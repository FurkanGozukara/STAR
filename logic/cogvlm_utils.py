import os
import torch
import threading
import gc
import io
import numpy as np
import gradio as gr
import logging
from .gpu_utils import get_gpu_device
from .cancellation_manager import cancellation_manager, CancelledError

# CogVLM availability flags
COG_VLM_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from decord import cpu, VideoReader, bridge
    try:
        from transformers import BitsAndBytesConfig
        BITSANDBYTES_AVAILABLE = True
    except ImportError:
        print("Warning: bitsandbytes not found. INT4/INT8 quantization for CogVLM2 will not be available.")
    COG_VLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CogVLM2 related components (transformers, decord) not fully found: {e}")
    print("Auto-captioning feature will be disabled.")

# Global state for CogVLM model
cogvlm_model_state = {"model": None, "tokenizer": None, "device": None, "quant": None}
cogvlm_lock = threading.RLock()

def load_cogvlm_model(quantization, device, cog_vlm_model_path, logger=None):
    """Load CogVLM model with specified quantization and device."""
    global cogvlm_model_state
    if logger:
        logger.info(f"Attempting to load CogVLM2 model with quantization: {quantization} on device: {device}")

    # Check for cancellation before starting model loading
    cancellation_manager.check_cancel("model loading start")

    with cogvlm_lock:
        if cogvlm_model_state["model"] is not None and cogvlm_model_state["quant"] == quantization and cogvlm_model_state["device"] == device:
            if logger:
                logger.info("CogVLM2 model already loaded with correct settings.")
            return cogvlm_model_state["model"], cogvlm_model_state["tokenizer"]
        elif cogvlm_model_state["model"] is not None:
            if logger:
                logger.info("Different CogVLM2 model/settings currently loaded, unloading before loading new one.")
            unload_cogvlm_model('full', logger)

        # Check for cancellation after unloading
        cancellation_manager.check_cancel("after model unloading")

        try:
            if logger:
                logger.info(f"Loading tokenizer from: {cog_vlm_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(cog_vlm_model_path, trust_remote_code=True)

            # Check for cancellation after tokenizer loading
            cancellation_manager.check_cancel("after tokenizer loading")

            if logger:
                logger.info("Tokenizer loaded successfully.")

            bnb_config = None
            model_dtype = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

            if BITSANDBYTES_AVAILABLE and device == 'cuda':
                if quantization == 4:
                    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=model_dtype)
                elif quantization == 8:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization in [4, 8] and device != 'cuda':
                if logger:
                    logger.warning("BitsAndBytes quantization is only available on CUDA. Loading in FP16/BF16.")
                quantization = 0

            current_device_map = None
            if bnb_config and device == 'cuda':
                current_device_map = "auto"

            effective_low_cpu_mem_usage = True if bnb_config else False

            if logger:
                logger.info(f"Preparing to load model from: {cog_vlm_model_path} with quant: {quantization}, dtype: {model_dtype}, device: {device}, device_map: {current_device_map}, low_cpu_mem: {effective_low_cpu_mem_usage}")

            # Final check for cancellation before model loading (this is the longest operation)  
            cancellation_manager.check_cancel("before model loading")
            
            # Note to user that model loading cannot be interrupted once started
            if logger:
                logger.info("Starting model loading - this operation cannot be interrupted once started")

            # Use a separate thread for model loading with periodic cancellation checks
            model_loading_result = {"model": None, "error": None, "cancelled_before_start": False}
            model_loading_complete = threading.Event()
            
            def load_model_thread():
                try:
                    # One final check right before starting the loading
                    if cancellation_manager.is_cancelled():
                        model_loading_result["cancelled_before_start"] = True
                        model_loading_complete.set()
                        return
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        cog_vlm_model_path,
                        torch_dtype=model_dtype if device == 'cuda' else torch.float32,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        low_cpu_mem_usage=effective_low_cpu_mem_usage,
                        device_map=current_device_map
                    )
                    model_loading_result["model"] = model
                except Exception as e:
                    model_loading_result["error"] = e
                finally:
                    model_loading_complete.set()
            
            # Start model loading in separate thread
            loading_thread = threading.Thread(target=load_model_thread, daemon=True)
            loading_thread.start()
            
            # Wait for loading to complete or cancellation, checking periodically
            while not model_loading_complete.is_set():
                # Check for cancellation every 0.5 seconds during model loading
                if model_loading_complete.wait(timeout=0.5):
                    break
                # If user cancels during loading, we can't stop the loading process
                # but we can detect it and handle it appropriately
                if cancellation_manager.is_cancelled():
                    if logger:
                        logger.warning("Cancellation requested during model loading - waiting for loading to complete before cleanup")
            
            # Wait for thread to complete
            loading_thread.join(timeout=30)  # Give it 30 seconds to complete
            
            # Check if cancellation was requested before loading started
            if model_loading_result["cancelled_before_start"]:
                if logger:
                    logger.info("Model loading cancelled before it started")
                raise CancelledError("Model loading was cancelled by the user.")
            
            # Check for loading errors
            if model_loading_result["error"]:
                raise model_loading_result["error"]
            
            model = model_loading_result["model"]
            if model is None:
                raise RuntimeError("Model loading thread completed but no model was returned")
            
            # Check for cancellation after model loading completes
            # If user cancelled during loading, we cleanup and raise error
            if cancellation_manager.is_cancelled():
                if logger:
                    logger.info("Cancellation detected after model loading completed - cleaning up loaded model")
                # Clean up the loaded model since it was cancelled
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise CancelledError("Model loading was cancelled by the user.")

            if not bnb_config and current_device_map is None:
                if logger:
                    logger.info(f"Moving non-quantized model to target device: {device}")
                actual_device = get_gpu_device() if device == 'cuda' else device
                model = model.to(actual_device)
            elif bnb_config and current_device_map == "auto":
                if logger:
                    logger.info(f"BNB model loaded with device_map='auto'. Should be on target CUDA device(s).")

            model = model.eval()

            cogvlm_model_state["model"] = model
            cogvlm_model_state["tokenizer"] = tokenizer
            cogvlm_model_state["device"] = device
            cogvlm_model_state["quant"] = quantization

            final_device_str = "N/A"
            final_dtype_str = "N/A"
            try:
                first_param = next(model.parameters(), None)
                if hasattr(model, 'device') and isinstance(model.device, torch.device):
                    final_device_str = str(model.device)
                elif hasattr(model, 'hf_device_map'):
                    final_device_str = str(model.hf_device_map)
                elif first_param is not None and hasattr(first_param, 'device'):
                    final_device_str = str(first_param.device)

                if first_param is not None and hasattr(first_param, 'dtype'):
                    final_dtype_str = str(first_param.dtype)
            except Exception as e_dev_dtype:
                if logger:
                    logger.warning(f"Could not reliably determine final model device/dtype: {e_dev_dtype}")

            if logger:
                logger.info(f"CogVLM2 model loaded (Quant: {quantization}, Requested Device: {device}, Final Device(s): {final_device_str}, Dtype: {final_dtype_str}).")
            return model, tokenizer
        except CancelledError:
            # Handle cancellation gracefully during model loading
            if logger:
                logger.info("CogVLM2 model loading cancelled by user.")
            # Clean up partial state
            _model_ref = cogvlm_model_state.pop("model", None)
            _tokenizer_ref = cogvlm_model_state.pop("tokenizer", None)
            if _model_ref:
                del _model_ref
            if _tokenizer_ref:
                del _tokenizer_ref
            cogvlm_model_state.update({"model": None, "tokenizer": None, "device": None, "quant": None})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise  # Re-raise the CancelledError
        except Exception as e:
            if logger:
                logger.error(f"Failed to load CogVLM2 model from path: {cog_vlm_model_path}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception details: {e}", exc_info=True)

            _model_ref = cogvlm_model_state.pop("model", None)
            _tokenizer_ref = cogvlm_model_state.pop("tokenizer", None)
            if _model_ref:
                del _model_ref
            if _tokenizer_ref:
                del _tokenizer_ref
            cogvlm_model_state.update({"model": None, "tokenizer": None, "device": None, "quant": None})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(f"Could not load CogVLM2 model (check logs for details): {str(e)[:200]}")

def unload_cogvlm_model(strategy, logger=None):
    """Unload CogVLM model with specified strategy."""
    global cogvlm_model_state
    if logger:
        logger.info(f"Unloading CogVLM2 model with strategy: {strategy}")
    with cogvlm_lock:
        if cogvlm_model_state.get("model") is None and cogvlm_model_state.get("tokenizer") is None:
            if logger:
                logger.info("CogVLM2 model/tokenizer not loaded or already unloaded.")
            return

        if strategy == 'cpu':
            try:
                model = cogvlm_model_state.get("model")
                if model is not None:
                    if cogvlm_model_state.get("quant") in [4, 8]:
                        if logger:
                            logger.info("BNB quantized model cannot be moved to CPU. Keeping on GPU or use 'full' unload.")
                        return

                    if cogvlm_model_state.get("device") == 'cuda' and torch.cuda.is_available():
                        model.to('cpu')
                        cogvlm_model_state["device"] = 'cpu'
                        if logger:
                            logger.info("CogVLM2 model moved to CPU.")
                    else:
                        if logger:
                            logger.info("CogVLM2 model already on CPU or CUDA not available for move.")
                else:
                    if logger:
                        logger.info("No model found in state to move to CPU.")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to move CogVLM2 model to CPU: {e}")
        elif strategy == 'full':
            model_obj = cogvlm_model_state.pop("model", None)
            tokenizer_obj = cogvlm_model_state.pop("tokenizer", None)

            cogvlm_model_state.update({"model": None, "tokenizer": None, "device": None, "quant": None})

            if model_obj is not None:
                del model_obj
                if logger:
                    logger.info("Explicitly deleted popped model object.")
            if tokenizer_obj is not None:
                del tokenizer_obj
                if logger:
                    logger.info("Explicitly deleted popped tokenizer object.")

            gc.collect()
            if logger:
                logger.info("Python garbage collection triggered.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if logger:
                    logger.info("CUDA cache emptied.")
            if logger:
                logger.info("CogVLM2 model and tokenizer fully unloaded and CUDA cache (if applicable) cleared.")
        else:
            if logger:
                logger.warning(f"Unknown unload strategy: {strategy}")

def auto_caption(video_path, quantization, unload_strategy, cog_vlm_model_path, logger=None, progress=gr.Progress(track_tqdm=True)):
    """Generate automatic caption for video using CogVLM model."""
    if not COG_VLM_AVAILABLE:
        raise gr.Error("CogVLM2 components not available. Captioning disabled.")
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Please provide a valid video file for captioning.")

    # Check for cancellation at the start
    cancellation_manager.check_cancel("auto_caption start")

    cogvlm_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if quantization in [4, 8] and cogvlm_device == 'cpu':
        raise gr.Error("INT4/INT8 quantization requires CUDA. Please select FP16/BF16 for CPU or ensure CUDA is available.")

    caption = "Error: Caption generation failed."

    local_model_ref = None
    local_tokenizer_ref = None
    inputs_on_device = None
    video_data_cog = None
    outputs_tensor = None

    try:
        progress(0.1, desc="Loading CogVLM2 for captioning...")
        local_model_ref, local_tokenizer_ref = load_cogvlm_model(quantization, cogvlm_device, cog_vlm_model_path, logger)

        # Check for cancellation after model loading
        cancellation_manager.check_cancel("after model loading in auto_caption")

        model_compute_device = cogvlm_device
        model_actual_dtype = torch.float32

        if local_model_ref is not None:
            first_param = next(local_model_ref.parameters(), None)
            if hasattr(local_model_ref, 'device') and isinstance(local_model_ref.device, torch.device):
                model_compute_device = local_model_ref.device
            elif hasattr(local_model_ref, 'hf_device_map'):
                if first_param is not None:
                    model_compute_device = first_param.device
                else:
                    model_compute_device = torch.device(cogvlm_device)
            elif first_param is not None:
                model_compute_device = first_param.device

            if first_param is not None:
                model_actual_dtype = first_param.dtype
            elif cogvlm_device == 'cuda':
                model_actual_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        if isinstance(model_compute_device, str) and model_compute_device == 'cuda':
            model_compute_device = torch.device(get_gpu_device())
        elif not isinstance(model_compute_device, torch.device):
            model_compute_device = torch.device(model_compute_device)

        if logger:
            logger.info(f"CogVLM2 for captioning. Inputs will target device: {model_compute_device}, dtype: {model_actual_dtype}")

        progress(0.3, desc="Preparing video for CogVLM2...")
        # Check for cancellation before video processing
        cancellation_manager.check_cancel("before video processing")
        
        bridge.set_bridge('torch')
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()

            if len(video_bytes) == 0:
                raise gr.Error("Video file is empty or could not be read.")

            decord_vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
            num_frames_cog = 24
            total_frames_decord = len(decord_vr)

            if total_frames_decord == 0:
                raise gr.Error("Video has no frames or could not be read by decord.")
            
            # Validate video reader properties
            try:
                # Try to get basic video info to ensure the reader is working
                fps = decord_vr.get_avg_fps()
                if logger:
                    logger.info(f"Video loaded: {total_frames_decord} frames, {fps:.2f} FPS")
            except Exception as info_error:
                if logger:
                    logger.warning(f"Could not get video info, but proceeding: {info_error}")
                    
        except Exception as video_load_error:
            if logger:
                logger.error(f"Failed to load video with decord: {video_load_error}")
            raise gr.Error(f"Could not load video for captioning: {video_load_error}")

        # Check for cancellation after video loading
        cancellation_manager.check_cancel("after video loading")

        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames_decord))
        timestamps = [ts[0] for ts in timestamps]

        frame_id_list = []
        if timestamps:
            max_second = int(round(max(timestamps) if timestamps else 0)) + 1
            unique_indices = set()
            for sec in range(max_second):
                if len(unique_indices) >= num_frames_cog:
                    break
                closest_time_diff = float('inf')
                closest_idx = -1
                for idx, ts_val in enumerate(timestamps):
                    diff = abs(ts_val - sec)
                    if diff < closest_time_diff:
                        closest_time_diff = diff
                        closest_idx = idx
                if closest_idx != -1:
                    unique_indices.add(closest_idx)
            frame_id_list = sorted(list(unique_indices))[:num_frames_cog]

        if len(frame_id_list) < num_frames_cog:
            if logger:
                logger.warning(f"Sampled {len(frame_id_list)} frames, need {num_frames_cog}. Using linspace.")
            step = max(1, total_frames_decord // num_frames_cog)
            frame_id_list = [min(i * step, total_frames_decord - 1) for i in range(num_frames_cog)]
            frame_id_list = sorted(list(set(frame_id_list)))
            frame_id_list = frame_id_list[:num_frames_cog]

        if not frame_id_list:
            if logger:
                logger.warning("Frame ID list is empty, using first N frames or all if less than N.")
            frame_id_list = list(range(min(num_frames_cog, total_frames_decord)))
        
        # Ensure all frame indices are within valid bounds
        frame_id_list = [max(0, min(idx, total_frames_decord - 1)) for idx in frame_id_list]
        
        # Remove duplicates while preserving order and ensure we have enough frames
        seen = set()
        unique_frame_list = []
        for idx in frame_id_list:
            if idx not in seen:
                seen.add(idx)
                unique_frame_list.append(idx)
        
        # If we still don't have enough unique frames, fill with evenly spaced frames
        if len(unique_frame_list) < num_frames_cog:
            if logger:
                logger.warning(f"Only {len(unique_frame_list)} unique frames available, filling with evenly spaced frames.")
            # Create evenly spaced frame indices
            if total_frames_decord > 0:
                step = max(1, total_frames_decord // num_frames_cog)
                evenly_spaced = [min(i * step, total_frames_decord - 1) for i in range(num_frames_cog)]
                # Merge with existing unique frames
                all_frames = list(set(unique_frame_list + evenly_spaced))
                frame_id_list = sorted(all_frames)[:num_frames_cog]
            else:
                frame_id_list = [0] * num_frames_cog
        else:
            frame_id_list = unique_frame_list[:num_frames_cog]

        if logger:
            logger.info(f"CogVLM2 using frame indices: {frame_id_list}")
        
        # Check for cancellation before frame extraction
        cancellation_manager.check_cancel("before frame extraction")
        
        # Robust frame extraction with fallback for DECORD errors
        video_data_cog = None
        try:
            # First attempt: try to get batch of frames
            video_data_cog = decord_vr.get_batch(frame_id_list).permute(3, 0, 1, 2)
            
            # Convert RGBA to RGB if necessary (CogVLM expects 3 channels)
            if video_data_cog.shape[0] == 4:  # RGBA
                if logger:
                    logger.info("Converting RGBA frames to RGB (removing alpha channel)")
                video_data_cog = video_data_cog[:3, :, :, :]  # Keep only RGB channels
        except Exception as decord_error:
            if logger:
                logger.warning(f"DECORD batch extraction failed: {decord_error}")
                logger.info("Attempting frame-by-frame extraction as fallback...")
            
            # Fallback: extract frames one by one and handle size mismatches
            try:
                frames_list = []
                target_shape = None
                
                for frame_idx in frame_id_list:
                    try:
                        # Clamp frame index to valid range
                        safe_frame_idx = max(0, min(frame_idx, total_frames_decord - 1))
                        frame = decord_vr[safe_frame_idx]  # Get single frame
                        
                        # Check if this is the first valid frame to establish target shape
                        if target_shape is None:
                            target_shape = frame.shape
                            if logger:
                                logger.info(f"Established target frame shape: {target_shape}")
                        
                        # Ensure all frames have the same shape
                        if frame.shape != target_shape:
                            if logger:
                                logger.warning(f"Frame {safe_frame_idx} has different shape {frame.shape}, expected {target_shape}")
                            # Resize frame to match target shape if needed
                            # For now, skip this frame and use the previous valid frame
                            if frames_list:
                                frame = frames_list[-1].clone()
                            else:
                                continue
                        
                        frames_list.append(frame)
                        
                    except Exception as frame_error:
                        if logger:
                            logger.warning(f"Could not extract frame {frame_idx}: {frame_error}")
                        # Use previous frame if available, otherwise skip
                        if frames_list:
                            frames_list.append(frames_list[-1].clone())
                
                if not frames_list:
                    raise gr.Error("Could not extract any valid frames from video for captioning.")
                
                # Pad or truncate to exactly num_frames_cog frames
                while len(frames_list) < num_frames_cog:
                    frames_list.append(frames_list[-1].clone())
                frames_list = frames_list[:num_frames_cog]
                
                # Stack frames and permute dimensions
                video_data_cog = torch.stack(frames_list, dim=0).permute(3, 0, 1, 2)
                
                # Convert RGBA to RGB if necessary (CogVLM expects 3 channels)
                if video_data_cog.shape[0] == 4:  # RGBA
                    if logger:
                        logger.info("Converting RGBA frames to RGB (removing alpha channel)")
                    video_data_cog = video_data_cog[:3, :, :, :]  # Keep only RGB channels
                
                if logger:
                    logger.info(f"Successfully extracted {len(frames_list)} frames using fallback method")
                    logger.info(f"Final video tensor shape for CogVLM: {video_data_cog.shape}")
                    
            except Exception as fallback_error:
                if logger:
                    logger.error(f"Frame-by-frame extraction also failed: {fallback_error}")
                raise gr.Error(f"Could not extract video frames for captioning: {fallback_error}")
        
        if video_data_cog is None:
            raise gr.Error("Failed to extract video data for captioning.")
        
        # Final validation: Ensure video tensor has correct shape for CogVLM
        expected_channels = 3  # RGB
        if video_data_cog.shape[0] != expected_channels:
            if logger:
                logger.warning(f"Unexpected video tensor shape: {video_data_cog.shape}. Expected {expected_channels} channels.")
            if video_data_cog.shape[0] == 4:  # RGBA, convert to RGB
                if logger:
                    logger.info("Final conversion: RGBA to RGB")
                video_data_cog = video_data_cog[:3, :, :, :]
            elif video_data_cog.shape[0] == 1:  # Grayscale, convert to RGB
                if logger:
                    logger.info("Final conversion: Grayscale to RGB by repeating channel")
                video_data_cog = video_data_cog.repeat(3, 1, 1, 1)
            else:
                raise gr.Error(f"Unsupported video format: {video_data_cog.shape[0]} channels. CogVLM requires RGB (3 channels).")
        
        if logger:
            logger.info(f"Final video tensor shape for CogVLM processing: {video_data_cog.shape}")

        query = "Please describe this video in detail."
        inputs = local_model_ref.build_conversation_input_ids(
            tokenizer=local_tokenizer_ref, query=query, images=[video_data_cog], history=[], template_version='chat'
        )

        inputs_on_device = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(model_compute_device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(model_compute_device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(model_compute_device),
            'images': [[inputs['images'][0].to(model_compute_device).to(model_actual_dtype)]],
        }

        gen_kwargs = {"max_new_tokens": 256, "pad_token_id": local_tokenizer_ref.eos_token_id or 128002, "top_k": 5, "do_sample": True, "top_p": 0.8, "temperature": 0.8}

        progress(0.6, desc="Generating caption with CogVLM2...")
        # Check for cancellation before generation
        cancellation_manager.check_cancel("before caption generation")
        
        with torch.no_grad():
            outputs_tensor = local_model_ref.generate(**inputs_on_device, **gen_kwargs)
        outputs_tensor = outputs_tensor[:, inputs_on_device['input_ids'].shape[1]:]
        caption = local_tokenizer_ref.decode(outputs_tensor[0], skip_special_tokens=True).strip()
        if logger:
            logger.info(f"Generated Caption: {caption}")
        progress(0.9, desc="Caption generated.")

    except CancelledError:
        if logger:
            logger.info("CogVLM2 caption generation cancelled by user.")
        caption = "Caption generation cancelled by user."
        # Clean up will happen in finally block
    except Exception as e:
        if logger:
            logger.error(f"Error during auto-captioning: {e}", exc_info=True)
        caption = f"Error during captioning: {str(e)[:100]}"
    finally:
        progress(1.0, desc="Finalizing captioning...")

        if outputs_tensor is not None:
            del outputs_tensor
        if inputs_on_device is not None:
            inputs_on_device.clear()
            del inputs_on_device
        if video_data_cog is not None:
            del video_data_cog

        if local_model_ref is not None:
            del local_model_ref
        if local_tokenizer_ref is not None:
            del local_tokenizer_ref

        unload_cogvlm_model(unload_strategy, logger)
    return caption, f"Captioning status: {'Success' if not caption.startswith('Error') and not caption.startswith('Caption generation cancelled') else caption}" 