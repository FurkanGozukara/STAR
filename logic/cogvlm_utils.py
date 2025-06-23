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
    cancellation_manager.check_cancel()

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
        cancellation_manager.check_cancel()

        try:
            if logger:
                logger.info(f"Loading tokenizer from: {cog_vlm_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(cog_vlm_model_path, trust_remote_code=True)

            # Check for cancellation after tokenizer loading
            cancellation_manager.check_cancel()

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

            # Check for cancellation before model loading (this is the longest operation)  
            cancellation_manager.check_cancel()

            model = AutoModelForCausalLM.from_pretrained(
                cog_vlm_model_path,
                torch_dtype=model_dtype if device == 'cuda' else torch.float32,
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=effective_low_cpu_mem_usage,
                device_map=current_device_map
            )

            # Check for cancellation after model loading
            cancellation_manager.check_cancel()

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
    cancellation_manager.check_cancel()

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
        cancellation_manager.check_cancel()

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
        cancellation_manager.check_cancel()
        
        bridge.set_bridge('torch')
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        decord_vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        num_frames_cog = 24
        total_frames_decord = len(decord_vr)

        if total_frames_decord == 0:
            raise gr.Error("Video has no frames or could not be read by decord.")

        # Check for cancellation after video loading
        cancellation_manager.check_cancel()

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

        if logger:
            logger.info(f"CogVLM2 using frame indices: {frame_id_list}")
        
        # Check for cancellation before frame extraction
        cancellation_manager.check_cancel()
        
        video_data_cog = decord_vr.get_batch(frame_id_list).permute(3, 0, 1, 2)

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
        cancellation_manager.check_cancel()
        
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