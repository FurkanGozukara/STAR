# --- START OF FILE secourses_app.py ---

import gradio as gr
import os
import sys
import torch
import torchvision
import subprocess
import cv2
import numpy as np
import math
import time
import shutil
import tempfile
import threading
from easydict import EasyDict
from argparse import ArgumentParser, Namespace # Keep if STAR scripts expect it

try:

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = script_dir # If app.py is in STAR's root.

    if not os.path.isdir(os.path.join(base_path, 'video_to_video')):
        # Fallback or raise error if structure is not as expected
        print(f"Warning: 'video_to_video' directory not found in inferred base_path: {base_path}. Attempting to use parent directory.")
        base_path = os.path.dirname(base_path) # Try one level up if app.py is nested
        if not os.path.isdir(os.path.join(base_path, 'video_to_video')):
            print(f"Error: Could not auto-determine STAR repository root. Please set 'base_path' manually.")
            print(f"Current inferred base_path: {base_path}")

    print(f"Using STAR repository base_path: {base_path}")
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

except Exception as e_path:
    print(f"Error setting up base_path: {e_path}")
    print("Please ensure app.py is correctly placed or base_path is manually set.")
    sys.exit(1)


# --- Try importing STAR components ---
try:
    from video_to_video.video_to_video_model import VideoToVideo_sr
    from video_to_video.utils.seed import setup_seed
    from video_to_video.utils.logger import get_logger
    from video_super_resolution.color_fix import adain_color_fix, wavelet_color_fix
    from inference_utils import tensor2vid, preprocess, collate_fn
    from video_super_resolution.scripts.util_image import ImageSpliterTh
    from video_to_video.utils.config import cfg as star_cfg
except ImportError as e:
    print(f"Error importing STAR components: {e}")
    print(f"Searched in sys.path: {sys.path}")
    print("Please ensure the STAR repository is correctly in the Python path (set by base_path) and all dependencies from 'requirements.txt' are installed.")
    sys.exit(1)

# --- Try importing CogVLM2 components ---
COG_VLM_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from decord import cpu, VideoReader, bridge
    import io # For CogVLM video loading
    try:
        from transformers import BitsAndBytesConfig
        BITSANDBYTES_AVAILABLE = True
    except ImportError:
        print("Warning: bitsandbytes not found. INT4/INT8 quantization for CogVLM2 will not be available.")
    COG_VLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CogVLM2 related components (transformers, decord) not fully found: {e}")
    print("Auto-captioning feature will be disabled.")


# --- Global Variables & Constants ---
logger = get_logger()
DEFAULT_OUTPUT_DIR = "upscaled_videos"
# COG_VLM_MODEL_PATH = "/models/cogvlm2-video-llama3-chat" # Incorrect hardcoded path

# Construct model paths relative to the determined base_path
COG_VLM_MODEL_PATH = os.path.join(base_path, 'models', 'cogvlm2-video-llama3-chat') # Correct relative path
LIGHT_DEG_MODEL = os.path.join(base_path, 'pretrained_weight', 'light_deg.pt')
HEAVY_DEG_MODEL = os.path.join(base_path, 'pretrained_weight', 'heavy_deg.pt')

DEFAULT_POS_PROMPT = star_cfg.positive_prompt
DEFAULT_NEG_PROMPT = star_cfg.negative_prompt

# Check model paths after definition
if not os.path.exists(LIGHT_DEG_MODEL):
     logger.error(f"FATAL: Light degradation model not found at {LIGHT_DEG_MODEL}.")
     # sys.exit(1) # Exit if models are critical
if not os.path.exists(HEAVY_DEG_MODEL):
     logger.error(f"FATAL: Heavy degradation model not found at {HEAVY_DEG_MODEL}.")
     # sys.exit(1)

cogvlm_model_state = {"model": None, "tokenizer": None, "device": None, "quant": None}
cogvlm_lock = threading.Lock()

def run_ffmpeg_command(cmd, desc="ffmpeg command"):
    logger.info(f"Running {desc}: {cmd}")
    try:
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if process.stdout: logger.info(f"{desc} stdout: {process.stdout.strip()}")
        if process.stderr: logger.info(f"{desc} stderr: {process.stderr.strip()}") # Log stderr as info too
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {desc}:")
        logger.error(f"  Command: {e.cmd}")
        logger.error(f"  Return code: {e.returncode}")
        if e.stdout: logger.error(f"  Stdout: {e.stdout.strip()}")
        if e.stderr: logger.error(f"  Stderr: {e.stderr.strip()}")
        raise gr.Error(f"ffmpeg command failed (see console for details): {e.stderr.strip()[:500] if e.stderr else 'Unknown ffmpeg error'}")
    except Exception as e_gen:
        logger.error(f"Unexpected error preparing/running {desc} for command '{cmd}': {e_gen}")
        raise gr.Error(f"ffmpeg command failed: {e_gen}")


def extract_frames(video_path, temp_dir):
    logger.info(f"Extracting frames from '{video_path}' to '{temp_dir}'")
    os.makedirs(temp_dir, exist_ok=True)
    fps = 30.0 # Default
    try:
        probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        process = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        rate_str = process.stdout.strip()
        if '/' in rate_str:
            num, den = map(int, rate_str.split('/'))
            if den != 0: fps = num / den
        elif rate_str: # If it's a single number
            fps = float(rate_str)
        logger.info(f"Detected FPS: {fps}")
    except Exception as e:
        logger.warning(f"Could not get FPS using ffprobe for '{video_path}': {e}. Using default {fps} FPS.")

    cmd = f'ffmpeg -i "{video_path}" -vsync vfr -qscale:v 2 "{os.path.join(temp_dir, "frame_%06d.png")}"' # Use os.path.join
    run_ffmpeg_command(cmd, "Frame Extraction")

    frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')])
    frame_count = len(frame_files)
    logger.info(f"Extracted {frame_count} frames.")
    if frame_count == 0:
        raise gr.Error("Failed to extract any frames. Check video file and ffmpeg installation.")
    return frame_count, fps, frame_files

def create_video_from_frames(frame_dir, output_path, fps):
    logger.info(f"Creating video from frames in '{frame_dir}' to '{output_path}' at {fps} FPS")
    input_pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = f'ffmpeg -y -framerate {fps} -i "{input_pattern}" -c:v libx264 -preset ultrafast -qp 0 -pix_fmt yuv420p "{output_path}"'
    run_ffmpeg_command(cmd, "Video Reassembly")

def get_next_filename(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    max_num = 0
    for f_name in existing_files:
        try:
            num = int(os.path.splitext(f_name)[0])
            if num > max_num: max_num = num
        except ValueError: continue
    return os.path.join(output_dir, f"{max_num + 1:04d}.mp4")

def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir) and os.path.isdir(temp_dir): # Added isdir check
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error removing temporary directory '{temp_dir}': {e}")

def get_video_resolution(video_path):
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "{video_path}"'
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        w_str, h_str = process.stdout.strip().split('x')
        w, h = int(w_str), int(h_str)
        logger.info(f"Video resolution (wxh) from ffprobe for '{video_path}': {w}x{h}")
        return h, w
    except Exception as e:
        logger.warning(f"ffprobe failed for '{video_path}' ({e}), trying OpenCV...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise gr.Error(f"Cannot open video file: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if h > 0 and w > 0:
             logger.info(f"Video resolution (wxh) from OpenCV for '{video_path}': {w}x{h}")
             return h, w
        raise gr.Error(f"Could not determine resolution for video: {video_path}")

def calculate_upscale_params(orig_h, orig_w, target_h, target_w, target_res_mode):
    final_h = int(target_h)
    final_w = int(target_w)
    needs_downscale = False
    downscale_h, downscale_w = orig_h, orig_w

    if target_res_mode == 'Downscale then 4x':
        intermediate_h = final_h / 4.0
        intermediate_w = final_w / 4.0
        if orig_h > intermediate_h or orig_w > intermediate_w:
            needs_downscale = True
            ratio = min(intermediate_h / orig_h, intermediate_w / orig_w)
            downscale_h = int(round(orig_h * ratio / 2) * 2)
            downscale_w = int(round(orig_w * ratio / 2) * 2)
            logger.info(f"Downscaling required: {orig_h}x{orig_w} -> {downscale_w}x{downscale_h} for 4x target.")
        else:
            logger.info("No downscaling needed for 'Downscale then 4x' mode.")
        final_upscale_factor = 4.0
        # Recalculate actual final H, W based on (potentially) downscaled source
        final_h = int(round(downscale_h * final_upscale_factor / 2) * 2)
        final_w = int(round(downscale_w * final_upscale_factor / 2) * 2)

    elif target_res_mode == 'Ratio Upscale':
        if orig_h == 0 or orig_w == 0: raise ValueError("Original dimensions cannot be zero.")
        ratio_h = final_h / orig_h
        ratio_w = final_w / orig_w
        final_upscale_factor = min(ratio_h, ratio_w)
        final_h = int(round(orig_h * final_upscale_factor / 2) * 2)
        final_w = int(round(orig_w * final_upscale_factor / 2) * 2)
        logger.info(f"Ratio Upscale mode: Using upscale factor {final_upscale_factor:.2f}")
    else:
        raise ValueError(f"Invalid target_res_mode: {target_res_mode}")

    logger.info(f"Calculated final target resolution: {final_w}x{final_h} with upscale {final_upscale_factor:.2f}")
    return needs_downscale, downscale_h, downscale_w, final_upscale_factor, final_h, final_w


def load_cogvlm_model(quantization, device):
    global cogvlm_model_state
    logger.info(f"Attempting to load CogVLM2 model with quantization: {quantization} on device: {device}")

    with cogvlm_lock:
        if cogvlm_model_state["model"] is not None and \
           cogvlm_model_state["quant"] == quantization and \
           cogvlm_model_state["device"] == device:
            logger.info("CogVLM2 model already loaded with correct settings.")
            return cogvlm_model_state["model"], cogvlm_model_state["tokenizer"]
        elif cogvlm_model_state["model"] is not None:
            logger.info("Different CogVLM2 model/settings currently loaded, unloading before loading new one.")
            unload_cogvlm_model('full')

        try:
            logger.info(f"Loading tokenizer from: {COG_VLM_MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(COG_VLM_MODEL_PATH, trust_remote_code=True)
            cogvlm_model_state["tokenizer"] = tokenizer
            logger.info("Tokenizer loaded successfully.")

            bnb_config = None
            model_dtype = torch.bfloat16 if (device=='cuda' and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

            if BITSANDBYTES_AVAILABLE and device == 'cuda':
                if quantization == 4:
                    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=model_dtype)
                elif quantization == 8:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization in [4, 8] and device != 'cuda':
                 logger.warning("BitsAndBytes quantization is only available on CUDA. Loading in FP16/BF16.")
                 quantization = 0 # Fallback, bnb_config will remain None

            # Determine device_map strategy
            current_device_map = None
            if bnb_config and device == 'cuda':
                # For BitsAndBytes quantized models on CUDA, "auto" lets accelerate handle placement.
                current_device_map = "auto"
            
            effective_low_cpu_mem_usage = True if bnb_config else False

            logger.info(f"Preparing to load model from: {COG_VLM_MODEL_PATH} with quant: {quantization}, dtype: {model_dtype}, device: {device}, device_map: {current_device_map}, low_cpu_mem: {effective_low_cpu_mem_usage}")
            
            model = AutoModelForCausalLM.from_pretrained(
                COG_VLM_MODEL_PATH,
                torch_dtype=model_dtype if device == 'cuda' else torch.float32,
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=effective_low_cpu_mem_usage,
                device_map=current_device_map # MODIFIED HERE
            )

            if not bnb_config and current_device_map is None : 
                logger.info(f"Moving non-quantized model to target device: {device}")
                model = model.to(device)
            elif bnb_config and current_device_map == "auto":
                 logger.info(f"BNB model loaded with device_map='auto'. Should be on target CUDA device(s).")


            model = model.eval()
            cogvlm_model_state["model"] = model
            cogvlm_model_state["device"] = device 
            cogvlm_model_state["quant"] = quantization
            
            final_device_str = "N/A"
            final_dtype_str = "N/A"
            try:
                if hasattr(model, 'device'):
                    final_device_str = str(model.device) # For BNB models, model.device gives the map or main device
                elif hasattr(next(model.parameters(), None), 'device'):
                     final_device_str = str(next(model.parameters()).device)

                if hasattr(next(model.parameters(), None), 'dtype'):
                    final_dtype_str = str(next(model.parameters()).dtype)

            except Exception as e_dev_dtype:
                logger.warning(f"Could not reliably determine final model device/dtype: {e_dev_dtype}")

            logger.info(f"CogVLM2 model loaded (Quant: {quantization}, Requested Device: {device}, Final Device(s): {final_device_str}, Dtype: {final_dtype_str}).")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load CogVLM2 model from path: {COG_VLM_MODEL_PATH}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {e}", exc_info=True)
            unload_cogvlm_model('full') 
            raise gr.Error(f"Could not load CogVLM2 model (check logs for details): {str(e)[:200]}")


def unload_cogvlm_model(strategy):
    global cogvlm_model_state
    logger.info(f"Unloading CogVLM2 model with strategy: {strategy}")
    with cogvlm_lock:
        if cogvlm_model_state["model"] is None:
            logger.info("CogVLM2 model not loaded or already unloaded.")
            return
        if strategy == 'cpu':
            try:
                # For BNB models, .to('cpu') is not supported. Unload fully or keep on GPU.
                if cogvlm_model_state["quant"] in [4, 8]:
                     logger.info("BNB quantized model cannot be moved to CPU. Keeping on GPU or use 'full' unload.")
                     return # Or proceed to full unload if 'cpu' is not viable
                
                if cogvlm_model_state["device"] == 'cuda' and torch.cuda.is_available():
                    cogvlm_model_state["model"].to('cpu')
                    cogvlm_model_state["device"] = 'cpu'
                    logger.info("CogVLM2 model moved to CPU.")
                else:
                    logger.info("CogVLM2 model already on CPU or CUDA not available for move.")
            except Exception as e: logger.error(f"Failed to move CogVLM2 model to CPU: {e}")
        elif strategy == 'full':
            try:
                del cogvlm_model_state["model"]
                del cogvlm_model_state["tokenizer"]
            except AttributeError: pass 
            cogvlm_model_state.update({"model": None, "tokenizer": None, "device": None, "quant": None})
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info("CogVLM2 model fully unloaded and CUDA cache cleared.")
        else:
            logger.warning(f"Unknown unload strategy: {strategy}")

def auto_caption(video_path, quantization, unload_strategy, progress=gr.Progress(track_tqdm=True)):
    if not COG_VLM_AVAILABLE:
        raise gr.Error("CogVLM2 components not available. Captioning disabled.")
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Please provide a valid video file for captioning.")

    # Determine target device for CogVLM model loading
    cogvlm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if quantization in [4,8] and cogvlm_device == 'cpu':
        # This case should be prevented by UI or load_cogvlm_model logic, but double check
        raise gr.Error("INT4/INT8 quantization requires CUDA. Please select FP16/BF16 for CPU or ensure CUDA is available.")

    caption = "Error: Caption generation failed."
    
    try:
        progress(0.1, desc="Loading CogVLM2 for captioning...")
        model, tokenizer = load_cogvlm_model(quantization, cogvlm_device)
        
        if hasattr(model, 'device') and isinstance(model.device, torch.device) : # Non-BNB or BNB on single device after loading
            model_compute_device = model.device
        elif hasattr(model, 'hf_device_map'): # BNB model with device_map

            model_compute_device = torch.device(cogvlm_device) # Default to the main requested device
            # Try to get a more specific device if possible, e.g., from a known submodule
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'device'): # Llama-like structure
                 model_compute_device = model.transformer.device
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'device'): # CogVLM-like structure
                 model_compute_model_device = model.language_model.device
            elif next(model.parameters(), None) is not None:
                 model_compute_device = next(model.parameters()).device


        else: # Fallback
            model_compute_device = torch.device(cogvlm_device)
        
        model_actual_dtype = next(model.parameters()).dtype # Get dtype from model parameters
        logger.info(f"CogVLM2 for captioning. Inputs will target device: {model_compute_device}, dtype: {model_actual_dtype}")


        progress(0.3, desc="Preparing video for CogVLM2...")
        bridge.set_bridge('torch')
        with open(video_path, 'rb') as f: video_bytes = f.read()
        
        decord_vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        num_frames_cog = 24
        total_frames_decord = len(decord_vr)

        if total_frames_decord == 0:
            raise gr.Error("Video has no frames or could not be read by decord.")

        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames_decord))
        timestamps = [ts[0] for ts in timestamps]
        
        frame_id_list = []
        if timestamps: 
            max_second = int(round(max(timestamps) if timestamps else 0)) + 1
            unique_indices = set()
            for sec in range(max_second):
                if len(unique_indices) >= num_frames_cog: break
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
            logger.warning(f"Sampled {len(frame_id_list)} frames, need {num_frames_cog}. Using linspace.")
            step = max(1, total_frames_decord // num_frames_cog)
            frame_id_list = [min(i * step, total_frames_decord-1) for i in range(num_frames_cog)]
            frame_id_list = sorted(list(set(frame_id_list))) 
            frame_id_list = frame_id_list[:num_frames_cog] 

        if not frame_id_list: 
            logger.warning("Frame ID list is empty, using first N frames or all if less than N.")
            frame_id_list = list(range(min(num_frames_cog, total_frames_decord)))

        logger.info(f"CogVLM2 using frame indices: {frame_id_list}")
        video_data_cog = decord_vr.get_batch(frame_id_list).permute(3, 0, 1, 2) # CTHW

        query = "Please describe this video in detail."
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer, query=query, images=[video_data_cog], history=[], template_version='chat'
        )
        
        inputs_on_device = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(model_compute_device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(model_compute_device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(model_compute_device),
            # Ensure image tensor is also on the correct device and dtype
            'images': [[inputs['images'][0].to(model_compute_device).to(model_actual_dtype)]],
        }

        gen_kwargs = {"max_new_tokens": 256, "pad_token_id": tokenizer.eos_token_id or 128002, "top_k": 5, "do_sample": True, "top_p": 0.8, "temperature": 0.8}
        
        progress(0.6, desc="Generating caption with CogVLM2...")
        with torch.no_grad():
            outputs = model.generate(**inputs_on_device, **gen_kwargs)
        outputs = outputs[:, inputs_on_device['input_ids'].shape[1]:]
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.info(f"Generated Caption: {caption}")
        progress(0.9, desc="Caption generated.")

    except Exception as e:
        logger.error(f"Error during auto-captioning: {e}", exc_info=True)
        caption = f"Error during captioning: {str(e)[:100]}" 
    finally:
        progress(1.0, desc="Finalizing captioning...")

        unload_cogvlm_model(unload_strategy)
    return caption, f"Captioning status: {'Success' if not caption.startswith('Error') else caption}"


def run_upscale(
    input_video_path, user_prompt, positive_prompt, negative_prompt, model_choice,
    upscale_factor_slider, cfg_scale, steps, solver_mode,
    max_chunk_len, vae_chunk, color_fix_method,
    enable_tiling, tile_size, tile_overlap,
    enable_sliding_window, window_size, window_step,
    enable_target_res, target_h, target_w, target_res_mode,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_video_path or not os.path.exists(input_video_path):
        raise gr.Error("Please upload a valid input video.")

    setup_seed(666)
    output_path = get_next_filename(DEFAULT_OUTPUT_DIR)
    # Create a unique temp_dir for each run to avoid conflicts
    run_id = f"star_run_{int(time.time())}_{np.random.randint(1000, 9999)}"
    temp_dir_base = tempfile.gettempdir()
    temp_dir = os.path.join(temp_dir_base, run_id)
    
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    
    # Ensure base temp dir exists
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    star_model = None
    current_input_video_for_frames = input_video_path # Video from which frames will be extracted
    downscaled_temp_video = None
    status_log = ["Process started..."]

    try:
        progress(0, desc="Initializing...")
        status_log.append("Initializing upscaling process...")
        yield None, "\n".join(status_log) # Initial status update

        final_prompt = (user_prompt.strip() + ". " + positive_prompt.strip()).strip()
        
        model_file_path = LIGHT_DEG_MODEL if model_choice == "Light Degradation" else HEAVY_DEG_MODEL
        if not os.path.exists(model_file_path):
            raise gr.Error(f"STAR model weight not found: {model_file_path}")

        orig_h, orig_w = get_video_resolution(input_video_path)
        status_log.append(f"Original resolution: {orig_w}x{orig_h}")
        progress(0.05, desc="Calculating target resolution...")

        if enable_target_res:
            needs_downscale, ds_h, ds_w, upscale_factor, final_h, final_w = calculate_upscale_params(
                orig_h, orig_w, target_h, target_w, target_res_mode
            )
            status_log.append(f"Target resolution mode: {target_res_mode}. Calculated upscale: {upscale_factor:.2f}x. Target output: {final_w}x{final_h}")
            if needs_downscale:
                progress(0.07, desc="Downscaling input video...")
                status_log.append(f"Downscaling input to {ds_w}x{ds_h} before upscaling.")
                yield None, "\n".join(status_log)
                downscaled_temp_video = os.path.join(temp_dir, "downscaled_input.mp4")
                scale_filter = f"scale='trunc(iw*min({ds_w}/iw,{ds_h}/ih)/2)*2':'trunc(ih*min({ds_w}/iw,{ds_h}/ih)/2)*2'"
                cmd = f'ffmpeg -i "{input_video_path}" -vf "{scale_filter}" -c:a copy "{downscaled_temp_video}"'
                run_ffmpeg_command(cmd, "Input Downscaling")
                current_input_video_for_frames = downscaled_temp_video
                # Update effective input dimensions based on actual output of ffmpeg if possible, or use ds_h, ds_w
                orig_h, orig_w = get_video_resolution(downscaled_temp_video) # Get actual res after ffmpeg
        else:
            upscale_factor = upscale_factor_slider
            final_h = int(round(orig_h * upscale_factor / 2) * 2) 
            final_w = int(round(orig_w * upscale_factor / 2) * 2) 
            status_log.append(f"Direct upscale: {upscale_factor:.2f}x. Target output: {final_w}x{final_h}")
        
        yield None, "\n".join(status_log)

        progress(0.1, desc="Loading STAR model...")
        model_cfg = EasyDict() 
        model_cfg.model_path = model_file_path

        star_model = VideoToVideo_sr(model_cfg)
        status_log.append("STAR model loaded.")
        yield None, "\n".join(status_log)

        progress(0.15, desc="Extracting frames...")
        frame_count, input_fps, frame_files = extract_frames(current_input_video_for_frames, input_frames_dir)
        status_log.append(f"Extracted {frame_count} frames at {input_fps:.2f} FPS.")
        yield None, "\n".join(status_log)

        progress(0.2, desc="Upscaling frames...")
        total_noise_levels = 900
        
        all_lr_frames_bgr_for_preprocess = [] # Renamed for clarity: stores BGR frames
        for frame_filename in frame_files:
            frame_lr_bgr = cv2.imread(os.path.join(input_frames_dir, frame_filename))
            if frame_lr_bgr is None:
                logger.error(f"Could not read frame {frame_filename} from {input_frames_dir}. Skipping.")
                # Add a placeholder; its color format for this specific error case is less critical
                all_lr_frames_bgr_for_preprocess.append(np.zeros((orig_h, orig_w, 3), dtype=np.uint8)) 
                continue
            # frame_lr_rgb = cv2.cvtColor(frame_lr_bgr, cv2.COLOR_BGR2RGB) # REMOVED: No explicit RGB conversion here
            all_lr_frames_bgr_for_preprocess.append(frame_lr_bgr) # Store BGR frames
        
        if len(all_lr_frames_bgr_for_preprocess) != frame_count:
             logger.warning(f"Mismatch in frame count and loaded LR frames for colorfix: {len(all_lr_frames_bgr_for_preprocess)} vs {frame_count}")

        if enable_tiling:
            status_log.append(f"Tiling enabled: Tile Size={tile_size}, Overlap={tile_overlap}.")
            yield None, "\n".join(status_log)
            for i, frame_filename in enumerate(progress.tqdm(frame_files, desc="Frames (Tiling)")):
                frame_lr_bgr = cv2.imread(os.path.join(input_frames_dir, frame_filename))
                if frame_lr_bgr is None: 
                    logger.warning(f"Skipping frame {frame_filename} due to read error during tiling.")
                    # Copy LR frame if SR fails for this frame to maintain sequence
                    placeholder_path = os.path.join(input_frames_dir, frame_filename)
                    if os.path.exists(placeholder_path):
                        shutil.copy2(placeholder_path, os.path.join(output_frames_dir, frame_filename))
                    continue
                # frame_lr_rgb = cv2.cvtColor(frame_lr_bgr, cv2.COLOR_BGR2RGB) # REMOVED
                single_lr_frame_tensor_norm = preprocess([frame_lr_bgr]) # Pass BGR frame to preprocess
                
                spliter = ImageSpliterTh(single_lr_frame_tensor_norm, int(tile_size), int(tile_overlap), sf=upscale_factor)

                for patch_lr_tensor_norm, patch_coords in progress.tqdm(spliter, desc=f"Patches Frame {i+1}"):
                    patch_lr_video_data = patch_lr_tensor_norm.unsqueeze(2) 

                    patch_pre_data = {'video_data': patch_lr_video_data, 'y': final_prompt,
                                      'target_res': (int(round(patch_lr_tensor_norm.shape[-2] * upscale_factor)), 
                                                     int(round(patch_lr_tensor_norm.shape[-1] * upscale_factor)))} # target_res based on patch size * sf
                    patch_data_tensor_cuda = collate_fn(patch_pre_data, 'cuda:0')
                    
                    with torch.no_grad():
                        patch_sr_tensor_bcthw = star_model.test( 
                            patch_data_tensor_cuda, total_noise_levels, steps=steps, solver_mode=solver_mode,
                            guide_scale=cfg_scale, max_chunk_len=1, vae_decoder_chunk_size=1
                        )
                    patch_sr_frames_uint8 = tensor2vid(patch_sr_tensor_bcthw)
                    
                    if color_fix_method != 'None':
                        if color_fix_method == 'AdaIN':
                            patch_sr_frames_uint8 = adain_color_fix(patch_sr_frames_uint8, patch_lr_video_data)
                        elif color_fix_method == 'Wavelet':
                            patch_sr_frames_uint8 = wavelet_color_fix(patch_sr_frames_uint8, patch_lr_video_data)

                    result_patch_chw_01 = torch.from_numpy(patch_sr_frames_uint8.cpu().numpy()[0,0]).permute(2,0,1).float() / 255.0
                    spliter.update_gaussian(result_patch_chw_01.unsqueeze(0), patch_coords) 
                    
                    del patch_data_tensor_cuda, patch_sr_tensor_bcthw, patch_sr_frames_uint8
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                final_frame_tensor_chw = spliter.gather() 
                final_frame_np_hwc_uint8 = (final_frame_tensor_chw.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                final_frame_bgr = cv2.cvtColor(final_frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_frames_dir, frame_filename), final_frame_bgr)
                status_log[-1] = f"Tiling: Processed frame {i+1}/{frame_count}"
                yield None, "\n".join(status_log)


        elif enable_sliding_window:
            status_log.append(f"Sliding Window: Size={window_size}, Step={window_step}.")
            yield None, "\n".join(status_log)
            processed_frame_filenames = [None] * frame_count 

            effective_window_size = int(window_size)
            effective_window_step = int(window_step)
            
            n_windows = math.ceil(frame_count / effective_window_step) if effective_window_step > 0 else 1
            if effective_window_step == 0 and frame_count > 0 : n_windows = 1 # Process as one large window if step is 0

            # Overlap calculation for saving frames from window
            save_from_start_offset = 0
            save_to_end_offset = effective_window_size
            if n_windows > 1 and effective_window_step < effective_window_size : # Overlap exists
                overlap_amount = effective_window_size - effective_window_step
                save_from_start_offset = overlap_amount // 2
                save_to_end_offset = effective_window_size - (overlap_amount - save_from_start_offset)


            for i in progress.tqdm(range(0, frame_count, effective_window_step), desc="Windows"):
                start_idx = i
                end_idx = min(i + effective_window_size, frame_count)
                current_window_len = end_idx - start_idx

                if current_window_len == 0: continue
                
                # Adjust last window to be full size if possible, by shifting its start
                is_last_window_iteration = (i + effective_window_step >= frame_count)
                if is_last_window_iteration and current_window_len < effective_window_size and frame_count >= effective_window_size:
                    start_idx = max(0, frame_count - effective_window_size)
                    end_idx = frame_count
                    current_window_len = end_idx - start_idx


                window_frame_names = frame_files[start_idx:end_idx]
                # Ensure all_lr_frames_for_colorfix has enough frames for this window
                if end_idx > len(all_lr_frames_bgr_for_preprocess): # Check against the BGR list
                    logger.error(f"Sliding window range {start_idx}-{end_idx} exceeds available LR frames {len(all_lr_frames_bgr_for_preprocess)}")
                    continue # Skip this malformed window
                
                window_lr_frames_bgr = [all_lr_frames_bgr_for_preprocess[j] for j in range(start_idx, end_idx)] # Get BGR frames
                if not window_lr_frames_bgr: continue

                window_lr_video_data = preprocess(window_lr_frames_bgr) # Pass BGR frames to preprocess

                window_pre_data = {'video_data': window_lr_video_data, 'y': final_prompt,
                                   'target_res': (final_h, final_w)}
                window_data_cuda = collate_fn(window_pre_data, 'cuda:0')

                with torch.no_grad():
                    window_sr_tensor_bcthw = star_model.test(
                        window_data_cuda, total_noise_levels, steps=steps, solver_mode=solver_mode,
                        guide_scale=cfg_scale, max_chunk_len=current_window_len, vae_decoder_chunk_size=min(vae_chunk, current_window_len)
                    )
                window_sr_frames_uint8 = tensor2vid(window_sr_tensor_bcthw) 

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        window_sr_frames_uint8 = adain_color_fix(window_sr_frames_uint8, window_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        window_sr_frames_uint8 = wavelet_color_fix(window_sr_frames_uint8, window_lr_video_data)
                
                # Determine which frames from this window's output to save (local indices within window_sr_frames_uint8)
                local_save_start = 0
                local_save_end = current_window_len

                if n_windows > 1:
                    if i == 0: # First window
                        local_save_end = save_to_end_offset
                    elif is_last_window_iteration: # Last window
                        local_save_start = save_from_start_offset if current_window_len == effective_window_size else 0 # If it was adjusted, take all
                    else: # Middle window
                        local_save_start = save_from_start_offset
                        local_save_end = save_to_end_offset
                
                local_save_start = max(0, min(local_save_start, current_window_len -1))
                local_save_end = max(local_save_start, min(local_save_end, current_window_len))


                for k_local in range(local_save_start, local_save_end):
                    k_global = start_idx + k_local
                    if k_global < frame_count and processed_frame_filenames[k_global] is None: 
                        frame_np_hwc_uint8 = window_sr_frames_uint8[k_local].cpu().numpy()
                        frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                        out_f_path = os.path.join(output_frames_dir, frame_files[k_global])
                        cv2.imwrite(out_f_path, frame_bgr)
                        processed_frame_filenames[k_global] = frame_files[k_global]

                del window_data_cuda, window_sr_tensor_bcthw, window_sr_frames_uint8
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                status_log[-1] = f"Sliding Window: Processed window for frames {start_idx}-{end_idx-1}/{frame_count}"
                yield None, "\n".join(status_log)

            for idx, fname in enumerate(frame_files):
                if processed_frame_filenames[idx] is None:
                    logger.warning(f"Frame {fname} (index {idx}) was not processed by sliding window, copying LR frame.")
                    lr_frame_path = os.path.join(input_frames_dir, fname)
                    if os.path.exists(lr_frame_path):
                         shutil.copy2(lr_frame_path, os.path.join(output_frames_dir, fname))
                    else: # Fallback: create empty frame or error
                         logger.error(f"LR frame {lr_frame_path} not found for fallback copy.")


        else: # Normal Chunked Processing
            status_log.append("Normal chunked processing.")
            yield None, "\n".join(status_log)
            num_chunks = math.ceil(frame_count / max_chunk_len)
            for i in progress.tqdm(range(num_chunks), desc="Chunks"):
                start_idx = i * max_chunk_len
                end_idx = min((i + 1) * max_chunk_len, frame_count)
                current_chunk_len = end_idx - start_idx
                if current_chunk_len == 0: continue
                
                if end_idx > len(all_lr_frames_bgr_for_preprocess): # Check against BGR list
                     logger.error(f"Chunk range {start_idx}-{end_idx} exceeds available LR frames {len(all_lr_frames_bgr_for_preprocess)}")
                     continue

                chunk_lr_frames_bgr = all_lr_frames_bgr_for_preprocess[start_idx:end_idx] # Get BGR frames
                if not chunk_lr_frames_bgr: continue

                chunk_lr_video_data = preprocess(chunk_lr_frames_bgr) # Pass BGR frames to preprocess

                chunk_pre_data = {'video_data': chunk_lr_video_data, 'y': final_prompt,
                                  'target_res': (final_h, final_w)}
                chunk_data_cuda = collate_fn(chunk_pre_data, 'cuda:0')

                with torch.no_grad():
                    chunk_sr_tensor_bcthw = star_model.test(
                        chunk_data_cuda, total_noise_levels, steps=steps, solver_mode=solver_mode,
                        guide_scale=cfg_scale, max_chunk_len=current_chunk_len, vae_decoder_chunk_size=min(vae_chunk, current_chunk_len)
                    )
                chunk_sr_frames_uint8 = tensor2vid(chunk_sr_tensor_bcthw) 

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        chunk_sr_frames_uint8 = adain_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        chunk_sr_frames_uint8 = wavelet_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)
                
                for k, frame_name in enumerate(frame_files[start_idx:end_idx]):
                    frame_np_hwc_uint8 = chunk_sr_frames_uint8[k].cpu().numpy()
                    frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_frames_dir, frame_name), frame_bgr)
                
                del chunk_data_cuda, chunk_sr_tensor_bcthw, chunk_sr_frames_uint8
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                status_log[-1] = f"Chunked Processing: Processed chunk {i+1}/{num_chunks} (frames {start_idx}-{end_idx-1})"
                yield None, "\n".join(status_log)


        progress(0.9, desc="Reassembling video...")
        status_log.append("Reassembling final video...")
        yield None, "\n".join(status_log)
        create_video_from_frames(output_frames_dir, output_path, input_fps)
        
        status_log.append(f"Upscaled video saved to: {output_path}")
        progress(1.0, "Finished!")
        yield output_path, "\n".join(status_log)

    except gr.Error as e: 
        logger.error(f"A Gradio UI Error occurred: {e}", exc_info=True)
        status_log.append(f"Error: {e}")
        yield None, "\n".join(status_log) 
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during upscaling: {e}", exc_info=True)
        status_log.append(f"Critical Error: {e}")
        yield None, "\n".join(status_log)
        raise gr.Error(f"Upscaling failed critically: {e}")
    finally:
        # cleanup_temp_dir(temp_dir) # TEMP: Disabled for debugging
        # downscaled_temp_video is inside temp_dir, so it's removed by cleanup_temp_dir
        # if downscaled_temp_video and os.path.exists(downscaled_temp_video):
        #     try: os.remove(downscaled_temp_video) # No longer needed if temp_dir is removed
        #     except: pass
        if star_model is not None:
            try:
                if hasattr(star_model, 'to'): star_model.to('cpu')
                del star_model
            except: pass
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("STAR upscaling process finished and cleaned up.")
        # Final status update if an error occurred before output_path was yielded
        # Check if 'output_path' was defined and if it exists, otherwise update status
        if 'output_path' not in locals() or not os.path.exists(output_path):
             if status_log: # Ensure status_log is not empty
                yield None, "\n".join(status_log)


# --- Gradio UI Definition ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="zinc", neutral_hue="zinc")) as demo:
    gr.Markdown("# 🌟 STAR Video Upscaler")
    gr.Markdown("Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Inputs & Settings")
            with gr.Group():
                input_video = gr.Video(
                    label="Input Video",
                    sources=["upload"],
                    interactive=True
                )
                user_prompt = gr.Textbox(
                    label="Describe the Video Content (Prompt)",
                    lines=3,
                    placeholder="e.g., A panda playing guitar by a lake at sunset.",
                    info="""Describe the main subject and action in the video. This guides the upscaling process.
Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens.
If CogVLM2 is available, you can use the button below to generate a caption automatically."""
                )
                if COG_VLM_AVAILABLE:
                    auto_caption_btn = gr.Button("Generate Caption with CogVLM2")
                    caption_status = gr.Textbox(label="Captioning Status", interactive=False, visible=False)

            with gr.Accordion("Prompt Settings", open=True):
                 pos_prompt = gr.Textbox(
                     label="Default Positive Prompt (Appended)",
                     value=DEFAULT_POS_PROMPT,
                     lines=2,
                     info="""Appended to your 'Describe Video Content' prompt. Focuses on desired quality aspects (e.g., realism, detail).
The total combined prompt length is limited to 77 tokens."""
                 )
                 neg_prompt = gr.Textbox(
                     label="Default Negative Prompt (Appended)",
                     value=DEFAULT_NEG_PROMPT,
                     lines=2,
                     info="Guides the model *away* from undesired aspects (e.g., bad quality, artifacts, specific styles). This does NOT count towards the 77 token limit for positive guidance."
                 )

            with gr.Group():
                gr.Markdown("### Core Upscaling Settings")
                model_selector = gr.Dropdown(
                    label="STAR Model",
                    choices=["Light Degradation", "Heavy Degradation"],
                    value="Light Degradation",
                    info="""Choose the model based on input video quality.
'Light Degradation': Better for relatively clean inputs (e.g., downloaded web videos).
'Heavy Degradation': Better for inputs with significant compression artifacts, noise, or blur."""
                )
                upscale_factor_slider = gr.Slider(
                    label="Upscale Factor (if Target Res disabled)",
                    minimum=1.0, maximum=8.0, value=4.0, step=0.1,
                    info="Simple multiplication factor for output resolution if 'Enable Max Target Resolution' is OFF. E.g., 4.0 means 4x height and 4x width."
                )
                cfg_slider = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                    info="Controls how strongly the model follows your combined text prompt. Higher values mean stricter adherence, lower values allow more creativity. Typical values: 5.0-10.0."
                )
                solver_mode_radio = gr.Radio(
                    label="Solver Mode",
                    choices=['fast', 'normal'], value='fast',
                    info="""Diffusion solver type.
'fast': Fewer steps (default ~15), much faster, good quality usually.
'normal': More steps (default ~50), slower, potentially slightly better detail/coherence."""
                )
                steps_slider = gr.Slider(
                    label="Diffusion Steps",
                    minimum=5, maximum=100, value=15, step=1,
                    info="Number of denoising steps. Value changes automatically based on Solver Mode (Fast: ~15, Normal: ~50). Higher steps take longer."
                )
                color_fix_dropdown = gr.Dropdown(
                    label="Color Correction",
                    choices=['AdaIN', 'Wavelet', 'None'], value='AdaIN',
                    info="""Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""
                 )

            with gr.Accordion("Performance & VRAM Optimization", open=True):
                max_chunk_len_slider = gr.Slider(
                    label="Max Frames per Batch (VRAM)",
                    minimum=4, maximum=96, value=32, step=4,
                    info="""IMPORTANT for VRAM. This is the standard way the application manages VRAM. It divides the entire sequence of video frames into sequential, non-overlapping chunks.
- Mechanism: The STAR model processes one complete chunk (of this many frames) at a time.
- VRAM Impact: High Reduction (Lower value = Less VRAM).
- Quality Impact: Can reduce Temporal Consistency (flicker/motion issues) between chunks if too low, as the model doesn't have context across chunk boundaries. Keep as high as VRAM allows.
- Speed Impact: Slower (Lower value = Slower, as more chunks are processed)."""
                )
                vae_chunk_slider = gr.Slider(
                    label="VAE Decode Chunk (VRAM)",
                    minimum=1, maximum=16, value=3, step=1,
                    info="""Controls max latent frames decoded back to pixels by VAE simultaneously.
- VRAM Impact: High Reduction (Lower value = Less VRAM during decode stage).
- Quality Impact: Minimal / Negligible. Safe to lower.
- Speed Impact: Slower (Lower value = Slower decoding)."""
                )

            with gr.Accordion("Advanced: Target Resolution", open=True):
                 enable_target_res_check = gr.Checkbox(
                     label="Enable Max Target Resolution",
                     value=True,
                     info="Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
                 )
                 target_res_mode_radio = gr.Radio(
                     label="Target Resolution Mode",
                     choices=['Ratio Upscale', 'Downscale then 4x'], value='Downscale then 4x', interactive=False,
                     info="""How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio.
'Downscale then 4x': If input is large, downscales it towards Target H/W divided by 4, THEN applies a 4x upscale. Can clean noisy high-res input before upscaling."""
                 )
                 with gr.Row():
                     target_h_num = gr.Number(
                         label="Max Target Height (px)",
                         value=1920, minimum=128, step=16, interactive=False,
                         info="""Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
    - VRAM Impact: Very High (Lower value = Less VRAM).
    - Quality Impact: Direct (Lower value = Less detail).
    - Speed Impact: Faster (Lower value = Faster)."""
                     )
                     target_w_num = gr.Number(
                         label="Max Target Width (px)",
                         value=1920, minimum=128, step=16, interactive=False,
                         info="""Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
    - VRAM Impact: Very High (Lower value = Less VRAM).
    - Quality Impact: Direct (Lower value = Less detail).
    - Speed Impact: Faster (Lower value = Faster)."""
                     )


            with gr.Accordion("Advanced: Tiling (Very High Res / Low VRAM)", open=True):
                 enable_tiling_check = gr.Checkbox(
                     label="Enable Tiled Upscaling",
                     value=False,
                     info="""Processes each frame in small spatial patches (tiles). Use ONLY if necessary for extreme resolutions or very low VRAM.
- VRAM Impact: Very High Reduction.
- Quality Impact: High risk of tile seams/artifacts. Can harm global coherence and severely reduce temporal consistency.
- Speed Impact: Extremely Slow."""
                )
                 tile_size_num = gr.Number(
                     label="Tile Size (px, input res)",
                     value=256, minimum=64, step=32, interactive=False,
                     info="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                 )
                 tile_overlap_num = gr.Number(
                     label="Tile Overlap (px, input res)",
                     value=64, minimum=0, step=16, interactive=False,
                     info="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                 )

            with gr.Accordion("Advanced: Sliding Window (Long Videos)", open=True):
                 enable_sliding_window_check = gr.Checkbox(
                     label="Enable Sliding Window",
                     value=False,
                     info="""Processes the video in overlapping temporal chunks (windows). Use for very long videos where 'Max Frames per Batch' isn't enough or causes too many artifacts.
- Mechanism: Takes a 'Window Size' of frames, processes it, saves results from the central part, then slides the window forward by 'Window Step', processing overlapping frames.
- VRAM Impact: High Reduction (limits frames processed temporally, similar to Max Frames per Batch but with overlap).
- Quality Impact: Moderate risk of discontinuities at window boundaries if overlap (Window Size - Window Step) is small. Aims for better consistency than small non-overlapping chunks.
- Speed Impact: Slower (due to processing overlapping frames multiple times). When enabled, 'Window Size' dictates batch size instead of 'Max Frames per Batch'."""
                 )
                 window_size_num = gr.Slider(
                     label="Window Size (frames)",
                     value=32, minimum=2, step=4, interactive=False,
                     info="Number of frames in each temporal window. Acts like 'Max Frames per Batch' but applied as a sliding window. Lower value = less VRAM, less temporal context."
                 )
                 window_step_num = gr.Slider(
                     label="Window Step (frames)",
                     value=16, minimum=1, step=1, interactive=False,
                     info="How many frames to advance for the next window. (Window Size - Window Step) = Overlap. Smaller step = more overlap = better consistency but slower. Recommended: Step = Size / 2."
                 )
            
            if COG_VLM_AVAILABLE:
                with gr.Accordion("Auto-Captioning Settings (CogVLM2)", open=True):
                    cogvlm_quant_choices_map = {0: "FP16/BF16"}
                    if torch.cuda.is_available() and BITSANDBYTES_AVAILABLE:
                        cogvlm_quant_choices_map[4] = "INT4 (CUDA)"
                        cogvlm_quant_choices_map[8] = "INT8 (CUDA)"
                    
                    # Ensure choices are available ones
                    cogvlm_quant_radio_choices_display = list(cogvlm_quant_choices_map.values())
                    # Default to INT4 if available, else FP16/BF16
                    default_quant_display = cogvlm_quant_choices_map.get(4, cogvlm_quant_choices_map.get(0))


                    cogvlm_quant_radio = gr.Radio(
                        label="CogVLM2 Quantization",
                        choices=cogvlm_quant_radio_choices_display,
                        value=default_quant_display,
                        info="Quantization for the CogVLM2 captioning model (uses less VRAM). INT4/8 require CUDA & bitsandbytes.",
                        interactive=True if len(cogvlm_quant_radio_choices_display) > 1 else False
                    )
                    cogvlm_unload_radio = gr.Radio(
                        label="CogVLM2 After-Use",
                        choices=['full', 'cpu'], value='full',
                        info="""Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""
                    )
            else:
                gr.Markdown("_(Auto-captioning disabled as CogVLM2 components are not fully available.)_")


            upscale_button = gr.Button("Upscale Video", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## Output")
            output_video = gr.Video(label="Upscaled Video", interactive=False)
            status_textbox = gr.Textbox(label="Log", interactive=False, lines=8, max_lines=20)

    # --- Event Listeners ---
    def update_steps_display(mode):
        return gr.Slider(value=15 if mode == 'fast' else 50) 
    solver_mode_radio.change(update_steps_display, solver_mode_radio, steps_slider)

    enable_target_res_check.change(lambda x: [gr.update(interactive=x)]*3, inputs=enable_target_res_check, outputs=[target_h_num, target_w_num, target_res_mode_radio])
    enable_tiling_check.change(lambda x: [gr.update(interactive=x)]*2, inputs=enable_tiling_check, outputs=[tile_size_num, tile_overlap_num])
    enable_sliding_window_check.change(lambda x: [gr.update(interactive=x)]*2, inputs=enable_sliding_window_check, outputs=[window_size_num, window_step_num])

    # --- Precision Note for User ---
    # The STAR model uses automatic mixed precision (FP16/FP32) on CUDA by default (`cfg.use_fp16 = True`).
    # There is no UI toggle for this; it's automatically enabled for performance and VRAM savings.
    # Forcing full FP16 is generally not recommended due to potential instability.

    upscale_button.click(
        fn=run_upscale,
        inputs=[
            input_video, user_prompt, pos_prompt, neg_prompt, model_selector,
            upscale_factor_slider, cfg_slider, steps_slider, solver_mode_radio,
            max_chunk_len_slider, vae_chunk_slider, color_fix_dropdown,
            enable_tiling_check, tile_size_num, tile_overlap_num,
            enable_sliding_window_check, window_size_num, window_step_num,
            enable_target_res_check, target_h_num, target_w_num, target_res_mode_radio
        ],
        outputs=[output_video, status_textbox]
    )

    if COG_VLM_AVAILABLE:
        # Invert the map for lookup
        cogvlm_display_to_quant_val_map = {v: k for k, v in cogvlm_quant_choices_map.items()}
        
        def get_quant_value_from_display(display_val):
            return cogvlm_display_to_quant_val_map.get(display_val, 0) # Default to 0 if not found

        auto_caption_btn.click(
            fn=lambda vid, quant_display, unload_strat, progress=gr.Progress(track_tqdm=True): auto_caption(vid, get_quant_value_from_display(quant_display), unload_strat, progress),
            inputs=[input_video, cogvlm_quant_radio, cogvlm_unload_radio],
            outputs=[user_prompt, caption_status]
        ).then(lambda: gr.update(visible=True), outputs=caption_status)


if __name__ == "__main__":
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Gradio App Starting. Default output to: {os.path.abspath(DEFAULT_OUTPUT_DIR)}")
    logger.info(f"STAR Models expected at: {LIGHT_DEG_MODEL}, {HEAVY_DEG_MODEL}")
    if COG_VLM_AVAILABLE:
        logger.info(f"CogVLM2 Model expected at: {COG_VLM_MODEL_PATH}")
    demo.queue().launch(debug=True, max_threads=100, inbrowser=True)

# --- END OF FILE secourses_app.py ---
