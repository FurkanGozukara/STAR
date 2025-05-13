# --- START OF FILE secourses_app.py ---

import gradio as gr
import os
import platform
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
import gc # Import gc for garbage collection
from easydict import EasyDict
from argparse import ArgumentParser, Namespace
import logging

parser = ArgumentParser(description="Ultimate SECourses STAR Video Upscaler")
parser.add_argument('--share', action='store_true', help="Enable Gradio live share")
parser.add_argument('--outputs_folder', type=str, default="outputs", help="Main folder for output videos and related files")
args = parser.parse_args()

# Helper function to format seconds into HH:MM:SS
def format_time(seconds):
    if seconds is None or seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

try:

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = script_dir 

    if not os.path.isdir(os.path.join(base_path, 'video_to_video')):
        print(f"Warning: 'video_to_video' directory not found in inferred base_path: {base_path}. Attempting to use parent directory.")
        base_path = os.path.dirname(base_path) 
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

COG_VLM_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from decord import cpu, VideoReader, bridge
    import io 
    try:
        from transformers import BitsAndBytesConfig
        BITSANDBYTES_AVAILABLE = True
    except ImportError:
        print("Warning: bitsandbytes not found. INT4/INT8 quantization for CogVLM2 will not be available.")
    COG_VLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CogVLM2 related components (transformers, decord) not fully found: {e}")
    print("Auto-captioning feature will be disabled.")


logger = get_logger()
# --- Diagnostic: Force logger level and console handler level ---
logger.setLevel(logging.INFO)
found_stream_handler = False
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
        found_stream_handler = True
        logger.info("Diagnostic: Explicitly set StreamHandler level to INFO.")
if not found_stream_handler:
    # If no stream handler was found on our specific logger, let's add one.
    # This is unlikely given the logger.py code but good for robustness.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Assuming 'formatter' is defined globally in logger.py, 
    # we might need to define a basic one here if it's not accessible.
    # For now, let's rely on the default formatter or one potentially set by get_logger.
    # If logs appear unformatted, this is a point to revisit.
    logger.addHandler(ch)
    logger.info("Diagnostic: No StreamHandler found, added a new one with INFO level.")
logger.info(f"Logger '{logger.name}' configured with level: {logging.getLevelName(logger.level)}. Handlers: {logger.handlers}")
# --- End Diagnostic ---

DEFAULT_OUTPUT_DIR = "upscaled_videos"

DEFAULT_OUTPUT_DIR = os.path.abspath(args.outputs_folder)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True) 

COG_VLM_MODEL_PATH = os.path.join(base_path, 'models', 'cogvlm2-video-llama3-chat') 
LIGHT_DEG_MODEL = os.path.join(base_path, 'pretrained_weight', 'light_deg.pt')
HEAVY_DEG_MODEL = os.path.join(base_path, 'pretrained_weight', 'heavy_deg.pt')

DEFAULT_POS_PROMPT = star_cfg.positive_prompt
DEFAULT_NEG_PROMPT = star_cfg.negative_prompt

if not os.path.exists(LIGHT_DEG_MODEL):
     logger.error(f"FATAL: Light degradation model not found at {LIGHT_DEG_MODEL}.")
if not os.path.exists(HEAVY_DEG_MODEL):
     logger.error(f"FATAL: Heavy degradation model not found at {HEAVY_DEG_MODEL}.")

cogvlm_model_state = {"model": None, "tokenizer": None, "device": None, "quant": None}
cogvlm_lock = threading.RLock() # Changed to RLock for re-entrancy

def run_ffmpeg_command(cmd, desc="ffmpeg command"):
    logger.info(f"Running {desc}: {cmd}")
    try:
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if process.stdout: logger.info(f"{desc} stdout: {process.stdout.strip()}")
        if process.stderr: logger.info(f"{desc} stderr: {process.stderr.strip()}")
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

def open_folder(folder_path):
    logger.info(f"Attempting to open folder: {folder_path}")
    if not os.path.isdir(folder_path):
        logger.warning(f"Folder does not exist or is not a directory: {folder_path}")
        gr.Warning(f"Output folder '{folder_path}' does not exist yet. Please run an upscale first.")
        return
    try:
        if sys.platform == "win32":
            os.startfile(os.path.normpath(folder_path))
        elif sys.platform == "darwin": 
            subprocess.run(['open', folder_path], check=True)
        else: 
            subprocess.run(['xdg-open', folder_path], check=True)
        logger.info(f"Successfully requested to open folder: {folder_path}")
    except FileNotFoundError:
        logger.error(f"File explorer command (e.g., xdg-open, open) not found for platform {sys.platform}. Cannot open folder.")
        gr.Error(f"Could not find a file explorer utility for your system ({sys.platform}).")
    except Exception as e:
        logger.error(f"Failed to open folder '{folder_path}': {e}")
        gr.Error(f"Failed to open folder: {e}")

def extract_frames(video_path, temp_dir):
    logger.info(f"Extracting frames from '{video_path}' to '{temp_dir}'")
    os.makedirs(temp_dir, exist_ok=True)
    fps = 30.0 
    try:
        probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        process = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        rate_str = process.stdout.strip()
        if '/' in rate_str:
            num, den = map(int, rate_str.split('/'))
            if den != 0: fps = num / den
        elif rate_str: 
            fps = float(rate_str)
        logger.info(f"Detected FPS: {fps}")
    except Exception as e:
        logger.warning(f"Could not get FPS using ffprobe for '{video_path}': {e}. Using default {fps} FPS.")

    cmd = f'ffmpeg -i "{video_path}" -vsync vfr -qscale:v 2 "{os.path.join(temp_dir, "frame_%06d.png")}"' 
    run_ffmpeg_command(cmd, "Frame Extraction")

    frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')])
    frame_count = len(frame_files)
    logger.info(f"Extracted {frame_count} frames.")
    if frame_count == 0:
        raise gr.Error("Failed to extract any frames. Check video file and ffmpeg installation.")
    return frame_count, fps, frame_files

def create_video_from_frames(frame_dir, output_path, fps, ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu):
    logger.info(f"Creating video from frames in '{frame_dir}' to '{output_path}' at {fps} FPS with preset: {ffmpeg_preset}, quality: {ffmpeg_quality_value}, GPU: {ffmpeg_use_gpu}")
    input_pattern = os.path.join(frame_dir, "frame_%06d.png")
    
    video_codec_opts = ""
    if ffmpeg_use_gpu:
        nvenc_preset = ffmpeg_preset
        if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
            nvenc_preset = "fast" 
        elif ffmpeg_preset in ["slower", "veryslow"]:
            nvenc_preset = "slow"
        
        video_codec_opts = f'-c:v h264_nvenc -preset:v {nvenc_preset} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
        logger.info(f"Using NVIDIA NVENC with preset {nvenc_preset} and CQ {ffmpeg_quality_value}.")
    else:
        video_codec_opts = f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'
        logger.info(f"Using libx264 with preset {ffmpeg_preset} and CRF {ffmpeg_quality_value}.")

    cmd = f'ffmpeg -y -framerate {fps} -i "{input_pattern}" {video_codec_opts} "{output_path}"'
    run_ffmpeg_command(cmd, "Video Reassembly (silent)")

def get_next_filename(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    max_num = 0
    existing_mp4_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    for f_name in existing_mp4_files:
        try:
            num = int(os.path.splitext(f_name)[0])
            if num > max_num: max_num = num
        except ValueError: continue

    current_num_to_try = max_num + 1
    while True:
        base_filename_no_ext = f"{current_num_to_try:04d}"
        tmp_lock_file_path = os.path.join(output_dir, f"{base_filename_no_ext}.tmp")
        full_output_path = os.path.join(output_dir, f"{base_filename_no_ext}.mp4")

        try:
            fd = os.open(tmp_lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            logger.info(f"Successfully created lock file: {tmp_lock_file_path}")
            return base_filename_no_ext, full_output_path
        except FileExistsError:
            logger.warning(f"Lock file {tmp_lock_file_path} already exists. Trying next number.")
            current_num_to_try += 1
        except Exception as e:
            logger.error(f"Error trying to create lock file {tmp_lock_file_path}: {e}")
            current_num_to_try += 1 
            if current_num_to_try > max_num + 1000: 
                 logger.error("Failed to secure a lock file after many attempts. Aborting get_next_filename.")
                 raise IOError("Could not secure a unique filename lock.")

def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir) and os.path.isdir(temp_dir): 
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
            unload_cogvlm_model('full') # RLock allows this call now

        try:
            logger.info(f"Loading tokenizer from: {COG_VLM_MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(COG_VLM_MODEL_PATH, trust_remote_code=True)
            # Tokenizer stored in global state later, after model is successfully loaded
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
                 quantization = 0 

            
            current_device_map = None
            if bnb_config and device == 'cuda':
                current_device_map = "auto" # Recommended for BNB
            
            effective_low_cpu_mem_usage = True if bnb_config else False # low_cpu_mem_usage useful with device_map or BNB

            logger.info(f"Preparing to load model from: {COG_VLM_MODEL_PATH} with quant: {quantization}, dtype: {model_dtype}, device: {device}, device_map: {current_device_map}, low_cpu_mem: {effective_low_cpu_mem_usage}")
            
            model = AutoModelForCausalLM.from_pretrained(
                COG_VLM_MODEL_PATH,
                torch_dtype=model_dtype if device == 'cuda' else torch.float32, # float32 for CPU
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=effective_low_cpu_mem_usage,
                device_map=current_device_map 
            )

            if not bnb_config and current_device_map is None : 
                logger.info(f"Moving non-quantized model to target device: {device}")
                model = model.to(device)
            elif bnb_config and current_device_map == "auto":
                 logger.info(f"BNB model loaded with device_map='auto'. Should be on target CUDA device(s).")
            # If device_map is e.g. {"": "cuda:0"} and not BNB, model is already on device.

            model = model.eval()
            
            # Store in global state only after successful load
            cogvlm_model_state["model"] = model
            cogvlm_model_state["tokenizer"] = tokenizer # Store tokenizer now
            cogvlm_model_state["device"] = device 
            cogvlm_model_state["quant"] = quantization
            
            final_device_str = "N/A"
            final_dtype_str = "N/A"
            try: # Try to get device and dtype info
                first_param = next(model.parameters(), None)
                if hasattr(model, 'device') and isinstance(model.device, torch.device): # For non-sharded, non-accelerate models
                    final_device_str = str(model.device)
                elif hasattr(model, 'hf_device_map'): # For models loaded with device_map by accelerate
                    final_device_str = str(model.hf_device_map) # Or extract primary device
                elif first_param is not None and hasattr(first_param, 'device'):
                     final_device_str = str(first_param.device)

                if first_param is not None and hasattr(first_param, 'dtype'):
                    final_dtype_str = str(first_param.dtype)
            except Exception as e_dev_dtype:
                logger.warning(f"Could not reliably determine final model device/dtype: {e_dev_dtype}")

            logger.info(f"CogVLM2 model loaded (Quant: {quantization}, Requested Device: {device}, Final Device(s): {final_device_str}, Dtype: {final_dtype_str}).")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load CogVLM2 model from path: {COG_VLM_MODEL_PATH}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {e}", exc_info=True)
            # Attempt cleanup even if loading failed partially
            # unload_cogvlm_model is called within the same lock context if an error occurs here.
            # To be safe, ensure it can be called if tokenizer was set but model failed.
            _model_ref = cogvlm_model_state.pop("model", None) # clear potentially partially set state
            _tokenizer_ref = cogvlm_model_state.pop("tokenizer", None)
            if _model_ref: del _model_ref
            if _tokenizer_ref: del _tokenizer_ref
            cogvlm_model_state.update({"model": None, "tokenizer": None, "device": None, "quant": None})
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise gr.Error(f"Could not load CogVLM2 model (check logs for details): {str(e)[:200]}")


def unload_cogvlm_model(strategy):
    global cogvlm_model_state
    logger.info(f"Unloading CogVLM2 model with strategy: {strategy}")
    with cogvlm_lock:
        if cogvlm_model_state.get("model") is None and cogvlm_model_state.get("tokenizer") is None:
            logger.info("CogVLM2 model/tokenizer not loaded or already unloaded.")
            return

        if strategy == 'cpu':
            try:
                model = cogvlm_model_state.get("model")
                if model is not None:
                    if cogvlm_model_state.get("quant") in [4, 8]:
                        logger.info("BNB quantized model cannot be moved to CPU. Keeping on GPU or use 'full' unload.")
                        return 
                    
                    if cogvlm_model_state.get("device") == 'cuda' and torch.cuda.is_available():
                        model.to('cpu')
                        cogvlm_model_state["device"] = 'cpu'
                        logger.info("CogVLM2 model moved to CPU.")
                    else:
                        logger.info("CogVLM2 model already on CPU or CUDA not available for move.")
                else:
                    logger.info("No model found in state to move to CPU.")
            except Exception as e:
                logger.error(f"Failed to move CogVLM2 model to CPU: {e}")
        elif strategy == 'full':
            model_obj = cogvlm_model_state.pop("model", None)
            tokenizer_obj = cogvlm_model_state.pop("tokenizer", None)
            
            cogvlm_model_state.update({"model": None, "tokenizer": None, "device": None, "quant": None})

            if model_obj is not None:
                del model_obj
                logger.info("Explicitly deleted popped model object.")
            if tokenizer_obj is not None:
                del tokenizer_obj
                logger.info("Explicitly deleted popped tokenizer object.")
            
            gc.collect() # Force Python garbage collection
            logger.info("Python garbage collection triggered.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache emptied.")
            logger.info("CogVLM2 model and tokenizer fully unloaded and CUDA cache (if applicable) cleared.")
        else:
            logger.warning(f"Unknown unload strategy: {strategy}")

def auto_caption(video_path, quantization, unload_strategy, progress=gr.Progress(track_tqdm=True)):
    if not COG_VLM_AVAILABLE:
        raise gr.Error("CogVLM2 components not available. Captioning disabled.")
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Please provide a valid video file for captioning.")

    cogvlm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if quantization in [4,8] and cogvlm_device == 'cpu':
        raise gr.Error("INT4/INT8 quantization requires CUDA. Please select FP16/BF16 for CPU or ensure CUDA is available.")

    caption = "Error: Caption generation failed."
    # Local variables for model and tokenizer to ensure proper scope and deletion
    local_model_ref = None
    local_tokenizer_ref = None
    inputs_on_device = None
    video_data_cog = None
    outputs_tensor = None 
    
    try:
        progress(0.1, desc="Loading CogVLM2 for captioning...")
        local_model_ref, local_tokenizer_ref = load_cogvlm_model(quantization, cogvlm_device)
        
        # Determine compute device and dtype from the loaded model
        model_compute_device = cogvlm_device # Default
        model_actual_dtype = torch.float32 # Default
        
        if local_model_ref is not None:
            first_param = next(local_model_ref.parameters(), None)
            if hasattr(local_model_ref, 'device') and isinstance(local_model_ref.device, torch.device):
                 model_compute_device = local_model_ref.device
            elif hasattr(local_model_ref, 'hf_device_map'): # For accelerate
                 # Heuristic: use the device of the first parameter or the main requested device
                 if first_param is not None: model_compute_device = first_param.device
                 else: model_compute_device = torch.device(cogvlm_device) # Fallback
            elif first_param is not None:
                 model_compute_device = first_param.device
            
            if first_param is not None:
                model_actual_dtype = first_param.dtype
            elif cogvlm_device == 'cuda': # Fallback for dtype
                model_actual_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >=8 else torch.float16


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
        video_data_cog = decord_vr.get_batch(frame_id_list).permute(3, 0, 1, 2) 

        query = "Please describe this video in detail."
        inputs = local_model_ref.build_conversation_input_ids( # Use local_model_ref
            tokenizer=local_tokenizer_ref, query=query, images=[video_data_cog], history=[], template_version='chat' # Use local_tokenizer_ref
        )
        
        inputs_on_device = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(model_compute_device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(model_compute_device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(model_compute_device),
            'images': [[inputs['images'][0].to(model_compute_device).to(model_actual_dtype)]],
        }

        gen_kwargs = {"max_new_tokens": 256, "pad_token_id": local_tokenizer_ref.eos_token_id or 128002, "top_k": 5, "do_sample": True, "top_p": 0.8, "temperature": 0.8}
        
        progress(0.6, desc="Generating caption with CogVLM2...")
        with torch.no_grad():
            outputs_tensor = local_model_ref.generate(**inputs_on_device, **gen_kwargs) # Use local_model_ref
        outputs_tensor = outputs_tensor[:, inputs_on_device['input_ids'].shape[1]:]
        caption = local_tokenizer_ref.decode(outputs_tensor[0], skip_special_tokens=True).strip() # Use local_tokenizer_ref
        logger.info(f"Generated Caption: {caption}")
        progress(0.9, desc="Caption generated.")

    except Exception as e:
        logger.error(f"Error during auto-captioning: {e}", exc_info=True)
        caption = f"Error during captioning: {str(e)[:100]}" 
    finally:
        progress(1.0, desc="Finalizing captioning...")

        if outputs_tensor is not None: del outputs_tensor
        if inputs_on_device is not None: inputs_on_device.clear(); del inputs_on_device
        if video_data_cog is not None: del video_data_cog
        
        # Delete local references to model and tokenizer if they were assigned
        if local_model_ref is not None: del local_model_ref
        if local_tokenizer_ref is not None: del local_tokenizer_ref
        
        unload_cogvlm_model(unload_strategy) # This will act on the global state
    return caption, f"Captioning status: {'Success' if not caption.startswith('Error') else caption}"


def run_upscale(
    input_video_path, user_prompt, positive_prompt, negative_prompt, model_choice,
    upscale_factor_slider, cfg_scale, steps, solver_mode,
    max_chunk_len, vae_chunk, color_fix_method,
    enable_tiling, tile_size, tile_overlap,
    enable_sliding_window, window_size, window_step,
    enable_target_res, target_h, target_w, target_res_mode,
    ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu,
    save_frames, save_metadata,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_video_path or not os.path.exists(input_video_path):
        raise gr.Error("Please upload a valid input video.")

    setup_seed(666)
    overall_process_start_time = time.time() 
    logger.info("Overall upscaling process started.")

    base_output_filename_no_ext, output_video_path = get_next_filename(DEFAULT_OUTPUT_DIR)
    
    run_id = f"star_run_{int(time.time())}_{np.random.randint(1000, 9999)}"
    temp_dir_base = tempfile.gettempdir()
    temp_dir = os.path.join(temp_dir_base, run_id)
    
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    
    frames_output_subfolder = None
    input_frames_permanent_save_path = None
    processed_frames_permanent_save_path = None
    if save_frames:
        frames_output_subfolder = os.path.join(DEFAULT_OUTPUT_DIR, base_output_filename_no_ext)
        input_frames_permanent_save_path = os.path.join(frames_output_subfolder, "input_frames")
        processed_frames_permanent_save_path = os.path.join(frames_output_subfolder, "processed_frames")
        os.makedirs(input_frames_permanent_save_path, exist_ok=True)
        os.makedirs(processed_frames_permanent_save_path, exist_ok=True)
        logger.info(f"Saving frames to: {frames_output_subfolder}")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    star_model = None
    current_input_video_for_frames = input_video_path 
    downscaled_temp_video = None
    status_log = ["Process started..."]

    try:
        progress(0, desc="Initializing...")
        status_log.append("Initializing upscaling process...")
        logger.info("Initializing upscaling process...")
        yield None, "\n".join(status_log) 

        final_prompt = (user_prompt.strip() + ". " + positive_prompt.strip()).strip()
        
        model_file_path = LIGHT_DEG_MODEL if model_choice == "Light Degradation" else HEAVY_DEG_MODEL
        if not os.path.exists(model_file_path):
            raise gr.Error(f"STAR model weight not found: {model_file_path}")

        orig_h, orig_w = get_video_resolution(input_video_path)
        status_log.append(f"Original resolution: {orig_w}x{orig_h}")
        logger.info(f"Original resolution: {orig_w}x{orig_h}")
        progress(0.05, desc="Calculating target resolution...")

        if enable_target_res:
            needs_downscale, ds_h, ds_w, upscale_factor, final_h, final_w = calculate_upscale_params(
                orig_h, orig_w, target_h, target_w, target_res_mode
            )
            status_log.append(f"Target resolution mode: {target_res_mode}. Calculated upscale: {upscale_factor:.2f}x. Target output: {final_w}x{final_h}")
            logger.info(f"Target resolution mode: {target_res_mode}. Calculated upscale: {upscale_factor:.2f}x. Target output: {final_w}x{final_h}")
            if needs_downscale:
                downscale_stage_start_time = time.time()
                progress(0.07, desc="Downscaling input video...")
                downscale_status_msg = f"Downscaling input to {ds_w}x{ds_h} before upscaling."
                status_log.append(downscale_status_msg)
                logger.info(downscale_status_msg)
                yield None, "\n".join(status_log)
                downscaled_temp_video = os.path.join(temp_dir, "downscaled_input.mp4")
                scale_filter = f"scale='trunc(iw*min({ds_w}/iw,{ds_h}/ih)/2)*2':'trunc(ih*min({ds_w}/iw,{ds_h}/ih)/2)*2'"
                
                ffmpeg_opts_downscale = ""
                if ffmpeg_use_gpu:
                    nvenc_preset_down = ffmpeg_preset
                    if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]: nvenc_preset_down = "fast"
                    elif ffmpeg_preset in ["slower", "veryslow"]: nvenc_preset_down = "slow"
                    ffmpeg_opts_downscale = f'-c:v h264_nvenc -preset:v {nvenc_preset_down} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
                else:
                    ffmpeg_opts_downscale = f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'

                cmd = f'ffmpeg -y -i "{input_video_path}" -vf "{scale_filter}" {ffmpeg_opts_downscale} -c:a copy "{downscaled_temp_video}"'
                run_ffmpeg_command(cmd, "Input Downscaling with Audio Copy")
                current_input_video_for_frames = downscaled_temp_video
                orig_h, orig_w = get_video_resolution(downscaled_temp_video) 
                downscale_duration_msg = f"Input downscaling finished. Time: {format_time(time.time() - downscale_stage_start_time)}"
                status_log.append(downscale_duration_msg)
                logger.info(downscale_duration_msg)
                yield None, "\n".join(status_log)
        else:
            upscale_factor = upscale_factor_slider
            final_h = int(round(orig_h * upscale_factor / 2) * 2) 
            final_w = int(round(orig_w * upscale_factor / 2) * 2) 
            direct_upscale_msg = f"Direct upscale: {upscale_factor:.2f}x. Target output: {final_w}x{final_h}"
            status_log.append(direct_upscale_msg)
            logger.info(direct_upscale_msg)
        
        yield None, "\n".join(status_log)

        progress(0.1, desc="Loading STAR model...")
        star_model_load_start_time = time.time()
        model_cfg = EasyDict() 
        model_cfg.model_path = model_file_path

        star_model = VideoToVideo_sr(model_cfg)
        model_load_msg = f"STAR model loaded. Time: {format_time(time.time() - star_model_load_start_time)}"
        status_log.append(model_load_msg)
        logger.info(model_load_msg)
        yield None, "\n".join(status_log)

        progress(0.15, desc="Extracting frames...")
        frame_extraction_start_time = time.time()
        frame_count, input_fps, frame_files = extract_frames(current_input_video_for_frames, input_frames_dir)
        frame_extract_msg = f"Extracted {frame_count} frames at {input_fps:.2f} FPS. Time: {format_time(time.time() - frame_extraction_start_time)}"
        status_log.append(frame_extract_msg)
        logger.info(frame_extract_msg)
        yield None, "\n".join(status_log)

        if save_frames and input_frames_dir and input_frames_permanent_save_path:
            copy_input_frames_start_time = time.time()
            copy_input_msg = f"Copying {frame_count} input frames to permanent storage: {input_frames_permanent_save_path}"
            status_log.append(copy_input_msg)
            logger.info(copy_input_msg)
            yield None, "\n".join(status_log)
            for frame_file in os.listdir(input_frames_dir):
                shutil.copy2(os.path.join(input_frames_dir, frame_file), os.path.join(input_frames_permanent_save_path, frame_file))
            copied_input_msg = f"Input frames copied. Time: {format_time(time.time() - copy_input_frames_start_time)}"
            status_log.append(copied_input_msg)
            logger.info(copied_input_msg)
            yield None, "\n".join(status_log)

        progress(0.2, desc="Upscaling frames...")
        total_noise_levels = 900
        
        all_lr_frames_bgr_for_preprocess = [] 
        for frame_filename in frame_files:
            frame_lr_bgr = cv2.imread(os.path.join(input_frames_dir, frame_filename))
            if frame_lr_bgr is None:
                logger.error(f"Could not read frame {frame_filename} from {input_frames_dir}. Skipping.")
                all_lr_frames_bgr_for_preprocess.append(np.zeros((orig_h, orig_w, 3), dtype=np.uint8)) 
                continue
            all_lr_frames_bgr_for_preprocess.append(frame_lr_bgr) 
        
        if len(all_lr_frames_bgr_for_preprocess) != frame_count:
             logger.warning(f"Mismatch in frame count and loaded LR frames for colorfix: {len(all_lr_frames_bgr_for_preprocess)} vs {frame_count}")

        # Main upscaling process start time
        upscaling_loop_start_time = time.time()

        if enable_tiling:
            loop_name = "Tiling Process"
            tiling_status_msg = f"Tiling enabled: Tile Size={tile_size}, Overlap={tile_overlap}. Processing {len(frame_files)} frames."
            status_log.append(tiling_status_msg)
            logger.info(tiling_status_msg)
            yield None, "\n".join(status_log)
            
            total_frames_to_tile = len(frame_files)
            # Outer loop for frames
            frame_tqdm_iterator = progress.tqdm(enumerate(frame_files), total=total_frames_to_tile, desc=f"{loop_name} - Initializing...")

            for i, frame_filename in frame_tqdm_iterator:
                frame_proc_start_time = time.time()
                frame_lr_bgr = cv2.imread(os.path.join(input_frames_dir, frame_filename))
                if frame_lr_bgr is None: 
                    logger.warning(f"Skipping frame {frame_filename} due to read error during tiling.")
                    placeholder_path = os.path.join(input_frames_dir, frame_filename)
                    if os.path.exists(placeholder_path):
                        shutil.copy2(placeholder_path, os.path.join(output_frames_dir, frame_filename))
                    continue
                
                single_lr_frame_tensor_norm = preprocess([frame_lr_bgr]) 
                spliter = ImageSpliterTh(single_lr_frame_tensor_norm, int(tile_size), int(tile_overlap), sf=upscale_factor)
                
                try:
                    num_patches_this_frame = len(list(spliter)) # Consume to get len, re-init for iteration
                    spliter = ImageSpliterTh(single_lr_frame_tensor_norm, int(tile_size), int(tile_overlap), sf=upscale_factor) # Re-initialize
                except: # Fallback if len() not directly available or spliter is not trivially re-initializable for len
                    num_patches_this_frame = getattr(spliter, 'num_patches', 'N/A') # Hypothetical attribute

                patch_tqdm_iterator = progress.tqdm(enumerate(spliter), total=num_patches_this_frame if isinstance(num_patches_this_frame, int) else None, desc=f"Frame {i+1}/{total_frames_to_tile} Patches")
                
                for patch_idx, (patch_lr_tensor_norm, patch_coords) in patch_tqdm_iterator:
                    patch_proc_start_time = time.time()
                    patch_lr_video_data = patch_lr_tensor_norm

                    patch_pre_data = {'video_data': patch_lr_video_data, 'y': final_prompt,
                                      'target_res': (int(round(patch_lr_tensor_norm.shape[-2] * upscale_factor)), 
                                                     int(round(patch_lr_tensor_norm.shape[-1] * upscale_factor)))} 
                    patch_data_tensor_cuda = collate_fn(patch_pre_data, 'cuda:0')
                    
                    # Log before STAR model call
                    logger.info(f"{loop_name} - Frame {i+1}/{total_frames_to_tile}, Patch {patch_idx+1}/{num_patches_this_frame}: Starting STAR model processing.")
                    star_model_call_patch_start_time = time.time()
                    with torch.no_grad():
                        patch_sr_tensor_bcthw = star_model.test( 
                            patch_data_tensor_cuda, total_noise_levels, steps=steps, solver_mode=solver_mode,
                            guide_scale=cfg_scale, max_chunk_len=1, vae_decoder_chunk_size=1
                        )
                    star_model_call_patch_duration = time.time() - star_model_call_patch_start_time
                    logger.info(f"{loop_name} - Frame {i+1}/{total_frames_to_tile}, Patch {patch_idx+1}/{num_patches_this_frame}: Finished STAR model processing. Duration: {format_time(star_model_call_patch_duration)}")
                    patch_sr_frames_uint8 = tensor2vid(patch_sr_tensor_bcthw)
                    
                    if color_fix_method != 'None':
                        if color_fix_method == 'AdaIN':
                            patch_sr_frames_uint8 = adain_color_fix(patch_sr_frames_uint8, patch_lr_video_data)
                        elif color_fix_method == 'Wavelet':
                            patch_sr_frames_uint8 = wavelet_color_fix(patch_sr_frames_uint8, patch_lr_video_data)

                    single_patch_frame_hwc = patch_sr_frames_uint8[0] 
                    result_patch_chw_01 = single_patch_frame_hwc.permute(2,0,1).float() / 255.0

                    spliter.update_gaussian(result_patch_chw_01.unsqueeze(0), patch_coords) 
                    
                    del patch_data_tensor_cuda, patch_sr_tensor_bcthw, patch_sr_frames_uint8
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                    patch_duration = time.time() - patch_proc_start_time
                    patch_tqdm_iterator.set_description_str(f"Frame {i+1} Patch {patch_idx+1}/{num_patches_this_frame} (took {patch_duration:.2f}s)")
                
                final_frame_tensor_chw = spliter.gather() 
                final_frame_np_hwc_uint8 = (final_frame_tensor_chw.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                final_frame_bgr = cv2.cvtColor(final_frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_frames_dir, frame_filename), final_frame_bgr)

                # Frame-level progress for tiling
                frames_processed_tile = i + 1
                current_tile_loop_time = time.time() - upscaling_loop_start_time
                avg_time_per_frame_tile = current_tile_loop_time / frames_processed_tile
                eta_seconds_tile = (total_frames_to_tile - frames_processed_tile) * avg_time_per_frame_tile if frames_processed_tile < total_frames_to_tile else 0
                speed_tile = 1 / avg_time_per_frame_tile if avg_time_per_frame_tile > 0 else 0
                
                frame_tqdm_desc = f"{loop_name}: {frames_processed_tile}/{total_frames_to_tile} frames | ETA: {format_time(eta_seconds_tile)} | Speed: {speed_tile:.2f} f/s"
                frame_tqdm_iterator.set_description_str(frame_tqdm_desc)
                
                detailed_frame_msg = f"{frame_tqdm_desc} | Current frame processed in {time.time() - frame_proc_start_time:.2f}s. Total elapsed: {format_time(current_tile_loop_time)}"
                status_log.append(detailed_frame_msg)
                logger.info(detailed_frame_msg)
                # Update Gradio progress bar
                progress(frames_processed_tile / total_frames_to_tile, desc=frame_tqdm_desc)
                yield None, "\n".join(status_log)


        elif enable_sliding_window:
            loop_name = "Sliding Window Process"
            sliding_status_msg = f"Sliding Window: Size={window_size}, Step={window_step}."
            status_log.append(sliding_status_msg)
            logger.info(sliding_status_msg)
            yield None, "\n".join(status_log)
            
            processed_frame_filenames = [None] * frame_count 
            effective_window_size = int(window_size)
            effective_window_step = int(window_step)
            
            # Calculate total number of windows
            if effective_window_step <= 0 or frame_count == 0:
                n_windows = 1 if frame_count > 0 else 0
            else:
                n_windows = math.ceil(frame_count / effective_window_step)
                # A common adjustment for the last window: if the last step doesn't make a full window,
                # it might be handled by extending the second to last, or processing a smaller final one.
                # The current logic already handles `end_idx` and `is_last_window_iteration` for length.
                # For `n_windows` for progress, this initial calculation should be fine for `tqdm`.


            window_indices_to_process = list(range(0, frame_count, effective_window_step))
            total_windows_to_process = len(window_indices_to_process)

            sliding_tqdm_iterator = progress.tqdm(enumerate(window_indices_to_process), total=total_windows_to_process, desc=f"{loop_name} - Initializing...")

            for window_iter_idx, i in sliding_tqdm_iterator: # i is start_idx of window
                window_proc_start_time = time.time()
                start_idx = i
                end_idx = min(i + effective_window_size, frame_count)
                current_window_len = end_idx - start_idx

                if current_window_len == 0: continue
                
                is_last_window_iteration = (i + effective_window_step >= frame_count) # or window_iter_idx == total_windows_to_process -1
                # Adjust start_idx for the very last iteration if it results in a window smaller than effective_window_size
                # and we want to ensure the last few frames are processed within a full-size (or near full-size) context.
                if is_last_window_iteration and current_window_len < effective_window_size and frame_count >= effective_window_size :
                    start_idx = max(0, frame_count - effective_window_size)
                    end_idx = frame_count # Ensure it goes to the end
                    current_window_len = end_idx - start_idx
                    logger.info(f"{loop_name} - Adjusted last window to start_idx: {start_idx} to cover final frames.")


                window_frame_names = frame_files[start_idx:end_idx]

                if end_idx > len(all_lr_frames_bgr_for_preprocess): 
                    logger.error(f"Sliding window range {start_idx}-{end_idx} exceeds available LR frames {len(all_lr_frames_bgr_for_preprocess)}")
                    continue 
                
                window_lr_frames_bgr = [all_lr_frames_bgr_for_preprocess[j] for j in range(start_idx, end_idx)] 
                if not window_lr_frames_bgr: continue

                window_lr_video_data = preprocess(window_lr_frames_bgr) 

                window_pre_data = {'video_data': window_lr_video_data, 'y': final_prompt,
                                   'target_res': (final_h, final_w)}
                window_data_cuda = collate_fn(window_pre_data, 'cuda:0')

                logger.info(f"{loop_name} - Window {window_iter_idx+1}/{total_windows_to_process} (frames {start_idx}-{end_idx-1}): Starting STAR model processing.")
                star_model_call_window_start_time = time.time()
                with torch.no_grad():
                    window_sr_tensor_bcthw = star_model.test(
                        window_data_cuda, total_noise_levels, steps=steps, solver_mode=solver_mode,
                        guide_scale=cfg_scale, max_chunk_len=current_window_len, vae_decoder_chunk_size=min(vae_chunk, current_window_len)
                    )
                star_model_call_window_duration = time.time() - star_model_call_window_start_time
                logger.info(f"{loop_name} - Window {window_iter_idx+1}/{total_windows_to_process}: Finished STAR model processing. Duration: {format_time(star_model_call_window_duration)}")
                window_sr_frames_uint8 = tensor2vid(window_sr_tensor_bcthw) 

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        window_sr_frames_uint8 = adain_color_fix(window_sr_frames_uint8, window_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        window_sr_frames_uint8 = wavelet_color_fix(window_sr_frames_uint8, window_lr_video_data)
                
                # Determine which frames from this window to save (handling overlaps)
                save_from_start_offset_local = 0
                save_to_end_offset_local = current_window_len
                
                if total_windows_to_process > 1: # Only apply overlap logic if there's more than one window
                    overlap_amount = effective_window_size - effective_window_step
                    if overlap_amount > 0 : # Ensure there is actual overlap
                        # First window: save from start up to a point before overlap affects next window's center
                        if window_iter_idx == 0:
                            save_to_end_offset_local = effective_window_size - (overlap_amount // 2) if overlap_amount > 0 else current_window_len
                        # Last window: save from a point after previous window's center up to end
                        elif is_last_window_iteration : # Check based on iteration index
                            save_from_start_offset_local = (overlap_amount // 2) if overlap_amount > 0 else 0
                        # Middle windows: save central part
                        else:
                            save_from_start_offset_local = (overlap_amount // 2) if overlap_amount > 0 else 0
                            save_to_end_offset_local = effective_window_size - (overlap_amount - save_from_start_offset_local) if overlap_amount > 0 else current_window_len
                    # Ensure valid ranges
                    save_from_start_offset_local = max(0, min(save_from_start_offset_local, current_window_len -1 if current_window_len > 0 else 0))
                    save_to_end_offset_local = max(save_from_start_offset_local, min(save_to_end_offset_local, current_window_len))


                for k_local in range(save_from_start_offset_local, save_to_end_offset_local):
                    k_global = start_idx + k_local
                    if k_global < frame_count and processed_frame_filenames[k_global] is None: 
                        frame_np_hwc_uint8 = window_sr_frames_uint8[k_local].cpu().numpy()
                        frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                        out_f_path = os.path.join(output_frames_dir, frame_files[k_global])
                        cv2.imwrite(out_f_path, frame_bgr)
                        processed_frame_filenames[k_global] = frame_files[k_global]

                del window_data_cuda, window_sr_tensor_bcthw, window_sr_frames_uint8
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                windows_processed_slide = window_iter_idx + 1
                current_slide_loop_time = time.time() - upscaling_loop_start_time # upscaling_loop_start_time is the start of this whole stage
                avg_time_per_window = current_slide_loop_time / windows_processed_slide
                eta_seconds_slide = (total_windows_to_process - windows_processed_slide) * avg_time_per_window if windows_processed_slide < total_windows_to_process else 0
                speed_slide = 1 / avg_time_per_window if avg_time_per_window > 0 else 0
                
                slide_tqdm_desc = f"{loop_name}: {windows_processed_slide}/{total_windows_to_process} windows | ETA: {format_time(eta_seconds_slide)} | Speed: {speed_slide:.2f} w/s"
                sliding_tqdm_iterator.set_description_str(slide_tqdm_desc)
                
                detailed_slide_msg = f"{slide_tqdm_desc} | Current window (frames {start_idx}-{end_idx-1}) processed in {time.time() - window_proc_start_time:.2f}s. Total elapsed: {format_time(current_slide_loop_time)}"
                status_log.append(detailed_slide_msg)
                logger.info(detailed_slide_msg)
                # Update Gradio progress bar
                progress(windows_processed_slide / total_windows_to_process, desc=slide_tqdm_desc)
                yield None, "\n".join(status_log)
            
            # Fallback for any missed frames
            num_missed_fallback = 0
            for idx, fname in enumerate(frame_files):
                if processed_frame_filenames[idx] is None:
                    num_missed_fallback +=1
                    logger.warning(f"Frame {fname} (index {idx}) was not processed by sliding window, copying LR frame.")
                    lr_frame_path = os.path.join(input_frames_dir, fname)
                    if os.path.exists(lr_frame_path):
                         shutil.copy2(lr_frame_path, os.path.join(output_frames_dir, fname))
                    else: 
                         logger.error(f"LR frame {lr_frame_path} not found for fallback copy.")
            if num_missed_fallback > 0:
                missed_msg = f"{loop_name} - Copied {num_missed_fallback} LR frames as fallback for unprocessed frames."
                status_log.append(missed_msg)
                logger.info(missed_msg)
                yield None, "\n".join(status_log)


        else: 
            loop_name = "Chunked Processing"
            chunk_status_msg = "Normal chunked processing."
            status_log.append(chunk_status_msg)
            logger.info(chunk_status_msg)
            yield None, "\n".join(status_log)

            num_chunks = math.ceil(frame_count / max_chunk_len) if max_chunk_len > 0 else (1 if frame_count > 0 else 0)
            if num_chunks == 0 and frame_count > 0: num_chunks = 1 # Ensure at least one chunk if frames exist

            chunk_tqdm_iterator = progress.tqdm(range(num_chunks), total=num_chunks, desc=f"{loop_name} - Initializing...")

            for i in chunk_tqdm_iterator: # i is current chunk index
                chunk_proc_start_time = time.time()
                start_idx = i * max_chunk_len
                end_idx = min((i + 1) * max_chunk_len, frame_count)
                current_chunk_len = end_idx - start_idx
                if current_chunk_len == 0: continue
                
                if end_idx > len(all_lr_frames_bgr_for_preprocess): 
                     logger.error(f"Chunk range {start_idx}-{end_idx} exceeds available LR frames {len(all_lr_frames_bgr_for_preprocess)}")
                     continue

                chunk_lr_frames_bgr = all_lr_frames_bgr_for_preprocess[start_idx:end_idx] 
                if not chunk_lr_frames_bgr: continue

                chunk_lr_video_data = preprocess(chunk_lr_frames_bgr) 

                chunk_pre_data = {'video_data': chunk_lr_video_data, 'y': final_prompt,
                                  'target_res': (final_h, final_w)}
                chunk_data_cuda = collate_fn(chunk_pre_data, 'cuda:0')

                logger.info(f"{loop_name} - Chunk {i+1}/{num_chunks} (frames {start_idx}-{end_idx-1}): Starting STAR model processing.")
                star_model_call_chunk_start_time = time.time()
                with torch.no_grad():
                    chunk_sr_tensor_bcthw = star_model.test(
                        chunk_data_cuda, total_noise_levels, steps=steps, solver_mode=solver_mode,
                        guide_scale=cfg_scale, max_chunk_len=current_chunk_len, vae_decoder_chunk_size=min(vae_chunk, current_chunk_len)
                    )
                star_model_call_chunk_duration = time.time() - star_model_call_chunk_start_time
                logger.info(f"{loop_name} - Chunk {i+1}/{num_chunks}: Finished STAR model processing. Duration: {format_time(star_model_call_chunk_duration)}")
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

                chunks_processed = i + 1
                current_chunk_loop_time = time.time() - upscaling_loop_start_time # upscaling_loop_start_time is the start of this whole stage
                avg_time_per_chunk = current_chunk_loop_time / chunks_processed
                eta_seconds_chunk = (num_chunks - chunks_processed) * avg_time_per_chunk if chunks_processed < num_chunks else 0
                speed_chunk = 1 / avg_time_per_chunk if avg_time_per_chunk > 0 else 0
                
                chunk_tqdm_desc = f"{loop_name}: {chunks_processed}/{num_chunks} chunks | ETA: {format_time(eta_seconds_chunk)} | Speed: {speed_chunk:.2f} ch/s"
                # chunk_tqdm_iterator.set_description_str(chunk_tqdm_desc)
                
                detailed_chunk_msg = f"{chunk_tqdm_desc} | Current chunk (frames {start_idx}-{end_idx-1}) processed in {time.time() - chunk_proc_start_time:.2f}s. Total elapsed: {format_time(current_chunk_loop_time)}"
                status_log.append(detailed_chunk_msg)
                logger.info(detailed_chunk_msg)
                # Update Gradio progress bar
                progress(chunks_processed / num_chunks, desc=chunk_tqdm_desc)
                yield None, "\n".join(status_log)
        
        upscaling_total_duration_msg = f"All frame upscaling operations finished. Total upscaling time: {format_time(time.time() - upscaling_loop_start_time)}"
        status_log.append(upscaling_total_duration_msg)
        logger.info(upscaling_total_duration_msg)
        yield None, "\n".join(status_log)


        progress(0.9, desc="Reassembling video...")
        reassembly_start_time = time.time()
        status_log.append("Reassembling final video...")
        logger.info("Reassembling final video...")
        yield None, "\n".join(status_log)
        
        if save_frames and output_frames_dir and processed_frames_permanent_save_path:
            copy_processed_start_time = time.time()
            num_processed_frames_to_copy = len(os.listdir(output_frames_dir))
            copy_proc_msg = f"Copying {num_processed_frames_to_copy} processed frames to permanent storage: {processed_frames_permanent_save_path}"
            status_log.append(copy_proc_msg)
            logger.info(copy_proc_msg)
            yield None, "\n".join(status_log)
            for frame_file in os.listdir(output_frames_dir):
                shutil.copy2(os.path.join(output_frames_dir, frame_file), os.path.join(processed_frames_permanent_save_path, frame_file))
            copied_proc_msg = f"Processed frames copied. Time: {format_time(time.time() - copy_processed_start_time)}"
            status_log.append(copied_proc_msg)
            logger.info(copied_proc_msg)
            yield None, "\n".join(status_log)

        silent_upscaled_video_path = os.path.join(temp_dir, "silent_upscaled_video.mp4")
        create_video_from_frames(output_frames_dir, silent_upscaled_video_path, input_fps, ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu)
        
        silent_video_msg = "Silent upscaled video created. Merging audio..."
        status_log.append(silent_video_msg)
        logger.info(silent_video_msg)
        yield None, "\n".join(status_log)

        audio_source_video = current_input_video_for_frames
        final_output_path = output_video_path 

        if not os.path.exists(audio_source_video):
            logger.warning(f"Audio source video '{audio_source_video}' not found. Output will be video-only.")
            shutil.copy2(silent_upscaled_video_path, final_output_path)
        else:
            run_ffmpeg_command(f'ffmpeg -y -i "{silent_upscaled_video_path}" -i "{audio_source_video}" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? -shortest "{final_output_path}"', "Final Video and Audio Merge")

        reassembly_done_msg = f"Video reassembly and audio merge finished. Time: {format_time(time.time() - reassembly_start_time)}"
        status_log.append(reassembly_done_msg)
        logger.info(reassembly_done_msg)

        final_save_msg = f"Upscaled video saved to: {final_output_path}"
        status_log.append(final_save_msg)
        logger.info(final_save_msg)
        progress(1.0, "Finished!")

        if save_metadata:
            metadata_save_start_time = time.time()
            processing_time = time.time() - overall_process_start_time # Use overall start time
            metadata_filepath = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_output_filename_no_ext}.txt")
            params_to_save = {
                "input_video_path": os.path.abspath(input_video_path) if input_video_path else "N/A",
                "user_prompt": user_prompt,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "model_choice": model_choice,
                "upscale_factor_slider_if_target_res_disabled": upscale_factor_slider,
                "cfg_scale": cfg_scale,
                "steps": steps,
                "solver_mode": solver_mode,
                "max_chunk_len": max_chunk_len,
                "vae_chunk": vae_chunk,
                "color_fix_method": color_fix_method,
                "enable_tiling": enable_tiling,
                "tile_size_if_tiling_enabled": tile_size if enable_tiling else "N/A",
                "tile_overlap_if_tiling_enabled": tile_overlap if enable_tiling else "N/A",
                "enable_sliding_window": enable_sliding_window,
                "window_size_if_sliding_enabled": window_size if enable_sliding_window else "N/A",
                "window_step_if_sliding_enabled": window_step if enable_sliding_window else "N/A",
                "enable_target_res": enable_target_res,
                "target_h_if_target_res_enabled": target_h if enable_target_res else "N/A",
                "target_w_if_target_res_enabled": target_w if enable_target_res else "N/A",
                "target_res_mode_if_target_res_enabled": target_res_mode if enable_target_res else "N/A",
                "ffmpeg_preset": ffmpeg_preset,
                "ffmpeg_quality_value": ffmpeg_quality_value,
                "ffmpeg_use_gpu": ffmpeg_use_gpu,
                "final_output_video_path": os.path.abspath(final_output_path),
                "original_video_resolution_wh": (orig_w, orig_h) if 'orig_w' in locals() and 'orig_h' in locals() else "N/A",
                "effective_input_fps": f"{input_fps:.2f}" if 'input_fps' in locals() else "N/A",
                "calculated_upscale_factor": f"{upscale_factor:.2f}" if 'upscale_factor' in locals() else "N/A",
                "final_output_resolution_wh": (final_w, final_h) if 'final_w' in locals() and 'final_h' in locals() else "N/A",
                "processing_time_seconds": f"{processing_time:.2f}",
                "processing_time_formatted": format_time(processing_time)
            }
            try:
                with open(metadata_filepath, 'w', encoding='utf-8') as f:
                    for key, value in params_to_save.items():
                        f.write(f"{key}: {value}\n")
                meta_saved_msg = f"Metadata saved to: {metadata_filepath}. Time to save: {format_time(time.time() - metadata_save_start_time)}"
                status_log.append(meta_saved_msg)
                logger.info(meta_saved_msg)
            except Exception as e_meta:
                status_log.append(f"Error saving metadata: {e_meta}")
                logger.error(f"Error saving metadata to {metadata_filepath}: {e_meta}")

        yield final_output_path, "\n".join(status_log)

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
        if star_model is not None:
            try:
                if hasattr(star_model, 'to'): star_model.to('cpu')
                del star_model
            except: pass
        
        gc.collect() # Add garbage collection for star_model too
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("STAR upscaling process finished and cleaned up.")
        
        cleanup_temp_dir(temp_dir) # Ensure temp dir is cleaned up
        
        total_process_duration = time.time() - overall_process_start_time
        final_cleanup_msg = f"STAR upscaling process finished and cleaned up. Total processing time: {format_time(total_process_duration)}"
        logger.info(final_cleanup_msg)
        # status_log.append(final_cleanup_msg) # This might be yielded too late if error occurred earlier

        if 'output_video_path' not in locals() or not os.path.exists(output_video_path):
             if status_log and status_log[-1] and not status_log[-1].startswith("Error:") and not status_log[-1].startswith("Critical Error:"):
                no_output_msg = "Processing finished, but output video was not found or not created."
                status_log.append(no_output_msg)
                logger.warning(no_output_msg)
             # Yield last status if an error didn't already cause a yield/raise
             # This path might be tricky if an error was already raised.
             # For now, let's assume the error handlers manage the final yield.
             # yield None, "\n".join(status_log)


        if 'base_output_filename_no_ext' in locals() and base_output_filename_no_ext:
            tmp_lock_file_to_delete = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_output_filename_no_ext}.tmp")
            if os.path.exists(tmp_lock_file_to_delete):
                try:
                    os.remove(tmp_lock_file_to_delete)
                    logger.info(f"Successfully deleted lock file: {tmp_lock_file_to_delete}")
                except Exception as e_lock_del:
                    logger.error(f"Failed to delete lock file {tmp_lock_file_to_delete}: {e_lock_del}")
            else:
                logger.warning(f"Lock file {tmp_lock_file_to_delete} not found for deletion.")


css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultimate SECourses STAR Video Upscaler V9")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                input_video = gr.Video(
                    label="Input Video",
                    sources=["upload"],
                    interactive=True,height=512
                )
                with gr.Row():
                    user_prompt = gr.Textbox(
                        label="Describe the Video Content (Prompt)",
                        lines=3,
                        placeholder="e.g., A panda playing guitar by a lake at sunset.",
                        info="""Describe the main subject and action in the video. This guides the upscaling process.
    Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens.
    If CogVLM2 is available, you can use the button below to generate a caption automatically."""
                    )
                with gr.Row():
                    auto_caption_then_upscale_check = gr.Checkbox(label="Auto-caption then Upscale", value=False, info="If checked, clicking 'Upscale Video' will first generate a caption and use it as the prompt.")
                    
                if COG_VLM_AVAILABLE:
                    with gr.Row():
                        auto_caption_btn = gr.Button("Generate Caption with CogVLM2",variant="primary",icon="icons/caption.png")
                        upscale_button = gr.Button("Upscale Video", variant="primary",icon="icons/upscale.png")
                    caption_status = gr.Textbox(label="Captioning Status", interactive=False, visible=False)
                else:
                    upscale_button = gr.Button("Upscale Video", variant="primary",icon="icons/upscale.png")

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

            with gr.Accordion("Advanced: Target Resolution", open=True):
                 enable_target_res_check = gr.Checkbox(
                     label="Enable Max Target Resolution",
                     value=True,
                     info="Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
                 )
                 target_res_mode_radio = gr.Radio(
                     label="Target Resolution Mode",
                     choices=['Ratio Upscale', 'Downscale then 4x'], value='Downscale then 4x', 
                     info="""How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio.
'Downscale then 4x': If input is large, downscales it towards Target H/W divided by 4, THEN applies a 4x upscale. Can clean noisy high-res input before upscaling."""
                 )
                 with gr.Row():
                     target_h_num = gr.Slider(
                         label="Max Target Height (px)",
                         value=512, minimum=128, maximum=4096, step=16, # Max increased for slider
                         info="""Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
    - VRAM Impact: Very High (Lower value = Less VRAM).
    - Quality Impact: Direct (Lower value = Less detail).
    - Speed Impact: Faster (Lower value = Faster)."""
                     )
                     target_w_num = gr.Slider(
                         label="Max Target Width (px)",
                         value=512, minimum=128, maximum=4096, step=16, # Max increased for slider
                         info="""Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
    - VRAM Impact: Very High (Lower value = Less VRAM).
    - Quality Impact: Direct (Lower value = Less detail).
    - Speed Impact: Faster (Lower value = Faster)."""
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


            
            if COG_VLM_AVAILABLE:
                with gr.Accordion("Auto-Captioning Settings (CogVLM2)", open=True):
                    cogvlm_quant_choices_map = {0: "FP16/BF16"}
                    if torch.cuda.is_available() and BITSANDBYTES_AVAILABLE:
                        cogvlm_quant_choices_map[4] = "INT4 (CUDA)"
                        cogvlm_quant_choices_map[8] = "INT8 (CUDA)"
                    
                    cogvlm_quant_radio_choices_display = list(cogvlm_quant_choices_map.values())
                    default_quant_display_val = "INT4 (CUDA)" if 4 in cogvlm_quant_choices_map else "FP16/BF16"

                    with gr.Row():
                        cogvlm_quant_radio = gr.Radio(
                            label="CogVLM2 Quantization",
                            choices=cogvlm_quant_radio_choices_display,
                            value=default_quant_display_val,
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

            with gr.Accordion("FFmpeg Encoding Settings", open=True): 
                ffmpeg_use_gpu_check = gr.Checkbox(
                    label="Use NVIDIA GPU for FFmpeg (h264_nvenc)",
                    value=False,
                    info="If checked, uses NVIDIA's NVENC for FFmpeg video encoding (downscaling and final video creation). Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."
                )
                with gr.Row():
                    ffmpeg_preset_dropdown = gr.Dropdown(
                        label="FFmpeg Preset",
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        value='medium',
                        info="Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently (e.g. p1-p7 or specific names like 'slow', 'medium', 'fast')."
                    )
                    ffmpeg_quality_slider = gr.Slider(
                        label="FFmpeg Quality (CRF for libx264 / CQ for NVENC)", 
                        minimum=0, maximum=51, value=23, step=1, 
                        info="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28."
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
                 with gr.Row():
                     tile_size_num = gr.Number(
                         label="Tile Size (px, input res)",
                         value=256, minimum=64, step=32, 
                         info="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                     )
                     tile_overlap_num = gr.Number(
                         label="Tile Overlap (px, input res)",
                         value=64, minimum=0, step=16, 
                         info="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                     )                
            

        with gr.Column(scale=1):
            output_video = gr.Video(label="Upscaled Video", interactive=False)
            status_textbox = gr.Textbox(label="Log", interactive=False, lines=8, max_lines=20)
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
                with gr.Row():
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
                        info="Number of denoising steps. 'Fast' mode uses a fixed ~15 steps. 'Normal' mode uses the value set here.",
                        interactive=False # Default to non-interactive as 'fast' mode is default and fixed
                    )
                color_fix_dropdown = gr.Dropdown(
                    label="Color Correction",
                    choices=['AdaIN', 'Wavelet', 'None'], value='AdaIN',
                    info="""Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""
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
                 with gr.Row():
                     window_size_num = gr.Slider(
                         label="Window Size (frames)",
                         value=32, minimum=2, step=4, 
                         info="Number of frames in each temporal window. Acts like 'Max Frames per Batch' but applied as a sliding window. Lower value = less VRAM, less temporal context."
                     )
                     window_step_num = gr.Slider(
                         label="Window Step (frames)",
                         value=16, minimum=1, step=1, 
                         info="How many frames to advance for the next window. (Window Size - Window Step) = Overlap. Smaller step = more overlap = better consistency but slower. Recommended: Step = Size / 2."
                     )
            with gr.Accordion("Output Options", open=True): 
                save_frames_checkbox = gr.Checkbox(
                    label="Save Input and Processed Frames",
                    value=True,
                    info="If checked, saves the extracted input frames and the upscaled output frames into a subfolder named after the output video (e.g., '0001/input_frames' and '0001/processed_frames')."
                )
                save_metadata_checkbox = gr.Checkbox(
                    label="Save Processing Metadata",
                    value=True,
                    info="If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time."
                )
                open_output_folder_button = gr.Button("Open Output Folder")

    def update_steps_display(mode):
        if mode == 'fast':
            return gr.update(value=15, interactive=False)
        else:  # 'normal'
            return gr.update(value=50, interactive=True)
    solver_mode_radio.change(update_steps_display, solver_mode_radio, steps_slider)

    enable_target_res_check.change(lambda x: [gr.update(interactive=x)]*3, inputs=enable_target_res_check, outputs=[target_h_num, target_w_num, target_res_mode_radio])
    enable_tiling_check.change(lambda x: [gr.update(interactive=x)]*2, inputs=enable_tiling_check, outputs=[tile_size_num, tile_overlap_num])
    enable_sliding_window_check.change(lambda x: [gr.update(interactive=x)]*2, inputs=enable_sliding_window_check, outputs=[window_size_num, window_step_num])

    def update_ffmpeg_quality_settings(use_gpu):
        if use_gpu:
            return gr.Slider(label="FFmpeg Quality (CQ for NVENC)", value=25, info="For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28.")
        else:
            return gr.Slider(label="FFmpeg Quality (CRF for libx264)", value=23, info="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default).")

    ffmpeg_use_gpu_check.change(
        fn=update_ffmpeg_quality_settings,
        inputs=ffmpeg_use_gpu_check,
        outputs=ffmpeg_quality_slider
    )

    open_output_folder_button.click(
        fn=lambda: open_folder(DEFAULT_OUTPUT_DIR),
        inputs=[],
        outputs=[]
    )

    # Helper to map display quant string to int value for CogVLM
    cogvlm_display_to_quant_val_map_global = {}
    if COG_VLM_AVAILABLE: # Define map only if COG_VLM_AVAILABLE
        _temp_map = {0: "FP16/BF16"}
        if torch.cuda.is_available() and BITSANDBYTES_AVAILABLE:
            _temp_map[4] = "INT4 (CUDA)"
            _temp_map[8] = "INT8 (CUDA)"
        cogvlm_display_to_quant_val_map_global = {v: k for k, v in _temp_map.items()}

    def get_quant_value_from_display(display_val):
        return cogvlm_display_to_quant_val_map_global.get(display_val, 0) # Default to 0 (FP16/BF16)

    def upscale_director_logic(
        input_video_val, user_prompt_val, pos_prompt_val, neg_prompt_val, model_selector_val,
        upscale_factor_slider_val, cfg_slider_val, steps_slider_val, solver_mode_radio_val,
        max_chunk_len_slider_val, vae_chunk_slider_val, color_fix_dropdown_val,
        enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
        enable_sliding_window_check_val, window_size_num_val, window_step_num_val,
        enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
        ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
        save_frames_checkbox_val, save_metadata_checkbox_val,
        # Optional CogVLM params, will be None if not COG_VLM_AVAILABLE
        cogvlm_quant_radio_val=None, cogvlm_unload_radio_val=None,
        do_auto_caption_first_val=False, # Checkbox value
        progress=gr.Progress(track_tqdm=True)
    ):
        current_user_prompt = user_prompt_val
        status_updates = []
        output_updates_for_prompt_box = gr.update() 

        if COG_VLM_AVAILABLE and do_auto_caption_first_val:
            progress(0, desc="Starting auto-captioning before upscale...")
            yield None, "Starting auto-captioning...", output_updates_for_prompt_box, gr.update(visible=True) # Show caption status
            try:
                quant_val = get_quant_value_from_display(cogvlm_quant_radio_val)
                caption_text, caption_stat_msg = auto_caption(input_video_val, quant_val, cogvlm_unload_radio_val, progress=progress)
                status_updates.append(f"Auto-caption status: {caption_stat_msg}")
                if not caption_text.startswith("Error:"):
                    current_user_prompt = caption_text
                    status_updates.append(f"Using generated caption as prompt: '{caption_text[:50]}...'")
                    output_updates_for_prompt_box = gr.update(value=current_user_prompt)
                else:
                    status_updates.append("Caption generation failed. Using original prompt.")
                # Yield intermediate status for captioning
                yield None, "\n".join(status_updates), output_updates_for_prompt_box, caption_stat_msg
            except Exception as e_ac:
                status_updates.append(f"Error during auto-caption pre-step: {e_ac}")
                yield None, "\n".join(status_updates), gr.update(), str(e_ac) # Update caption status with error
        
        # Always proceed to upscaling
        upscale_generator = run_upscale(
            input_video_val, current_user_prompt, pos_prompt_val, neg_prompt_val, model_selector_val,
            upscale_factor_slider_val, cfg_slider_val, steps_slider_val, solver_mode_radio_val,
            max_chunk_len_slider_val, vae_chunk_slider_val, color_fix_dropdown_val,
            enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
            enable_sliding_window_check_val, window_size_num_val, window_step_num_val,
            enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
            ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
            save_frames_checkbox_val, save_metadata_checkbox_val,
            progress=progress
        )
        # Iteratively yield from the upscaling generator
        final_video_output = None
        final_status_output = ""
        for output_val, status_val in upscale_generator:
            final_video_output = output_val # Keep track of the latest video output
            final_status_output = (("\n".join(status_updates) + "\n") if status_updates else "") + (status_val if status_val else "")
            
            # For caption status, keep the last message or clear if only upscaling
            caption_status_update = caption_status.value if COG_VLM_AVAILABLE and do_auto_caption_first_val else ""
            
            yield final_video_output, final_status_output.strip(), output_updates_for_prompt_box, caption_status_update
            output_updates_for_prompt_box = gr.update() # Reset after first yield if prompt was updated

        # Final yield to ensure the last state is correctly presented
        final_caption_status = caption_status.value if COG_VLM_AVAILABLE and hasattr(caption_status, 'value') else ""
        yield final_video_output, final_status_output.strip(), output_updates_for_prompt_box, final_caption_status


    # Define all possible inputs for the click handler
    click_inputs = [
        input_video, user_prompt, pos_prompt, neg_prompt, model_selector,
        upscale_factor_slider, cfg_slider, steps_slider, solver_mode_radio,
        max_chunk_len_slider, vae_chunk_slider, color_fix_dropdown,
        enable_tiling_check, tile_size_num, tile_overlap_num,
        enable_sliding_window_check, window_size_num, window_step_num,
        enable_target_res_check, target_h_num, target_w_num, target_res_mode_radio,
        ffmpeg_preset_dropdown, ffmpeg_quality_slider, ffmpeg_use_gpu_check,
        save_frames_checkbox, save_metadata_checkbox
    ]
    
    # Outputs for the main upscale button
    # If CogVLM available, caption_status is an output, otherwise it's not part of this button's direct outputs.
    click_outputs_list = [output_video, status_textbox, user_prompt]
    if COG_VLM_AVAILABLE:
        click_outputs_list.append(caption_status)


    if COG_VLM_AVAILABLE:
        click_inputs.extend([cogvlm_quant_radio, cogvlm_unload_radio, auto_caption_then_upscale_check])
    else:
        click_inputs.extend([gr.State(None), gr.State(None), gr.State(False)]) # Placeholders

    upscale_button.click(
        fn=upscale_director_logic,
        inputs=click_inputs,
        outputs=click_outputs_list
    )

    if COG_VLM_AVAILABLE:
        auto_caption_btn.click(
            fn=lambda vid, quant_display, unload_strat, progress_bar=gr.Progress(track_tqdm=True): auto_caption(vid, get_quant_value_from_display(quant_display), unload_strat, progress_bar),
            inputs=[input_video, cogvlm_quant_radio, cogvlm_unload_radio],
            outputs=[user_prompt, caption_status] # caption_status is Textbox
        ).then(lambda: gr.update(visible=True), outputs=caption_status)


def get_available_drives():
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(f"{letter}:\\")
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin":
         available_paths = ["/", "/Volumes"] 
    else: 
        available_paths = ["/", "/mnt", "/media"] 
        
        home_dir = os.path.expanduser("~")
        if home_dir not in available_paths:
            available_paths.append(home_dir)
    
    existing_paths = [p for p in available_paths if os.path.exists(p) and os.path.isdir(p)]
    
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if cwd not in existing_paths: existing_paths.append(cwd)
    if script_dir not in existing_paths: existing_paths.append(script_dir)

    # Add DEFAULT_OUTPUT_DIR to allowed_paths if it's not obviously covered
    abs_default_output_dir = os.path.abspath(DEFAULT_OUTPUT_DIR)
    if not any(abs_default_output_dir.startswith(os.path.abspath(p)) for p in existing_paths):
        # Add the parent of DEFAULT_OUTPUT_DIR to be safe, or DEFAULT_OUTPUT_DIR itself
        parent_default_output_dir = os.path.dirname(abs_default_output_dir)
        if parent_default_output_dir not in existing_paths:
             existing_paths.append(parent_default_output_dir)
    
    # Add base_path (STAR repo root)
    if base_path not in existing_paths and os.path.isdir(base_path):
        existing_paths.append(base_path)


    unique_paths = sorted(list(set(os.path.abspath(p) for p in existing_paths if os.path.isdir(p))))
    logger.info(f"Effective Gradio allowed_paths: {unique_paths}")
    return unique_paths


if __name__ == "__main__":
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Gradio App Starting. Default output to: {os.path.abspath(DEFAULT_OUTPUT_DIR)}")
    logger.info(f"STAR Models expected at: {LIGHT_DEG_MODEL}, {HEAVY_DEG_MODEL}")
    if COG_VLM_AVAILABLE:
        logger.info(f"CogVLM2 Model expected at: {COG_VLM_MODEL_PATH}")
    
    effective_allowed_paths = get_available_drives()

    demo.queue().launch(
        debug=True, 
        max_threads=100, # Default Gradio is 40, this might be high but matches original
        inbrowser=True, 
        share=args.share, 
        allowed_paths=effective_allowed_paths,
        prevent_thread_lock=True # Might help with thread-related issues during long processes
    )