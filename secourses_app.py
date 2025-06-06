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
import gc 
from easydict import EasyDict 
from argparse import ArgumentParser ,Namespace 
import logging 
import re 
from pathlib import Path 
from functools import partial 

from logic import config as app_config 
from logic import metadata_handler 

from logic .cogvlm_utils import (
load_cogvlm_model as util_load_cogvlm_model ,
unload_cogvlm_model as util_unload_cogvlm_model ,
auto_caption as util_auto_caption ,
COG_VLM_AVAILABLE as UTIL_COG_VLM_AVAILABLE ,
BITSANDBYTES_AVAILABLE as UTIL_BITSANDBYTES_AVAILABLE 
)

from logic .common_utils import format_time 

from logic .ffmpeg_utils import (
run_ffmpeg_command as util_run_ffmpeg_command ,
extract_frames as util_extract_frames ,
create_video_from_frames as util_create_video_from_frames ,
decrease_fps as util_decrease_fps ,
decrease_fps_with_multiplier as util_decrease_fps_with_multiplier ,
calculate_target_fps_from_multiplier as util_calculate_target_fps_from_multiplier ,
get_common_fps_multipliers as util_get_common_fps_multipliers 
)

from logic .file_utils import (
sanitize_filename as util_sanitize_filename ,
get_batch_filename as util_get_batch_filename ,
get_next_filename as util_get_next_filename ,
cleanup_temp_dir as util_cleanup_temp_dir ,
get_video_resolution as util_get_video_resolution ,
get_available_drives as util_get_available_drives ,
open_folder as util_open_folder 
)

from logic .scene_utils import (
split_video_into_scenes as util_split_video_into_scenes ,
merge_scene_videos as util_merge_scene_videos ,
split_video_only as util_split_video_only 
)

from logic .upscaling_utils import (
calculate_upscale_params as util_calculate_upscale_params 
)

from logic .gpu_utils import (
get_available_gpus as util_get_available_gpus ,
set_gpu_device as util_set_gpu_device ,
get_gpu_device as util_get_gpu_device ,
validate_gpu_availability as util_validate_gpu_availability 
)

from logic .nvenc_utils import (
is_resolution_too_small_for_nvenc 
)

from logic .batch_operations import (
process_batch_videos 
)

from logic .batch_processing_help import create_batch_processing_help

from logic .upscaling_core import run_upscale as core_run_upscale 

from logic .manual_comparison import (
generate_manual_comparison_video as util_generate_manual_comparison_video 
)

from logic .rife_interpolation import (
rife_fps_only_wrapper as util_rife_fps_only_wrapper 
)

SELECTED_GPU_ID =0 

parser =ArgumentParser (description ="Ultimate SECourses STAR Video Upscaler")
parser .add_argument ('--share',action ='store_true',help ="Enable Gradio live share")
parser .add_argument ('--outputs_folder',type =str ,default ="outputs",help ="Main folder for output videos and related files")
args =parser .parse_args ()

try :
    script_dir =os .path .dirname (os .path .abspath (__file__ ))
    base_path =script_dir 

    if not os .path .isdir (os .path .join (base_path ,'video_to_video')):
        print (f"Warning: 'video_to_video' directory not found in inferred base_path: {base_path}. Attempting to use parent directory.")
        base_path =os .path .dirname (base_path )
        if not os .path .isdir (os .path .join (base_path ,'video_to_video')):
            print (f"Error: Could not auto-determine STAR repository root. Please set 'base_path' manually.")
            print (f"Current inferred base_path: {base_path}")

    print (f"Using STAR repository base_path: {base_path}")
    if base_path not in sys .path :
        sys .path .insert (0 ,base_path )

except Exception as e_path :
    print (f"Error setting up base_path: {e_path}")
    print ("Please ensure app.py is correctly placed or base_path is manually set.")
    sys .exit (1 )

try :
    from video_to_video .video_to_video_model import VideoToVideo_sr 
    from video_to_video .utils .seed import setup_seed 
    from video_to_video .utils .logger import get_logger 
    from video_super_resolution .color_fix import adain_color_fix ,wavelet_color_fix 
    from inference_utils import tensor2vid ,preprocess ,collate_fn 
    from video_super_resolution .scripts .util_image import ImageSpliterTh 
    from video_to_video .utils .config import cfg as star_cfg 
except ImportError as e :
    print (f"Error importing STAR components: {e}")
    print (f"Searched in sys.path: {sys.path}")
    print ("Please ensure the STAR repository is correctly in the Python path (set by base_path) and all dependencies from 'requirements.txt' are installed.")
    sys .exit (1 )

logger =get_logger ()
logger .setLevel (logging .INFO )
found_stream_handler =False 
for handler in logger .handlers :
    if isinstance (handler ,logging .StreamHandler ):
        handler .setLevel (logging .INFO )
        found_stream_handler =True 
        logger .info ("Diagnostic: Explicitly set StreamHandler level to INFO.")
if not found_stream_handler :
    ch =logging .StreamHandler ()
    ch .setLevel (logging .INFO )
    logger .addHandler (ch )
    logger .info ("Diagnostic: No StreamHandler found, added a new one with INFO level.")
logger .info (f"Logger '{logger.name}' configured with level: {logging.getLevelName(logger.level)}. Handlers: {logger.handlers}")

app_config .initialize_paths_and_prompts (base_path ,args .outputs_folder ,star_cfg )

os .makedirs (app_config .DEFAULT_OUTPUT_DIR ,exist_ok =True )

if not os .path .exists (app_config .LIGHT_DEG_MODEL_PATH ):
     logger .error (f"FATAL: Light degradation model not found at {app_config.LIGHT_DEG_MODEL_PATH}.")
if not os .path .exists (app_config .HEAVY_DEG_MODEL_PATH ):
     logger .error (f"FATAL: Heavy degradation model not found at {app_config.HEAVY_DEG_MODEL_PATH}.")

css ="""
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

def wrapper_split_video_only_for_gradio (
input_video_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,
progress =gr .Progress (track_tqdm =True )
):
    return util_split_video_only (
    input_video_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
    scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
    scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
    scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,
    app_config .DEFAULT_OUTPUT_DIR ,
    logger ,
    progress =progress 
    )

with gr .Blocks (css =css ,theme =gr .themes .Soft ())as demo :
    gr .Markdown ("# Ultimate SECourses STAR Video Upscaler V48")

    with gr .Tabs ():
        with gr .Tab ("Main"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        input_video =gr .Video (
                        label ="Input Video",
                        sources =["upload"],
                        interactive =True ,height =512 
                        )
                        with gr .Row ():
                            user_prompt =gr .Textbox (
                            label ="Describe the Video Content (Prompt)",
                            lines =3 ,
                            placeholder ="e.g., A panda playing guitar by a lake at sunset.",
                            info ="""Describe the main subject and action in the video. This guides the upscaling process.
Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens.
If CogVLM2 is available, you can use the button below to generate a caption automatically."""
                            )
                        with gr .Row ():
                            auto_caption_then_upscale_check =gr .Checkbox (label ="Auto-caption then Upscale",value =app_config .DEFAULT_AUTO_CAPTION_THEN_UPSCALE ,info ="If checked, clicking 'Upscale Video' will first generate a caption and use it as the prompt.")

                            available_gpus =util_get_available_gpus ()
                            gpu_choices =["Auto"]+available_gpus if available_gpus else ["Auto","No CUDA GPUs detected"]
                            default_gpu =available_gpus [0 ]if available_gpus else "Auto"

                            gpu_selector =gr .Dropdown (
                            label ="GPU Selection",
                            choices =gpu_choices ,
                            value =default_gpu ,
                            info ="Select which GPU to use for processing. 'Auto' uses default GPU or CPU if none.",
                            scale =1 
                            )

                        if app_config .UTIL_COG_VLM_AVAILABLE :
                            with gr .Row ():
                                auto_caption_btn =gr .Button ("Generate Caption with CogVLM2 (No Upscale)",variant ="primary",icon ="icons/caption.png")
                                rife_fps_button =gr .Button ("RIFE FPS Increase (No Upscale)",variant ="primary",icon ="icons/fps.png")
                            with gr .Row ():
                                upscale_button =gr .Button ("Upscale Video",variant ="primary",icon ="icons/upscale.png")
                            caption_status =gr .Textbox (label ="Captioning Status",interactive =False ,visible =False )

                    with gr .Accordion ("Prompt Settings",open =True ):
                        pos_prompt =gr .Textbox (
                        label ="Default Positive Prompt (Appended)",
                        value =app_config .DEFAULT_POS_PROMPT ,
                        lines =2 ,
                        info ="""Appended to your 'Describe Video Content' prompt. Focuses on desired quality aspects (e.g., realism, detail).
The total combined prompt length is limited to 77 tokens."""
                        )
                        neg_prompt =gr .Textbox (
                        label ="Default Negative Prompt (Appended)",
                        value =app_config .DEFAULT_NEG_PROMPT ,
                        lines =2 ,
                        info ="Guides the model *away* from undesired aspects (e.g., bad quality, artifacts, specific styles). This does NOT count towards the 77 token limit for positive guidance."
                        )
                    open_output_folder_button =gr .Button ("Open Outputs Folder",icon ="icons/folder.png",variant ="primary")

                with gr .Column (scale =1 ):
                    output_video =gr .Video (label ="Upscaled Video",interactive =False ,height =512 )
                    status_textbox =gr .Textbox (label ="Log",interactive =False ,lines =8 ,max_lines =15 )

                    with gr .Accordion ("Last Processed Chunk",open =True ):
                        last_chunk_video =gr .Video (
                        label ="Last Processed Chunk Preview",
                        interactive =False ,
                        height =512 ,
                        visible =True 
                        )
                        chunk_status_text =gr .Textbox (
                        label ="Chunk Status",
                        interactive =False ,
                        lines =1 ,
                        value ="No chunks processed yet"
                        )

        with gr .Tab ("Resolution & Scene Split"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("Target Resolution - Maintains Your Input Video Aspect Ratio",open =True ):
                        enable_target_res_check =gr .Checkbox (
                        label ="Enable Max Target Resolution",
                        value =app_config .DEFAULT_ENABLE_TARGET_RES ,
                        info ="Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
                        )
                        target_res_mode_radio =gr .Radio (
                        label ="Target Resolution Mode",
                        choices =['Ratio Upscale','Downscale then 4x'],value =app_config .DEFAULT_TARGET_RES_MODE ,
                        info ="""How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio.
'Downscale then 4x': If input is large, downscales it towards Target H/W divided by 4, THEN applies a 4x upscale. Can clean noisy high-res input before upscaling."""
                        )
                        with gr .Row ():
                            target_h_num =gr .Slider (
                            label ="Max Target Height (px)",
                            value =app_config .DEFAULT_TARGET_H ,minimum =128 ,maximum =4096 ,step =16 ,
                            info ="""Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                            )
                            target_w_num =gr .Slider (
                            label ="Max Target Width (px)",
                            value =app_config .DEFAULT_TARGET_W ,minimum =128 ,maximum =4096 ,step =16 ,
                            info ="""Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                            )
                with gr .Column (scale =1 ):
                    split_only_button =gr .Button ("Split Video Only (No Upscaling)",icon ="icons/split.png",variant ="primary")
                    with gr .Accordion ("Scene Splitting",open =True ):
                        enable_scene_split_check =gr .Checkbox (
                        label ="Enable Scene Splitting",
                        value =app_config .DEFAULT_ENABLE_SCENE_SPLIT ,
                        info ="""Split video into scenes and process each scene individually. This can improve quality and speed by processing similar content together.
- Quality Impact: Better temporal consistency within scenes, improved auto-captioning per scene.
- Speed Impact: Can be faster for long videos with distinct scenes.
- Memory Impact: Reduces peak memory usage by processing smaller segments."""
                        )
                        with gr .Row ():
                            scene_split_mode_radio =gr .Radio (
                            label ="Split Mode",
                            choices =['automatic','manual'],value =app_config .DEFAULT_SCENE_SPLIT_MODE ,
                            info ="""'automatic': Uses scene detection algorithms to find natural scene boundaries.
'manual': Splits video at fixed intervals (duration or frame count)."""
                            )
                        with gr .Group ():
                            gr .Markdown ("**Automatic Scene Detection Settings**")
                            with gr .Row ():
                                scene_min_scene_len_num =gr .Number (label ="Min Scene Length (seconds)",value =app_config .DEFAULT_SCENE_MIN_SCENE_LEN ,minimum =0.1 ,step =0.1 ,info ="Minimum duration for a scene. Shorter scenes will be merged or dropped.")
                                scene_threshold_num =gr .Number (label ="Detection Threshold",value =app_config .DEFAULT_SCENE_THRESHOLD ,minimum =0.1 ,maximum =10.0 ,step =0.1 ,info ="Sensitivity of scene detection. Lower values detect more scenes.")
                            with gr .Row ():
                                scene_drop_short_check =gr .Checkbox (label ="Drop Short Scenes",value =app_config .DEFAULT_SCENE_DROP_SHORT ,info ="If enabled, scenes shorter than minimum length are dropped instead of merged.")
                                scene_merge_last_check =gr .Checkbox (label ="Merge Last Scene",value =app_config .DEFAULT_SCENE_MERGE_LAST ,info ="If the last scene is too short, merge it with the previous scene.")
                            with gr .Row ():
                                scene_frame_skip_num =gr .Number (label ="Frame Skip",value =app_config .DEFAULT_SCENE_FRAME_SKIP ,minimum =0 ,step =1 ,info ="Skip frames during detection to speed up processing. 0 = analyze every frame.")
                                scene_min_content_val_num =gr .Number (label ="Min Content Value",value =app_config .DEFAULT_SCENE_MIN_CONTENT_VAL ,minimum =0.0 ,step =1.0 ,info ="Minimum content change required to detect a scene boundary.")
                                scene_frame_window_num =gr .Number (label ="Frame Window",value =app_config .DEFAULT_SCENE_FRAME_WINDOW ,minimum =1 ,step =1 ,info ="Number of frames to analyze for scene detection.")
                        with gr .Group ():
                            gr .Markdown ("**Manual Split Settings**")
                            with gr .Row ():
                                scene_manual_split_type_radio =gr .Radio (label ="Manual Split Type",choices =['duration','frame_count'],value =app_config .DEFAULT_SCENE_MANUAL_SPLIT_TYPE ,info ="'duration': Split every N seconds.\n'frame_count': Split every N frames.")
                                scene_manual_split_value_num =gr .Number (label ="Split Value",value =app_config .DEFAULT_SCENE_MANUAL_SPLIT_VALUE ,minimum =1.0 ,step =1.0 ,info ="Duration in seconds or number of frames for manual splitting.")
                        with gr .Group ():
                            gr .Markdown ("**Encoding Settings (for scene segments)**")
                            with gr .Row ():
                                scene_copy_streams_check =gr .Checkbox (label ="Copy Streams",value =app_config .DEFAULT_SCENE_COPY_STREAMS ,info ="Copy video/audio streams without re-encoding during scene splitting (faster) but can generate inaccurate splits.")
                                scene_use_mkvmerge_check =gr .Checkbox (label ="Use MKVMerge",value =app_config .DEFAULT_SCENE_USE_MKVMERGE ,info ="Use mkvmerge instead of ffmpeg for splitting (if available).")
                            with gr .Row ():
                                scene_rate_factor_num =gr .Number (label ="Rate Factor (CRF)",value =app_config .DEFAULT_SCENE_RATE_FACTOR ,minimum =0 ,maximum =51 ,step =1 ,info ="Quality setting for re-encoding (lower = better quality). Only used if Copy Streams is disabled.")
                                scene_preset_dropdown =gr .Dropdown (label ="Encoding Preset",choices =['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],value =app_config .DEFAULT_SCENE_ENCODING_PRESET ,info ="Encoding speed vs quality trade-off. Only used if Copy Streams is disabled.")
                            scene_quiet_ffmpeg_check =gr .Checkbox (label ="Quiet FFmpeg",value =app_config .DEFAULT_SCENE_QUIET_FFMPEG ,info ="Suppress ffmpeg output during scene splitting.")

        with gr .Tab ("Core Settings"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        gr .Markdown ("### Core Upscaling Settings")
                        model_selector =gr .Dropdown (
                        label ="STAR Model",
                        choices =["Light Degradation","Heavy Degradation"],
                        value =app_config .DEFAULT_MODEL_CHOICE ,
                        info ="""Choose the model based on input video quality.
'Light Degradation': Better for relatively clean inputs (e.g., downloaded web videos).
'Heavy Degradation': Better for inputs with significant compression artifacts, noise, or blur."""
                        )
                        upscale_factor_slider =gr .Slider (
                        label ="Upscale Factor (if Target Res disabled)",
                        minimum =1.0 ,maximum =8.0 ,value =app_config .DEFAULT_UPSCALE_FACTOR ,step =0.1 ,
                        info ="Simple multiplication factor for output resolution if 'Enable Max Target Resolution' is OFF. E.g., 4.0 means 4x height and 4x width."
                        )
                        cfg_slider =gr .Slider (
                        label ="Guidance Scale (CFG)",
                        minimum =1.0 ,maximum =15.0 ,value =app_config .DEFAULT_CFG_SCALE ,step =0.5 ,
                        info ="Controls how strongly the model follows your combined text prompt. Higher values mean stricter adherence, lower values allow more creativity. Typical values: 5.0-10.0."
                        )
                        with gr .Row ():
                            solver_mode_radio =gr .Radio (
                            label ="Solver Mode",
                            choices =['fast','normal'],value =app_config .DEFAULT_SOLVER_MODE ,
                            info ="""Diffusion solver type.
'fast': Fewer steps (default ~15), much faster, good quality usually.
'normal': More steps (default ~50), slower, potentially slightly better detail/coherence."""
                            )
                            steps_slider =gr .Slider (
                            label ="Diffusion Steps",
                            minimum =5 ,maximum =100 ,value =app_config .DEFAULT_DIFFUSION_STEPS_FAST ,step =1 ,
                            info ="Number of denoising steps. 'Fast' mode uses a fixed ~15 steps. 'Normal' mode uses the value set here.",
                            interactive =False 
                            )
                        color_fix_dropdown =gr .Dropdown (
                        label ="Color Correction",
                        choices =['AdaIN','Wavelet','None'],value =app_config .DEFAULT_COLOR_FIX_METHOD ,
                        info ="""Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""
                        )
                with gr .Column (scale =1 ):
                    if app_config .UTIL_COG_VLM_AVAILABLE :
                        with gr .Accordion ("Auto-Captioning Settings (CogVLM2)",open =True ):
                            cogvlm_quant_choices_map =app_config .get_cogvlm_quant_choices_map (torch .cuda .is_available (),app_config .UTIL_BITSANDBYTES_AVAILABLE )
                            cogvlm_quant_radio_choices_display =list (cogvlm_quant_choices_map .values ())
                            default_quant_display_val =app_config .get_default_cogvlm_quant_display (cogvlm_quant_choices_map )

                            with gr .Row ():
                                cogvlm_quant_radio =gr .Radio (
                                label ="CogVLM2 Quantization",
                                choices =cogvlm_quant_radio_choices_display ,
                                value =default_quant_display_val ,
                                info ="Quantization for the CogVLM2 captioning model (uses less VRAM). INT4/8 require CUDA & bitsandbytes.",
                                interactive =True if len (cogvlm_quant_radio_choices_display )>1 else False 
                                )
                                cogvlm_unload_radio =gr .Radio (
                                label ="CogVLM2 After-Use",
                                choices =['full','cpu'],value =app_config .DEFAULT_COGVLM_UNLOAD_AFTER_USE ,
                                info ="""Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""
                                )
                    else :
                        gr .Markdown ("_(Auto-captioning disabled as CogVLM2 components are not fully available.)_")

                    with gr .Accordion ("Performance & VRAM Optimization",open =True ):
                        max_chunk_len_slider =gr .Slider (
                        label ="Max Frames per Chunk (VRAM)",
                        minimum =4 ,maximum =96 ,value =app_config .DEFAULT_MAX_CHUNK_LEN ,step =4 ,
                        info ="""IMPORTANT for VRAM. This is the standard way the application manages VRAM. It divides the entire sequence of video frames into sequential, non-overlapping chunks.
- Mechanism: The STAR model processes one complete chunk (of this many frames) at a time.
- VRAM Impact: High Reduction (Lower value = Less VRAM).
- Bigger Chunk + Bigger VRAM = Faster Processing and Better Quality 
- Quality Impact: Can reduce Temporal Consistency (flicker/motion issues) between chunks if too low, as the model doesn't have context across chunk boundaries. Keep as high as VRAM allows.
- Speed Impact: Slower (Lower value = Slower, as more chunks are processed)."""
                        )
                        vae_chunk_slider =gr .Slider (
                        label ="VAE Decode Chunk (VRAM)",
                        minimum =1 ,maximum =16 ,value =app_config .DEFAULT_VAE_CHUNK ,step =1 ,
                        info ="""Controls max latent frames decoded back to pixels by VAE simultaneously.
- VRAM Impact: High Reduction (Lower value = Less VRAM during decode stage).
- Quality Impact: Minimal / Negligible. Safe to lower.
- Speed Impact: Slower (Lower value = Slower decoding)."""
                        )

            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("Sliding Window - Overlapped Processing Thus Better Quality but Slower",open =True ):
                        enable_sliding_window_check =gr .Checkbox (
                        label ="Enable Sliding Window",
                        value =app_config .DEFAULT_ENABLE_SLIDING_WINDOW ,
                        info ="""Processes the video in overlapping temporal chunks (windows). Use if you aim maximum quality and have better consistency between chunks.
- Mechanism: Takes a 'Window Size' of frames, processes it, saves results from the central part, then slides the window forward by 'Window Step', processing overlapping frames.
- Quality Impact: Can improve quality of processing if you use higher Windows and overlapped frame count
- VRAM Impact : To lower VRAM you can set like Window Size 8 and Window Step Size 4
- Speed Impact: Slower (due to processing overlapping frames multiple times). When enabled, 'Window Size' dictates chunk size instead of 'Max Frames per Chunk'."""
                        )
                        with gr .Row ():
                            window_size_num =gr .Slider (
                            label ="Window Size (frames)",
                            value =app_config .DEFAULT_WINDOW_SIZE ,minimum =2 ,maximum =256 ,step =4 ,
                            info ="Number of frames in each temporal window. Acts like 'Max Frames per Chunk' but applied as a sliding window. Lower value = less VRAM, less temporal context."
                            )
                            window_step_num =gr .Slider (
                            label ="Window Step (frames)",
                            value =app_config .DEFAULT_WINDOW_STEP ,minimum =1 ,maximum =128 ,step =1 ,
                            info ="How many frames to advance for the next window. (Window Size - Window Step) = Overlap. Smaller step = more overlap = better consistency but slower. Recommended: Step = Size / 2."
                            )

                with gr .Column (scale =1 ):
                    with gr .Accordion ("Advanced: Tiling (Very High Res / Low VRAM)",open =True ):
                        enable_tiling_check =gr .Checkbox (
                        label ="Enable Tiled Upscaling",
                        value =app_config .DEFAULT_ENABLE_TILING ,
                        info ="""Processes each frame in small spatial patches (tiles). Use ONLY if necessary for extreme resolutions or very low VRAM.
- VRAM Impact: Very High Reduction.
- Quality Impact: High risk of tile seams/artifacts. Can harm global coherence and severely reduce temporal consistency.
- Speed Impact: Extremely Slow."""
                        )
                        with gr .Row ():
                            tile_size_num =gr .Number (
                            label ="Tile Size (px, input res)",
                            value =app_config .DEFAULT_TILE_SIZE ,minimum =64 ,step =32 ,
                            info ="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                            )
                            tile_overlap_num =gr .Number (
                            label ="Tile Overlap (px, input res)",
                            value =app_config .DEFAULT_TILE_OVERLAP ,minimum =0 ,step =16 ,
                            info ="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                            )

        with gr .Tab ("Output & Comparison"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("FFmpeg Encoding Settings",open =True ):
                        ffmpeg_use_gpu_check =gr .Checkbox (
                        label ="Use NVIDIA GPU for FFmpeg (h264_nvenc)",
                        value =app_config .DEFAULT_FFMPEG_USE_GPU ,
                        info ="If checked, uses NVIDIA's NVENC for FFmpeg video encoding (downscaling and final video creation). Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."
                        )
                        with gr .Row ():
                            ffmpeg_preset_dropdown =gr .Dropdown (
                            label ="FFmpeg Preset",
                            choices =['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],
                            value =app_config .DEFAULT_FFMPEG_PRESET ,
                            info ="Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently (e.g. p1-p7 or specific names like 'slow', 'medium', 'fast')."
                            )
                            ffmpeg_quality_slider =gr .Slider (
                            label ="FFmpeg Quality (CRF for libx264 / CQ for NVENC)",
                            minimum =0 ,maximum =51 ,value =app_config .DEFAULT_FFMPEG_QUALITY_CPU ,step =1 ,
                            info ="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28."
                            )

                with gr .Column (scale =1 ):
                    with gr .Accordion ("Output Options",open =True ):
                        create_comparison_video_check =gr .Checkbox (
                        label ="Generate Comparison Video",
                        value =app_config .DEFAULT_CREATE_COMPARISON_VIDEO ,
                        info ="""Create a side-by-side or top-bottom comparison video showing original vs upscaled quality.
The layout is automatically chosen based on aspect ratio to stay within 1920x1080 bounds when possible.
This helps visualize the quality improvement from upscaling."""
                        )
                        save_frames_checkbox =gr .Checkbox (label ="Save Input and Processed Frames",value =app_config .DEFAULT_SAVE_FRAMES ,info ="If checked, saves the extracted input frames and the upscaled output frames into a subfolder named after the output video (e.g., '0001/input_frames' and '0001/processed_frames').")
                        save_metadata_checkbox =gr .Checkbox (label ="Save Processing Metadata",value =app_config .DEFAULT_SAVE_METADATA ,info ="If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time.")
                        save_chunks_checkbox =gr .Checkbox (label ="Save Processed Chunks",value =app_config .DEFAULT_SAVE_CHUNKS ,info ="If checked, saves each processed chunk as a video file in a 'chunks' subfolder (e.g., '0001/chunks/chunk_0001.mp4'). Uses the same FFmpeg settings as the final video.")

                    with gr .Accordion ("Advanced: Seeding (Reproducibility)",open =True ):
                        with gr .Row ():
                            seed_num =gr .Number (
                            label ="Seed",
                            value =app_config .DEFAULT_SEED ,
                            minimum =-1 ,
                            maximum =2 **32 -1 ,
                            step =1 ,
                            info ="Seed for random number generation. Used for reproducibility. Set to -1 or check 'Random Seed' for a random seed. Value is ignored if 'Random Seed' is checked.",
                            interactive =not app_config .DEFAULT_RANDOM_SEED 
                            )
                            random_seed_check =gr .Checkbox (
                            label ="Random Seed",
                            value =app_config .DEFAULT_RANDOM_SEED ,
                            info ="If checked, a random seed will be generated and used, ignoring the 'Seed' value."
                            )

                with gr .Column (scale =1 ):
                    with gr .Accordion ("Manual Comparison Video Generator",open =True ):
                        gr .Markdown ("### Generate Custom Comparison Videos")
                        gr .Markdown ("Upload two videos to create a manual side-by-side or top-bottom comparison video using the same FFmpeg settings and layout logic as the automatic comparison feature.")

                        gr .Markdown ("**Step 1:** Upload the original or reference video for comparison")
                        manual_original_video =gr .Video (
                        label ="Original/Reference Video",
                        sources =["upload"],
                        interactive =True ,
                        height =200 
                        )

                        gr .Markdown ("**Step 2:** Upload the upscaled or enhanced video for comparison")
                        manual_upscaled_video =gr .Video (
                        label ="Upscaled/Enhanced Video",
                        sources =["upload"],
                        interactive =True ,
                        height =200 
                        )

                        gr .Markdown ("**Step 3:** Generate the comparison video using current FFmpeg settings")
                        manual_comparison_button =gr .Button (
                        "Generate Manual Comparison Video",
                        variant ="primary",
                        size ="lg"
                        )

                        manual_comparison_status =gr .Textbox (
                        label ="Manual Comparison Status",
                        lines =2 ,
                        interactive =False ,
                        visible =False 
                        )

            with gr .Accordion ("Comparison Video To See Difference",open =True ):
                comparison_video =gr .Video (label ="Comparison Video",interactive =False ,height =512 )

        with gr .Tab ("Batch Upscaling"):

            with gr .Accordion ("Batch Processing Options",open =True ):
                with gr .Row ():
                    batch_input_folder =gr .Textbox (
                    label ="Input Folder",
                    placeholder ="Path to folder containing videos to process...",
                    info ="Folder containing video files to process in batch mode."
                    )
                    batch_output_folder =gr .Textbox (
                    label ="Output Folder",
                    placeholder ="Path to output folder for processed videos...",
                    info ="Folder where processed videos will be saved with organized structure."
                    )

                with gr .Row ():

                    batch_skip_existing =gr .Checkbox (
                    label ="Skip Existing Outputs",
                    value =app_config .DEFAULT_BATCH_SKIP_EXISTING ,
                    info ="Skip processing if the output file already exists. Useful for resuming interrupted batch jobs."
                    )

                    batch_use_prompt_files =gr .Checkbox (
                    label ="Use Prompt Files (filename.txt)",
                    value =app_config .DEFAULT_BATCH_USE_PROMPT_FILES ,
                    info ="Look for text files with same name as video (e.g., video.txt) to use as custom prompts. Takes priority over user prompt and auto-caption."
                    )

                    batch_save_captions =gr .Checkbox (
                    label ="Save Auto-Generated Captions",
                    value =app_config .DEFAULT_BATCH_SAVE_CAPTIONS ,
                    info ="Save auto-generated captions as filename.txt in the input folder for future reuse. Never overwrites existing prompt files."
                    )

                if app_config .UTIL_COG_VLM_AVAILABLE :
                    with gr .Row ():
                        batch_enable_auto_caption =gr .Checkbox (
                        label ="Enable Auto-Caption for Batch",
                        value =True ,
                        info ="Generate automatic captions for videos that don't have prompt files. Uses the same CogVLM2 settings as Core Settings tab."
                        )
                else :

                    batch_enable_auto_caption =gr .Checkbox (visible =False ,value =False )

            with gr .Row ():
                batch_process_button =gr .Button ("Start Batch Upscaling",variant ="primary",icon ="icons/split.png")

            # Add the help content at the bottom in 3 columns
            create_batch_processing_help()

        with gr .Tab ("FPS Increase"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("RIFE Interpolation Settings",open =True ):
                        gr .Markdown ("### Frame Interpolation (RIFE)")
                        gr .Markdown ("**RIFE (Real-time Intermediate Flow Estimation)** uses AI to intelligently generate intermediate frames between existing frames, increasing video smoothness and frame rate.")

                        enable_rife_interpolation =gr .Checkbox (
                        label ="Enable RIFE Interpolation",
                        value =app_config .DEFAULT_RIFE_ENABLE_INTERPOLATION ,
                        info ="Enable AI-powered frame interpolation to increase video FPS. When enabled, RIFE will be applied to the final upscaled video and can optionally be applied to intermediate chunks and scenes."
                        )

                        with gr .Row ():
                            rife_multiplier =gr .Radio (
                            label ="FPS Multiplier",
                            choices =[2 ,4 ],
                            value =app_config .DEFAULT_RIFE_MULTIPLIER ,
                            info ="Choose how much to increase the frame rate. 2x doubles FPS (e.g., 24→48), 4x quadruples FPS (e.g., 24→96)."
                            )

                        with gr .Row ():
                            rife_fp16 =gr .Checkbox (
                            label ="Use FP16 Precision",
                            value =app_config .DEFAULT_RIFE_FP16 ,
                            info ="Use half-precision floating point for faster processing and lower VRAM usage. Recommended for most users."
                            )
                            rife_uhd =gr .Checkbox (
                            label ="UHD Mode",
                            value =app_config .DEFAULT_RIFE_UHD ,
                            info ="Enable UHD mode for 4K+ videos. May improve quality for very high resolution content but requires more VRAM."
                            )

                        rife_scale =gr .Slider (
                        label ="Scale Factor",
                        minimum =0.25 ,maximum =2.0 ,value =app_config .DEFAULT_RIFE_SCALE ,step =0.25 ,
                        info ="Scale factor for RIFE processing. 1.0 = original size. Lower values use less VRAM but may reduce quality. Higher values may improve quality but use more VRAM."
                        )

                        rife_skip_static =gr .Checkbox (
                        label ="Skip Static Frames",
                        value =app_config .DEFAULT_RIFE_SKIP_STATIC ,
                        info ="Automatically detect and skip interpolating static (non-moving) frames to save processing time and avoid unnecessary interpolation."
                        )

                    with gr .Accordion ("Intermediate Processing",open =True ):
                        gr .Markdown ("**Apply RIFE to intermediate videos (recommended)**")
                        rife_apply_to_chunks =gr .Checkbox (
                        label ="Apply to Chunks",
                        value =app_config .DEFAULT_RIFE_APPLY_TO_CHUNKS ,
                        info ="Apply RIFE interpolation to individual video chunks during processing. Enabled by default for smoother intermediate results."
                        )
                        rife_apply_to_scenes =gr .Checkbox (
                        label ="Apply to Scenes",
                        value =app_config .DEFAULT_RIFE_APPLY_TO_SCENES ,
                        info ="Apply RIFE interpolation to individual scene videos when scene splitting is enabled. Enabled by default for consistent results."
                        )

                        gr .Markdown ("**Note:** When RIFE is enabled, the system will return RIFE-interpolated versions to the interface instead of originals, ensuring you get the smoothest possible results throughout the processing pipeline.")

                with gr .Column (scale =1 ):
                    with gr .Accordion ("FPS Decrease",open =True ):
                        gr .Markdown ("### Pre-Processing FPS Reduction")
                        gr .Markdown ("**Reduce FPS before upscaling** to speed up processing and reduce VRAM usage. You can then use RIFE interpolation to restore smooth motion afterward.")

                        enable_fps_decrease =gr .Checkbox (
                        label ="Enable FPS Decrease",
                        value =app_config .DEFAULT_ENABLE_FPS_DECREASE ,
                        info ="Reduce video FPS before upscaling to speed up processing. Fewer frames = faster upscaling and lower VRAM usage."
                        )

                        fps_decrease_mode =gr .Radio (
                        label ="FPS Reduction Mode",
                        choices =["multiplier","fixed"],
                        value =app_config .DEFAULT_FPS_DECREASE_MODE ,
                        info ="Multiplier: Reduce by fraction (1/2x, 1/4x). Fixed: Set specific FPS value. Multiplier is recommended for automatic adaptation to input video."
                        )

                        with gr .Group ()as multiplier_controls :
                            with gr .Row ():
                                fps_multiplier_preset =gr .Dropdown (
                                label ="FPS Multiplier",
                                choices =list (util_get_common_fps_multipliers ().values ())+["Custom"],
                                value ="1/2x (Half FPS)",
                                info ="Choose common multiplier. 1/2x is recommended for good speed/quality balance."
                                )
                                fps_multiplier_custom =gr .Number (
                                label ="Custom Multiplier",
                                value =app_config .DEFAULT_FPS_MULTIPLIER ,
                                minimum =0.1 ,
                                maximum =1.0 ,
                                step =0.05 ,
                                precision =2 ,
                                visible =False ,
                                info ="Custom multiplier value (0.1 to 1.0). Lower = fewer frames."
                                )

                        with gr .Group ()as fixed_controls :
                            target_fps =gr .Dropdown (
                            label ="Target FPS",
                            choices =[8.0 ,10.0 ,12.0 ,15.0 ,18.0 ,20.0 ,23.976 ,24.0 ,25.0 ,29.970 ,30.0 ],
                            value =app_config .DEFAULT_TARGET_FPS ,
                            info ="Target FPS for the reduced video. Lower FPS = faster upscaling. Common choices: 12-15 FPS for fast processing, 24 FPS for cinema standard.",
                            visible =False 
                            )

                        fps_interpolation_method =gr .Radio (
                        label ="Frame Reduction Method",
                        choices =["drop","blend"],
                        value =app_config .DEFAULT_FPS_INTERPOLATION_METHOD ,
                        info ="Drop: Faster, simply removes frames. Blend: Smoother, blends frames together (slower but may preserve motion better)."
                        )

                        fps_calculation_info =gr .Markdown (
                        "**📊 Calculation:** Upload a video to see FPS reduction preview",
                        visible =True 
                        )

                        gr .Markdown ("**💡 Workflow Tip:** Use FPS decrease (1/2x for balanced speed/quality) for faster upscaling, then enable RIFE 2x-4x to restore smooth 24-60 FPS output!")

                    with gr .Accordion ("FPS Limiting & Output Control",open =True ):
                        rife_enable_fps_limit =gr .Checkbox (
                        label ="Enable FPS Limiting",
                        value =app_config .DEFAULT_RIFE_ENABLE_FPS_LIMIT ,
                        info ="Limit the output FPS to specific common values instead of unlimited interpolation. Useful for compatibility with displays and media players."
                        )

                        rife_max_fps_limit =gr .Radio (
                        label ="Max FPS Limit",
                        choices =[23.976 ,24 ,25 ,29.970 ,30 ,47.952 ,48 ,50 ,59.940 ,60 ,75 ,90 ,100 ,119.880 ,120 ,144 ,165 ,180 ,240 ,360 ],
                        value =app_config .DEFAULT_RIFE_MAX_FPS_LIMIT ,
                        info ="Maximum FPS when limiting is enabled. NTSC rates: 23.976/29.970/59.940 (film/TV), Standard: 24/25/30/50/60, Gaming: 120/144/240+. Choose based on your target format and display."
                        )

                        with gr .Row ():
                            rife_keep_original =gr .Checkbox (
                            label ="Keep Original Files",
                            value =app_config .DEFAULT_RIFE_KEEP_ORIGINAL ,
                            info ="Keep the original (non-interpolated) video files alongside the RIFE-processed versions. Recommended to compare results."
                            )
                            rife_overwrite_original =gr .Checkbox (
                            label ="Overwrite Original",
                            value =app_config .DEFAULT_RIFE_OVERWRITE_ORIGINAL ,
                            info ="Replace the original upscaled video with the RIFE version as the primary output. When disabled, both versions are available."
                            )

    def update_steps_display (mode ):
        if mode =='fast':
            return gr .update (value =app_config .DEFAULT_DIFFUSION_STEPS_FAST ,interactive =False )
        else :
            return gr .update (value =app_config .DEFAULT_DIFFUSION_STEPS_NORMAL ,interactive =True )
    solver_mode_radio .change (update_steps_display ,solver_mode_radio ,steps_slider )

    enable_target_res_check .change (
    lambda x :[gr .update (interactive =x )]*3 ,
    inputs =enable_target_res_check ,
    outputs =[target_h_num ,target_w_num ,target_res_mode_radio ]
    )
    enable_tiling_check .change (
    lambda x :[gr .update (interactive =x )]*2 ,
    inputs =enable_tiling_check ,
    outputs =[tile_size_num ,tile_overlap_num ]
    )
    enable_sliding_window_check .change (
    lambda x :[gr .update (interactive =x )]*2 ,
    inputs =enable_sliding_window_check ,
    outputs =[window_size_num ,window_step_num ]
    )
    scene_splitting_outputs =[
    scene_split_mode_radio ,scene_min_scene_len_num ,scene_threshold_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_manual_split_type_radio ,scene_manual_split_value_num ,scene_copy_streams_check ,
    scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check 
    ]
    enable_scene_split_check .change (
    lambda x :[gr .update (interactive =x )]*len (scene_splitting_outputs ),
    inputs =enable_scene_split_check ,
    outputs =scene_splitting_outputs 
    )

    def update_ffmpeg_quality_settings (use_gpu_ffmpeg ):
        if use_gpu_ffmpeg :
            return gr .Slider (label ="FFmpeg Quality (CQ for NVENC)",value =app_config .DEFAULT_FFMPEG_QUALITY_GPU ,info ="For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28.")
        else :
            return gr .Slider (label ="FFmpeg Quality (CRF for libx264)",value =app_config .DEFAULT_FFMPEG_QUALITY_CPU ,info ="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default).")

    ffmpeg_use_gpu_check .change (
    fn =update_ffmpeg_quality_settings ,
    inputs =ffmpeg_use_gpu_check ,
    outputs =ffmpeg_quality_slider 
    )

    open_output_folder_button .click (
    fn =lambda :util_open_folder (app_config .DEFAULT_OUTPUT_DIR ,logger =logger ),
    inputs =[],
    outputs =[]
    )

    cogvlm_display_to_quant_val_map_global ={}
    if app_config .UTIL_COG_VLM_AVAILABLE :
        _temp_map =app_config .get_cogvlm_quant_choices_map (torch .cuda .is_available (),app_config .UTIL_BITSANDBYTES_AVAILABLE )
        cogvlm_display_to_quant_val_map_global ={v :k for k ,v in _temp_map .items ()}

    def get_quant_value_from_display (display_val ):
        if display_val is None :return 0 
        if isinstance (display_val ,int ):return display_val 
        return cogvlm_display_to_quant_val_map_global .get (display_val ,0 )

    def upscale_director_logic (
    input_video_val ,user_prompt_val ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
    upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
    max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
    enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
    enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
    enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
    ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
    save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,
    create_comparison_video_check_val ,
    enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
    scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
    scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
    scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,

    enable_fps_decrease_val =False ,fps_decrease_mode_val ="multiplier",fps_multiplier_preset_val ="1/2x (Half FPS)",fps_multiplier_custom_val =0.5 ,target_fps_val =24.0 ,fps_interpolation_method_val ="drop",

    enable_rife_interpolation_val =False ,rife_multiplier_val =2 ,rife_fp16_val =True ,rife_uhd_val =False ,rife_scale_val =1.0 ,
    rife_skip_static_val =False ,rife_enable_fps_limit_val =False ,rife_max_fps_limit_val =60 ,
    rife_apply_to_chunks_val =True ,rife_apply_to_scenes_val =True ,rife_keep_original_val =True ,rife_overwrite_original_val =False ,
    cogvlm_quant_radio_val =None ,cogvlm_unload_radio_val =None ,
    do_auto_caption_first_val =False ,
    seed_num_val =99 ,random_seed_check_val =False ,
    progress =gr .Progress (track_tqdm =True )
    ):

        current_output_video_val =None 
        current_status_text_val =""
        current_user_prompt_val =user_prompt_val 
        current_caption_status_text_val =""
        current_caption_status_visible_val =False 
        current_last_chunk_video_val =None 
        current_chunk_status_text_val ="No chunks processed yet"
        current_comparison_video_val =None 

        auto_caption_completed_successfully =False 

        log_accumulator_director =[]

        logger .info (f"In upscale_director_logic. Auto-caption first: {do_auto_caption_first_val}, User prompt: '{user_prompt_val[:50]}...'")

        actual_cogvlm_quant_for_captioning =get_quant_value_from_display (cogvlm_quant_radio_val )

        should_auto_caption_entire_video =(do_auto_caption_first_val and 
        not enable_scene_split_check_val and 
        app_config .UTIL_COG_VLM_AVAILABLE )

        if should_auto_caption_entire_video :
            logger .info ("Attempting auto-captioning entire video before upscale (scene splitting disabled).")
            progress (0 ,desc ="Starting auto-captioning before upscale...")

            current_status_text_val ="Starting auto-captioning..."
            if app_config .UTIL_COG_VLM_AVAILABLE :
                current_caption_status_text_val ="Starting auto-captioning..."
                current_caption_status_visible_val =True 

            yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
            gr .update (value =current_user_prompt_val ),
            gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
            gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
            gr .update (value =current_comparison_video_val ))

            try :
                caption_text ,caption_stat_msg =util_auto_caption (
                input_video_val ,actual_cogvlm_quant_for_captioning ,cogvlm_unload_radio_val ,
                app_config .COG_VLM_MODEL_PATH ,logger =logger ,progress =progress 
                )
                log_accumulator_director .append (f"Auto-caption status: {caption_stat_msg}")

                if not caption_text .startswith ("Error:"):
                    current_user_prompt_val =caption_text 
                    auto_caption_completed_successfully =True 
                    log_accumulator_director .append (f"Using generated caption as prompt: '{caption_text[:50]}...'")
                    logger .info (f"Auto-caption successful. Updated current_user_prompt_val to: '{current_user_prompt_val[:100]}...'")
                else :
                    log_accumulator_director .append ("Caption generation failed. Using original prompt.")
                    logger .warning (f"Auto-caption failed. Keeping original prompt: '{current_user_prompt_val[:100]}...'")

                current_status_text_val ="\n".join (log_accumulator_director )
                if app_config .UTIL_COG_VLM_AVAILABLE :
                    current_caption_status_text_val =caption_stat_msg 

                logger .info (f"About to yield auto-caption result. current_user_prompt_val: '{current_user_prompt_val[:100]}...'")
                yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
                gr .update (value =current_user_prompt_val ),
                gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
                gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
                gr .update (value =current_comparison_video_val ))
                logger .info ("Auto-caption yield completed.")

                if app_config .UTIL_COG_VLM_AVAILABLE :
                    current_caption_status_visible_val =False 

            except Exception as e_ac :
                logger .error (f"Exception during auto-caption call or its setup: {e_ac}",exc_info =True )
                log_accumulator_director .append (f"Error during auto-caption pre-step: {e_ac}")
                current_status_text_val ="\n".join (log_accumulator_director )
                if app_config .UTIL_COG_VLM_AVAILABLE :
                    current_caption_status_text_val =str (e_ac )
                    yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
                    gr .update (value =current_user_prompt_val ),
                    gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
                    gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
                    gr .update (value =current_comparison_video_val ))
                    current_caption_status_visible_val =False 
                else :
                    yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
                    gr .update (value =current_user_prompt_val ),
                    gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
                    gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
                    gr .update (value =current_comparison_video_val ))
            log_accumulator_director =[]

        elif do_auto_caption_first_val and enable_scene_split_check_val and app_config .UTIL_COG_VLM_AVAILABLE :
            msg ="Scene splitting enabled: Auto-captioning will be done per scene during upscaling."
            logger .info (msg )
            log_accumulator_director .append (msg )
            current_status_text_val ="\n".join (log_accumulator_director )

            yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
            gr .update (value =current_user_prompt_val ),
            gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
            gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
            gr .update (value =current_comparison_video_val ))

        elif do_auto_caption_first_val and not app_config .UTIL_COG_VLM_AVAILABLE :
            msg ="Auto-captioning requested but CogVLM2 is not available. Using original prompt."
            logger .warning (msg )
            log_accumulator_director .append (msg )
            current_status_text_val ="\n".join (log_accumulator_director )

            yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
            gr .update (value =current_user_prompt_val ),
            gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
            gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
            gr .update (value =current_comparison_video_val ))

        actual_seed_to_use =seed_num_val 
        if random_seed_check_val :
            actual_seed_to_use =np .random .randint (0 ,2 **31 )
            logger .info (f"Random seed checkbox is checked. Using generated seed: {actual_seed_to_use}")
        elif seed_num_val ==-1 :
            actual_seed_to_use =np .random .randint (0 ,2 **31 )
            logger .info (f"Seed input is -1. Using generated seed: {actual_seed_to_use}")
        else :
            logger .info (f"Using provided seed: {actual_seed_to_use}")

        upscale_generator =core_run_upscale (
        input_video_path =input_video_val ,user_prompt =current_user_prompt_val ,
        positive_prompt =pos_prompt_val ,negative_prompt =neg_prompt_val ,model_choice =model_selector_val ,
        upscale_factor_slider =upscale_factor_slider_val ,cfg_scale =cfg_slider_val ,steps =steps_slider_val ,solver_mode =solver_mode_radio_val ,
        max_chunk_len =max_chunk_len_slider_val ,vae_chunk =vae_chunk_slider_val ,color_fix_method =color_fix_dropdown_val ,
        enable_tiling =enable_tiling_check_val ,tile_size =tile_size_num_val ,tile_overlap =tile_overlap_num_val ,
        enable_sliding_window =enable_sliding_window_check_val ,window_size =window_size_num_val ,window_step =window_step_num_val ,
        enable_target_res =enable_target_res_check_val ,target_h =target_h_num_val ,target_w =target_w_num_val ,target_res_mode =target_res_mode_radio_val ,
        ffmpeg_preset =ffmpeg_preset_dropdown_val ,ffmpeg_quality_value =ffmpeg_quality_slider_val ,ffmpeg_use_gpu =ffmpeg_use_gpu_check_val ,
        save_frames =save_frames_checkbox_val ,save_metadata =save_metadata_checkbox_val ,save_chunks =save_chunks_checkbox_val ,

        enable_scene_split =enable_scene_split_check_val ,scene_split_mode =scene_split_mode_radio_val ,scene_min_scene_len =scene_min_scene_len_num_val ,scene_drop_short =scene_drop_short_check_val ,scene_merge_last =scene_merge_last_check_val ,
        scene_frame_skip =scene_frame_skip_num_val ,scene_threshold =scene_threshold_num_val ,scene_min_content_val =scene_min_content_val_num_val ,scene_frame_window =scene_frame_window_num_val ,
        scene_copy_streams =scene_copy_streams_check_val ,scene_use_mkvmerge =scene_use_mkvmerge_check_val ,scene_rate_factor =scene_rate_factor_num_val ,scene_preset =scene_preset_dropdown_val ,scene_quiet_ffmpeg =scene_quiet_ffmpeg_check_val ,
        scene_manual_split_type =scene_manual_split_type_radio_val ,scene_manual_split_value =scene_manual_split_value_num_val ,

        create_comparison_video_enabled =create_comparison_video_check_val ,

        enable_fps_decrease =enable_fps_decrease_val ,fps_decrease_mode =fps_decrease_mode_val ,
        fps_multiplier_preset =fps_multiplier_preset_val ,fps_multiplier_custom =fps_multiplier_custom_val ,
        target_fps =target_fps_val ,fps_interpolation_method =fps_interpolation_method_val ,

        enable_rife_interpolation =enable_rife_interpolation_val ,rife_multiplier =rife_multiplier_val ,rife_fp16 =rife_fp16_val ,rife_uhd =rife_uhd_val ,rife_scale =rife_scale_val ,
        rife_skip_static =rife_skip_static_val ,rife_enable_fps_limit =rife_enable_fps_limit_val ,rife_max_fps_limit =rife_max_fps_limit_val ,
        rife_apply_to_chunks =rife_apply_to_chunks_val ,rife_apply_to_scenes =rife_apply_to_scenes_val ,rife_keep_original =rife_keep_original_val ,rife_overwrite_original =rife_overwrite_original_val ,

        is_batch_mode =False ,batch_output_dir =None ,original_filename =None ,

        enable_auto_caption_per_scene =(do_auto_caption_first_val and enable_scene_split_check_val and app_config .UTIL_COG_VLM_AVAILABLE ),
        cogvlm_quant =actual_cogvlm_quant_for_captioning ,
        cogvlm_unload =cogvlm_unload_radio_val if cogvlm_unload_radio_val else 'full',

        logger =logger ,
        app_config_module =app_config ,
        metadata_handler_module =metadata_handler ,
        VideoToVideo_sr_class =VideoToVideo_sr ,
        setup_seed_func =setup_seed ,
        EasyDict_class =EasyDict ,
        preprocess_func =preprocess ,
        collate_fn_func =collate_fn ,
        tensor2vid_func =tensor2vid ,
        ImageSpliterTh_class =ImageSpliterTh ,
        adain_color_fix_func =adain_color_fix ,
        wavelet_color_fix_func =wavelet_color_fix ,
        progress =progress ,
        current_seed =actual_seed_to_use 
        )

        for yielded_output_video ,yielded_status_log ,yielded_chunk_video ,yielded_chunk_status ,yielded_comparison_video in upscale_generator :

            output_video_update =gr .update ()
            if yielded_output_video is not None :
                current_output_video_val =yielded_output_video 
                output_video_update =gr .update (value =current_output_video_val )
            elif current_output_video_val is None :
                output_video_update =gr .update (value =None )
            else :
                output_video_update =gr .update (value =current_output_video_val )

            combined_log_director =""
            if log_accumulator_director :
                combined_log_director ="\n".join (log_accumulator_director )+"\n"
                log_accumulator_director =[]
            if yielded_status_log :
                combined_log_director +=yielded_status_log 
            current_status_text_val =combined_log_director .strip ()
            status_text_update =gr .update (value =current_status_text_val )

            if yielded_status_log and "[FIRST_SCENE_CAPTION:"in yielded_status_log and not auto_caption_completed_successfully :
                try :
                    caption_start =yielded_status_log .find ("[FIRST_SCENE_CAPTION:")+len ("[FIRST_SCENE_CAPTION:")
                    caption_end =yielded_status_log .find ("]",caption_start )
                    if caption_start >len ("[FIRST_SCENE_CAPTION:")and caption_end >caption_start :
                        extracted_caption =yielded_status_log [caption_start :caption_end ]
                        current_user_prompt_val =extracted_caption 
                        auto_caption_completed_successfully =True 
                        logger .info (f"Updated main prompt from first scene caption: '{extracted_caption[:100]}...'")

                        log_accumulator_director .append (f"Main prompt updated with first scene caption: '{extracted_caption[:50]}...'")
                        current_status_text_val =(combined_log_director +"\n"+"\n".join (log_accumulator_director )).strip ()
                        status_text_update =gr .update (value =current_status_text_val )
                except Exception as e :
                    logger .error (f"Error extracting first scene caption: {e}")

            elif yielded_status_log and "FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE:"in yielded_status_log and not auto_caption_completed_successfully :
                try :
                    caption_start =yielded_status_log .find ("FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE:")+len ("FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE:")
                    extracted_caption =yielded_status_log [caption_start :].strip ()
                    if extracted_caption :
                        current_user_prompt_val =extracted_caption 
                        auto_caption_completed_successfully =True 
                        logger .info (f"Updated main prompt from immediate first scene caption: '{extracted_caption[:100]}...'")
                        log_accumulator_director .append (f"Main prompt updated with first scene caption: '{extracted_caption[:50]}...'")
                        current_status_text_val =(combined_log_director +"\n"+"\n".join (log_accumulator_director )).strip ()
                        status_text_update =gr .update (value =current_status_text_val )
                except Exception as e :
                    logger .error (f"Error extracting immediate first scene caption: {e}")

            user_prompt_update =gr .update (value =current_user_prompt_val )

            caption_status_update =gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val )

            chunk_video_update =gr .update ()
            if yielded_chunk_video is not None :
                current_last_chunk_video_val =yielded_chunk_video 
                chunk_video_update =gr .update (value =current_last_chunk_video_val )
            elif current_last_chunk_video_val is None :
                chunk_video_update =gr .update (value =None )
            else :
                chunk_video_update =gr .update (value =current_last_chunk_video_val )

            if yielded_chunk_status is not None :
                current_chunk_status_text_val =yielded_chunk_status 
            chunk_status_text_update =gr .update (value =current_chunk_status_text_val )

            comparison_video_update =gr .update ()
            if yielded_comparison_video is not None :
                current_comparison_video_val =yielded_comparison_video 
                comparison_video_update =gr .update (value =current_comparison_video_val )
            elif current_comparison_video_val is None :
                comparison_video_update =gr .update (value =None )
            else :
                comparison_video_update =gr .update (value =current_comparison_video_val )

            yield (
            output_video_update ,status_text_update ,user_prompt_update ,
            caption_status_update ,
            chunk_video_update ,chunk_status_text_update ,
            comparison_video_update 
            )

        logger .info (f"Final yield: current_user_prompt_val = '{current_user_prompt_val[:100]}...', auto_caption_completed = {auto_caption_completed_successfully}")
        yield (
        gr .update (value =current_output_video_val ),
        gr .update (value =current_status_text_val ),
        gr .update (value =current_user_prompt_val ),
        gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
        gr .update (value =current_last_chunk_video_val ),
        gr .update (value =current_chunk_status_text_val ),
        gr .update (value =current_comparison_video_val )
        )

    click_inputs =[
    input_video ,user_prompt ,pos_prompt ,neg_prompt ,model_selector ,
    upscale_factor_slider ,cfg_slider ,steps_slider ,solver_mode_radio ,
    max_chunk_len_slider ,vae_chunk_slider ,color_fix_dropdown ,
    enable_tiling_check ,tile_size_num ,tile_overlap_num ,
    enable_sliding_window_check ,window_size_num ,window_step_num ,
    enable_target_res_check ,target_h_num ,target_w_num ,target_res_mode_radio ,
    ffmpeg_preset_dropdown ,ffmpeg_quality_slider ,ffmpeg_use_gpu_check ,
    save_frames_checkbox ,save_metadata_checkbox ,save_chunks_checkbox ,
    create_comparison_video_check ,
    enable_scene_split_check ,scene_split_mode_radio ,scene_min_scene_len_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_threshold_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_copy_streams_check ,scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check ,
    scene_manual_split_type_radio ,scene_manual_split_value_num ,

    enable_fps_decrease ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ,fps_interpolation_method ,

    enable_rife_interpolation ,rife_multiplier ,rife_fp16 ,rife_uhd ,rife_scale ,
    rife_skip_static ,rife_enable_fps_limit ,rife_max_fps_limit ,
    rife_apply_to_chunks ,rife_apply_to_scenes ,rife_keep_original ,rife_overwrite_original 
    ]

    click_outputs_list =[output_video ,status_textbox ,user_prompt ]
    if app_config .UTIL_COG_VLM_AVAILABLE :
        click_outputs_list .append (caption_status )
        click_inputs .extend ([cogvlm_quant_radio ,cogvlm_unload_radio ,auto_caption_then_upscale_check ])
    else :
        click_outputs_list .append (gr .State (None ))
        click_inputs .extend ([gr .State (None ),gr .State (None ),gr .State (False )])

    click_inputs .extend ([seed_num ,random_seed_check ])

    click_outputs_list .extend ([last_chunk_video ,chunk_status_text ])

    click_outputs_list .append (comparison_video )

    upscale_button .click (
    fn =upscale_director_logic ,
    inputs =click_inputs ,
    outputs =click_outputs_list ,
    show_progress_on =[output_video ]
    )

    if app_config .UTIL_COG_VLM_AVAILABLE :
        def auto_caption_wrapper (vid ,quant_display ,unload_strat ,progress =gr .Progress (track_tqdm =True )):
            caption_text ,caption_stat_msg =util_auto_caption (
            vid ,
            get_quant_value_from_display (quant_display ),
            unload_strat ,
            app_config .COG_VLM_MODEL_PATH ,
            logger =logger ,
            progress =progress 
            )
            return caption_text ,caption_stat_msg 

        def rife_fps_increase_wrapper (
        input_video_val ,
        rife_multiplier_val =2 ,
        rife_fp16_val =True ,
        rife_uhd_val =False ,
        rife_scale_val =1.0 ,
        rife_skip_static_val =False ,
        rife_enable_fps_limit_val =False ,
        rife_max_fps_limit_val =60 ,
        ffmpeg_preset_dropdown_val ="medium",
        ffmpeg_quality_slider_val =18 ,
        ffmpeg_use_gpu_check_val =True ,
        seed_num_val =99 ,
        random_seed_check_val =False ,
        progress =gr .Progress (track_tqdm =True )
        ):

            return util_rife_fps_only_wrapper (
            input_video_val =input_video_val ,
            rife_multiplier_val =rife_multiplier_val ,
            rife_fp16_val =rife_fp16_val ,
            rife_uhd_val =rife_uhd_val ,
            rife_scale_val =rife_scale_val ,
            rife_skip_static_val =rife_skip_static_val ,
            rife_enable_fps_limit_val =rife_enable_fps_limit_val ,
            rife_max_fps_limit_val =rife_max_fps_limit_val ,
            ffmpeg_preset_dropdown_val =ffmpeg_preset_dropdown_val ,
            ffmpeg_quality_slider_val =ffmpeg_quality_slider_val ,
            ffmpeg_use_gpu_check_val =ffmpeg_use_gpu_check_val ,
            seed_num_val =seed_num_val ,
            random_seed_check_val =random_seed_check_val ,
            output_dir =app_config .DEFAULT_OUTPUT_DIR ,
            logger =logger ,
            progress =progress 
            )

        auto_caption_btn .click (
        fn =auto_caption_wrapper ,
        inputs =[input_video ,cogvlm_quant_radio ,cogvlm_unload_radio ],
        outputs =[user_prompt ,caption_status ],
        show_progress_on =[user_prompt ]
        ).then (lambda :gr .update (visible =True ),None ,caption_status )

    rife_fps_button .click (
    fn =rife_fps_increase_wrapper ,
    inputs =[
    input_video ,
    rife_multiplier ,rife_fp16 ,rife_uhd ,rife_scale ,rife_skip_static ,
    rife_enable_fps_limit ,rife_max_fps_limit ,
    ffmpeg_preset_dropdown ,ffmpeg_quality_slider ,ffmpeg_use_gpu_check ,
    seed_num ,random_seed_check 
    ],
    outputs =[output_video ,status_textbox ],
    show_progress_on =[output_video ]
    )

    split_only_button .click (
    fn =wrapper_split_video_only_for_gradio ,
    inputs =[
    input_video ,scene_split_mode_radio ,scene_min_scene_len_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_threshold_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_copy_streams_check ,scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check ,
    scene_manual_split_type_radio ,scene_manual_split_value_num 
    ],
    outputs =[output_video ,status_textbox ],
    show_progress_on =[output_video ]
    )

    def process_batch_videos_wrapper (
    batch_input_folder_val ,batch_output_folder_val ,
    user_prompt_val ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
    upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
    max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
    enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
    enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
    enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
    ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
    save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,
    create_comparison_video_check_val ,

    enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
    scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
    scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
    scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,

    enable_fps_decrease_val =False ,fps_decrease_mode_val ="multiplier",fps_multiplier_preset_val ="1/2x (Half FPS)",fps_multiplier_custom_val =0.5 ,target_fps_val =24.0 ,fps_interpolation_method_val ="drop",

    enable_rife_interpolation_val =False ,rife_multiplier_val =2 ,rife_fp16_val =True ,rife_uhd_val =False ,rife_scale_val =1.0 ,
    rife_skip_static_val =False ,rife_enable_fps_limit_val =False ,rife_max_fps_limit_val =60 ,
    rife_apply_to_chunks_val =True ,rife_apply_to_scenes_val =True ,rife_keep_original_val =True ,rife_overwrite_original_val =False ,

    batch_skip_existing_val =True ,batch_use_prompt_files_val =True ,batch_save_captions_val =True ,
    batch_enable_auto_caption_val =False ,cogvlm_quant_radio_val =None ,cogvlm_unload_radio_val =None ,
    seed_num_val =99 ,random_seed_check_val =False ,
    progress =gr .Progress (track_tqdm =True )
    ):

        actual_seed_for_batch =seed_num_val 
        if random_seed_check_val :
            actual_seed_for_batch =np .random .randint (0 ,2 **31 )
            logger .info (f"Batch: Random seed checked. Using generated seed: {actual_seed_for_batch}")
        elif seed_num_val ==-1 :
            actual_seed_for_batch =np .random .randint (0 ,2 **31 )
            logger .info (f"Batch: Seed input is -1. Using generated seed: {actual_seed_for_batch}")
        else :
            logger .info (f"Batch: Using provided seed: {actual_seed_for_batch}")

        partial_run_upscale_for_batch =partial (core_run_upscale ,
        logger =logger ,
        app_config_module =app_config ,
        metadata_handler_module =metadata_handler ,
        VideoToVideo_sr_class =VideoToVideo_sr ,
        setup_seed_func =setup_seed ,
        EasyDict_class =EasyDict ,
        preprocess_func =preprocess ,
        collate_fn_func =collate_fn ,
        tensor2vid_func =tensor2vid ,
        ImageSpliterTh_class =ImageSpliterTh ,
        adain_color_fix_func =adain_color_fix ,
        wavelet_color_fix_func =wavelet_color_fix ,
        current_seed =actual_seed_for_batch 
        )

        return process_batch_videos (
        batch_input_folder_val ,batch_output_folder_val ,
        user_prompt_val ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
        upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
        max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
        enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
        enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
        enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
        ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
        save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,
        create_comparison_video_check_val ,

        enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
        scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
        scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
        scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,

        enable_fps_decrease_val ,target_fps_val ,fps_interpolation_method_val ,

        enable_rife_interpolation_val ,rife_multiplier_val ,rife_fp16_val ,rife_uhd_val ,rife_scale_val ,
        rife_skip_static_val ,rife_enable_fps_limit_val ,rife_max_fps_limit_val ,
        rife_apply_to_chunks_val ,rife_apply_to_scenes_val ,rife_keep_original_val ,rife_overwrite_original_val ,

        run_upscale_func =partial_run_upscale_for_batch ,
        logger =logger ,

        batch_skip_existing_val =batch_skip_existing_val ,
        batch_use_prompt_files_val =batch_use_prompt_files_val ,
        batch_save_captions_val =batch_save_captions_val ,
        batch_enable_auto_caption_val =batch_enable_auto_caption_val ,
        batch_cogvlm_quant_val =get_quant_value_from_display (cogvlm_quant_radio_val )if batch_enable_auto_caption_val else None ,
        batch_cogvlm_unload_val =cogvlm_unload_radio_val if batch_enable_auto_caption_val else 'full',

        progress =progress 
        )

    batch_process_inputs =[
    batch_input_folder ,batch_output_folder ,
    user_prompt ,pos_prompt ,neg_prompt ,model_selector ,
    upscale_factor_slider ,cfg_slider ,steps_slider ,solver_mode_radio ,
    max_chunk_len_slider ,vae_chunk_slider ,color_fix_dropdown ,
    enable_tiling_check ,tile_size_num ,tile_overlap_num ,
    enable_sliding_window_check ,window_size_num ,window_step_num ,
    enable_target_res_check ,target_h_num ,target_w_num ,target_res_mode_radio ,
    ffmpeg_preset_dropdown ,ffmpeg_quality_slider ,ffmpeg_use_gpu_check ,
    save_frames_checkbox ,save_metadata_checkbox ,save_chunks_checkbox ,
    create_comparison_video_check ,
    enable_scene_split_check ,scene_split_mode_radio ,scene_min_scene_len_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_threshold_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_copy_streams_check ,scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check ,
    scene_manual_split_type_radio ,scene_manual_split_value_num ,

    enable_fps_decrease ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ,fps_interpolation_method ,

    enable_rife_interpolation ,rife_multiplier ,rife_fp16 ,rife_uhd ,rife_scale ,
    rife_skip_static ,rife_enable_fps_limit ,rife_max_fps_limit ,
    rife_apply_to_chunks ,rife_apply_to_scenes ,rife_keep_original ,rife_overwrite_original ,

    batch_skip_existing ,batch_use_prompt_files ,batch_save_captions ,
    batch_enable_auto_caption 
    ]
    
    if app_config .UTIL_COG_VLM_AVAILABLE :
        batch_process_inputs .extend ([cogvlm_quant_radio ,cogvlm_unload_radio ])
    else :
        batch_process_inputs .extend ([gr .State (None ),gr .State (None )])

    batch_process_inputs .extend ([seed_num ,random_seed_check ])

    batch_process_button .click (
    fn =process_batch_videos_wrapper ,
    inputs =batch_process_inputs ,
    outputs =[output_video ,status_textbox ],
    show_progress_on =[output_video ]
    )

    gpu_selector .change (
    fn =lambda gpu_id :util_set_gpu_device (gpu_id ,logger =logger ),
    inputs =gpu_selector ,
    outputs =status_textbox 
    )

    def manual_comparison_wrapper (
    manual_original_video_val ,
    manual_upscaled_video_val ,
    ffmpeg_preset_dropdown_val ,
    ffmpeg_quality_slider_val ,
    ffmpeg_use_gpu_check_val ,
    seed_num_val ,
    random_seed_check_val ,
    progress =gr .Progress (track_tqdm =True )
    ):

        if random_seed_check_val :
            import random 
            current_seed =random .randint (0 ,2 **32 -1 )
        else :
            current_seed =seed_num_val if seed_num_val >=0 else 99 

        if manual_original_video_val is None :
            error_msg ="Please upload an original/reference video"
            return gr .update (value =None ),gr .update (value =error_msg ,visible =True )

        if manual_upscaled_video_val is None :
            error_msg ="Please upload an upscaled/enhanced video"
            return gr .update (value =None ),gr .update (value =error_msg ,visible =True )

        try :

            output_path ,status_message =util_generate_manual_comparison_video (
            original_video_path =manual_original_video_val ,
            upscaled_video_path =manual_upscaled_video_val ,
            ffmpeg_preset =ffmpeg_preset_dropdown_val ,
            ffmpeg_quality =ffmpeg_quality_slider_val ,
            ffmpeg_use_gpu =ffmpeg_use_gpu_check_val ,
            output_dir =app_config .DEFAULT_OUTPUT_DIR ,
            seed_value =current_seed ,
            logger =logger ,
            progress =progress 
            )

            if output_path :

                return gr .update (value =output_path ),gr .update (value =status_message ,visible =True )
            else :

                return gr .update (value =None ),gr .update (value =status_message ,visible =True )

        except Exception as e :
            error_msg =f"Unexpected error during manual comparison: {str(e)}"
            logger .error (error_msg ,exc_info =True )
            return gr .update (value =None ),gr .update (value =error_msg ,visible =True )

    def update_seed_num_interactive (is_random_seed_checked ):
        return gr .update (interactive =not is_random_seed_checked )

    random_seed_check .change (
    fn =update_seed_num_interactive ,
    inputs =random_seed_check ,
    outputs =seed_num 
    )

    manual_comparison_button .click (
    fn =manual_comparison_wrapper ,
    inputs =[
    manual_original_video ,
    manual_upscaled_video ,
    ffmpeg_preset_dropdown ,
    ffmpeg_quality_slider ,
    ffmpeg_use_gpu_check ,
    seed_num ,
    random_seed_check 
    ],
    outputs =[
    comparison_video ,
    manual_comparison_status 
    ],
    show_progress_on =[comparison_video ]
    )

    def update_rife_fps_limit_interactive (enable_fps_limit ):
        return gr .update (interactive =enable_fps_limit )

    def update_rife_controls_interactive (enable_rife ):
        return [gr .update (interactive =enable_rife )]*10 

    rife_enable_fps_limit .change (
    fn =update_rife_fps_limit_interactive ,
    inputs =rife_enable_fps_limit ,
    outputs =rife_max_fps_limit 
    )

    enable_rife_interpolation .change (
    fn =update_rife_controls_interactive ,
    inputs =enable_rife_interpolation ,
    outputs =[
    rife_multiplier ,rife_fp16 ,rife_uhd ,rife_scale ,rife_skip_static ,
    rife_enable_fps_limit ,rife_max_fps_limit ,rife_apply_to_chunks ,
    rife_apply_to_scenes ,rife_keep_original 
    ]
    )

    def update_fps_decrease_controls_interactive (enable_fps_decrease ):

        if enable_fps_decrease :
            return [
            gr .update (interactive =True ),
            gr .update (interactive =True ),
            gr .update (interactive =True ),
            gr .update (interactive =True ),
            gr .update (interactive =True ),
            ]
        else :
            return [
            gr .update (interactive =False ),
            gr .update (interactive =False ),
            gr .update (interactive =False ),
            gr .update (interactive =False ),
            gr .update (interactive =False ),
            ]

    def update_fps_mode_controls (fps_mode ):

        if fps_mode =="multiplier":
            return [
            gr .update (visible =True ),
            gr .update (visible =False ),
            ]
        else :
            return [
            gr .update (visible =False ),
            gr .update (visible =True ),
            ]

    def update_multiplier_preset (preset_choice ):

        multiplier_map ={v :k for k ,v in util_get_common_fps_multipliers ().items ()}

        if preset_choice =="Custom":
            return [
            gr .update (visible =True ),
            0.5 
            ]
        else :
            multiplier_value =multiplier_map .get (preset_choice ,0.5 )
            return [
            gr .update (visible =False ),
            multiplier_value 
            ]

    def calculate_fps_preview (input_video ,fps_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ):

        if input_video is None :
            return "**📊 Calculation:** Upload a video to see FPS reduction preview"

        try :

            input_fps =30.0 

            if fps_mode =="multiplier":

                if fps_multiplier_preset =="Custom":
                    multiplier =fps_multiplier_custom 
                else :
                    multiplier_map ={v :k for k ,v in util_get_common_fps_multipliers ().items ()}
                    multiplier =multiplier_map .get (fps_multiplier_preset ,0.5 )

                calculated_fps =input_fps *multiplier 
                if calculated_fps <1.0 :
                    calculated_fps =1.0 

                return f"**📊 Calculation:** {input_fps:.1f} FPS × {multiplier:.2f} = {calculated_fps:.1f} FPS ({fps_multiplier_preset})"
            else :
                return f"**📊 Calculation:** Fixed mode → {target_fps} FPS"

        except Exception as e :
            return f"**📊 Calculation:** Error calculating preview: {str(e)}"

    enable_fps_decrease .change (
    fn =update_fps_decrease_controls_interactive ,
    inputs =enable_fps_decrease ,
    outputs =[fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ,fps_interpolation_method ]
    )

    fps_decrease_mode .change (
    fn =update_fps_mode_controls ,
    inputs =fps_decrease_mode ,
    outputs =[multiplier_controls ,fixed_controls ]
    )

    fps_multiplier_preset .change (
    fn =update_multiplier_preset ,
    inputs =fps_multiplier_preset ,
    outputs =[fps_multiplier_custom ,fps_multiplier_custom ]
    )

    for component in [input_video ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ]:
        component .change (
        fn =calculate_fps_preview ,
        inputs =[input_video ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ],
        outputs =fps_calculation_info 
        )



if __name__ =="__main__":
    os .makedirs (app_config .DEFAULT_OUTPUT_DIR ,exist_ok =True )
    logger .info (f"Gradio App Starting. Default output to: {os.path.abspath(app_config.DEFAULT_OUTPUT_DIR)}")
    logger .info (f"STAR Models expected at: {app_config.LIGHT_DEG_MODEL_PATH}, {app_config.HEAVY_DEG_MODEL_PATH}")
    if app_config .UTIL_COG_VLM_AVAILABLE :
        logger .info (f"CogVLM2 Model expected at: {app_config.COG_VLM_MODEL_PATH}")

    available_gpus_main =util_get_available_gpus ()
    if available_gpus_main :
        default_gpu_main_val =available_gpus_main [0 ]

        util_set_gpu_device (default_gpu_main_val ,logger =logger )
        logger .info (f"Attempted to initialize with default GPU: {default_gpu_main_val}")
    else :
        logger .info ("No CUDA GPUs detected, attempting to set to 'Auto' (CPU or default).")
        util_set_gpu_device ("Auto",logger =logger )

    effective_allowed_paths =util_get_available_drives (app_config .DEFAULT_OUTPUT_DIR ,base_path ,logger =logger )

    demo .queue ().launch (
    debug =True ,
    max_threads =100 ,
    inbrowser =True ,
    share =args .share ,
    allowed_paths =effective_allowed_paths ,
    prevent_thread_lock =True 
    )
