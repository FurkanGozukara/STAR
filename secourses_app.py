# secourses_app.py

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

from logic.dataclasses import (
    AppConfig, create_app_config,
    UTIL_COG_VLM_AVAILABLE, UTIL_BITSANDBYTES_AVAILABLE,
    get_cogvlm_quant_choices_map, get_default_cogvlm_quant_display,
    PathConfig, PromptConfig, StarModelConfig, PerformanceConfig, ResolutionConfig,
    ContextWindowConfig, TilingConfig, FfmpegConfig, FrameFolderConfig, SceneSplitConfig,
    CogVLMConfig, OutputConfig, SeedConfig, RifeConfig, FpsDecreaseConfig, BatchConfig,
    ImageUpscalerConfig, FaceRestorationConfig, GpuConfig
)
from logic import metadata_handler
from logic import config as app_config_module 
from logic import preset_handler

from logic .cogvlm_utils import (
load_cogvlm_model as util_load_cogvlm_model ,
unload_cogvlm_model as util_unload_cogvlm_model ,
auto_caption as util_auto_caption ,
)

from logic .common_utils import format_time 

from logic .ffmpeg_utils import (
run_ffmpeg_command as util_run_ffmpeg_command ,
extract_frames as util_extract_frames ,
create_video_from_frames as util_create_video_from_frames ,
decrease_fps as util_decrease_fps ,
decrease_fps_with_multiplier as util_decrease_fps_with_multiplier ,
calculate_target_fps_from_multiplier as util_calculate_target_fps_from_multiplier ,
get_common_fps_multipliers as util_get_common_fps_multipliers ,
get_video_info as util_get_video_info ,
format_video_info_message as util_format_video_info_message 
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

from logic.cancellation_manager import cancellation_manager, CancelledError

from logic .image_upscaler_utils import (
scan_for_models as util_scan_for_models,
get_model_info as util_get_model_info
)

from logic .video_editor import (
parse_time_ranges as util_parse_time_ranges,
parse_frame_ranges as util_parse_frame_ranges,
validate_ranges as util_validate_ranges,
get_video_detailed_info as util_get_video_detailed_info,
cut_video_segments as util_cut_video_segments,
create_preview_segment as util_create_preview_segment,
estimate_processing_time as util_estimate_processing_time,
format_video_info_for_display as util_format_video_info_for_display
)

from logic .temp_folder_utils import (
get_temp_folder_path as util_get_temp_folder_path ,
calculate_temp_folder_size as util_calculate_temp_folder_size ,
format_temp_folder_size as util_format_temp_folder_size ,
clear_temp_folder as util_clear_temp_folder
)

from logic .frame_folder_utils import (
validate_frame_folder_input as util_validate_frame_folder_input,
process_frame_folder_to_video as util_process_frame_folder_to_video,
find_frame_folders_in_directory as util_find_frame_folders_in_directory
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

APP_CONFIG = create_app_config(base_path, args.outputs_folder, star_cfg)

# Initialize the config module paths (needed for backward compatibility with core functions)
app_config_module.initialize_paths_and_prompts(base_path, args.outputs_folder, star_cfg)

os .makedirs (APP_CONFIG.paths.outputs_dir ,exist_ok =True )

if not os .path .exists (APP_CONFIG.paths.light_deg_model_path ):
     logger .error (f"FATAL: Light degradation model not found at {APP_CONFIG.paths.light_deg_model_path}.")
if not os .path .exists (APP_CONFIG.paths.heavy_deg_model_path ):
     logger .error (f"FATAL: Heavy degradation model not found at {APP_CONFIG.paths.heavy_deg_model_path}.")

css ="""
.gradio-container { font-family: 'Inter', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important; font-size: 16px !important; }
.gradio-container * { font-family: 'Inter', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important; font-size: 16px !important; }
.gr-textbox, .gr-dropdown, .gr-radio, .gr-checkbox, .gr-slider, .gr-number, .gr-markdown { font-size: 16px !important; }
label, .gr-form > label { font-size: 16px !important; }
.gr-button { color: white; border-color: black; background: black; font-size: 16px !important; }
#row1, #row2, #row3, #row4 {
    margin-bottom: 20px !important;
}
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

def load_initial_preset():
    """Loads the last used preset, or the 'Default' preset, or returns a fresh config."""
    base_config = create_app_config(base_path, args.outputs_folder, star_cfg)
    
    # Always ensure GPU is set to 0 as default
    base_config.gpu.device = "0"
    
    preset_to_load = preset_handler.get_last_used_preset_name()
    if preset_to_load:
        logger.info(f"Found last used preset: '{preset_to_load}'. Attempting to load.")
    else:
        logger.info("No last used preset found. Looking for 'Default' preset.")
        preset_to_load = "Default"

    config_dict, message = preset_handler.load_preset(preset_to_load)
    
    if config_dict:
        logger.info(f"Successfully loaded initial preset '{preset_to_load}'.")
        # Robustly update the base_config with loaded values
        for section_name, section_data in config_dict.items():
            if hasattr(base_config, section_name):
                section_obj = getattr(base_config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        # Special handling for GPU device to ensure it's valid
                        if section_name == 'gpu' and key == 'device':
                            available_gpus = util_get_available_gpus()
                            if available_gpus:
                                try:
                                    gpu_num = int(value) if value != "Auto" else 0
                                    if 0 <= gpu_num < len(available_gpus):
                                        setattr(section_obj, key, str(gpu_num))
                                    else:
                                        logger.warning(f"GPU index {gpu_num} out of range. Defaulting to GPU 0.")
                                        setattr(section_obj, key, "0")
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid GPU device value '{value}'. Defaulting to GPU 0.")
                                    setattr(section_obj, key, "0")
                            else:
                                setattr(section_obj, key, "0")
                        else:
                            setattr(section_obj, key, value)
        return base_config
    else:
        logger.warning(f"Could not load initial preset '{preset_to_load}'. Reason: {message}. Starting with application defaults.")
        return base_config

# Load initial settings from presets or defaults
INITIAL_APP_CONFIG = load_initial_preset()

def get_filtered_preset_list():
    """Get preset list excluding 'last_preset' from dropdown display"""
    all_presets = preset_handler.get_preset_list()
    filtered_presets = [preset for preset in all_presets if preset != "last_preset"]
    logger.debug(f"All presets: {all_presets}, Filtered presets: {filtered_presets}")
    return filtered_presets

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
    APP_CONFIG.paths.outputs_dir ,
    logger ,
    progress =progress 
    )

with gr .Blocks (css =css ,theme =gr .themes .Soft ())as demo :
    gr .Markdown ("# Ultimate SECourses STAR Video Upscaler V100 Pre Release")

    with gr .Tabs ()as main_tabs :
        with gr .Tab ("Main",id ="main_tab"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        input_video =gr .Video (
                        label ="Input Video",
                        sources =["upload"],
                        interactive =True ,height =512 
                        )
                                                
                        integration_status =gr .Textbox (
                        label ="Integration Status",
                        interactive =False ,
                        lines =2 ,
                        visible =False ,
                        value =""
                        )
                        with gr .Row ():
                            user_prompt =gr .Textbox (
                            label ="Describe the Video Content (Prompt) (Useful only for STAR Model)",
                            lines =3 ,
                            placeholder ="e.g., A panda playing guitar by a lake at sunset.",
                            value=INITIAL_APP_CONFIG.prompts.user,
                            info ="""Describe the main subject and action in the video. This guides the upscaling process.
Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens.
If CogVLM2 is available, you can use the button below to generate a caption automatically."""
                            )
                        with gr .Row ():
                            auto_caption_then_upscale_check =gr .Checkbox (label ="Auto-caption then Upscale (Useful only for STAR Model)",value =INITIAL_APP_CONFIG.cogvlm.auto_caption_then_upscale ,info ="If checked, clicking 'Upscale Video' will first generate a caption and use it as the prompt.")

                            available_gpus =util_get_available_gpus ()
                            gpu_choices = available_gpus if available_gpus else ["No CUDA GPUs detected"]
                            # Always default to GPU 0 (first GPU in the list)
                            default_gpu = available_gpus[0] if available_gpus else "No CUDA GPUs detected"

                            gpu_selector =gr .Dropdown (
                            label ="GPU Selection",
                            choices =gpu_choices ,
                            value =default_gpu ,
                            info ="Select which GPU to use for processing. Defaults to GPU 0.",
                            scale =1 
                            )

                        if UTIL_COG_VLM_AVAILABLE:
                            with gr.Row(elem_id="row1"):
                                auto_caption_btn = gr.Button("Generate Caption with CogVLM2 (No Upscale)", variant="primary", icon="icons/caption.png")
                                rife_fps_button = gr.Button("RIFE FPS Increase (No Upscale)", variant="primary", icon="icons/fps.png")
    
                            with gr.Row(elem_id="row2"):
                                upscale_button = gr.Button("Upscale Video", variant="primary", icon="icons/upscale.png")
    
                            with gr.Row(elem_id="row3"):
                                cancel_button = gr.Button("Cancel Upscaling", variant="stop", visible=True, interactive=False, icon="icons/cancel.png")
    
                            with gr.Row(elem_id="row4"):
                                initial_temp_size_label = util_format_temp_folder_size(logger)
                                delete_temp_button = gr.Button(f"Delete Temp Folder ({initial_temp_size_label})", variant="stop")


                            caption_status =gr .Textbox (label ="Captioning Status",interactive =False ,visible =False )
                        else:
                            with gr .Row ():
                                rife_fps_button =gr .Button ("RIFE FPS Increase (No Upscale)",variant ="primary",icon ="icons/fps.png")
                            with gr .Row ():
                                upscale_button =gr .Button ("Upscale Video",variant ="primary",icon ="icons/upscale.png")
                                cancel_button =gr .Button ("Cancel",variant ="stop",visible =True, interactive=False)

                            with gr .Row ():
                                initial_temp_size_label = util_format_temp_folder_size(logger)
                                delete_temp_button = gr.Button(f"Delete Temp Folder ({initial_temp_size_label})", variant="stop")

                            caption_status =gr .Textbox (label ="Captioning Status",interactive =False ,visible =False )

                    with gr .Accordion ("Prompt Settings (Useful only for STAR Model)",open =True ):
                        pos_prompt =gr .Textbox (
                        label ="Default Positive Prompt (Appended)",
                        value =INITIAL_APP_CONFIG.prompts.positive ,
                        lines =2 ,
                        info ="""Appended to your 'Describe Video Content' prompt. Focuses on desired quality aspects (e.g., realism, detail).
The total combined prompt length is limited to 77 tokens."""
                        )
                        neg_prompt =gr .Textbox (
                        label ="Default Negative Prompt (Appended)",
                        value =INITIAL_APP_CONFIG.prompts.negative ,
                        lines =2 ,
                        info ="Guides the model *away* from undesired aspects (e.g., bad quality, artifacts, specific styles). This does NOT count towards the 77 token limit for positive guidance."
                        )
                    with gr .Group ():
                        gr.Markdown("### 📁 Enhanced Input: Video Files & Frame Folders")
                        gr.Markdown("*Auto-detects whether your input is a single video file or a folder containing frame sequences*")
                        
                        input_frames_folder =gr .Textbox (
                        label ="Input Video or Frames Folder Path",
                        placeholder ="C:/path/to/video.mp4 or C:/path/to/frames/folder/",
                        interactive =True ,
                        info ="Enter path to either a video file (mp4, avi, mov, etc.) or folder containing image frames (jpg, png, tiff, etc.). Automatically detected - works on Windows and Linux."
                        )
                        frames_folder_status =gr .Textbox (
                        label ="Input Path Status",
                        interactive =False ,
                        lines =3 ,
                        visible =True ,
                        value ="Enter a video file path or frames folder path above to validate"
                        )
                    open_output_folder_button =gr .Button ("Open Outputs Folder",icon ="icons/folder.png",variant ="primary")

                with gr .Column (scale =1 ):
                    output_video =gr .Video (label ="Upscaled Video",interactive =False ,height =512 )
                    status_textbox =gr .Textbox (label ="Log",interactive =False ,lines =8 ,max_lines =15 )
                    
                    with gr.Accordion("Save/Load Presets", open=True):
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Select or Create Preset",
                                choices=get_filtered_preset_list(),
                                value=preset_handler.get_last_used_preset_name() or "Default",
                                allow_custom_value=True,
                                scale=3,
                                info="Select a preset to auto-load, or type a new name and click Save."
                            )
                            refresh_presets_btn = gr.Button("🔄", scale=1, variant="secondary")
                            save_preset_btn = gr.Button("Save", variant="primary", scale=1)
                        preset_status = gr.Textbox(label="Preset Status", show_label=False, interactive=False, lines=1, placeholder="...")

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

        with gr .Tab ("Resolution & Scene Split",id ="resolution_tab"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("Target Resolution - Maintains Your Input Video Aspect Ratio",open =True ):
                        gr .Markdown ("📝 **Image Upscaler Support:** Target resolution now works with image upscalers! The system automatically adapts based on your selected model's scale factor (2x, 4x, etc.).")
                        enable_target_res_check =gr .Checkbox (
                        label ="Enable Max Target Resolution",
                        value =INITIAL_APP_CONFIG.resolution.enable_target_res ,
                        info ="Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
                        )
                        target_res_mode_radio =gr .Radio (
                        label ="Target Resolution Mode",
                        choices =['Ratio Upscale','Downscale then 4x'],value =INITIAL_APP_CONFIG.resolution.target_res_mode ,
                        info ="""How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio.
'Downscale then 4x': For STAR models, downscales towards Target H/W ÷ 4, then applies 4x upscale. For image upscalers, adapts to model scale (e.g., 2x model = Target H/W ÷ 2). Can clean noisy high-res input before upscaling."""
                        )
                        with gr .Row ():
                            target_h_num =gr .Slider (
                            label ="Max Target Height (px)",
                            value =INITIAL_APP_CONFIG.resolution.target_h ,minimum =128 ,maximum =4096 ,step =16 ,
                            info ="""Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                            )
                            target_w_num =gr .Slider (
                            label ="Max Target Width (px)",
                            value =INITIAL_APP_CONFIG.resolution.target_w ,minimum =128 ,maximum =4096 ,step =16 ,
                            info ="""Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                            )
                        
                        # Auto-Resolution (Aspect Ratio Aware) Feature
                        gr.Markdown("---")
                        gr.Markdown("### 🎯 Auto-Resolution (Aspect Ratio Aware)")
                        
                        enable_auto_aspect_resolution_check = gr.Checkbox(
                            label="Enable Auto Aspect Resolution",
                            value=INITIAL_APP_CONFIG.resolution.enable_auto_aspect_resolution,
                            info="""Automatically calculate optimal resolution that maintains input video aspect ratio within the pixel budget (Target H × Target W).
- Triggered whenever you change video or process batch videos
- Maintains exact aspect ratio while maximizing quality within pixel limits
- Prevents manual resolution adjustment for different aspect ratios
- Example: 1024×1024 budget + 360×640 input → auto-sets to 720×1280 (maintains 9:16 ratio, uses 921,600 pixels)"""
                        )
                        
                        auto_resolution_status_display = gr.Textbox(
                            label="Auto-Resolution Status",
                            value=INITIAL_APP_CONFIG.resolution.auto_resolution_status,
                            interactive=False,
                            lines=3,
                            info="Shows current auto-calculated resolution and aspect ratio information"
                        )
                with gr .Column (scale =1 ):
                    split_only_button =gr .Button ("Split Video Only (No Upscaling)",icon ="icons/split.png",variant ="primary")
                    with gr .Accordion ("Scene Splitting",open =True ):
                        enable_scene_split_check =gr .Checkbox (
                        label ="Enable Scene Splitting",
                        value =INITIAL_APP_CONFIG.scene_split.enable ,
                        info ="""Split video into scenes and process each scene individually. This can improve quality and speed by processing similar content together.
- Quality Impact: Better temporal consistency within scenes, improved auto-captioning per scene.
- Speed Impact: Can be faster for long videos with distinct scenes.
- Memory Impact: Reduces peak memory usage by processing smaller segments."""
                        )
                        with gr .Row ():
                            scene_split_mode_radio =gr .Radio (
                            label ="Split Mode",
                            choices =['automatic','manual'],value =INITIAL_APP_CONFIG.scene_split.mode ,
                            info ="""'automatic': Uses scene detection algorithms to find natural scene boundaries.
'manual': Splits video at fixed intervals (duration or frame count)."""
                            )
                        with gr .Group ():
                            gr .Markdown ("**Automatic Scene Detection Settings**")
                            with gr .Row ():
                                scene_min_scene_len_num =gr .Number (label ="Min Scene Length (seconds)",value =INITIAL_APP_CONFIG.scene_split.min_scene_len ,minimum =0.1 ,step =0.1 ,info ="Minimum duration for a scene. Shorter scenes will be merged or dropped.")
                                scene_threshold_num =gr .Number (label ="Detection Threshold",value =INITIAL_APP_CONFIG.scene_split.threshold ,minimum =0.1 ,maximum =10.0 ,step =0.1 ,info ="Sensitivity of scene detection. Lower values detect more scenes.")
                            with gr .Row ():
                                scene_drop_short_check =gr .Checkbox (label ="Drop Short Scenes",value =INITIAL_APP_CONFIG.scene_split.drop_short ,info ="If enabled, scenes shorter than minimum length are dropped instead of merged.")
                                scene_merge_last_check =gr .Checkbox (label ="Merge Last Scene",value =INITIAL_APP_CONFIG.scene_split.merge_last ,info ="If the last scene is too short, merge it with the previous scene.")
                            with gr .Row ():
                                scene_frame_skip_num =gr .Number (label ="Frame Skip",value =INITIAL_APP_CONFIG.scene_split.frame_skip ,minimum =0 ,step =1 ,info ="Skip frames during detection to speed up processing. 0 = analyze every frame.")
                                scene_min_content_val_num =gr .Number (label ="Min Content Value",value =INITIAL_APP_CONFIG.scene_split.min_content_val ,minimum =0.0 ,step =1.0 ,info ="Minimum content change required to detect a scene boundary.")
                                scene_frame_window_num =gr .Number (label ="Frame Window",value =INITIAL_APP_CONFIG.scene_split.frame_window ,minimum =1 ,step =1 ,info ="Number of frames to analyze for scene detection.")
                        with gr .Group ():
                            gr .Markdown ("**Manual Split Settings**")
                            with gr .Row ():
                                scene_manual_split_type_radio =gr .Radio (label ="Manual Split Type",choices =['duration','frame_count'],value =INITIAL_APP_CONFIG.scene_split.manual_split_type ,info ="'duration': Split every N seconds.\n'frame_count': Split every N frames.")
                                scene_manual_split_value_num =gr .Number (label ="Split Value",value =INITIAL_APP_CONFIG.scene_split.manual_split_value ,minimum =1.0 ,step =1.0 ,info ="Duration in seconds or number of frames for manual splitting.")
                        with gr .Group ():
                            gr .Markdown ("**Encoding Settings (for scene segments)**")
                            with gr .Row ():
                                scene_copy_streams_check =gr .Checkbox (label ="Copy Streams",value =INITIAL_APP_CONFIG.scene_split.copy_streams ,info ="Copy video/audio streams without re-encoding during scene splitting (faster) but can generate inaccurate splits.")
                                scene_use_mkvmerge_check =gr .Checkbox (label ="Use MKVMerge",value =INITIAL_APP_CONFIG.scene_split.use_mkvmerge ,info ="Use mkvmerge instead of ffmpeg for splitting (if available).")
                            with gr .Row ():
                                scene_rate_factor_num =gr .Number (label ="Rate Factor (CRF)",value =INITIAL_APP_CONFIG.scene_split.rate_factor ,minimum =0 ,maximum =51 ,step =1 ,info ="Quality setting for re-encoding (lower = better quality). Only used if Copy Streams is disabled.")
                                scene_preset_dropdown =gr .Dropdown (label ="Encoding Preset",choices =['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],value =INITIAL_APP_CONFIG.scene_split.encoding_preset ,info ="Encoding speed vs quality trade-off. Only used if Copy Streams is disabled.")
                            scene_quiet_ffmpeg_check =gr .Checkbox (label ="Quiet FFmpeg",value =INITIAL_APP_CONFIG.scene_split.quiet_ffmpeg ,info ="Suppress ffmpeg output during scene splitting.")

        with gr .Tab ("Core Settings",id ="core_tab"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        gr .Markdown ("### Core Upscaling Settings for STAR Model - Temporal Upscaling")
                        model_selector =gr .Dropdown (
                        label ="STAR Model - Temporal Upscaling",
                        choices =["Light Degradation","Heavy Degradation"],
                        value =INITIAL_APP_CONFIG.star_model.model_choice ,
                        info ="""Choose the model based on input video quality.
'Light Degradation': Better for relatively clean inputs (e.g., downloaded web videos).
'Heavy Degradation': Better for inputs with significant compression artifacts, noise, or blur."""
                        )
                        upscale_factor_slider =gr .Slider (
                        label ="Upscale Factor (if Target Res disabled)",
                        minimum =1.0 ,maximum =8.0 ,value =INITIAL_APP_CONFIG.resolution.upscale_factor ,step =0.1 ,
                        info ="Simple multiplication factor for output resolution if 'Enable Max Target Resolution' is OFF. E.g., 4.0 means 4x height and 4x width."
                        )
                        cfg_slider =gr .Slider (
                        label ="Guidance Scale (CFG)",
                        minimum =1.0 ,maximum =15.0 ,value =INITIAL_APP_CONFIG.star_model.cfg_scale ,step =0.5 ,
                        info ="Controls how strongly the model follows your combined text prompt. Higher values mean stricter adherence, lower values allow more creativity. Typical values: 5.0-10.0."
                        )
                        with gr .Row ():
                            solver_mode_radio =gr .Radio (
                            label ="Solver Mode",
                            choices =['fast','normal'],value =INITIAL_APP_CONFIG.star_model.solver_mode ,
                            info ="""Diffusion solver type.
'fast': Fewer steps (default ~15), much faster, good quality usually.
'normal': More steps (default ~50), slower, potentially slightly better detail/coherence."""
                            )
                            steps_slider =gr .Slider (
                            label ="Diffusion Steps",
                            minimum =5 ,maximum =100 ,value =INITIAL_APP_CONFIG.star_model.steps ,step =1 ,
                            info ="Number of denoising steps. 'Fast' mode uses a fixed ~15 steps. 'Normal' mode uses the value set here.",
                            interactive =False 
                            )
                        color_fix_dropdown =gr .Dropdown (
                        label ="Color Correction",
                        choices =['AdaIN','Wavelet','None'],value =INITIAL_APP_CONFIG.star_model.color_fix_method ,
                        info ="""Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""
                        )
                    
                    with gr .Accordion ("QUALITY: Performance & VRAM Optimization (Only for STAR Model)",open =True ):
                        max_chunk_len_slider =gr .Slider (
                        label ="Max Frames per Chunk (VRAM) - Bigger = Better QUALITY",
                        minimum =1 ,maximum =192 ,value =INITIAL_APP_CONFIG.performance.max_chunk_len ,step =1 ,
                        info ="""IMPORTANT for VRAM. This is the standard way the application manages VRAM. It divides the entire sequence of video frames into sequential, non-overlapping chunks.
- Mechanism: The STAR model processes one complete chunk (of this many frames) at a time.
- VRAM Impact: High Reduction (Lower value = Less VRAM).
- Bigger Chunk + Bigger VRAM = Faster Processing and Better Quality 
- Quality Impact: Can reduce Temporal Consistency (flicker/motion issues) between chunks if too low, as the model doesn't have context across chunk boundaries. Keep as high as VRAM allows.
- Speed Impact: Slower (Lower value = Slower, as more chunks are processed)."""
                        )
                        enable_chunk_optimization_check =gr .Checkbox (
                        label ="Optimize Last Chunk Quality",
                        value =INITIAL_APP_CONFIG.performance.enable_chunk_optimization ,
                        info ="""Process extra frames for small last chunks to improve quality. When the last chunk has fewer frames than target size (causing quality drops), this processes additional frames but only keeps the necessary output.
- Example: For 70 frames with 32-frame chunks, instead of processing only 6 frames for the last chunk (poor quality), it processes 23 frames (48-70) but keeps only the last 6 (65-70).
- Quality Impact: Significantly improves quality for small last chunks.
- Speed Impact: Minimal impact on total processing time.
- VRAM Impact: No additional VRAM usage."""
                        )
                        vae_chunk_slider =gr .Slider (
                        label ="VAE Decode Chunk (VRAM)",
                        minimum =1 ,maximum =16 ,value =INITIAL_APP_CONFIG.performance.vae_chunk ,step =1 ,
                        info ="""Controls max latent frames decoded back to pixels by VAE simultaneously.
- VRAM Impact: High Reduction (Lower value = Less VRAM during decode stage).
- Quality Impact: Minimal / Negligible. Safe to lower.
- Speed Impact: Slower (Lower value = Slower decoding)."""
                        )

                with gr .Column (scale =1 ):
                    with gr .Group ():
                        gr .Markdown ("### Image-Based Upscaler (Alternative to STAR) - Spatial Upscaling")
                        enable_image_upscaler_check =gr .Checkbox (
                        label ="Enable Image-Based Upscaling (Disables STAR Model)",
                        value =INITIAL_APP_CONFIG.image_upscaler.enable ,
                        info ="""Use deterministic image upscaler models instead of STAR. When enabled:
- Processes frames individually using spandrel-compatible models
- Ignores prompts, auto-caption, context window, and tiling settings
- Supports various architectures: DAT-2, ESRGAN, HAT, RCAN, OmniSR, CUGAN
- Much faster processing with batch support
- Uses way lesser VRAM
- Lower quality compared to high Max Frames per Chunk STAR model upscale"""
                        )
                        
                        try:
                            available_model_files = util_scan_for_models(APP_CONFIG.paths.upscale_models_dir, logger)
                            if available_model_files:
                                model_choices = available_model_files
                                default_model_choice = INITIAL_APP_CONFIG.image_upscaler.model if INITIAL_APP_CONFIG.image_upscaler.model in model_choices else model_choices[0]
                            else:
                                model_choices = ["No models found - place models in upscale_models/"]
                                default_model_choice = model_choices[0]
                        except Exception as e:
                            logger.warning(f"Failed to scan for upscaler models: {e}")
                            model_choices = ["Error scanning models - check upscale_models/ directory"]
                            default_model_choice = model_choices[0]
                        
                        image_upscaler_model_dropdown =gr .Dropdown (
                        label ="Select Upscaler Model - Spatial Upscaling",
                        choices =model_choices ,
                        value =default_model_choice ,
                        info ="Select the image upscaler model. Models should be placed in the 'upscale_models/' directory.",
                        interactive =True 
                        )
                        
                        image_upscaler_batch_size_slider =gr .Slider (
                        label ="Batch Size",
                        minimum =1 ,
                        maximum =1000 ,
                        value =INITIAL_APP_CONFIG.image_upscaler.batch_size ,
                        step =1 ,
                        info ="Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage. Adjust based on your GPU memory.",
                        interactive =True 
                        )
                    
                    with gr .Group ():
                        gr .Markdown ("### Face Restoration (CodeFormer)")
                        enable_face_restoration_check =gr .Checkbox (
                        label ="Enable Face Restoration",
                        value =INITIAL_APP_CONFIG.face_restoration.enable ,
                        info ="""Enhance faces in the video using CodeFormer. Works with both STAR and image-based upscaling.
- Detects and restores faces automatically
- Can be applied before or after upscaling
- Supports both face restoration and colorization
- Requires CodeFormer models in pretrained_weight/ directory"""
                        )
                        
                        face_restoration_fidelity_slider =gr .Slider (
                        label ="Fidelity Weight",
                        minimum =0.0 ,
                        maximum =1.0 ,
                        value =INITIAL_APP_CONFIG.face_restoration.fidelity_weight ,
                        step =0.1 ,
                        info ="""Balance between quality and identity preservation:
- 0.0-0.3: Prioritize quality/detail (may change facial features)
- 0.4-0.6: Balanced approach
- 0.7-1.0: Prioritize identity preservation (may reduce enhancement)""",
                        interactive =True 
                        )
                        
                        with gr .Row ():
                            enable_face_colorization_check =gr .Checkbox (
                            label ="Enable Colorization",
                            value =INITIAL_APP_CONFIG.face_restoration.enable_colorization ,
                            info ="Apply colorization to grayscale faces (experimental feature)",
                            interactive =True 
                            )
                            
                            face_restoration_when_radio =gr .Radio (
                            label ="Apply Timing",
                            choices =['before','after'],
                            value =INITIAL_APP_CONFIG.face_restoration.when ,
                            info ="""When to apply face restoration:
'before': Apply before upscaling (may be enhanced further)
'after': Apply after upscaling (final enhancement)""",
                            interactive =True 
                            )
                        
                        with gr .Row ():
                            with gr .Row ():
                                model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                                default_model_choice = "Auto (Default)"
                                if INITIAL_APP_CONFIG.face_restoration.model and "codeformer.pth" in INITIAL_APP_CONFIG.face_restoration.model:
                                    default_model_choice = "codeformer.pth (359.2MB)"

                                codeformer_model_dropdown =gr .Dropdown (
                                label ="CodeFormer Model",
                                choices =model_choices ,
                                value =default_model_choice ,
                                info ="Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory.",
                                interactive =True 
                                )
                        
                        face_restoration_batch_size_slider =gr .Slider (
                        label ="Face Restoration Batch Size",
                        minimum =1 ,
                        maximum =32 ,
                        value =INITIAL_APP_CONFIG.face_restoration.batch_size ,
                        step =1 ,
                        info ="Number of frames to process simultaneously for face restoration. Higher values = faster processing but more VRAM usage.",
                        interactive =True 
                        )

                    if UTIL_COG_VLM_AVAILABLE :
                        with gr .Accordion ("Auto-Captioning Settings (CogVLM2) (Only for STAR Model)",open =True ):
                            cogvlm_quant_choices_map =get_cogvlm_quant_choices_map (torch .cuda .is_available (),UTIL_BITSANDBYTES_AVAILABLE )
                            cogvlm_quant_radio_choices_display =list (cogvlm_quant_choices_map .values ())
                            default_quant_display_val =get_default_cogvlm_quant_display (cogvlm_quant_choices_map )

                            with gr .Row ():
                                cogvlm_quant_radio =gr .Radio (
                                label ="CogVLM2 Quantization",
                                choices =cogvlm_quant_radio_choices_display ,
                                value =INITIAL_APP_CONFIG.cogvlm.quant_display ,
                                info ="Quantization for the CogVLM2 captioning model (uses less VRAM). INT4/8 require CUDA & bitsandbytes.",
                                interactive =True if len (cogvlm_quant_radio_choices_display )>1 else False 
                                )
                                cogvlm_unload_radio =gr .Radio (
                                label ="CogVLM2 After-Use",
                                choices =['full','cpu'],value =INITIAL_APP_CONFIG.cogvlm.unload_after_use ,
                                info ="""Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""
                                )
                    else :
                        gr .Markdown ("_(Auto-captioning disabled as CogVLM2 components are not fully available.)_")

                    with gr .Accordion ("Context Window - Previous Frames for Better Consistency (Only for STAR Model)",open =True ):
                        enable_context_window_check =gr .Checkbox (
                        label ="Enable Context Window",
                        value =INITIAL_APP_CONFIG.context_window.enable ,
                        info ="""Include previous frames as context when processing each chunk (except the first). Similar to "Optimize Last Chunk Quality" but applied to all chunks.
- Mechanism: Each chunk (except first) includes N previous frames as context, but only outputs new frames. Provides temporal consistency without complex overlap logic.
- Quality Impact: Better temporal consistency and reduced flickering between chunks. More context = better consistency.
- VRAM Impact: Medium increase due to processing context frames (recommend 25-50% of Max Frames per Chunk).
- Speed Impact: Slower due to processing additional context frames, but simpler and more predictable than traditional sliding window."""
                        )
                        context_overlap_num =gr .Slider (
                        label ="Context Overlap (frames)",
                        value =INITIAL_APP_CONFIG.context_window.overlap ,minimum =0 ,maximum =31 ,step =1 ,
                        info ="Number of previous frames to include as context for each chunk (except first). 0 = disabled (same as normal chunking). Higher values = better consistency but more VRAM and slower processing. Recommend: 25-50% of Max Frames per Chunk."
                        )

                    with gr .Accordion ("Advanced: Tiling (Very High Res / Low VRAM)",open =True, visible=False ):
                        enable_tiling_check =gr .Checkbox (
                        label ="Enable Tiled Upscaling",
                        value =INITIAL_APP_CONFIG.tiling.enable ,
                        info ="""Processes each frame in small spatial patches (tiles). Use ONLY if necessary for extreme resolutions or very low VRAM.
- VRAM Impact: Very High Reduction.
- Quality Impact: High risk of tile seams/artifacts. Can harm global coherence and severely reduce temporal consistency.
- Speed Impact: Extremely Slow."""
                        )
                        with gr .Row ():
                            tile_size_num =gr .Number (
                            label ="Tile Size (px, input res)",
                            value =INITIAL_APP_CONFIG.tiling.tile_size ,minimum =64 ,step =32 ,
                            info ="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                            )
                            tile_overlap_num =gr .Number (
                            label ="Tile Overlap (px, input res)",
                            value =INITIAL_APP_CONFIG.tiling.tile_overlap ,minimum =0 ,step =16 ,
                            info ="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                            )

        with gr .Tab ("Output & Comparison",id ="output_tab"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("FFmpeg Encoding Settings",open =True ):
                        ffmpeg_use_gpu_check =gr .Checkbox (
                        label ="Use NVIDIA GPU for FFmpeg (h264_nvenc)",
                        value =INITIAL_APP_CONFIG.ffmpeg.use_gpu ,
                        info ="If checked, uses NVIDIA's NVENC for FFmpeg video encoding (downscaling and final video creation). Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."
                        )
                        with gr .Row ():
                            ffmpeg_preset_dropdown =gr .Dropdown (
                            label ="FFmpeg Preset",
                            choices =['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],
                            value =INITIAL_APP_CONFIG.ffmpeg.preset ,
                            info ="Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently (e.g. p1-p7 or specific names like 'slow', 'medium', 'fast')."
                            )
                            ffmpeg_quality_slider =gr .Slider (
                            label ="FFmpeg Quality (CRF for libx264 / CQ for NVENC)",
                            minimum =0 ,maximum =51 ,value =INITIAL_APP_CONFIG.ffmpeg.quality ,step =1 ,
                            info ="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28."
                            )
                        
                        frame_folder_fps_slider =gr .Slider (
                        label ="Frame Folder FPS",
                        minimum =1.0 ,maximum =120.0 ,value =INITIAL_APP_CONFIG.frame_folder.fps ,step =0.001 ,
                        info ="FPS to use when converting frame folders to videos. This setting only applies when processing input frame folders (not regular videos). Common values: 23.976, 24, 25, 29.97, 30, 60."
                        )

                with gr .Column (scale =1 ):
                    with gr .Accordion ("Output Options",open =True ):
                        create_comparison_video_check =gr .Checkbox (
                        label ="Generate Comparison Video",
                        value =INITIAL_APP_CONFIG.outputs.create_comparison_video ,
                        info ="""Create a side-by-side or top-bottom comparison video showing original vs upscaled quality.
The layout is automatically chosen based on aspect ratio to stay within 1920x1080 bounds when possible.
This helps visualize the quality improvement from upscaling."""
                        )
                        save_frames_checkbox =gr .Checkbox (label ="Save Input and Processed Frames",value =INITIAL_APP_CONFIG.outputs.save_frames ,info ="If checked, saves the extracted input frames and the upscaled output frames into a subfolder named after the output video (e.g., '0001/input_frames' and '0001/processed_frames').")
                        save_metadata_checkbox =gr .Checkbox (label ="Save Processing Metadata",value =INITIAL_APP_CONFIG.outputs.save_metadata ,info ="If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time.")
                        save_chunks_checkbox =gr .Checkbox (label ="Save Processed Chunks",value =INITIAL_APP_CONFIG.outputs.save_chunks ,info ="If checked, saves each processed chunk as a video file in a 'chunks' subfolder (e.g., '0001/chunks/chunk_0001.mp4'). Uses the same FFmpeg settings as the final video.")
                        save_chunk_frames_checkbox =gr .Checkbox (label ="Save Chunk Input Frames (Debug)",value =INITIAL_APP_CONFIG.outputs.save_chunk_frames ,info ="If checked, saves the input frames for each chunk before processing into a 'chunk_frames' subfolder (e.g., '0001/chunk_frames/chunk_01_frame_012.png'). Useful for debugging which frames are processed in each chunk.")

                    with gr .Accordion ("Advanced: Seeding (Reproducibility)",open =True ):
                        with gr .Row ():
                            seed_num =gr .Number (
                            label ="Seed",
                            value =INITIAL_APP_CONFIG.seed.seed ,
                            minimum =-1 ,
                            maximum =2 **32 -1 ,
                            step =1 ,
                            info ="Seed for random number generation. Used for reproducibility. Set to -1 or check 'Random Seed' for a random seed. Value is ignored if 'Random Seed' is checked.",
                            interactive =not INITIAL_APP_CONFIG.seed.use_random 
                            )
                            random_seed_check =gr .Checkbox (
                            label ="Random Seed",
                            value =INITIAL_APP_CONFIG.seed.use_random ,
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
                        interactive =True ,
                        visible =False 
                        )

            with gr .Accordion ("Comparison Video To See Difference",open =True ):
                comparison_video =gr .Video (label ="Comparison Video",interactive =False ,height =512 )

        with gr .Tab ("Batch Upscaling",id ="batch_tab"):

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
                    enable_batch_frame_folders =gr .Checkbox (
                    label ="Process Frame Folders in Batch",
                    value =False ,
                    info ="Enable to process subfolders containing frame sequences instead of video files. Each subfolder with images will be converted to video first."
                    )

                with gr .Row ():

                    batch_skip_existing =gr .Checkbox (
                    label ="Skip Existing Outputs",
                    value =INITIAL_APP_CONFIG.batch.skip_existing ,
                    info ="Skip processing if the output file already exists. Useful for resuming interrupted batch jobs."
                    )

                    batch_use_prompt_files =gr .Checkbox (
                    label ="Use Prompt Files (filename.txt)",
                    value =INITIAL_APP_CONFIG.batch.use_prompt_files ,
                    info ="Look for text files with same name as video (e.g., video.txt) to use as custom prompts. Takes priority over user prompt and auto-caption."
                    )

                    batch_save_captions =gr .Checkbox (
                    label ="Save Auto-Generated Captions",
                    value =INITIAL_APP_CONFIG.batch.save_captions ,
                    info ="Save auto-generated captions as filename.txt in the input folder for future reuse. Never overwrites existing prompt files."
                    )

                if UTIL_COG_VLM_AVAILABLE :
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

            create_batch_processing_help()

        with gr .Tab ("FPS Increase - Decrease",id ="fps_tab"):
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Accordion ("RIFE Interpolation Settings",open =True ):
                        gr .Markdown ("### Frame Interpolation (RIFE)")
                        gr .Markdown ("**RIFE (Real-time Intermediate Flow Estimation)** uses AI to intelligently generate intermediate frames between existing frames, increasing video smoothness and frame rate.")

                        enable_rife_interpolation =gr .Checkbox (
                        label ="Enable RIFE Interpolation",
                        value =INITIAL_APP_CONFIG.rife.enable ,
                        info ="Enable AI-powered frame interpolation to increase video FPS. When enabled, RIFE will be applied to the final upscaled video and can optionally be applied to intermediate chunks and scenes."
                        )

                        with gr .Row ():
                            rife_multiplier =gr .Radio (
                            label ="FPS Multiplier",
                            choices =[2 ,4 ],
                            value =INITIAL_APP_CONFIG.rife.multiplier ,
                            info ="Choose how much to increase the frame rate. 2x doubles FPS (e.g., 24→48), 4x quadruples FPS (e.g., 24→96)."
                            )

                        with gr .Row ():
                            rife_fp16 =gr .Checkbox (
                            label ="Use FP16 Precision",
                            value =INITIAL_APP_CONFIG.rife.fp16 ,
                            info ="Use half-precision floating point for faster processing and lower VRAM usage. Recommended for most users."
                            )
                            rife_uhd =gr .Checkbox (
                            label ="UHD Mode",
                            value =INITIAL_APP_CONFIG.rife.uhd ,
                            info ="Enable UHD mode for 4K+ videos. May improve quality for very high resolution content but requires more VRAM."
                            )

                        rife_scale =gr .Slider (
                        label ="Scale Factor",
                        minimum =0.25 ,maximum =2.0 ,value =INITIAL_APP_CONFIG.rife.scale ,step =0.25 ,
                        info ="Scale factor for RIFE processing. 1.0 = original size. Lower values use less VRAM but may reduce quality. Higher values may improve quality but use more VRAM."
                        )

                        rife_skip_static =gr .Checkbox (
                        label ="Skip Static Frames",
                        value =INITIAL_APP_CONFIG.rife.skip_static ,
                        info ="Automatically detect and skip interpolating static (non-moving) frames to save processing time and avoid unnecessary interpolation."
                        )

                    with gr .Accordion ("Intermediate Processing",open =True ):
                        gr .Markdown ("**Apply RIFE to intermediate videos (recommended)**")
                        rife_apply_to_chunks =gr .Checkbox (
                        label ="Apply to Chunks",
                        value =INITIAL_APP_CONFIG.rife.apply_to_chunks ,
                        info ="Apply RIFE interpolation to individual video chunks during processing. Enabled by default for smoother intermediate results."
                        )
                        rife_apply_to_scenes =gr .Checkbox (
                        label ="Apply to Scenes",
                        value =INITIAL_APP_CONFIG.rife.apply_to_scenes ,
                        info ="Apply RIFE interpolation to individual scene videos when scene splitting is enabled. Enabled by default for consistent results."
                        )

                        gr .Markdown ("**Note:** When RIFE is enabled, the system will return RIFE-interpolated versions to the interface instead of originals, ensuring you get the smoothest possible results throughout the processing pipeline.")

                with gr .Column (scale =1 ):
                    with gr .Accordion ("FPS Decrease",open =True ):
                        gr .Markdown ("### Pre-Processing FPS Reduction")
                        gr .Markdown ("**Reduce FPS before upscaling** to speed up processing and reduce VRAM usage. You can then use RIFE interpolation to restore smooth motion afterward.")

                        enable_fps_decrease =gr .Checkbox (
                        label ="Enable FPS Decrease",
                        value =INITIAL_APP_CONFIG.fps_decrease.enable ,
                        info ="Reduce video FPS before upscaling to speed up processing. Fewer frames = faster upscaling and lower VRAM usage."
                        )

                        fps_decrease_mode =gr .Radio (
                        label ="FPS Reduction Mode",
                        choices =["multiplier","fixed"],
                        value =INITIAL_APP_CONFIG.fps_decrease.mode ,
                        info ="Multiplier: Reduce by fraction (1/2x, 1/4x). Fixed: Set specific FPS value. Multiplier is recommended for automatic adaptation to input video."
                        )

                        with gr .Group ()as multiplier_controls :
                            with gr .Row ():
                                fps_multiplier_preset =gr .Dropdown (
                                label ="FPS Multiplier",
                                choices =list (util_get_common_fps_multipliers ().values ())+["Custom"],
                                value =INITIAL_APP_CONFIG.fps_decrease.multiplier_preset,
                                info ="Choose common multiplier. 1/2x is recommended for good speed/quality balance."
                                )
                                fps_multiplier_custom =gr .Number (
                                label ="Custom Multiplier",
                                value =INITIAL_APP_CONFIG.fps_decrease.multiplier_custom ,
                                minimum =0.1 ,
                                maximum =1.0 ,
                                step =0.05 ,
                                precision =2 ,
                                visible =False ,
                                info ="Custom multiplier value (0.1 to 1.0). Lower = fewer frames."
                                )

                        with gr .Group (visible =False )as fixed_controls :
                            target_fps =gr .Slider (
                            label ="Target FPS",
                            minimum =1.0 ,
                            maximum =60.0 ,
                            value =INITIAL_APP_CONFIG.fps_decrease.target_fps ,
                            step =0.001 ,
                            info ="Target FPS for the reduced video. Lower FPS = faster upscaling. Common choices: 12-15 FPS for fast processing, 24 FPS for cinema standard. Supports precise values like 23.976."
                            )

                        fps_interpolation_method =gr .Radio (
                        label ="Frame Reduction Method",
                        choices =["drop","blend"],
                        value =INITIAL_APP_CONFIG.fps_decrease.interpolation_method ,
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
                        value =INITIAL_APP_CONFIG.rife.enable_fps_limit ,
                        info ="Limit the output FPS to specific common values instead of unlimited interpolation. Useful for compatibility with displays and media players."
                        )

                        rife_max_fps_limit =gr .Radio (
                        label ="Max FPS Limit",
                        choices =[23.976 ,24 ,25 ,29.970 ,30 ,47.952 ,48 ,50 ,59.940 ,60 ,75 ,90 ,100 ,119.880 ,120 ,144 ,165 ,180 ,240 ,360 ],
                        value =INITIAL_APP_CONFIG.rife.max_fps_limit ,
                        info ="Maximum FPS when limiting is enabled. NTSC rates: 23.976/29.970/59.940 (film/TV), Standard: 24/25/30/50/60, Gaming: 120/144/240+. Choose based on your target format and display."
                        )

                        with gr .Row ():
                            rife_keep_original =gr .Checkbox (
                            label ="Keep Original Files",
                            value =INITIAL_APP_CONFIG.rife.keep_original ,
                            info ="Keep the original (non-interpolated) video files alongside the RIFE-processed versions. Recommended to compare results."
                            )
                            rife_overwrite_original =gr .Checkbox (
                            label ="Overwrite Original",
                            value =INITIAL_APP_CONFIG.rife.overwrite_original ,
                            info ="Replace the original upscaled video with the RIFE version as the primary output. When disabled, both versions are available."
                            )

        with gr .Tab ("Edit Videos",id ="edit_tab"):
            gr .Markdown ("# Video Editor - Cut and Extract Video Segments")
            gr .Markdown ("**Cut specific time ranges or frame ranges from your videos with precise FFmpeg encoding.**")
            
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        input_video_edit =gr .Video (
                        label ="Input Video for Editing",
                        sources =["upload"],
                        interactive =True ,
                        height =300 
                        )
                        
                        video_info_display =gr .Textbox (
                        label ="Video Information",
                        interactive =False ,
                        lines =6 ,
                        info ="Shows duration, FPS, frame count, resolution",
                        value ="📹 Upload a video to see detailed information"
                        )

                    with gr .Row ():
                        cut_and_save_btn =gr .Button ("Cut and Save",variant ="primary",icon ="icons/cut_paste.png")
                        cut_and_upscale_btn =gr .Button ("Cut and Move to Upscale",variant ="primary",icon ="icons/move_icon.png")
                    
                    with gr .Accordion ("Cutting Settings",open =True ):
                        cutting_mode =gr .Radio (
                        label ="Cutting Mode",
                        choices =['time_ranges','frame_ranges'],
                        value ='time_ranges',
                        info ="Choose between time-based or frame-based cutting"
                        )
                        
                        with gr .Group ()as time_range_controls :
                            time_ranges_input =gr .Textbox (
                            label ="Time Ranges (seconds)",
                            placeholder ="1-3,5-8,10-15 or 1:30-2:45,3:00-4:30",
                            info ="Format: start1-end1,start2-end2,... (supports decimal seconds and MM:SS format)",
                            lines =2 
                            )
                        
                        with gr .Group (visible =False )as frame_range_controls :
                            frame_ranges_input =gr .Textbox (
                            label ="Frame Ranges",
                            placeholder ="30-90,150-210,300-450",
                            info ="Format: start1-end1,start2-end2,... (frame numbers are 0-indexed)",
                            lines =2 
                            )
                        
                        cut_info_display =gr .Textbox (
                        label ="Cut Analysis",
                        interactive =False ,
                        lines =3 ,
                        info ="Shows details about the cuts being made",
                        value ="✏️ Enter ranges above to see cut analysis"
                        )
                    
                    with gr .Accordion ("Options",open =True ):
                        precise_cutting_mode =gr .Radio (
                        label ="Cutting Precision",
                        choices =['precise','fast'],
                        value ='precise',
                        info ="Precise: Frame-accurate re-encoding. Fast: Stream copy (faster but may be less accurate)"
                        )
                        
                        preview_first_segment =gr .Checkbox (
                        label ="Generate Preview of First Segment",
                        value =True ,
                        info ="Create a preview video of the first cut segment for verification"
                        )
                        
                        processing_estimate =gr .Textbox (
                        label ="Processing Time Estimate",
                        interactive =False ,
                        lines =1 ,
                        value ="📊 Upload video and enter ranges to see time estimate"
                        )
                    

                    
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        output_video_edit =gr .Video (
                        label ="Cut Video Output",
                        interactive =False ,
                        height =400 
                        )
                        
                        preview_video_edit =gr .Video (
                        label ="Preview (First Segment)",
                        interactive =False ,
                        height =300 
                        )
                    
                    with gr .Group ():
                        edit_status_textbox =gr .Textbox (
                        label ="Edit Status & Log",
                        interactive =False ,
                        lines =8 ,
                        max_lines =15 ,
                        value ="🎞️ Ready to edit videos. Upload a video and specify cut ranges to begin."
                        )
                    
                    with gr .Accordion ("Quick Help & Examples",open =True ):
                        gr .Markdown ("""
**Time Range Examples:**
- `1-3` → Cut from 1 to 3 seconds
- `1.5-3.2` → Cut from 1.5 to 3.2 seconds  
- `1:30-2:45` → Cut from 1 minute 30 seconds to 2 minutes 45 seconds
- `0:05-0:10,0:20-0:30` → Multiple segments

**Frame Range Examples:**
- `30-90` → Cut frames 30 to 90
- `30-90,150-210` → Cut frames 30-90 and 150-210
- `0-120,240-360` → Multiple frame segments

**Tips:**
- Use time ranges for easier input (supports MM:SS format)
- Use frame ranges for frame-perfect editing
- Preview first segment to verify before processing
- All cuts use your current FFmpeg settings from Output & Comparison tab
- Cut videos are saved in organized folders with metadata
""")

        with gr .Tab ("Face Restoration",id ="face_restoration_tab"):
            gr .Markdown ("# Standalone Face Restoration - CodeFormer Processing")
            gr .Markdown ("**Apply face restoration to videos using CodeFormer without upscaling. Perfect for improving face quality in existing videos.**")
            
            with gr .Row ():
                with gr .Column (scale =1 ):
                    with gr .Group ():
                        input_video_face_restoration =gr .Video (
                        label ="Input Video for Face Restoration",
                        sources =["upload"],
                        interactive =True ,
                        height =400 
                        )
                        
                        face_restoration_mode =gr .Radio (
                        label ="Processing Mode",
                        choices =["Single Video","Batch Folder"],
                        value ="Single Video",
                        info ="Choose between processing a single video or batch processing a folder of videos"
                        )
                        
                        with gr .Group (visible =False )as batch_folder_controls :
                            batch_input_folder_face =gr .Textbox (
                            label ="Input Folder Path",
                            placeholder ="C:/path/to/input/videos/",
                            info ="Folder containing videos to process with face restoration"
                            )
                            batch_output_folder_face =gr .Textbox (
                            label ="Output Folder Path", 
                            placeholder ="C:/path/to/output/videos/",
                            info ="Folder where face-restored videos will be saved"
                            )

                    with gr .Row ():
                        face_restoration_process_btn =gr .Button ("Process Face Restoration",variant ="primary",icon ="icons/face_restoration.png")
                        face_restoration_stop_btn =gr .Button ("Stop Processing",variant ="stop")

                    with gr .Accordion ("Face Restoration Settings",open =True ):
                        standalone_enable_face_restoration =gr .Checkbox (
                        label ="Enable Face Restoration",
                        value =True ,
                        info ="Enable CodeFormer face restoration processing. Must be enabled for any processing to occur."
                        )
                        
                        standalone_face_restoration_fidelity =gr .Slider (
                        label ="Face Restoration Fidelity Weight",
                        minimum =0.0 ,maximum =1.0 ,value =0.7 ,step =0.05 ,
                        info ="Balance between quality (0.3) and identity preservation (0.8). 0.7 is recommended for most videos."
                        )
                        
                        standalone_enable_face_colorization =gr .Checkbox (
                        label ="Enable Face Colorization",
                        value =False ,
                        info ="Enable colorization for grayscale faces. Useful for old black & white videos or grayscale content."
                        )
                        
                        with gr .Row ():
                            standalone_model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            standalone_default_model_choice = "Auto (Default)"
                            
                            standalone_codeformer_model_dropdown =gr .Dropdown (
                            label ="CodeFormer Model",
                            choices =standalone_model_choices,
                            value =standalone_default_model_choice,
                            info ="Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory."
                            )
                        
                        standalone_face_restoration_batch_size =gr .Slider (
                        label ="Processing Batch Size",
                        minimum =1 ,maximum =16 ,value =4 ,step =1 ,
                        info ="Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage."
                        )


                    
                    with gr .Accordion ("Advanced Options",open =True ):
                        standalone_save_frames =gr .Checkbox (
                        label ="Save Individual Frames",
                        value =False ,
                        info ="Save processed frames as individual image files alongside the video"
                        )
                        
                        standalone_create_comparison =gr .Checkbox (
                        label ="Create Before/After Comparison Video",
                        value =True ,
                        info ="Create a side-by-side comparison video showing original vs face-restored results"
                        )
                        
                        standalone_preserve_audio =gr .Checkbox (
                        label ="Preserve Original Audio",
                        value =True ,
                        info ="Keep the original audio track in the processed video"
                        )

                with gr .Column (scale =1 ):
                    with gr .Group ():
                        output_video_face_restoration =gr .Video (
                        label ="Face Restored Video",
                        interactive =False ,
                        height =400 
                        )
                        
                        comparison_video_face_restoration =gr .Video (
                        label ="Before/After Comparison",
                        interactive =False ,
                        height =300 ,
                        visible =True 
                        )
                    
                    with gr .Group ():
                        face_restoration_status =gr .Textbox (
                        label ="Face Restoration Status & Log",
                        interactive =False ,
                        lines =10 ,
                        max_lines =20 ,
                        value ="🎭 Ready for face restoration processing. Upload a video and configure settings to begin."
                        )
                    
                    with gr .Accordion ("Processing Statistics",open =True ):
                        face_restoration_stats =gr .Textbox (
                        label ="Processing Stats",
                        interactive =False ,
                        lines =4 ,
                        value ="📊 Processing statistics will appear here during face restoration."
                        )
                    
                    with gr .Accordion ("Face Restoration Help",open =True ):
                        gr .Markdown ("""
**Face Restoration Tips:**
- **Fidelity Weight 0.3-0.5**: Focus on quality enhancement, may change face slightly
- **Fidelity Weight 0.7-0.8**: Balance between quality and identity preservation  
- **Fidelity Weight 0.8-1.0**: Maximum identity preservation, minimal changes

**Best Practices:**
- Use comparison video to evaluate results before final processing
- Start with fidelity weight 0.7 for most videos
- Enable colorization only for grayscale/black & white content
- Higher batch sizes speed up processing but use more VRAM
- Face restoration works best on videos with clear, visible faces

**Model Requirements:**
- CodeFormer models should be placed in `pretrained_weight/` directory
- The system will automatically detect available models
- 'Auto' mode uses the default CodeFormer model with optimal settings
""")

    def update_steps_display (mode ):
        if mode =='fast':
            return gr .update (value =INITIAL_APP_CONFIG.star_model.steps ,interactive =False )
        else :
            from logic.dataclasses import DEFAULT_DIFFUSION_STEPS_NORMAL
            return gr .update (value =DEFAULT_DIFFUSION_STEPS_NORMAL ,interactive =True )
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
    enable_context_window_check .change (
    lambda x :gr .update (interactive =x ),
    inputs =enable_context_window_check ,
    outputs =context_overlap_num 
    )

    def validate_enhanced_input_wrapper(input_path):
        """Validate enhanced input path (video file or frames folder) and show detailed status."""
        if not input_path or not input_path.strip():
            return "Enter a video file path or frames folder path above to validate"
        
        from logic.frame_folder_utils import validate_input_path
        is_valid, message, metadata = validate_input_path(input_path, logger)
        
        if is_valid:
            # Show additional details for successful validation
            if metadata.get("frame_count"):
                # It's a frames folder
                formats = metadata.get("supported_formats", [])
                detail_msg = f"\n📊 Detected formats: {', '.join(formats)}" if formats else ""
                return f"{message}{detail_msg}"
            elif metadata.get("duration"):
                # It's a video file
                return f"{message}\n🎬 Input Type: Video File"
            else:
                return message
        else:
            return message
    
    input_frames_folder.change(
        fn=validate_enhanced_input_wrapper,
        inputs=[input_frames_folder],
        outputs=[frames_folder_status]
    )

    def update_image_upscaler_controls(enable_image_upscaler):
        return [
            gr.update(interactive=enable_image_upscaler),
            gr.update(interactive=enable_image_upscaler)
        ]
    
    enable_image_upscaler_check.change(
        fn=update_image_upscaler_controls,
        inputs=enable_image_upscaler_check,
        outputs=[image_upscaler_model_dropdown, image_upscaler_batch_size_slider]
    )
    
    def update_face_restoration_controls(enable_face_restoration):
        return [
            gr.update(interactive=enable_face_restoration),
            gr.update(interactive=enable_face_restoration),
            gr.update(interactive=enable_face_restoration),
            gr.update(interactive=enable_face_restoration),
            gr.update(interactive=enable_face_restoration)
        ]
    
    enable_face_restoration_check.change(
        fn=update_face_restoration_controls,
        inputs=enable_face_restoration_check,
        outputs=[
            face_restoration_fidelity_slider, 
            enable_face_colorization_check, 
            face_restoration_when_radio,
            codeformer_model_dropdown, 
            face_restoration_batch_size_slider
        ]
    )
    
    def update_face_restoration_mode_controls(mode):
        if mode == "Batch Folder":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    
    face_restoration_mode.change(
        fn=update_face_restoration_mode_controls,
        inputs=face_restoration_mode,
        outputs=batch_folder_controls
    )
    
    def refresh_upscaler_models():
        try:
            available_model_files = util_scan_for_models(APP_CONFIG.paths.upscale_models_dir, logger)
            if available_model_files:
                model_choices = available_model_files
                default_choice = model_choices[0]
            else:
                model_choices = ["No models found - place models in upscale_models/"]
                default_choice = model_choices[0]
            return gr.update(choices=model_choices, value=default_choice)
        except Exception as e:
            logger.warning(f"Failed to refresh upscaler models: {e}")
            return gr.update(choices=["Error scanning models - check upscale_models/ directory"], 
                           value="Error scanning models - check upscale_models/ directory")
    
    def update_context_overlap_max (max_chunk_len ):
        new_max =max (0 ,int (max_chunk_len )-1 )
        return gr .update (maximum =new_max )

    max_chunk_len_slider .change (
    fn =update_context_overlap_max ,
    inputs =max_chunk_len_slider ,
    outputs =context_overlap_num 
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
        from logic.dataclasses import DEFAULT_FFMPEG_QUALITY_GPU, DEFAULT_FFMPEG_QUALITY_CPU
        if use_gpu_ffmpeg :
            return gr .Slider (label ="FFmpeg Quality (CQ for NVENC)",value =DEFAULT_FFMPEG_QUALITY_GPU ,info ="For h24_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28.")
        else :
            return gr .Slider (label ="FFmpeg Quality (CRF for libx264)",value =DEFAULT_FFMPEG_QUALITY_CPU ,info ="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default).")

    ffmpeg_use_gpu_check .change (
    fn =update_ffmpeg_quality_settings ,
    inputs =ffmpeg_use_gpu_check ,
    outputs =ffmpeg_quality_slider 
    )

    open_output_folder_button .click (
    fn =lambda :util_open_folder (APP_CONFIG.paths.outputs_dir ,logger =logger ),
    inputs =[],
    outputs =[]
    )

    cogvlm_display_to_quant_val_map_global ={}
    if UTIL_COG_VLM_AVAILABLE :
        _temp_map =get_cogvlm_quant_choices_map (torch .cuda .is_available (),UTIL_BITSANDBYTES_AVAILABLE )
        cogvlm_display_to_quant_val_map_global ={v :k for k ,v in _temp_map .items ()}

    def get_quant_value_from_display (display_val ):
        if display_val is None :return 0 
        if isinstance (display_val ,int ):return display_val 
        return cogvlm_display_to_quant_val_map_global .get (display_val ,0 )
    
    def extract_codeformer_model_path_from_dropdown(dropdown_choice):
        if not dropdown_choice or dropdown_choice.startswith("No CodeFormer") or dropdown_choice.startswith("Error"):
            return None
        if dropdown_choice == "Auto (Default)":
            return None
        
        if dropdown_choice.startswith("codeformer.pth"):
            return os.path.join(APP_CONFIG.paths.face_restoration_models_dir, "CodeFormer", "codeformer.pth")
        
        return None

    def extract_gpu_index_from_dropdown(dropdown_choice):
        """Extract GPU index from dropdown choice for preset saving."""
        if dropdown_choice is None or dropdown_choice == "No CUDA GPUs detected":
            return 0  # Default to GPU 0 index
        
        available_gpus = util_get_available_gpus()
        if not available_gpus:
            return 0
        
        # Handle case where dropdown returns an integer (choice index) - this is what we want!
        if isinstance(dropdown_choice, int):
            # Ensure it's within valid range
            if 0 <= dropdown_choice < len(available_gpus):
                return dropdown_choice
            else:
                logger.warning(f"GPU index {dropdown_choice} is out of range. Available GPUs: {len(available_gpus)}. Defaulting to 0.")
                return 0
        
        # Handle string format like "GPU 0: Device Name" - convert to index
        if isinstance(dropdown_choice, str) and dropdown_choice.startswith("GPU "):
            try:
                # Extract number from "GPU 0: Device Name" format
                gpu_index = int(dropdown_choice.split(":")[0].replace("GPU ", "").strip())
                # Ensure it's within valid range
                if 0 <= gpu_index < len(available_gpus):
                    return gpu_index
                else:
                    logger.warning(f"GPU index {gpu_index} is out of range. Available GPUs: {len(available_gpus)}. Defaulting to 0.")
                    return 0
            except:
                logger.warning(f"Failed to parse GPU index from '{dropdown_choice}'. Defaulting to 0.")
                return 0
        
        # Find the choice in the available GPUs list
        try:
            choice_index = available_gpus.index(dropdown_choice)
            return choice_index
        except ValueError:
            logger.warning(f"GPU choice '{dropdown_choice}' not found in available GPUs. Defaulting to 0.")
            return 0

    def convert_gpu_index_to_dropdown(gpu_index, available_gpus):
        """Convert GPU index back to dropdown format for preset loading."""
        if not available_gpus:
            return "No CUDA GPUs detected"
            
        if gpu_index is None or gpu_index == "Auto":
            # Default to GPU 0 (first GPU) 
            return available_gpus[0] if available_gpus else "No CUDA GPUs detected"
        
        try:
            # Handle both integer and string inputs
            if isinstance(gpu_index, int):
                gpu_num = gpu_index
            else:
                gpu_num = int(gpu_index)
            
            # Ensure the GPU index is within bounds
            if 0 <= gpu_num < len(available_gpus):
                return available_gpus[gpu_num]  # This will be "GPU X: Device Name"
            else:
                # If index is out of range, default to GPU 0
                logger.warning(f"GPU index {gpu_num} is out of range. Available GPUs: {len(available_gpus)}. Defaulting to GPU 0.")
                return available_gpus[0] if available_gpus else "No CUDA GPUs detected"
        except (ValueError, TypeError):
            # If conversion fails, default to GPU 0
            logger.warning(f"Failed to convert GPU index '{gpu_index}' to integer. Defaulting to GPU 0.")
            return available_gpus[0] if available_gpus else "No CUDA GPUs detected"

    def build_app_config_from_ui(*args):
        # This function takes all UI component values and builds an AppConfig object
        (
            input_video_val, user_prompt_val, pos_prompt_val, neg_prompt_val, model_selector_val,
            upscale_factor_slider_val, cfg_slider_val, steps_slider_val, solver_mode_radio_val,
            max_chunk_len_slider_val, enable_chunk_optimization_check_val, vae_chunk_slider_val, color_fix_dropdown_val,
            enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
            enable_context_window_check_val, context_overlap_num_val,
            enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
            enable_auto_aspect_resolution_check_val, auto_resolution_status_display_val,
            ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
            save_frames_checkbox_val, save_metadata_checkbox_val, save_chunks_checkbox_val, save_chunk_frames_checkbox_val,
            create_comparison_video_check_val,
            enable_scene_split_check_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
            scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
            scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
            scene_manual_split_type_radio_val, scene_manual_split_value_num_val,
            enable_fps_decrease_val, fps_decrease_mode_val, fps_multiplier_preset_val, fps_multiplier_custom_val, target_fps_val, fps_interpolation_method_val,
            enable_rife_interpolation_val, rife_multiplier_val, rife_fp16_val, rife_uhd_val, rife_scale_val,
            rife_skip_static_val, rife_enable_fps_limit_val, rife_max_fps_limit_val,
            rife_apply_to_chunks_val, rife_apply_to_scenes_val, rife_keep_original_val, rife_overwrite_original_val,
            cogvlm_quant_radio_val, cogvlm_unload_radio_val, do_auto_caption_first_val,
            seed_num_val, random_seed_check_val,
            enable_image_upscaler_val, image_upscaler_model_val, image_upscaler_batch_size_val,
            enable_face_restoration_val, face_restoration_fidelity_val, enable_face_colorization_val,
            face_restoration_when_val, codeformer_model_val, face_restoration_batch_size_val,
            input_frames_folder_val, frame_folder_fps_slider_val,
            gpu_selector_val
        ) = args

        # Auto-detect if frame folder processing should be enabled based on input path
        frame_folder_enable = False
        if input_frames_folder_val and input_frames_folder_val.strip():
            from logic.frame_folder_utils import detect_input_type
            input_type, _, _ = detect_input_type(input_frames_folder_val, logger)
            frame_folder_enable = (input_type == "frames_folder")
            if frame_folder_enable:
                logger.info(f"Auto-detected frames folder: {input_frames_folder_val}")
            elif input_type == "video_file":
                logger.info(f"Auto-detected video file: {input_frames_folder_val}")

        config = AppConfig(
            input_video_path=input_video_val,
            paths=APP_CONFIG.paths, # Use globally initialized paths
            prompts=PromptConfig(
                user=user_prompt_val,
                positive=pos_prompt_val,
                negative=neg_prompt_val
            ),
            star_model=StarModelConfig(
                model_choice=model_selector_val,
                cfg_scale=cfg_slider_val,
                solver_mode=solver_mode_radio_val,
                steps=steps_slider_val,
                color_fix_method=color_fix_dropdown_val
            ),
            performance=PerformanceConfig(
                max_chunk_len=max_chunk_len_slider_val,
                vae_chunk=vae_chunk_slider_val,
                enable_chunk_optimization=enable_chunk_optimization_check_val
            ),
            resolution=ResolutionConfig(
                enable_target_res=enable_target_res_check_val,
                target_res_mode=target_res_mode_radio_val,
                target_h=target_h_num_val,
                target_w=target_w_num_val,
                upscale_factor=upscale_factor_slider_val,
                enable_auto_aspect_resolution=enable_auto_aspect_resolution_check_val,
                auto_resolution_status=auto_resolution_status_display_val
            ),
            context_window=ContextWindowConfig(
                enable=enable_context_window_check_val,
                overlap=context_overlap_num_val
            ),
            tiling=TilingConfig(
                enable=enable_tiling_check_val,
                tile_size=tile_size_num_val,
                tile_overlap=tile_overlap_num_val
            ),
            ffmpeg=FfmpegConfig(
                use_gpu=ffmpeg_use_gpu_check_val,
                preset=ffmpeg_preset_dropdown_val,
                quality=ffmpeg_quality_slider_val
            ),
            frame_folder=FrameFolderConfig(
                enable=frame_folder_enable,
                input_path=input_frames_folder_val,
                fps=frame_folder_fps_slider_val
            ),
            scene_split=SceneSplitConfig(
                enable=enable_scene_split_check_val,
                mode=scene_split_mode_radio_val,
                min_scene_len=scene_min_scene_len_num_val,
                threshold=scene_threshold_num_val,
                drop_short=scene_drop_short_check_val,
                merge_last=scene_merge_last_check_val,
                frame_skip=scene_frame_skip_num_val,
                min_content_val=scene_min_content_val_num_val,
                frame_window=scene_frame_window_num_val,
                manual_split_type=scene_manual_split_type_radio_val,
                manual_split_value=scene_manual_split_value_num_val,
                copy_streams=scene_copy_streams_check_val,
                use_mkvmerge=scene_use_mkvmerge_check_val,
                rate_factor=scene_rate_factor_num_val,
                encoding_preset=scene_preset_dropdown_val,
                quiet_ffmpeg=scene_quiet_ffmpeg_check_val
            ),
            cogvlm=CogVLMConfig(
                quant_display=cogvlm_quant_radio_val,
                unload_after_use=cogvlm_unload_radio_val,
                auto_caption_then_upscale=do_auto_caption_first_val,
                quant_value=get_quant_value_from_display(cogvlm_quant_radio_val)
            ),
            outputs=OutputConfig(
                save_frames=save_frames_checkbox_val,
                save_metadata=save_metadata_checkbox_val,
                save_chunks=save_chunks_checkbox_val,
                save_chunk_frames=save_chunk_frames_checkbox_val,
                create_comparison_video=create_comparison_video_check_val
            ),
            seed=SeedConfig(
                seed=seed_num_val,
                use_random=random_seed_check_val
            ),
            rife=RifeConfig(
                enable=enable_rife_interpolation_val,
                multiplier=rife_multiplier_val,
                fp16=rife_fp16_val,
                uhd=rife_uhd_val,
                scale=rife_scale_val,
                skip_static=rife_skip_static_val,
                enable_fps_limit=rife_enable_fps_limit_val,
                max_fps_limit=rife_max_fps_limit_val,
                apply_to_chunks=rife_apply_to_chunks_val,
                apply_to_scenes=rife_apply_to_scenes_val,
                keep_original=rife_keep_original_val,
                overwrite_original=rife_overwrite_original_val
            ),
            fps_decrease=FpsDecreaseConfig(
                enable=enable_fps_decrease_val,
                mode=fps_decrease_mode_val,
                multiplier_preset=fps_multiplier_preset_val,
                multiplier_custom=fps_multiplier_custom_val,
                target_fps=target_fps_val,
                interpolation_method=fps_interpolation_method_val
            ),
            batch=BatchConfig(
                input_folder="",
                output_folder="",
                skip_existing=True,
                save_captions=True,
                use_prompt_files=True,
                enable_auto_caption=False,
                enable_frame_folders=False
            ),
            image_upscaler=ImageUpscalerConfig(
                enable=enable_image_upscaler_val,
                model=image_upscaler_model_val,
                batch_size=image_upscaler_batch_size_val
            ),
            face_restoration=FaceRestorationConfig(
                enable=enable_face_restoration_val,
                fidelity_weight=face_restoration_fidelity_val,
                enable_colorization=enable_face_colorization_val,
                when=face_restoration_when_val,
                model=extract_codeformer_model_path_from_dropdown(codeformer_model_val),
                batch_size=face_restoration_batch_size_val
            ),
            gpu=GpuConfig(
                device=str(extract_gpu_index_from_dropdown(gpu_selector_val))
            )
        )
        return config

    def upscale_director_logic (app_config: AppConfig, progress =gr .Progress (track_tqdm =True )):
        # Reset cancellation state at the very beginning of new processing
        cancellation_manager.reset()
        logger.info("Starting new upscale process - cancellation state reset")
        
        current_output_video_val =None 
        current_status_text_val =""
        current_user_prompt_val =app_config.prompts.user
        current_caption_status_text_val =""
        current_caption_status_visible_val =False 
        current_last_chunk_video_val =None 
        current_chunk_status_text_val ="No chunks processed yet"
        current_comparison_video_val =None 

        auto_caption_completed_successfully =False 

        log_accumulator_director =[]

        logger .info (f"In upscale_director_logic. Auto-caption first: {app_config.cogvlm.auto_caption_then_upscale}, User prompt: '{app_config.prompts.user[:50]}...'")

        actual_input_video_path = app_config.input_video_path

        # Check if enhanced input path contains a video file or frame folder
        if app_config.frame_folder.input_path and app_config.frame_folder.input_path.strip():
            from logic.frame_folder_utils import detect_input_type, validate_input_path
            
            input_type, validated_path, metadata = detect_input_type(app_config.frame_folder.input_path, logger)
            
            if input_type == "video_file":
                # If it's a video file, use it as the main input video instead of the frame folder
                logger.info(f"Enhanced input detected video file: {validated_path}")
                actual_input_video_path = validated_path
                
                # Update status to show video file was detected
                info_msg = f"✅ Using enhanced input video file: {os.path.basename(validated_path)}"
                if metadata.get("duration"):
                    info_msg += f" ({metadata['duration']:.1f}s, {metadata.get('width', '?')}x{metadata.get('height', '?')})"
                log_accumulator_director.append(info_msg)
                
            elif input_type == "frames_folder":
                logger.info("Enhanced input detected frames folder - converting to video")
                progress(0, desc="Converting frame folder to video...")
                
                is_valid, validation_msg, validation_metadata = validate_input_path(app_config.frame_folder.input_path, logger)
                
                if not is_valid:
                    error_msg = f"Frame folder validation failed: {validation_msg}"
                    logger.error(error_msg)
                    log_accumulator_director.append(error_msg)
                    current_status_text_val = "\n".join(log_accumulator_director)
                    yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                           gr.update(value=current_user_prompt_val),
                           gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                           gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                                                      gr.update(value=current_comparison_video_val))
                    return
                
                # Extract frame count from validation metadata
                frame_count = validation_metadata.get("frame_count", 0)
                
                temp_video_dir = tempfile.mkdtemp(prefix="frame_folder_")
                frame_folder_name = os.path.basename(app_config.frame_folder.input_path.rstrip(os.sep))
                temp_video_path = os.path.join(temp_video_dir, f"{frame_folder_name}_from_frames.mp4")
                
                conversion_msg = f"Converting {frame_count} frames to video using global encoding settings..."
                log_accumulator_director.append(conversion_msg)
                current_status_text_val = "\n".join(log_accumulator_director)
                yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                       gr.update(value=current_user_prompt_val),
                       gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                       gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                       gr.update(value=current_comparison_video_val))
                
                try:
                    success, conv_msg = util_process_frame_folder_to_video(
                        app_config.frame_folder.input_path, temp_video_path, fps=app_config.frame_folder.fps,
                        ffmpeg_preset=app_config.ffmpeg.preset,
                        ffmpeg_quality_value=app_config.ffmpeg.quality,
                        ffmpeg_use_gpu=app_config.ffmpeg.use_gpu,
                        logger=logger
                    )
                    
                    if success:
                        actual_input_video_path = temp_video_path
                        app_config.input_video_path = actual_input_video_path
                        success_msg = f"✅ Successfully converted frame folder to video: {conv_msg}"
                        log_accumulator_director.append(success_msg)
                        logger.info(f"Frame folder converted successfully. Using: {actual_input_video_path}")
                    else:
                        error_msg = f"❌ Failed to convert frame folder: {conv_msg}"
                        log_accumulator_director.append(error_msg)
                        logger.error(error_msg)
                        current_status_text_val = "\n".join(log_accumulator_director)
                        yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                               gr.update(value=current_user_prompt_val),
                               gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                               gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                               gr.update(value=current_comparison_video_val))
                        return
                        
                except Exception as e:
                    error_msg = f"❌ Exception during frame folder conversion: {str(e)}"
                    log_accumulator_director.append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    current_status_text_val = "\n".join(log_accumulator_director)
                    yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                           gr.update(value=current_user_prompt_val),
                           gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                           gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                           gr.update(value=current_comparison_video_val))
                    return
                
                current_status_text_val = "\n".join(log_accumulator_director)
                yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                       gr.update(value=current_user_prompt_val),
                       gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                       gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                       gr.update(value=current_comparison_video_val))
                
                log_accumulator_director = []
                
            elif input_type == "invalid":
                error_msg = f"❌ Enhanced input validation failed: {metadata.get('error', 'Unknown error')}"
                logger.error(error_msg)
                log_accumulator_director.append(error_msg)
                current_status_text_val = "\n".join(log_accumulator_director)
                yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                       gr.update(value=current_user_prompt_val),
                       gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                       gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                       gr.update(value=current_comparison_video_val))
                return
            
            # Update current status after enhanced input processing
            current_status_text_val = "\n".join(log_accumulator_director)
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
            
            log_accumulator_director = []

        should_auto_caption_entire_video =(app_config.cogvlm.auto_caption_then_upscale and 
        not app_config.scene_split.enable and 
        not app_config.image_upscaler.enable and 
        UTIL_COG_VLM_AVAILABLE )

        if should_auto_caption_entire_video :
            # Don't auto-caption if user prompt contains previous cancellation messages
            if "cancelled" in current_user_prompt_val.lower() or "error" in current_user_prompt_val.lower():
                logger.info("Skipping auto-caption because prompt contains error/cancelled text from previous run")
                current_user_prompt_val = "..."  # Reset to default prompt
                app_config.prompts.user = current_user_prompt_val
            logger .info ("Attempting auto-captioning entire video before upscale (scene splitting disabled).")
            progress (0 ,desc ="Starting auto-captioning before upscale...")

            current_status_text_val ="Starting auto-captioning..."
            if UTIL_COG_VLM_AVAILABLE :
                current_caption_status_text_val ="Starting auto-captioning..."
                current_caption_status_visible_val =True 

            yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
            gr .update (value =current_user_prompt_val ),
            gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
            gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
            gr .update (value =current_comparison_video_val ))

            try :
                # Check for cancellation before auto-captioning
                cancellation_manager.check_cancel()
                
                caption_text ,caption_stat_msg =util_auto_caption (
                app_config.input_video_path ,app_config.cogvlm.quant_value ,app_config.cogvlm.unload_after_use ,
                app_config.paths.cogvlm_model_path ,logger =logger ,progress =progress 
                )
                
                # Check for cancellation after auto-captioning
                cancellation_manager.check_cancel()
                
                log_accumulator_director .append (f"Auto-caption status: {caption_stat_msg}")

                if not caption_text .startswith ("Error:") and not caption_text.startswith("Caption generation cancelled"):
                    current_user_prompt_val =caption_text 
                    app_config.prompts.user = caption_text
                    auto_caption_completed_successfully =True 
                    log_accumulator_director .append (f"Using generated caption as prompt: '{caption_text[:50]}...'")
                    logger .info (f"Auto-caption successful. Updated current_user_prompt_val to: '{current_user_prompt_val[:100]}...'")
                else :
                    # Keep the original prompt if auto-captioning failed/cancelled
                    log_accumulator_director .append ("Caption generation failed or was cancelled. Using original prompt.")
                    logger .warning (f"Auto-caption failed or cancelled. Keeping original prompt: '{current_user_prompt_val[:100]}...'")
                    # Don't update current_user_prompt_val with error text - keep original

                current_status_text_val ="\n".join (log_accumulator_director )
                if UTIL_COG_VLM_AVAILABLE :
                    current_caption_status_text_val =caption_stat_msg 

                logger .info (f"About to yield auto-caption result. current_user_prompt_val: '{current_user_prompt_val[:100]}...'")
                yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
                gr .update (value =current_user_prompt_val ),
                gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
                gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
                gr .update (value =current_comparison_video_val ))
                logger .info ("Auto-caption yield completed.")

                if UTIL_COG_VLM_AVAILABLE :
                    current_caption_status_visible_val =False 

            except CancelledError as e_cancel:
                logger.info("Auto-captioning cancelled by user - stopping process")
                log_accumulator_director.append("⚠️ Process cancelled by user during auto-captioning")
                current_status_text_val = "\n".join(log_accumulator_director)
                if UTIL_COG_VLM_AVAILABLE:
                    current_caption_status_text_val = "Auto-captioning cancelled by user"
                    current_caption_status_visible_val = False
                yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                       gr.update(value=current_user_prompt_val),
                       gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                       gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                       gr.update(value=current_comparison_video_val))
                return
            except Exception as e_ac :
                logger .error (f"Exception during auto-caption call or its setup: {e_ac}",exc_info =True )
                log_accumulator_director .append (f"Error during auto-caption pre-step: {e_ac}")
                current_status_text_val ="\n".join (log_accumulator_director )
                if UTIL_COG_VLM_AVAILABLE :
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

        elif app_config.cogvlm.auto_caption_then_upscale and app_config.scene_split.enable and not app_config.image_upscaler.enable and UTIL_COG_VLM_AVAILABLE :
            msg ="Scene splitting enabled: Auto-captioning will be done per scene during upscaling."
            logger .info (msg )
            log_accumulator_director .append (msg )
            current_status_text_val ="\n".join (log_accumulator_director )

            yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
            gr .update (value =current_user_prompt_val ),
            gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
            gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
            gr .update (value =current_comparison_video_val ))

        elif app_config.cogvlm.auto_caption_then_upscale and app_config.image_upscaler.enable :
            msg ="Image-based upscaling is enabled. Auto-captioning is disabled as image upscalers don't use prompts."
            logger .info (msg )
            log_accumulator_director .append (msg )
            current_status_text_val ="\n".join (log_accumulator_director )

            yield (gr .update (value =current_output_video_val ),gr .update (value =current_status_text_val ),
            gr .update (value =current_user_prompt_val ),
            gr .update (value =current_caption_status_text_val ,visible =current_caption_status_visible_val ),
            gr .update (value =current_last_chunk_video_val ),gr .update (value =current_chunk_status_text_val ),
            gr .update (value =current_comparison_video_val ))

        elif app_config.cogvlm.auto_caption_then_upscale and not UTIL_COG_VLM_AVAILABLE :
            msg ="Auto-captioning requested but CogVLM2 is not available. Using original prompt."
            logger .warning (msg )

        # Check for cancellation before proceeding to the main upscaling step
        try:
            cancellation_manager.check_cancel()
        except CancelledError:
            logger.info("Process cancelled before main upscaling - stopping")
            log_accumulator_director.append("⚠️ Process cancelled by user")
            current_status_text_val = "\n".join(log_accumulator_director)
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
            return

        actual_seed_to_use =app_config.seed.seed 
        if app_config.seed.use_random :
            actual_seed_to_use =np .random .randint (0 ,2 **31 )
            logger .info (f"Random seed checkbox is checked. Using generated seed: {actual_seed_to_use}")
        elif app_config.seed.seed ==-1 :
            actual_seed_to_use =np .random .randint (0 ,2 **31 )
            logger .info (f"Seed input is -1. Using generated seed: {actual_seed_to_use}")
        else :
            logger .info (f"Using provided seed: {actual_seed_to_use}")
        
        app_config.seed.seed = actual_seed_to_use

        try:
            upscale_generator =core_run_upscale (
            input_video_path =actual_input_video_path ,user_prompt =app_config.prompts.user ,
            positive_prompt =app_config.prompts.positive ,negative_prompt =app_config.prompts.negative ,model_choice =app_config.star_model.model_choice ,
            upscale_factor_slider =app_config.resolution.upscale_factor ,cfg_scale =app_config.star_model.cfg_scale ,steps =app_config.star_model.steps ,solver_mode =app_config.star_model.solver_mode ,
            max_chunk_len =app_config.performance.max_chunk_len ,enable_chunk_optimization =app_config.performance.enable_chunk_optimization ,vae_chunk =app_config.performance.vae_chunk ,color_fix_method =app_config.star_model.color_fix_method ,
            enable_tiling =app_config.tiling.enable ,tile_size =app_config.tiling.tile_size ,tile_overlap =app_config.tiling.tile_overlap ,
            enable_context_window =app_config.context_window.enable ,context_overlap =app_config.context_window.overlap ,
            enable_target_res =app_config.resolution.enable_target_res ,target_h =app_config.resolution.target_h ,target_w =app_config.resolution.target_w ,target_res_mode =app_config.resolution.target_res_mode ,
            ffmpeg_preset =app_config.ffmpeg.preset ,ffmpeg_quality_value =app_config.ffmpeg.quality ,ffmpeg_use_gpu =app_config.ffmpeg.use_gpu ,
            save_frames =app_config.outputs.save_frames ,save_metadata =app_config.outputs.save_metadata ,save_chunks =app_config.outputs.save_chunks ,save_chunk_frames =app_config.outputs.save_chunk_frames ,

            enable_scene_split =app_config.scene_split.enable ,scene_split_mode =app_config.scene_split.mode ,scene_min_scene_len =app_config.scene_split.min_scene_len ,scene_drop_short =app_config.scene_split.drop_short ,scene_merge_last =app_config.scene_split.merge_last ,
            scene_frame_skip =app_config.scene_split.frame_skip ,scene_threshold =app_config.scene_split.threshold ,scene_min_content_val =app_config.scene_split.min_content_val ,scene_frame_window =app_config.scene_split.frame_window ,
            scene_copy_streams =app_config.scene_split.copy_streams ,scene_use_mkvmerge =app_config.scene_split.use_mkvmerge ,scene_rate_factor =app_config.scene_split.rate_factor ,scene_preset =app_config.scene_split.encoding_preset ,scene_quiet_ffmpeg =app_config.scene_split.quiet_ffmpeg ,
            scene_manual_split_type =app_config.scene_split.manual_split_type ,scene_manual_split_value =app_config.scene_split.manual_split_value ,

            create_comparison_video_enabled =app_config.outputs.create_comparison_video ,

            enable_fps_decrease =app_config.fps_decrease.enable ,fps_decrease_mode =app_config.fps_decrease.mode ,
            fps_multiplier_preset =app_config.fps_decrease.multiplier_preset ,fps_multiplier_custom =app_config.fps_decrease.multiplier_custom ,
            target_fps =app_config.fps_decrease.target_fps ,fps_interpolation_method =app_config.fps_decrease.interpolation_method ,

            enable_rife_interpolation =app_config.rife.enable ,rife_multiplier =app_config.rife.multiplier ,rife_fp16 =app_config.rife.fp16 ,rife_uhd =app_config.rife.uhd ,rife_scale =app_config.rife.scale ,
            rife_skip_static =app_config.rife.skip_static ,rife_enable_fps_limit =app_config.rife.enable_fps_limit ,rife_max_fps_limit =app_config.rife.max_fps_limit ,
            rife_apply_to_chunks =app_config.rife.apply_to_chunks ,rife_apply_to_scenes =app_config.rife.apply_to_scenes ,rife_keep_original =app_config.rife.keep_original ,rife_overwrite_original =app_config.rife.overwrite_original ,

            is_batch_mode =False ,batch_output_dir =None ,original_filename =None ,

            enable_auto_caption_per_scene =(app_config.cogvlm.auto_caption_then_upscale and app_config.scene_split.enable and not app_config.image_upscaler.enable and UTIL_COG_VLM_AVAILABLE ),
            cogvlm_quant =app_config.cogvlm.quant_value ,
            cogvlm_unload =app_config.cogvlm.unload_after_use if app_config.cogvlm.unload_after_use else 'full',

            logger =logger ,
            app_config_module =app_config_module ,
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
            current_seed =actual_seed_to_use ,
            
            # Image upscaler parameters
            enable_image_upscaler =app_config.image_upscaler.enable ,
            image_upscaler_model =app_config.image_upscaler.model ,
            image_upscaler_batch_size =app_config.image_upscaler.batch_size ,
            
            # Face restoration parameters
            enable_face_restoration =app_config.face_restoration.enable ,
            face_restoration_fidelity =app_config.face_restoration.fidelity_weight ,
            enable_face_colorization =app_config.face_restoration.enable_colorization ,
            face_restoration_timing ="after_upscale" ,  # Fixed timing mode for single video processing
            face_restoration_when =app_config.face_restoration.when ,
            codeformer_model =app_config.face_restoration.model ,
            face_restoration_batch_size =app_config.face_restoration.batch_size 
            )

            cancellation_detected = False
            for yielded_output_video ,yielded_status_log ,yielded_chunk_video ,yielded_chunk_status ,yielded_comparison_video in upscale_generator :

                # Check if this is a partial video result after cancellation
                is_partial_video = yielded_output_video and "partial_cancelled" in yielded_output_video
                
                # If we already detected cancellation and have a partial video, don't overwrite it
                if cancellation_detected and current_output_video_val and "partial_cancelled" in current_output_video_val:
                    if not is_partial_video:
                        logger.info(f"Skipping output video update to preserve partial video: {os.path.basename(current_output_video_val)}")
                        output_video_update = gr.update(value=current_output_video_val)
                    else:
                        # Allow partial video updates
                        current_output_video_val = yielded_output_video
                        output_video_update = gr.update(value=current_output_video_val)
                        logger.info(f"Updated partial video from generator: {os.path.basename(yielded_output_video)}")
                else:
                    output_video_update =gr .update ()
                    if yielded_output_video is not None :
                        current_output_video_val =yielded_output_video 
                        output_video_update =gr .update (value =current_output_video_val )
                        logger.info(f"Updated output video from generator: {os.path.basename(yielded_output_video) if yielded_output_video else 'None'}")
                        
                        # Detect if this is a partial video (indicates cancellation occurred)
                        if is_partial_video:
                            cancellation_detected = True
                            logger.info("Detected partial video - cancellation mode activated")
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

        except CancelledError:
            logger.warning("Processing was cancelled by user.")
            logger.info(f"Current output video value at cancellation: {current_output_video_val}")
            
            # Check if we already have a partial video from the generator
            if current_output_video_val and "partial_cancelled" in current_output_video_val:
                logger.info(f"Partial video already set from generator: {current_output_video_val}")
                current_status_text_val = f"⚠️ Processing cancelled by user. Partial video saved: {os.path.basename(current_output_video_val)}"
            else:
                # Check if a partial video was generated during cancellation
                partial_video_found = False
                # Look for partial video in outputs directory
                import glob
                partial_pattern = os.path.join(APP_CONFIG.paths.outputs_dir, "*_partial_cancelled.mp4")
                partial_files = glob.glob(partial_pattern)
                logger.info(f"Searching for partial videos with pattern: {partial_pattern}")
                logger.info(f"Found partial files: {partial_files}")
                if partial_files:
                    # Get the most recent partial file
                    latest_partial = max(partial_files, key=os.path.getctime)
                    current_output_video_val = latest_partial
                    partial_video_found = True
                    logger.info(f"Found partial video after cancellation: {latest_partial}")
                
                if partial_video_found:
                    current_status_text_val = f"⚠️ Processing cancelled by user. Partial video saved: {os.path.basename(current_output_video_val)}"
                else:
                    current_status_text_val = "❌ Processing cancelled by user."
            
            logger.info(f"Final cancellation yield - output video: {current_output_video_val}, status: {current_status_text_val}")
            yield (
                gr.update(value=current_output_video_val),
                gr.update(value=current_status_text_val),
                gr.update(value=current_user_prompt_val),
                gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                gr.update(value=current_last_chunk_video_val),
                gr.update(value=current_chunk_status_text_val),
                gr.update(value=current_comparison_video_val)
            )
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            current_status_text_val = f"❌ Error during processing: {str(e)}"
            yield (
                gr.update(value=current_output_video_val),
                gr.update(value=current_status_text_val),
                gr.update(value=current_user_prompt_val),
                gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                gr.update(value=current_last_chunk_video_val),
                gr.update(value=current_chunk_status_text_val),
                gr.update(value=current_comparison_video_val)
            )
        finally:
            # Reset cancellation state and ensure UI is ready for next operation
            cancellation_manager.reset()
            logger.info("Processing completed - UI state reset")

    click_inputs =[
        input_video, user_prompt, pos_prompt, neg_prompt, model_selector,
        upscale_factor_slider, cfg_slider, steps_slider, solver_mode_radio,
        max_chunk_len_slider, enable_chunk_optimization_check, vae_chunk_slider, color_fix_dropdown,
        enable_tiling_check, tile_size_num, tile_overlap_num,
        enable_context_window_check, context_overlap_num,
        enable_target_res_check, target_h_num, target_w_num, target_res_mode_radio,
        enable_auto_aspect_resolution_check, auto_resolution_status_display,
        ffmpeg_preset_dropdown, ffmpeg_quality_slider, ffmpeg_use_gpu_check,
        save_frames_checkbox, save_metadata_checkbox, save_chunks_checkbox, save_chunk_frames_checkbox,
        create_comparison_video_check,
        enable_scene_split_check, scene_split_mode_radio, scene_min_scene_len_num, scene_drop_short_check, scene_merge_last_check,
        scene_frame_skip_num, scene_threshold_num, scene_min_content_val_num, scene_frame_window_num,
        scene_copy_streams_check, scene_use_mkvmerge_check, scene_rate_factor_num, scene_preset_dropdown, scene_quiet_ffmpeg_check,
        scene_manual_split_type_radio, scene_manual_split_value_num,
        enable_fps_decrease, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps, fps_interpolation_method,
        enable_rife_interpolation, rife_multiplier, rife_fp16, rife_uhd, rife_scale,
        rife_skip_static, rife_enable_fps_limit, rife_max_fps_limit,
        rife_apply_to_chunks, rife_apply_to_scenes, rife_keep_original, rife_overwrite_original
    ]

    if UTIL_COG_VLM_AVAILABLE:
        click_inputs.extend([cogvlm_quant_radio, cogvlm_unload_radio, auto_caption_then_upscale_check])
    else:
        click_inputs.extend([gr.State(None), gr.State(None), gr.State(False)])

    click_inputs.extend([
        seed_num, random_seed_check,
        enable_image_upscaler_check, image_upscaler_model_dropdown, image_upscaler_batch_size_slider,
        enable_face_restoration_check, face_restoration_fidelity_slider, enable_face_colorization_check,
        face_restoration_when_radio, codeformer_model_dropdown, face_restoration_batch_size_slider,
        input_frames_folder, frame_folder_fps_slider,
        gpu_selector
    ])

    click_outputs_list =[output_video ,status_textbox ,user_prompt ]
    if UTIL_COG_VLM_AVAILABLE :
        click_outputs_list .append (caption_status )
    else :
        click_outputs_list .append (gr .State (None ))

    click_outputs_list .extend ([last_chunk_video ,chunk_status_text, comparison_video, cancel_button])

    def upscale_wrapper(*args):
        # Show cancel button when processing starts
        first_result = None
        processing_started = False
        last_valid_result = None
        
        try:
            for result in upscale_director_logic(build_app_config_from_ui(*args)):
                last_valid_result = result  # Keep track of the last valid result
                if not processing_started:
                    # First yield: enable cancel button and yield first result
                    processing_started = True
                    if isinstance(result, tuple):
                        yield result + (gr.update(interactive=True),)
                    else:
                        yield (result, gr.update(interactive=True))
                else:
                    # Subsequent yields: maintain cancel button enabled
                    if isinstance(result, tuple):
                        yield result + (gr.update(interactive=True),)
                    else:
                        yield (result, gr.update(interactive=True))
        except Exception as e:
            # Disable cancel button on any error
            logger.error(f"Error in upscale_wrapper: {e}")
            if processing_started and last_valid_result:
                # Try to preserve the last valid result when an error occurs
                if isinstance(last_valid_result, tuple):
                    yield last_valid_result + (gr.update(interactive=False),)
                else:
                    yield (last_valid_result, gr.update(interactive=False))
            else:
                yield (gr.update(), gr.update(value=f"Error: {e}"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False))
        finally:
            # Always disable cancel button when processing completes
            # Don't override the last valid result if we have one
            if not last_valid_result:
                yield (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False))

    upscale_button .click (
    fn =upscale_wrapper,
    inputs =click_inputs ,
    outputs =click_outputs_list ,
    show_progress_on =[output_video ]
    )

    # Cancel button click handler
    def cancel_processing():
        """Handle cancel button click - request cancellation via the global manager."""
        logger.warning("🚨 CANCEL button clicked by user. Requesting cancellation.")
        success = cancellation_manager.request_cancellation()
        
        if success:
            return "🚨 CANCELLATION REQUESTED: Processing will stop safely at the next checkpoint. Please wait for current operation to complete..."
        else:
            return "⚠️ No active processing to cancel or cancellation already requested."

    cancel_button.click(
        fn=cancel_processing,
        inputs=[],
        outputs=[status_textbox]
    )

    if UTIL_COG_VLM_AVAILABLE :
        def auto_caption_wrapper (vid ,quant_display ,unload_strat ,progress =gr .Progress (track_tqdm =True )):
            caption_text ,caption_stat_msg =util_auto_caption (
            vid ,
            get_quant_value_from_display (quant_display ),
            unload_strat ,
            APP_CONFIG.paths.cogvlm_model_path ,
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
            output_dir =APP_CONFIG.paths.outputs_dir ,
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

    def process_batch_videos_wrapper (app_config: AppConfig, progress =gr .Progress (track_tqdm =True )):
        actual_seed_for_batch = app_config.seed.seed
        if app_config.seed.use_random:
            actual_seed_for_batch = np.random.randint(0, 2**31)
            logger.info(f"Batch: Random seed checked. Using generated seed: {actual_seed_for_batch}")
        elif app_config.seed.seed == -1:
            actual_seed_for_batch = np.random.randint(0, 2**31)
            logger.info(f"Batch: Seed input is -1. Using generated seed: {actual_seed_for_batch}")
        else:
            logger.info(f"Batch: Using provided seed: {actual_seed_for_batch}")
        
        app_config.seed.seed = actual_seed_for_batch

        partial_run_upscale_for_batch =partial (core_run_upscale ,
            logger =logger ,
            app_config_module=None, # Deprecated
            metadata_handler_module =metadata_handler ,
            VideoToVideo_sr_class =VideoToVideo_sr ,
            setup_seed_func =setup_seed ,
            EasyDict_class =EasyDict ,
            preprocess_func =preprocess ,
            collate_fn_func =collate_fn ,
            tensor2vid_func =tensor2vid ,
            ImageSpliterTh_class =ImageSpliterTh ,
            adain_color_fix_func =adain_color_fix ,
            wavelet_color_fix_func =wavelet_color_fix
        )

        actual_batch_input_folder = app_config.batch.input_folder
        temp_video_conversions = []
        
        if app_config.batch.enable_frame_folders:
            logger.info("Batch frame folder processing mode enabled")
            
            frame_folders = util_find_frame_folders_in_directory(app_config.batch.input_folder, logger)
            
            if not frame_folders:
                return None, f"❌ No frame folders found in: {app_config.batch.input_folder}"
            
            temp_batch_dir = tempfile.mkdtemp(prefix="batch_frame_folders_")
            actual_batch_input_folder = temp_batch_dir
            
            logger.info(f"Found {len(frame_folders)} frame folders to convert")
            
            for i, frame_folder in enumerate(frame_folders):
                folder_name = os.path.basename(frame_folder.rstrip(os.sep))
                temp_video_path = os.path.join(temp_batch_dir, f"{folder_name}.mp4")
                
                logger.info(f"Converting frame folder {i+1}/{len(frame_folders)}: {folder_name}")
                
                success, conv_msg = util_process_frame_folder_to_video(
                    frame_folder, temp_video_path, fps=app_config.frame_folder.fps,
                    ffmpeg_preset=app_config.ffmpeg.preset,
                    ffmpeg_quality_value=app_config.ffmpeg.quality,
                    ffmpeg_use_gpu=app_config.ffmpeg.use_gpu,
                    logger=logger
                )
                
                if success:
                    temp_video_conversions.append(temp_video_path)
                    logger.info(f"✅ Converted {folder_name}: {conv_msg}")
                else:
                    logger.error(f"❌ Failed to convert {folder_name}: {conv_msg}")
            
            if not temp_video_conversions:
                return None, f"❌ Failed to convert any frame folders to videos"
            
            logger.info(f"Successfully converted {len(temp_video_conversions)} frame folders to videos")
            app_config.batch.input_folder = actual_batch_input_folder

        from logic.batch_operations import process_batch_videos_from_app_config
        return process_batch_videos_from_app_config(
            app_config=app_config,
            run_upscale_func=partial_run_upscale_for_batch,
            logger=logger,
            progress=progress
        )

    def build_batch_app_config_from_ui(*args):
        # Build AppConfig with all the needed parameters for batch processing
        (
            input_video_val, user_prompt_val, pos_prompt_val, neg_prompt_val, model_selector_val,
            upscale_factor_slider_val, cfg_slider_val, steps_slider_val, solver_mode_radio_val,
            max_chunk_len_slider_val, enable_chunk_optimization_check_val, vae_chunk_slider_val, color_fix_dropdown_val,
            enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
            enable_context_window_check_val, context_overlap_num_val,
            enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
            enable_auto_aspect_resolution_check_val, auto_resolution_status_display_val,
            ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
            save_frames_checkbox_val, save_metadata_checkbox_val, save_chunks_checkbox_val, save_chunk_frames_checkbox_val,
            create_comparison_video_check_val,
            enable_scene_split_check_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
            scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
            scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
            scene_manual_split_type_radio_val, scene_manual_split_value_num_val,
            enable_fps_decrease_val, fps_decrease_mode_val, fps_multiplier_preset_val, fps_multiplier_custom_val, target_fps_val, fps_interpolation_method_val,
            enable_rife_interpolation_val, rife_multiplier_val, rife_fp16_val, rife_uhd_val, rife_scale_val,
            rife_skip_static_val, rife_enable_fps_limit_val, rife_max_fps_limit_val,
            rife_apply_to_chunks_val, rife_apply_to_scenes_val, rife_keep_original_val, rife_overwrite_original_val,
            cogvlm_quant_radio_val, cogvlm_unload_radio_val, do_auto_caption_first_val,
            seed_num_val, random_seed_check_val,
            enable_image_upscaler_val, image_upscaler_model_val, image_upscaler_batch_size_val,
            enable_face_restoration_val, face_restoration_fidelity_val, enable_face_colorization_val,
            face_restoration_when_val, codeformer_model_val, face_restoration_batch_size_val,
            input_frames_folder_val, frame_folder_fps_slider_val,
            gpu_selector_val,
            batch_input_folder_val, batch_output_folder_val, enable_batch_frame_folders_val,
            batch_skip_existing_val, batch_use_prompt_files_val, batch_save_captions_val, batch_enable_auto_caption_val
        ) = args
        
        # Auto-detect if frame folder processing should be enabled based on input path
        frame_folder_enable = False
        if input_frames_folder_val and input_frames_folder_val.strip():
            from logic.frame_folder_utils import detect_input_type
            input_type, _, _ = detect_input_type(input_frames_folder_val, logger)
            frame_folder_enable = (input_type == "frames_folder")
        
        config = AppConfig(
            input_video_path=input_video_val,
            paths=APP_CONFIG.paths,
            prompts=PromptConfig(
                user=user_prompt_val,
                positive=pos_prompt_val,
                negative=neg_prompt_val
            ),
            star_model=StarModelConfig(
                model_choice=model_selector_val,
                cfg_scale=cfg_slider_val,
                solver_mode=solver_mode_radio_val,
                steps=steps_slider_val,
                color_fix_method=color_fix_dropdown_val
            ),
            performance=PerformanceConfig(
                max_chunk_len=max_chunk_len_slider_val,
                vae_chunk=vae_chunk_slider_val,
                enable_chunk_optimization=enable_chunk_optimization_check_val
            ),
            resolution=ResolutionConfig(
                enable_target_res=enable_target_res_check_val,
                target_res_mode=target_res_mode_radio_val,
                target_h=target_h_num_val,
                target_w=target_w_num_val,
                upscale_factor=upscale_factor_slider_val,
                enable_auto_aspect_resolution=enable_auto_aspect_resolution_check_val,
                auto_resolution_status=auto_resolution_status_display_val
            ),
            context_window=ContextWindowConfig(
                enable=enable_context_window_check_val,
                overlap=context_overlap_num_val
            ),
            tiling=TilingConfig(
                enable=enable_tiling_check_val,
                tile_size=tile_size_num_val,
                tile_overlap=tile_overlap_num_val
            ),
            ffmpeg=FfmpegConfig(
                use_gpu=ffmpeg_use_gpu_check_val,
                preset=ffmpeg_preset_dropdown_val,
                quality=ffmpeg_quality_slider_val
            ),
            frame_folder=FrameFolderConfig(
                enable=frame_folder_enable,
                input_path=input_frames_folder_val,
                fps=frame_folder_fps_slider_val
            ),
            scene_split=SceneSplitConfig(
                enable=enable_scene_split_check_val,
                mode=scene_split_mode_radio_val,
                min_scene_len=scene_min_scene_len_num_val,
                threshold=scene_threshold_num_val,
                drop_short=scene_drop_short_check_val,
                merge_last=scene_merge_last_check_val,
                frame_skip=scene_frame_skip_num_val,
                min_content_val=scene_min_content_val_num_val,
                frame_window=scene_frame_window_num_val,
                manual_split_type=scene_manual_split_type_radio_val,
                manual_split_value=scene_manual_split_value_num_val,
                copy_streams=scene_copy_streams_check_val,
                use_mkvmerge=scene_use_mkvmerge_check_val,
                rate_factor=scene_rate_factor_num_val,
                encoding_preset=scene_preset_dropdown_val,
                quiet_ffmpeg=scene_quiet_ffmpeg_check_val
            ),
            cogvlm=CogVLMConfig(
                quant_display=cogvlm_quant_radio_val,
                unload_after_use=cogvlm_unload_radio_val,
                auto_caption_then_upscale=do_auto_caption_first_val,
                quant_value=get_quant_value_from_display(cogvlm_quant_radio_val)
            ),
            outputs=OutputConfig(
                save_frames=save_frames_checkbox_val,
                save_metadata=save_metadata_checkbox_val,
                save_chunks=save_chunks_checkbox_val,
                save_chunk_frames=save_chunk_frames_checkbox_val,
                create_comparison_video=create_comparison_video_check_val
            ),
            seed=SeedConfig(
                seed=seed_num_val,
                use_random=random_seed_check_val
            ),
            rife=RifeConfig(
                enable=enable_rife_interpolation_val,
                multiplier=rife_multiplier_val,
                fp16=rife_fp16_val,
                uhd=rife_uhd_val,
                scale=rife_scale_val,
                skip_static=rife_skip_static_val,
                enable_fps_limit=rife_enable_fps_limit_val,
                max_fps_limit=rife_max_fps_limit_val,
                apply_to_chunks=rife_apply_to_chunks_val,
                apply_to_scenes=rife_apply_to_scenes_val,
                keep_original=rife_keep_original_val,
                overwrite_original=rife_overwrite_original_val
            ),
            fps_decrease=FpsDecreaseConfig(
                enable=enable_fps_decrease_val,
                mode=fps_decrease_mode_val,
                multiplier_preset=fps_multiplier_preset_val,
                multiplier_custom=fps_multiplier_custom_val,
                target_fps=target_fps_val,
                interpolation_method=fps_interpolation_method_val
            ),
            batch=BatchConfig(
                input_folder=batch_input_folder_val,
                output_folder=batch_output_folder_val,
                skip_existing=batch_skip_existing_val,
                save_captions=batch_save_captions_val,
                use_prompt_files=batch_use_prompt_files_val,
                enable_auto_caption=batch_enable_auto_caption_val,
                enable_frame_folders=enable_batch_frame_folders_val
            ),
            image_upscaler=ImageUpscalerConfig(
                enable=enable_image_upscaler_val,
                model=image_upscaler_model_val,
                batch_size=image_upscaler_batch_size_val
            ),
            face_restoration=FaceRestorationConfig(
                enable=enable_face_restoration_val,
                fidelity_weight=face_restoration_fidelity_val,
                enable_colorization=enable_face_colorization_val,
                when=face_restoration_when_val,
                model=extract_codeformer_model_path_from_dropdown(codeformer_model_val),
                batch_size=face_restoration_batch_size_val
            ),
            gpu=GpuConfig(
                device=str(extract_gpu_index_from_dropdown(gpu_selector_val))
            )
        )
        return config

    def batch_wrapper(*args):
        result = process_batch_videos_wrapper(build_batch_app_config_from_ui(*args))
        return result

    # Create batch inputs list that includes batch-specific parameters
    batch_click_inputs = click_inputs + [
        batch_input_folder, batch_output_folder, enable_batch_frame_folders,
        batch_skip_existing, batch_use_prompt_files, batch_save_captions, batch_enable_auto_caption
    ]

    batch_process_button .click (
    fn =batch_wrapper,
    inputs =batch_click_inputs,
    outputs =[output_video ,status_textbox ],
    show_progress_on =[output_video ]
    )

    gpu_selector .change (
    fn =lambda gpu_id :util_set_gpu_device (gpu_id ,logger =logger ),
    inputs =gpu_selector ,
    outputs =status_textbox 
    )

    def standalone_face_restoration_wrapper(
        input_video_val,
        face_restoration_mode_val,
        batch_input_folder_val,
        batch_output_folder_val,
        enable_face_restoration_val,
        face_restoration_fidelity_val,
        enable_face_colorization_val,
        codeformer_model_val,
        face_restoration_batch_size_val,
        save_frames_val,
        create_comparison_val,
        preserve_audio_val,
        seed_num_val=99,
        random_seed_check_val=False,
        progress=gr.Progress(track_tqdm=True)
    ):
        try:
            from logic.face_restoration_utils import restore_video_frames, scan_codeformer_models
            import os
            import time
            import random
            
            if random_seed_check_val:
                actual_seed = random.randint(0, 2**32 - 1)
                logger.info(f"Generated random seed: {actual_seed}")
            else:
                actual_seed = seed_num_val if seed_num_val >= 0 else 99
            
            if not enable_face_restoration_val:
                return None, None, "⚠️ Face restoration is disabled. Please enable it to process videos.", "❌ Processing disabled"
            
            if face_restoration_mode_val == "Single Video":
                if not input_video_val:
                    return None, None, "⚠️ Please upload a video file to process.", "❌ No input video"
                input_path = input_video_val
                output_dir = APP_CONFIG.paths.outputs_dir
                processing_mode = "single"
            else:
                if not batch_input_folder_val or not batch_output_folder_val:
                    return None, None, "⚠️ Please specify both input and output folder paths for batch processing.", "❌ Missing folder paths"
                if not os.path.exists(batch_input_folder_val):
                    return None, None, f"⚠️ Input folder does not exist: {batch_input_folder_val}", "❌ Input folder not found"
                input_path = batch_input_folder_val
                output_dir = batch_output_folder_val
                processing_mode = "batch"
            
            actual_model_path = extract_codeformer_model_path_from_dropdown(codeformer_model_val)
            
            progress(0.0, "🎭 Starting face restoration processing...")
            start_time = time.time()
            
            if processing_mode == "single":
                progress(0.1, "📹 Processing single video...")
                
                def progress_callback(current_progress, status_msg):
                    mapped_progress = 0.1 + (current_progress * 0.8)
                    progress(mapped_progress, f"🎭 {status_msg}")
                
                result = restore_video_frames(
                    video_path=input_path,
                    output_dir=output_dir,
                    fidelity_weight=face_restoration_fidelity_val,
                    enable_colorization=enable_face_colorization_val,
                    model_path=actual_model_path,
                    batch_size=face_restoration_batch_size_val,
                    save_frames=save_frames_val,
                    create_comparison=create_comparison_val,
                    preserve_audio=preserve_audio_val,
                    ffmpeg_preset=ffmpeg_preset_dropdown.value if 'ffmpeg_preset_dropdown' in globals() else "medium",
                    ffmpeg_quality=ffmpeg_quality_slider.value if 'ffmpeg_quality_slider' in globals() else 23,
                    ffmpeg_use_gpu=ffmpeg_use_gpu_check.value if 'ffmpeg_use_gpu_check' in globals() else False,
                    progress_callback=progress_callback,
                    logger=logger
                )
                
                if result['success']:
                    output_video = result['output_video_path']
                    comparison_video = result.get('comparison_video_path', None)
                    
                    processing_time = time.time() - start_time
                    stats_msg = f"""📊 Processing Complete!
⏱️ Total Time: {processing_time:.1f} seconds
🎬 Input: {os.path.basename(input_path)}
📁 Output: {os.path.basename(output_video) if output_video else 'N/A'}
🎯 Fidelity: {face_restoration_fidelity_val}
🔧 Batch Size: {face_restoration_batch_size_val}
✅ Status: Success"""
                    
                    progress(1.0, "✅ Face restoration completed successfully!")
                    return output_video, comparison_video, result['message'], stats_msg
                else:
                    return None, None, f"❌ Processing failed: {result['message']}", "❌ Processing failed"
            
            else:
                progress(0.1, "📁 Starting batch face restoration...")
                
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
                video_files = []
                for file in os.listdir(input_path):
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(input_path, file))
                
                if not video_files:
                    return None, None, f"⚠️ No video files found in input folder: {input_path}", "❌ No videos found"
                
                processed_count = 0
                failed_count = 0
                total_files = len(video_files)
                
                for i, video_file in enumerate(video_files):
                    video_name = os.path.basename(video_file)
                    file_progress = i / total_files
                    base_progress = 0.1 + (file_progress * 0.8)
                    
                    progress(base_progress, f"🎭 Processing {video_name} ({i+1}/{total_files})...")
                    
                    def batch_progress_callback(current_progress, status_msg):
                        file_progress_range = 0.8 / total_files
                        mapped_progress = base_progress + (current_progress * file_progress_range)
                        progress(mapped_progress, f"🎭 [{i+1}/{total_files}] {video_name}: {status_msg}")
                    
                    result = restore_video_frames(
                        video_path=video_file,
                        output_dir=output_dir,
                        fidelity_weight=face_restoration_fidelity_val,
                        enable_colorization=enable_face_colorization_val,
                        model_path=actual_model_path,
                        batch_size=face_restoration_batch_size_val,
                        save_frames=save_frames_val,
                        create_comparison=create_comparison_val,
                        preserve_audio=preserve_audio_val,
                        ffmpeg_preset=ffmpeg_preset_dropdown.value if 'ffmpeg_preset_dropdown' in globals() else "medium",
                        ffmpeg_quality=ffmpeg_quality_slider.value if 'ffmpeg_quality_slider' in globals() else 23,
                        ffmpeg_use_gpu=ffmpeg_use_gpu_check.value if 'ffmpeg_use_gpu_check' in globals() else False,
                        progress_callback=batch_progress_callback,
                        logger=logger
                    )
                    
                    if result['success']:
                        processed_count += 1
                        logger.info(f"Successfully processed: {video_name}")
                    else:
                        failed_count += 1
                        logger.error(f"Failed to process {video_name}: {result['message']}")
                
                processing_time = time.time() - start_time
                stats_msg = f"""📊 Batch Processing Complete!
⏱️ Total Time: {processing_time:.1f} seconds
📁 Input Folder: {os.path.basename(input_path)}
📁 Output Folder: {os.path.basename(output_dir)}
✅ Processed: {processed_count} videos
❌ Failed: {failed_count} videos
🎯 Fidelity: {face_restoration_fidelity_val}
🔧 Batch Size: {face_restoration_batch_size_val}"""
                
                if processed_count > 0:
                    progress(1.0, f"✅ Batch processing completed! {processed_count}/{total_files} videos processed successfully.")
                    status_msg = f"✅ Batch processing completed!\n\n📊 Results:\n✅ Successfully processed: {processed_count} videos\n❌ Failed: {failed_count} videos\n📁 Output saved to: {output_dir}\n\n⏱️ Total processing time: {processing_time:.1f} seconds"
                    return None, None, status_msg, stats_msg
                else:
                    return None, None, f"❌ Batch processing failed: No videos were processed successfully.", "❌ Batch processing failed"
        
        except Exception as e:
            logger.error(f"Standalone face restoration error: {str(e)}")
            return None, None, f"❌ Error during face restoration: {str(e)}", "❌ Processing error"

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
            output_dir =APP_CONFIG.paths.outputs_dir ,
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

    # --- AUTO-RESOLUTION FUNCTIONS ---
    def calculate_auto_resolution(video_path, enable_auto_aspect_resolution, target_h, target_w):
        """
        Calculate optimal resolution maintaining aspect ratio within pixel budget.
        
        Args:
            video_path: Path to the input video
            enable_auto_aspect_resolution: Whether auto-resolution is enabled
            target_h: Current target height setting (pixel budget height)
            target_w: Current target width setting (pixel budget width)
            
        Returns:
            tuple: (new_target_h, new_target_w, status_message)
        """
        if not enable_auto_aspect_resolution:
            return target_h, target_w, "Auto-resolution disabled"
        
        if video_path is None:
            return target_h, target_w, "No video loaded"
        
        try:
            from logic.auto_resolution_utils import update_resolution_from_video
            
            # Calculate pixel budget from current target resolution
            pixel_budget = target_h * target_w
            
            # Get updated resolution maintaining aspect ratio
            result = update_resolution_from_video(
                video_path=video_path,
                pixel_budget=pixel_budget,
                logger=logger
            )
            
            if result['success']:
                new_h = result['optimal_height']
                new_w = result['optimal_width']
                status_msg = result['status_message']
                
                logger.info(f"Auto-resolution: {video_path} -> {new_w}x{new_h} (was {target_w}x{target_h})")
                logger.info(f"Auto-resolution status: {status_msg}")
                
                return new_h, new_w, status_msg
            else:
                error_msg = f"Auto-resolution calculation failed: {result.get('error', 'Unknown error')}"
                logger.warning(error_msg)
                return target_h, target_w, error_msg
                
        except Exception as e:
            error_msg = f"Auto-resolution error: {str(e)}"
            logger.error(f"Exception in calculate_auto_resolution: {e}")
            return target_h, target_w, error_msg

    def handle_video_change_with_auto_resolution(
        video_path, 
        enable_auto_aspect_resolution, 
        target_h, 
        target_w
    ):
        """
        Handle video change event with auto-resolution calculation and video info display.
        
        Returns:
            tuple: (status_message, new_target_h, new_target_w, auto_resolution_status)
        """
        # Always display video info regardless of auto-resolution setting
        if video_path is None:
            return "", target_h, target_w, "No video loaded"
        
        # Get basic video info for status display
        try:
            video_info = util_get_video_info(video_path, logger)
            if video_info:
                filename = os.path.basename(video_path) if video_path else None
                info_message = util_format_video_info_message(video_info, filename)
                
                logger.info(f"Video uploaded: {filename}")
                logger.info(f"Video details: {video_info['frames']} frames, {video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']}")
            else:
                info_message = "❌ Could not read video information"
                logger.warning(f"Failed to get video info for: {video_path}")
        except Exception as e:
            info_message = f"❌ Error reading video: {str(e)}"
            logger.error(f"Exception in video info: {e}")
        
        # Calculate auto-resolution if enabled
        new_target_h, new_target_w, auto_status = calculate_auto_resolution(
            video_path, enable_auto_aspect_resolution, target_h, target_w
        )
        
        return info_message, new_target_h, new_target_w, auto_status

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



    input_video.change(
        fn=handle_video_change_with_auto_resolution,
        inputs=[input_video, enable_auto_aspect_resolution_check, target_h_num, target_w_num],
        outputs=[status_textbox, target_h_num, target_w_num, auto_resolution_status_display]
    )

    for component in [input_video ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ]:
        component .change (
        fn =calculate_fps_preview ,
        inputs =[input_video ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ],
        outputs =fps_calculation_info 
        )

    def clear_temp_folder_wrapper(progress: gr.Progress = gr.Progress(track_tqdm=True)):
        logger.info("Temp cleanup requested via UI button")
        before_bytes = util_calculate_temp_folder_size(logger)
        success = util_clear_temp_folder(logger)
        after_bytes = util_calculate_temp_folder_size(logger)
        freed_bytes = max(before_bytes - after_bytes, 0)

        logger.info(
            f"Temp cleanup completed. Freed {freed_bytes / (1024**3):.2f} GB (Remaining: {after_bytes / (1024**3):.2f} GB)"
        )

        after_label = util_format_temp_folder_size(logger)
        status_message = (
            f"✅ Temp folder cleared. Freed {freed_bytes / (1024**3):.2f} GB. Remaining: {after_label}"
            if success
            else "⚠️ Temp folder cleanup encountered errors. Check logs."
        )

        return gr.update(value=status_message), gr.update(value=f"Delete Temp Folder ({after_label})")

    delete_temp_button.click(
        fn=clear_temp_folder_wrapper,
        inputs=[],
        outputs=[status_textbox, delete_temp_button],
        show_progress_on=[status_textbox]
    )
    
    def update_cutting_mode_controls(cutting_mode_val):
        if cutting_mode_val == "time_ranges":
            return [
                gr.update(visible=True),
                gr.update(visible=False)
            ]
        else:
            return [
                gr.update(visible=False),
                gr.update(visible=True)
            ]
    
    def display_detailed_video_info_edit(video_path):
        if video_path is None:
            return "📹 Upload a video to see detailed information"
        
        try:
            video_info = util_get_video_detailed_info(video_path, logger)
            if video_info:
                formatted_info = util_format_video_info_for_display(video_info)
                logger.info(f"Video editor: Video uploaded - {video_info.get('filename', 'Unknown')}")
                return formatted_info
            else:
                error_msg = "❌ Could not read video information"
                logger.warning(f"Video editor: Failed to get video info for: {video_path}")
                return error_msg
                
        except Exception as e:
            error_msg = f"❌ Error reading video: {str(e)}"
            logger.error(f"Video editor: Exception in video info: {e}")
            return error_msg
    
    def validate_and_analyze_cuts(ranges_input, cutting_mode_val, video_path):
        if video_path is None:
            return "✏️ Upload a video first", "📊 Upload video and enter ranges to see time estimate"
        
        if not ranges_input or not ranges_input.strip():
            return "✏️ Enter ranges above to see cut analysis", "📊 Upload video and enter ranges to see time estimate"
        
        try:
            video_info = util_get_video_detailed_info(video_path, logger)
            if not video_info:
                return "❌ Could not read video information", "❌ Cannot estimate without video info"
            
            if cutting_mode_val == "time_ranges":
                duration = video_info.get("accurate_duration", video_info.get("duration", 0))
                ranges = util_parse_time_ranges(ranges_input, duration)
                max_value = duration
                range_type = "time"
            else:
                total_frames = video_info.get("total_frames", 0)
                ranges = util_parse_frame_ranges(ranges_input, total_frames)
                max_value = total_frames
                range_type = "frame"
            
            validation_result = util_validate_ranges(ranges, max_value, range_type)
            
            ffmpeg_settings = {
                "use_gpu": ffmpeg_use_gpu_check.value if 'ffmpeg_use_gpu_check' in globals() else False,
                "preset": ffmpeg_preset_dropdown.value if 'ffmpeg_preset_dropdown' in globals() else "medium",
                "quality": ffmpeg_quality_slider.value if 'ffmpeg_quality_slider' in globals() else 23
            }
            
            time_estimate = util_estimate_processing_time(ranges, video_info, ffmpeg_settings)
            
            return validation_result["analysis_text"], time_estimate.get("time_estimate_text", "Could not estimate time")
            
        except ValueError as e:
            return f"❌ Validation Error: {str(e)}", "❌ Cannot estimate due to validation error"
        except Exception as e:
            logger.error(f"Video editor: Error in validation: {e}")
            return f"❌ Error: {str(e)}", "❌ Error during analysis"
    
    def get_current_ffmpeg_settings():
        try:
            return {
                "use_gpu": ffmpeg_use_gpu_check.value if 'ffmpeg_use_gpu_check' in globals() else False,
                "preset": ffmpeg_preset_dropdown.value if 'ffmpeg_preset_dropdown' in globals() else "medium", 
                "quality": ffmpeg_quality_slider.value if 'ffmpeg_quality_slider' in globals() else 23
            }
        except:
            return {
                "use_gpu": False,
                "preset": "medium", 
                "quality": 23
            }
    
    def cut_video_wrapper(
        input_video_val, ranges_input, cutting_mode_val, precise_cutting_mode_val, 
        preview_first_segment_val, seed_num_val=99, random_seed_check_val=False, 
        progress=gr.Progress(track_tqdm=True)
    ):
        if input_video_val is None:
            return None, None, "❌ Please upload a video first"
        
        if not ranges_input or not ranges_input.strip():
            return None, None, "❌ Please enter cut ranges"
        
        try:
            progress(0, desc="Analyzing video and ranges...")
            
            video_info = util_get_video_detailed_info(input_video_val, logger)
            if not video_info:
                return None, None, "❌ Could not read video information"
            
            if cutting_mode_val == "time_ranges":
                duration = video_info.get("accurate_duration", video_info.get("duration", 0))
                ranges = util_parse_time_ranges(ranges_input, duration)
                range_type = "time"
                max_value = duration
            else:
                total_frames = video_info.get("total_frames", 0)
                ranges = util_parse_frame_ranges(ranges_input, total_frames)
                range_type = "frame"
                max_value = total_frames
            
            validation_result = util_validate_ranges(ranges, max_value, range_type)
            logger.info(f"Video editor: Validated {len(ranges)} ranges: {validation_result['analysis_text']}")
            
            ffmpeg_settings = get_current_ffmpeg_settings()
            
            if precise_cutting_mode_val == "fast":
                ffmpeg_settings["stream_copy"] = True
            
            progress(0.1, desc="Starting video cutting...")
            
            result = util_cut_video_segments(
                video_path=input_video_val,
                ranges=ranges,
                range_type=range_type,
                output_dir=APP_CONFIG.paths.outputs_dir,
                ffmpeg_settings=ffmpeg_settings,
                logger=logger,
                progress=progress,
                seed=seed_num_val if not random_seed_check_val else np.random.randint(0, 2**31)
            )
            
            if not result["success"]:
                return None, None, f"❌ Cutting failed: {result.get('error', 'Unknown error')}"
            
            preview_path = None
            if preview_first_segment_val and ranges:
                progress(0.95, desc="Creating preview...")
                preview_path = util_create_preview_segment(
                    video_path=input_video_val,
                    first_range=ranges[0],
                    range_type=range_type,
                    output_dir=APP_CONFIG.paths.outputs_dir,
                    ffmpeg_settings=ffmpeg_settings,
                    logger=logger
                )
            
            progress(1.0, desc="Video cutting completed!")
            
            status_msg = f"""✅ Video cutting completed successfully!
            
📁 Output: {result['final_output']}
📊 Processed: {len(ranges)} segments
📝 Session: {result['session_dir']}
💾 Metadata: {result['metadata_path']}

{validation_result['analysis_text']}"""
            
            return result["final_output"], preview_path, status_msg
            
        except ValueError as e:
            error_msg = f"❌ Input Error: {str(e)}"
            logger.error(f"Video editor: {error_msg}")
            return None, None, error_msg
        except Exception as e:
            error_msg = f"❌ Processing Error: {str(e)}"
            logger.error(f"Video editor: {error_msg}", exc_info=True)
            return None, None, error_msg
    
    def cut_and_move_to_upscale(
        input_video_val, ranges_input, cutting_mode_val, precise_cutting_mode_val, 
        preview_first_segment_val, seed_num_val=99, random_seed_check_val=False, 
        progress=gr.Progress(track_tqdm=True)
    ):
        final_output, preview_path, status_msg = cut_video_wrapper(
            input_video_val, ranges_input, cutting_mode_val, precise_cutting_mode_val,
            preview_first_segment_val, seed_num_val, random_seed_check_val, progress
        )
        
        if final_output is None:
            return None, None, status_msg, gr.update(), gr.update(visible=False)
        
        integration_msg = f"""🎬✅ Video cutting completed successfully!

📁 Cut Video: {os.path.basename(final_output)}
📍 Ready for upscaling in Main tab

💡 Next Steps:
1. Switch to the 'Main' tab
2. The cut video has been automatically loaded
3. Configure upscaling settings as needed
4. Click 'Upscale Video' to begin processing

{status_msg}"""
        
        logger.info(f"Video editor: Cut completed, updating main tab input: {final_output}")
        
        return (
            final_output,
            preview_path,
            integration_msg,
            gr.update(value=final_output),
            gr.update(visible=True, value=f"✅ Cut video loaded from Edit Videos tab: {os.path.basename(final_output)}")
        )
    
    def enhanced_cut_and_move_to_upscale(
        input_video_val, time_ranges_val, frame_ranges_val, cutting_mode_val, 
        precise_cutting_mode_val, preview_first_segment_val, seed_num_val=99, 
        random_seed_check_val=False, progress=gr.Progress(track_tqdm=True)
    ):
        ranges_input = time_ranges_val if cutting_mode_val == "time_ranges" else frame_ranges_val
        
        return cut_and_move_to_upscale(
            input_video_val, ranges_input, cutting_mode_val, 
            precise_cutting_mode_val, preview_first_segment_val, 
            seed_num_val, random_seed_check_val, progress
        )

    cutting_mode.change(
        fn=update_cutting_mode_controls,
        inputs=cutting_mode,
        outputs=[time_range_controls, frame_range_controls]
    )
    
    input_video_edit.change(
        fn=display_detailed_video_info_edit,
        inputs=input_video_edit,
        outputs=video_info_display
    )
    
    def validate_ranges_wrapper(time_ranges_val, frame_ranges_val, cutting_mode_val, video_path):
        ranges_input = time_ranges_val if cutting_mode_val == "time_ranges" else frame_ranges_val
        return validate_and_analyze_cuts(ranges_input, cutting_mode_val, video_path)
    
    for component in [time_ranges_input, frame_ranges_input, cutting_mode]:
        component.change(
            fn=validate_ranges_wrapper,
            inputs=[time_ranges_input, frame_ranges_input, cutting_mode, input_video_edit],
            outputs=[cut_info_display, processing_estimate]
        )
    
    cut_and_save_btn.click(
        fn=lambda input_video_val, time_ranges_val, frame_ranges_val, cutting_mode_val, precise_cutting_mode_val, preview_first_segment_val, seed_num_val, random_seed_check_val: cut_video_wrapper(
            input_video_val, 
            time_ranges_val if cutting_mode_val == "time_ranges" else frame_ranges_val,
            cutting_mode_val, 
            precise_cutting_mode_val, 
            preview_first_segment_val,
            seed_num_val,
            random_seed_check_val
        ),
        inputs=[
            input_video_edit, time_ranges_input, frame_ranges_input, cutting_mode, 
            precise_cutting_mode, preview_first_segment, seed_num, random_seed_check
        ],
        outputs=[output_video_edit, preview_video_edit, edit_status_textbox],
        show_progress_on=[output_video_edit]
    )
    
    cut_and_upscale_btn.click(
        fn=enhanced_cut_and_move_to_upscale,
        inputs=[
            input_video_edit, time_ranges_input, frame_ranges_input, cutting_mode, 
            precise_cutting_mode, preview_first_segment, seed_num, random_seed_check
        ],
        outputs=[output_video_edit, preview_video_edit, edit_status_textbox, input_video, integration_status],
        show_progress_on=[output_video_edit]
    )
    
    def handle_main_input_change(video_path):
        if video_path is None:
            return gr.update(visible=False, value="")
        else:
            if video_path and "logic/" in video_path:
                return gr.update(visible=True, value=f"✅ Cut video from Edit Videos tab: {os.path.basename(video_path)}")
            else:
                return gr.update(visible=False, value="")
    
    input_video.change(
        fn=handle_main_input_change,
        inputs=input_video,
        outputs=integration_status
    )
    
    face_restoration_process_btn.click(
        fn=standalone_face_restoration_wrapper,
        inputs=[
            input_video_face_restoration,
            face_restoration_mode,
            batch_input_folder_face,
            batch_output_folder_face,
            standalone_enable_face_restoration,
            standalone_face_restoration_fidelity,
            standalone_enable_face_colorization,
            standalone_codeformer_model_dropdown,
            standalone_face_restoration_batch_size,
            standalone_save_frames,
            standalone_create_comparison,
            standalone_preserve_audio,
            seed_num,
            random_seed_check
        ],
        outputs=[
            output_video_face_restoration,
            comparison_video_face_restoration,
            face_restoration_status,
            face_restoration_stats
        ],
        show_progress_on=[output_video_face_restoration]
    )

    # --- PRESET LOGIC ---

    # This list defines all UI components that are part of a preset.
    # The order is critical and must be maintained for both saving and loading.
    # IMPORTANT: This order must exactly match the order in click_inputs for preset saving/loading to work!
    
    # Create preset_components by copying click_inputs and excluding input_video (first component)
    # This ensures exact order matching between save and load operations
    preset_components = click_inputs[1:]  # Skip input_video which is at index 0

    # Define preset helper functions before they are used

    # Map component objects to their location in the AppConfig structure for robust loading.
    component_key_map = {
        user_prompt: ('prompts', 'user'), pos_prompt: ('prompts', 'positive'), neg_prompt: ('prompts', 'negative'),
        model_selector: ('star_model', 'model_choice'), cfg_slider: ('star_model', 'cfg_scale'), steps_slider: ('star_model', 'steps'), solver_mode_radio: ('star_model', 'solver_mode'), color_fix_dropdown: ('star_model', 'color_fix_method'),
        max_chunk_len_slider: ('performance', 'max_chunk_len'), enable_chunk_optimization_check: ('performance', 'enable_chunk_optimization'), vae_chunk_slider: ('performance', 'vae_chunk'),
        enable_target_res_check: ('resolution', 'enable_target_res'), target_h_num: ('resolution', 'target_h'), target_w_num: ('resolution', 'target_w'), target_res_mode_radio: ('resolution', 'target_res_mode'), upscale_factor_slider: ('resolution', 'upscale_factor'),
        enable_auto_aspect_resolution_check: ('resolution', 'enable_auto_aspect_resolution'), auto_resolution_status_display: ('resolution', 'auto_resolution_status'),
        enable_context_window_check: ('context_window', 'enable'), context_overlap_num: ('context_window', 'overlap'),
        enable_tiling_check: ('tiling', 'enable'), tile_size_num: ('tiling', 'tile_size'), tile_overlap_num: ('tiling', 'tile_overlap'),
        ffmpeg_use_gpu_check: ('ffmpeg', 'use_gpu'), ffmpeg_preset_dropdown: ('ffmpeg', 'preset'), ffmpeg_quality_slider: ('ffmpeg', 'quality'),
        input_frames_folder: ('frame_folder', 'input_path'), frame_folder_fps_slider: ('frame_folder', 'fps'),
        enable_scene_split_check: ('scene_split', 'enable'), scene_split_mode_radio: ('scene_split', 'mode'), scene_min_scene_len_num: ('scene_split', 'min_scene_len'), scene_drop_short_check: ('scene_split', 'drop_short'), scene_merge_last_check: ('scene_split', 'merge_last'), scene_frame_skip_num: ('scene_split', 'frame_skip'), scene_threshold_num: ('scene_split', 'threshold'), scene_min_content_val_num: ('scene_split', 'min_content_val'), scene_frame_window_num: ('scene_split', 'frame_window'), scene_manual_split_type_radio: ('scene_split', 'manual_split_type'), scene_manual_split_value_num: ('scene_split', 'manual_split_value'), scene_copy_streams_check: ('scene_split', 'copy_streams'), scene_use_mkvmerge_check: ('scene_split', 'use_mkvmerge'), scene_rate_factor_num: ('scene_split', 'rate_factor'), scene_preset_dropdown: ('scene_split', 'encoding_preset'), scene_quiet_ffmpeg_check: ('scene_split', 'quiet_ffmpeg'),
        (cogvlm_quant_radio if UTIL_COG_VLM_AVAILABLE else None): ('cogvlm', 'quant_display'), (cogvlm_unload_radio if UTIL_COG_VLM_AVAILABLE else None): ('cogvlm', 'unload_after_use'), auto_caption_then_upscale_check: ('cogvlm', 'auto_caption_then_upscale'),
        save_frames_checkbox: ('outputs', 'save_frames'), save_metadata_checkbox: ('outputs', 'save_metadata'), save_chunks_checkbox: ('outputs', 'save_chunks'), save_chunk_frames_checkbox: ('outputs', 'save_chunk_frames'), create_comparison_video_check: ('outputs', 'create_comparison_video'),
        seed_num: ('seed', 'seed'), random_seed_check: ('seed', 'use_random'),
        enable_rife_interpolation: ('rife', 'enable'), rife_multiplier: ('rife', 'multiplier'), rife_fp16: ('rife', 'fp16'), rife_uhd: ('rife', 'uhd'), rife_scale: ('rife', 'scale'), rife_skip_static: ('rife', 'skip_static'), rife_enable_fps_limit: ('rife', 'enable_fps_limit'), rife_max_fps_limit: ('rife', 'max_fps_limit'), rife_apply_to_chunks: ('rife', 'apply_to_chunks'), rife_apply_to_scenes: ('rife', 'apply_to_scenes'), rife_keep_original: ('rife', 'keep_original'), rife_overwrite_original: ('rife', 'overwrite_original'),
        enable_fps_decrease: ('fps_decrease', 'enable'), fps_decrease_mode: ('fps_decrease', 'mode'), fps_multiplier_preset: ('fps_decrease', 'multiplier_preset'), fps_multiplier_custom: ('fps_decrease', 'multiplier_custom'), target_fps: ('fps_decrease', 'target_fps'), fps_interpolation_method: ('fps_decrease', 'interpolation_method'),
        enable_image_upscaler_check: ('image_upscaler', 'enable'), image_upscaler_model_dropdown: ('image_upscaler', 'model'), image_upscaler_batch_size_slider: ('image_upscaler', 'batch_size'),
        enable_face_restoration_check: ('face_restoration', 'enable'), face_restoration_fidelity_slider: ('face_restoration', 'fidelity_weight'), enable_face_colorization_check: ('face_restoration', 'enable_colorization'), face_restoration_when_radio: ('face_restoration', 'when'), codeformer_model_dropdown: ('face_restoration', 'model'), face_restoration_batch_size_slider: ('face_restoration', 'batch_size'),
        gpu_selector: ('gpu', 'device'),
    }

    def save_preset_wrapper(preset_name, *all_ui_values):
        import time
        
        app_config = build_app_config_from_ui(*all_ui_values)
        success, message = preset_handler.save_preset(app_config, preset_name)
        
        if success:
            # Sanitize name for dropdown value
            safe_preset_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            
            # Ensure file system has committed the file before updating dropdown
            time.sleep(0.1)  # Small delay to ensure file is written
            
            # Verify the file was actually written
            presets_dir = preset_handler.get_presets_dir()
            filepath = os.path.join(presets_dir, f"{safe_preset_name}.json")
            if not os.path.exists(filepath):
                logger.warning(f"Preset file not immediately available after save: {filepath}")
                time.sleep(0.2)  # Additional delay if file not found
            
            new_choices = get_filtered_preset_list()
            
            # Set the cached preset to prevent unnecessary reload after save
            last_loaded_preset[0] = safe_preset_name
            
            return message, gr.update(choices=new_choices, value=safe_preset_name)
        else:
            return message, gr.update()

    def load_preset_wrapper(preset_name):
        import time
        
        # Skip loading if preset_name is None, empty, or just whitespace
        if not preset_name or not preset_name.strip():
            return [gr.update(value="No preset selected")] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(20)]
        
        # Sanitize preset name to prevent issues
        preset_name = preset_name.strip()
        
        # Check if this is the system file that should be excluded
        if preset_name == "last_preset":
            return [gr.update(value="Cannot load 'last_preset' - this is a system file")] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(20)]
        
        # Try to load the preset with retry logic to handle timing issues
        config_dict, message = None, None
        max_retries = 3
        
        for attempt in range(max_retries):
            config_dict, message = preset_handler.load_preset(preset_name)
            if config_dict:
                break
            
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                logger.debug(f"Preset load attempt {attempt + 1} failed, retrying in 0.1s...")
                time.sleep(0.1)
        
        if not config_dict:
            # Suppress error logging for expected failures (file might not exist yet)
            logger.debug(f"Failed to load preset '{preset_name}' after {max_retries} attempts: {message}")
            return [gr.update(value=f"Could not load preset: {preset_name}")] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(20)]  # Additional outputs for conditional controls

        # Get a fresh AppConfig with default values for any missing keys in the preset file
        default_config = create_app_config(base_path, args.outputs_folder, star_cfg)
        updates = []
        
        def reverse_extract_codeformer_model_path(path):
            if not path: return "Auto (Default)"
            if "codeformer.pth" in path: return "codeformer.pth (359.2MB)"
            return "Auto (Default)"

        # Extract checkbox values for conditional control updates
        enable_image_upscaler_val = config_dict.get('image_upscaler', {}).get('enable', default_config.image_upscaler.enable)
        enable_face_restoration_val = config_dict.get('face_restoration', {}).get('enable', default_config.face_restoration.enable)
        enable_target_res_val = config_dict.get('resolution', {}).get('enable_target_res', default_config.resolution.enable_target_res)
        enable_tiling_val = config_dict.get('tiling', {}).get('enable', default_config.tiling.enable)
        enable_context_window_val = config_dict.get('context_window', {}).get('enable', default_config.context_window.enable)
        enable_scene_split_val = config_dict.get('scene_split', {}).get('enable', default_config.scene_split.enable)
        enable_rife_val = config_dict.get('rife', {}).get('enable', default_config.rife.enable)
        enable_fps_decrease_val = config_dict.get('fps_decrease', {}).get('enable', default_config.fps_decrease.enable)
        random_seed_val = config_dict.get('seed', {}).get('use_random', default_config.seed.use_random)
        rife_enable_fps_limit_val = config_dict.get('rife', {}).get('enable_fps_limit', default_config.rife.enable_fps_limit)

        for component in preset_components:
            if isinstance(component, gr.State) or component is None:
                updates.append(gr.update())
                continue
            
            if component in component_key_map:
                section, key = component_key_map[component]
                # Get the default value from a fresh config object
                default_value = getattr(getattr(default_config, section), key)
                # Get value from loaded dict, falling back to the default
                value = config_dict.get(section, {}).get(key, default_value)
                
                # Special handling for components that need it
                if component is codeformer_model_dropdown:
                    # The saved value is a path, we need to convert it back to the dropdown choice
                    value = reverse_extract_codeformer_model_path(value)
                elif component is gpu_selector:
                    # The saved value is a GPU index, we need to convert it back to the dropdown format
                    available_gpus = util_get_available_gpus()
                    # Handle legacy "Auto" values by converting to "0"
                    if value == "Auto":
                        value = 0
                    # Convert string index to integer if needed
                    if isinstance(value, str) and value.isdigit():
                        value = int(value)
                    value = convert_gpu_index_to_dropdown(value, available_gpus)
                    logger.info(f"Loading preset: {preset_name}")
                
                updates.append(gr.update(value=value))
            else:
                logger.warning(f"Component with label '{getattr(component, 'label', 'N/A')}' not found in component_key_map. Skipping update.")
                updates.append(gr.update())
        
        # Add conditional control updates
        conditional_updates = [
            # Image upscaler controls
            gr.update(interactive=enable_image_upscaler_val),  # image_upscaler_model_dropdown
            gr.update(interactive=enable_image_upscaler_val),  # image_upscaler_batch_size_slider
            # Face restoration controls
            gr.update(interactive=enable_face_restoration_val),  # face_restoration_fidelity_slider
            gr.update(interactive=enable_face_restoration_val),  # enable_face_colorization_check
            gr.update(interactive=enable_face_restoration_val),  # face_restoration_when_radio
            gr.update(interactive=enable_face_restoration_val),  # codeformer_model_dropdown
            gr.update(interactive=enable_face_restoration_val),  # face_restoration_batch_size_slider
            # Target resolution controls
            gr.update(interactive=enable_target_res_val),  # target_h_num
            gr.update(interactive=enable_target_res_val),  # target_w_num
            gr.update(interactive=enable_target_res_val),  # target_res_mode_radio
            # Tiling controls
            gr.update(interactive=enable_tiling_val),  # tile_size_num
            gr.update(interactive=enable_tiling_val),  # tile_overlap_num
            # Context window control
            gr.update(interactive=enable_context_window_val),  # context_overlap_num
            # Scene splitting controls (15 controls)
            gr.update(interactive=enable_scene_split_val),  # scene_split_mode_radio
            gr.update(interactive=enable_scene_split_val),  # scene_min_scene_len_num
            gr.update(interactive=enable_scene_split_val),  # scene_threshold_num
            gr.update(interactive=enable_scene_split_val),  # scene_drop_short_check
            gr.update(interactive=enable_scene_split_val),  # scene_merge_last_check
            # Seed control
            gr.update(interactive=not random_seed_val),  # seed_num
        ]
        
        return [gr.update(value=message)] + updates + conditional_updates

    def refresh_presets_list():
        updated_choices = get_filtered_preset_list()
        logger.info(f"Refreshing preset list: {updated_choices}")
        return gr.update(choices=updated_choices)
    
    # Cache the last loaded preset to prevent unnecessary reloads
    last_loaded_preset = [None]
    
    def safe_load_preset_wrapper(preset_name):
        # Skip if this is the same preset we just loaded
        if preset_name == last_loaded_preset[0]:
            logger.debug(f"Skipping reload of already loaded preset: '{preset_name}'")
            return [gr.update()] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(20)]
        
        last_loaded_preset[0] = preset_name
        return load_preset_wrapper(preset_name)

    save_preset_btn.click(
        fn=save_preset_wrapper,
        inputs=[preset_dropdown] + click_inputs,
        outputs=[preset_status, preset_dropdown]
    )

    preset_dropdown.change(
        fn=safe_load_preset_wrapper,
        inputs=[preset_dropdown],
        outputs=[preset_status] + preset_components + [
            # Image upscaler controls
            image_upscaler_model_dropdown, image_upscaler_batch_size_slider,
            # Face restoration controls
            face_restoration_fidelity_slider, enable_face_colorization_check, 
            face_restoration_when_radio, codeformer_model_dropdown, face_restoration_batch_size_slider,
            # Target resolution controls
            target_h_num, target_w_num, target_res_mode_radio,
            # Tiling controls
            tile_size_num, tile_overlap_num,
            # Context window control
            context_overlap_num,
            # Scene splitting controls
            scene_split_mode_radio, scene_min_scene_len_num, scene_threshold_num, 
            scene_drop_short_check, scene_merge_last_check,
            # Seed control
            seed_num
        ]
    )

    refresh_presets_btn.click(
        fn=refresh_presets_list,
        inputs=[],
        outputs=[preset_dropdown]
    )

    # --- END OF PRESET LOGIC ---

    for component in [input_video ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ]:
        component .change (
        fn =calculate_fps_preview ,
        inputs =[input_video ,fps_decrease_mode ,fps_multiplier_preset ,fps_multiplier_custom ,target_fps ],
        outputs =fps_calculation_info 
        )

if __name__ =="__main__":
    os .makedirs (APP_CONFIG.paths.outputs_dir ,exist_ok =True )
    logger .info (f"Gradio App Starting. Default output to: {os.path.abspath(APP_CONFIG.paths.outputs_dir)}")
    logger .info (f"STAR Models expected at: {APP_CONFIG.paths.light_deg_model_path}, {APP_CONFIG.paths.heavy_deg_model_path}")
    if UTIL_COG_VLM_AVAILABLE :
        logger .info (f"CogVLM2 Model expected at: {APP_CONFIG.paths.cogvlm_model_path}")

    available_gpus_main =util_get_available_gpus ()
    if available_gpus_main :
        # Always use GPU 0 as default (first GPU in the list)
        default_gpu_main_val =available_gpus_main [0 ]

        util_set_gpu_device (default_gpu_main_val ,logger =logger )
        logger .info (f"Initialized with default GPU: {default_gpu_main_val} (GPU 0)")
    else :
        logger .info ("No CUDA GPUs detected, attempting to set to GPU 0 (CPU fallback).")
        util_set_gpu_device (None ,logger =logger )

    effective_allowed_paths =util_get_available_drives (APP_CONFIG.paths.outputs_dir ,base_path ,logger =logger )

    demo .queue ().launch (
    debug =True ,
    max_threads =100 ,
    inbrowser =True ,
    share =args .share ,
    allowed_paths =effective_allowed_paths ,
    prevent_thread_lock =True 
    )