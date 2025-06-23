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
from argparse import ArgumentParser, Namespace 
import logging 
import re 
from pathlib import Path 
from functools import partial 

# Import dataclasses instead of config
from logic.dataclasses import (
    ProcessingParams, PromptParams, CoreParams, PerformanceParams, TilingParams,
    ContextWindowParams, TargetResolutionParams, FFmpegParams, OutputParams,
    SceneSplittingParams, FpsParams, RifeParams, CogVLMParams, SeedParams,
    ImageUpscalerParams, FaceRestorationParams, FrameFolderParams, BatchParams,
    DEFAULT_OUTPUT_DIR, LIGHT_DEG_MODEL_PATH, HEAVY_DEG_MODEL_PATH, COG_VLM_MODEL_PATH,
    UPSCALE_MODELS_DIR, FACE_RESTORATION_MODELS_DIR, UTIL_COG_VLM_AVAILABLE,
    UTIL_BITSANDBYTES_AVAILABLE, get_cogvlm_quant_choices_map, 
    get_default_cogvlm_quant_display, initialize_paths_and_prompts,
    DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT, DEFAULT_AUTO_CAPTION_THEN_UPSCALE,
    DEFAULT_MODEL_CHOICE, DEFAULT_UPSCALE_FACTOR, DEFAULT_CFG_SCALE,
    DEFAULT_DIFFUSION_STEPS_FAST, DEFAULT_DIFFUSION_STEPS_NORMAL, DEFAULT_SOLVER_MODE,
    DEFAULT_COLOR_FIX_METHOD, DEFAULT_MAX_CHUNK_LEN, DEFAULT_ENABLE_CHUNK_OPTIMIZATION,
    DEFAULT_VAE_CHUNK, DEFAULT_ENABLE_TILING, DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP,
    DEFAULT_ENABLE_CONTEXT_WINDOW, DEFAULT_CONTEXT_OVERLAP, DEFAULT_ENABLE_TARGET_RES,
    DEFAULT_TARGET_H, DEFAULT_TARGET_W, DEFAULT_TARGET_RES_MODE, DEFAULT_FFMPEG_USE_GPU,
    DEFAULT_FFMPEG_PRESET, DEFAULT_FFMPEG_QUALITY_CPU, DEFAULT_FFMPEG_QUALITY_GPU,
    DEFAULT_FRAME_FOLDER_FPS, DEFAULT_SAVE_FRAMES, DEFAULT_SAVE_METADATA,
    DEFAULT_SAVE_CHUNKS, DEFAULT_SAVE_CHUNK_FRAMES, DEFAULT_CREATE_COMPARISON_VIDEO,
    DEFAULT_ENABLE_SCENE_SPLIT, DEFAULT_SCENE_SPLIT_MODE, DEFAULT_SCENE_MIN_SCENE_LEN,
    DEFAULT_SCENE_DROP_SHORT, DEFAULT_SCENE_MERGE_LAST, DEFAULT_SCENE_FRAME_SKIP,
    DEFAULT_SCENE_THRESHOLD, DEFAULT_SCENE_MIN_CONTENT_VAL, DEFAULT_SCENE_FRAME_WINDOW,
    DEFAULT_SCENE_COPY_STREAMS, DEFAULT_SCENE_USE_MKVMERGE, DEFAULT_SCENE_RATE_FACTOR,
    DEFAULT_SCENE_ENCODING_PRESET, DEFAULT_SCENE_QUIET_FFMPEG, DEFAULT_SCENE_MANUAL_SPLIT_TYPE,
    DEFAULT_SCENE_MANUAL_SPLIT_VALUE, DEFAULT_ENABLE_FPS_DECREASE, DEFAULT_FPS_DECREASE_MODE,
    DEFAULT_FPS_MULTIPLIER, DEFAULT_TARGET_FPS, DEFAULT_FPS_INTERPOLATION_METHOD,
    DEFAULT_RIFE_ENABLE_INTERPOLATION, DEFAULT_RIFE_MULTIPLIER, DEFAULT_RIFE_FP16,
    DEFAULT_RIFE_UHD, DEFAULT_RIFE_SCALE, DEFAULT_RIFE_SKIP_STATIC,
    DEFAULT_RIFE_ENABLE_FPS_LIMIT, DEFAULT_RIFE_MAX_FPS_LIMIT, DEFAULT_RIFE_APPLY_TO_CHUNKS,
    DEFAULT_RIFE_APPLY_TO_SCENES, DEFAULT_RIFE_KEEP_ORIGINAL, DEFAULT_RIFE_OVERWRITE_ORIGINAL,
    DEFAULT_COGVLM_UNLOAD_AFTER_USE, DEFAULT_SEED, DEFAULT_RANDOM_SEED,
    DEFAULT_ENABLE_IMAGE_UPSCALER, DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE,
    DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE, DEFAULT_IMAGE_UPSCALER_BATCH_SIZE,
    DEFAULT_ENABLE_FACE_RESTORATION, DEFAULT_FACE_RESTORATION_FIDELITY,
    DEFAULT_ENABLE_FACE_COLORIZATION, DEFAULT_FACE_RESTORATION_WHEN,
    DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE, DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE,
    DEFAULT_FACE_RESTORATION_BATCH_SIZE, DEFAULT_BATCH_SKIP_EXISTING,
    DEFAULT_BATCH_USE_PROMPT_FILES, DEFAULT_BATCH_SAVE_CAPTIONS
)

from logic import metadata_handler 

from logic.cogvlm_utils import (
    load_cogvlm_model as util_load_cogvlm_model,
    unload_cogvlm_model as util_unload_cogvlm_model,
    auto_caption as util_auto_caption,
    COG_VLM_AVAILABLE as UTIL_COG_VLM_AVAILABLE,
    BITSANDBYTES_AVAILABLE as UTIL_BITSANDBYTES_AVAILABLE
)

from logic.common_utils import format_time 

from logic.ffmpeg_utils import (
    run_ffmpeg_command as util_run_ffmpeg_command,
    extract_frames as util_extract_frames,
    create_video_from_frames as util_create_video_from_frames,
    decrease_fps as util_decrease_fps,
    decrease_fps_with_multiplier as util_decrease_fps_with_multiplier,
    calculate_target_fps_from_multiplier as util_calculate_target_fps_from_multiplier,
    get_common_fps_multipliers as util_get_common_fps_multipliers,
    get_video_info as util_get_video_info,
    format_video_info_message as util_format_video_info_message
)

from logic.file_utils import (
    sanitize_filename as util_sanitize_filename,
    get_batch_filename as util_get_batch_filename,
    get_next_filename as util_get_next_filename,
    cleanup_temp_dir as util_cleanup_temp_dir,
    get_video_resolution as util_get_video_resolution,
    get_available_drives as util_get_available_drives,
    open_folder as util_open_folder
)

from logic.scene_utils import (
    split_video_into_scenes as util_split_video_into_scenes,
    merge_scene_videos as util_merge_scene_videos,
    split_video_only as util_split_video_only
)

from logic.upscaling_utils import (
    calculate_upscale_params as util_calculate_upscale_params
)

from logic.gpu_utils import (
    get_available_gpus as util_get_available_gpus,
    set_gpu_device as util_set_gpu_device,
    get_gpu_device as util_get_gpu_device,
    validate_gpu_availability as util_validate_gpu_availability
)

from logic.nvenc_utils import (
    is_resolution_too_small_for_nvenc
)

from logic.batch_operations import (
    process_batch_videos
)

from logic.batch_processing_help import create_batch_processing_help

from logic.upscaling_core import run_upscale as core_run_upscale 

from logic.manual_comparison import (
    generate_manual_comparison_video as util_generate_manual_comparison_video
)

from logic.rife_interpolation import (
    rife_fps_only_wrapper as util_rife_fps_only_wrapper
)

from logic.image_upscaler_utils import (
    scan_for_models as util_scan_for_models,
    get_model_info as util_get_model_info
)

from logic.video_editor import (
    parse_time_ranges as util_parse_time_ranges,
    parse_frame_ranges as util_parse_frame_ranges,
    validate_ranges as util_validate_ranges,
    get_video_detailed_info as util_get_video_detailed_info,
    cut_video_segments as util_cut_video_segments,
    create_preview_segment as util_create_preview_segment,
    estimate_processing_time as util_estimate_processing_time,
    format_video_info_for_display as util_format_video_info_for_display
)

from logic.temp_folder_utils import (
    get_temp_folder_path as util_get_temp_folder_path,
    calculate_temp_folder_size as util_calculate_temp_folder_size,
    format_temp_folder_size as util_format_temp_folder_size,
    clear_temp_folder as util_clear_temp_folder
)

from logic.frame_folder_utils import (
    validate_frame_folder_input as util_validate_frame_folder_input,
    process_frame_folder_to_video as util_process_frame_folder_to_video,
    find_frame_folders_in_directory as util_find_frame_folders_in_directory
)

SELECTED_GPU_ID = 0

parser = ArgumentParser(description="Ultimate SECourses STAR Video Upscaler")
parser.add_argument('--share', action='store_true', help="Enable Gradio live share")
parser.add_argument('--outputs_folder', type=str, default="outputs", help="Main folder for output videos and related files")
args = parser.parse_args()

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

logger = get_logger()
logger.setLevel(logging.INFO)
found_stream_handler = False
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
        found_stream_handler = True
        logger.info("Diagnostic: Explicitly set StreamHandler level to INFO.")
if not found_stream_handler:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("Diagnostic: No StreamHandler found, added a new one with INFO level.")
logger.info(f"Logger '{logger.name}' configured with level: {logging.getLevelName(logger.level)}. Handlers: {logger.handlers}")

# Initialize paths and prompts
initialize_paths_and_prompts(base_path, args.outputs_folder, star_cfg)

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

if not os.path.exists(LIGHT_DEG_MODEL_PATH):
    logger.error(f"FATAL: Light degradation model not found at {LIGHT_DEG_MODEL_PATH}.")
if not os.path.exists(HEAVY_DEG_MODEL_PATH):
    logger.error(f"FATAL: Heavy degradation model not found at {HEAVY_DEG_MODEL_PATH}.")

css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

# Event Handlers and UI Logic
def update_steps_display(mode):
    if mode == 'fast':
        return gr.update(value=DEFAULT_DIFFUSION_STEPS_FAST, interactive=False)
    else:
        return gr.update(value=DEFAULT_DIFFUSION_STEPS_NORMAL, interactive=True)

solver_mode_radio.change(update_steps_display, solver_mode_radio, steps_slider)

enable_target_res_check.change(
    lambda x: [gr.update(interactive=x)] * 3,
    inputs=enable_target_res_check,
    outputs=[target_h_num, target_w_num, target_res_mode_radio]
)

enable_tiling_check.change(
    lambda x: [gr.update(interactive=x)] * 2,
    inputs=enable_tiling_check,
    outputs=[tile_size_num, tile_overlap_num]
)

enable_context_window_check.change(
    lambda x: gr.update(interactive=x),
    inputs=enable_context_window_check,
    outputs=context_overlap_num
)

# Frame folder controls
def update_frame_folder_controls(enable_frame_folder):
    """Enable/disable frame folder controls based on checkbox state."""
    return [
        gr.update(interactive=enable_frame_folder),  # frames folder textbox
        gr.update(visible=enable_frame_folder)       # status textbox
    ]

def validate_frame_folder_input_wrapper(frames_folder_path, enable_frame_folder):
    """Validate frame folder input and show status."""
    if not enable_frame_folder or not frames_folder_path:
        return gr.update(visible=False, value="")
    
    is_valid, message, frame_count = util_validate_frame_folder_input(frames_folder_path, logger)
    
    if is_valid:
        status_msg = f"✅ {message}"
    else:
        status_msg = f"❌ {message}"
    
    return gr.update(visible=True, value=status_msg)

enable_frame_folder_check.change(
    fn=update_frame_folder_controls,
    inputs=enable_frame_folder_check,
    outputs=[input_frames_folder, frames_folder_status]
)

input_frames_folder.change(
    fn=validate_frame_folder_input_wrapper,
    inputs=[input_frames_folder, enable_frame_folder_check],
    outputs=frames_folder_status
)

# Image upscaler controls
def update_image_upscaler_controls(enable_image_upscaler):
    """Enable/disable image upscaler controls based on checkbox state."""
    return [
        gr.update(interactive=enable_image_upscaler),  # model dropdown
        gr.update(interactive=enable_image_upscaler)   # batch size slider
    ]

enable_image_upscaler_check.change(
    fn=update_image_upscaler_controls,
    inputs=enable_image_upscaler_check,
    outputs=[image_upscaler_model_dropdown, image_upscaler_batch_size_slider]
)

# Face restoration controls
def update_face_restoration_controls(enable_face_restoration):
    """Enable/disable face restoration controls based on checkbox state."""
    return [
        gr.update(interactive=enable_face_restoration),  # fidelity slider
        gr.update(interactive=enable_face_restoration),  # colorization checkbox
        gr.update(interactive=enable_face_restoration),  # timing radio
        gr.update(interactive=enable_face_restoration),  # model dropdown
        gr.update(interactive=enable_face_restoration)   # batch size slider
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

def update_context_overlap_max(max_chunk_len):
    """Update the maximum value of context overlap based on max_chunk_len"""
    new_max = max(0, int(max_chunk_len) - 1)
    return gr.update(maximum=new_max)

max_chunk_len_slider.change(
    fn=update_context_overlap_max,
    inputs=max_chunk_len_slider,
    outputs=context_overlap_num
)

scene_splitting_outputs = [
    scene_split_mode_radio, scene_min_scene_len_num, scene_threshold_num, scene_drop_short_check, scene_merge_last_check,
    scene_frame_skip_num, scene_min_content_val_num, scene_frame_window_num,
    scene_manual_split_type_radio, scene_manual_split_value_num, scene_copy_streams_check,
    scene_use_mkvmerge_check, scene_rate_factor_num, scene_preset_dropdown, scene_quiet_ffmpeg_check
]
enable_scene_split_check.change(
    lambda x: [gr.update(interactive=x)] * len(scene_splitting_outputs),
    inputs=enable_scene_split_check,
    outputs=scene_splitting_outputs
)

def update_ffmpeg_quality_settings(use_gpu_ffmpeg):
    if use_gpu_ffmpeg:
        return gr.Slider(label="FFmpeg Quality (CQ for NVENC)", value=DEFAULT_FFMPEG_QUALITY_GPU, info="For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28.")
    else:
        return gr.Slider(label="FFmpeg Quality (CRF for libx264)", value=DEFAULT_FFMPEG_QUALITY_CPU, info="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default).")

ffmpeg_use_gpu_check.change(
    fn=update_ffmpeg_quality_settings,
    inputs=ffmpeg_use_gpu_check,
    outputs=ffmpeg_quality_slider
)

open_output_folder_button.click(
    fn=lambda: util_open_folder(DEFAULT_OUTPUT_DIR, logger=logger),
    inputs=[],
    outputs=[]
)

# Main upscaling button click handler
upscale_button.click(
    fn=upscale_director_logic_v2,
    inputs=[
        input_video, user_prompt, pos_prompt, neg_prompt, auto_caption_then_upscale_check, gpu_selector,
        enable_frame_folder_check, input_frames_folder, model_selector, upscale_factor_slider, cfg_slider,
        steps_slider, solver_mode_radio, color_fix_dropdown, max_chunk_len_slider, enable_chunk_optimization_check,
        vae_chunk_slider, enable_image_upscaler_check, image_upscaler_model_dropdown, image_upscaler_batch_size_slider,
        enable_face_restoration_check, face_restoration_fidelity_slider, enable_face_colorization_check,
        face_restoration_when_radio, codeformer_model_dropdown, face_restoration_batch_size_slider,
        cogvlm_quant_radio, cogvlm_unload_radio, enable_context_window_check, context_overlap_num,
        enable_tiling_check, tile_size_num, tile_overlap_num, enable_target_res_check, target_h_num,
        target_w_num, target_res_mode_radio, ffmpeg_use_gpu_check, ffmpeg_preset_dropdown, ffmpeg_quality_slider,
        frame_folder_fps_slider, create_comparison_video_check, save_frames_checkbox, save_metadata_checkbox,
        save_chunks_checkbox, save_chunk_frames_checkbox, seed_num, random_seed_check,
        enable_scene_split_check, scene_split_mode_radio, scene_min_scene_len_num, scene_drop_short_check,
        scene_merge_last_check, scene_frame_skip_num, scene_threshold_num, scene_min_content_val_num,
        scene_frame_window_num, scene_copy_streams_check, scene_use_mkvmerge_check, scene_rate_factor_num,
        scene_preset_dropdown, scene_quiet_ffmpeg_check, scene_manual_split_type_radio, scene_manual_split_value_num,
        enable_fps_decrease, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps,
        fps_interpolation_method, enable_rife_interpolation, rife_multiplier, rife_fp16, rife_uhd, rife_scale,
        rife_skip_static, rife_enable_fps_limit, rife_max_fps_limit, rife_apply_to_chunks, rife_apply_to_scenes,
        rife_keep_original, rife_overwrite_original
    ],
    outputs=[
        output_video, status_textbox, user_prompt,
        caption_status if UTIL_COG_VLM_AVAILABLE else gr.State(None),
        last_chunk_video, chunk_status_text, comparison_video
    ],
    show_progress_on=[output_video]
)

# Auto-caption and RIFE buttons
if UTIL_COG_VLM_AVAILABLE:
    def auto_caption_wrapper(vid, quant_display, unload_strat, progress=gr.Progress(track_tqdm=True)):
        caption_text, caption_stat_msg = util_auto_caption(
            vid,
            get_quant_value_from_display(quant_display),
            unload_strat,
            COG_VLM_MODEL_PATH,
            logger=logger,
            progress=progress
        )
        return caption_text, caption_stat_msg

    auto_caption_btn.click(
        fn=auto_caption_wrapper,
        inputs=[input_video, cogvlm_quant_radio, cogvlm_unload_radio],
        outputs=[user_prompt, caption_status],
        show_progress_on=[user_prompt]
    ).then(lambda: gr.update(visible=True), None, caption_status)

def rife_fps_increase_wrapper(
    input_video_val,
    rife_multiplier_val=2,
    rife_fp16_val=True,
    rife_uhd_val=False,
    rife_scale_val=1.0,
    rife_skip_static_val=False,
    rife_enable_fps_limit_val=False,
    rife_max_fps_limit_val=60,
    ffmpeg_preset_dropdown_val="medium",
    ffmpeg_quality_slider_val=18,
    ffmpeg_use_gpu_check_val=True,
    seed_num_val=99,
    random_seed_check_val=False,
    progress=gr.Progress(track_tqdm=True)
):
    return util_rife_fps_only_wrapper(
        input_video_val=input_video_val,
        rife_multiplier_val=rife_multiplier_val,
        rife_fp16_val=rife_fp16_val,
        rife_uhd_val=rife_uhd_val,
        rife_scale_val=rife_scale_val,
        rife_skip_static_val=rife_skip_static_val,
        rife_enable_fps_limit_val=rife_enable_fps_limit_val,
        rife_max_fps_limit_val=rife_max_fps_limit_val,
        ffmpeg_preset_dropdown_val=ffmpeg_preset_dropdown_val,
        ffmpeg_quality_slider_val=ffmpeg_quality_slider_val,
        ffmpeg_use_gpu_check_val=ffmpeg_use_gpu_check_val,
        seed_num_val=seed_num_val,
        random_seed_check_val=random_seed_check_val,
        output_dir=DEFAULT_OUTPUT_DIR,
        logger=logger,
        progress=progress
    )

rife_fps_button.click(
    fn=rife_fps_increase_wrapper,
    inputs=[
        input_video,
        rife_multiplier, rife_fp16, rife_uhd, rife_scale, rife_skip_static,
        rife_enable_fps_limit, rife_max_fps_limit,
        ffmpeg_preset_dropdown, ffmpeg_quality_slider, ffmpeg_use_gpu_check,
        seed_num, random_seed_check
    ],
    outputs=[output_video, status_textbox],
    show_progress_on=[output_video]
)

# Split video only button
split_only_button.click(
    fn=wrapper_split_video_only_for_gradio,
    inputs=[
        input_video, scene_split_mode_radio, scene_min_scene_len_num, scene_drop_short_check, scene_merge_last_check,
        scene_frame_skip_num, scene_threshold_num, scene_min_content_val_num, scene_frame_window_num,
        scene_copy_streams_check, scene_use_mkvmerge_check, scene_rate_factor_num, scene_preset_dropdown, scene_quiet_ffmpeg_check,
        scene_manual_split_type_radio, scene_manual_split_value_num
    ],
    outputs=[output_video, status_textbox],
    show_progress_on=[output_video]
)

# GPU selector change handler
gpu_selector.change(
    fn=lambda gpu_id: util_set_gpu_device(gpu_id, logger=logger),
    inputs=gpu_selector,
    outputs=status_textbox
)

# Seed controls
def update_seed_num_interactive(is_random_seed_checked):
    return gr.update(interactive=not is_random_seed_checked)

random_seed_check.change(
    fn=update_seed_num_interactive,
    inputs=random_seed_check,
    outputs=seed_num
)

# Video info display when video is uploaded
def display_video_info(video_path):
    """Display video information when a video is uploaded."""
    if video_path is None:
        return ""
    
    try:
        # Get video information
        video_info = util_get_video_info(video_path, logger)
        
        if video_info:
            # Format the message
            filename = os.path.basename(video_path) if video_path else None
            info_message = util_format_video_info_message(video_info, filename)
            
            # Log the information
            logger.info(f"Video uploaded: {filename}")
            logger.info(f"Video details: {video_info['frames']} frames, {video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']}")
            
            return info_message
        else:
            error_msg = "❌ Could not read video information"
            logger.warning(f"Failed to get video info for: {video_path}")
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ Error reading video: {str(e)}"
        logger.error(f"Exception in display_video_info: {e}")
        return error_msg

# Add video info display when video is uploaded
input_video.change(
    fn=display_video_info,
    inputs=input_video,
    outputs=status_textbox
)

# Temp folder cleanup button
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

# Attach callback to the delete temp button
delete_temp_button.click(
    fn=clear_temp_folder_wrapper,
    inputs=[],
    outputs=[status_textbox, delete_temp_button],
    show_progress_on=[status_textbox]
)

# Logger setup and main execution
logger = get_logger()
logger.setLevel(logging.INFO)
found_stream_handler = False
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
        found_stream_handler = True
        logger.info("Diagnostic: Explicitly set StreamHandler level to INFO.")
if not found_stream_handler:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("Diagnostic: No StreamHandler found, added a new one with INFO level.")
logger.info(f"Logger '{logger.name}' configured with level: {logging.getLevelName(logger.level)}. Handlers: {logger.handlers}")

# Initialize paths and prompts using dataclasses function
initialize_paths_and_prompts(base_path, args.outputs_folder, star_cfg)

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

if not os.path.exists(LIGHT_DEG_MODEL_PATH):
    logger.error(f"FATAL: Light degradation model not found at {LIGHT_DEG_MODEL_PATH}.")
if not os.path.exists(HEAVY_DEG_MODEL_PATH):
    logger.error(f"FATAL: Heavy degradation model not found at {HEAVY_DEG_MODEL_PATH}.")

# Create the complete Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultimate SECourses STAR Video Upscaler V100 Pre Release")

    with gr.Tabs() as main_tabs:
        with gr.Tab("Main", id="main_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        input_video = gr.Video(
                            label="Input Video",
                            sources=["upload"],
                            interactive=True, height=512
                        )
                        
                        # Integration status (hidden by default)
                        integration_status = gr.Textbox(
                            label="Integration Status",
                            interactive=False,
                            lines=2,
                            visible=False,
                            value=""
                        )
                        
                        with gr.Row():
                            user_prompt = gr.Textbox(
                                label="Describe the Video Content (Prompt) (Useful only for STAR Model)",
                                lines=3,
                                placeholder="e.g., A panda playing guitar by a lake at sunset.",
                                info="""Describe the main subject and action in the video. This guides the upscaling process.
Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens.
If CogVLM2 is available, you can use the button below to generate a caption automatically."""
                            )
                        
                        with gr.Row():
                            auto_caption_then_upscale_check = gr.Checkbox(
                                label="Auto-caption then Upscale (Useful only for STAR Model)",
                                value=DEFAULT_AUTO_CAPTION_THEN_UPSCALE,
                                info="If checked, clicking 'Upscale Video' will first generate a caption and use it as the prompt."
                            )

                            available_gpus = util_get_available_gpus()
                            gpu_choices = ["Auto"] + available_gpus if available_gpus else ["Auto", "No CUDA GPUs detected"]
                            default_gpu = available_gpus[0] if available_gpus else "Auto"

                            gpu_selector = gr.Dropdown(
                                label="GPU Selection",
                                choices=gpu_choices,
                                value=default_gpu,
                                info="Select which GPU to use for processing. 'Auto' uses default GPU or CPU if none.",
                                scale=1
                            )

                        if UTIL_COG_VLM_AVAILABLE:
                            with gr.Row():
                                auto_caption_btn = gr.Button("Generate Caption with CogVLM2 (No Upscale)", variant="primary", icon="icons/caption.png")
                                rife_fps_button = gr.Button("RIFE FPS Increase (No Upscale)", variant="primary", icon="icons/fps.png")
                            with gr.Row():
                                upscale_button = gr.Button("Upscale Video", variant="primary", icon="icons/upscale.png")

                            # Add Delete Temp Folder button directly under Upscale Video
                            with gr.Row():
                                initial_temp_size_label = util_format_temp_folder_size(logger)
                                delete_temp_button = gr.Button(f"Delete Temp Folder ({initial_temp_size_label})", variant="stop")

                            caption_status = gr.Textbox(label="Captioning Status", interactive=False, visible=False)

                    with gr.Accordion("Prompt Settings (Useful only for STAR Model)", open=True):
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
                    
                    # Frame Folder Input Feature
                    with gr.Group():
                        enable_frame_folder_check = gr.Checkbox(
                            label="Process Input Frames Folder (instead of video)",
                            value=False,
                            info="Enable to process a folder of image frames instead of a video file. Supports jpg, png, tiff, jp2, dpx and other formats. Frames will be sorted naturally (2.png before 12.png)."
                        )
                        input_frames_folder = gr.Textbox(
                            label="Input Frames Folder Path",
                            placeholder="C:/path/to/frames/folder/",
                            interactive=False,
                            info="Path to folder containing image frames. Will be processed as a video sequence using global encoding settings."
                        )
                        frames_folder_status = gr.Textbox(
                            label="Frame Folder Status",
                            interactive=False,
                            lines=2,
                            visible=False,
                            value=""
                        )
                    
                    open_output_folder_button = gr.Button("Open Outputs Folder", icon="icons/folder.png", variant="primary")

                with gr.Column(scale=1):
                    output_video = gr.Video(label="Upscaled Video", interactive=False, height=512)
                    status_textbox = gr.Textbox(label="Log", interactive=False, lines=8, max_lines=15)

                    with gr.Accordion("Last Processed Chunk", open=True):
                        last_chunk_video = gr.Video(
                            label="Last Processed Chunk Preview",
                            interactive=False,
                            height=512,
                            visible=True
                        )
                        chunk_status_text = gr.Textbox(
                            label="Chunk Status",
                            interactive=False,
                            lines=1,
                            value="No chunks processed yet"
                        )

        with gr.Tab("Resolution & Scene Split", id="resolution_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Target Resolution - Maintains Your Input Video Aspect Ratio", open=True):
                        enable_target_res_check = gr.Checkbox(
                            label="Enable Max Target Resolution",
                            value=DEFAULT_ENABLE_TARGET_RES,
                            info="Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
                        )
                        target_res_mode_radio = gr.Radio(
                            label="Target Resolution Mode",
                            choices=['Ratio Upscale', 'Downscale then 4x'], value=DEFAULT_TARGET_RES_MODE,
                            info="""How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio.
'Downscale then 4x': If input is large, downscales it towards Target H/W divided by 4, THEN applies a 4x upscale. Can clean noisy high-res input before upscaling."""
                        )
                        with gr.Row():
                            target_h_num = gr.Slider(
                                label="Max Target Height (px)",
                                value=DEFAULT_TARGET_H, minimum=128, maximum=4096, step=16,
                                info="""Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                            )
                            target_w_num = gr.Slider(
                                label="Max Target Width (px)",
                                value=DEFAULT_TARGET_W, minimum=128, maximum=4096, step=16,
                                info="""Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                            )
                
                with gr.Column(scale=1):
                    split_only_button = gr.Button("Split Video Only (No Upscaling)", icon="icons/split.png", variant="primary")
                    with gr.Accordion("Scene Splitting", open=True):
                        enable_scene_split_check = gr.Checkbox(
                            label="Enable Scene Splitting",
                            value=DEFAULT_ENABLE_SCENE_SPLIT,
                            info="""Split video into scenes and process each scene individually. This can improve quality and speed by processing similar content together.
- Quality Impact: Better temporal consistency within scenes, improved auto-captioning per scene.
- Speed Impact: Can be faster for long videos with distinct scenes.
- Memory Impact: Reduces peak memory usage by processing smaller segments."""
                        )
                        with gr.Row():
                            scene_split_mode_radio = gr.Radio(
                                label="Split Mode",
                                choices=['automatic', 'manual'], value=DEFAULT_SCENE_SPLIT_MODE,
                                info="""'automatic': Uses scene detection algorithms to find natural scene boundaries.
'manual': Splits video at fixed intervals (duration or frame count)."""
                            )
                        with gr.Group():
                            gr.Markdown("**Automatic Scene Detection Settings**")
                            with gr.Row():
                                scene_min_scene_len_num = gr.Number(
                                    label="Min Scene Length (seconds)", value=DEFAULT_SCENE_MIN_SCENE_LEN,
                                    minimum=0.1, step=0.1, info="Minimum duration for a scene. Shorter scenes will be merged or dropped."
                                )
                                scene_threshold_num = gr.Number(
                                    label="Detection Threshold", value=DEFAULT_SCENE_THRESHOLD,
                                    minimum=0.1, maximum=10.0, step=0.1, info="Sensitivity of scene detection. Lower values detect more scenes."
                                )
                            with gr.Row():
                                scene_drop_short_check = gr.Checkbox(
                                    label="Drop Short Scenes", value=DEFAULT_SCENE_DROP_SHORT,
                                    info="If enabled, scenes shorter than minimum length are dropped instead of merged."
                                )
                                scene_merge_last_check = gr.Checkbox(
                                    label="Merge Last Scene", value=DEFAULT_SCENE_MERGE_LAST,
                                    info="If the last scene is too short, merge it with the previous scene."
                                )
                            with gr.Row():
                                scene_frame_skip_num = gr.Number(
                                    label="Frame Skip", value=DEFAULT_SCENE_FRAME_SKIP,
                                    minimum=0, step=1, info="Skip frames during detection to speed up processing. 0 = analyze every frame."
                                )
                                scene_min_content_val_num = gr.Number(
                                    label="Min Content Value", value=DEFAULT_SCENE_MIN_CONTENT_VAL,
                                    minimum=0.0, step=1.0, info="Minimum content change required to detect a scene boundary."
                                )
                                scene_frame_window_num = gr.Number(
                                    label="Frame Window", value=DEFAULT_SCENE_FRAME_WINDOW,
                                    minimum=1, step=1, info="Number of frames to analyze for scene detection."
                                )
                        with gr.Group():
                            gr.Markdown("**Manual Split Settings**")
                            with gr.Row():
                                scene_manual_split_type_radio = gr.Radio(
                                    label="Manual Split Type", choices=['duration', 'frame_count'],
                                    value=DEFAULT_SCENE_MANUAL_SPLIT_TYPE,
                                    info="'duration': Split every N seconds.\n'frame_count': Split every N frames."
                                )
                                scene_manual_split_value_num = gr.Number(
                                    label="Split Value", value=DEFAULT_SCENE_MANUAL_SPLIT_VALUE,
                                    minimum=1.0, step=1.0, info="Duration in seconds or number of frames for manual splitting."
                                )
                        with gr.Group():
                            gr.Markdown("**Encoding Settings (for scene segments)**")
                            with gr.Row():
                                scene_copy_streams_check = gr.Checkbox(
                                    label="Copy Streams", value=DEFAULT_SCENE_COPY_STREAMS,
                                    info="Copy video/audio streams without re-encoding during scene splitting (faster) but can generate inaccurate splits."
                                )
                                scene_use_mkvmerge_check = gr.Checkbox(
                                    label="Use MKVMerge", value=DEFAULT_SCENE_USE_MKVMERGE,
                                    info="Use mkvmerge instead of ffmpeg for splitting (if available)."
                                )
                            with gr.Row():
                                scene_rate_factor_num = gr.Number(
                                    label="Rate Factor (CRF)", value=DEFAULT_SCENE_RATE_FACTOR,
                                    minimum=0, maximum=51, step=1,
                                    info="Quality setting for re-encoding (lower = better quality). Only used if Copy Streams is disabled."
                                )
                                scene_preset_dropdown = gr.Dropdown(
                                    label="Encoding Preset",
                                    choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                                    value=DEFAULT_SCENE_ENCODING_PRESET,
                                    info="Encoding speed vs quality trade-off. Only used if Copy Streams is disabled."
                                )
                            scene_quiet_ffmpeg_check = gr.Checkbox(
                                label="Quiet FFmpeg", value=DEFAULT_SCENE_QUIET_FFMPEG,
                                info="Suppress ffmpeg output during scene splitting."
                            )

        with gr.Tab("Core Settings", id="core_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Core Upscaling Settings for STAR Model")
                        model_selector = gr.Dropdown(
                            label="STAR Model",
                            choices=["Light Degradation", "Heavy Degradation"],
                            value=DEFAULT_MODEL_CHOICE,
                            info="""Choose the model based on input video quality.
'Light Degradation': Better for relatively clean inputs (e.g., downloaded web videos).
'Heavy Degradation': Better for inputs with significant compression artifacts, noise, or blur."""
                        )
                        upscale_factor_slider = gr.Slider(
                            label="Upscale Factor (if Target Res disabled)",
                            minimum=1.0, maximum=8.0, value=DEFAULT_UPSCALE_FACTOR, step=0.1,
                            info="Simple multiplication factor for output resolution if 'Enable Max Target Resolution' is OFF. E.g., 4.0 means 4x height and 4x width."
                        )
                        cfg_slider = gr.Slider(
                            label="Guidance Scale (CFG)",
                            minimum=1.0, maximum=15.0, value=DEFAULT_CFG_SCALE, step=0.5,
                            info="Controls how strongly the model follows your combined text prompt. Higher values mean stricter adherence, lower values allow more creativity. Typical values: 5.0-10.0."
                        )
                        with gr.Row():
                            solver_mode_radio = gr.Radio(
                                label="Solver Mode",
                                choices=['fast', 'normal'], value=DEFAULT_SOLVER_MODE,
                                info="""Diffusion solver type.
'fast': Fewer steps (default ~15), much faster, good quality usually.
'normal': More steps (default ~50), slower, potentially slightly better detail/coherence."""
                            )
                            steps_slider = gr.Slider(
                                label="Diffusion Steps",
                                minimum=5, maximum=100, value=DEFAULT_DIFFUSION_STEPS_FAST, step=1,
                                info="Number of denoising steps. 'Fast' mode uses a fixed ~15 steps. 'Normal' mode uses the value set here.",
                                interactive=False
                            )
                        color_fix_dropdown = gr.Dropdown(
                            label="Color Correction",
                            choices=['AdaIN', 'Wavelet', 'None'], value=DEFAULT_COLOR_FIX_METHOD,
                            info="""Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""
                        )

                    with gr.Accordion("Performance & VRAM Optimization (Only for STAR Model)", open=True):
                        max_chunk_len_slider = gr.Slider(
                            label="Max Frames per Chunk (VRAM)",
                            minimum=4, maximum=96, value=DEFAULT_MAX_CHUNK_LEN, step=4,
                            info="""IMPORTANT for VRAM. This is the standard way the application manages VRAM. It divides the entire sequence of video frames into sequential, non-overlapping chunks.
- Mechanism: The STAR model processes one complete chunk (of this many frames) at a time.
- VRAM Impact: High Reduction (Lower value = Less VRAM).
- Bigger Chunk + Bigger VRAM = Faster Processing and Better Quality 
- Quality Impact: Can reduce Temporal Consistency (flicker/motion issues) between chunks if too low, as the model doesn't have context across chunk boundaries. Keep as high as VRAM allows.
- Speed Impact: Slower (Lower value = Slower, as more chunks are processed)."""
                        )
                        enable_chunk_optimization_check = gr.Checkbox(
                            label="Optimize Last Chunk Quality",
                            value=DEFAULT_ENABLE_CHUNK_OPTIMIZATION,
                            info="""Process extra frames for small last chunks to improve quality. When the last chunk has fewer frames than target size (causing quality drops), this processes additional frames but only keeps the necessary output.
- Example: For 70 frames with 32-frame chunks, instead of processing only 6 frames for the last chunk (poor quality), it processes 23 frames (48-70) but keeps only the last 6 (65-70).
- Quality Impact: Significantly improves quality for small last chunks.
- Speed Impact: Minimal impact on total processing time.
- VRAM Impact: No additional VRAM usage."""
                        )
                        vae_chunk_slider = gr.Slider(
                            label="VAE Decode Chunk (VRAM)",
                            minimum=1, maximum=16, value=DEFAULT_VAE_CHUNK, step=1,
                            info="""Controls max latent frames decoded back to pixels by VAE simultaneously.
- VRAM Impact: High Reduction (Lower value = Less VRAM during decode stage).
- Quality Impact: Minimal / Negligible. Safe to lower.
- Speed Impact: Slower (Lower value = Slower decoding)."""
                        )

                with gr.Column(scale=1):
                    # Image Upscaler Panel
                    with gr.Group():
                        gr.Markdown("### Image-Based Upscaler (Alternative to STAR)")
                        enable_image_upscaler_check = gr.Checkbox(
                            label="Enable Image-Based Upscaling (Disables STAR Model)",
                            value=DEFAULT_ENABLE_IMAGE_UPSCALER,
                            info="""Use deterministic image upscaler models instead of STAR. When enabled:
- Processes frames individually using spandrel-compatible models
- Ignores prompts, auto-caption, context window, and tiling settings
- Supports various architectures: DAT-2, ESRGAN, HAT, RCAN, OmniSR, CUGAN
- Much faster processing with batch support
- Uses way lesser VRAM
- Lower quality compared to high Max Frames per Chunk STAR model upscale"""
                        )

                        # Scan for available models
                        try:
                            available_model_files = util_scan_for_models(UPSCALE_MODELS_DIR, logger)
                            if available_model_files:
                                model_choices = available_model_files
                                default_model_choice = model_choices[0]
                            else:
                                model_choices = ["No models found - place models in upscale_models/"]
                                default_model_choice = model_choices[0]
                        except Exception as e:
                            logger.warning(f"Failed to scan for upscaler models: {e}")
                            model_choices = ["Error scanning models - check upscale_models/ directory"]
                            default_model_choice = model_choices[0]

                        image_upscaler_model_dropdown = gr.Dropdown(
                            label="Upscaler Model",
                            choices=model_choices,
                            value=default_model_choice,
                            info="Select the image upscaler model. Models should be placed in the 'upscale_models/' directory.",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                        image_upscaler_batch_size_slider = gr.Slider(
                            label="Batch Size",
                            minimum=DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE,
                            maximum=DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE,
                            value=DEFAULT_IMAGE_UPSCALER_BATCH_SIZE,
                            step=1,
                            info="Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage. Adjust based on your GPU memory.",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                    # Face Restoration Panel
                    with gr.Group():
                        gr.Markdown("### Face Restoration (CodeFormer)")
                        enable_face_restoration_check = gr.Checkbox(
                            label="Enable Face Restoration",
                            value=DEFAULT_ENABLE_FACE_RESTORATION,
                            info="""Enhance faces in the video using CodeFormer. Works with both STAR and image-based upscaling.
- Detects and restores faces automatically
- Can be applied before or after upscaling
- Supports both face restoration and colorization
- Requires CodeFormer models in pretrained_weight/ directory"""
                        )

                        face_restoration_fidelity_slider = gr.Slider(
                            label="Fidelity Weight",
                            minimum=0.0,
                            maximum=1.0,
                            value=DEFAULT_FACE_RESTORATION_FIDELITY,
                            step=0.1,
                            info="""Balance between quality and identity preservation:
- 0.0-0.3: Prioritize quality/detail (may change facial features)
- 0.4-0.6: Balanced approach
- 0.7-1.0: Prioritize identity preservation (may reduce enhancement)""",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                        with gr.Row():
                            enable_face_colorization_check = gr.Checkbox(
                                label="Enable Colorization",
                                value=DEFAULT_ENABLE_FACE_COLORIZATION,
                                info="Apply colorization to grayscale faces (experimental feature)",
                                interactive=False  # Will be enabled when checkbox is checked
                            )

                            face_restoration_when_radio = gr.Radio(
                                label="Apply Timing",
                                choices=['before', 'after'],
                                value=DEFAULT_FACE_RESTORATION_WHEN,
                                info="""When to apply face restoration:
'before': Apply before upscaling (may be enhanced further)
'after': Apply after upscaling (final enhancement)""",
                                interactive=False  # Will be enabled when checkbox is checked
                            )

                        with gr.Row():
                            # CodeFormer Model Selection
                            model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            default_model_choice = "Auto (Default)"

                            codeformer_model_dropdown = gr.Dropdown(
                                label="CodeFormer Model",
                                choices=model_choices,
                                value=default_model_choice,
                                info="Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory.",
                                interactive=False   # Will be enabled when checkbox is checked
                            )

                        face_restoration_batch_size_slider = gr.Slider(
                            label="Face Restoration Batch Size",
                            minimum=DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE,
                            maximum=DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE,
                            value=DEFAULT_FACE_RESTORATION_BATCH_SIZE,
                            step=1,
                            info="Number of frames to process simultaneously for face restoration. Higher values = faster processing but more VRAM usage.",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                    if UTIL_COG_VLM_AVAILABLE:
                        with gr.Accordion("Auto-Captioning Settings (CogVLM2) (Only for STAR Model)", open=True):
                            cogvlm_quant_choices_map = get_cogvlm_quant_choices_map(torch.cuda.is_available(), UTIL_BITSANDBYTES_AVAILABLE)
                            cogvlm_quant_radio_choices_display = list(cogvlm_quant_choices_map.values())
                            default_quant_display_val = get_default_cogvlm_quant_display(cogvlm_quant_choices_map)

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
                                    choices=['full', 'cpu'], value=DEFAULT_COGVLM_UNLOAD_AFTER_USE,
                                    info="""Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""
                                )
                    else:
                        gr.Markdown("_(Auto-captioning disabled as CogVLM2 components are not fully available.)_")
                        cogvlm_quant_radio = gr.State(None)
                        cogvlm_unload_radio = gr.State(None)

                    with gr.Accordion("Context Window - Previous Frames for Better Consistency (Only for STAR Model)", open=True):
                        enable_context_window_check = gr.Checkbox(
                            label="Enable Context Window",
                            value=DEFAULT_ENABLE_CONTEXT_WINDOW,
                            info="""Include previous frames as context when processing each chunk (except the first). Similar to "Optimize Last Chunk Quality" but applied to all chunks.
- Mechanism: Each chunk (except first) includes N previous frames as context, but only outputs new frames. Provides temporal consistency without complex overlap logic.
- Quality Impact: Better temporal consistency and reduced flickering between chunks. More context = better consistency.
- VRAM Impact: Medium increase due to processing context frames (recommend 25-50% of Max Frames per Chunk).
- Speed Impact: Slower due to processing additional context frames, but simpler and more predictable than traditional sliding window."""
                        )
                        context_overlap_num = gr.Slider(
                            label="Context Overlap (frames)",
                            value=DEFAULT_CONTEXT_OVERLAP, minimum=0, maximum=31, step=1,
                            info="Number of previous frames to include as context for each chunk (except first). 0 = disabled (same as normal chunking). Higher values = better consistency but more VRAM and slower processing. Recommend: 25-50% of Max Frames per Chunk."
                        )

                    with gr.Accordion("Advanced: Tiling (Very High Res / Low VRAM)", open=True, visible=False):
                        enable_tiling_check = gr.Checkbox(
                            label="Enable Tiled Upscaling",
                            value=DEFAULT_ENABLE_TILING,
                            info="""Processes each frame in small spatial patches (tiles). Use ONLY if necessary for extreme resolutions or very low VRAM.
- VRAM Impact: Very High Reduction.
- Quality Impact: High risk of tile seams/artifacts. Can harm global coherence and severely reduce temporal consistency.
- Speed Impact: Extremely Slow."""
                        )
                        with gr.Row():
                            tile_size_num = gr.Number(
                                label="Tile Size (px, input res)",
                                value=DEFAULT_TILE_SIZE, minimum=64, step=32,
                                info="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                            )
                            tile_overlap_num = gr.Number(
                                label="Tile Overlap (px, input res)",
                                value=DEFAULT_TILE_OVERLAP, minimum=0, step=16,
                                info="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                            )

                        image_upscaler_model_dropdown = gr.Dropdown(
                            label="Upscaler Model",
                            choices=model_choices,
                            value=default_model_choice,
                            info="Select the image upscaler model. Models should be placed in the 'upscale_models/' directory.",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                        image_upscaler_batch_size_slider = gr.Slider(
                            label="Batch Size",
                            minimum=DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE,
                            maximum=DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE,
                            value=DEFAULT_IMAGE_UPSCALER_BATCH_SIZE,
                            step=1,
                            info="Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage. Adjust based on your GPU memory.",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                    # Face Restoration Panel
                    with gr.Group():
                        gr.Markdown("### Face Restoration (CodeFormer)")
                        enable_face_restoration_check = gr.Checkbox(
                            label="Enable Face Restoration",
                            value=DEFAULT_ENABLE_FACE_RESTORATION,
                            info="""Enhance faces in the video using CodeFormer. Works with both STAR and image-based upscaling.
- Detects and restores faces automatically
- Can be applied before or after upscaling
- Supports both face restoration and colorization
- Requires CodeFormer models in pretrained_weight/ directory"""
                        )

                        face_restoration_fidelity_slider = gr.Slider(
                            label="Fidelity Weight",
                            minimum=0.0,
                            maximum=1.0,
                            value=DEFAULT_FACE_RESTORATION_FIDELITY,
                            step=0.1,
                            info="""Balance between quality and identity preservation:
- 0.0-0.3: Prioritize quality/detail (may change facial features)
- 0.4-0.6: Balanced approach
- 0.7-1.0: Prioritize identity preservation (may reduce enhancement)""",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                        with gr.Row():
                            enable_face_colorization_check = gr.Checkbox(
                                label="Enable Colorization",
                                value=DEFAULT_ENABLE_FACE_COLORIZATION,
                                info="Apply colorization to grayscale faces (experimental feature)",
                                interactive=False  # Will be enabled when checkbox is checked
                            )

                            face_restoration_when_radio = gr.Radio(
                                label="Apply Timing",
                                choices=['before', 'after'],
                                value=DEFAULT_FACE_RESTORATION_WHEN,
                                info="""When to apply face restoration:
'before': Apply before upscaling (may be enhanced further)
'after': Apply after upscaling (final enhancement)""",
                                interactive=False  # Will be enabled when checkbox is checked
                            )

                        with gr.Row():
                            # CodeFormer Model Selection
                            model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            default_model_choice = "Auto (Default)"

                            codeformer_model_dropdown = gr.Dropdown(
                                label="CodeFormer Model",
                                choices=model_choices,
                                value=default_model_choice,
                                info="Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory.",
                                interactive=False   # Will be enabled when checkbox is checked
                            )

                        face_restoration_batch_size_slider = gr.Slider(
                            label="Face Restoration Batch Size",
                            minimum=DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE,
                            maximum=DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE,
                            value=DEFAULT_FACE_RESTORATION_BATCH_SIZE,
                            step=1,
                            info="Number of frames to process simultaneously for face restoration. Higher values = faster processing but more VRAM usage.",
                            interactive=False  # Will be enabled when checkbox is checked
                        )

                    if UTIL_COG_VLM_AVAILABLE:
                        with gr.Accordion("Auto-Captioning Settings (CogVLM2) (Only for STAR Model)", open=True):
                            cogvlm_quant_choices_map = get_cogvlm_quant_choices_map(torch.cuda.is_available(), UTIL_BITSANDBYTES_AVAILABLE)
                            cogvlm_quant_radio_choices_display = list(cogvlm_quant_choices_map.values())
                            default_quant_display_val = get_default_cogvlm_quant_display(cogvlm_quant_choices_map)

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
                                    choices=['full', 'cpu'], value=DEFAULT_COGVLM_UNLOAD_AFTER_USE,
                                    info="""Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""
                                )
                    else:
                        gr.Markdown("_(Auto-captioning disabled as CogVLM2 components are not fully available.)_")
                        cogvlm_quant_radio = gr.State(None)
                        cogvlm_unload_radio = gr.State(None)

                    with gr.Accordion("Context Window - Previous Frames for Better Consistency (Only for STAR Model)", open=True):
                        enable_context_window_check = gr.Checkbox(
                            label="Enable Context Window",
                            value=DEFAULT_ENABLE_CONTEXT_WINDOW,
                            info="""Include previous frames as context when processing each chunk (except the first). Similar to "Optimize Last Chunk Quality" but applied to all chunks.
- Mechanism: Each chunk (except first) includes N previous frames as context, but only outputs new frames. Provides temporal consistency without complex overlap logic.
- Quality Impact: Better temporal consistency and reduced flickering between chunks. More context = better consistency.
- VRAM Impact: Medium increase due to processing context frames (recommend 25-50% of Max Frames per Chunk).
- Speed Impact: Slower due to processing additional context frames, but simpler and more predictable than traditional sliding window."""
                        )
                        context_overlap_num = gr.Slider(
                            label="Context Overlap (frames)",
                            value=DEFAULT_CONTEXT_OVERLAP, minimum=0, maximum=31, step=1,
                            info="Number of previous frames to include as context for each chunk (except first). 0 = disabled (same as normal chunking). Higher values = better consistency but more VRAM and slower processing. Recommend: 25-50% of Max Frames per Chunk."
                        )

                    with gr.Accordion("Advanced: Tiling (Very High Res / Low VRAM)", open=True, visible=False):
                        enable_tiling_check = gr.Checkbox(
                            label="Enable Tiled Upscaling",
                            value=DEFAULT_ENABLE_TILING,
                            info="""Processes each frame in small spatial patches (tiles). Use ONLY if necessary for extreme resolutions or very low VRAM.
- VRAM Impact: Very High Reduction.
- Quality Impact: High risk of tile seams/artifacts. Can harm global coherence and severely reduce temporal consistency.
- Speed Impact: Extremely Slow."""
                        )
                        with gr.Row():
                            tile_size_num = gr.Number(
                                label="Tile Size (px, input res)",
                                value=DEFAULT_TILE_SIZE, minimum=64, step=32,
                                info="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                            )
                            tile_overlap_num = gr.Number(
                                label="Tile Overlap (px, input res)",
                                value=DEFAULT_TILE_OVERLAP, minimum=0, step=16,
                                info="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                            )

        with gr.Tab("Output & Comparison", id="output_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("FFmpeg Encoding Settings", open=True):
                        ffmpeg_use_gpu_check = gr.Checkbox(
                            label="Use NVIDIA GPU for FFmpeg (h264_nvenc)",
                            value=DEFAULT_FFMPEG_USE_GPU,
                            info="If checked, uses NVIDIA's NVENC for FFmpeg video encoding (downscaling and final video creation). Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."
                        )
                        with gr.Row():
                            ffmpeg_preset_dropdown = gr.Dropdown(
                                label="FFmpeg Preset",
                                choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                                value=DEFAULT_FFMPEG_PRESET,
                                info="Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently (e.g. p1-p7 or specific names like 'slow', 'medium', 'fast')."
                            )
                            ffmpeg_quality_slider = gr.Slider(
                                label="FFmpeg Quality (CRF for libx264 / CQ for NVENC)",
                                minimum=0, maximum=51, value=DEFAULT_FFMPEG_QUALITY_CPU, step=1,
                                info="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28."
                            )
                        
                        frame_folder_fps_slider = gr.Slider(
                            label="Frame Folder FPS",
                            minimum=1.0, maximum=120.0, value=DEFAULT_FRAME_FOLDER_FPS, step=0.001,
                            info="FPS to use when converting frame folders to videos. This setting only applies when processing input frame folders (not regular videos). Common values: 23.976, 24, 25, 29.97, 30, 60."
                        )

                with gr.Column(scale=1):
                    with gr.Accordion("Output Options", open=True):
                        create_comparison_video_check = gr.Checkbox(
                            label="Generate Comparison Video",
                            value=DEFAULT_CREATE_COMPARISON_VIDEO,
                            info="""Create a side-by-side or top-bottom comparison video showing original vs upscaled quality.
The layout is automatically chosen based on aspect ratio to stay within 1920x1080 bounds when possible.
This helps visualize the quality improvement from upscaling."""
                        )
                        save_frames_checkbox = gr.Checkbox(
                            label="Save Input and Processed Frames", value=DEFAULT_SAVE_FRAMES,
                            info="If checked, saves the extracted input frames and the upscaled output frames into a subfolder named after the output video (e.g., '0001/input_frames' and '0001/processed_frames')."
                        )
                        save_metadata_checkbox = gr.Checkbox(
                            label="Save Processing Metadata", value=DEFAULT_SAVE_METADATA,
                            info="If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time."
                        )
                        save_chunks_checkbox = gr.Checkbox(
                            label="Save Processed Chunks", value=DEFAULT_SAVE_CHUNKS,
                            info="If checked, saves each processed chunk as a video file in a 'chunks' subfolder (e.g., '0001/chunks/chunk_0001.mp4'). Uses the same FFmpeg settings as the final video."
                        )
                        save_chunk_frames_checkbox = gr.Checkbox(
                            label="Save Chunk Input Frames (Debug)", value=DEFAULT_SAVE_CHUNK_FRAMES,
                            info="If checked, saves the input frames for each chunk before processing into a 'chunk_frames' subfolder (e.g., '0001/chunk_frames/chunk_01_frame_012.png'). Useful for debugging which frames are processed in each chunk."
                        )

                    with gr.Accordion("Advanced: Seeding (Reproducibility)", open=True):
                        with gr.Row():
                            seed_num = gr.Number(
                                label="Seed",
                                value=DEFAULT_SEED,
                                minimum=-1,
                                maximum=2**32 - 1,
                                step=1,
                                info="Seed for random number generation. Used for reproducibility. Set to -1 or check 'Random Seed' for a random seed. Value is ignored if 'Random Seed' is checked.",
                                interactive=not DEFAULT_RANDOM_SEED
                            )
                            random_seed_check = gr.Checkbox(
                                label="Random Seed",
                                value=DEFAULT_RANDOM_SEED,
                                info="If checked, a random seed will be generated and used, ignoring the 'Seed' value."
                            )

                with gr.Column(scale=1):
                    with gr.Accordion("Manual Comparison Video Generator", open=True):
                        gr.Markdown("### Generate Custom Comparison Videos")
                        gr.Markdown("Upload two videos to create a manual side-by-side or top-bottom comparison video using the same FFmpeg settings and layout logic as the automatic comparison feature.")

                        gr.Markdown("**Step 1:** Upload the original or reference video for comparison")
                        manual_original_video = gr.Video(
                            label="Original/Reference Video",
                            sources=["upload"],
                            interactive=True,
                            height=200
                        )

                        gr.Markdown("**Step 2:** Upload the upscaled or enhanced video for comparison")
                        manual_upscaled_video = gr.Video(
                            label="Upscaled/Enhanced Video",
                            sources=["upload"],
                            interactive=True,
                            height=200
                        )

                        gr.Markdown("**Step 3:** Generate the comparison video using current FFmpeg settings")
                        manual_comparison_button = gr.Button(
                            "Generate Manual Comparison Video",
                            variant="primary",
                            size="lg"
                        )

                        manual_comparison_status = gr.Textbox(
                            label="Manual Comparison Status",
                            lines=2,
                            interactive=False,
                            visible=False
                        )

            with gr.Accordion("Comparison Video To See Difference", open=True):
                comparison_video = gr.Video(label="Comparison Video", interactive=False, height=512)

        with gr.Tab("Batch Upscaling", id="batch_tab"):
            with gr.Accordion("Batch Processing Options", open=True):
                with gr.Row():
                    batch_input_folder = gr.Textbox(
                        label="Input Folder",
                        placeholder="Path to folder containing videos to process...",
                        info="Folder containing video files to process in batch mode."
                    )
                    batch_output_folder = gr.Textbox(
                        label="Output Folder",
                        placeholder="Path to output folder for processed videos...",
                        info="Folder where processed videos will be saved with organized structure."
                    )
                
                with gr.Row():
                    enable_batch_frame_folders = gr.Checkbox(
                        label="Process Frame Folders in Batch",
                        value=False,
                        info="Enable to process subfolders containing frame sequences instead of video files. Each subfolder with images will be converted to video first."
                    )

                with gr.Row():
                    batch_skip_existing = gr.Checkbox(
                        label="Skip Existing Outputs",
                        value=DEFAULT_BATCH_SKIP_EXISTING,
                        info="Skip processing if the output file already exists. Useful for resuming interrupted batch jobs."
                    )

                    batch_use_prompt_files = gr.Checkbox(
                        label="Use Prompt Files (filename.txt)",
                        value=DEFAULT_BATCH_USE_PROMPT_FILES,
                        info="Look for text files with same name as video (e.g., video.txt) to use as custom prompts. Takes priority over user prompt and auto-caption."
                    )

                    batch_save_captions = gr.Checkbox(
                        label="Save Auto-Generated Captions",
                        value=DEFAULT_BATCH_SAVE_CAPTIONS,
                        info="Save auto-generated captions as filename.txt in the input folder for future reuse. Never overwrites existing prompt files."
                    )

                if UTIL_COG_VLM_AVAILABLE:
                    with gr.Row():
                        batch_enable_auto_caption = gr.Checkbox(
                            label="Enable Auto-Caption for Batch",
                            value=True,
                            info="Generate automatic captions for videos that don't have prompt files. Uses the same CogVLM2 settings as Core Settings tab."
                        )
                else:
                    batch_enable_auto_caption = gr.Checkbox(visible=False, value=False)

            with gr.Row():
                batch_process_button = gr.Button("Start Batch Upscaling", variant="primary", icon="icons/split.png")

            # Add the help content at the bottom in 3 columns
            create_batch_processing_help()

        with gr.Tab("FPS Increase - Decrease", id="fps_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("RIFE Interpolation Settings", open=True):
                        gr.Markdown("### Frame Interpolation (RIFE)")
                        gr.Markdown("**RIFE (Real-time Intermediate Flow Estimation)** uses AI to intelligently generate intermediate frames between existing frames, increasing video smoothness and frame rate.")

                        enable_rife_interpolation = gr.Checkbox(
                            label="Enable RIFE Interpolation",
                            value=DEFAULT_RIFE_ENABLE_INTERPOLATION,
                            info="Enable AI-powered frame interpolation to increase video FPS. When enabled, RIFE will be applied to the final upscaled video and can optionally be applied to intermediate chunks and scenes."
                        )

                        with gr.Row():
                            rife_multiplier = gr.Radio(
                                label="FPS Multiplier",
                                choices=[2, 4],
                                value=DEFAULT_RIFE_MULTIPLIER,
                                info="Choose how much to increase the frame rate. 2x doubles FPS (e.g., 24→48), 4x quadruples FPS (e.g., 24→96)."
                            )

                        with gr.Row():
                            rife_fp16 = gr.Checkbox(
                                label="Use FP16 Precision",
                                value=DEFAULT_RIFE_FP16,
                                info="Use half-precision floating point for faster processing and lower VRAM usage. Recommended for most users."
                            )
                            rife_uhd = gr.Checkbox(
                                label="UHD Mode",
                                value=DEFAULT_RIFE_UHD,
                                info="Enable UHD mode for 4K+ videos. May improve quality for very high resolution content but requires more VRAM."
                            )

                        rife_scale = gr.Slider(
                            label="Scale Factor",
                            minimum=0.25, maximum=2.0, value=DEFAULT_RIFE_SCALE, step=0.25,
                            info="Scale factor for RIFE processing. 1.0 = original size. Lower values use less VRAM but may reduce quality. Higher values may improve quality but use more VRAM."
                        )

                        rife_skip_static = gr.Checkbox(
                            label="Skip Static Frames",
                            value=DEFAULT_RIFE_SKIP_STATIC,
                            info="Automatically detect and skip interpolating static (non-moving) frames to save processing time and avoid unnecessary interpolation."
                        )

                    with gr.Accordion("Intermediate Processing", open=True):
                        gr.Markdown("**Apply RIFE to intermediate videos (recommended)**")
                        rife_apply_to_chunks = gr.Checkbox(
                            label="Apply to Chunks",
                            value=DEFAULT_RIFE_APPLY_TO_CHUNKS,
                            info="Apply RIFE interpolation to individual video chunks during processing. Enabled by default for smoother intermediate results."
                        )
                        rife_apply_to_scenes = gr.Checkbox(
                            label="Apply to Scenes",
                            value=DEFAULT_RIFE_APPLY_TO_SCENES,
                            info="Apply RIFE interpolation to individual scene videos when scene splitting is enabled. Enabled by default for consistent results."
                        )

                        gr.Markdown("**Note:** When RIFE is enabled, the system will return RIFE-interpolated versions to the interface instead of originals, ensuring you get the smoothest possible results throughout the processing pipeline.")

                with gr.Column(scale=1):
                    with gr.Accordion("FPS Decrease", open=True):
                        gr.Markdown("### Pre-Processing FPS Reduction")
                        gr.Markdown("**Reduce FPS before upscaling** to speed up processing and reduce VRAM usage. You can then use RIFE interpolation to restore smooth motion afterward.")

                        enable_fps_decrease = gr.Checkbox(
                            label="Enable FPS Decrease",
                            value=DEFAULT_ENABLE_FPS_DECREASE,
                            info="Reduce video FPS before upscaling to speed up processing. Fewer frames = faster upscaling and lower VRAM usage."
                        )

                        fps_decrease_mode = gr.Radio(
                            label="FPS Reduction Mode",
                            choices=["multiplier", "fixed"],
                            value=DEFAULT_FPS_DECREASE_MODE,
                            info="Multiplier: Reduce by fraction (1/2x, 1/4x). Fixed: Set specific FPS value. Multiplier is recommended for automatic adaptation to input video."
                        )

                        with gr.Group() as multiplier_controls:
                            with gr.Row():
                                fps_multiplier_preset = gr.Dropdown(
                                    label="FPS Multiplier",
                                    choices=list(util_get_common_fps_multipliers().values()) + ["Custom"],
                                    value="1/2x (Half FPS)",
                                    info="Choose common multiplier. 1/2x is recommended for good speed/quality balance."
                                )
                                fps_multiplier_custom = gr.Number(
                                    label="Custom Multiplier",
                                    value=DEFAULT_FPS_MULTIPLIER,
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.05,
                                    precision=2,
                                    visible=False,
                                    info="Custom multiplier value (0.1 to 1.0). Lower = fewer frames."
                                )

                        with gr.Group(visible=False) as fixed_controls:
                            target_fps = gr.Slider(
                                label="Target FPS",
                                minimum=1.0,
                                maximum=60.0,
                                value=DEFAULT_TARGET_FPS,
                                step=0.001,
                                info="Target FPS for the reduced video. Lower FPS = faster upscaling. Common choices: 12-15 FPS for fast processing, 24 FPS for cinema standard. Supports precise values like 23.976."
                            )

                        fps_interpolation_method = gr.Radio(
                            label="Frame Reduction Method",
                            choices=["drop", "blend"],
                            value=DEFAULT_FPS_INTERPOLATION_METHOD,
                            info="Drop: Faster, simply removes frames. Blend: Smoother, blends frames together (slower but may preserve motion better)."
                        )

                        fps_calculation_info = gr.Markdown(
                            "**📊 Calculation:** Upload a video to see FPS reduction preview",
                            visible=True
                        )

                        gr.Markdown("**💡 Workflow Tip:** Use FPS decrease (1/2x for balanced speed/quality) for faster upscaling, then enable RIFE 2x-4x to restore smooth 24-60 FPS output!")

                    with gr.Accordion("FPS Limiting & Output Control", open=True):
                        rife_enable_fps_limit = gr.Checkbox(
                            label="Enable FPS Limiting",
                            value=DEFAULT_RIFE_ENABLE_FPS_LIMIT,
                            info="Limit the output FPS to specific common values instead of unlimited interpolation. Useful for compatibility with displays and media players."
                        )

                        rife_max_fps_limit = gr.Radio(
                            label="Max FPS Limit",
                            choices=[23.976, 24, 25, 29.970, 30, 47.952, 48, 50, 59.940, 60, 75, 90, 100, 119.880, 120, 144, 165, 180, 240, 360],
                            value=DEFAULT_RIFE_MAX_FPS_LIMIT,
                            info="Maximum FPS when limiting is enabled. NTSC rates: 23.976/29.970/59.940 (film/TV), Standard: 24/25/30/50/60, Gaming: 120/144/240+. Choose based on your target format and display."
                        )

                        with gr.Row():
                            rife_keep_original = gr.Checkbox(
                                label="Keep Original Files",
                                value=DEFAULT_RIFE_KEEP_ORIGINAL,
                                info="Keep the original (non-interpolated) video files alongside the RIFE-processed versions. Recommended to compare results."
                            )
                            rife_overwrite_original = gr.Checkbox(
                                label="Overwrite Original",
                                value=DEFAULT_RIFE_OVERWRITE_ORIGINAL,
                                info="Replace the original upscaled video with the RIFE version as the primary output. When disabled, both versions are available."
                            )

        with gr.Tab("Edit Videos", id="edit_tab"):
            gr.Markdown("# Video Editor - Cut and Extract Video Segments")
            gr.Markdown("**Cut specific time ranges or frame ranges from your videos with precise FFmpeg encoding.**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Video Upload Section
                    with gr.Group():
                        input_video_edit = gr.Video(
                            label="Input Video for Editing",
                            sources=["upload"],
                            interactive=True,
                            height=300
                        )
                        
                        # Video Information Display
                        video_info_display = gr.Textbox(
                            label="Video Information",
                            interactive=False,
                            lines=6,
                            info="Shows duration, FPS, frame count, resolution",
                            value="📹 Upload a video to see detailed information"
                        )

                    # Action Buttons
                    with gr.Row():
                        cut_and_save_btn = gr.Button("Cut and Save", variant="primary", icon="icons/cut_paste.png")
                        cut_and_upscale_btn = gr.Button("Cut and Move to Upscale", variant="primary", icon="icons/move_icon.png")
                    
                    # Cutting Mode Selection
                    with gr.Accordion("Cutting Settings", open=True):
                        cutting_mode = gr.Radio(
                            label="Cutting Mode",
                            choices=['time_ranges', 'frame_ranges'],
                            value='time_ranges',
                            info="Choose between time-based or frame-based cutting"
                        )
                        
                        # Time Range Inputs (visible when time_ranges selected)
                        with gr.Group() as time_range_controls:
                            time_ranges_input = gr.Textbox(
                                label="Time Ranges (seconds)",
                                placeholder="1-3,5-8,10-15 or 1:30-2:45,3:00-4:30",
                                info="Format: start1-end1,start2-end2,... (supports decimal seconds and MM:SS format)",
                                lines=2
                            )
                        
                        # Frame Range Inputs (hidden when time_ranges selected)
                        with gr.Group(visible=False) as frame_range_controls:
                            frame_ranges_input = gr.Textbox(
                                label="Frame Ranges",
                                placeholder="30-90,150-210,300-450",
                                info="Format: start1-end1,start2-end2,... (frame numbers are 0-indexed)",
                                lines=2
                            )
                        
                        # Cut Information Display
                        cut_info_display = gr.Textbox(
                            label="Cut Analysis",
                            interactive=False,
                            lines=3,
                            info="Shows details about the cuts being made",
                            value="✏️ Enter ranges above to see cut analysis"
                        )
                    
                    # Precision and Preview Options
                    with gr.Accordion("Options", open=True):
                        precise_cutting_mode = gr.Radio(
                            label="Cutting Precision",
                            choices=['precise', 'fast'],
                            value='precise',
                            info="Precise: Frame-accurate re-encoding. Fast: Stream copy (faster but may be less accurate)"
                        )
                        
                        preview_first_segment = gr.Checkbox(
                            label="Generate Preview of First Segment",
                            value=True,
                            info="Create a preview video of the first cut segment for verification"
                        )
                        
                        # Processing Time Estimate
                        processing_estimate = gr.Textbox(
                            label="Processing Time Estimate",
                            interactive=False,
                            lines=1,
                            value="📊 Upload video and enter ranges to see time estimate"
                        )

                with gr.Column(scale=1):
                    # Output Video Display
                    with gr.Group():
                        output_video_edit = gr.Video(
                            label="Cut Video Output",
                            interactive=False,
                            height=400
                        )
                        
                        # Preview Video (for first segment)
                        preview_video_edit = gr.Video(
                            label="Preview (First Segment)",
                            interactive=False,
                            height=300
                        )
                    
                    # Status and Progress
                    with gr.Group():
                        edit_status_textbox = gr.Textbox(
                            label="Edit Status & Log",
                            interactive=False,
                            lines=8,
                            max_lines=15,
                            value="🎞️ Ready to edit videos. Upload a video and specify cut ranges to begin."
                        )
                    
                    # Quick Help
                    with gr.Accordion("Quick Help & Examples", open=False):
                        gr.Markdown("""
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

        with gr.Tab("Face Restoration", id="face_restoration_tab"):
            gr.Markdown("# Standalone Face Restoration - CodeFormer Processing")
            gr.Markdown("**Apply face restoration to videos using CodeFormer without upscaling. Perfect for improving face quality in existing videos.**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Video Upload Section
                    with gr.Group():
                        input_video_face_restoration = gr.Video(
                            label="Input Video for Face Restoration",
                            sources=["upload"],
                            interactive=True,
                            height=400
                        )
                        
                        # Face Restoration Mode Selection
                        face_restoration_mode = gr.Radio(
                            label="Processing Mode",
                            choices=["Single Video", "Batch Folder"],
                            value="Single Video",
                            info="Choose between processing a single video or batch processing a folder of videos"
                        )
                        
                        # Batch folder inputs (hidden by default)
                        with gr.Group(visible=False) as batch_folder_controls:
                            batch_input_folder_face = gr.Textbox(
                                label="Input Folder Path",
                                placeholder="C:/path/to/input/videos/",
                                info="Folder containing videos to process with face restoration"
                            )
                            batch_output_folder_face = gr.Textbox(
                                label="Output Folder Path", 
                                placeholder="C:/path/to/output/videos/",
                                info="Folder where face-restored videos will be saved"
                            )

                    # Processing Controls
                    with gr.Row():
                        face_restoration_process_btn = gr.Button("Process Face Restoration", variant="primary", icon="icons/face_restoration.png")
                        face_restoration_stop_btn = gr.Button("Stop Processing", variant="stop")

                    # Face Restoration Settings
                    with gr.Accordion("Face Restoration Settings", open=True):
                        standalone_enable_face_restoration = gr.Checkbox(
                            label="Enable Face Restoration",
                            value=True,
                            info="Enable CodeFormer face restoration processing. Must be enabled for any processing to occur."
                        )
                        
                        standalone_face_restoration_fidelity = gr.Slider(
                            label="Face Restoration Fidelity Weight",
                            minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                            info="Balance between quality (0.3) and identity preservation (0.8). 0.7 is recommended for most videos."
                        )
                        
                        standalone_enable_face_colorization = gr.Checkbox(
                            label="Enable Face Colorization",
                            value=False,
                            info="Enable colorization for grayscale faces. Useful for old black & white videos or grayscale content."
                        )
                        
                        with gr.Row():
                            # Since CodeFormer models are always available, use a simple dropdown
                            standalone_model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            standalone_default_model_choice = "Auto (Default)"
                            
                            standalone_codeformer_model_dropdown = gr.Dropdown(
                                label="CodeFormer Model",
                                choices=standalone_model_choices,
                                value=standalone_default_model_choice,
                                info="Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory."
                            )
                        
                        standalone_face_restoration_batch_size = gr.Slider(
                            label="Processing Batch Size",
                            minimum=1, maximum=16, value=4, step=1,
                            info="Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage."
                        )

                    # Processing Options
                    with gr.Accordion("Advanced Options", open=False):
                        standalone_save_frames = gr.Checkbox(
                            label="Save Individual Frames",
                            value=False,
                            info="Save processed frames as individual image files alongside the video"
                        )
                        
                        standalone_create_comparison = gr.Checkbox(
                            label="Create Before/After Comparison Video",
                            value=True,
                            info="Create a side-by-side comparison video showing original vs face-restored results"
                        )
                        
                        standalone_preserve_audio = gr.Checkbox(
                            label="Preserve Original Audio",
                            value=True,
                            info="Keep the original audio track in the processed video"
                        )

                with gr.Column(scale=1):
                    # Output Video Display
                    with gr.Group():
                        output_video_face_restoration = gr.Video(
                            label="Face Restored Video",
                            interactive=False,
                            height=400
                        )
                        
                        # Comparison Video (if enabled)
                        comparison_video_face_restoration = gr.Video(
                            label="Before/After Comparison",
                            interactive=False,
                            height=300,
                            visible=True
                        )
                    
                    # Status and Progress
                    with gr.Group():
                        face_restoration_status = gr.Textbox(
                            label="Face Restoration Status & Log",
                            interactive=False,
                            lines=10,
                            max_lines=20,
                            value="🎭 Ready for face restoration processing. Upload a video and configure settings to begin."
                        )
                    
                    # Processing Statistics
                    with gr.Accordion("Processing Statistics", open=True):
                        face_restoration_stats = gr.Textbox(
                            label="Processing Stats",
                            interactive=False,
                            lines=4,
                            value="📊 Processing statistics will appear here during face restoration."
                        )
                    
                    # Quick Help
                    with gr.Accordion("Face Restoration Help", open=False):
                        gr.Markdown("""
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

        # Event handlers and interactions
    def update_steps_display(mode):

        with gr.Tab("Edit Videos", id="edit_tab"):
            gr.Markdown("# Video Editor - Cut and Extract Video Segments")
            gr.Markdown("**Cut specific time ranges or frame ranges from your videos with precise FFmpeg encoding.**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Video Upload Section
                    with gr.Group():
                        input_video_edit = gr.Video(
                            label="Input Video for Editing",
                            sources=["upload"],
                            interactive=True,
                            height=300
                        )
                        
                        # Video Information Display
                        video_info_display = gr.Textbox(
                            label="Video Information",
                            interactive=False,
                            lines=6,
                            info="Shows duration, FPS, frame count, resolution",
                            value="📹 Upload a video to see detailed information"
                        )

                    # Action Buttons
                    with gr.Row():
                        cut_and_save_btn = gr.Button("Cut and Save", variant="primary", icon="icons/cut_paste.png")
                        cut_and_upscale_btn = gr.Button("Cut and Move to Upscale", variant="primary", icon="icons/move_icon.png")
                    
                    # Cutting Mode Selection
                    with gr.Accordion("Cutting Settings", open=True):
                        cutting_mode = gr.Radio(
                            label="Cutting Mode",
                            choices=['time_ranges', 'frame_ranges'],
                            value='time_ranges',
                            info="Choose between time-based or frame-based cutting"
                        )
                        
                        # Time Range Inputs (visible when time_ranges selected)
                        with gr.Group() as time_range_controls:
                            time_ranges_input = gr.Textbox(
                                label="Time Ranges (seconds)",
                                placeholder="1-3,5-8,10-15 or 1:30-2:45,3:00-4:30",
                                info="Format: start1-end1,start2-end2,... (supports decimal seconds and MM:SS format)",
                                lines=2
                            )
                        
                        # Frame Range Inputs (hidden when time_ranges selected)
                        with gr.Group(visible=False) as frame_range_controls:
                            frame_ranges_input = gr.Textbox(
                                label="Frame Ranges",
                                placeholder="30-90,150-210,300-450",
                                info="Format: start1-end1,start2-end2,... (frame numbers are 0-indexed)",
                                lines=2
                            )
                        
                        # Cut Information Display
                        cut_info_display = gr.Textbox(
                            label="Cut Analysis",
                            interactive=False,
                            lines=3,
                            info="Shows details about the cuts being made",
                            value="✏️ Enter ranges above to see cut analysis"
                        )
                    
                    # Precision and Preview Options
                    with gr.Accordion("Options", open=True):
                        precise_cutting_mode = gr.Radio(
                            label="Cutting Precision",
                            choices=['precise', 'fast'],
                            value='precise',
                            info="Precise: Frame-accurate re-encoding. Fast: Stream copy (faster but may be less accurate)"
                        )
                        
                        preview_first_segment = gr.Checkbox(
                            label="Generate Preview of First Segment",
                            value=True,
                            info="Create a preview video of the first cut segment for verification"
                        )
                        
                        # Processing Time Estimate
                        processing_estimate = gr.Textbox(
                            label="Processing Time Estimate",
                            interactive=False,
                            lines=1,
                            value="📊 Upload video and enter ranges to see time estimate"
                        )
                    
                with gr.Column(scale=1):
                    # Output Video Display
                    with gr.Group():
                        output_video_edit = gr.Video(
                            label="Cut Video Output",
                            interactive=False,
                            height=400
                        )
                        
                        # Preview Video (for first segment)
                        preview_video_edit = gr.Video(
                            label="Preview (First Segment)",
                            interactive=False,
                            height=300
                        )
                    
                    # Status and Progress
                    with gr.Group():
                        edit_status_textbox = gr.Textbox(
                            label="Edit Status & Log",
                            interactive=False,
                            lines=8,
                            max_lines=15,
                            value="🎞️ Ready to edit videos. Upload a video and specify cut ranges to begin."
                        )
                    
                    # Quick Help
                    with gr.Accordion("Quick Help & Examples", open=False):
                        gr.Markdown("""
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

        with gr.Tab("Face Restoration", id="face_restoration_tab"):
            gr.Markdown("# Standalone Face Restoration - CodeFormer Processing")
            gr.Markdown("**Apply face restoration to videos using CodeFormer without upscaling. Perfect for improving face quality in existing videos.**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Video Upload Section
                    with gr.Group():
                        input_video_face_restoration = gr.Video(
                            label="Input Video for Face Restoration",
                            sources=["upload"],
                            interactive=True,
                            height=400
                        )
                        
                        # Face Restoration Mode Selection
                        face_restoration_mode = gr.Radio(
                            label="Processing Mode",
                            choices=["Single Video", "Batch Folder"],
                            value="Single Video",
                            info="Choose between processing a single video or batch processing a folder of videos"
                        )
                        
                        # Batch folder inputs (hidden by default)
                        with gr.Group(visible=False) as batch_folder_controls:
                            batch_input_folder_face = gr.Textbox(
                                label="Input Folder Path",
                                placeholder="C:/path/to/input/videos/",
                                info="Folder containing videos to process with face restoration"
                            )
                            batch_output_folder_face = gr.Textbox(
                                label="Output Folder Path", 
                                placeholder="C:/path/to/output/videos/",
                                info="Folder where face-restored videos will be saved"
                            )

                    # Processing Controls
                    with gr.Row():
                        face_restoration_process_btn = gr.Button("Process Face Restoration", variant="primary", icon="icons/face_restoration.png")
                        face_restoration_stop_btn = gr.Button("Stop Processing", variant="stop")

                    # Face Restoration Settings
                    with gr.Accordion("Face Restoration Settings", open=True):
                        standalone_enable_face_restoration = gr.Checkbox(
                            label="Enable Face Restoration",
                            value=True,
                            info="Enable CodeFormer face restoration processing. Must be enabled for any processing to occur."
                        )
                        
                        standalone_face_restoration_fidelity = gr.Slider(
                            label="Face Restoration Fidelity Weight",
                            minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                            info="Balance between quality (0.3) and identity preservation (0.8). 0.7 is recommended for most videos."
                        )
                        
                        standalone_enable_face_colorization = gr.Checkbox(
                            label="Enable Face Colorization",
                            value=False,
                            info="Enable colorization for grayscale faces. Useful for old black & white videos or grayscale content."
                        )
                        
                        with gr.Row():
                            # Since CodeFormer models are always available, use a simple dropdown
                            standalone_model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            standalone_default_model_choice = "Auto (Default)"
                            
                            standalone_codeformer_model_dropdown = gr.Dropdown(
                                label="CodeFormer Model",
                                choices=standalone_model_choices,
                                value=standalone_default_model_choice,
                                info="Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory."
                            )
                        
                        standalone_face_restoration_batch_size = gr.Slider(
                            label="Processing Batch Size",
                            minimum=1, maximum=16, value=4, step=1,
                            info="Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage."
                        )

                    # Processing Options
                    with gr.Accordion("Advanced Options", open=False):
                        standalone_save_frames = gr.Checkbox(
                            label="Save Individual Frames",
                            value=False,
                            info="Save processed frames as individual image files alongside the video"
                        )
                        
                        standalone_create_comparison = gr.Checkbox(
                            label="Create Before/After Comparison Video",
                            value=True,
                            info="Create a side-by-side comparison video showing original vs face-restored results"
                        )
                        
                        standalone_preserve_audio = gr.Checkbox(
                            label="Preserve Original Audio",
                            value=True,
                            info="Keep the original audio track in the processed video"
                        )

                with gr.Column(scale=1):
                    # Output Video Display
                    with gr.Group():
                        output_video_face_restoration = gr.Video(
                            label="Face Restored Video",
                            interactive=False,
                            height=400
                        )
                        
                        # Comparison Video (if enabled)
                        comparison_video_face_restoration = gr.Video(
                            label="Before/After Comparison",
                            interactive=False,
                            height=300,
                            visible=True
                        )
                    
                    # Status and Progress
                    with gr.Group():
                        face_restoration_status = gr.Textbox(
                            label="Face Restoration Status & Log",
                            interactive=False,
                            lines=10,
                            max_lines=20,
                            value="🎭 Ready for face restoration processing. Upload a video and configure settings to begin."
                        )
                    
                    # Processing Statistics
                    with gr.Accordion("Processing Statistics", open=True):
                        face_restoration_stats = gr.Textbox(
                            label="Processing Stats",
                            interactive=False,
                            lines=4,
                            value="📊 Processing statistics will appear here during face restoration."
                        )
                    
                    # Quick Help
                    with gr.Accordion("Face Restoration Help", open=False):
                        gr.Markdown("""
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

    # Initialize global variables
cogvlm_display_to_quant_val_map_global = {}
if UTIL_COG_VLM_AVAILABLE:
    _temp_map = get_cogvlm_quant_choices_map(torch.cuda.is_available(), UTIL_BITSANDBYTES_AVAILABLE)
    cogvlm_display_to_quant_val_map_global = {v: k for k, v in _temp_map.items()}

def get_quant_value_from_display(display_val):
    """Convert CogVLM quantization display value to actual value"""
    if display_val is None:
        return 0
    if isinstance(display_val, int):
        return display_val
    return cogvlm_display_to_quant_val_map_global.get(display_val, 0)

def extract_codeformer_model_path_from_dropdown(dropdown_choice):
    """Extract the actual model path from the dropdown choice."""
    if not dropdown_choice or dropdown_choice.startswith("No CodeFormer") or dropdown_choice.startswith("Error"):
        return None
    if dropdown_choice == "Auto (Default)":
        return None  # Use default model
    
    # For the specific codeformer.pth model, return the known path
    if dropdown_choice.startswith("codeformer.pth"):
        return os.path.join(FACE_RESTORATION_MODELS_DIR, "CodeFormer", "codeformer.pth")
    
    return None

def create_processing_params_from_gradio_inputs(*gradio_values):
    """Convert all Gradio input values into ProcessingParams dataclass"""
    # Unpack the 70+ Gradio input values in the correct order
    (input_video, user_prompt, pos_prompt, neg_prompt, auto_caption_then_upscale_check, gpu_selector,
     enable_frame_folder_check, input_frames_folder, model_selector, upscale_factor_slider, cfg_slider, 
     steps_slider, solver_mode_radio, color_fix_dropdown, max_chunk_len_slider, enable_chunk_optimization_check, 
     vae_chunk_slider, enable_image_upscaler_check, image_upscaler_model_dropdown, image_upscaler_batch_size_slider,
     enable_face_restoration_check, face_restoration_fidelity_slider, enable_face_colorization_check, 
     face_restoration_when_radio, codeformer_model_dropdown, face_restoration_batch_size_slider,
     cogvlm_quant_radio, cogvlm_unload_radio, enable_context_window_check, context_overlap_num,
     enable_tiling_check, tile_size_num, tile_overlap_num, enable_target_res_check, target_h_num, 
     target_w_num, target_res_mode_radio, ffmpeg_use_gpu_check, ffmpeg_preset_dropdown, ffmpeg_quality_slider,
     frame_folder_fps_slider, create_comparison_video_check, save_frames_checkbox, save_metadata_checkbox,
     save_chunks_checkbox, save_chunk_frames_checkbox, seed_num, random_seed_check,
     enable_scene_split_check, scene_split_mode_radio, scene_min_scene_len_num, scene_drop_short_check,
     scene_merge_last_check, scene_frame_skip_num, scene_threshold_num, scene_min_content_val_num,
     scene_frame_window_num, scene_copy_streams_check, scene_use_mkvmerge_check, scene_rate_factor_num,
     scene_preset_dropdown, scene_quiet_ffmpeg_check, scene_manual_split_type_radio, scene_manual_split_value_num,
     enable_fps_decrease, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps,
     fps_interpolation_method, enable_rife_interpolation, rife_multiplier, rife_fp16, rife_uhd, rife_scale,
     rife_skip_static, rife_enable_fps_limit, rife_max_fps_limit, rife_apply_to_chunks, rife_apply_to_scenes,
     rife_keep_original, rife_overwrite_original) = gradio_values

    # Create all parameter dataclasses
    prompts = PromptParams(
        user_prompt=user_prompt or "",
        positive_prompt=pos_prompt or DEFAULT_POS_PROMPT,
        negative_prompt=neg_prompt or DEFAULT_NEG_PROMPT
    )
    
    core = CoreParams(
        model_choice=model_selector or DEFAULT_MODEL_CHOICE,
        upscale_factor=upscale_factor_slider or DEFAULT_UPSCALE_FACTOR,
        cfg_scale=cfg_slider or DEFAULT_CFG_SCALE,
        steps=steps_slider or DEFAULT_DIFFUSION_STEPS_FAST,
        solver_mode=solver_mode_radio or DEFAULT_SOLVER_MODE,
        color_fix_method=color_fix_dropdown or DEFAULT_COLOR_FIX_METHOD
    )
    
    performance = PerformanceParams(
        max_chunk_len=max_chunk_len_slider or DEFAULT_MAX_CHUNK_LEN,
        enable_chunk_optimization=enable_chunk_optimization_check if enable_chunk_optimization_check is not None else DEFAULT_ENABLE_CHUNK_OPTIMIZATION,
        vae_chunk=vae_chunk_slider or DEFAULT_VAE_CHUNK
    )
    
    tiling = TilingParams(
        enable_tiling=enable_tiling_check if enable_tiling_check is not None else DEFAULT_ENABLE_TILING,
        tile_size=tile_size_num or DEFAULT_TILE_SIZE,
        tile_overlap=tile_overlap_num or DEFAULT_TILE_OVERLAP
    )
    
    context_window = ContextWindowParams(
        enable_context_window=enable_context_window_check if enable_context_window_check is not None else DEFAULT_ENABLE_CONTEXT_WINDOW,
        context_overlap=context_overlap_num or DEFAULT_CONTEXT_OVERLAP
    )
    
    target_resolution = TargetResolutionParams(
        enable_target_res=enable_target_res_check if enable_target_res_check is not None else DEFAULT_ENABLE_TARGET_RES,
        target_h=target_h_num or DEFAULT_TARGET_H,
        target_w=target_w_num or DEFAULT_TARGET_W,
        target_res_mode=target_res_mode_radio or DEFAULT_TARGET_RES_MODE
    )
    
    ffmpeg = FFmpegParams(
        use_gpu=ffmpeg_use_gpu_check if ffmpeg_use_gpu_check is not None else DEFAULT_FFMPEG_USE_GPU,
        preset=ffmpeg_preset_dropdown or DEFAULT_FFMPEG_PRESET,
        quality=ffmpeg_quality_slider or (DEFAULT_FFMPEG_QUALITY_GPU if ffmpeg_use_gpu_check else DEFAULT_FFMPEG_QUALITY_CPU)
    )
    
    output = OutputParams(
        create_comparison_video=create_comparison_video_check if create_comparison_video_check is not None else DEFAULT_CREATE_COMPARISON_VIDEO,
        save_frames=save_frames_checkbox if save_frames_checkbox is not None else DEFAULT_SAVE_FRAMES,
        save_metadata=save_metadata_checkbox if save_metadata_checkbox is not None else DEFAULT_SAVE_METADATA,
        save_chunks=save_chunks_checkbox if save_chunks_checkbox is not None else DEFAULT_SAVE_CHUNKS,
        save_chunk_frames=save_chunk_frames_checkbox if save_chunk_frames_checkbox is not None else DEFAULT_SAVE_CHUNK_FRAMES
    )
    
    scene_splitting = SceneSplittingParams(
        enable_scene_split=enable_scene_split_check if enable_scene_split_check is not None else DEFAULT_ENABLE_SCENE_SPLIT,
        scene_split_mode=scene_split_mode_radio or DEFAULT_SCENE_SPLIT_MODE,
        scene_min_scene_len=scene_min_scene_len_num or DEFAULT_SCENE_MIN_SCENE_LEN,
        scene_drop_short=scene_drop_short_check if scene_drop_short_check is not None else DEFAULT_SCENE_DROP_SHORT,
        scene_merge_last=scene_merge_last_check if scene_merge_last_check is not None else DEFAULT_SCENE_MERGE_LAST,
        scene_frame_skip=scene_frame_skip_num or DEFAULT_SCENE_FRAME_SKIP,
        scene_threshold=scene_threshold_num or DEFAULT_SCENE_THRESHOLD,
        scene_min_content_val=scene_min_content_val_num or DEFAULT_SCENE_MIN_CONTENT_VAL,
        scene_frame_window=scene_frame_window_num or DEFAULT_SCENE_FRAME_WINDOW,
        scene_copy_streams=scene_copy_streams_check if scene_copy_streams_check is not None else DEFAULT_SCENE_COPY_STREAMS,
        scene_use_mkvmerge=scene_use_mkvmerge_check if scene_use_mkvmerge_check is not None else DEFAULT_SCENE_USE_MKVMERGE,
        scene_rate_factor=scene_rate_factor_num or DEFAULT_SCENE_RATE_FACTOR,
        scene_preset=scene_preset_dropdown or DEFAULT_SCENE_ENCODING_PRESET,
        scene_quiet_ffmpeg=scene_quiet_ffmpeg_check if scene_quiet_ffmpeg_check is not None else DEFAULT_SCENE_QUIET_FFMPEG,
        scene_manual_split_type=scene_manual_split_type_radio or DEFAULT_SCENE_MANUAL_SPLIT_TYPE,
        scene_manual_split_value=scene_manual_split_value_num or DEFAULT_SCENE_MANUAL_SPLIT_VALUE
    )
    
    fps = FpsParams(
        enable_fps_decrease=enable_fps_decrease if enable_fps_decrease is not None else DEFAULT_ENABLE_FPS_DECREASE,
        fps_decrease_mode=fps_decrease_mode or DEFAULT_FPS_DECREASE_MODE,
        fps_multiplier_preset=fps_multiplier_preset or "1/2x (Half FPS)",
        fps_multiplier_custom=fps_multiplier_custom or DEFAULT_FPS_MULTIPLIER,
        target_fps=target_fps or DEFAULT_TARGET_FPS,
        fps_interpolation_method=fps_interpolation_method or DEFAULT_FPS_INTERPOLATION_METHOD
    )
    
    rife = RifeParams(
        enable_interpolation=enable_rife_interpolation if enable_rife_interpolation is not None else DEFAULT_RIFE_ENABLE_INTERPOLATION,
        multiplier=rife_multiplier or DEFAULT_RIFE_MULTIPLIER,
        fp16=rife_fp16 if rife_fp16 is not None else DEFAULT_RIFE_FP16,
        uhd=rife_uhd if rife_uhd is not None else DEFAULT_RIFE_UHD,
        scale=rife_scale or DEFAULT_RIFE_SCALE,
        skip_static=rife_skip_static if rife_skip_static is not None else DEFAULT_RIFE_SKIP_STATIC,
        enable_fps_limit=rife_enable_fps_limit if rife_enable_fps_limit is not None else DEFAULT_RIFE_ENABLE_FPS_LIMIT,
        max_fps_limit=rife_max_fps_limit or DEFAULT_RIFE_MAX_FPS_LIMIT,
        apply_to_chunks=rife_apply_to_chunks if rife_apply_to_chunks is not None else DEFAULT_RIFE_APPLY_TO_CHUNKS,
        apply_to_scenes=rife_apply_to_scenes if rife_apply_to_scenes is not None else DEFAULT_RIFE_APPLY_TO_SCENES,
        keep_original=rife_keep_original if rife_keep_original is not None else DEFAULT_RIFE_KEEP_ORIGINAL,
        overwrite_original=rife_overwrite_original if rife_overwrite_original is not None else DEFAULT_RIFE_OVERWRITE_ORIGINAL
    )
    
    cogvlm = CogVLMParams(
        quant=get_quant_value_from_display(cogvlm_quant_radio),
        unload_strategy=cogvlm_unload_radio or DEFAULT_COGVLM_UNLOAD_AFTER_USE,
        auto_caption_then_upscale=auto_caption_then_upscale_check if auto_caption_then_upscale_check is not None else DEFAULT_AUTO_CAPTION_THEN_UPSCALE
    )
    
    seed = SeedParams(
        seed_value=seed_num or DEFAULT_SEED,
        random_seed=random_seed_check if random_seed_check is not None else DEFAULT_RANDOM_SEED
    )
    
    image_upscaler = ImageUpscalerParams(
        enable=enable_image_upscaler_check if enable_image_upscaler_check is not None else DEFAULT_ENABLE_IMAGE_UPSCALER,
        model=image_upscaler_model_dropdown,
        batch_size=image_upscaler_batch_size_slider or DEFAULT_IMAGE_UPSCALER_BATCH_SIZE
    )
    
    face_restoration = FaceRestorationParams(
        enable=enable_face_restoration_check if enable_face_restoration_check is not None else DEFAULT_ENABLE_FACE_RESTORATION,
        fidelity=face_restoration_fidelity_slider or DEFAULT_FACE_RESTORATION_FIDELITY,
        enable_colorization=enable_face_colorization_check if enable_face_colorization_check is not None else DEFAULT_ENABLE_FACE_COLORIZATION,
        timing="after_upscale",  # Fixed for single video processing
        when=face_restoration_when_radio or DEFAULT_FACE_RESTORATION_WHEN,
        model=extract_codeformer_model_path_from_dropdown(codeformer_model_dropdown),
        batch_size=face_restoration_batch_size_slider or DEFAULT_FACE_RESTORATION_BATCH_SIZE
    )
    
    frame_folder = FrameFolderParams(
        enable=enable_frame_folder_check if enable_frame_folder_check is not None else False,
        folder_path=input_frames_folder or "",
        fps=frame_folder_fps_slider or DEFAULT_FRAME_FOLDER_FPS
    )
    
    batch = BatchParams()  # Will be set for batch processing
    
    # Create the master ProcessingParams
    params = ProcessingParams(
        input_video_path=input_video or "",
        prompts=prompts,
        core=core,
        performance=performance,
        tiling=tiling,
        context_window=context_window,
        target_resolution=target_resolution,
        ffmpeg=ffmpeg,
        output=output,
        scene_splitting=scene_splitting,
        fps=fps,
        rife=rife,
        cogvlm=cogvlm,
        seed=seed,
        image_upscaler=image_upscaler,
        face_restoration=face_restoration,
        frame_folder=frame_folder,
        batch=batch
    )
    
    return params

def wrapper_split_video_only_for_gradio(
    input_video_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
    scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
    scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
    scene_manual_split_type_radio_val, scene_manual_split_value_num_val,
    progress=gr.Progress(track_tqdm=True)
):
    return util_split_video_only(
        input_video_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
        scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
        scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
        scene_manual_split_type_radio_val, scene_manual_split_value_num_val,
        DEFAULT_OUTPUT_DIR,
        logger,
        progress=progress
    )

# CSS styling
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

def upscale_director_logic_v2(*gradio_values, progress=gr.Progress(track_tqdm=True)):
    """
    Main processing function that converts Gradio inputs to ProcessingParams
    and calls the core processing logic.
    """
    try:
        # Convert all Gradio inputs to ProcessingParams dataclass
        params = create_processing_params_from_gradio_inputs(*gradio_values)
        
        # Call the wrapper function that unpacks the dataclass for the core logic
        for result in core_run_upscale_with_params(params, progress=progress):
            yield result
            
    except Exception as e:
        logger.error(f"Error in upscale_director_logic_v2: {e}", exc_info=True)
        error_msg = f"❌ Processing error: {str(e)}"
        yield (None, error_msg, "", "", None, "Processing failed", None)

def core_run_upscale_with_params(params, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper function that unpacks ProcessingParams dataclass into individual parameters
    for compatibility with the existing core_run_upscale function.
    """
    # Handle seed generation
    actual_seed_to_use = params.seed.seed_value
    if params.seed.random_seed:
        actual_seed_to_use = np.random.randint(0, 2**31)
        logger.info(f"Random seed checkbox is checked. Using generated seed: {actual_seed_to_use}")
    elif params.seed.seed_value == -1:
        actual_seed_to_use = np.random.randint(0, 2**31)
        logger.info(f"Seed input is -1. Using generated seed: {actual_seed_to_use}")
    else:
        logger.info(f"Using provided seed: {actual_seed_to_use}")

    # Handle frame folder input preprocessing if enabled
    actual_input_video_path = params.input_video_path
    if params.frame_folder.enable and params.frame_folder.folder_path:
        logger.info("Frame folder processing mode enabled")
        progress(0, desc="Converting frame folder to video...")
        
        # Validate frame folder first
        is_valid, validation_msg, frame_count = util_validate_frame_folder_input(params.frame_folder.folder_path, logger)
        
        if not is_valid:
            error_msg = f"Frame folder validation failed: {validation_msg}"
            logger.error(error_msg)
            yield (None, error_msg, "", "", None, "Frame folder validation failed", None)
            return
        
        # Create temporary video from frames
        import tempfile
        temp_video_dir = tempfile.mkdtemp(prefix="frame_folder_")
        frame_folder_name = os.path.basename(params.frame_folder.folder_path.rstrip(os.sep))
        temp_video_path = os.path.join(temp_video_dir, f"{frame_folder_name}_from_frames.mp4")
        
        try:
            success, conv_msg = util_process_frame_folder_to_video(
                params.frame_folder.folder_path, temp_video_path, fps=params.frame_folder.fps,
                ffmpeg_preset=params.ffmpeg.preset,
                ffmpeg_quality_value=params.ffmpeg.quality,
                ffmpeg_use_gpu=params.ffmpeg.use_gpu,
                logger=logger
            )
            
            if success:
                actual_input_video_path = temp_video_path
                logger.info(f"Frame folder converted successfully. Using: {actual_input_video_path}")
            else:
                error_msg = f"❌ Failed to convert frame folder: {conv_msg}"
                logger.error(error_msg)
                yield (None, error_msg, "", "", None, "Frame folder conversion failed", None)
                return
                
        except Exception as e:
            error_msg = f"❌ Exception during frame folder conversion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield (None, error_msg, "", "", None, "Frame folder conversion error", None)
            return

    # Call the core upscale function with unpacked parameters
    upscale_generator = core_run_upscale(
        input_video_path=actual_input_video_path,
        user_prompt=params.prompts.user_prompt,
        positive_prompt=params.prompts.positive_prompt,
        negative_prompt=params.prompts.negative_prompt,
        model_choice=params.core.model_choice,
        upscale_factor_slider=params.core.upscale_factor,
        cfg_scale=params.core.cfg_scale,
        steps=params.core.steps,
        solver_mode=params.core.solver_mode,
        max_chunk_len=params.performance.max_chunk_len,
        enable_chunk_optimization=params.performance.enable_chunk_optimization,
        vae_chunk=params.performance.vae_chunk,
        color_fix_method=params.core.color_fix_method,
        enable_tiling=params.tiling.enable_tiling,
        tile_size=params.tiling.tile_size,
        tile_overlap=params.tiling.tile_overlap,
        enable_context_window=params.context_window.enable_context_window,
        context_overlap=params.context_window.context_overlap,
        enable_target_res=params.target_resolution.enable_target_res,
        target_h=params.target_resolution.target_h,
        target_w=params.target_resolution.target_w,
        target_res_mode=params.target_resolution.target_res_mode,
        ffmpeg_preset=params.ffmpeg.preset,
        ffmpeg_quality_value=params.ffmpeg.quality,
        ffmpeg_use_gpu=params.ffmpeg.use_gpu,
        save_frames=params.output.save_frames,
        save_metadata=params.output.save_metadata,
        save_chunks=params.output.save_chunks,
        save_chunk_frames=params.output.save_chunk_frames,
        enable_scene_split=params.scene_splitting.enable_scene_split,
        scene_split_mode=params.scene_splitting.scene_split_mode,
        scene_min_scene_len=params.scene_splitting.scene_min_scene_len,
        scene_drop_short=params.scene_splitting.scene_drop_short,
        scene_merge_last=params.scene_splitting.scene_merge_last,
        scene_frame_skip=params.scene_splitting.scene_frame_skip,
        scene_threshold=params.scene_splitting.scene_threshold,
        scene_min_content_val=params.scene_splitting.scene_min_content_val,
        scene_frame_window=params.scene_splitting.scene_frame_window,
        scene_copy_streams=params.scene_splitting.scene_copy_streams,
        scene_use_mkvmerge=params.scene_splitting.scene_use_mkvmerge,
        scene_rate_factor=params.scene_splitting.scene_rate_factor,
        scene_preset=params.scene_splitting.scene_preset,
        scene_quiet_ffmpeg=params.scene_splitting.scene_quiet_ffmpeg,
        scene_manual_split_type=params.scene_splitting.scene_manual_split_type,
        scene_manual_split_value=params.scene_splitting.scene_manual_split_value,
        create_comparison_video_enabled=params.output.create_comparison_video,
        enable_fps_decrease=params.fps.enable_fps_decrease,
        fps_decrease_mode=params.fps.fps_decrease_mode,
        fps_multiplier_preset=params.fps.fps_multiplier_preset,
        fps_multiplier_custom=params.fps.fps_multiplier_custom,
        target_fps=params.fps.target_fps,
        fps_interpolation_method=params.fps.fps_interpolation_method,
        enable_rife_interpolation=params.rife.enable_interpolation,
        rife_multiplier=params.rife.multiplier,
        rife_fp16=params.rife.fp16,
        rife_uhd=params.rife.uhd,
        rife_scale=params.rife.scale,
        rife_skip_static=params.rife.skip_static,
        rife_enable_fps_limit=params.rife.enable_fps_limit,
        rife_max_fps_limit=params.rife.max_fps_limit,
        rife_apply_to_chunks=params.rife.apply_to_chunks,
        rife_apply_to_scenes=params.rife.apply_to_scenes,
        rife_keep_original=params.rife.keep_original,
        rife_overwrite_original=params.rife.overwrite_original,
        is_batch_mode=False,
        batch_output_dir=None,
        original_filename=None,
        enable_auto_caption_per_scene=(params.cogvlm.auto_caption_then_upscale and 
                                      params.scene_splitting.enable_scene_split and 
                                      not params.image_upscaler.enable and 
                                      UTIL_COG_VLM_AVAILABLE),
        cogvlm_quant=params.cogvlm.quant,
        cogvlm_unload=params.cogvlm.unload_strategy,
        logger=logger,
        app_config_module=app_config,
        metadata_handler_module=metadata_handler,
        VideoToVideo_sr_class=VideoToVideo_sr,
        setup_seed_func=setup_seed,
        EasyDict_class=EasyDict,
        preprocess_func=preprocess,
        collate_fn_func=collate_fn,
        tensor2vid_func=tensor2vid,
        ImageSpliterTh_class=ImageSpliterTh,
        adain_color_fix_func=adain_color_fix,
        wavelet_color_fix_func=wavelet_color_fix,
        progress=progress,
        current_seed=actual_seed_to_use,
        # Image upscaler parameters
        enable_image_upscaler=params.image_upscaler.enable,
        image_upscaler_model=params.image_upscaler.model,
        image_upscaler_batch_size=params.image_upscaler.batch_size,
        # Face restoration parameters
        enable_face_restoration=params.face_restoration.enable,
        face_restoration_fidelity=params.face_restoration.fidelity,
        enable_face_colorization=params.face_restoration.enable_colorization,
        face_restoration_timing=params.face_restoration.timing,
        face_restoration_when=params.face_restoration.when,
        codeformer_model=params.face_restoration.model,
        face_restoration_batch_size=params.face_restoration.batch_size
    )

    # Yield results from the core processing
    for result in upscale_generator:
        yield result

if __name__ == "__main__":
    # Logger setup
    logger = get_logger()
    logger.setLevel(logging.INFO)
    found_stream_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.INFO)
            found_stream_handler = True
            logger.info("Diagnostic: Explicitly set StreamHandler level to INFO.")
    if not found_stream_handler:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.info("Diagnostic: No StreamHandler found, added a new one with INFO level.")
    logger.info(f"Logger '{logger.name}' configured with level: {logging.getLevelName(logger.level)}. Handlers: {logger.handlers}")

    # Initialize paths and prompts using dataclasses function
    initialize_paths_and_prompts(base_path, args.outputs_folder, star_cfg)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(LIGHT_DEG_MODEL_PATH):
        logger.error(f"FATAL: Light degradation model not found at {LIGHT_DEG_MODEL_PATH}.")
    if not os.path.exists(HEAVY_DEG_MODEL_PATH):
        logger.error(f"FATAL: Heavy degradation model not found at {HEAVY_DEG_MODEL_PATH}.")

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Gradio App Starting. Default output to: {os.path.abspath(DEFAULT_OUTPUT_DIR)}")
    logger.info(f"STAR Models expected at: {LIGHT_DEG_MODEL_PATH}, {HEAVY_DEG_MODEL_PATH}")
    if UTIL_COG_VLM_AVAILABLE:
        logger.info(f"CogVLM2 Model expected at: {COG_VLM_MODEL_PATH}")

    available_gpus_main = util_get_available_gpus()
    if available_gpus_main:
        default_gpu_main_val = available_gpus_main[0]
        util_set_gpu_device(default_gpu_main_val, logger=logger)
        logger.info(f"Attempted to initialize with default GPU: {default_gpu_main_val}")
    else:
        logger.info("No CUDA GPUs detected, attempting to set to 'Auto' (CPU or default).")
        util_set_gpu_device("Auto", logger=logger)

    effective_allowed_paths = util_get_available_drives(DEFAULT_OUTPUT_DIR, base_path, logger=logger)

    demo.queue().launch(
        debug=True,
        max_threads=100,
        inbrowser=True,
        share=args.share,
        allowed_paths=effective_allowed_paths,
        prevent_thread_lock=True
    )