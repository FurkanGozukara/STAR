# Standard library imports
import gc, logging, math, os, platform, re, shutil, subprocess, sys, tempfile, threading, time, webbrowser
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

# Set environment variable before other imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Third-party imports
import cv2, gradio as gr, numpy as np, torch, torchvision
from easydict import EasyDict

# Local logic imports
from logic import metadata_handler, config as app_config_module, preset_handler, info_strings
from logic.star_dataclasses import (
    AppConfig, create_app_config, UTIL_COG_VLM_AVAILABLE, UTIL_BITSANDBYTES_AVAILABLE,
    get_cogvlm_quant_choices_map, get_default_cogvlm_quant_display,
    PathConfig, PromptConfig, StarModelConfig, PerformanceConfig, ResolutionConfig,
    ContextWindowConfig, TilingConfig, FfmpegConfig, FrameFolderConfig, SceneSplitConfig,
    CogVLMConfig, OutputConfig, SeedConfig, RifeConfig, FpsDecreaseConfig, BatchConfig,
    ImageUpscalerConfig, FaceRestorationConfig, GpuConfig, UpscalerTypeConfig, SeedVR2Config,
    ManualComparisonConfig, StandaloneFaceRestorationConfig, PresetSystemConfig, VideoEditingConfig,
    DEFAULT_GPU_DEVICE, DEFAULT_VIDEO_COUNT_THRESHOLD_3, DEFAULT_VIDEO_COUNT_THRESHOLD_4,
    DEFAULT_PROGRESS_OFFSET, DEFAULT_PROGRESS_SCALE, DEFAULT_BATCH_PROGRESS_OFFSET, DEFAULT_BATCH_PROGRESS_SCALE
)
from logic.batch_operations import process_batch_videos
from logic.batch_processing_help import create_batch_processing_help
from logic.cancellation_manager import cancellation_manager, CancelledError
from logic.cogvlm_utils import (
    load_cogvlm_model as util_load_cogvlm_model,
    unload_cogvlm_model as util_unload_cogvlm_model,
    auto_caption as util_auto_caption
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
    get_video_info_fast as util_get_video_info_fast,
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
from logic.frame_folder_utils import (
    validate_frame_folder_input as util_validate_frame_folder_input,
    process_frame_folder_to_video as util_process_frame_folder_to_video,
    find_frame_folders_in_directory as util_find_frame_folders_in_directory
)
from logic.gpu_utils import (
    get_available_gpus as util_get_available_gpus,
    set_gpu_device as util_set_gpu_device,
    get_gpu_device as util_get_gpu_device,
    validate_gpu_availability as util_validate_gpu_availability
)
from logic.image_upscaler_utils import (
    scan_for_models as util_scan_for_models,
    get_model_info as util_get_model_info
)
from logic.manual_comparison import (
    generate_manual_comparison_video as util_generate_manual_comparison_video,
    generate_multi_video_comparison as util_generate_multi_video_comparison
)
from logic.nvenc_utils import is_resolution_too_small_for_nvenc
from logic.preview_utils import (
    preview_single_model as util_preview_single_model,
    preview_all_models as util_preview_all_models
)
from logic.rife_interpolation import rife_fps_only_wrapper as util_rife_fps_only_wrapper
from logic.scene_utils import (
    split_video_into_scenes as util_split_video_into_scenes,
    merge_scene_videos as util_merge_scene_videos,
    split_video_only as util_split_video_only
)
from logic.seedvr2_cli_core import (
    process_video_with_seedvr2_cli as util_process_video_with_seedvr2_cli,
    SeedVR2BlockSwap as util_SeedVR2BlockSwap,
    apply_wavelet_color_correction as util_apply_wavelet_color_correction
)
from logic.seedvr2_utils import (
    util_check_seedvr2_dependencies, util_scan_seedvr2_models, util_get_seedvr2_model_info,
    util_format_model_info_display, util_get_vram_info, util_get_block_swap_recommendations,
    util_format_model_display_name, util_validate_seedvr2_model, util_extract_model_filename_from_dropdown,
    util_format_vram_status, util_format_block_swap_status, util_validate_seedvr2_config,
    util_get_recommended_settings_for_vram, util_get_suggested_settings, util_estimate_processing_time,
    util_cleanup_seedvr2_resources, util_detect_available_gpus, util_validate_gpu_selection
)
from logic.temp_folder_utils import (
    get_temp_folder_path as util_get_temp_folder_path,
    calculate_temp_folder_size as util_calculate_temp_folder_size,
    format_temp_folder_size as util_format_temp_folder_size,
    clear_temp_folder as util_clear_temp_folder
)
from logic.upscaling_core import run_upscale as core_run_upscale
from logic.upscaling_utils import calculate_upscale_params as util_calculate_upscale_params
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
from logic.info_strings import *

parser = ArgumentParser(description="Ultimate SECourses STAR Video Upscaler")
parser.add_argument('--share', action='store_true', help="Enable Gradio live share")
parser.add_argument('--outputs_folder', type=str, default="outputs", help=PARSER_HELP_OUTPUTS_FOLDER)
args = parser.parse_args()

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = script_dir

    if not os.path.isdir(os.path.join(base_path, 'video_to_video')):
        print(VIDEO_TO_VIDEO_DIR_NOT_FOUND_WARNING.format(base_path=base_path))
        base_path = os.path.dirname(base_path)
        if not os.path.isdir(os.path.join(base_path, 'video_to_video')):
            print(BASE_PATH_AUTO_DETERMINE_ERROR)
            print(f"Current inferred base_path: {base_path}")

    print(BASE_PATH_USAGE_INFO.format(base_path=base_path))
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

except Exception as e_path:
    print(f"Error setting up base_path: {e_path}")
    print(APP_PLACEMENT_ERROR)
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
    print(DEPENDENCIES_INSTALL_ERROR)
    sys.exit(1)

# Custom logging filter to suppress progress messages from console
class ProgressMessageFilter(logging.Filter):
    def filter(self, record):
        # Suppress messages that are already shown in Gradio UI
        suppress_patterns = [
            "⏳ Processing...",
            "🎬 Batch",
            "frames left",
            "💓 Yielding heartbeat",
            "📦 Processing batch",
            "📤 Queued batch progress",
            "🔍 Debug: batch_progress_queue",
            "✅ Batch processed successfully",
            "❌ Batch processing error",
            "📊 Batch progress queue",
            "⏱️ Batch",
            "📊 Total ETA:",
            "🔍 Queue has",
            "🔍 Successfully got from queue",
            "🔍 Received processing result type",
            "📊 Status update - result_path",
            "UI received batch progress",
            "🎯 decode_dtype:",
            "🎄 Monitoring for chunk updates",
            "🔍 Monitor loop active",
            "🔄 video Compute dtype:",
            "📹 Sequence of",
            "🔄 VAE to GPU time:",
            "🔄 VAE dtype:",
            "🔄 VAE encode time:",
            "🔄 Transformed video to",
            "🔄 Cond latents shape:",
            "🎯 model_dtype:",
            "🎯 target_dtype:",
            "🔄 VAE to CPU time:",
            "🔄 Dit to GPU time:",
            "🔄 INFERENCE time:",
            "🧹 Clearing VRAM cache",
            "🔄 Dit to CPU time:",
            "🔄 shape of latents:",
            "🔄 DECODE time:",
            "🔄 Samples shape:",
            "🔧 Converting",
            "🔄 Time batch:",
            "⏱️  Batch time:",
            "💾 Saved",
            "📊 Total frames saved:",
            "🔄 VAE",
            "🔄 Dit",
            "EulerSampler:"
        ]
        return not any(pattern in record.getMessage() for pattern in suppress_patterns)

# Apply filter to all existing loggers
def apply_progress_filter_globally():
    # Get all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger_instance = logging.getLogger(logger_name)
        for handler in logger_instance.handlers:
            if isinstance(handler, logging.StreamHandler):
                # Check if filter already exists to avoid duplicates
                if not any(isinstance(f, ProgressMessageFilter) for f in handler.filters):
                    handler.addFilter(ProgressMessageFilter())

# Override getLogger to automatically apply filter to new loggers
original_getLogger = logging.getLogger
def getLogger_with_filter(name=None):
    logger_instance = original_getLogger(name)
    # Apply filter to all stream handlers
    for handler in logger_instance.handlers:
        if isinstance(handler, logging.StreamHandler):
            if not any(isinstance(f, ProgressMessageFilter) for f in handler.filters):
                handler.addFilter(ProgressMessageFilter())
    return logger_instance

# Replace the original getLogger
logging.getLogger = getLogger_with_filter

logger = get_logger()
logger.setLevel(logging.INFO)
found_stream_handler = False
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
        # Add the filter to suppress progress messages
        handler.addFilter(ProgressMessageFilter())
        found_stream_handler = True
        logger.info(info_strings.STREAM_HANDLER_LEVEL_SET_INFO)
if not found_stream_handler:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Add the filter to suppress progress messages
    ch.addFilter(ProgressMessageFilter())
    logger.addHandler(ch)
    logger.info(info_strings.STREAM_HANDLER_ADDED_INFO)
logger.info(info_strings.LOGGER_CONFIGURED_LEVEL_HANDLERS_INFO.format(logger_name=logger.name, level=logging.getLevelName(logger.level), handlers=logger.handlers))

# Apply the progress filter to all existing loggers
apply_progress_filter_globally()

APP_CONFIG = create_app_config(base_path, args.outputs_folder, star_cfg)
app_config_module.initialize_paths_and_prompts(base_path, args.outputs_folder, star_cfg)
os.makedirs(APP_CONFIG.paths.outputs_dir, exist_ok=True)

if not os.path.exists(APP_CONFIG.paths.light_deg_model_path):
     logger.error(LIGHT_DEG_MODEL_NOT_FOUND_ERROR.format(model_path=APP_CONFIG.paths.light_deg_model_path))
if not os.path.exists(APP_CONFIG.paths.heavy_deg_model_path):
     logger.error(HEAVY_DEG_MODEL_NOT_FOUND_ERROR.format(model_path=APP_CONFIG.paths.heavy_deg_model_path))

css = APP_CSS

def load_initial_preset():
    global LOADED_PRESET_NAME
    base_config = create_app_config(base_path, args.outputs_folder, star_cfg)
    base_config.gpu.device = DEFAULT_GPU_DEVICE

    preset_to_load = preset_handler.get_last_used_preset_name()
    if preset_to_load:
        logger.info(PRESET_LAST_USED_FOUND_INFO.format(preset_name=preset_to_load))
    else:
        logger.info(PRESET_NO_LAST_USED_INFO)
        preset_to_load = "Default"

    config_dict, message = preset_handler.load_preset(preset_to_load)

    if config_dict:
        logger.info(PRESET_SUCCESSFULLY_LOADED_INFO.format(preset_name=preset_to_load))
        LOADED_PRESET_NAME = preset_to_load
        for section_name, section_data in config_dict.items():
            if hasattr(base_config, section_name):
                section_obj = getattr(base_config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        if section_name == 'gpu' and key == 'device':
                            available_gpus = util_get_available_gpus()
                            if available_gpus:
                                try:
                                    gpu_num = int(value) if value != "Auto" else int(DEFAULT_GPU_DEVICE)
                                    if 0 <= gpu_num < len(available_gpus):
                                        setattr(section_obj, key, str(gpu_num))
                                    else:
                                        logger.warning(PRESET_GPU_INDEX_OUT_OF_RANGE_WARNING.format(gpu_num=gpu_num, default_gpu=DEFAULT_GPU_DEVICE))
                                        setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                                except (ValueError, TypeError):
                                    logger.warning(PRESET_INVALID_GPU_VALUE_WARNING.format(value=value, default_gpu=DEFAULT_GPU_DEVICE))
                                    setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                            else:
                                setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                        else:
                            setattr(section_obj, key, value)
        return base_config
    else:
        if preset_to_load != "Default":
            logger.warning(PRESET_CLEARING_FAILED_WARNING.format(preset_name=preset_to_load))
            preset_handler.save_last_used_preset_name("")

        logger.warning(PRESET_FALLBACK_ATTEMPT_WARNING.format(preset_name=preset_to_load, message=message))
        fallback_config_dict, fallback_message = preset_handler.load_preset("Image_Upscaler_Fast_Low_VRAM")

        if fallback_config_dict:
            logger.info(PRESET_FALLBACK_SUCCESS_INFO)
            LOADED_PRESET_NAME = "Image_Upscaler_Fast_Low_VRAM"
            for section_name, section_data in fallback_config_dict.items():
                if hasattr(base_config, section_name):
                    section_obj = getattr(base_config, section_name)
                    for key, value in section_data.items():
                        if hasattr(section_obj, key):
                            if section_name == 'gpu' and key == 'device':
                                available_gpus = util_get_available_gpus()
                                if available_gpus:
                                    try:
                                        gpu_num = int(value) if value != "Auto" else int(DEFAULT_GPU_DEVICE)
                                        if 0 <= gpu_num < len(available_gpus):
                                            setattr(section_obj, key, str(gpu_num))
                                        else:
                                            logger.warning(PRESET_GPU_INDEX_OUT_OF_RANGE_WARNING.format(gpu_num=gpu_num, default_gpu=DEFAULT_GPU_DEVICE))
                                            setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                                    except (ValueError, TypeError):
                                        logger.warning(PRESET_INVALID_GPU_VALUE_WARNING.format(value=value, default_gpu=DEFAULT_GPU_DEVICE))
                                        setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                                else:
                                    setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                            else:
                                setattr(section_obj, key, value)
            return base_config
        else:
            logger.warning(PRESET_FALLBACK_FAILED_WARNING.format(message=fallback_message))
            available_presets = preset_handler.get_preset_list()
            filtered_presets = [p for p in available_presets if p not in ["last_preset", preset_to_load, "Image_Upscaler_Fast_Low_VRAM"]]
            if filtered_presets:
                first_available = filtered_presets[0]
                logger.info(PRESET_TRYING_FIRST_AVAILABLE_INFO.format(preset_name=first_available))
                last_resort_config, last_resort_message = preset_handler.load_preset(first_available)
                if last_resort_config:
                    logger.info(PRESET_LAST_RESORT_SUCCESS_INFO.format(preset_name=first_available))
                    LOADED_PRESET_NAME = first_available
                    for section_name, section_data in last_resort_config.items():
                        if hasattr(base_config, section_name):
                            section_obj = getattr(base_config, section_name)
                            for key, value in section_data.items():
                                if hasattr(section_obj, key):
                                    if section_name == 'gpu' and key == 'device':
                                        available_gpus = util_get_available_gpus()
                                        if available_gpus:
                                            try:
                                                gpu_num = int(value) if value != "Auto" else int(DEFAULT_GPU_DEVICE)
                                                if 0 <= gpu_num < len(available_gpus):
                                                    setattr(section_obj, key, str(gpu_num))
                                                else:
                                                    logger.warning(PRESET_GPU_INDEX_OUT_OF_RANGE_WARNING.format(gpu_num=gpu_num, default_gpu=DEFAULT_GPU_DEVICE))
                                                    setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                                            except (ValueError, TypeError):
                                                logger.warning(PRESET_INVALID_GPU_VALUE_WARNING.format(value=value, default_gpu=DEFAULT_GPU_DEVICE))
                                                setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                                        else:
                                            setattr(section_obj, key, DEFAULT_GPU_DEVICE)
                                    else:
                                        setattr(section_obj, key, value)
                    return base_config
                else:
                    logger.warning(PRESET_LAST_RESORT_FAILED_WARNING.format(preset_name=first_available, message=last_resort_message))

            logger.warning(PRESET_NO_PRESETS_LOADED_WARNING)
            LOADED_PRESET_NAME = None
            return base_config

LOADED_PRESET_NAME = None
INITIAL_APP_CONFIG = load_initial_preset()

def get_filtered_preset_list():
    all_presets = preset_handler.get_preset_list()
    filtered_presets = [preset for preset in all_presets if preset != "last_preset"]
    logger.debug(f"All presets: {all_presets}, Filtered presets: {filtered_presets}")
    return filtered_presets

def get_initial_preset_name():
    if LOADED_PRESET_NAME:
        logger.info(PRESET_INITIAL_FOR_DROPDOWN_INFO.format(preset_name=LOADED_PRESET_NAME))
        return LOADED_PRESET_NAME
    else:
        logger.info(PRESET_NO_PRESET_SUCCESSFULLY_LOADED_INFO)
        return None

INITIAL_PRESET_NAME = get_initial_preset_name()

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
        APP_CONFIG.paths.outputs_dir,
        logger,
        progress=progress
    )

# --- Refactoring Helper Functions ---

def create_component(component_type, config_path, ui_dict, **kwargs):
    """
    Creates a Gradio component, sets its initial value from config,
    and registers it in the UI components dictionary.
    """
    section, key = config_path
    # Get the initial value from the loaded config if not explicitly provided
    if 'value' not in kwargs:
        # Gracefully handle nested attribute access
        section_obj = getattr(INITIAL_APP_CONFIG, section, None)
        if section_obj:
            kwargs['value'] = getattr(section_obj, key, None)
        else:
            kwargs['value'] = None

    # Create the component
    component = component_type(**kwargs)
    
    # Register it
    ui_dict[config_path] = component
    
    return component

# Create convenient partials for common components
create_slider = partial(create_component, gr.Slider)
create_checkbox = partial(create_component, gr.Checkbox)
create_textbox = partial(create_component, gr.Textbox)
create_dropdown = partial(create_component, gr.Dropdown)
create_radio = partial(create_component, gr.Radio)
create_number = partial(create_component, gr.Number)
create_video_component = partial(create_component, gr.Video)

# --- End of Refactoring Helper Functions ---


with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Central dictionary to hold UI components that map to AppConfig
    ui_components = {}

    gr.Markdown(APP_TITLE)

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
                        integration_status = gr.Textbox(
                            label="Integration Status",
                            interactive=False,
                            lines=2,
                            visible=False,
                            value=""
                        )
                        with gr.Row():
                            user_prompt = create_textbox(
                                config_path=('prompts', 'user'), ui_dict=ui_components,
                                label=DESCRIBE_VIDEO_CONTENT_LABEL, lines=6,
                                placeholder=PANDA_PLAYING_GUITAR_PLACEHOLDER, info=PROMPT_USER_INFO
                            )
                        with gr.Row():
                            auto_caption_then_upscale_check = create_checkbox(
                                config_path=('cogvlm', 'auto_caption_then_upscale'), ui_dict=ui_components,
                                label=AUTO_CAPTION_THEN_UPSCALE_LABEL, info=AUTO_CAPTION_THEN_UPSCALE_INFO
                            )

                            available_gpus = util_get_available_gpus()
                            gpu_choices = available_gpus if available_gpus else ["No CUDA GPUs detected"]
                            # The initial value must be converted from the stored index to the display string
                            def convert_gpu_index_to_dropdown(gpu_index, available_gpus_list):
                                if not available_gpus_list: return "No CUDA GPUs detected"
                                try:
                                    gpu_num = int(gpu_index)
                                    if 0 <= gpu_num < len(available_gpus_list): return available_gpus_list[gpu_num]
                                except (ValueError, TypeError): pass
                                return available_gpus_list[0]

                            initial_gpu_display = convert_gpu_index_to_dropdown(INITIAL_APP_CONFIG.gpu.device, available_gpus)
                            
                            gpu_selector = create_dropdown(
                                config_path=('gpu', 'device'), ui_dict=ui_components,
                                label="GPU Selection", choices=gpu_choices, value=initial_gpu_display,
                                info=GPU_SELECTOR_INFO, scale=1
                            )

                        if UTIL_COG_VLM_AVAILABLE:
                            with gr.Row(elem_id="row1"):
                                auto_caption_btn = gr.Button(GENERATE_CAPTION_NO_UPSCALE_BUTTON, variant="primary", icon="icons/caption.png")
                                rife_fps_button = gr.Button("RIFE FPS Increase (No Upscale)", variant="primary", icon="icons/fps.png")
                            with gr.Row(elem_id="row2"):
                                upscale_button = gr.Button("Upscale Video", variant="primary", icon="icons/upscale.png")
                            with gr.Row(elem_id="row3"):
                                cancel_button = gr.Button(CANCEL_UPSCALING_BUTTON, variant="stop", visible=True, interactive=False, icon="icons/cancel.png")
                            with gr.Row(elem_id="row4"):
                                initial_temp_size_label = util_format_temp_folder_size(logger)
                                delete_temp_button = gr.Button(f"Delete Temp Folder ({initial_temp_size_label})", variant="stop")
                            caption_status = gr.Textbox(label="Captioning Status", interactive=False, visible=False)
                        else:
                            with gr.Row():
                                rife_fps_button = gr.Button("RIFE FPS Increase (No Upscale)", variant="primary", icon="icons/fps.png")
                            with gr.Row():
                                upscale_button = gr.Button("Upscale Video", variant="primary", icon="icons/upscale.png")
                                cancel_button = gr.Button("Cancel", variant="stop", visible=True, interactive=False)
                            with gr.Row():
                                initial_temp_size_label = util_format_temp_folder_size(logger)
                                delete_temp_button = gr.Button(f"Delete Temp Folder ({initial_temp_size_label})", variant="stop")
                            caption_status = gr.Textbox(label="Captioning Status", interactive=False, visible=False)

                    with gr.Accordion(PROMPT_SETTINGS_STAR_MODEL_ACCORDION, open=True):
                        pos_prompt = create_textbox(
                            config_path=('prompts', 'positive'), ui_dict=ui_components,
                            label=DEFAULT_POSITIVE_PROMPT_LABEL, lines=2, info=PROMPT_POSITIVE_INFO
                        )
                        neg_prompt = create_textbox(
                            config_path=('prompts', 'negative'), ui_dict=ui_components,
                            label=DEFAULT_NEGATIVE_PROMPT_LABEL, lines=2, info=PROMPT_NEGATIVE_INFO
                        )
                    with gr.Group():
                        gr.Markdown(ENHANCED_INPUT_HEADER)
                        gr.Markdown(ENHANCED_INPUT_DESCRIPTION)
                        input_frames_folder = create_textbox(
                            config_path=('frame_folder', 'input_path'), ui_dict=ui_components,
                            label="Input Video or Frames Folder Path",
                            placeholder=VIDEO_FRAMES_FOLDER_PLACEHOLDER, interactive=True,
                            info=ENHANCED_INPUT_INFO
                        )
                        frames_folder_status = gr.Textbox(
                            label="Input Path Status", interactive=False, lines=3, visible=True,
                            value=DEFAULT_STATUS_MESSAGES['validate_input']
                        )

                with gr.Column(scale=1):
                    output_video = gr.Video(label="Upscaled Video", interactive=False, height=512)
                    with gr.Row():
                        how_to_use_button = gr.Button("📖 How To Use Documentation", variant="secondary")
                        version_history_button = gr.Button("📋 Version History V4", variant="secondary")
                        open_output_folder_button = gr.Button("Open Outputs Folder", icon="icons/folder.png", variant="primary")
                    status_textbox = gr.Textbox(label="Log", interactive=False, lines=10, max_lines=15)
                    with gr.Accordion("Save/Load Presets", open=True):
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Load Preset",
                                choices=get_filtered_preset_list(),
                                value=INITIAL_PRESET_NAME or "Default",
                                allow_custom_value=False, scale=3,
                                info="Select a preset to load"
                            )
                            refresh_presets_btn = gr.Button("🔄", scale=1, variant="secondary")
                        with gr.Row():
                            preset_save_textbox = gr.Textbox(
                                label="Save Preset As",
                                placeholder="Enter preset name...",
                                scale=3,
                                info="Enter a name for the new preset"
                            )
                            save_preset_btn = gr.Button("Save", variant="primary", scale=1)
                        preset_status = gr.Textbox(
                            label="Preset Status", show_label=False, interactive=False, lines=1,
                            value=PRESET_STATUS_LOADED.format(preset_name=LOADED_PRESET_NAME) if LOADED_PRESET_NAME else PRESET_STATUS_NO_PRESET,
                            placeholder="..."
                        )
                    with gr.Accordion("Last Processed Chunk", open=True):
                        last_chunk_video = gr.Video(
                            label="Last Processed Chunk Preview", interactive=False, height=512, visible=True
                        )
                        chunk_status_text = gr.Textbox(
                            label="Chunk Status", interactive=False, lines=1,
                            value="No chunks processed yet"
                        )

        with gr.Tab("Single Image Upscale", id="image_upscale_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("🚀 Processing Controls")
                        with gr.Row():
                            image_upscale_button = gr.Button("Upscale Image", variant="primary", icon="icons/upscale.png")
                            image_cancel_button = gr.Button(CANCEL_PROCESSING_BUTTON, variant="stop", visible=True, interactive=False, icon="icons/cancel.png")
                        image_processing_status = gr.Textbox(
                            label="Processing Status", interactive=False, lines=3,
                            value=DEFAULT_STATUS_MESSAGES['ready_image_processing']
                        )
                    with gr.Group():
                        input_image = gr.Image(
                            label="Input Image", sources=["upload"], interactive=True,
                            height=512, type="filepath"
                        )
                        image_integration_status = gr.Textbox(
                            label="Image Processing Status", interactive=False, lines=2,
                            visible=False, value=""
                        )
                    with gr.Accordion("Image Upscaler Selection", open=True):
                        gr.Markdown(CHOOSE_IMAGE_UPSCALING_METHOD)
                        image_upscaler_type_radio = create_radio(
                            config_path=('single_image_upscale', 'upscaler_type'), ui_dict=ui_components,
                            label="Select Image Upscaler Type", choices=["Use SeedVR2 for Images", "Use Image Based Upscalers"],
                            info=UPSCALER_TYPE_INFO
                        )
                    with gr.Accordion("Image Processing Settings", open=True):
                        with gr.Row():
                            image_preserve_aspect_ratio = create_checkbox(
                                config_path=('single_image_upscale', 'preserve_aspect_ratio'), ui_dict=ui_components,
                                label="Preserve Aspect Ratio", info=IMAGE_PRESERVE_ASPECT_RATIO_INFO
                            )
                            image_output_format = create_dropdown(
                                config_path=('single_image_upscale', 'output_format'), ui_dict=ui_components,
                                label="Output Format", choices=["PNG", "JPEG", "WEBP"],
                                info=IMAGE_OUTPUT_FORMAT_INFO
                            )
                        with gr.Row():
                            image_quality_level = create_slider(
                                config_path=('single_image_upscale', 'quality_level'), ui_dict=ui_components,
                                label="Output Quality (JPEG/WEBP only)", minimum=70, maximum=100,
                                step=1, info=IMAGE_QUALITY_INFO
                            )
                    with gr.Accordion("Advanced Image Settings", open=True):
                        with gr.Row():
                            image_preserve_metadata = create_checkbox(
                                config_path=('single_image_upscale', 'preserve_metadata'), ui_dict=ui_components,
                                label="Preserve Image Metadata",
                                info=IMAGE_PRESERVE_METADATA_INFO
                            )
                        with gr.Row():
                            image_custom_suffix = create_textbox(
                                config_path=('single_image_upscale', 'custom_suffix'), ui_dict=ui_components,
                                label="Custom Output Suffix", placeholder="_upscaled",
                                info=IMAGE_CUSTOM_SUFFIX_INFO
                            )
                    with gr.Accordion("Processing Log", open=True):
                        image_log_display = gr.Textbox(
                            label="Detailed Processing Log", interactive=False, lines=20, value=DEFAULT_STATUS_MESSAGES['ready_processing_log']
                        )
                with gr.Column(scale=1):
                    open_image_output_folder_button = gr.Button("Open Image Outputs Folder", icon="icons/folder.png", variant="primary")
                    with gr.Accordion("Upscaled Image Results", open=True):
                        output_image = gr.Image(label="Upscaled Image", interactive=False, height=512, type="filepath")
                    with gr.Accordion("Image Comparison", open=True):
                        image_comparison_slider = gr.ImageSlider(
                            label="Before/After Comparison", interactive=True, visible=False,
                            height=512, show_label=True, show_download_button=True,
                            show_fullscreen_button=True, slider_position=0.5
                        )
                    with gr.Accordion("Image Information", open=True):
                        image_info_display = gr.Textbox(
                            label="Image Details", interactive=False, lines=20, value=DEFAULT_STATUS_MESSAGES['ready_image_details']
                        )


        with gr.Tab("Resolution & Scene Split", id="resolution_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion(info_strings.TARGET_RESOLUTION_MAINTAINS_ASPECT_RATIO_ACCORDION, open=True):
                        gr.Markdown(IMAGE_UPSCALER_SUPPORT_NOTE)
                        enable_target_res_check = create_checkbox(
                            config_path=('resolution', 'enable_target_res'), ui_dict=ui_components,
                            label="Enable Max Target Resolution", info=info_strings.ENABLE_TARGET_RESOLUTION_MANUAL_CONTROL_INFO
                        )
                        target_res_mode_radio = create_radio(
                            config_path=('resolution', 'target_res_mode'), ui_dict=ui_components,
                            label="Target Resolution Mode", choices=['Ratio Upscale', 'Downscale then Upscale'],
                            info=TARGET_RESOLUTION_MODE_INFO
                        )
                        with gr.Row():
                            target_w_num = create_slider(
                                config_path=('resolution', 'target_w'), ui_dict=ui_components,
                                label="Max Target Width (px)", minimum=128, maximum=4096, step=16, info=TARGET_WIDTH_INFO
                            )
                            target_h_num = create_slider(
                                config_path=('resolution', 'target_h'), ui_dict=ui_components,
                                label="Max Target Height (px)", minimum=128, maximum=4096, step=16, info=TARGET_HEIGHT_INFO
                            )
                        gr.Markdown("---")
                        gr.Markdown(AUTO_RESOLUTION_HEADER)
                        enable_auto_aspect_resolution_check = create_checkbox(
                            config_path=('resolution', 'enable_auto_aspect_resolution'), ui_dict=ui_components,
                            label="Enable Auto Aspect Resolution", info=AUTO_ASPECT_RESOLUTION_INFO
                        )
                        auto_resolution_status_display = create_textbox(
                            config_path=('resolution', 'auto_resolution_status'), ui_dict=ui_components,
                            label="Auto-Resolution Status", interactive=False, lines=3,
                            info=info_strings.AUTO_CALCULATED_RESOLUTION_ASPECT_RATIO_INFO
                        )
                        gr.Markdown("---")
                        gr.Markdown(EXPECTED_OUTPUT_RESOLUTION_HEADER)
                        gr.Markdown(EXPECTED_OUTPUT_RESOLUTION_DESCRIPTION)
                        output_resolution_preview = gr.Textbox(
                            label="Expected Output Resolution", value=DEFAULT_STATUS_MESSAGES['expected_resolution'],
                            interactive=False, lines=10, info=info_strings.FINAL_OUTPUT_RESOLUTION_BASED_SETTINGS_INFO
                        )
                with gr.Column(scale=1):
                    split_only_button = gr.Button("Split Video Only (No Upscaling)", icon="icons/split.png", variant="primary")
                    with gr.Accordion("Scene Splitting", open=True):
                        enable_scene_split_check = create_checkbox(
                            config_path=('scene_split', 'enable'), ui_dict=ui_components,
                            label=ENABLE_SCENE_SPLITTING_LABEL, info=SCENE_SPLIT_INFO
                        )
                        with gr.Row():
                            scene_split_mode_radio = create_radio(
                                config_path=('scene_split', 'mode'), ui_dict=ui_components,
                                label="Split Mode", choices=['automatic', 'manual'], info=SCENE_SPLIT_MODE_INFO
                            )
                        with gr.Group():
                            gr.Markdown(AUTOMATIC_SCENE_DETECTION_SETTINGS)
                            with gr.Row():
                                scene_min_scene_len_num = create_number(
                                    config_path=('scene_split', 'min_scene_len'), ui_dict=ui_components,
                                    label="Min Scene Length (seconds)", minimum=0.1, step=0.1, info=SCENE_MIN_LENGTH_INFO
                                )
                                scene_threshold_num = create_number(
                                    config_path=('scene_split', 'threshold'), ui_dict=ui_components,
                                    label="Detection Threshold", minimum=0.1, maximum=10.0, step=0.1, info=SCENE_THRESHOLD_INFO
                                )
                            with gr.Row():
                                scene_drop_short_check = create_checkbox(
                                    config_path=('scene_split', 'drop_short'), ui_dict=ui_components,
                                    label="Drop Short Scenes", info=SCENE_DROP_SHORT_INFO
                                )
                                scene_merge_last_check = create_checkbox(
                                    config_path=('scene_split', 'merge_last'), ui_dict=ui_components,
                                    label="Merge Last Scene", info=SCENE_MERGE_LAST_INFO
                                )
                            with gr.Row():
                                scene_frame_skip_num = create_number(
                                    config_path=('scene_split', 'frame_skip'), ui_dict=ui_components,
                                    label="Frame Skip", minimum=0, step=1, info=SCENE_FRAME_SKIP_INFO
                                )
                                scene_min_content_val_num = create_number(
                                    config_path=('scene_split', 'min_content_val'), ui_dict=ui_components,
                                    label="Min Content Value", minimum=0.0, step=1.0, info=SCENE_MIN_CONTENT_INFO
                                )
                                scene_frame_window_num = create_number(
                                    config_path=('scene_split', 'frame_window'), ui_dict=ui_components,
                                    label="Frame Window", minimum=1, step=1, info=SCENE_FRAME_WINDOW_INFO
                                )
                        with gr.Group():
                            gr.Markdown(MANUAL_SPLIT_SETTINGS)
                            with gr.Row():
                                scene_manual_split_type_radio = create_radio(
                                    config_path=('scene_split', 'manual_split_type'), ui_dict=ui_components,
                                    label="Manual Split Type", choices=['duration', 'frame_count'], info=SCENE_MANUAL_SPLIT_TYPE_INFO
                                )
                                scene_manual_split_value_num = create_number(
                                    config_path=('scene_split', 'manual_split_value'), ui_dict=ui_components,
                                    label="Split Value", minimum=1.0, step=1.0, info=SCENE_MANUAL_SPLIT_VALUE_INFO
                                )
                        with gr.Group():
                            gr.Markdown(ENCODING_SETTINGS_SCENE_SEGMENTS)
                            with gr.Row():
                                scene_copy_streams_check = create_checkbox(
                                    config_path=('scene_split', 'copy_streams'), ui_dict=ui_components,
                                    label="Copy Streams", info=SCENE_COPY_STREAMS_INFO
                                )
                                scene_use_mkvmerge_check = create_checkbox(
                                    config_path=('scene_split', 'use_mkvmerge'), ui_dict=ui_components,
                                    label="Use MKVMerge", info=SCENE_USE_MKVMERGE_INFO
                                )
                            with gr.Row():
                                scene_rate_factor_num = create_number(
                                    config_path=('scene_split', 'rate_factor'), ui_dict=ui_components,
                                    label="Rate Factor (CRF)", minimum=0, maximum=51, step=1, info=SCENE_RATE_FACTOR_INFO
                                )
                                scene_preset_dropdown = create_dropdown(
                                    config_path=('scene_split', 'encoding_preset'), ui_dict=ui_components,
                                    label="Encoding Preset", choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                                    info=SCENE_ENCODING_PRESET_INFO
                                )
                            scene_quiet_ffmpeg_check = create_checkbox(
                                config_path=('scene_split', 'quiet_ffmpeg'), ui_dict=ui_components,
                                label="Quiet FFmpeg", info=SCENE_QUIET_FFMPEG_INFO
                            )

        with gr.Tab("Core Settings", id="core_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(UPSCALER_SELECTION_HEADER)
                        def get_upscaler_display_value(internal_value):
                            return {"star": "Use STAR Model Upscaler", "image_upscaler": "Use Image Based Upscalers", "seedvr2": "Use SeedVR2 Video Upscaler"}.get(internal_value, "Use Image Based Upscalers")
                        
                        initial_upscaler_display = get_upscaler_display_value(INITIAL_APP_CONFIG.upscaler_type.upscaler_type)
                        
                        upscaler_type_radio = create_radio(
                            config_path=('upscaler_type', 'upscaler_type'), ui_dict=ui_components,
                            label="Choose Your Upscaler Type",
                            choices=["Use STAR Model Upscaler", "Use Image Based Upscalers", "Use SeedVR2 Video Upscaler"],
                            value=initial_upscaler_display,
                            info=UPSCALER_TYPE_SELECTION_INFO
                        )
                    if UTIL_COG_VLM_AVAILABLE:
                        with gr.Accordion(info_strings.AUTO_CAPTIONING_COGVLM2_STAR_MODEL_ACCORDION, open=True):
                            cogvlm_quant_choices_map = get_cogvlm_quant_choices_map(torch.cuda.is_available(), UTIL_BITSANDBYTES_AVAILABLE)
                            cogvlm_quant_radio_choices_display = list(cogvlm_quant_choices_map.values())
                            
                            with gr.Row():
                                cogvlm_quant_radio = create_radio(
                                    config_path=('cogvlm', 'quant_display'), ui_dict=ui_components,
                                    label="CogVLM2 Quantization", choices=cogvlm_quant_radio_choices_display,
                                    info=COGVLM_QUANTIZATION_INFO, interactive=len(cogvlm_quant_radio_choices_display) > 1
                                )
                                cogvlm_unload_radio = create_radio(
                                    config_path=('cogvlm', 'unload_after_use'), ui_dict=ui_components,
                                    label="CogVLM2 After-Use", choices=['full', 'cpu'], info=CAPTION_UNLOAD_STRATEGY_INFO
                                )
                    else:
                        gr.Markdown(AUTO_CAPTIONING_DISABLED_NOTE)
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(FACE_RESTORATION_HEADER)
                        enable_face_restoration_check = create_checkbox(
                            config_path=('face_restoration', 'enable'), ui_dict=ui_components,
                            label="Enable Face Restoration", info=FACE_RESTORATION_ENABLE_INFO
                        )
                        face_restoration_fidelity_slider = create_slider(
                            config_path=('face_restoration', 'fidelity_weight'), ui_dict=ui_components,
                            label="Fidelity Weight", minimum=0.0, maximum=1.0, step=0.1,
                            info=FACE_RESTORATION_FIDELITY_INFO, interactive=True
                        )
                        with gr.Row():
                            enable_face_colorization_check = create_checkbox(
                                config_path=('face_restoration', 'enable_colorization'), ui_dict=ui_components,
                                label="Enable Colorization", info=FACE_COLORIZATION_INFO, interactive=True
                            )
                            face_restoration_when_radio = create_radio(
                                config_path=('face_restoration', 'when'), ui_dict=ui_components,
                                label="Apply Timing", choices=['before', 'after'], info=FACE_RESTORATION_TIMING_INFO, interactive=True
                            )
                        with gr.Row():
                            model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            default_model_choice = "codeformer.pth (359.2MB)" if INITIAL_APP_CONFIG.face_restoration.model and "codeformer.pth" in INITIAL_APP_CONFIG.face_restoration.model else "Auto (Default)"
                            
                            codeformer_model_dropdown = create_dropdown(
                                config_path=('face_restoration', 'model'), ui_dict=ui_components,
                                label="CodeFormer Model", choices=model_choices, value=default_model_choice,
                                info=CODEFORMER_MODEL_SELECTION_DETAILED_INFO, interactive=True
                            )
                        face_restoration_batch_size_slider = create_slider(
                            config_path=('face_restoration', 'batch_size'), ui_dict=ui_components,
                            label="Face Restoration Batch Size", minimum=1, maximum=50, step=1,
                            info=FACE_RESTORATION_BATCH_SIZE_INFO, interactive=True
                        )

        with gr.Tab("Star Upscaler", id="star_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(STAR_MODEL_SETTINGS_HEADER)
                        model_selector = create_dropdown(
                            config_path=('star_model', 'model_choice'), ui_dict=ui_components,
                            label=STAR_MODEL_TEMPORAL_UPSCALING_LABEL, choices=["Light Degradation", "Heavy Degradation"],
                            info=CODEFORMER_MODEL_SELECTION_INFO
                        )
                        upscale_factor_slider = create_slider(
                            config_path=('resolution', 'upscale_factor'), ui_dict=ui_components,
                            label=UPSCALE_FACTOR_TARGET_RES_DISABLED_LABEL, minimum=1.0, maximum=8.0, step=0.1,
                            info=UPSCALE_FACTOR_INFO
                        )
                        cfg_slider = create_slider(
                            config_path=('star_model', 'cfg_scale'), ui_dict=ui_components,
                            label="Guidance Scale (CFG)", minimum=1.0, maximum=15.0, step=0.5, info=GUIDANCE_SCALE_INFO
                        )
                        with gr.Row():
                            solver_mode_radio = create_radio(
                                config_path=('star_model', 'solver_mode'), ui_dict=ui_components,
                                label="Solver Mode", choices=['fast', 'normal'], info=SOLVER_MODE_INFO
                            )
                            steps_slider = create_slider(
                                config_path=('star_model', 'steps'), ui_dict=ui_components,
                                label="Diffusion Steps", minimum=5, maximum=100, step=1,
                                info=DENOISING_STEPS_INFO, interactive=False
                            )
                        color_fix_dropdown = create_dropdown(
                            config_path=('star_model', 'color_fix_method'), ui_dict=ui_components,
                            label="Color Correction", choices=['AdaIN', 'Wavelet', 'None'], info=COLOR_CORRECTION_INFO
                        )
                    with gr.Accordion(info_strings.CONTEXT_WINDOW_PREVIOUS_FRAMES_EXPERIMENTAL_ACCORDION, open=True):
                        enable_context_window_check = create_checkbox(
                            config_path=('context_window', 'enable'), ui_dict=ui_components,
                            label="Enable Context Window", info=CONTEXT_WINDOW_INFO
                        )
                        context_overlap_num = create_slider(
                            config_path=('context_window', 'overlap'), ui_dict=ui_components,
                            label="Context Overlap (frames)", minimum=0, maximum=31, step=1, info=CONTEXT_FRAMES_INFO
                        )
                with gr.Column(scale=1):
                    with gr.Accordion(info_strings.PERFORMANCE_VRAM_32_FRAMES_BEST_QUALITY_ACCORDION, open=True):
                        max_chunk_len_slider = create_slider(
                            config_path=('performance', 'max_chunk_len'), ui_dict=ui_components,
                            label="Max Frames per Chunk (VRAM)", minimum=1, maximum=1000, step=1, info=MAX_CHUNK_LEN_INFO
                        )
                        enable_chunk_optimization_check = create_checkbox(
                            config_path=('performance', 'enable_chunk_optimization'), ui_dict=ui_components,
                            label="Optimize Last Chunk Quality", info=CONTEXT_OVERLAP_INFO
                        )
                        vae_chunk_slider = create_slider(
                            config_path=('performance', 'vae_chunk'), ui_dict=ui_components,
                            label="VAE Decode Chunk (VRAM)", minimum=1, maximum=16, step=1, info=VAE_DECODE_BATCH_SIZE_INFO
                        )
                        enable_vram_optimization_check = create_checkbox(
                            config_path=('performance', 'enable_vram_optimization'), ui_dict=ui_components,
                            label=ENABLE_ADVANCED_VRAM_OPTIMIZATION_LABEL, info=ADVANCED_MEMORY_MANAGEMENT_INFO
                        )
                    with gr.Accordion(TILING_HIGH_RES_LOW_VRAM_ACCORDION, open=True, visible=False):
                        enable_tiling_check = create_checkbox(
                            config_path=('tiling', 'enable'), ui_dict=ui_components,
                            label="Enable Tiled Upscaling", info=TILING_INFO
                        )
                        with gr.Row():
                            tile_size_num = create_number(
                                config_path=('tiling', 'tile_size'), ui_dict=ui_components,
                                label="Tile Size (px, input res)", minimum=64, step=32, info=TILE_SIZE_INFO
                            )
                            tile_overlap_num = create_number(
                                config_path=('tiling', 'tile_overlap'), ui_dict=ui_components,
                                label="Tile Overlap (px, input res)", minimum=0, step=16, info=TILE_OVERLAP_INFO
                            )

        with gr.Tab("Image Based Upscalers", id="image_upscaler_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(IMAGE_BASED_UPSCALER_SETTINGS_HEADER)
                        gr.Markdown(IMAGE_BASED_UPSCALER_DESCRIPTION)
                        gr.Markdown(IMAGE_BASED_UPSCALER_NOTE)
                        try:
                            model_choices = util_scan_for_models(APP_CONFIG.paths.upscale_models_dir, logger) or [info_strings.NO_MODELS_FOUND_PLACE_UPSCALE_MODELS_STATUS]
                        except Exception as e:
                            logger.warning(f"Failed to scan for upscaler models: {e}")
                            model_choices = [info_strings.ERROR_SCANNING_UPSCALE_MODELS_DIRECTORY_STATUS]
                        
                        image_upscaler_model_dropdown = create_dropdown(
                            config_path=('image_upscaler', 'model'), ui_dict=ui_components,
                            label=SELECT_UPSCALER_MODEL_SPATIAL_LABEL, choices=model_choices,
                            info=IMAGE_UPSCALER_MODEL_INFO, interactive=True
                        )
                        image_upscaler_batch_size_slider = create_slider(
                            config_path=('image_upscaler', 'batch_size'), ui_dict=ui_components,
                            label="Batch Size", minimum=1, maximum=50, step=1,
                            info=IMAGE_UPSCALER_BATCH_SIZE_INFO, interactive=True
                        )
                    with gr.Group():
                        gr.Markdown(QUICK_PREVIEW_HEADER)
                        gr.Markdown(QUICK_PREVIEW_DESCRIPTION)
                        with gr.Row():
                            preview_single_btn = gr.Button("🖼️ Preview Current Model", variant="secondary", size="sm")
                            preview_all_models_btn = gr.Button("🔬 Test All Models & Save", variant="secondary", size="sm")
                        preview_status = gr.Textbox(
                            label="Preview Status", interactive=False, lines=2, visible=True,
                            value=DEFAULT_PREVIEW_STATUS, show_label=True
                        )
                        preview_slider = gr.ImageSlider(
                            label=BEFORE_AFTER_COMPARISON_SLIDER_LABEL, interactive=True, visible=False,
                            height=400, show_label=True, show_download_button=True,
                            show_fullscreen_button=True, slider_position=0.5
                        )
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(MODEL_INFORMATION_HEADER)
                        model_info_display = gr.Textbox(
                            label="Selected Model Info", value="Select a model to see its information",
                            interactive=False, lines=10, info=MODEL_DETAILS_DISPLAY_INFO
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh Model List", variant="secondary")
                        gr.Markdown(IMAGE_UPSCALER_OPTIMIZATION_TIPS)

        with gr.Tab("SeedVR2 Upscaler", id="seedvr2_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown(SEEDVR2_VIDEO_UPSCALER_HEADER)
                        gr.Markdown(SEEDVR2_DESCRIPTION)
                        seedvr2_dependency_status = gr.Textbox(
                            label="SeedVR2 Status", value="Checking SeedVR2 dependencies...", interactive=False,
                            lines=2, info=info_strings.SEEDVR2_INSTALLATION_DEPENDENCY_STATUS_INFO
                        )
                    with gr.Accordion("Model Selection", open=True):
                        try:
                            available_models = util_scan_seedvr2_models(logger=logger)
                            model_choices = [util_format_model_display_name(m) for m in available_models] if available_models else ["No SeedVR2 models found"]
                        except Exception as e:
                            logger.warning(f"Failed to scan for SeedVR2 models: {e}")
                            model_choices = [info_strings.ERROR_SCANNING_SEEDVR2_MODELS_DIRECTORY_STATUS]

                        # Convert stored filename to display name for initial value
                        def convert_seedvr2_filename_to_display_name(filename, available_models_list, model_choices_list):
                            if not model_choices_list:
                                return "No SeedVR2 models found"
                            
                            # If we have a filename from config/preset, try to find it
                            if filename and available_models_list:
                                for model_info in available_models_list:
                                    if model_info.get('filename') == filename:
                                        return util_format_model_display_name(model_info)
                            
                            # Only if no filename specified in config/preset - look for 3B FP8 model as default
                            if not filename:
                                for choice in model_choices_list:
                                    if "3B" in choice and "FP8" in choice:
                                        return choice
                            
                            # If 3B FP8 not found or filename was specified but not found, return first available choice
                            return model_choices_list[0]

                        # Get the model to display (respecting config/preset if set, otherwise defaulting to 3B FP8)
                        initial_seedvr2_display = convert_seedvr2_filename_to_display_name(
                            INITIAL_APP_CONFIG.seedvr2.model, available_models, model_choices
                        )
                        
                        logger.info(f"SeedVR2 Model Dropdown - Config model: {INITIAL_APP_CONFIG.seedvr2.model}, Choices: {model_choices}, Selected: {initial_seedvr2_display}")

                        seedvr2_model_dropdown = create_dropdown(
                            config_path=('seedvr2', 'model'), ui_dict=ui_components,
                            label="SeedVR2 Model", choices=model_choices, value=initial_seedvr2_display,
                            info=info_strings.SEEDVR2_MODEL_3B_7B_SPEED_QUALITY_INFO
                        )
                        with gr.Row():
                            refresh_seedvr2_models_btn = gr.Button("🔄 Refresh Models", variant="secondary", scale=1)
                            apply_recommended_settings_btn = gr.Button("⚡ Apply Optimal Settings", variant="primary", scale=1)
                        with gr.Row():
                            get_block_swap_recommendations_btn = gr.Button("🧠 Smart Block Swap", variant="secondary", scale=1)
                            get_multi_gpu_recommendations_btn = gr.Button("🚀 Multi-GPU Analysis", variant="secondary", scale=1)
                        seedvr2_model_info_display = gr.Textbox(
                            label="Model Information", value=info_strings.SELECT_MODEL_DETAILED_SPECIFICATIONS_INFO,
                            interactive=False, lines=8, info=info_strings.DETAILED_MODEL_SPECIFICATIONS_REQUIREMENTS_INFO
                        )
                    with gr.Accordion("Processing Settings", open=True):
                        seedvr2_batch_size_slider = create_slider(
                            config_path=('seedvr2', 'batch_size'), ui_dict=ui_components,
                            label=BATCH_SIZE_TEMPORAL_CONSISTENCY_LABEL, minimum=5, maximum=32, step=1,
                            info=SEEDVR2_BATCH_SIZE_INFO
                        )
                        seedvr2_temporal_overlap_slider = create_slider(
                            config_path=('seedvr2', 'temporal_overlap'), ui_dict=ui_components,
                            label="Temporal Overlap", minimum=0, maximum=8, step=1, info=SEEDVR2_TEMPORAL_OVERLAP_INFO
                        )
                        seedvr2_quality_preset_radio = create_radio(
                            config_path=('seedvr2', 'quality_preset'), ui_dict=ui_components,
                            label="Quality Preset", choices=["fast", "balanced", "quality"],
                            info=info_strings.PROCESSING_QUALITY_FAST_BALANCED_QUALITY_INFO
                        )
                        with gr.Row():
                            seedvr2_preserve_vram_check = create_checkbox(
                                config_path=('seedvr2', 'preserve_vram'), ui_dict=ui_components,
                                label="Preserve VRAM", info=info_strings.OPTIMIZE_VRAM_USAGE_RECOMMENDED_SYSTEMS_INFO
                            )
                            seedvr2_color_correction_check = create_checkbox(
                                config_path=('seedvr2', 'color_correction'), ui_dict=ui_components,
                                label="Color Correction (Wavelet)", info=info_strings.COLOR_FIX_WAVELET_RECONSTRUCTION_RECOMMENDED_INFO
                            )
                        with gr.Row():
                            seedvr2_enable_frame_padding_check = create_checkbox(
                                config_path=('seedvr2', 'enable_frame_padding'), ui_dict=ui_components,
                                label="Automatic Frame Padding", info=info_strings.OPTIMIZE_LAST_CHUNK_STAR_MODEL_RECOMMENDED_INFO
                            )
                            seedvr2_flash_attention_check = create_checkbox(
                                config_path=('seedvr2', 'flash_attention'), ui_dict=ui_components,
                                label="Flash Attention", info=info_strings.MEMORY_EFFICIENT_ATTENTION_DEFAULT_ENABLED_INFO
                            )
                        with gr.Accordion("🎬 Temporal Consistency", open=False):
                            with gr.Row():
                                seedvr2_scene_awareness_check = create_checkbox(
                                    config_path=('seedvr2', 'scene_awareness'), ui_dict=ui_components,
                                    label="🎭 Scene Awareness", info=info_strings.SCENE_AWARE_TEMPORAL_PROCESSING_BOUNDARIES_INFO
                                )
                                seedvr2_consistency_validation_check = create_checkbox(
                                    config_path=('seedvr2', 'consistency_validation'), ui_dict=ui_components,
                                    label="🎯 Consistency Validation", info=info_strings.TEMPORAL_CONSISTENCY_VALIDATION_QUALITY_METRICS_INFO
                                )
                            with gr.Row():
                                seedvr2_chunk_optimization_check = create_checkbox(
                                    config_path=('seedvr2', 'chunk_optimization'), ui_dict=ui_components,
                                    label="🔧 Chunk Optimization", info=info_strings.CHUNK_BOUNDARIES_TEMPORAL_COHERENCE_RECOMMENDED_INFO
                                )
                                seedvr2_temporal_quality_radio = create_radio(
                                    config_path=('seedvr2', 'temporal_quality'), ui_dict=ui_components,
                                    choices=["fast", "balanced", "quality"], label="🏆 Temporal Quality",
                                    info=info_strings.PROCESSING_SPEED_TEMPORAL_CONSISTENCY_BALANCE_INFO
                                )
                with gr.Column(scale=1):
                    with gr.Accordion("GPU Configuration", open=True):
                        with gr.Row():
                            seedvr2_use_gpu_check = create_checkbox(
                                config_path=('seedvr2', 'use_gpu'), ui_dict=ui_components,
                                label="Use GPU", info=info_strings.GPU_ACCELERATION_SEEDVR2_PROCESSING_INFO
                            )
                            seedvr2_enable_multi_gpu_check = create_checkbox(
                                config_path=('seedvr2', 'enable_multi_gpu'), ui_dict=ui_components,
                                label="Enable Multi-GPU", info=info_strings.MULTI_GPU_DISTRIBUTE_FASTER_PROCESSING_INFO
                            )
                        seedvr2_gpu_devices_textbox = create_textbox(
                            config_path=('seedvr2', 'gpu_devices'), ui_dict=ui_components,
                            label="GPU Device IDs", placeholder="0,1,2", info=info_strings.GPU_DEVICES_COMMA_SEPARATED_IDS_INFO
                        )
                        seedvr2_gpu_status_display = gr.Textbox(
                            label="GPU Status", value="Detecting available GPUs...", interactive=False,
                            lines=4, info=info_strings.AVAILABLE_GPUS_VRAM_STATUS_INFO
                        )
                    with gr.Accordion("Block Swap - VRAM Optimization", open=False):
                        gr.Markdown(BLOCK_SWAP_DESCRIPTION)
                        seedvr2_enable_block_swap_check = create_checkbox(
                            config_path=('seedvr2', 'enable_block_swap'), ui_dict=ui_components,
                            label="Enable Block Swap", info=info_strings.BLOCK_SWAP_LARGE_MODELS_LIMITED_VRAM_INFO
                        )
                        seedvr2_block_swap_counter_slider = create_slider(
                            config_path=('seedvr2', 'block_swap_counter'), ui_dict=ui_components,
                            label="Block Swap Counter", minimum=0, maximum=20, step=1,
                            info=info_strings.BLOCK_SWAP_COUNTER_VRAM_SAVINGS_SLOWER_INFO
                        )
                        with gr.Row():
                            seedvr2_block_swap_offload_io_check = create_checkbox(
                                config_path=('seedvr2', 'block_swap_offload_io'), ui_dict=ui_components,
                                label="I/O Component Offloading", info=info_strings.OFFLOAD_IO_LAYERS_MAXIMUM_VRAM_SAVINGS_INFO
                            )
                            seedvr2_block_swap_model_caching_check = create_checkbox(
                                config_path=('seedvr2', 'block_swap_model_caching'), ui_dict=ui_components,
                                label="Model Caching", info=info_strings.MODEL_CACHE_RAM_FASTER_BATCH_PROCESSING_INFO
                            )
                        seedvr2_block_swap_info_display = gr.Textbox(
                            label="Block Swap Status", value="Block swap disabled", interactive=False,
                            lines=3, info=info_strings.BLOCK_SWAP_CONFIGURATION_ESTIMATED_SAVINGS_INFO
                        )
                    with gr.Accordion("Chunk Preview Settings", open=True):
                        gr.Markdown(CHUNK_PREVIEW_DESCRIPTION)
                        seedvr2_enable_chunk_preview_check = create_checkbox(
                            config_path=('seedvr2', 'enable_chunk_preview'), ui_dict=ui_components,
                            label="Enable Chunk Preview", info=info_strings.CHUNK_PREVIEW_FUNCTIONALITY_MAIN_TAB_INFO
                        )
                        with gr.Row():
                            seedvr2_chunk_preview_frames_slider = create_slider(
                                config_path=('seedvr2', 'chunk_preview_frames'), ui_dict=ui_components,
                                label="Preview Frame Count", minimum=5, maximum=500, step=25,
                                info=info_strings.PREVIEW_FRAMES_DEFAULT_125_FRAMES_INFO
                            )
                            seedvr2_keep_last_chunks_slider = create_slider(
                                config_path=('seedvr2', 'keep_last_chunks'), ui_dict=ui_components,
                                label="Keep Last N Chunks", minimum=1, maximum=1000, step=1,
                                info=info_strings.CHUNK_RETENTION_DEFAULT_5_VIDEOS_INFO
                            )
                    with gr.Accordion("Advanced Settings", open=False):
                        seedvr2_cfg_scale_slider = create_slider(
                            config_path=('seedvr2', 'cfg_scale'), ui_dict=ui_components,
                            label="CFG Scale", minimum=0.5, maximum=2.0, step=0.1,
                            info=info_strings.GUIDANCE_SCALE_GENERATION_USUALLY_1_0_INFO
                        )

        with gr.Tab("Output & Comparison", id="output_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("FFmpeg Encoding Settings", open=True):
                        ffmpeg_use_gpu_check = create_checkbox(
                            config_path=('ffmpeg', 'use_gpu'), ui_dict=ui_components,
                            label=USE_NVIDIA_GPU_FFMPEG_LABEL, info=FFMPEG_GPU_ENCODING_INFO
                        )
                        with gr.Row():
                            ffmpeg_preset_dropdown = create_dropdown(
                                config_path=('ffmpeg', 'preset'), ui_dict=ui_components,
                                label="FFmpeg Preset", choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                                info=info_strings.FFMPEG_PRESET_ENCODING_SPEED_COMPRESSION_INFO
                            )
                            ffmpeg_quality_slider = create_slider(
                                config_path=('ffmpeg', 'quality'), ui_dict=ui_components,
                                label=info_strings.FFMPEG_QUALITY_CRF_LIBX264_CQ_NVENC_INFO, minimum=0, maximum=51, step=1,
                                info=info_strings.FFMPEG_QUALITY_DETAILED_CRF_CQ_RANGES_INFO
                            )
                        frame_folder_fps_slider = create_slider(
                            config_path=('frame_folder', 'fps'), ui_dict=ui_components,
                            label="Frame Folder FPS", minimum=1.0, maximum=120.0, step=0.001,
                            info=info_strings.FRAME_FOLDER_FPS_CONVERSION_COMMON_VALUES_INFO
                        )
                with gr.Column(scale=1):
                    with gr.Accordion("Output Options", open=True):
                        create_comparison_video_check = create_checkbox(
                            config_path=('outputs', 'create_comparison_video'), ui_dict=ui_components,
                            label="Generate Comparison Video", info=COMPARISON_VIDEO_INFO
                        )
                        save_frames_checkbox = create_checkbox(
                            config_path=('outputs', 'save_frames'), ui_dict=ui_components,
                            label=SAVE_INPUT_PROCESSED_FRAMES_LABEL, info=SAVE_FRAMES_INFO
                        )
                        save_metadata_checkbox = create_checkbox(
                            config_path=('outputs', 'save_metadata'), ui_dict=ui_components,
                            label="Save Processing Metadata", info=info_strings.SAVE_METADATA_TXT_PROCESSING_PARAMETERS_INFO
                        )
                        save_chunks_checkbox = create_checkbox(
                            config_path=('outputs', 'save_chunks'), ui_dict=ui_components,
                            label="Save Processed Chunks", info=info_strings.SAVE_CHUNKS_SUBFOLDER_FFMPEG_SETTINGS_INFO
                        )
                        save_chunk_frames_checkbox = create_checkbox(
                            config_path=('outputs', 'save_chunk_frames'), ui_dict=ui_components,
                            label=SAVE_CHUNK_INPUT_FRAMES_DEBUG_LABEL, info=SAVE_CHUNK_FRAMES_INFO
                        )
                    with gr.Accordion("Advanced: Seeding (Reproducibility)", open=True):
                        with gr.Row():
                            seed_num = create_number(
                                config_path=('seed', 'seed'), ui_dict=ui_components,
                                label="Seed", minimum=-1, maximum=2**32 - 1, step=1,
                                info=info_strings.SEED_REPRODUCIBILITY_RANDOM_IGNORED_INFO,
                                interactive=not INITIAL_APP_CONFIG.seed.use_random
                            )
                            random_seed_check = create_checkbox(
                                config_path=('seed', 'use_random'), ui_dict=ui_components,
                                label="Random Seed", info=info_strings.RANDOM_SEED_GENERATED_IGNORING_VALUE_INFO
                            )
                with gr.Column(scale=1):
                    with gr.Accordion("Manual Comparison Video Generator", open=True):
                        gr.Markdown(GENERATE_CUSTOM_COMPARISON_VIDEOS)
                        gr.Markdown(CUSTOM_COMPARISON_DESCRIPTION)
                        gr.Markdown(CUSTOM_COMPARISON_STEP1)
                        manual_video_count = create_radio(
                            config_path=('manual_comparison', 'video_count'), ui_dict=ui_components,
                            label="Number of Videos", choices=[2, 3, 4],
                            info=info_strings.VIDEO_COUNT_ADDITIONAL_INPUTS_SELECTION_INFO
                        )
                        gr.Markdown(CUSTOM_COMPARISON_STEP2)
                        manual_original_video = create_video_component(
                            config_path=('manual_comparison', 'original_video'), ui_dict=ui_components,
                            label="Video 1 (Original/Reference)", sources=["upload"], interactive=True, height=200
                        )
                        manual_upscaled_video = create_video_component(
                            config_path=('manual_comparison', 'upscaled_video'), ui_dict=ui_components,
                            label="Video 2 (Upscaled/Enhanced)", sources=["upload"], interactive=True, height=200
                        )
                        manual_third_video = create_video_component(
                            config_path=('manual_comparison', 'third_video'), ui_dict=ui_components,
                            label="Video 3 (Optional)", sources=["upload"], interactive=True, height=200, visible=False
                        )
                        manual_fourth_video = create_video_component(
                            config_path=('manual_comparison', 'fourth_video'), ui_dict=ui_components,
                            label="Video 4 (Optional)", sources=["upload"], interactive=True, height=200, visible=False
                        )
                        gr.Markdown(CUSTOM_COMPARISON_STEP3)
                        manual_comparison_layout = create_radio(
                            config_path=('manual_comparison', 'layout'), ui_dict=ui_components,
                            label="Comparison Layout", choices=["auto", "side_by_side", "top_bottom"], value="auto",
                            info=info_strings.LAYOUT_OPTIONS_VIDEO_SELECTION_INFO, interactive=True
                        )
                        gr.Markdown(CUSTOM_COMPARISON_STEP4)
                        manual_comparison_button = gr.Button("Generate Multi-Video Comparison", variant="primary", size="lg")
                        manual_comparison_status = gr.Textbox(label="Manual Comparison Status", lines=2, interactive=True, visible=False)
            with gr.Accordion("Comparison Video To See Difference", open=True):
                comparison_video = gr.Video(label="Comparison Video", interactive=False, height=512)

        with gr.Tab("Batch Upscaling", id="batch_tab"):
            with gr.Accordion("Batch Processing Options", open=True):
                with gr.Row():
                    batch_input_folder = create_textbox(
                        config_path=('batch', 'input_folder'), ui_dict=ui_components,
                        label="Input Folder", placeholder=info_strings.BATCH_INPUT_FOLDER_VIDEOS_PROCESS_PLACEHOLDER,
                        info=info_strings.BATCH_INPUT_FOLDER_VIDEO_FILES_MODE_INFO
                    )
                    batch_output_folder = create_textbox(
                        config_path=('batch', 'output_folder'), ui_dict=ui_components,
                        label="Output Folder", placeholder=info_strings.BATCH_OUTPUT_FOLDER_PROCESSED_VIDEOS_PLACEHOLDER,
                        info=info_strings.BATCH_OUTPUT_FOLDER_ORGANIZED_STRUCTURE_INFO
                    )
                with gr.Row():
                    enable_batch_frame_folders = create_checkbox(
                        config_path=('batch', 'enable_frame_folders'), ui_dict=ui_components,
                        label=PROCESS_FRAME_FOLDERS_BATCH_LABEL, value=False,
                        info=info_strings.BATCH_FRAME_FOLDERS_SUBFOLDERS_SEQUENCES_INFO
                    )
                    enable_direct_image_upscaling = create_checkbox(
                        config_path=('batch', 'enable_direct_image_upscaling'), ui_dict=ui_components,
                        label="Direct Image Upscaling", value=False,
                        info=info_strings.BATCH_DIRECT_IMAGE_JPG_PNG_UPSCALER_INFO
                    )
                with gr.Row():
                    batch_skip_existing = create_checkbox(
                        config_path=('batch', 'skip_existing'), ui_dict=ui_components,
                        label="Skip Existing Outputs", info=info_strings.BATCH_SKIP_EXISTING_INTERRUPTED_JOBS_INFO
                    )
                    batch_use_prompt_files = create_checkbox(
                        config_path=('batch', 'use_prompt_files'), ui_dict=ui_components,
                        label=USE_PROMPT_FILES_FILENAME_LABEL, info=BATCH_USE_PROMPT_FILES_INFO
                    )
                    batch_save_captions = create_checkbox(
                        config_path=('batch', 'save_captions'), ui_dict=ui_components,
                        label="Save Auto-Generated Captions", info=BATCH_SAVE_CAPTIONS_INFO
                    )
                if UTIL_COG_VLM_AVAILABLE:
                    with gr.Row():
                        batch_enable_auto_caption = create_checkbox(
                            config_path=('batch', 'enable_auto_caption'), ui_dict=ui_components,
                            label="Enable Auto-Caption for Batch", info=BATCH_ENABLE_AUTO_CAPTION_INFO
                        )
                else:
                    batch_enable_auto_caption = gr.Checkbox(visible=False, value=False)
            with gr.Row():
                batch_process_button = gr.Button("Start Batch Upscaling", variant="primary", icon="icons/split.png")
            create_batch_processing_help()

        with gr.Tab("FPS Increase - Decrease", id="fps_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("RIFE Interpolation Settings", open=True):
                        gr.Markdown(FRAME_INTERPOLATION_HEADER)
                        gr.Markdown(RIFE_DESCRIPTION)
                        enable_rife_interpolation = create_checkbox(
                            config_path=('rife', 'enable'), ui_dict=ui_components,
                            label="Enable RIFE Interpolation", info=RIFE_INTERPOLATION_INFO
                        )
                        rife_multiplier = create_radio(
                            config_path=('rife', 'multiplier'), ui_dict=ui_components,
                            label="FPS Multiplier", choices=[2, 4], info=RIFE_MULTIPLIER_INFO
                        )
                        with gr.Row():
                            rife_fp16 = create_checkbox(
                                config_path=('rife', 'fp16'), ui_dict=ui_components,
                                label="Use FP16 Precision", info=RIFE_FP16_INFO
                            )
                            rife_uhd = create_checkbox(
                                config_path=('rife', 'uhd'), ui_dict=ui_components,
                                label="UHD Mode", info=RIFE_UHD_INFO
                            )
                        rife_scale = create_slider(
                            config_path=('rife', 'scale'), ui_dict=ui_components,
                            label="Scale Factor", minimum=0.25, maximum=2.0, step=0.25, info=RIFE_SCALE_INFO
                        )
                        rife_skip_static = create_checkbox(
                            config_path=('rife', 'skip_static'), ui_dict=ui_components,
                            label="Skip Static Frames", info=RIFE_SKIP_STATIC_INFO
                        )
                    with gr.Accordion("Intermediate Processing", open=True):
                        gr.Markdown(APPLY_RIFE_TO_INTERMEDIATE)
                        rife_apply_to_chunks = create_checkbox(
                            config_path=('rife', 'apply_to_chunks'), ui_dict=ui_components,
                            label="Apply to Chunks", info=RIFE_APPLY_TO_CHUNKS_INFO
                        )
                        rife_apply_to_scenes = create_checkbox(
                            config_path=('rife', 'apply_to_scenes'), ui_dict=ui_components,
                            label="Apply to Scenes", info=RIFE_APPLY_TO_SCENES_INFO
                        )
                        gr.Markdown(RIFE_NOTE)
                with gr.Column(scale=1):
                    with gr.Accordion("FPS Decrease", open=True):
                        gr.Markdown(PRE_PROCESSING_FPS_REDUCTION_HEADER)
                        gr.Markdown(PRE_PROCESSING_FPS_REDUCTION_DESCRIPTION)
                        enable_fps_decrease = create_checkbox(
                            config_path=('fps_decrease', 'enable'), ui_dict=ui_components,
                            label="Enable FPS Decrease", info=info_strings.FPS_REDUCE_BEFORE_UPSCALING_SPEED_VRAM_INFO
                        )
                        fps_decrease_mode = create_radio(
                            config_path=('fps_decrease', 'mode'), ui_dict=ui_components,
                            label="FPS Reduction Mode", choices=["multiplier", "fixed"],
                            info=info_strings.FPS_MODE_MULTIPLIER_FIXED_AUTOMATIC_ADAPTATION_INFO
                        )
                        with gr.Group() as multiplier_controls:
                            with gr.Row():
                                fps_multiplier_preset = create_dropdown(
                                    config_path=('fps_decrease', 'multiplier_preset'), ui_dict=ui_components,
                                    label="FPS Multiplier", choices=list(util_get_common_fps_multipliers().values()) + ["Custom"],
                                    info=info_strings.FPS_MULTIPLIER_PRESET_SPEED_QUALITY_BALANCE_INFO
                                )
                                fps_multiplier_custom = create_number(
                                    config_path=('fps_decrease', 'multiplier_custom'), ui_dict=ui_components,
                                    label="Custom Multiplier", minimum=0.1, maximum=1.0, step=0.05,
                                    precision=2, visible=False, info=info_strings.FPS_MULTIPLIER_CUSTOM_LOWER_FRAMES_INFO
                                )
                        with gr.Group(visible=False) as fixed_controls:
                            target_fps = create_slider(
                                config_path=('fps_decrease', 'target_fps'), ui_dict=ui_components,
                                label="Target FPS", minimum=1.0, maximum=60.0, step=0.001,
                                info=info_strings.FPS_TARGET_FAST_CINEMA_STANDARD_INFO
                            )
                        fps_interpolation_method = create_radio(
                            config_path=('fps_decrease', 'interpolation_method'), ui_dict=ui_components,
                            label="Frame Reduction Method", choices=["drop", "blend"],
                            info=info_strings.FPS_REDUCTION_DROP_BLEND_MOTION_PRESERVATION_INFO
                        )
                        fps_calculation_info = gr.Markdown("**📊 Calculation:** Upload a video to see FPS reduction preview", visible=True)
                        gr.Markdown(WORKFLOW_TIP)
                    with gr.Accordion("FPS Limiting & Output Control", open=True):
                        rife_enable_fps_limit = create_checkbox(
                            config_path=('rife', 'enable_fps_limit'), ui_dict=ui_components,
                            label="Enable FPS Limiting", info=info_strings.RIFE_FPS_LIMIT_COMMON_VALUES_COMPATIBILITY_INFO
                        )
                        rife_max_fps_limit = create_radio(
                            config_path=('rife', 'max_fps_limit'), ui_dict=ui_components,
                            label="Max FPS Limit", choices=[23.976, 24, 25, 29.970, 30, 47.952, 48, 50, 59.940, 60, 75, 90, 100, 119.880, 120, 144, 165, 180, 240, 360],
                            info=info_strings.RIFE_MAX_FPS_NTSC_STANDARD_GAMING_INFO
                        )
                        with gr.Row():
                            rife_keep_original = create_checkbox(
                                config_path=('rife', 'keep_original'), ui_dict=ui_components,
                                label="Keep Original Files", info=info_strings.RIFE_KEEP_ORIGINAL_COMPARE_RESULTS_INFO
                            )
                            rife_overwrite_original = create_checkbox(
                                config_path=('rife', 'overwrite_original'), ui_dict=ui_components,
                                label="Overwrite Original", info=info_strings.RIFE_REPLACE_OUTPUT_PRIMARY_VERSION_INFO
                            )

        with gr.Tab("Edit Videos", id="edit_tab"):
            gr.Markdown(VIDEO_EDITOR_HEADER)
            gr.Markdown(VIDEO_EDITOR_DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        input_video_edit = gr.Video(label="Input Video for Editing", sources=["upload"], interactive=True, height=300)
                        video_info_display = gr.Textbox(
                            label="Video Information", interactive=False, lines=6,
                            info=info_strings.VIDEO_INFO_DURATION_FPS_FRAMES_RESOLUTION_INFO, value=info_strings.VIDEO_INFO_UPLOAD_DETAILED_INFORMATION
                        )
                    with gr.Row():
                        cut_and_save_btn = gr.Button("Cut and Save", variant="primary", icon="icons/cut_paste.png")
                        cut_and_upscale_btn = gr.Button("Cut and Move to Upscale", variant="primary", icon="icons/move_icon.png")
                    with gr.Accordion("Cutting Settings", open=True):
                        cutting_mode = gr.Radio(label="Cutting Mode", choices=['time_ranges', 'frame_ranges'], value='time_ranges', info=info_strings.CUTTING_MODE_TIME_FRAME_BASED_INFO)
                        with gr.Group() as time_range_controls:
                            time_ranges_input = gr.Textbox(
                                label="Time Ranges (seconds)", placeholder="1-3,5-8,10-15 or 1:30-2:45,3:00-4:30",
                                info=info_strings.TIME_RANGES_FORMAT_DECIMAL_MM_SS_INFO, lines=2
                            )
                        with gr.Group(visible=False) as frame_range_controls:
                            frame_ranges_input = gr.Textbox(
                                label="Frame Ranges", placeholder="30-90,150-210,300-450",
                                info=info_strings.FRAME_RANGES_FORMAT_ZERO_INDEXED_INFO, lines=2
                            )
                        cut_info_display = gr.Textbox(label="Cut Analysis", interactive=False, lines=3, info=CUT_ANALYSIS_INFO, value=DEFAULT_STATUS_MESSAGES['cut_analysis'])
                    with gr.Accordion("Options", open=True):
                        precise_cutting_mode = create_radio(
                            config_path=('video_editing', 'precise_cutting_mode'), ui_dict=ui_components,
                            label="Cutting Precision", choices=['precise', 'fast'], value='precise', info=info_strings.PRECISE_CUTTING_FRAME_ACCURATE_FAST_COPY_INFO
                        )
                        preview_first_segment = create_checkbox(
                            config_path=('video_editing', 'preview_first_segment'), ui_dict=ui_components,
                            label=GENERATE_PREVIEW_FIRST_SEGMENT_LABEL, info=info_strings.PREVIEW_FIRST_SEGMENT_VERIFICATION_INFO
                        )
                        processing_estimate = gr.Textbox(label="Processing Time Estimate", interactive=False, lines=1, value=DEFAULT_TIME_ESTIMATE_STATUS)
                with gr.Column(scale=1):
                    with gr.Group():
                        output_video_edit = gr.Video(label="Cut Video Output", interactive=False, height=400)
                        preview_video_edit = gr.Video(label="Preview (First Segment)", interactive=False, height=300)
                    with gr.Group():
                        edit_status_textbox = gr.Textbox(label="Edit Status & Log", interactive=False, lines=8, max_lines=15, value=DEFAULT_STATUS_MESSAGES['ready_video'])
                    with gr.Accordion("Quick Help & Examples", open=True):
                        gr.Markdown(VIDEO_EDITING_HELP)

        with gr.Tab("Face Restoration", id="face_restoration_tab"):
            gr.Markdown(STANDALONE_FACE_RESTORATION_HEADER)
            gr.Markdown(STANDALONE_FACE_RESTORATION_DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        input_video_face_restoration = create_video_component(
                            config_path=('standalone_face_restoration', 'input_video'), ui_dict=ui_components,
                            label=INPUT_VIDEO_FACE_RESTORATION_LABEL, sources=["upload"], interactive=True, height=400
                        )
                        face_restoration_mode = create_radio(
                            config_path=('standalone_face_restoration', 'mode'), ui_dict=ui_components,
                            label="Processing Mode", choices=["Single Video", "Batch Folder"], value="Single Video", info=info_strings.PROCESSING_MODE_SINGLE_BATCH_FOLDER_INFO
                        )
                        with gr.Group(visible=False) as batch_folder_controls:
                            batch_input_folder_face = create_textbox(
                                config_path=('standalone_face_restoration', 'batch_input_folder'), ui_dict=ui_components,
                                label="Input Folder Path", placeholder="C:/path/to/input/videos/", info=info_strings.FACE_RESTORATION_INPUT_FOLDER_VIDEOS_INFO
                            )
                            batch_output_folder_face = create_textbox(
                                config_path=('standalone_face_restoration', 'batch_output_folder'), ui_dict=ui_components,
                                label="Output Folder Path", placeholder="C:/path/to/output/videos/", info=info_strings.FACE_RESTORATION_OUTPUT_FOLDER_VIDEOS_INFO
                            )
                    with gr.Row():
                        face_restoration_process_btn = gr.Button("Process Face Restoration", variant="primary", icon="icons/face_restoration.png")
                        face_restoration_stop_btn = gr.Button("Stop Processing", variant="stop")
                    with gr.Accordion("Face Restoration Settings", open=True):
                        standalone_enable_face_restoration = create_checkbox(
                            config_path=('standalone_face_restoration', 'enable'), ui_dict=ui_components,
                            label="Enable Face Restoration", info=info_strings.FACE_RESTORATION_ENABLE_PROCESSING_OCCUR_INFO
                        )
                        standalone_face_restoration_fidelity = create_slider(
                            config_path=('standalone_face_restoration', 'fidelity_weight'), ui_dict=ui_components,
                            label=FACE_RESTORATION_FIDELITY_WEIGHT_LABEL, minimum=0.0, maximum=1.0, step=0.05, info=info_strings.FACE_RESTORATION_FIDELITY_BALANCE_RECOMMENDED_INFO
                        )
                        standalone_enable_face_colorization = create_checkbox(
                            config_path=('standalone_face_restoration', 'enable_colorization'), ui_dict=ui_components,
                            label="Enable Face Colorization", info=info_strings.FACE_COLORIZATION_GRAYSCALE_OLD_VIDEOS_INFO
                        )
                        with gr.Row():
                            standalone_model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
                            standalone_default_model_choice = "Auto (Default)"
                            standalone_codeformer_model_dropdown = create_dropdown(
                                config_path=('standalone_face_restoration', 'codeformer_model'), ui_dict=ui_components,
                                label="CodeFormer Model", choices=standalone_model_choices, value=standalone_default_model_choice,
                                info=info_strings.CODEFORMER_MODEL_AUTO_PRETRAINED_WEIGHT_INFO
                            )
                        standalone_face_restoration_batch_size = create_slider(
                            config_path=('standalone_face_restoration', 'batch_size'), ui_dict=ui_components,
                            label="Processing Batch Size", minimum=1, maximum=50, step=1, info=info_strings.FACE_RESTORATION_BATCH_SIZE_SIMULTANEOUS_VRAM_INFO
                        )
                    with gr.Accordion("Advanced Options", open=True):
                        standalone_save_frames = create_checkbox(
                            config_path=('standalone_face_restoration', 'save_frames'), ui_dict=ui_components,
                            label="Save Individual Frames", info=info_strings.SAVE_PROCESSED_FRAMES_INDIVIDUAL_FILES_INFO
                        )
                        standalone_create_comparison = create_checkbox(
                            config_path=('standalone_face_restoration', 'create_comparison'), ui_dict=ui_components,
                            label=CREATE_BEFORE_AFTER_COMPARISON_VIDEO_LABEL, info=info_strings.COMPARISON_VIDEO_SIDE_BY_SIDE_ORIGINAL_RESTORED_INFO
                        )
                        standalone_preserve_audio = create_checkbox(
                            config_path=('standalone_face_restoration', 'preserve_audio'), ui_dict=ui_components,
                            label="Preserve Original Audio", info=info_strings.PRESERVE_AUDIO_TRACK_PROCESSED_VIDEO_INFO
                        )
                with gr.Column(scale=1):
                    with gr.Group():
                        output_video_face_restoration = gr.Video(label="Face Restored Video", interactive=False, height=400)
                        comparison_video_face_restoration = gr.Video(label="Before/After Comparison", interactive=False, height=300, visible=True)
                    with gr.Group():
                        face_restoration_status = gr.Textbox(
                            label="Face Restoration Status & Log", interactive=False, lines=10, max_lines=20,
                            value=DEFAULT_STATUS_MESSAGES['ready_face_restoration']
                        )
                    with gr.Accordion("Processing Statistics", open=True):
                        face_restoration_stats = gr.Textbox(
                            label="Processing Stats", interactive=False, lines=4, value=DEFAULT_STATUS_MESSAGES['ready_processing_stats']
                        )
                    with gr.Accordion("Face Restoration Help", open=True):
                        gr.Markdown(FACE_RESTORATION_HELP)

    # --- End of UI Definition ---

    # Create an ordered list of all components that map to the AppConfig
    # This order MUST match the order of components in the UI definition
    # This list will be used to build the `inputs` for event handlers
    component_order = [
        # Main Tab
        user_prompt, pos_prompt, neg_prompt, auto_caption_then_upscale_check, gpu_selector,
        # CogVLM (conditional)
        cogvlm_quant_radio if UTIL_COG_VLM_AVAILABLE else gr.State(None),
        cogvlm_unload_radio if UTIL_COG_VLM_AVAILABLE else gr.State(None),
        # Enhanced Input
        input_frames_folder,
        # Resolution Tab
        enable_target_res_check, target_res_mode_radio, target_w_num, target_h_num,
        enable_auto_aspect_resolution_check, auto_resolution_status_display,
        # Scene Split Tab
        enable_scene_split_check, scene_split_mode_radio, scene_min_scene_len_num, scene_threshold_num,
        scene_drop_short_check, scene_merge_last_check, scene_frame_skip_num, scene_min_content_val_num,
        scene_frame_window_num, scene_manual_split_type_radio, scene_manual_split_value_num,
        scene_copy_streams_check, scene_use_mkvmerge_check, scene_rate_factor_num, scene_preset_dropdown,
        scene_quiet_ffmpeg_check,
        # Core Settings Tab
        upscaler_type_radio, enable_face_restoration_check, face_restoration_fidelity_slider,
        enable_face_colorization_check, face_restoration_when_radio, codeformer_model_dropdown,
        face_restoration_batch_size_slider,
        # Star Upscaler Tab
        model_selector, upscale_factor_slider, cfg_slider, solver_mode_radio, steps_slider,
        color_fix_dropdown, enable_context_window_check, context_overlap_num,
        # Performance & VRAM
        max_chunk_len_slider, enable_chunk_optimization_check, vae_chunk_slider, enable_vram_optimization_check,
        # Tiling
        enable_tiling_check, tile_size_num, tile_overlap_num,
        # Image Based Upscaler Tab
        image_upscaler_model_dropdown, image_upscaler_batch_size_slider,
        # SeedVR2 Tab
        seedvr2_model_dropdown, seedvr2_batch_size_slider, seedvr2_temporal_overlap_slider,
        seedvr2_quality_preset_radio, seedvr2_preserve_vram_check, seedvr2_color_correction_check,
        seedvr2_enable_frame_padding_check, seedvr2_flash_attention_check, seedvr2_scene_awareness_check,
        seedvr2_consistency_validation_check, seedvr2_chunk_optimization_check, seedvr2_temporal_quality_radio,
        seedvr2_use_gpu_check, seedvr2_enable_multi_gpu_check, seedvr2_gpu_devices_textbox,
        seedvr2_enable_block_swap_check, seedvr2_block_swap_counter_slider, seedvr2_block_swap_offload_io_check,
        seedvr2_block_swap_model_caching_check, seedvr2_cfg_scale_slider, seedvr2_enable_chunk_preview_check,
        seedvr2_chunk_preview_frames_slider, seedvr2_keep_last_chunks_slider,
        # Output & Comparison Tab
        ffmpeg_use_gpu_check, ffmpeg_preset_dropdown, ffmpeg_quality_slider, frame_folder_fps_slider,
        create_comparison_video_check, save_frames_checkbox, save_metadata_checkbox, save_chunks_checkbox,
        save_chunk_frames_checkbox, seed_num, random_seed_check,
        # Manual Comparison
        manual_video_count, manual_original_video, manual_upscaled_video, manual_third_video,
        manual_fourth_video, manual_comparison_layout,
        # FPS Tab
        enable_rife_interpolation, rife_multiplier, rife_fp16, rife_uhd, rife_scale, rife_skip_static,
        rife_apply_to_chunks, rife_apply_to_scenes, rife_enable_fps_limit, rife_max_fps_limit,
        rife_keep_original, rife_overwrite_original,
        enable_fps_decrease, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom,
        target_fps, fps_interpolation_method,
        # Edit Videos Tab
        precise_cutting_mode, preview_first_segment,
        # Face Restoration Tab
        input_video_face_restoration, face_restoration_mode, batch_input_folder_face, batch_output_folder_face,
        standalone_enable_face_restoration, standalone_face_restoration_fidelity, standalone_enable_face_colorization,
        standalone_codeformer_model_dropdown, standalone_face_restoration_batch_size,
        standalone_save_frames, standalone_create_comparison, standalone_preserve_audio,
        # Batch Tab
        batch_input_folder, batch_output_folder, enable_batch_frame_folders, enable_direct_image_upscaling,
        batch_skip_existing, batch_use_prompt_files, batch_save_captions,
        batch_enable_auto_caption if UTIL_COG_VLM_AVAILABLE else gr.State(False),
        # Single Image Upscale Tab
        image_upscaler_type_radio, image_preserve_aspect_ratio, image_output_format,
        image_quality_level, image_preserve_metadata, image_custom_suffix
    ]

    # Create a reverse map for faster lookups in build_app_config
    component_map = {comp: key for key, comp in ui_components.items()}

    def update_steps_display(mode):
        if mode == 'fast':
            return gr.update(value=INITIAL_APP_CONFIG.star_model.steps, interactive=False)
        else:
            from logic.star_dataclasses import DEFAULT_DIFFUSION_STEPS_NORMAL
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

    def validate_enhanced_input_wrapper(input_path):
        if not input_path or not input_path.strip():
            return DEFAULT_STATUS_MESSAGES['validate_input']
        from logic.frame_folder_utils import validate_input_path
        is_valid, message, metadata = validate_input_path(input_path, logger)
        if is_valid:
            if metadata.get("frame_count"):
                formats = metadata.get("supported_formats", [])
                detail_msg = f"\n📊 Detected formats: {', '.join(formats)}" if formats else ""
                return f"{message}{detail_msg}"
            elif metadata.get("duration"):
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

    def update_image_upscaler_controls(upscaler_type_selection):
        enable_controls = (upscaler_type_selection == "Use Image Based Upscalers")
        return [
            gr.update(interactive=enable_controls),
            gr.update(interactive=enable_controls)
        ]

    upscaler_type_radio.change(
        fn=update_image_upscaler_controls,
        inputs=upscaler_type_radio,
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

    def update_direct_image_upscaling_info(enable_direct_upscaling, upscaler_type):
        if enable_direct_upscaling:
            if upscaler_type != "Use Image Based Upscalers":
                return gr.update(info=DIRECT_IMAGE_UPSCALING_WARNING)
            else:
                return gr.update(info=DIRECT_IMAGE_UPSCALING_SUCCESS)
        else:
            return gr.update(info=DIRECT_IMAGE_UPSCALING_DEFAULT)

    face_restoration_mode.change(
        fn=update_face_restoration_mode_controls,
        inputs=face_restoration_mode,
        outputs=batch_folder_controls
    )

    for component in [enable_direct_image_upscaling, upscaler_type_radio]:
        component.change(
            fn=update_direct_image_upscaling_info,
            inputs=[enable_direct_image_upscaling, upscaler_type_radio],
            outputs=[enable_direct_image_upscaling]
        )

    def refresh_upscaler_models():
        try:
            available_model_files = util_scan_for_models(APP_CONFIG.paths.upscale_models_dir, logger)
            if available_model_files:
                model_choices = available_model_files
                default_choice = model_choices[0]
            else:
                model_choices = [info_strings.NO_MODELS_FOUND_PLACE_UPSCALE_MODELS_STATUS]
                default_choice = model_choices[0]
            return gr.update(choices=model_choices, value=default_choice)
        except Exception as e:
            logger.warning(f"Failed to refresh upscaler models: {e}")
            return gr.update(choices=[info_strings.ERROR_SCANNING_UPSCALE_MODELS_DIRECTORY_STATUS],
                                    value=MODEL_SCAN_ERROR_STATUS)

    refresh_models_btn.click(
        fn=refresh_upscaler_models,
        inputs=[],
        outputs=[image_upscaler_model_dropdown]
    )

    def update_model_info_display(selected_model):
        if not selected_model or selected_model.startswith("No models") or selected_model.startswith("Error"):
            return "Select a model to see its information"
        try:
            model_path = os.path.join(APP_CONFIG.paths.upscale_models_dir, selected_model)
            model_info = util_get_model_info(model_path, logger)
            if model_info and "error" not in model_info:
                try:
                    file_size = os.path.getsize(model_path)
                    file_size_mb = file_size / (1024 * 1024)
                    file_size_str = f"{file_size_mb:.1f} MB"
                except:
                    file_size_str = "Unknown"
                model_name = os.path.splitext(selected_model)[0]
                info_text = MODEL_INFO_DISPLAY_TEMPLATE.format(
                    model_name=model_name,
                    scale_factor=model_info.get('scale', 'Unknown'),
                    vram_usage="Unknown",
                    architecture=model_info.get('architecture_name', model_info.get('architecture', 'Unknown')),
                    input_size=f"{model_info.get('input_channels', 'Unknown')} channels",
                    output_size=f"{model_info.get('output_channels', 'Unknown')} channels",
                    supports_bfloat16=model_info.get('supports_bfloat16', False)
                )
                return info_text
            elif model_info and "error" in model_info:
                return f"Error loading model: {model_info['error']}"
            else:
                return f"Could not load information for {selected_model}"
        except Exception as e:
            logger.warning(f"Failed to get model info for {selected_model}: {e}")
            return f"Error loading model info: {str(e)}"

    image_upscaler_model_dropdown.change(
        fn=update_model_info_display,
        inputs=[image_upscaler_model_dropdown],
        outputs=[model_info_display]
    )

    def preview_single_model_wrapper(
        input_video_val, model_dropdown_val,
        enable_target_res_val, target_h_val, target_w_val, target_res_mode_val,
        gpu_selector_val, progress=gr.Progress(track_tqdm=True)
    ):
        try:
            if not input_video_val:
                return (
                    gr.update(visible=True, value="❌ Please upload a video first"),
                    gr.update(visible=False)
                )
            progress(0.0, "🚀 Starting preview generation...")
            available_gpus = util_get_available_gpus()
            if available_gpus and gpu_selector_val:
                device = "cuda" if "cuda" in str(gpu_selector_val).lower() else "cpu"
            else:
                device = "cpu"
            progress(0.2, "📹 Extracting first frame from video...")
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="preview_single_")
            progress(0.4, f"🔧 Loading {model_dropdown_val} model...")
            progress(0.6, f"🎨 Upscaling frame with {model_dropdown_val}...")
            result = util_preview_single_model(
                video_path=input_video_val,
                model_name=model_dropdown_val,
                upscale_models_dir=APP_CONFIG.paths.upscale_models_dir,
                temp_dir=temp_dir,
                device=device,
                apply_resolution_constraints=enable_target_res_val,
                target_h=target_h_val,
                target_w=target_w_val,
                target_res_mode=target_res_mode_val,
                logger=logger
            )
            progress(0.9, "✨ Finalizing preview...")
            if result['success']:
                progress(1.0, "✅ Preview generation complete!")
                status_msg = f"✅ {result['message']}\n📊 {result['original_resolution']} → {result['output_resolution']} in {result['processing_time']:.2f}s"
                slider_images = (result['original_image_path'], result['preview_image_path'])
                return (
                    gr.update(visible=True, value=status_msg),
                    gr.update(visible=True, value=slider_images)
                )
            else:
                return (
                    gr.update(visible=True, value=f"❌ {result['error']}"),
                    gr.update(visible=False)
                )
        except Exception as e:
            logger.error(f"Preview single model error: {e}")
            return (
                gr.update(visible=True, value=f"❌ Error: {str(e)}"),
                gr.update(visible=False)
            )

    def preview_all_models_wrapper(
        input_video_val,
        enable_target_res_val, target_h_val, target_w_val, target_res_mode_val,
        gpu_selector_val, progress=gr.Progress(track_tqdm=True)
    ):
        try:
            if not input_video_val:
                return (
                    gr.update(visible=True, value="❌ Please upload a video first"),
                    gr.update(visible=False)
                )
            progress(0.0, "🚀 Initializing model comparison test...")
            available_gpus = util_get_available_gpus()
            if available_gpus and gpu_selector_val:
                device = "cuda" if "cuda" in str(gpu_selector_val).lower() else "cpu"
            else:
                device = "cpu"
            progress(0.1, "🔍 Scanning for upscaler models...")
            def progress_callback(prog_val, desc):
                mapped_progress = DEFAULT_PROGRESS_OFFSET + (prog_val * DEFAULT_PROGRESS_SCALE)
                progress(mapped_progress, f"🎨 {desc}")
            result = util_preview_all_models(
                video_path=input_video_val,
                upscale_models_dir=APP_CONFIG.paths.upscale_models_dir,
                output_dir=APP_CONFIG.paths.outputs_dir,
                device=device,
                apply_resolution_constraints=enable_target_res_val,
                target_h=target_h_val,
                target_w=target_w_val,
                target_res_mode=target_res_mode_val,
                logger=logger,
                progress_callback=progress_callback
            )
            progress(0.95, "📁 Organizing results...")
            if result['success']:
                progress(1.0, "✅ Model comparison test complete!")
                status_msg = f"✅ {result['message']}\n⏱️ Processed in {result['processing_time']:.1f}s"
                if result['failed_models']:
                    status_msg += f"\n⚠️ {len(result['failed_models'])} models failed"
                original_frame_path = os.path.join(result['output_folder'], "00_original.png")
                if os.path.exists(original_frame_path):
                    import glob
                    processed_files = glob.glob(os.path.join(result['output_folder'], "[0-9][0-9]_*.png"))
                    if processed_files:
                        processed_files.sort()
                        first_processed = processed_files[0]
                        slider_images = (original_frame_path, first_processed)
                        return (
                            gr.update(visible=True, value=status_msg),
                            gr.update(visible=True, value=slider_images)
                        )
                    else:
                        slider_images = (original_frame_path, original_frame_path)
                        return (
                            gr.update(visible=True, value=status_msg),
                            gr.update(visible=True, value=slider_images)
                        )
                else:
                    return (
                        gr.update(visible=True, value=status_msg),
                        gr.update(visible=False)
                    )
            else:
                return (
                    gr.update(visible=True, value=f"❌ {result['error']}"),
                    gr.update(visible=False)
                )
        except Exception as e:
            logger.error(f"Preview all models error: {e}")
            return (
                gr.update(visible=True, value=f"❌ Error: {str(e)}"),
                gr.update(visible=False)
            )

    preview_single_btn.click(
        fn=preview_single_model_wrapper,
        inputs=[
            input_video, image_upscaler_model_dropdown,
            enable_target_res_check, target_h_num, target_w_num, target_res_mode_radio,
            gpu_selector
        ],
        outputs=[preview_status, preview_slider],
        show_progress_on=[preview_status]
    )

    preview_all_models_btn.click(
        fn=preview_all_models_wrapper,
        inputs=[
            input_video,
            enable_target_res_check, target_h_num, target_w_num, target_res_mode_radio,
            gpu_selector
        ],
        outputs=[preview_status, preview_slider],
        show_progress_on=[preview_status]
    )

    def update_context_overlap_max(max_chunk_len):
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
        from logic.star_dataclasses import DEFAULT_FFMPEG_QUALITY_GPU, DEFAULT_FFMPEG_QUALITY_CPU
        if use_gpu_ffmpeg:
            return gr.Slider(label=FFMPEG_QUALITY_CRF_LIBX264_CQ_NVENC_LABEL, value=DEFAULT_FFMPEG_QUALITY_GPU, info=info_strings.FFMPEG_H264_NVENC_CQ_QUALITY_RANGE_INFO)
        else:
            return gr.Slider(label=FFMPEG_QUALITY_CRF_LIBX264_LABEL, value=DEFAULT_FFMPEG_QUALITY_CPU, info=info_strings.FFMPEG_LIBX264_CRF_QUALITY_LOSSLESS_DEFAULT_INFO)

    ffmpeg_use_gpu_check.change(
        fn=update_ffmpeg_quality_settings,
        inputs=ffmpeg_use_gpu_check,
        outputs=ffmpeg_quality_slider
    )

    open_output_folder_button.click(
        fn=lambda: util_open_folder(APP_CONFIG.paths.outputs_dir, logger=logger),
        inputs=[],
        outputs=[]
    )

    def open_how_to_use():
        try:
            how_to_use_path = os.path.join(base_path, "How_To_Use.html")
            if os.path.exists(how_to_use_path):
                webbrowser.open(f"file://{os.path.abspath(how_to_use_path)}")
                logger.info(f"Opened How To Use documentation: {how_to_use_path}")
            else:
                logger.warning(f"How To Use documentation not found: {how_to_use_path}")
        except Exception as e:
            logger.error(f"Failed to open How To Use documentation: {e}")

    def open_version_history():
        try:
            version_history_path = os.path.join(base_path, "Version_History.html")
            if os.path.exists(version_history_path):
                webbrowser.open(f"file://{os.path.abspath(version_history_path)}")
                logger.info(f"Opened Version History: {version_history_path}")
            else:
                logger.warning(f"Version History not found: {version_history_path}")
        except Exception as e:
            logger.error(f"Failed to open Version History: {e}")

    how_to_use_button.click(
        fn=open_how_to_use,
        inputs=[],
        outputs=[]
    )

    version_history_button.click(
        fn=open_version_history,
        inputs=[],
        outputs=[]
    )

    cogvlm_display_to_quant_val_map_global = {}
    if UTIL_COG_VLM_AVAILABLE:
        _temp_map = get_cogvlm_quant_choices_map(torch.cuda.is_available(), UTIL_BITSANDBYTES_AVAILABLE)
        cogvlm_display_to_quant_val_map_global = {v: k for k, v in _temp_map.items()}

    def get_quant_value_from_display(display_val):
        if display_val is None: return 0
        if isinstance(display_val, int): return display_val
        return cogvlm_display_to_quant_val_map_global.get(display_val, 0)

    def extract_codeformer_model_path_from_dropdown(dropdown_choice):
        if not dropdown_choice or dropdown_choice.startswith("No CodeFormer") or dropdown_choice.startswith("Error"):
            return None
        if dropdown_choice == "Auto (Default)":
            return None
        if dropdown_choice.startswith("codeformer.pth"):
            return os.path.join(APP_CONFIG.paths.face_restoration_models_dir, "CodeFormer", "codeformer.pth")
        return None

    def extract_gpu_index_from_dropdown(dropdown_choice):
        if dropdown_choice is None or dropdown_choice == "No CUDA GPUs detected":
            return 0
        available_gpus = util_get_available_gpus()
        if not available_gpus:
            return 0
        if isinstance(dropdown_choice, int):
            if 0 <= dropdown_choice < len(available_gpus):
                return dropdown_choice
            else:
                logger.warning(f"GPU index {dropdown_choice} is out of range. Available GPUs: {len(available_gpus)}. Defaulting to 0.")
                return 0
        if isinstance(dropdown_choice, str) and dropdown_choice.startswith("GPU "):
            try:
                gpu_index = int(dropdown_choice.split(":")[0].replace("GPU ", "").strip())
                if 0 <= gpu_index < len(available_gpus):
                    return gpu_index
                else:
                    logger.warning(f"GPU index {gpu_index} is out of range. Available GPUs: {len(available_gpus)}. Defaulting to 0.")
                    return 0
            except:
                logger.warning(f"Failed to parse GPU index from '{dropdown_choice}'. Defaulting to 0.")
                return 0
        try:
            choice_index = available_gpus.index(dropdown_choice)
            return choice_index
        except ValueError:
            logger.warning(f"GPU choice '{dropdown_choice}' not found in available GPUs. Defaulting to 0.")
            return 0

    def convert_gpu_index_to_dropdown(gpu_index, available_gpus):
        if not available_gpus:
            return "No CUDA GPUs detected"
        if gpu_index is None or gpu_index == "Auto":
            return available_gpus[0] if available_gpus else "No CUDA GPUs detected"
        try:
            if isinstance(gpu_index, int):
                gpu_num = gpu_index
            else:
                gpu_num = int(gpu_index)
            if 0 <= gpu_num < len(available_gpus):
                return available_gpus[gpu_num]
            else:
                logger.warning(f"GPU index {gpu_num} is out of range. Available GPUs: {len(available_gpus)}. Defaulting to GPU 0.")
                return available_gpus[0] if available_gpus else "No CUDA GPUs detected"
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert GPU index '{gpu_index}' to integer. Defaulting to GPU 0.")
            return available_gpus[0] if available_gpus else "No CUDA GPUs detected"

    def build_app_config_from_ui(*component_values):
        # Create a mapping from component object to its value from the *component_values list
        ui_values = dict(zip([input_video] + component_order, component_values))

        # Create a default config object to populate
        config = create_app_config(base_path, args.outputs_folder, star_cfg)

        # Iterate through our registered components and set values on the config object
        for (section, key), component in ui_components.items():
            if component in ui_values:
                value = ui_values[component]

                # Apply special transformations for specific components
                if component is upscaler_type_radio:
                    value = {"Use STAR Model Upscaler": "star", "Use Image Based Upscalers": "image_upscaler", "Use SeedVR2 Video Upscaler": "seedvr2"}.get(value, "image_upscaler")
                elif component is gpu_selector:
                    value = str(extract_gpu_index_from_dropdown(value))
                elif component is codeformer_model_dropdown:
                    value = extract_codeformer_model_path_from_dropdown(value)
                elif component is standalone_codeformer_model_dropdown:
                    value = extract_codeformer_model_path_from_dropdown(value)
                elif component is seedvr2_model_dropdown:
                    value = util_extract_model_filename_from_dropdown(value) if value else None
                elif UTIL_COG_VLM_AVAILABLE and component is cogvlm_quant_radio:
                    # This component's value is already the display string, which is what we save.
                    # The actual quant value is derived from it later.
                    pass

                # Set the attribute on the config object
                if hasattr(config, section) and hasattr(getattr(config, section), key):
                    setattr(getattr(config, section), key, value)

        # Handle values that are not in ui_components or need special logic
        config.input_video_path = ui_values.get(input_video)

        if UTIL_COG_VLM_AVAILABLE:
            config.cogvlm.quant_value = get_quant_value_from_display(ui_values.get(cogvlm_quant_radio))

        # Handle frame folder detection
        frame_folder_enable = False
        input_path = ui_values.get(input_frames_folder)
        if input_path and input_path.strip():
            from logic.frame_folder_utils import detect_input_type
            input_type, _, _ = detect_input_type(input_path, logger)
            frame_folder_enable = (input_type == "frames_folder")
        config.frame_folder.enable = frame_folder_enable

        # Handle upscaler type enable flags
        selected_upscaler_type = config.upscaler_type.upscaler_type
        config.image_upscaler.enable = (selected_upscaler_type == "image_upscaler")
        config.seedvr2.enable = (selected_upscaler_type == "seedvr2")

        return config

    def upscale_director_logic(app_config: AppConfig, progress=gr.Progress(track_tqdm=True)):
        cancellation_manager.reset()
        logger.info(info_strings.STARTING_NEW_UPSCALE_CANCELLATION_RESET_STATUS)

        current_output_video_val = None
        current_status_text_val = ""
        current_user_prompt_val = app_config.prompts.user
        current_caption_status_text_val = ""
        current_caption_status_visible_val = False
        current_last_chunk_video_val = None
        current_chunk_status_text_val = "No chunks processed yet"
        current_comparison_video_val = None
        auto_caption_completed_successfully = False
        log_accumulator_director = []

        logger.info(f"In upscale_director_logic. Auto-caption first: {app_config.cogvlm.auto_caption_then_upscale}, User prompt: '{app_config.prompts.user[:50]}...'")

        actual_input_video_path = app_config.input_video_path

        if app_config.frame_folder.input_path and app_config.frame_folder.input_path.strip():
            from logic.frame_folder_utils import detect_input_type, validate_input_path
            input_type, validated_path, metadata = detect_input_type(app_config.frame_folder.input_path, logger)
            if input_type == "video_file":
                logger.info(f"Enhanced input detected video file: {validated_path}")
                actual_input_video_path = validated_path
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
            current_status_text_val = "\n".join(log_accumulator_director)
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
            log_accumulator_director = []

        should_auto_caption_entire_video = (app_config.cogvlm.auto_caption_then_upscale and
                                            not app_config.scene_split.enable and
                                            not app_config.image_upscaler.enable and
                                            UTIL_COG_VLM_AVAILABLE)

        if should_auto_caption_entire_video:
            if "cancelled" in current_user_prompt_val.lower() or "error" in current_user_prompt_val.lower():
                logger.info(info_strings.SKIPPING_AUTO_CAPTION_ERROR_CANCELLED_PROMPT)
                current_user_prompt_val = "..."
                app_config.prompts.user = current_user_prompt_val
            logger.info(info_strings.AUTO_CAPTIONING_ENTIRE_VIDEO_SCENE_DISABLED)
            progress(0, desc="Starting auto-captioning before upscale...")
            current_status_text_val = "Starting auto-captioning..."
            if UTIL_COG_VLM_AVAILABLE:
                current_caption_status_text_val = "Starting auto-captioning..."
                current_caption_status_visible_val = True
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
            try:
                cancellation_manager.check_cancel()
                caption_text, caption_stat_msg = util_auto_caption(
                    app_config.input_video_path, app_config.cogvlm.quant_value, app_config.cogvlm.unload_after_use,
                    app_config.paths.cogvlm_model_path, logger=logger, progress=progress
                )
                cancellation_manager.check_cancel()
                log_accumulator_director.append(f"Auto-caption status: {caption_stat_msg}")
                if not caption_text.startswith("Error:") and not caption_text.startswith("Caption generation cancelled"):
                    current_user_prompt_val = caption_text
                    app_config.prompts.user = caption_text
                    auto_caption_completed_successfully = True
                    log_accumulator_director.append(f"Using generated caption as prompt: '{caption_text[:50]}...'")
                    logger.info(f"Auto-caption successful. Updated current_user_prompt_val to: '{current_user_prompt_val[:100]}...'")
                else:
                    log_accumulator_director.append(info_strings.CAPTION_GENERATION_FAILED_ORIGINAL_PROMPT)
                    logger.warning(f"Auto-caption failed or cancelled. Keeping original prompt: '{current_user_prompt_val[:100]}...'")
                current_status_text_val = "\n".join(log_accumulator_director)
                if UTIL_COG_VLM_AVAILABLE:
                    current_caption_status_text_val = caption_stat_msg
                logger.info(f"About to yield auto-caption result. current_user_prompt_val: '{current_user_prompt_val[:100]}...'")
                yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                       gr.update(value=current_user_prompt_val),
                       gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                       gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                       gr.update(value=current_comparison_video_val))
                logger.info("Auto-caption yield completed.")
                if UTIL_COG_VLM_AVAILABLE:
                    current_caption_status_visible_val = False
            except CancelledError as e_cancel:
                logger.info(info_strings.AUTO_CAPTIONING_CANCELLED_USER_STOPPING)
                log_accumulator_director.append(info_strings.PROCESS_CANCELLED_USER_AUTO_CAPTIONING)
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
            except Exception as e_ac:
                logger.error(f"Exception during auto-caption call or its setup: {e_ac}", exc_info=True)
                log_accumulator_director.append(f"Error during auto-caption pre-step: {e_ac}")
                current_status_text_val = "\n".join(log_accumulator_director)
                if UTIL_COG_VLM_AVAILABLE:
                    current_caption_status_text_val = str(e_ac)
                    yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                           gr.update(value=current_user_prompt_val),
                           gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                           gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                           gr.update(value=current_comparison_video_val))
                    current_caption_status_visible_val = False
                else:
                    yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                           gr.update(value=current_user_prompt_val),
                           gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                           gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                           gr.update(value=current_comparison_video_val))
            log_accumulator_director = []
        elif app_config.cogvlm.auto_caption_then_upscale and app_config.scene_split.enable and not app_config.image_upscaler.enable and UTIL_COG_VLM_AVAILABLE:
            msg = info_strings.SCENE_SPLITTING_AUTO_CAPTIONING_PER_SCENE
            logger.info(msg)
            log_accumulator_director.append(msg)
            current_status_text_val = "\n".join(log_accumulator_director)
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
        elif app_config.cogvlm.auto_caption_then_upscale and app_config.image_upscaler.enable:
            msg = info_strings.IMAGE_UPSCALING_AUTO_CAPTIONING_DISABLED
            logger.info(msg)
            log_accumulator_director.append(msg)
            current_status_text_val = "\n".join(log_accumulator_director)
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
        elif app_config.cogvlm.auto_caption_then_upscale and not UTIL_COG_VLM_AVAILABLE:
            msg = info_strings.AUTO_CAPTIONING_COGVLM2_UNAVAILABLE_ORIGINAL
            logger.warning(msg)

        try:
            cancellation_manager.check_cancel()
        except CancelledError:
            logger.info(info_strings.PROCESS_CANCELLED_MAIN_UPSCALING_STOPPING)
            log_accumulator_director.append("⚠️ Process cancelled by user")
            current_status_text_val = "\n".join(log_accumulator_director)
            yield (gr.update(value=current_output_video_val), gr.update(value=current_status_text_val),
                   gr.update(value=current_user_prompt_val),
                   gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                   gr.update(value=current_last_chunk_video_val), gr.update(value=current_chunk_status_text_val),
                   gr.update(value=current_comparison_video_val))
            return

        actual_seed_to_use = app_config.seed.seed
        if app_config.seed.use_random:
            actual_seed_to_use = np.random.randint(0, 2**31)
            logger.info(f"Random seed checkbox is checked. Using generated seed: {actual_seed_to_use}")
        elif app_config.seed.seed == -1:
            actual_seed_to_use = np.random.randint(0, 2**31)
            logger.info(f"Seed input is -1. Using generated seed: {actual_seed_to_use}")
        else:
            logger.info(f"Using provided seed: {actual_seed_to_use}")
        app_config.seed.seed = actual_seed_to_use

        try:
            upscale_generator = core_run_upscale(
                input_video_path=actual_input_video_path, user_prompt=app_config.prompts.user,
                positive_prompt=app_config.prompts.positive, negative_prompt=app_config.prompts.negative, model_choice=app_config.star_model.model_choice,
                upscale_factor_slider=app_config.resolution.upscale_factor, cfg_scale=app_config.star_model.cfg_scale, steps=app_config.star_model.steps, solver_mode=app_config.star_model.solver_mode,
                max_chunk_len=app_config.performance.max_chunk_len, enable_chunk_optimization=app_config.performance.enable_chunk_optimization, vae_chunk=app_config.performance.vae_chunk, enable_vram_optimization=app_config.performance.enable_vram_optimization, color_fix_method=app_config.star_model.color_fix_method,
                enable_tiling=app_config.tiling.enable, tile_size=app_config.tiling.tile_size, tile_overlap=app_config.tiling.tile_overlap,
                enable_context_window=app_config.context_window.enable, context_overlap=app_config.context_window.overlap,
                enable_target_res=app_config.resolution.enable_target_res, target_h=app_config.resolution.target_h, target_w=app_config.resolution.target_w, target_res_mode=app_config.resolution.target_res_mode,
                ffmpeg_preset=app_config.ffmpeg.preset, ffmpeg_quality_value=app_config.ffmpeg.quality, ffmpeg_use_gpu=app_config.ffmpeg.use_gpu,
                save_frames=app_config.outputs.save_frames, save_metadata=app_config.outputs.save_metadata, save_chunks=app_config.outputs.save_chunks, save_chunk_frames=app_config.outputs.save_chunk_frames,
                enable_scene_split=app_config.scene_split.enable, scene_split_mode=app_config.scene_split.mode, scene_min_scene_len=app_config.scene_split.min_scene_len, scene_drop_short=app_config.scene_split.drop_short, scene_merge_last=app_config.scene_split.merge_last,
                scene_frame_skip=app_config.scene_split.frame_skip, scene_threshold=app_config.scene_split.threshold, scene_min_content_val=app_config.scene_split.min_content_val, scene_frame_window=app_config.scene_split.frame_window,
                scene_copy_streams=app_config.scene_split.copy_streams, scene_use_mkvmerge=app_config.scene_split.use_mkvmerge, scene_rate_factor=app_config.scene_split.rate_factor, scene_preset=app_config.scene_split.encoding_preset, scene_quiet_ffmpeg=app_config.scene_split.quiet_ffmpeg,
                scene_manual_split_type=app_config.scene_split.manual_split_type, scene_manual_split_value=app_config.scene_split.manual_split_value,
                create_comparison_video_enabled=app_config.outputs.create_comparison_video,
                enable_fps_decrease=app_config.fps_decrease.enable, fps_decrease_mode=app_config.fps_decrease.mode,
                fps_multiplier_preset=app_config.fps_decrease.multiplier_preset, fps_multiplier_custom=app_config.fps_decrease.multiplier_custom,
                target_fps=app_config.fps_decrease.target_fps, fps_interpolation_method=app_config.fps_decrease.interpolation_method,
                enable_rife_interpolation=app_config.rife.enable, rife_multiplier=app_config.rife.multiplier, rife_fp16=app_config.rife.fp16, rife_uhd=app_config.rife.uhd, rife_scale=app_config.rife.scale,
                rife_skip_static=app_config.rife.skip_static, rife_enable_fps_limit=app_config.rife.enable_fps_limit, rife_max_fps_limit=app_config.rife.max_fps_limit,
                rife_apply_to_chunks=app_config.rife.apply_to_chunks, rife_apply_to_scenes=app_config.rife.apply_to_scenes, rife_keep_original=app_config.rife.keep_original, rife_overwrite_original=app_config.rife.overwrite_original,
                is_batch_mode=False, batch_output_dir=None, original_filename=None,
                enable_auto_caption_per_scene=(app_config.cogvlm.auto_caption_then_upscale and app_config.scene_split.enable and not app_config.image_upscaler.enable and UTIL_COG_VLM_AVAILABLE),
                cogvlm_quant=app_config.cogvlm.quant_value,
                cogvlm_unload=app_config.cogvlm.unload_after_use if app_config.cogvlm.unload_after_use else 'full',
                logger=logger,
                app_config_module=app_config_module,
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
                enable_image_upscaler=app_config.image_upscaler.enable,
                image_upscaler_model=app_config.image_upscaler.model,
                image_upscaler_batch_size=app_config.image_upscaler.batch_size,
                enable_face_restoration=app_config.face_restoration.enable,
                face_restoration_fidelity=app_config.face_restoration.fidelity_weight,
                enable_face_colorization=app_config.face_restoration.enable_colorization,
                face_restoration_timing="after_upscale",
                face_restoration_when=app_config.face_restoration.when,
                codeformer_model=app_config.face_restoration.model,
                face_restoration_batch_size=app_config.face_restoration.batch_size,
                enable_seedvr2=app_config.seedvr2.enable,
                seedvr2_config=app_config.seedvr2
            )
            cancellation_detected = False
            for yielded_output_video, yielded_status_log, yielded_chunk_video, yielded_chunk_status, yielded_comparison_video in upscale_generator:
                # Debug log what we're receiving
                if yielded_status_log and "🎬 Batch" in yielded_status_log:
                    # Remove or comment out this duplicate logging
                    # logger.info(f"UI received batch progress: {yielded_status_log[:100]}...")
                    pass  # Do nothing, just continue
                elif yielded_chunk_video:
                    logger.info(f"UI received chunk video update: {yielded_chunk_video}")
                is_partial_video = yielded_output_video and "partial_cancelled" in yielded_output_video
                if cancellation_detected and current_output_video_val and "partial_cancelled" in current_output_video_val:
                    if not is_partial_video:
                        logger.info(f"Skipping output video update to preserve partial video: {os.path.basename(current_output_video_val)}")
                        output_video_update = gr.update(value=current_output_video_val)
                    else:
                        current_output_video_val = yielded_output_video
                        output_video_update = gr.update(value=current_output_video_val)
                        logger.info(f"Updated partial video from generator: {os.path.basename(yielded_output_video)}")
                else:
                    output_video_update = gr.update()
                    if yielded_output_video is not None:
                        current_output_video_val = yielded_output_video
                        output_video_update = gr.update(value=current_output_video_val)
                        logger.info(f"Updated output video from generator: {os.path.basename(yielded_output_video) if yielded_output_video else 'None'}")
                        if is_partial_video:
                            cancellation_detected = True
                            logger.info("Detected partial video - cancellation mode activated")
                    elif current_output_video_val is None:
                        output_video_update = gr.update(value=None)
                    else:
                        output_video_update = gr.update(value=current_output_video_val)
                # Handle status log updates
                if yielded_status_log:
                    # Check if this is a SeedVR2 batch progress or important update
                    if any(marker in yielded_status_log for marker in ["🎬 Batch", "⏳ Processing...", "Chunk", "ETA:"]):
                        # This is an important progress update - show it directly
                        current_status_text_val = yielded_status_log
                        # Also log it for debugging
                        if "🎬 Batch" in yielded_status_log:
                            logger.info(f"Displaying batch progress in UI: {yielded_status_log}")
                    else:
                        # For other messages, append to existing status
                        if current_status_text_val and not current_status_text_val.endswith('\n'):
                            current_status_text_val += '\n'
                        current_status_text_val += yielded_status_log
                        # Keep only last 15 lines
                        status_lines = current_status_text_val.split('\n')
                        if len(status_lines) > 15:
                            current_status_text_val = '\n'.join(status_lines[-15:])
                else:
                    # No new status, keep current
                    pass
                status_text_update = gr.update(value=current_status_text_val)
                if yielded_status_log and "[FIRST_SCENE_CAPTION:" in yielded_status_log and not auto_caption_completed_successfully:
                    try:
                        caption_start = yielded_status_log.find("[FIRST_SCENE_CAPTION:") + len("[FIRST_SCENE_CAPTION:")
                        caption_end = yielded_status_log.find("]", caption_start)
                        if caption_start > len("[FIRST_SCENE_CAPTION:") and caption_end > caption_start:
                            extracted_caption = yielded_status_log[caption_start:caption_end]
                            current_user_prompt_val = extracted_caption
                            auto_caption_completed_successfully = True
                            logger.info(f"Updated main prompt from first scene caption: '{extracted_caption[:100]}...'")
                            log_accumulator_director.append(f"Main prompt updated with first scene caption: '{extracted_caption[:50]}...'")
                            current_status_text_val = (combined_log_director + "\n" + "\n".join(log_accumulator_director)).strip()
                            status_text_update = gr.update(value=current_status_text_val)
                    except Exception as e:
                        logger.error(f"Error extracting first scene caption: {e}")
                elif yielded_status_log and "FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE:" in yielded_status_log and not auto_caption_completed_successfully:
                    try:
                        caption_start = yielded_status_log.find("FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE:") + len("FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE:")
                        extracted_caption = yielded_status_log[caption_start:].strip()
                        if extracted_caption:
                            current_user_prompt_val = extracted_caption
                            auto_caption_completed_successfully = True
                            logger.info(f"Updated main prompt from immediate first scene caption: '{extracted_caption[:100]}...'")
                            log_accumulator_director.append(f"Main prompt updated with first scene caption: '{extracted_caption[:50]}...'")
                            current_status_text_val = (combined_log_director + "\n" + "\n".join(log_accumulator_director)).strip()
                            status_text_update = gr.update(value=current_status_text_val)
                    except Exception as e:
                        logger.error(f"Error extracting immediate first scene caption: {e}")
                user_prompt_update = gr.update(value=current_user_prompt_val)
                caption_status_update = gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val)
                chunk_video_update = gr.update()
                if yielded_chunk_video is not None:
                    current_last_chunk_video_val = yielded_chunk_video
                    chunk_video_update = gr.update(value=current_last_chunk_video_val)
                    logger.info(f"Updating chunk video in UI to: {current_last_chunk_video_val}")
                elif current_last_chunk_video_val is None:
                    chunk_video_update = gr.update(value=None)
                else:
                    chunk_video_update = gr.update(value=current_last_chunk_video_val)
                if yielded_chunk_status is not None:
                    current_chunk_status_text_val = yielded_chunk_status
                    if "Batch" in yielded_chunk_status:
                        logger.info(f"Updating chunk status text to: {yielded_chunk_status}")
                chunk_status_text_update = gr.update(value=current_chunk_status_text_val)
                comparison_video_update = gr.update()
                if yielded_comparison_video is not None:
                    current_comparison_video_val = yielded_comparison_video
                    comparison_video_update = gr.update(value=current_comparison_video_val)
                elif current_comparison_video_val is None:
                    comparison_video_update = gr.update(value=None)
                else:
                    comparison_video_update = gr.update(value=current_comparison_video_val)
                yield (
                    output_video_update, status_text_update, user_prompt_update,
                    caption_status_update,
                    chunk_video_update, chunk_status_text_update,
                    comparison_video_update
                )
            logger.info(f"Final yield: current_user_prompt_val = '{current_user_prompt_val[:100]}...', auto_caption_completed = {auto_caption_completed_successfully}")
            yield (
                gr.update(value=current_output_video_val),
                gr.update(value=current_status_text_val),
                gr.update(value=current_user_prompt_val),
                gr.update(value=current_caption_status_text_val, visible=current_caption_status_visible_val),
                gr.update(value=current_last_chunk_video_val),
                gr.update(value=current_chunk_status_text_val),
                gr.update(value=current_comparison_video_val)
            )
        except CancelledError:
            logger.warning("Processing was cancelled by user.")
            logger.info(f"Current output video value at cancellation: {current_output_video_val}")
            if current_output_video_val and "partial_cancelled" in current_output_video_val:
                logger.info(f"Partial video already set from generator: {current_output_video_val}")
                current_status_text_val = f"⚠️ Processing cancelled by user. Partial video saved: {os.path.basename(current_output_video_val)}"
            else:
                partial_video_found = False
                import glob
                partial_pattern = os.path.join(APP_CONFIG.paths.outputs_dir, "*_partial_cancelled.mp4")
                partial_files = glob.glob(partial_pattern)
                logger.info(f"Searching for partial videos with pattern: {partial_pattern}")
                logger.info(f"Found partial files: {partial_files}")
                if partial_files:
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
            cancellation_manager.reset()
            logger.info("Processing completed - UI state reset")

    # Define the list of inputs for the main upscale and batch buttons
    click_inputs = [input_video] + component_order

    click_outputs_list = [output_video, status_textbox, user_prompt]
    if UTIL_COG_VLM_AVAILABLE:
        click_outputs_list.append(caption_status)
    else:
        click_outputs_list.append(gr.State(None))
    click_outputs_list.extend([last_chunk_video, chunk_status_text, comparison_video, cancel_button])

    def upscale_wrapper(*args):
        last_valid_result = None
        try:
            app_config = build_app_config_from_ui(*args)
            processing_started = False
            for result in upscale_director_logic(app_config):
                last_valid_result = result
                if not processing_started:
                    processing_started = True
                    yield result + (gr.update(interactive=True),)
                else:
                    yield result + (gr.update(interactive=True),)
        except Exception as e:
            logger.error(f"Error in upscale_wrapper: {e}", exc_info=True)
            if last_valid_result:
                yield last_valid_result + (gr.update(interactive=False),)
            else:
                yield (gr.update(), gr.update(value=f"Error: {e}"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False))
        finally:
            if not last_valid_result:
                yield (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False))

    upscale_button.click(
        fn=upscale_wrapper,
        inputs=click_inputs,
        outputs=click_outputs_list,
        show_progress_on=[output_video]
    )

    def cancel_processing():
        logger.warning(info_strings.CANCEL_BUTTON_CLICKED_USER_REQUESTING)
        success = cancellation_manager.request_cancellation()
        if success:
            return info_strings.CANCELLATION_REQUESTED_STOP_SAFELY_CHECKPOINT
        else:
            return info_strings.NO_ACTIVE_PROCESSING_CANCEL_REQUESTED

    cancel_button.click(
        fn=cancel_processing,
        inputs=[],
        outputs=[status_textbox]
    )

    if UTIL_COG_VLM_AVAILABLE:
        def auto_caption_wrapper(vid, quant_display, unload_strat, progress=gr.Progress(track_tqdm=True)):
            caption_text, caption_stat_msg = util_auto_caption(
                vid,
                get_quant_value_from_display(quant_display),
                unload_strat,
                APP_CONFIG.paths.cogvlm_model_path,
                logger=logger,
                progress=progress
            )
            return caption_text, caption_stat_msg

        def rife_fps_increase_wrapper(
            input_video_val,
            rife_multiplier_val,
            rife_fp16_val,
            rife_uhd_val,
            rife_scale_val,
            rife_skip_static_val,
            rife_enable_fps_limit_val,
            rife_max_fps_limit_val,
            ffmpeg_preset_dropdown_val,
            ffmpeg_quality_slider_val,
            ffmpeg_use_gpu_check_val,
            seed_num_val,
            random_seed_check_val,
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
                output_dir=APP_CONFIG.paths.outputs_dir,
                logger=logger,
                progress=progress
            )

        auto_caption_btn.click(
            fn=auto_caption_wrapper,
            inputs=[input_video, cogvlm_quant_radio, cogvlm_unload_radio],
            outputs=[user_prompt, caption_status],
            show_progress_on=[user_prompt]
        ).then(lambda: gr.update(visible=True), None, caption_status)

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

    def process_batch_videos_wrapper(app_config: AppConfig, progress=gr.Progress(track_tqdm=True)):
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

        partial_run_upscale_for_batch = partial(core_run_upscale,
                                                logger=logger,
                                                app_config_module=app_config_module,
                                                metadata_handler_module=metadata_handler,
                                                VideoToVideo_sr_class=VideoToVideo_sr,
                                                setup_seed_func=setup_seed,
                                                EasyDict_class=EasyDict,
                                                preprocess_func=preprocess,
                                                collate_fn_func=collate_fn,
                                                tensor2vid_func=tensor2vid,
                                                ImageSpliterTh_class=ImageSpliterTh,
                                                adain_color_fix_func=adain_color_fix,
                                                wavelet_color_fix_func=wavelet_color_fix
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
                logger.info(f"Converting frame folder {i + 1}/{len(frame_folders)}: {folder_name}")
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
                return None, "❌ Failed to convert any frame folders to videos"
            logger.info(f"Successfully converted {len(temp_video_conversions)} frame folders to videos")
            app_config.batch.input_folder = actual_batch_input_folder

        from logic.batch_operations import process_batch_videos_from_app_config
        return process_batch_videos_from_app_config(
            app_config=app_config,
            run_upscale_func=partial_run_upscale_for_batch,
            logger=logger,
            progress=progress
        )

    def batch_wrapper(*args):
        app_config = build_app_config_from_ui(*args)
        result = process_batch_videos_wrapper(app_config)
        return result

    batch_process_button.click(
        fn=batch_wrapper,
        inputs=click_inputs,
        outputs=[output_video, status_textbox],
        show_progress_on=[output_video]
    )

    gpu_selector.change(
        fn=lambda gpu_id: util_set_gpu_device(gpu_id, logger=logger),
        inputs=gpu_selector,
        outputs=status_textbox
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
        seed_num_val,
        random_seed_check_val,
        ffmpeg_preset_val,
        ffmpeg_quality_val,
        ffmpeg_use_gpu_val,
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
                actual_seed = seed_num_val

            if not enable_face_restoration_val:
                return None, None, info_strings.FACE_RESTORATION_DISABLED_ENABLE_PROCESS, "❌ Processing disabled"

            if face_restoration_mode_val == "Single Video":
                if not input_video_val:
                    return None, None, "⚠️ Please upload a video file to process.", "❌ No input video"
                input_path = input_video_val
                output_dir = APP_CONFIG.paths.outputs_dir
                processing_mode = "single"
            else:
                if not batch_input_folder_val or not batch_output_folder_val:
                    return None, None, info_strings.BATCH_PROCESSING_SPECIFY_FOLDER_PATHS, "❌ Missing folder paths"
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
                    mapped_progress = DEFAULT_BATCH_PROGRESS_OFFSET + (current_progress * DEFAULT_BATCH_PROGRESS_SCALE)
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
                    ffmpeg_preset=ffmpeg_preset_val,
                    ffmpeg_quality=ffmpeg_quality_val,
                    ffmpeg_use_gpu=ffmpeg_use_gpu_val,
                    progress_callback=progress_callback,
                    logger=logger
                )
                if result['success']:
                    output_video = result['output_video_path']
                    comparison_video = result.get('comparison_video_path', None)
                    processing_time = time.time() - start_time
                    session_info = ""
                    if 'session_name' in result and 'session_folder' in result:
                        session_info = f"\n📂 Session Folder: {result['session_name']}"
                        if save_frames_val:
                            session_info += f"\n🖼️ Extracted Frames: {result['session_name']}/extracted_frames/"
                            session_info += f"\n🎨 Processed Frames: {result['session_name']}/processed_frames/"
                    stats_msg = PROCESSING_COMPLETE_TEMPLATE.format(
                        total_time=processing_time,
                        frames_processed=f"Input: {os.path.basename(input_path)}, Output: {os.path.basename(output_video) if output_video else 'N/A'}, Fidelity: {face_restoration_fidelity_val}, Batch Size: {face_restoration_batch_size_val}{session_info}",
                        output_filename=os.path.basename(output_video) if output_video else 'N/A'
                    )
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
                    base_progress = DEFAULT_BATCH_PROGRESS_OFFSET + (file_progress * DEFAULT_BATCH_PROGRESS_SCALE)
                    progress(base_progress, f"🎭 Processing {video_name} ({i + 1}/{total_files})...")
                    def batch_progress_callback(current_progress, status_msg):
                        file_progress_range = DEFAULT_BATCH_PROGRESS_SCALE / total_files
                        mapped_progress = base_progress + (current_progress * file_progress_range)
                        progress(mapped_progress, f"🎭 [{i + 1}/{total_files}] {video_name}: {status_msg}")
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
                        ffmpeg_preset=ffmpeg_preset_val,
                        ffmpeg_quality=ffmpeg_quality_val,
                        ffmpeg_use_gpu=ffmpeg_use_gpu_val,
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
                stats_msg = BATCH_PROCESSING_COMPLETE_TEMPLATE.format(
                    total_time=processing_time,
                    videos_processed=f"Processed: {processed_count}, Failed: {failed_count}",
                    output_folder=os.path.basename(output_dir),
                    face_restoration_batch_size_val=face_restoration_batch_size_val
                )
                if processed_count > 0:
                    progress(1.0, f"✅ Batch processing completed! {processed_count}/{total_files} videos processed successfully.")
                    status_msg = f"✅ Batch processing completed!\n\n📊 Results:\n✅ Successfully processed: {processed_count} videos\n❌ Failed: {failed_count} videos\n📁 Output saved to: {output_dir}\n\n⏱️ Total processing time: {processing_time:.1f} seconds"
                    return None, None, status_msg, stats_msg
                else:
                    return None, None, info_strings.BATCH_PROCESSING_FAILED_NO_VIDEOS, "❌ Batch processing failed"
        except Exception as e:
            logger.error(f"Standalone face restoration error: {str(e)}")
            return None, None, f"❌ Error during face restoration: {str(e)}", "❌ Processing error"

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
            random_seed_check,
            ffmpeg_preset_dropdown,
            ffmpeg_quality_slider,
            ffmpeg_use_gpu_check
        ],
        outputs=[
            output_video_face_restoration,
            comparison_video_face_restoration,
            face_restoration_status,
            face_restoration_stats
        ],
        show_progress_on=[output_video_face_restoration]
    )

    def update_multi_video_ui(video_count):
        layout_choices_map = {
            2: ["auto", "side_by_side", "top_bottom"],
            3: ["auto", "3x1_horizontal", "1x3_vertical", "L_shape"],
            4: ["auto", "2x2_grid", "4x1_horizontal", "1x4_vertical"]
        }
        layout_info_map = {
            2: info_strings.COMPARISON_AUTO_SIDE_BY_SIDE_TOP_BOTTOM_INFO,
            3: info_strings.COMPARISON_AUTO_3X1_1X3_L_SHAPE_INFO,
            4: info_strings.COMPARISON_AUTO_2X2_4X1_1X4_INFO
        }
        choices = layout_choices_map.get(video_count, layout_choices_map[2])
        info = layout_info_map.get(video_count, layout_info_map[2])
        layout_update = gr.update(choices=choices, value="auto", info=info)
        if video_count >= DEFAULT_VIDEO_COUNT_THRESHOLD_3:
            third_video_update = gr.update(visible=True)
        else:
            third_video_update = gr.update(visible=False)
        if video_count >= DEFAULT_VIDEO_COUNT_THRESHOLD_4:
            fourth_video_update = gr.update(visible=True)
        else:
            fourth_video_update = gr.update(visible=False)
        return layout_update, third_video_update, fourth_video_update

    def manual_comparison_wrapper(
        manual_video_count_val,
        manual_original_video_val,
        manual_upscaled_video_val,
        manual_third_video_val,
        manual_fourth_video_val,
        manual_comparison_layout_val,
        ffmpeg_preset_dropdown_val,
        ffmpeg_quality_slider_val,
        ffmpeg_use_gpu_check_val,
        seed_num_val,
        random_seed_check_val,
        progress=gr.Progress(track_tqdm=True)
    ):
        if random_seed_check_val:
            import random
            current_seed = random.randint(0, 2**32 - 1)
        else:
            current_seed = seed_num_val if seed_num_val >= 0 else -1
        video_paths = [manual_original_video_val, manual_upscaled_video_val]
        if manual_video_count_val >= DEFAULT_VIDEO_COUNT_THRESHOLD_3:
            video_paths.append(manual_third_video_val)
        if manual_video_count_val >= DEFAULT_VIDEO_COUNT_THRESHOLD_4:
            video_paths.append(manual_fourth_video_val)
        valid_videos = [path for path in video_paths if path is not None]
        if len(valid_videos) < 2:
            error_msg = "Please upload at least 2 videos for comparison"
            return gr.update(value=None), gr.update(value=error_msg, visible=True)
        if len(valid_videos) < manual_video_count_val:
            error_msg = f"Please upload all {manual_video_count_val} videos as selected"
            return gr.update(value=None), gr.update(value=error_msg, visible=True)
        try:
            if len(valid_videos) == 2 and manual_comparison_layout_val in ["auto", "side_by_side", "top_bottom"]:
                output_path, status_message = util_generate_manual_comparison_video(
                    original_video_path=valid_videos[0],
                    upscaled_video_path=valid_videos[1],
                    ffmpeg_preset=ffmpeg_preset_dropdown_val,
                    ffmpeg_quality=ffmpeg_quality_slider_val,
                    ffmpeg_use_gpu=ffmpeg_use_gpu_check_val,
                    output_dir=APP_CONFIG.paths.outputs_dir,
                    comparison_layout=manual_comparison_layout_val,
                    seed_value=current_seed,
                    logger=logger,
                    progress=progress
                )
            else:
                output_path, status_message = util_generate_multi_video_comparison(
                    video_paths=valid_videos,
                    ffmpeg_preset=ffmpeg_preset_dropdown_val,
                    ffmpeg_quality=ffmpeg_quality_slider_val,
                    ffmpeg_use_gpu=ffmpeg_use_gpu_check_val,
                    output_dir=APP_CONFIG.paths.outputs_dir,
                    comparison_layout=manual_comparison_layout_val,
                    seed_value=current_seed,
                    logger=logger,
                    progress=progress
                )
            if output_path:
                return gr.update(value=output_path), gr.update(value=status_message, visible=True)
            else:
                return gr.update(value=None), gr.update(value=status_message, visible=True)
        except Exception as e:
            error_msg = f"Unexpected error during multi-video comparison: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return gr.update(value=None), gr.update(value=error_msg, visible=True)

    def update_seed_num_interactive(is_random_seed_checked):
        return gr.update(interactive=not is_random_seed_checked)

    random_seed_check.change(
        fn=update_seed_num_interactive,
        inputs=random_seed_check,
        outputs=seed_num
    )

    manual_video_count.change(
        fn=update_multi_video_ui,
        inputs=[manual_video_count],
        outputs=[manual_comparison_layout, manual_third_video, manual_fourth_video]
    )

    manual_comparison_button.click(
        fn=manual_comparison_wrapper,
        inputs=[
            manual_video_count,
            manual_original_video,
            manual_upscaled_video,
            manual_third_video,
            manual_fourth_video,
            manual_comparison_layout,
            ffmpeg_preset_dropdown,
            ffmpeg_quality_slider,
            ffmpeg_use_gpu_check,
            seed_num,
            random_seed_check
        ],
        outputs=[
            comparison_video,
            manual_comparison_status
        ],
        show_progress_on=[comparison_video]
    )

    def update_rife_fps_limit_interactive(enable_fps_limit):
        return gr.update(interactive=enable_fps_limit)

    def update_rife_controls_interactive(enable_rife):
        return [gr.update(interactive=enable_rife)] * 10

    rife_enable_fps_limit.change(
        fn=update_rife_fps_limit_interactive,
        inputs=rife_enable_fps_limit,
        outputs=rife_max_fps_limit
    )

    enable_rife_interpolation.change(
        fn=update_rife_controls_interactive,
        inputs=enable_rife_interpolation,
        outputs=[
            rife_multiplier, rife_fp16, rife_uhd, rife_scale, rife_skip_static,
            rife_enable_fps_limit, rife_max_fps_limit, rife_apply_to_chunks,
            rife_apply_to_scenes, rife_keep_original
        ]
    )

    def update_fps_decrease_controls_interactive(enable_fps_decrease):
        if enable_fps_decrease:
            return [
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            ]
        else:
            return [
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

    def update_fps_mode_controls(fps_mode):
        if fps_mode == "multiplier":
            return [
                gr.update(visible=True),
                gr.update(visible=False),
            ]
        else:
            return [
                gr.update(visible=False),
                gr.update(visible=True),
            ]

    def update_multiplier_preset(preset_choice):
        multiplier_map = {v: k for k, v in util_get_common_fps_multipliers().items()}
        if preset_choice == "Custom":
            return [
                gr.update(visible=True),
                0.5
            ]
        else:
            multiplier_value = multiplier_map.get(preset_choice, 0.5)
            return [
                gr.update(visible=False),
                multiplier_value
            ]

    def calculate_fps_preview(input_video, fps_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps):
        if input_video is None:
            return "**📊 Calculation:** Upload a video to see FPS reduction preview"
        try:
            video_info = util_get_video_info_fast(input_video, logger)
            input_fps = video_info.get('fps', 30.0) if video_info else 30.0
            if fps_mode == "multiplier":
                if fps_multiplier_preset == "Custom":
                    multiplier = fps_multiplier_custom
                else:
                    multiplier_map = {v: k for k, v in util_get_common_fps_multipliers().items()}
                    multiplier = multiplier_map.get(fps_multiplier_preset, 0.5)
                calculated_fps = input_fps * multiplier
                if calculated_fps < 1.0:
                    calculated_fps = 1.0
                return f"**📊 Calculation:** {input_fps:.1f} FPS × {multiplier:.2f} = {calculated_fps:.1f} FPS ({fps_multiplier_preset})"
            else:
                return f"**📊 Calculation:** Fixed mode → {target_fps} FPS"
        except Exception as e:
            return f"**📊 Calculation:** Error calculating preview: {str(e)}"

    def calculate_auto_resolution(video_path, enable_auto_aspect_resolution, target_h, target_w):
        if not enable_auto_aspect_resolution:
            return target_h, target_w, "Auto-resolution disabled"
        if video_path is None:
            return target_h, target_w, "No video loaded"
        try:
            from logic.auto_resolution_utils import update_resolution_from_video
            constraint_width = target_w
            constraint_height = target_h
            result = update_resolution_from_video(
                video_path=video_path,
                constraint_width=constraint_width,
                constraint_height=constraint_height,
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

    def calculate_expected_output_resolution(
        video_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        upscale_factor,
        image_upscaler_model,
        enable_auto_aspect_resolution
    ):
        if video_path is None:
            return info_strings.UPLOAD_VIDEO_EXPECTED_OUTPUT_RESOLUTION
        try:
            video_info = util_get_video_info_fast(video_path, logger)
            if not video_info:
                return "❌ Could not read video information"
            orig_w = video_info.get('width', 0)
            orig_h = video_info.get('height', 0)
            if orig_w <= 0 or orig_h <= 0:
                return f"❌ Invalid video dimensions: {orig_w}x{orig_h}"
            upscaler_type_map = {
                "Use STAR Model Upscaler": "star",
                "Use Image Based Upscalers": "image_upscaler",
                "Use SeedVR2 Video Upscaler": "seedvr2"
            }
            internal_upscaler_type = upscaler_type_map.get(upscaler_type, "star")
            if internal_upscaler_type == "image_upscaler" and image_upscaler_model:
                try:
                    model_path = os.path.join(APP_CONFIG.paths.upscale_models_dir, image_upscaler_model)
                    if os.path.exists(model_path):
                        model_info = util_get_model_info(model_path, logger)
                        if "error" not in model_info:
                            effective_upscale_factor = float(model_info.get("scale", upscale_factor))
                            model_name = f"{image_upscaler_model} ({effective_upscale_factor}x)"
                        else:
                            effective_upscale_factor = upscale_factor
                            model_name = f"{image_upscaler_model} (scale unknown, using {upscale_factor}x)"
                    else:
                        effective_upscale_factor = upscale_factor
                        model_name = f"{image_upscaler_model} (not found, using {upscale_factor}x)"
                except Exception as e:
                    effective_upscale_factor = upscale_factor
                    model_name = f"{image_upscaler_model} (error reading scale, using {upscale_factor}x)"
            elif internal_upscaler_type == "star":
                effective_upscale_factor = app_config.resolution.upscale_factor
                model_name = "STAR Model (4x)"
            elif internal_upscaler_type == "seedvr2":
                from logic.star_dataclasses import DEFAULT_SEEDVR2_UPSCALE_FACTOR
                effective_upscale_factor = DEFAULT_SEEDVR2_UPSCALE_FACTOR
                model_name = "SeedVR2 (4x)"
            else:
                effective_upscale_factor = upscale_factor
                model_name = f"Unknown Upscaler ({upscale_factor}x)"
            if enable_target_res:
                try:
                    custom_upscale_factor = effective_upscale_factor if internal_upscaler_type == "seedvr2" else None
                    needs_downscale, ds_h, ds_w, upscale_factor_calc, final_h_calc, final_w_calc = util_calculate_upscale_params(
                        orig_h, orig_w, target_h, target_w, target_res_mode,
                        logger=logger,
                        image_upscaler_model=image_upscaler_model if internal_upscaler_type == "image_upscaler" else None,
                        custom_upscale_factor=custom_upscale_factor
                    )
                    final_h = final_h_calc
                    final_w = final_w_calc
                    actual_upscale_factor = upscale_factor_calc
                    info_lines = [
                        f"🎯 **Target Resolution Mode**: {target_res_mode}",
                        f"📹 **Input**: {orig_w}x{orig_h} ({orig_w * orig_h:,} pixels)",
                        f"🚀 **Upscaler**: {model_name}",
                        f"📐 **Output**: {final_w}x{final_h} ({final_w * final_h:,} pixels)",
                        f"📊 **Effective Scale**: {actual_upscale_factor:.2f}x"
                    ]
                    if needs_downscale:
                        info_lines.append(f"⬇️ **Downscale First**: {orig_w}x{orig_h} → {ds_w}x{ds_h}")
                    target_pixels = target_h * target_w
                    output_pixels = final_w * final_h
                    budget_usage = (output_pixels / target_pixels) * 100
                    info_lines.append(f"💾 **Pixel Budget**: {budget_usage:.1f}% of {target_w}x{target_h}")
                    return "\n".join(info_lines)
                except Exception as e:
                    return f"❌ Error calculating target resolution: {str(e)}"
            else:
                final_h = int(round(orig_h * effective_upscale_factor / 2) * 2)
                final_w = int(round(orig_w * effective_upscale_factor / 2) * 2)
                info_lines = [
                    f"🎯 **Mode**: Simple {effective_upscale_factor}x Upscale",
                    f"📹 **Input**: {orig_w}x{orig_h} ({orig_w * orig_h:,} pixels)",
                    f"🚀 **Upscaler**: {model_name}",
                    f"📐 **Output**: {final_w}x{final_h} ({final_w * final_h:,} pixels)",
                    f"📊 **Scale Factor**: {effective_upscale_factor}x"
                ]
                return "\n".join(info_lines)
        except Exception as e:
            logger.error(f"Error calculating expected output resolution: {e}")
            return f"❌ Error calculating resolution: {str(e)}"

    def handle_video_change_with_auto_resolution(
        video_path,
        enable_auto_aspect_resolution,
        target_h,
        target_w
    ):
        if video_path is None:
            return "", target_h, target_w, "No video loaded"
        try:
            video_info = util_get_video_info_fast(video_path, logger)
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
        new_target_h, new_target_w, auto_status = calculate_auto_resolution(
            video_path, enable_auto_aspect_resolution, target_h, target_w
        )
        return info_message, new_target_h, new_target_w, auto_status

    def calculate_compact_resolution_preview(
        video_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        upscale_factor,
        image_upscaler_model,
        enable_auto_aspect_resolution
    ):
        if video_path is None:
            return ""
        try:
            video_info = util_get_video_info_fast(video_path, logger)
            if not video_info:
                return ""
            orig_w = video_info.get('width', 0)
            orig_h = video_info.get('height', 0)
            if orig_w <= 0 or orig_h <= 0:
                return ""
            upscaler_type_map = {
                "Use STAR Model Upscaler": "star",
                "Use Image Based Upscalers": "image_upscaler",
                "Use SeedVR2 Video Upscaler": "seedvr2"
            }
            internal_upscaler_type = upscaler_type_map.get(upscaler_type, "star")
            if internal_upscaler_type == "image_upscaler" and image_upscaler_model:
                try:
                    model_path = os.path.join(APP_CONFIG.paths.upscale_models_dir, image_upscaler_model)
                    if os.path.exists(model_path):
                        model_info = util_get_model_info(model_path, logger)
                        if "error" not in model_info:
                            effective_upscale_factor = float(model_info.get("scale", upscale_factor))
                            model_name = f"{image_upscaler_model.split('.')[0]} ({effective_upscale_factor}x)"
                        else:
                            effective_upscale_factor = upscale_factor
                            model_name = f"{image_upscaler_model.split('.')[0]} ({upscale_factor}x)"
                    else:
                        effective_upscale_factor = upscale_factor
                        model_name = f"Image Upscaler ({upscale_factor}x)"
                except Exception:
                    effective_upscale_factor = upscale_factor
                    model_name = f"Image Upscaler ({upscale_factor}x)"
            elif internal_upscaler_type == "star":
                effective_upscale_factor = upscale_factor
                model_name = "STAR Model (4x)"
            elif internal_upscaler_type == "seedvr2":
                from logic.star_dataclasses import DEFAULT_SEEDVR2_UPSCALE_FACTOR
                effective_upscale_factor = DEFAULT_SEEDVR2_UPSCALE_FACTOR
                model_name = "SeedVR2 (4x)"
            else:
                effective_upscale_factor = upscale_factor
                model_name = f"Upscaler ({upscale_factor}x)"
            if enable_target_res:
                try:
                    custom_upscale_factor = effective_upscale_factor if internal_upscaler_type == "seedvr2" else None
                    needs_downscale, ds_h, ds_w, upscale_factor_calc, final_h_calc, final_w_calc = util_calculate_upscale_params(
                        orig_h, orig_w, target_h, target_w, target_res_mode,
                        logger=logger,
                        image_upscaler_model=image_upscaler_model if internal_upscaler_type == "image_upscaler" else None,
                        custom_upscale_factor=custom_upscale_factor
                    )
                    final_h = final_h_calc
                    final_w = final_w_calc
                    actual_upscale_factor = upscale_factor_calc
                    downscale_info = f" (⬇️ {ds_w}x{ds_h} first)" if needs_downscale else ""
                    return f"🎯 **Expected Output with {model_name}:**\n• Output Resolution: {final_w}x{final_h} ({actual_upscale_factor:.2f}x){downscale_info}\n• Target Mode: {target_res_mode}"
                except Exception as e:
                    return f"🎯 **Expected Output:** Error calculating ({str(e)})"
            else:
                final_h = int(round(orig_h * effective_upscale_factor / 2) * 2)
                final_w = int(round(orig_w * effective_upscale_factor / 2) * 2)
                return f"🎯 **Expected Output with {model_name}:**\n• Output Resolution: {final_w}x{final_h} ({effective_upscale_factor}x scale)"
        except Exception as e:
            return f"🎯 **Expected Output:** Error calculating ({str(e)})"

    def update_resolution_preview_wrapper(
        video_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        upscale_factor,
        image_upscaler_model,
        enable_auto_aspect_resolution
    ):
        return calculate_expected_output_resolution(
            video_path=video_path,
            upscaler_type=upscaler_type,
            enable_target_res=enable_target_res,
            target_h=target_h,
            target_w=target_w,
            target_res_mode=target_res_mode,
            upscale_factor=upscale_factor,
            image_upscaler_model=image_upscaler_model,
            enable_auto_aspect_resolution=enable_auto_aspect_resolution
        )

    enable_fps_decrease.change(
        fn=update_fps_decrease_controls_interactive,
        inputs=enable_fps_decrease,
        outputs=[fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps, fps_interpolation_method]
    )

    fps_decrease_mode.change(
        fn=update_fps_mode_controls,
        inputs=fps_decrease_mode,
        outputs=[multiplier_controls, fixed_controls]
    )

    fps_multiplier_preset.change(
        fn=update_multiplier_preset,
        inputs=fps_multiplier_preset,
        outputs=[fps_multiplier_custom, fps_multiplier_custom]
    )

    def enhanced_video_change_handler(
        video_path,
        enable_auto_aspect_resolution,
        target_h,
        target_w,
        upscaler_type,
        enable_target_res,
        target_res_mode,
        upscale_factor,
        image_upscaler_model
    ):
        status_msg, new_target_h, new_target_w, auto_status = handle_video_change_with_auto_resolution(
            video_path, enable_auto_aspect_resolution, target_h, target_w
        )
        resolution_preview = update_resolution_preview_wrapper(
            video_path=video_path,
            upscaler_type=upscaler_type,
            enable_target_res=enable_target_res,
            target_h=new_target_h,
            target_w=new_target_w,
            target_res_mode=target_res_mode,
            upscale_factor=upscale_factor,
            image_upscaler_model=image_upscaler_model,
            enable_auto_aspect_resolution=enable_auto_aspect_resolution
        )
        if video_path and not status_msg.startswith("❌"):
            compact_resolution_preview = calculate_compact_resolution_preview(
                video_path=video_path,
                upscaler_type=upscaler_type,
                enable_target_res=enable_target_res,
                target_h=new_target_h,
                target_w=new_target_w,
                target_res_mode=target_res_mode,
                upscale_factor=upscale_factor,
                image_upscaler_model=image_upscaler_model,
                enable_auto_aspect_resolution=enable_auto_aspect_resolution
            )
            enhanced_status_msg = status_msg + "\n\n" + compact_resolution_preview
        else:
            enhanced_status_msg = status_msg
        return enhanced_status_msg, new_target_h, new_target_w, auto_status, resolution_preview

    input_video.change(
        fn=enhanced_video_change_handler,
        inputs=[
            input_video,
            enable_auto_aspect_resolution_check,
            target_h_num,
            target_w_num,
            upscaler_type_radio,
            enable_target_res_check,
            target_res_mode_radio,
            upscale_factor_slider,
            image_upscaler_model_dropdown
        ],
        outputs=[
            status_textbox,
            target_h_num,
            target_w_num,
            auto_resolution_status_display,
            output_resolution_preview
        ]
    )

    def update_both_resolution_displays(
        video_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        upscale_factor,
        image_upscaler_model,
        enable_auto_aspect_resolution,
        current_status_text
    ):
        detailed_preview = update_resolution_preview_wrapper(
            video_path=video_path,
            upscaler_type=upscaler_type,
            enable_target_res=enable_target_res,
            target_h=target_h,
            target_w=target_w,
            target_res_mode=target_res_mode,
            upscale_factor=upscale_factor,
            image_upscaler_model=image_upscaler_model,
            enable_auto_aspect_resolution=enable_auto_aspect_resolution
        )
        compact_preview = calculate_compact_resolution_preview(
            video_path=video_path,
            upscaler_type=upscaler_type,
            enable_target_res=enable_target_res,
            target_h=target_h,
            target_w=target_w,
            target_res_mode=target_res_mode,
            upscale_factor=upscale_factor,
            image_upscaler_model=image_upscaler_model,
            enable_auto_aspect_resolution=enable_auto_aspect_resolution
        )
        if video_path and compact_preview and current_status_text:
            lines = current_status_text.split('\n')
            new_lines = []
            skip_next_lines = False
            for line in lines:
                if line.startswith('🎯 **Expected Output'):
                    skip_next_lines = True
                    continue
                elif skip_next_lines and line.startswith('•'):
                    continue
                else:
                    skip_next_lines = False
                    new_lines.append(line)
            updated_status = '\n'.join(new_lines).rstrip() + '\n\n' + compact_preview
        elif not video_path and current_status_text:
            updated_status = current_status_text
        else:
            updated_status = current_status_text if current_status_text else ""
        return updated_status, detailed_preview

    resolution_preview_components = [
        upscaler_type_radio,
        enable_target_res_check,
        target_h_num,
        target_w_num,
        target_res_mode_radio,
        upscale_factor_slider,
        image_upscaler_model_dropdown,
        enable_auto_aspect_resolution_check
    ]

    for component in resolution_preview_components:
        component.change(
            fn=update_both_resolution_displays,
            inputs=[
                input_video,
                upscaler_type_radio,
                enable_target_res_check,
                target_h_num,
                target_w_num,
                target_res_mode_radio,
                upscale_factor_slider,
                image_upscaler_model_dropdown,
                enable_auto_aspect_resolution_check,
                status_textbox
            ],
            outputs=[status_textbox, output_resolution_preview]
        )

    for component in [input_video, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps]:
        component.change(
            fn=calculate_fps_preview,
            inputs=[input_video, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps],
            outputs=fps_calculation_info
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
            PROCESSING_SUCCESS_TEMPLATES['temp_cleanup_success'].format(freed_gb=freed_bytes / (1024**3), remaining_label=after_label)
            if success
            else PROCESSING_SUCCESS_TEMPLATES['temp_cleanup_error']
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
            return info_strings.VIDEO_INFO_UPLOAD_DETAILED_INFORMATION
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
            return "✏️ Upload a video first", info_strings.CUT_ANALYSIS_UPLOAD_RANGES_TIME_ESTIMATE_INFO
        if not ranges_input or not ranges_input.strip():
            return "✏️ Enter ranges above to see cut analysis", info_strings.CUT_ANALYSIS_UPLOAD_RANGES_TIME_ESTIMATE_INFO
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
        preview_first_segment_val, seed_num_val, random_seed_check_val,
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
            start_time = time.time()
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
            status_msg = VIDEO_CUTTING_SUCCESS_TEMPLATE.format(
                output_filename=result['final_output'],
                processing_time=time.time() - start_time,
                cuts_applied=len(ranges),
                analysis_text=validation_result['analysis_text']
            )
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
        preview_first_segment_val, seed_num_val, random_seed_check_val,
        progress=gr.Progress(track_tqdm=True)
    ):
        start_time = time.time()
        final_output, preview_path, status_msg = cut_video_wrapper(
            input_video_val, ranges_input, cutting_mode_val, precise_cutting_mode_val,
            preview_first_segment_val, seed_num_val, random_seed_check_val, progress
        )
        if final_output is None:
            return None, None, status_msg, gr.update(), gr.update(visible=False)
        
        # Determine number of cuts for the message
        try:
            if cutting_mode_val == "time_ranges":
                ranges = util_parse_time_ranges(ranges_input, 1e9) # Use large duration for parsing
            else:
                ranges = util_parse_frame_ranges(ranges_input, 1e9) # Use large frame count for parsing
            num_cuts = len(ranges)
        except:
            num_cuts = 1

        integration_msg = VIDEO_CUTTING_INTEGRATION_TEMPLATE.format(
            output_filename=os.path.basename(final_output),
            processing_time=time.time() - start_time,
            cuts_applied=num_cuts,
            status_msg=status_msg
        )
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
        precise_cutting_mode_val, preview_first_segment_val, seed_num_val,
        random_seed_check_val, progress=gr.Progress(track_tqdm=True)
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
            # A simple heuristic to check if the video came from the temp dir used by the editor
            if "temp" in video_path and "gradio" in video_path:
                return gr.update(visible=True, value=f"✅ Cut video from Edit Videos tab: {os.path.basename(video_path)}")
            else:
                return gr.update(visible=False, value="")

    input_video.change(
        fn=handle_main_input_change,
        inputs=input_video,
        outputs=integration_status
    )

    # This list is now used for loading presets
    preset_components = component_order[1:] # Exclude input_video

    def save_preset_wrapper(save_textbox_name, selected_preset, *all_ui_values):
        import time
        
        # If no name entered in save textbox, use the currently selected preset
        if not save_textbox_name or not save_textbox_name.strip():
            if selected_preset and selected_preset.strip():
                preset_name = selected_preset.strip()
                overwrite_msg = f" (overwriting existing preset '{preset_name}')"
            else:
                return "Please enter a preset name or select a preset to overwrite", gr.update(), gr.update()
        else:
            preset_name = save_textbox_name.strip()
            overwrite_msg = ""
        
        app_config = build_app_config_from_ui(*all_ui_values)
        success, message = preset_handler.save_preset(app_config, preset_name)
        if success:
            safe_preset_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            time.sleep(APP_CONFIG.preset_system.save_delay)
            presets_dir = preset_handler.get_presets_dir()
            filepath = os.path.join(presets_dir, f"{safe_preset_name}.json")
            if not os.path.exists(filepath):
                logger.warning(f"Preset file not immediately available after save: {filepath}")
                time.sleep(APP_CONFIG.preset_system.retry_delay)
            new_choices = get_filtered_preset_list()
            # Return: status message with overwrite info, updated dropdown with saved preset selected, clear the save textbox
            return message + overwrite_msg, gr.update(choices=new_choices, value=safe_preset_name), gr.update(value="")
        else:
            # On failure, don't clear the textbox so user can fix the name
            return message, gr.update(), gr.update()

    def load_preset_wrapper(preset_name):
        import time
        if not preset_name or not preset_name.strip():
            return [gr.update(value="No preset selected")] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(16)]
        preset_name = preset_name.strip()
        if preset_name == "last_preset":
            return [gr.update(value=info_strings.CANNOT_LOAD_LAST_PRESET_SYSTEM_FILE)] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(16)]
        
        config_dict, message = None, None
        max_retries = APP_CONFIG.preset_system.load_retries
        for attempt in range(max_retries):
            config_dict, message = preset_handler.load_preset(preset_name)
            if config_dict:
                break
            if attempt < max_retries - 1:
                logger.debug(f"Preset load attempt {attempt + 1} failed, retrying in {APP_CONFIG.preset_system.load_delay}s...")
                time.sleep(APP_CONFIG.preset_system.load_delay)
        
        if not config_dict:
            logger.debug(f"Failed to load preset '{preset_name}' after {max_retries} attempts: {message}")
            return [gr.update(value=f"Could not load preset: {preset_name}")] + [gr.update() for _ in preset_components] + [gr.update() for _ in range(16)]

        default_config = create_app_config(base_path, args.outputs_folder, star_cfg)
        updates = []

        def reverse_extract_codeformer_model_path(path):
            if not path: return "Auto (Default)"
            if "codeformer.pth" in path: return "codeformer.pth (359.2MB)"
            return "Auto (Default)"

        def reverse_upscaler_type_mapping(internal_value):
            if not internal_value: return "Use Image Based Upscalers"
            reverse_map = {"star": "Use STAR Model Upscaler", "image_upscaler": "Use Image Based Upscalers", "seedvr2": "Use SeedVR2 Video Upscaler"}
            return reverse_map.get(internal_value, "Use Image Based Upscalers")

        def reverse_seedvr2_model_filename_to_display_name(filename):
            """Convert SeedVR2 model filename to display name for dropdown"""
            try:
                available_models = util_scan_seedvr2_models(logger=logger)
                if available_models:
                    model_choices = [util_format_model_display_name(model) for model in available_models]
                    
                    # If we have a filename from preset, try to find it
                    if filename:
                        for model_info in available_models:
                            if model_info.get('filename') == filename:
                                return util_format_model_display_name(model_info)
                    
                    # Only if no filename in preset - look for 3B FP8 model as default
                    if not filename:
                        for choice in model_choices:
                            if "3B" in choice and "FP8" in choice:
                                return choice
                    
                    # If 3B FP8 not found or filename was specified but not found, return first available model
                    return model_choices[0]
                else:
                    return "No SeedVR2 models found"
            except Exception as e:
                logger.warning(f"Failed to convert SeedVR2 filename to display name: {e}")
                return "No SeedVR2 models found"

        for component in preset_components:
            if isinstance(component, gr.State) or component is None:
                updates.append(gr.update())
                continue
            if component in component_map:
                section, key = component_map[component]
                default_value = getattr(getattr(default_config, section, {}), key, None)
                value = config_dict.get(section, {}).get(key, default_value)

                if component is codeformer_model_dropdown or component is standalone_codeformer_model_dropdown:
                    value = reverse_extract_codeformer_model_path(value)
                elif component is upscaler_type_radio:
                    value = reverse_upscaler_type_mapping(value)
                elif component is gpu_selector:
                    available_gpus = util_get_available_gpus()
                    value = convert_gpu_index_to_dropdown(value, available_gpus)
                elif component is seedvr2_model_dropdown:
                    value = reverse_seedvr2_model_filename_to_display_name(value)
                
                updates.append(gr.update(value=value))
            else:
                updates.append(gr.update())

        # Conditional updates for interactivity
        try:
            upscaler_type_val = config_dict.get('upscaler_type', {}).get('upscaler_type', default_config.upscaler_type.upscaler_type)
        except AttributeError:
            upscaler_type_val = "image_upscaler"  # fallback default
        enable_image_upscaler_val = (upscaler_type_val == "image_upscaler")
        enable_face_restoration_val = config_dict.get('face_restoration', {}).get('enable', default_config.face_restoration.enable)
        enable_target_res_val = config_dict.get('resolution', {}).get('enable_target_res', default_config.resolution.enable_target_res)
        enable_tiling_val = config_dict.get('tiling', {}).get('enable', default_config.tiling.enable)
        enable_context_window_val = config_dict.get('context_window', {}).get('enable', default_config.context_window.enable)
        enable_scene_split_val = config_dict.get('scene_split', {}).get('enable', default_config.scene_split.enable)
        random_seed_val = config_dict.get('seed', {}).get('use_random', default_config.seed.use_random)

        conditional_updates = [
            gr.update(interactive=enable_image_upscaler_val), gr.update(interactive=enable_image_upscaler_val),
            gr.update(interactive=enable_face_restoration_val), gr.update(interactive=enable_face_restoration_val),
            gr.update(interactive=enable_face_restoration_val), gr.update(interactive=enable_face_restoration_val),
            gr.update(interactive=enable_face_restoration_val),
            gr.update(interactive=enable_target_res_val), gr.update(interactive=enable_target_res_val),
            gr.update(interactive=enable_target_res_val),
            gr.update(interactive=enable_tiling_val), gr.update(interactive=enable_tiling_val),
            gr.update(interactive=enable_context_window_val),
            gr.update(interactive=enable_scene_split_val), gr.update(interactive=enable_scene_split_val),
            gr.update(interactive=enable_scene_split_val), gr.update(interactive=enable_scene_split_val),
            gr.update(interactive=enable_scene_split_val),
            gr.update(interactive=not random_seed_val)
        ]
        
        return [gr.update(value=message)] + updates + conditional_updates

    def refresh_presets_list():
        updated_choices = get_filtered_preset_list()
        logger.info(f"Refreshing preset list: {updated_choices}")
        return gr.update(choices=updated_choices)

    save_preset_btn.click(
        fn=save_preset_wrapper,
        inputs=[preset_save_textbox, preset_dropdown] + click_inputs,
        outputs=[preset_status, preset_dropdown, preset_save_textbox]
    )

    # Add a function to update resolution preview after preset loading
    def update_after_preset_load(
        video_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        upscale_factor,
        image_upscaler_model,
        enable_auto_aspect_resolution,
        current_status_text
    ):
        """Update resolution preview after preset is loaded"""
        if video_path is None:
            return current_status_text, gr.update()
        
        # Update both displays
        updated_status, detailed_preview = update_both_resolution_displays(
            video_path=video_path,
            upscaler_type=upscaler_type,
            enable_target_res=enable_target_res,
            target_h=target_h,
            target_w=target_w,
            target_res_mode=target_res_mode,
            upscale_factor=upscale_factor,
            image_upscaler_model=image_upscaler_model,
            enable_auto_aspect_resolution=enable_auto_aspect_resolution,
            current_status_text=current_status_text
        )
        return updated_status, detailed_preview

    preset_dropdown.change(
        fn=load_preset_wrapper,
        inputs=[preset_dropdown],
        outputs=[preset_status] + preset_components + [
            # Conditional components
            image_upscaler_model_dropdown, image_upscaler_batch_size_slider,
            face_restoration_fidelity_slider, enable_face_colorization_check,
            face_restoration_when_radio, codeformer_model_dropdown, face_restoration_batch_size_slider,
            target_h_num, target_w_num, target_res_mode_radio,
            tile_size_num, tile_overlap_num,
            context_overlap_num,
            scene_split_mode_radio, scene_min_scene_len_num, scene_threshold_num,
            scene_drop_short_check, scene_merge_last_check,
            seed_num
        ]
    ).then(
        fn=update_after_preset_load,
        inputs=[
            input_video,
            upscaler_type_radio,
            enable_target_res_check,
            target_h_num,
            target_w_num,
            target_res_mode_radio,
            upscale_factor_slider,
            image_upscaler_model_dropdown,
            enable_auto_aspect_resolution_check,
            status_textbox
        ],
        outputs=[status_textbox, output_resolution_preview]
    )

    refresh_presets_btn.click(
        fn=refresh_presets_list,
        inputs=[],
        outputs=[preset_dropdown]
    )

    for component in [input_video, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps]:
        component.change(
            fn=calculate_fps_preview,
            inputs=[input_video, fps_decrease_mode, fps_multiplier_preset, fps_multiplier_custom, target_fps],
            outputs=fps_calculation_info
        )

    def refresh_seedvr2_models(current_selection=None):
        try:
            available_models = util_scan_seedvr2_models(logger=logger)
            if available_models:
                model_choices = [util_format_model_display_name(model) for model in available_models]
                
                # Keep current selection if it's still valid
                selected_model = None
                if current_selection and current_selection in model_choices:
                    selected_model = current_selection
                else:
                    # Only default to 3B FP8 if no valid current selection
                    for choice in model_choices:
                        if "3B" in choice and "FP8" in choice:
                            selected_model = choice
                            break
                    
                    # If not found, use the first available model
                    if not selected_model:
                        selected_model = model_choices[0]
                
                logger.info(f"Found {len(available_models)} SeedVR2 models, selected: {selected_model}")
                return gr.update(choices=model_choices, value=selected_model)
            else:
                seedvr2_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SeedVR2', 'models')
                if not os.path.exists(seedvr2_models_path):
                    logger.warning(f"SeedVR2 models directory not found: {seedvr2_models_path}")
                    return gr.update(choices=["SeedVR2 directory not found"], value="SeedVR2 directory not found")
                else:
                    logger.info("SeedVR2 models directory exists but no models found")
                    return gr.update(choices=[SEEDVR2_NO_MODELS_STATUS], value=SEEDVR2_NO_MODELS_STATUS)
        except Exception as e:
            logger.error(f"Failed to refresh SeedVR2 models: {e}")
            return gr.update(choices=[f"Error: {str(e)}"], value=f"Error: {str(e)}")

    def update_seedvr2_model_info(model_choice):
        if not model_choice:
            return gr.update(value="No model selected")
        if "SeedVR2 directory not found" in model_choice:
            return gr.update(value=SEEDVR2_INSTALLATION_MISSING)
        if "No SeedVR2 models found" in model_choice:
            return gr.update(value=SEEDVR2_NO_MODELS_FOUND)
        if "Error:" in model_choice:
            return gr.update(value=SEEDVR2_ERROR_DETECTED_TEMPLATE.format(error_message=model_choice))
        try:
            model_filename = util_extract_model_filename_from_dropdown(model_choice)
            if model_filename:
                model_info = util_format_model_info_display(model_filename)
                logger.info(f"Displaying info for model: {model_filename}")
                return gr.update(value=model_info)
            else:
                return gr.update(value=MODEL_INFO_EXTRACTION_ERROR)
        except Exception as e:
            logger.error(f"Failed to get model info for '{model_choice}': {e}")
            return gr.update(value=f"Error loading model information: {e}")

    def update_block_swap_controls(enable_block_swap, block_swap_counter):
        if not enable_block_swap:
            return (
                gr.update(interactive=False, value=0),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value="Block swap disabled")
            )
        elif block_swap_counter == 0:
            # Block swap is enabled but counter is 0
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(value="⚠️ Block swap enabled but counter is 0. Increase the counter to activate block swapping.")
            )
        else:
            try:
                system_status = util_get_vram_info(logger=logger)
                memory_info = util_format_vram_status(system_status)
                vram_allocated = system_status.get("total_allocated_gb", 0)
                estimated_savings_gb = block_swap_counter * 0.3
                estimated_performance_impact = min(block_swap_counter * 2.5, 60)
                if vram_allocated > 8:
                    status_icon = "🔴"
                    status_msg = "High VRAM usage detected"
                elif vram_allocated > 6:
                    status_icon = "🟡"
                    status_msg = "Moderate VRAM usage"
                else:
                    status_icon = "🟢"
                    status_msg = "VRAM usage normal"
                info_text = BLOCK_SWAP_ADVANCED_ENABLED_TEMPLATE.format(
                    status_icon=status_icon,
                    status_msg=status_msg,
                    memory_info=memory_info,
                    block_swap_counter=block_swap_counter,
                    estimated_savings_gb=estimated_savings_gb,
                    estimated_performance_impact=estimated_performance_impact
                )
            except Exception as e:
                logger.warning(f"Block swap monitoring error: {e}")
                estimated_savings = min(block_swap_counter * 5, 50)
                info_text = BLOCK_SWAP_ENABLED_FALLBACK_TEMPLATE.format(
                    block_swap_counter=block_swap_counter,
                    estimated_savings=estimated_savings,
                    performance_impact=block_swap_counter * 2
                )
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(value=info_text)
            )

    def check_seedvr2_dependencies():
        try:
            all_available, missing_deps = util_check_seedvr2_dependencies(logger=logger)
            if all_available:
                status_text = SEEDVR2_READY_ALL_DEPENDENCIES_AVAILABLE
            else:
                status_text = f"❌ Missing dependencies:\n{'; '.join(missing_deps[:3])}"
            return gr.update(value=status_text)
        except Exception as e:
            logger.error(f"Failed to check SeedVR2 dependencies: {e}")
            return gr.update(value=f"❌ Dependency check failed: {e}")

    def initialize_seedvr2_tab():
        try:
            dep_status = check_seedvr2_dependencies()
            gpu_status = update_gpu_status()
            models_update = refresh_seedvr2_models()
            default_model = models_update.get('value', 'No SeedVR2 models found')
            model_info = update_seedvr2_model_info(default_model) if default_model != 'No SeedVR2 models found' else gr.update(value="No models available")
            return [dep_status, gpu_status, models_update, model_info]
        except Exception as e:
            logger.error(f"Failed to initialize SeedVR2 tab: {e}")
            error_status = gr.update(value=f"❌ Initialization failed: {e}")
            return [error_status, error_status, error_status, error_status]

    def update_gpu_status():
        try:
            gpus = util_detect_available_gpus(logger=logger)
            if not gpus:
                status_text = NO_CUDA_GPUS_DETECTED_DETAILED
                return gr.update(value=status_text)
            vram_info = util_get_vram_info(logger=logger)
            vram_status = util_format_vram_status(vram_info)
            status_lines = [vram_status, "\n🖥️ Available GPUs:"]
            for gpu in gpus:
                gpu_line = f"  • {gpu['display_name']}"
                status_lines.append(gpu_line)
            total_vram = sum(gpu.get('memory_gb', 0) for gpu in gpus)
            recommendation_lines = ["\n🎯 Professional Recommendations:"]
            if total_vram >= 16:
                recommendation_lines.append(info_strings.SINGLE_GPU_7B_FP16_MAXIMUM_QUALITY)
            elif total_vram >= 12:
                recommendation_lines.append(info_strings.SINGLE_GPU_7B_FP8_HIGH_QUALITY_EFFICIENT)
            elif total_vram >= 8:
                recommendation_lines.append("💡 Single GPU: 3B FP16 model (balanced)")
            elif total_vram >= 6:
                recommendation_lines.append(info_strings.SINGLE_GPU_3B_FP8_BLOCK_SWAP_OPTIMIZED)
            else:
                recommendation_lines.append(info_strings.SINGLE_GPU_3B_FP8_AGGRESSIVE_BLOCK_SWAP)
            if len(gpus) >= 2:
                recommendation_lines.append(f"🚀 Multi-GPU: {len(gpus)} GPUs detected for parallel processing")
                recommendation_lines.append(info_strings.MULTI_GPU_ENABLED_INCREASED_PERFORMANCE)
            else:
                recommendation_lines.append(info_strings.MULTI_GPU_NOT_AVAILABLE_NEED_2_PLUS)
            recommendation_lines.extend(["\n🔧 Performance Tips:", info_strings.OPTIMAL_SETTINGS_AUTOMATIC_CONFIGURATION, info_strings.FLASH_ATTENTION_15_20_SPEEDUP, info_strings.SMART_BLOCK_SWAP_VRAM_OPTIMIZATION])
            complete_status = "\n".join(status_lines + recommendation_lines)
            return gr.update(value=complete_status)
        except Exception as e:
            logger.error(f"Failed to update professional GPU status: {e}")
            return gr.update(value=f"❌ Professional GPU analysis failed: {e}")

    def validate_gpu_devices(gpu_devices_text, enable_multi_gpu):
        if not enable_multi_gpu:
            return gr.update(value="Multi-GPU disabled - using single GPU")
        try:
            is_valid, error_messages = util_validate_gpu_selection(gpu_devices_text)
            if is_valid:
                device_indices = [int(x.strip()) for x in gpu_devices_text.split(',') if x.strip()]
                if len(device_indices) > 1:
                    status_text = f"✅ Multi-GPU validated: {len(device_indices)} GPUs\nDevices: {', '.join(map(str, device_indices))}"
                else:
                    status_text = f"✅ Single GPU validated: GPU {device_indices[0]}"
            else:
                status_text = f"❌ {'; '.join(error_messages)}"
            return gr.update(value=status_text)
        except Exception as e:
            logger.error(f"Failed to validate GPU devices: {e}")
            return gr.update(value=f"❌ GPU validation failed: {e}")

    def apply_optimal_seedvr2_settings(current_model):
        try:
            if not current_model or any(error in current_model for error in ["Error:", "No SeedVR2", "SeedVR2 directory"]):
                return [gr.update()] * 12
            model_filename = util_extract_model_filename_from_dropdown(current_model)
            if not model_filename:
                return [gr.update()] * 12
            gpus = util_detect_available_gpus(logger=logger)
            total_vram = sum(gpu.get('free_memory_gb', 0) for gpu in gpus if gpu.get('is_available', False))
            num_gpus = len([gpu for gpu in gpus if gpu.get('is_available', False)])
            recommendations = util_get_recommended_settings_for_vram(
                total_vram_gb=total_vram,
                model_filename=model_filename,
                target_quality='balanced',
                logger=logger
            )
            logger.info(f"Applying optimal settings for {model_filename} with {total_vram:.1f}GB VRAM")
            batch_size = max(5, recommendations.get('batch_size', 8))
            temporal_overlap = 2 if total_vram >= 8 else 1
            preserve_vram = total_vram < 12
            color_correction = True
            enable_frame_padding = True
            flash_attention = True
            enable_multi_gpu = num_gpus > 1 and total_vram >= 8
            gpu_devices = ','.join([str(gpu['id']) for gpu in gpus if gpu.get('is_available', False)][:2]) if enable_multi_gpu else "0"
            enable_block_swap = recommendations.get('enable_block_swap', False)
            block_swap_counter = recommendations.get('block_swap_counter', 0) if enable_block_swap else 0
            block_swap_offload_io = enable_block_swap and total_vram < 6
            block_swap_model_caching = enable_block_swap and total_vram >= 4
            logger.info(f"Applied settings: batch_size={batch_size}, block_swap={enable_block_swap}, multi_gpu={enable_multi_gpu}")
            return [
                                gr.update(value=batch_size),
                gr.update(value=temporal_overlap),
                gr.update(value=preserve_vram),
                gr.update(value=color_correction),
                gr.update(value=enable_frame_padding),
                gr.update(value=flash_attention),
                gr.update(value=enable_multi_gpu),
                gr.update(value=gpu_devices),
                gr.update(value=enable_block_swap),
                gr.update(value=block_swap_counter),
                gr.update(value=block_swap_offload_io),
                gr.update(value=block_swap_model_caching)
            ]
        except Exception as e:
            logger.error(f"Failed to apply optimal settings: {e}")
            return [gr.update()] * 12

    def get_intelligent_block_swap_recommendations(current_model):
        try:
            if not current_model or any(error in current_model for error in ["Error:", "No SeedVR2", "SeedVR2 directory"]):
                return gr.update(value=info_strings.SELECT_VALID_SEEDVR2_MODEL_FIRST)
            model_filename = util_extract_model_filename_from_dropdown(current_model)
            if not model_filename:
                return gr.update(value="Could not extract model information.")
            recommendations = util_get_block_swap_recommendations(
                model_filename=model_filename,
                target_quality="balanced",
                logger=logger
            )
            formatted_recommendations = util_format_block_swap_status(recommendations)
            system_status = util_get_vram_info(logger=logger)
            memory_info = util_format_vram_status(system_status)
            enhanced_display = INTELLIGENT_BLOCK_SWAP_ANALYSIS_TEMPLATE.format(
                model_filename=model_filename,
                vram_requirements=f"Current System Status:\n{memory_info}",
                recommendations=formatted_recommendations
            )
            logger.info(f"Generated block swap recommendations for {model_filename}")
            return gr.update(value=enhanced_display)
        except Exception as e:
            logger.error(f"Failed to get block swap recommendations: {e}")
            return gr.update(value=f"Error generating recommendations: {e}")

    def get_professional_multi_gpu_recommendations(current_model):
        try:
            if not current_model or any(error in current_model for error in ["Error:", "No SeedVR2", "SeedVR2 directory"]):
                return gr.update(value=info_strings.SELECT_VALID_SEEDVR2_MODEL_FIRST)
            model_filename = util_extract_model_filename_from_dropdown(current_model)
            if not model_filename:
                return gr.update(value="Could not extract model information.")
            gpus = util_detect_available_gpus(logger=logger)
            multi_gpu_analysis = util_get_suggested_settings(
                gpus, model_filename, logger=logger
            )
            formatted_recommendations = util_format_block_swap_status(multi_gpu_analysis)
            multi_gpu_status = util_format_vram_status(util_get_vram_info(logger=logger))
            enhanced_display = PROFESSIONAL_MULTI_GPU_ANALYSIS_TEMPLATE.format(
                model_filename=model_filename,
                total_vram=sum(gpu.get('memory_gb', 0) for gpu in gpus),
                recommendations=f"{multi_gpu_status}\n\n{formatted_recommendations}"
            )
            logger.info(f"Generated professional multi-GPU recommendations for {model_filename}")
            return gr.update(value=enhanced_display)
        except Exception as e:
            logger.error(f"Failed to get multi-GPU recommendations: {e}")
            return gr.update(value=f"Error generating multi-GPU analysis: {e}")

    refresh_seedvr2_models_btn.click(
        fn=refresh_seedvr2_models,
        inputs=[seedvr2_model_dropdown],
        outputs=[seedvr2_model_dropdown]
    )

    apply_recommended_settings_btn.click(
        fn=apply_optimal_seedvr2_settings,
        inputs=[seedvr2_model_dropdown],
        outputs=[
            seedvr2_batch_size_slider,
            seedvr2_temporal_overlap_slider,
            seedvr2_preserve_vram_check,
            seedvr2_color_correction_check,
            seedvr2_enable_frame_padding_check,
            seedvr2_flash_attention_check,
            seedvr2_enable_multi_gpu_check,
            seedvr2_gpu_devices_textbox,
            seedvr2_enable_block_swap_check,
            seedvr2_block_swap_counter_slider,
            seedvr2_block_swap_offload_io_check,
            seedvr2_block_swap_model_caching_check
        ]
    )

    get_block_swap_recommendations_btn.click(
        fn=get_intelligent_block_swap_recommendations,
        inputs=[seedvr2_model_dropdown],
        outputs=[seedvr2_model_info_display]
    )

    get_multi_gpu_recommendations_btn.click(
        fn=get_professional_multi_gpu_recommendations,
        inputs=[seedvr2_model_dropdown],
        outputs=[seedvr2_model_info_display]
    )

    def validate_and_update_model_info(model_choice):
        try:
            basic_info = update_seedvr2_model_info(model_choice)
            if not any(error in model_choice for error in ["Error:", "No SeedVR2", "SeedVR2 directory"]):
                model_filename = util_extract_model_filename_from_dropdown(model_choice)
                if model_filename:
                    is_valid, validation_msg = util_validate_seedvr2_model(model_filename, logger=logger)
                    if is_valid:
                        gpus = util_detect_available_gpus(logger=logger)
                        total_vram = sum(gpu.get('free_memory_gb', 0) for gpu in gpus if gpu.get('is_available', False))
                        recommendations = util_get_recommended_settings_for_vram(
                            total_vram_gb=total_vram,
                            model_filename=model_filename,
                            target_quality='balanced',
                            logger=logger
                        )
                        current_info = basic_info.get('value', '')
                        
                        # Pre-calculate status values for template
                        block_swap_status = 'Enabled' if recommendations.get('enable_block_swap', False) else 'Disabled'
                        multi_gpu_status = 'Enabled' if (len(gpus) > 1 and total_vram >= 8) else 'Disabled'
                        
                        enhanced_info = MODEL_VALIDATION_ENHANCED_TEMPLATE.format(
                            current_info=current_info,
                            architecture="Unknown",
                            scale_factor="Unknown",
                            parameters="Unknown",
                            vram_usage="Unknown",
                            total_vram=total_vram,
                            block_swap_status=block_swap_status,
                            multi_gpu_status=multi_gpu_status
                        )
                        return gr.update(value=enhanced_info)
                    else:
                        error_info = MODEL_VALIDATION_FAILED_TEMPLATE.format(error_message=validation_msg)
                        return gr.update(value=error_info)
            return basic_info
        except Exception as e:
            logger.error(f"Failed to validate model '{model_choice}': {e}")
            return gr.update(value=f"Error validating model: {e}")

    seedvr2_model_dropdown.change(
        fn=validate_and_update_model_info,
        inputs=[seedvr2_model_dropdown],
        outputs=[seedvr2_model_info_display]
    )

    for component in [seedvr2_enable_block_swap_check, seedvr2_block_swap_counter_slider]:
        component.change(
            fn=update_block_swap_controls,
            inputs=[seedvr2_enable_block_swap_check, seedvr2_block_swap_counter_slider],
            outputs=[
                seedvr2_block_swap_counter_slider,
                seedvr2_block_swap_offload_io_check,
                seedvr2_block_swap_model_caching_check,
                seedvr2_block_swap_info_display
            ]
        )

    for component in [seedvr2_enable_multi_gpu_check, seedvr2_gpu_devices_textbox]:
        component.change(
            fn=validate_gpu_devices,
            inputs=[seedvr2_gpu_devices_textbox, seedvr2_enable_multi_gpu_check],
            outputs=[seedvr2_gpu_status_display]
        )

    demo.load(
        fn=initialize_seedvr2_tab,
        inputs=[],
        outputs=[seedvr2_dependency_status, seedvr2_gpu_status_display, seedvr2_model_dropdown, seedvr2_model_info_display]
    )

    def refresh_block_swap_status(enable_block_swap, block_swap_counter):
        if enable_block_swap and block_swap_counter > 0:
            try:
                system_status = util_get_vram_info(logger=logger)
                memory_info = util_format_vram_status(system_status)
                vram_allocated = system_status.get("total_allocated_gb", 0)
                if vram_allocated > 8:
                    status_icon = "🔴"
                    status_msg = "High VRAM usage"
                elif vram_allocated > 6:
                    status_icon = "🟡"
                    status_msg = "Moderate VRAM usage"
                else:
                    status_icon = "🟢"
                    status_msg = "VRAM usage normal"
                estimated_savings_gb = block_swap_counter * 0.3
                updated_status = BLOCK_SWAP_ACTIVE_STATUS_TEMPLATE.format(
                    status_msg=f"{status_icon} {status_msg}",
                    memory_info=memory_info,
                    block_swap_counter=block_swap_counter,
                    estimated_savings=estimated_savings_gb
                )
                return gr.update(value=updated_status)
            except Exception as e:
                return gr.update(value=f"⚠️ Block swap monitoring error: {e}")
        else:
            return gr.update()

    def calculate_image_resolution_preview(
        image_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        image_upscaler_model,
        seedvr2_model=None
    ):
        if not image_path:
            return "Upload an image to see resolution preview..."
        
        try:
            from logic.seedvr2_image_core import _extract_image_info
            image_info = _extract_image_info(image_path, logger)
            orig_w = image_info.get('width', 0)
            orig_h = image_info.get('height', 0)
            
            if orig_w <= 0 or orig_h <= 0:
                return "Invalid image dimensions"
            
            # Determine internal upscaler type and effective upscale factor
            if upscaler_type == "Use SeedVR2 for Images":
                internal_upscaler_type = "seedvr2"
                effective_upscale_factor = 4.0  # SeedVR2 is always 4x
                model_name = "SeedVR2 (4x)"
                # Add specific SeedVR2 model if available
                if seedvr2_model:
                    selected_model_display = f"\n🎯 **Selected Model:** {seedvr2_model}"
                else:
                    selected_model_display = "\n🎯 **Selected Model:** SeedVR2 (Default)"
            else:
                internal_upscaler_type = "image_upscaler"
                # Get model-specific upscale factor
                if image_upscaler_model:
                    model_scale = util_get_model_scale_from_name(image_upscaler_model)
                    effective_upscale_factor = model_scale if model_scale else 4.0
                    model_name = f"{image_upscaler_model} ({effective_upscale_factor}x)"
                    selected_model_display = f"\n🎯 **Selected Model:** {image_upscaler_model}"
                else:
                    effective_upscale_factor = 4.0
                    model_name = "Image Upscaler (4x)"
                    selected_model_display = "\n🎯 **Selected Model:** No model selected"
            
            if enable_target_res:
                try:
                    custom_upscale_factor = effective_upscale_factor if internal_upscaler_type == "seedvr2" else None
                    needs_downscale, ds_h, ds_w, upscale_factor_calc, final_h_calc, final_w_calc = util_calculate_upscale_params(
                        orig_h, orig_w, target_h, target_w, target_res_mode,
                        logger=logger,
                        image_upscaler_model=image_upscaler_model if internal_upscaler_type == "image_upscaler" else None,
                        custom_upscale_factor=custom_upscale_factor
                    )
                    
                    downscale_info = f"\n• Downscale first: {ds_w}×{ds_h}" if needs_downscale else ""
                    
                    return f"""{selected_model_display}

🎯 **Expected Output Resolution**
• Input: {orig_w}×{orig_h}
• Model: {model_name}
• Output: {final_w_calc}×{final_h_calc} ({upscale_factor_calc:.2f}x)
• Mode: {target_res_mode}{downscale_info}"""
                except Exception as e:
                    return f"❌ Error calculating resolution: {str(e)}"
            else:
                # Simple upscale without target resolution
                final_h = int(round(orig_h * effective_upscale_factor / 2) * 2)
                final_w = int(round(orig_w * effective_upscale_factor / 2) * 2)
                return f"""{selected_model_display}

🎯 **Expected Output Resolution**
• Input: {orig_w}×{orig_h}
• Model: {model_name}
• Output: {final_w}×{final_h} ({effective_upscale_factor}x scale)"""
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def handle_image_upload(image_path):
        if not image_path:
            return (
                gr.update(value="Upload an image to see details..."),
                gr.update(value="Ready to process images..."),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        try:
            from logic.seedvr2_image_core import _extract_image_info, format_image_info_display as util_format_image_info_display
            image_info = _extract_image_info(image_path, logger)
            
            # Get basic info first
            basic_info = util_format_image_info_display(image_info)
            
            # Return basic info - resolution preview will be updated separately
            return (
                gr.update(value=basic_info),
                gr.update(value=f"✅ Image loaded: {image_info.get('width', 0)}×{image_info.get('height', 0)} pixels"),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        except Exception as e:
            error_msg = f"❌ Error loading image: {e}"
            logger.error(error_msg)
            return (
                gr.update(value=error_msg),
                gr.update(value=error_msg),
                gr.update(visible=False),
                gr.update(visible=False)
            )

    def image_upscale_wrapper(
        input_image_path, upscaler_type, image_upscaler_model_val, output_format, output_quality,
        preserve_aspect_ratio, preserve_metadata, custom_suffix,
        seedvr2_model_val, seedvr2_batch_size_val, seedvr2_cfg_scale_val,
        seedvr2_enable_block_swap_val, seedvr2_block_swap_counter_val,
        enable_target_res_val, target_h_val, target_w_val, target_res_mode_val,
        seed_value, use_random_seed,
        # Additional SeedVR2 settings
        seedvr2_preserve_vram_val, seedvr2_flash_attention_val, seedvr2_color_correction_val,
        seedvr2_enable_multi_gpu_val, seedvr2_gpu_devices_val,
        seedvr2_block_swap_offload_io_val, seedvr2_block_swap_model_caching_val,
        seedvr2_quality_preset_val, seedvr2_use_gpu_val,
        progress=gr.Progress(track_tqdm=True)
    ):
        if not input_image_path:
            return (
                gr.update(),
                gr.update(visible=False),
                "❌ Please upload an image first",
                gr.update(),
                "❌ No image uploaded"
            )
        try:
            if upscaler_type == "Use SeedVR2 for Images":
                actual_upscaler_type = "seedvr2"
                from logic.star_dataclasses import SeedVR2Config
                seedvr2_config = SeedVR2Config()
                # Extract the actual model filename from dropdown selection
                if seedvr2_model_val:
                    model_filename = util_extract_model_filename_from_dropdown(seedvr2_model_val)
                    if model_filename:
                        seedvr2_config.model = model_filename  # Use 'model' field, not 'model_filename'
                        logger.info(f"Single image upscale using SeedVR2 model: {model_filename}")
                # Set batch size from UI
                if seedvr2_batch_size_val:
                    seedvr2_config.batch_size = seedvr2_batch_size_val
                # Set CFG scale
                if seedvr2_cfg_scale_val is not None:
                    seedvr2_config.cfg_scale = seedvr2_cfg_scale_val
                # Set block swap settings
                seedvr2_config.enable_block_swap = seedvr2_enable_block_swap_val
                seedvr2_config.block_swap_counter = seedvr2_block_swap_counter_val
                seedvr2_config.block_swap_offload_io = seedvr2_block_swap_offload_io_val
                seedvr2_config.block_swap_model_caching = seedvr2_block_swap_model_caching_val
                # Set resolution settings for SeedVR2
                seedvr2_config.enable_target_res = enable_target_res_val
                seedvr2_config.target_h = target_h_val
                seedvr2_config.target_w = target_w_val
                seedvr2_config.target_res_mode = target_res_mode_val
                # Set additional SeedVR2 settings
                seedvr2_config.preserve_vram = seedvr2_preserve_vram_val
                seedvr2_config.flash_attention = seedvr2_flash_attention_val
                seedvr2_config.color_correction = seedvr2_color_correction_val
                seedvr2_config.enable_multi_gpu = seedvr2_enable_multi_gpu_val
                seedvr2_config.gpu_devices = seedvr2_gpu_devices_val
                seedvr2_config.quality_preset = seedvr2_quality_preset_val
                seedvr2_config.use_gpu = seedvr2_use_gpu_val
                # Set seed
                seedvr2_config.seed = seed_value
                image_upscaler_model = None
            else:
                actual_upscaler_type = "image_upscaler"
                seedvr2_config = None
                image_upscaler_model = image_upscaler_model_val
            
            from logic.seedvr2_image_core import process_single_image as util_process_single_image, format_image_info_display as util_format_image_info_display
            
            # Handle seed generation similar to video processing
            actual_seed = seed_value
            if use_random_seed:
                actual_seed = np.random.randint(0, 2**31)
                logger.info(f"Random seed checkbox is checked. Using generated seed: {actual_seed}")
            elif seed_value == -1:
                actual_seed = np.random.randint(0, 2**31)
                logger.info(f"Seed input is -1. Using generated seed: {actual_seed}")
            else:
                logger.info(f"Using provided seed: {actual_seed}")
            current_seed = actual_seed
            
            # Always create comparison for the slider
            results = list(util_process_single_image(
                input_image_path=input_image_path,
                upscaler_type=actual_upscaler_type,
                seedvr2_config=seedvr2_config,
                image_upscaler_model=image_upscaler_model,
                output_format=output_format,
                output_quality=output_quality,
                preserve_aspect_ratio=preserve_aspect_ratio,
                preserve_metadata=preserve_metadata,
                custom_suffix=custom_suffix,
                create_comparison=True,  # Always create for slider
                output_dir=None,  # Will use default STAR/outputs_images
                logger=logger,
                progress=progress,
                util_get_gpu_device=util_get_gpu_device,
                format_time=format_time,
                current_seed=current_seed
            ))
            if results:
                output_image_path, comparison_image_path, status_message, image_info = results[0]
                logger.info(f"Image upscale wrapper - output path: {output_image_path}")
                logger.info(f"Image upscale wrapper - file exists: {os.path.exists(output_image_path) if output_image_path else 'None'}")
                
                formatted_info = util_format_image_info_display(image_info)
                
                # Ensure we have an absolute path
                if output_image_path and not os.path.isabs(output_image_path):
                    output_image_path = os.path.abspath(output_image_path)
                    logger.info(f"Converted to absolute path: {output_image_path}")
                
                # Force reload the image by creating a new path reference
                output_image_update = gr.update(value=str(output_image_path), visible=True)
                
                # Update image slider with before/after images
                if output_image_path:
                    slider_update = gr.update(value=(input_image_path, output_image_path), visible=True)
                else:
                    slider_update = gr.update(visible=False)
                
                processing_log = PROCESSING_SUCCESS_TEMPLATES['image_processing_complete'].format(
                    output_filename=os.path.basename(output_image_path),
                    width=image_info.get('output_width', 0),
                    height=image_info.get('output_height', 0),
                    processing_time=image_info.get('processing_time', 0),
                    upscale_factor=image_info.get('upscale_factor_x', 1)
                )
                
                logger.info(f"Returning output_image_update value: {output_image_update}")
                logger.info(f"Returning status_message: {status_message}")
                
                return (
                    output_image_update,
                    slider_update,
                    status_message,
                    gr.update(value=formatted_info),
                    processing_log
                )
            else:
                error_msg = GENERAL_ERROR_TEMPLATES['no_results_error']
                return (
                    gr.update(),
                    gr.update(visible=False),
                    error_msg,
                    gr.update(),
                    error_msg
                )
        except Exception as e:
            error_msg = f"❌ Image processing failed: {e}"
            logger.error(error_msg, exc_info=True)
            return (
                gr.update(),
                gr.update(visible=False),
                error_msg,
                gr.update(),
                error_msg
            )

    def update_image_upscaler_visibility(upscaler_type):
        if upscaler_type == "Use SeedVR2 for Images":
            return gr.update(visible=True)
        else:
            return gr.update(visible=True)

    def enhanced_image_upload_handler(
        image_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        image_upscaler_model,
        seedvr2_model,
        seedvr2_enable_multi_gpu,
        seedvr2_gpu_devices
    ):
        # Get basic image info
        basic_info, status, output_update, slider_update = handle_image_upload(image_path)
        
        if image_path and not status['value'].startswith("❌"):
            # Calculate resolution preview
            resolution_preview = calculate_image_resolution_preview(
                image_path=image_path,
                upscaler_type=upscaler_type,
                enable_target_res=enable_target_res,
                target_h=target_h,
                target_w=target_w,
                target_res_mode=target_res_mode,
                image_upscaler_model=image_upscaler_model,
                seedvr2_model=seedvr2_model
            )
            
            # Combine basic info with resolution preview
            combined_info = basic_info['value'] + "\n\n" + resolution_preview
            basic_info['value'] = combined_info
        
        return basic_info, status, output_update, slider_update
    
    input_image.change(
        fn=enhanced_image_upload_handler,
        inputs=[
            input_image,
            image_upscaler_type_radio,
            enable_target_res_check,
            target_h_num,
            target_w_num,
            target_res_mode_radio,
            image_upscaler_model_dropdown,
            seedvr2_model_dropdown,
            seedvr2_enable_multi_gpu_check,
            seedvr2_gpu_devices_textbox
        ],
        outputs=[
            image_info_display,
            image_processing_status,
            output_image,
            image_comparison_slider
        ]
    )

    image_upscale_button.click(
        fn=image_upscale_wrapper,
        inputs=[
            input_image,
            image_upscaler_type_radio,
            image_upscaler_model_dropdown,
            image_output_format,
            image_quality_level,
            image_preserve_aspect_ratio,
            image_preserve_metadata,
            image_custom_suffix,
            seedvr2_model_dropdown,
            seedvr2_batch_size_slider,
            seedvr2_cfg_scale_slider,
            seedvr2_enable_block_swap_check,
            seedvr2_block_swap_counter_slider,
            enable_target_res_check,
            target_h_num,
            target_w_num,
            target_res_mode_radio,
            seed_num,
            random_seed_check,
            # Additional SeedVR2 settings
            seedvr2_preserve_vram_check,
            seedvr2_flash_attention_check,
            seedvr2_color_correction_check,
            seedvr2_enable_multi_gpu_check,
            seedvr2_gpu_devices_textbox,
            seedvr2_block_swap_offload_io_check,
            seedvr2_block_swap_model_caching_check,
            seedvr2_quality_preset_radio,
            seedvr2_use_gpu_check
        ],
        outputs=[
            output_image,
            image_comparison_slider,
            image_processing_status,
            image_info_display,
            image_log_display
        ]
    )
    
    open_image_output_folder_button.click(
        fn=lambda: util_open_folder(os.path.join(os.path.dirname(__file__), 'outputs_images'), logger=logger),
        inputs=[],
        outputs=[]
    )

    def update_image_resolution_preview(
        image_path,
        upscaler_type,
        enable_target_res,
        target_h,
        target_w,
        target_res_mode,
        image_upscaler_model,
        seedvr2_model,
        seedvr2_enable_multi_gpu,
        seedvr2_gpu_devices
    ):
        if not image_path:
            return gr.update()  # Don't update if no image
        
        try:
            from logic.seedvr2_image_core import _extract_image_info, format_image_info_display as util_format_image_info_display
            image_info = _extract_image_info(image_path, logger)
            
            # Get basic info
            basic_info = util_format_image_info_display(image_info)
            
            # Calculate resolution preview
            resolution_preview = calculate_image_resolution_preview(
                image_path=image_path,
                upscaler_type=upscaler_type,
                enable_target_res=enable_target_res,
                target_h=target_h,
                target_w=target_w,
                target_res_mode=target_res_mode,
                image_upscaler_model=image_upscaler_model,
                seedvr2_model=seedvr2_model
            )
            
            # Combine info
            combined_info = basic_info + "\n\n" + resolution_preview
            return gr.update(value=combined_info)
            
        except Exception as e:
            logger.error(f"Error updating image resolution preview: {e}")
            return gr.update()
    
    # Components that affect resolution calculation
    resolution_affecting_components = [
        image_upscaler_type_radio,
        enable_target_res_check,
        target_h_num,
        target_w_num,
        target_res_mode_radio,
        image_upscaler_model_dropdown,
        seedvr2_model_dropdown,
        seedvr2_enable_multi_gpu_check,
        seedvr2_gpu_devices_textbox
    ]
    
    # Add change handlers for all components that affect resolution
    for component in resolution_affecting_components:
        component.change(
            fn=update_image_resolution_preview,
            inputs=[
                input_image,
                image_upscaler_type_radio,
                enable_target_res_check,
                target_h_num,
                target_w_num,
                target_res_mode_radio,
                image_upscaler_model_dropdown,
                seedvr2_model_dropdown,
                seedvr2_enable_multi_gpu_check,
                seedvr2_gpu_devices_textbox
            ],
            outputs=image_info_display
        )
    
    image_upscaler_type_radio.change(
        fn=update_image_upscaler_visibility,
        inputs=image_upscaler_type_radio,
        outputs=image_processing_status
    )

if __name__ == "__main__":
    os.makedirs(APP_CONFIG.paths.outputs_dir, exist_ok=True)
    logger.info(f"Gradio App Starting. Default output to: {os.path.abspath(APP_CONFIG.paths.outputs_dir)}")
    logger.info(f"STAR Models expected at: {APP_CONFIG.paths.light_deg_model_path}, {APP_CONFIG.paths.heavy_deg_model_path}")
    if UTIL_COG_VLM_AVAILABLE:
        logger.info(f"CogVLM2 Model expected at: {APP_CONFIG.paths.cogvlm_model_path}")

    available_gpus_main = util_get_available_gpus()
    if available_gpus_main:
        default_gpu_main_val = available_gpus_main[0]
        util_set_gpu_device(default_gpu_main_val, logger=logger)
        logger.info(f"Initialized with default GPU: {default_gpu_main_val} (GPU 0)")
    else:
        logger.info(info_strings.NO_CUDA_GPUS_CPU_FALLBACK)
        util_set_gpu_device(None, logger=logger)

    effective_allowed_paths = util_get_available_drives(APP_CONFIG.paths.outputs_dir, base_path, logger=logger)

    demo.queue().launch(
        debug=True,
        max_threads=100,
        inbrowser=True,
        share=args.share,
        allowed_paths=effective_allowed_paths,
        prevent_thread_lock=True
    )