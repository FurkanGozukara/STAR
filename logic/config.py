import os

# These will be imported from cogvlm_utils by the main app and then passed here or config will import them.
# For now, let them be importable directly if cogvlm_utils is in the path
try:
    from .cogvlm_utils import COG_VLM_AVAILABLE as UTIL_COG_VLM_AVAILABLE
    from .cogvlm_utils import BITSANDBYTES_AVAILABLE as UTIL_BITSANDBYTES_AVAILABLE
except ImportError:
    # Fallback if this module is run/imported in a context where relative import fails
    # This might happen during linting or testing in some setups.
    # The main app (secourses_app.py) will definitively use the correct import.
    UTIL_COG_VLM_AVAILABLE = False
    UTIL_BITSANDBYTES_AVAILABLE = False


# --- Path-dependent constants (to be initialized by secourses_app.py) ---
APP_BASE_PATH = None
# DEFAULT_OUTPUT_DIR will be initialized using args.outputs_folder
DEFAULT_OUTPUT_DIR = "outputs" # This is a fallback/initial default
COG_VLM_MODEL_PATH = None
LIGHT_DEG_MODEL_PATH = None
HEAVY_DEG_MODEL_PATH = None

# --- Prompt constants (to be initialized by secourses_app.py from star_cfg) ---
DEFAULT_POS_PROMPT = "Default Positive Prompt Placeholder"
DEFAULT_NEG_PROMPT = "Default Negative Prompt Placeholder"

# --- UI Default Values ---
# Gradio core upscaling settings
DEFAULT_MODEL_CHOICE = "Light Degradation"
DEFAULT_UPSCALE_FACTOR = 4.0
DEFAULT_CFG_SCALE = 7.5
DEFAULT_SOLVER_MODE = "fast"
DEFAULT_DIFFUSION_STEPS_FAST = 15
DEFAULT_DIFFUSION_STEPS_NORMAL = 50
DEFAULT_COLOR_FIX_METHOD = "AdaIN"

# Performance & VRAM
DEFAULT_MAX_CHUNK_LEN = 32
DEFAULT_VAE_CHUNK = 3

# Target Resolution
DEFAULT_ENABLE_TARGET_RES = True
DEFAULT_TARGET_RES_MODE = "Downscale then 4x"
DEFAULT_TARGET_H = 512
DEFAULT_TARGET_W = 512

# Sliding Window
DEFAULT_ENABLE_SLIDING_WINDOW = False
DEFAULT_WINDOW_SIZE = 32
DEFAULT_WINDOW_STEP = 16

# Tiling
DEFAULT_ENABLE_TILING = False
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_OVERLAP = 64

# FFmpeg
DEFAULT_FFMPEG_USE_GPU = True
DEFAULT_FFMPEG_PRESET = "medium"
DEFAULT_FFMPEG_QUALITY_CPU = 23
DEFAULT_FFMPEG_QUALITY_GPU = 25

# Scene Splitting
DEFAULT_ENABLE_SCENE_SPLIT = True
DEFAULT_SCENE_SPLIT_MODE = "automatic"
# Auto Scene Detection
DEFAULT_SCENE_MIN_SCENE_LEN = 0.6
DEFAULT_SCENE_THRESHOLD = 3.0
DEFAULT_SCENE_DROP_SHORT = False
DEFAULT_SCENE_MERGE_LAST = True
DEFAULT_SCENE_FRAME_SKIP = 0
DEFAULT_SCENE_MIN_CONTENT_VAL = 15.0
DEFAULT_SCENE_FRAME_WINDOW = 2
# Manual Scene Split
DEFAULT_SCENE_MANUAL_SPLIT_TYPE = "duration"
DEFAULT_SCENE_MANUAL_SPLIT_VALUE = 30.0
# Scene Encoding
DEFAULT_SCENE_COPY_STREAMS = False
DEFAULT_SCENE_USE_MKVMERGE = True
DEFAULT_SCENE_RATE_FACTOR = 12
DEFAULT_SCENE_ENCODING_PRESET = "slower"
DEFAULT_SCENE_QUIET_FFMPEG = True

# CogVLM (if available)
DEFAULT_COGVLM_QUANT_DISPLAY_FP16 = "FP16/BF16"
DEFAULT_COGVLM_QUANT_DISPLAY_INT4 = "INT4 (CUDA)"
DEFAULT_COGVLM_UNLOAD_AFTER_USE = "full"
DEFAULT_AUTO_CAPTION_THEN_UPSCALE = True

# Output Options
DEFAULT_SAVE_FRAMES = True
DEFAULT_SAVE_METADATA = True
DEFAULT_SAVE_CHUNKS = True

# --- Initialization functions ---
def initialize_paths_and_prompts(base_path_from_app, outputs_folder_from_args, star_cfg_from_app):
    global APP_BASE_PATH, DEFAULT_OUTPUT_DIR, COG_VLM_MODEL_PATH, LIGHT_DEG_MODEL_PATH, HEAVY_DEG_MODEL_PATH
    global DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT

    APP_BASE_PATH = base_path_from_app
    DEFAULT_OUTPUT_DIR = os.path.abspath(outputs_folder_from_args)
    COG_VLM_MODEL_PATH = os.path.join(APP_BASE_PATH, 'models', 'cogvlm2-video-llama3-chat')
    LIGHT_DEG_MODEL_PATH = os.path.join(APP_BASE_PATH, 'pretrained_weight', 'light_deg.pt')
    HEAVY_DEG_MODEL_PATH = os.path.join(APP_BASE_PATH, 'pretrained_weight', 'heavy_deg.pt')

    if star_cfg_from_app:
        DEFAULT_POS_PROMPT = star_cfg_from_app.positive_prompt
        DEFAULT_NEG_PROMPT = star_cfg_from_app.negative_prompt
    else:
        # Fallback if star_cfg is not available for some reason
        print("Warning: star_cfg not provided to config.initialize_paths_and_prompts. Using placeholder prompts.")


def get_cogvlm_quant_choices_map(torch_cuda_available, util_bitsandbytes_available_from_cogvlm_utils):
    choices_map = {0: DEFAULT_COGVLM_QUANT_DISPLAY_FP16}
    if torch_cuda_available and util_bitsandbytes_available_from_cogvlm_utils:
        choices_map[4] = DEFAULT_COGVLM_QUANT_DISPLAY_INT4
        choices_map[8] = "INT8 (CUDA)"
    return choices_map

def get_default_cogvlm_quant_display(cogvlm_quant_choices_map):
    return DEFAULT_COGVLM_QUANT_DISPLAY_INT4 if 4 in cogvlm_quant_choices_map else DEFAULT_COGVLM_QUANT_DISPLAY_FP16

# Expose UTIL_COG_VLM_AVAILABLE and UTIL_BITSANDBYTES_AVAILABLE from cogvlm_utils if they were imported successfully
# This makes them centrally accessible via this config module.
# The main app should ensure cogvlm_utils is in sys.path before importing config if direct import within config fails.

__all__ = [
    # Initialized values
    'APP_BASE_PATH', 'DEFAULT_OUTPUT_DIR', 'COG_VLM_MODEL_PATH', 'LIGHT_DEG_MODEL_PATH', 'HEAVY_DEG_MODEL_PATH',
    'DEFAULT_POS_PROMPT', 'DEFAULT_NEG_PROMPT',
    # Direct imports/values
    'UTIL_COG_VLM_AVAILABLE', 'UTIL_BITSANDBYTES_AVAILABLE',
    'DEFAULT_MODEL_CHOICE', 'DEFAULT_UPSCALE_FACTOR', 'DEFAULT_CFG_SCALE', 'DEFAULT_SOLVER_MODE',
    'DEFAULT_DIFFUSION_STEPS_FAST', 'DEFAULT_DIFFUSION_STEPS_NORMAL', 'DEFAULT_COLOR_FIX_METHOD',
    'DEFAULT_MAX_CHUNK_LEN', 'DEFAULT_VAE_CHUNK',
    'DEFAULT_ENABLE_TARGET_RES', 'DEFAULT_TARGET_RES_MODE', 'DEFAULT_TARGET_H', 'DEFAULT_TARGET_W',
    'DEFAULT_ENABLE_SLIDING_WINDOW', 'DEFAULT_WINDOW_SIZE', 'DEFAULT_WINDOW_STEP',
    'DEFAULT_ENABLE_TILING', 'DEFAULT_TILE_SIZE', 'DEFAULT_TILE_OVERLAP',
    'DEFAULT_FFMPEG_USE_GPU', 'DEFAULT_FFMPEG_PRESET', 'DEFAULT_FFMPEG_QUALITY_CPU', 'DEFAULT_FFMPEG_QUALITY_GPU',
    'DEFAULT_ENABLE_SCENE_SPLIT', 'DEFAULT_SCENE_SPLIT_MODE',
    'DEFAULT_SCENE_MIN_SCENE_LEN', 'DEFAULT_SCENE_THRESHOLD', 'DEFAULT_SCENE_DROP_SHORT', 'DEFAULT_SCENE_MERGE_LAST',
    'DEFAULT_SCENE_FRAME_SKIP', 'DEFAULT_SCENE_MIN_CONTENT_VAL', 'DEFAULT_SCENE_FRAME_WINDOW',
    'DEFAULT_SCENE_MANUAL_SPLIT_TYPE', 'DEFAULT_SCENE_MANUAL_SPLIT_VALUE',
    'DEFAULT_SCENE_COPY_STREAMS', 'DEFAULT_SCENE_USE_MKVMERGE', 'DEFAULT_SCENE_RATE_FACTOR',
    'DEFAULT_SCENE_ENCODING_PRESET', 'DEFAULT_SCENE_QUIET_FFMPEG',
    'DEFAULT_COGVLM_QUANT_DISPLAY_FP16', 'DEFAULT_COGVLM_QUANT_DISPLAY_INT4', 'DEFAULT_COGVLM_UNLOAD_AFTER_USE',
    'DEFAULT_AUTO_CAPTION_THEN_UPSCALE',
    'DEFAULT_SAVE_FRAMES', 'DEFAULT_SAVE_METADATA', 'DEFAULT_SAVE_CHUNKS',
    # Functions
    'initialize_paths_and_prompts', 'get_cogvlm_quant_choices_map', 'get_default_cogvlm_quant_display'
] 