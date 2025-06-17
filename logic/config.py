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
RIFE_MODEL_PATH = None
# Image upscaler models directory
UPSCALE_MODELS_DIR = None
# Face restoration paths
CODEFORMER_BASE_PATH = None
FACE_RESTORATION_MODELS_DIR = None

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
DEFAULT_COLOR_FIX_METHOD = "Wavelet"

# Performance & VRAM
DEFAULT_MAX_CHUNK_LEN = 32
DEFAULT_VAE_CHUNK = 3

# Chunk Optimization
DEFAULT_ENABLE_CHUNK_OPTIMIZATION = True
DEFAULT_CHUNK_OPTIMIZATION_MIN_RATIO = 0.5

# Target Resolution
DEFAULT_ENABLE_TARGET_RES = True
DEFAULT_TARGET_RES_MODE = "Downscale then 4x"
DEFAULT_TARGET_H = 512
DEFAULT_TARGET_W = 512

# Context Window (replaces Sliding Window)
DEFAULT_ENABLE_CONTEXT_WINDOW = False
DEFAULT_CONTEXT_OVERLAP = 8

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
DEFAULT_SAVE_CHUNK_FRAMES = True

# Comparison Video
DEFAULT_CREATE_COMPARISON_VIDEO = True

# Seeding
DEFAULT_SEED = 99
DEFAULT_RANDOM_SEED = False

# RIFE Interpolation Settings
DEFAULT_RIFE_ENABLE_INTERPOLATION = False
DEFAULT_RIFE_MULTIPLIER = 2
DEFAULT_RIFE_FP16 = True
DEFAULT_RIFE_UHD = False
DEFAULT_RIFE_SCALE = 1.0
DEFAULT_RIFE_SKIP_STATIC = False
DEFAULT_RIFE_ENABLE_FPS_LIMIT = False
DEFAULT_RIFE_MAX_FPS_LIMIT = 60
DEFAULT_RIFE_APPLY_TO_CHUNKS = True
DEFAULT_RIFE_APPLY_TO_SCENES = True
DEFAULT_RIFE_KEEP_ORIGINAL = True
DEFAULT_RIFE_OVERWRITE_ORIGINAL = False
DEFAULT_RIFE_SKIP_EXISTING = True
DEFAULT_RIFE_INCLUDE_SUBFOLDERS = True

# FPS Decrease Settings
DEFAULT_ENABLE_FPS_DECREASE = False
DEFAULT_FPS_DECREASE_MODE = "multiplier"  # "multiplier" or "fixed"
DEFAULT_FPS_MULTIPLIER = 0.5  # Default to 1/2x (half FPS)
DEFAULT_TARGET_FPS = 24.0  # For fixed mode backward compatibility
DEFAULT_FPS_INTERPOLATION_METHOD = "blend"

# Batch Processing Options
DEFAULT_BATCH_SKIP_EXISTING = True
DEFAULT_BATCH_SAVE_CAPTIONS = True
DEFAULT_BATCH_USE_PROMPT_FILES = True

# Image Upscaler Settings
DEFAULT_ENABLE_IMAGE_UPSCALER = False
DEFAULT_IMAGE_UPSCALER_MODEL = None  # Will be set to first available model
DEFAULT_IMAGE_UPSCALER_BATCH_SIZE = 1
DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE = 1
DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE = 1000
DEFAULT_IMAGE_UPSCALER_DEVICE = "cuda"  # "cuda" or "cpu"
DEFAULT_IMAGE_UPSCALER_CACHE_MODELS = True

# Face Restoration Settings (CodeFormer)
DEFAULT_ENABLE_FACE_RESTORATION = False
DEFAULT_FACE_RESTORATION_FIDELITY = 0.7  # 0.0-1.0, higher values preserve identity better
DEFAULT_ENABLE_FACE_COLORIZATION = False
DEFAULT_FACE_RESTORATION_TIMING = "after_upscale"  # "before_upscale", "after_upscale", "per_frame"
DEFAULT_FACE_RESTORATION_WHEN = "after"  # "before", "after" (relative to upscaling process)
DEFAULT_CODEFORMER_MODEL = None  # Will be set to first available model
DEFAULT_FACE_RESTORATION_BATCH_SIZE = 4  # Number of frames to process simultaneously
DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE = 1
DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE = 32

# --- Initialization functions ---
def initialize_paths_and_prompts(base_path_from_app, outputs_folder_from_args, star_cfg_from_app):
    global APP_BASE_PATH, DEFAULT_OUTPUT_DIR, COG_VLM_MODEL_PATH, LIGHT_DEG_MODEL_PATH, HEAVY_DEG_MODEL_PATH, RIFE_MODEL_PATH
    global DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT, UPSCALE_MODELS_DIR, CODEFORMER_BASE_PATH, FACE_RESTORATION_MODELS_DIR

    APP_BASE_PATH = base_path_from_app
    DEFAULT_OUTPUT_DIR = os.path.abspath(outputs_folder_from_args)
    COG_VLM_MODEL_PATH = os.path.join(APP_BASE_PATH, 'models', 'cogvlm2-video-llama3-chat')
    LIGHT_DEG_MODEL_PATH = os.path.join(APP_BASE_PATH, 'pretrained_weight', 'light_deg.pt')
    HEAVY_DEG_MODEL_PATH = os.path.join(APP_BASE_PATH, 'pretrained_weight', 'heavy_deg.pt')
    RIFE_MODEL_PATH = os.path.join(APP_BASE_PATH, '..', 'Practical-RIFE', 'train_log')
    # Initialize image upscaler models directory
    UPSCALE_MODELS_DIR = os.path.join(APP_BASE_PATH, 'upscale_models')
    # Initialize face restoration paths
    CODEFORMER_BASE_PATH = os.path.join(os.path.dirname(APP_BASE_PATH), 'CodeFormer_STAR')
    FACE_RESTORATION_MODELS_DIR = os.path.join(APP_BASE_PATH, 'pretrained_weight')

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

def get_image_upscaler_settings():
    """
    Get the current image upscaler configuration settings.
    
    Returns:
        Dict containing image upscaler configuration
    """
    return {
        'enabled': DEFAULT_ENABLE_IMAGE_UPSCALER,
        'model': DEFAULT_IMAGE_UPSCALER_MODEL,
        'batch_size': DEFAULT_IMAGE_UPSCALER_BATCH_SIZE,
        'min_batch_size': DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE,
        'max_batch_size': DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE,
        'device': DEFAULT_IMAGE_UPSCALER_DEVICE,
        'cache_models': DEFAULT_IMAGE_UPSCALER_CACHE_MODELS,
        'models_dir': UPSCALE_MODELS_DIR
    }

def validate_image_upscaler_batch_size(batch_size):
    """
    Validate and clamp image upscaler batch size to allowed range.
    
    Args:
        batch_size: Requested batch size
        
    Returns:
        Validated batch size within allowed range
    """
    return max(DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE, 
               min(DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE, batch_size))

def get_face_restoration_settings():
    """
    Get the current face restoration configuration settings.
    
    Returns:
        Dict containing face restoration configuration
    """
    return {
        'enabled': DEFAULT_ENABLE_FACE_RESTORATION,
        'fidelity_weight': DEFAULT_FACE_RESTORATION_FIDELITY,
        'enable_colorization': DEFAULT_ENABLE_FACE_COLORIZATION,
        'timing': DEFAULT_FACE_RESTORATION_TIMING,
        'when': DEFAULT_FACE_RESTORATION_WHEN,
        'model': DEFAULT_CODEFORMER_MODEL,
        'batch_size': DEFAULT_FACE_RESTORATION_BATCH_SIZE,
        'min_batch_size': DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE,
        'max_batch_size': DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE,
        'codeformer_base_path': CODEFORMER_BASE_PATH,
        'models_dir': FACE_RESTORATION_MODELS_DIR
    }

def validate_face_restoration_batch_size(batch_size):
    """
    Validate and clamp face restoration batch size to allowed range.
    
    Args:
        batch_size: Requested batch size
        
    Returns:
        Validated batch size within allowed range
    """
    return max(DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE, 
               min(DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE, batch_size))

def validate_face_restoration_fidelity(fidelity_weight):
    """
    Validate and clamp face restoration fidelity weight to allowed range.
    
    Args:
        fidelity_weight: Requested fidelity weight (0.0-1.0)
        
    Returns:
        Validated fidelity weight within allowed range
    """
    return max(0.0, min(1.0, fidelity_weight))

# Expose UTIL_COG_VLM_AVAILABLE and UTIL_BITSANDBYTES_AVAILABLE from cogvlm_utils if they were imported successfully
# This makes them centrally accessible via this config module.
# The main app should ensure cogvlm_utils is in sys.path before importing config if direct import within config fails.

__all__ = [
    # Initialized values
    'APP_BASE_PATH', 'DEFAULT_OUTPUT_DIR', 'COG_VLM_MODEL_PATH', 'LIGHT_DEG_MODEL_PATH', 'HEAVY_DEG_MODEL_PATH', 'RIFE_MODEL_PATH', 'UPSCALE_MODELS_DIR',
    'CODEFORMER_BASE_PATH', 'FACE_RESTORATION_MODELS_DIR',
    'DEFAULT_POS_PROMPT', 'DEFAULT_NEG_PROMPT',
    # Direct imports/values
    'UTIL_COG_VLM_AVAILABLE', 'UTIL_BITSANDBYTES_AVAILABLE',
    'DEFAULT_MODEL_CHOICE', 'DEFAULT_UPSCALE_FACTOR', 'DEFAULT_CFG_SCALE', 'DEFAULT_SOLVER_MODE',
    'DEFAULT_DIFFUSION_STEPS_FAST', 'DEFAULT_DIFFUSION_STEPS_NORMAL', 'DEFAULT_COLOR_FIX_METHOD',
    'DEFAULT_MAX_CHUNK_LEN', 'DEFAULT_VAE_CHUNK',
    'DEFAULT_ENABLE_CHUNK_OPTIMIZATION', 'DEFAULT_CHUNK_OPTIMIZATION_MIN_RATIO',
    'DEFAULT_ENABLE_TARGET_RES', 'DEFAULT_TARGET_RES_MODE', 'DEFAULT_TARGET_H', 'DEFAULT_TARGET_W',
    'DEFAULT_ENABLE_CONTEXT_WINDOW', 'DEFAULT_CONTEXT_OVERLAP',
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
    'DEFAULT_SAVE_FRAMES', 'DEFAULT_SAVE_METADATA', 'DEFAULT_SAVE_CHUNKS', 'DEFAULT_SAVE_CHUNK_FRAMES',
    'DEFAULT_CREATE_COMPARISON_VIDEO',
    # Seed defaults
    'DEFAULT_SEED', 'DEFAULT_RANDOM_SEED',
    # RIFE defaults
    'DEFAULT_RIFE_ENABLE_INTERPOLATION', 'DEFAULT_RIFE_MULTIPLIER', 'DEFAULT_RIFE_FP16', 'DEFAULT_RIFE_UHD', 'DEFAULT_RIFE_SCALE', 'DEFAULT_RIFE_SKIP_STATIC',
    'DEFAULT_RIFE_ENABLE_FPS_LIMIT', 'DEFAULT_RIFE_MAX_FPS_LIMIT', 'DEFAULT_RIFE_APPLY_TO_CHUNKS', 'DEFAULT_RIFE_APPLY_TO_SCENES',
    'DEFAULT_RIFE_KEEP_ORIGINAL', 'DEFAULT_RIFE_OVERWRITE_ORIGINAL', 'DEFAULT_RIFE_SKIP_EXISTING', 'DEFAULT_RIFE_INCLUDE_SUBFOLDERS',
    # FPS decrease defaults
    'DEFAULT_ENABLE_FPS_DECREASE', 'DEFAULT_FPS_DECREASE_MODE', 'DEFAULT_FPS_MULTIPLIER', 'DEFAULT_TARGET_FPS', 'DEFAULT_FPS_INTERPOLATION_METHOD',
    # Batch processing defaults
    'DEFAULT_BATCH_SKIP_EXISTING', 'DEFAULT_BATCH_SAVE_CAPTIONS', 'DEFAULT_BATCH_USE_PROMPT_FILES',
    # Image upscaler defaults
    'DEFAULT_ENABLE_IMAGE_UPSCALER', 'DEFAULT_IMAGE_UPSCALER_MODEL', 'DEFAULT_IMAGE_UPSCALER_BATCH_SIZE',
    'DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE', 'DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE', 'DEFAULT_IMAGE_UPSCALER_DEVICE', 'DEFAULT_IMAGE_UPSCALER_CACHE_MODELS',
    # Face restoration defaults
    'DEFAULT_ENABLE_FACE_RESTORATION', 'DEFAULT_FACE_RESTORATION_FIDELITY', 'DEFAULT_ENABLE_FACE_COLORIZATION',
    'DEFAULT_FACE_RESTORATION_TIMING', 'DEFAULT_FACE_RESTORATION_WHEN', 'DEFAULT_CODEFORMER_MODEL',
    'DEFAULT_FACE_RESTORATION_BATCH_SIZE', 'DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE', 'DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE',
    # Functions
    'initialize_paths_and_prompts', 'get_cogvlm_quant_choices_map', 'get_default_cogvlm_quant_display',
    'get_image_upscaler_settings', 'validate_image_upscaler_batch_size',
    'get_face_restoration_settings', 'validate_face_restoration_batch_size', 'validate_face_restoration_fidelity'
] 