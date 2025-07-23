import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Availability Checks (from original config.py) ---
try:
    from .cogvlm_utils import COG_VLM_AVAILABLE as UTIL_COG_VLM_AVAILABLE
    from .cogvlm_utils import BITSANDBYTES_AVAILABLE as UTIL_BITSANDBYTES_AVAILABLE
except ImportError:
    UTIL_COG_VLM_AVAILABLE = False
    UTIL_BITSANDBYTES_AVAILABLE = False

# --- Default Constants (migrated from config.py) ---

# Prompts (will be overwritten by star_cfg if available)
DEFAULT_POS_PROMPT = "best quality, high quality, absurdres, 4k, 8k, highres"
DEFAULT_NEG_PROMPT = "worst quality, low quality, lowres, blurry, pixelated, noisy, artifacts, watermark, text, signature, logo"

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
DEFAULT_VAE_CHUNK = 1
DEFAULT_ENABLE_VRAM_OPTIMIZATION = True
DEFAULT_ENABLE_FP16_PROCESSING = True  # Use fp16 tensors to reduce VRAM usage by ~50%

# Chunk Optimization
DEFAULT_ENABLE_CHUNK_OPTIMIZATION = True

# Target Resolution
DEFAULT_ENABLE_TARGET_RES = True
DEFAULT_TARGET_RES_MODE = "Ratio Upscale"
DEFAULT_TARGET_H = 1024
DEFAULT_TARGET_W = 1024

# Auto-Resolution (Aspect Ratio Aware)
DEFAULT_ENABLE_AUTO_ASPECT_RESOLUTION = False
DEFAULT_AUTO_RESOLUTION_STATUS = "No video loaded"
DEFAULT_PIXEL_BUDGET = DEFAULT_TARGET_H * DEFAULT_TARGET_W  # 1,048,576 pixels (1024x1024)
DEFAULT_LAST_VIDEO_ASPECT_RATIO = 1.0
DEFAULT_AUTO_CALCULATED_H = DEFAULT_TARGET_H
DEFAULT_AUTO_CALCULATED_W = DEFAULT_TARGET_W

# Context Window
DEFAULT_ENABLE_CONTEXT_WINDOW = False
DEFAULT_CONTEXT_OVERLAP = 8

# Tiling
DEFAULT_ENABLE_TILING = False
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_OVERLAP = 64

# FFmpeg
DEFAULT_FFMPEG_USE_GPU = True
DEFAULT_FFMPEG_PRESET = "slower"
DEFAULT_FFMPEG_QUALITY_CPU = 12
DEFAULT_FFMPEG_QUALITY_GPU = 12

# Frame Folder Processing
DEFAULT_FRAME_FOLDER_FPS = 24.0

# Scene Splitting
DEFAULT_ENABLE_SCENE_SPLIT = True
DEFAULT_SCENE_SPLIT_MODE = "automatic"
DEFAULT_SCENE_MIN_SCENE_LEN = 0.6
DEFAULT_SCENE_THRESHOLD = 3.0
DEFAULT_SCENE_DROP_SHORT = False
DEFAULT_SCENE_MERGE_LAST = True
DEFAULT_SCENE_FRAME_SKIP = 0
DEFAULT_SCENE_MIN_CONTENT_VAL = 15.0
DEFAULT_SCENE_FRAME_WINDOW = 2
DEFAULT_SCENE_MANUAL_SPLIT_TYPE = "duration"
DEFAULT_SCENE_MANUAL_SPLIT_VALUE = 30.0
DEFAULT_SCENE_COPY_STREAMS = False
DEFAULT_SCENE_USE_MKVMERGE = True
DEFAULT_SCENE_RATE_FACTOR = 12
DEFAULT_SCENE_ENCODING_PRESET = "slower"
DEFAULT_SCENE_QUIET_FFMPEG = True

# CogVLM
DEFAULT_COGVLM_QUANT_DISPLAY_FP16 = "FP16/BF16"
DEFAULT_COGVLM_QUANT_DISPLAY_INT4 = "INT4 (CUDA)"
DEFAULT_COGVLM_UNLOAD_AFTER_USE = "full"
DEFAULT_AUTO_CAPTION_THEN_UPSCALE = True

# Output Options
DEFAULT_SAVE_FRAMES = True
DEFAULT_SAVE_METADATA = True
DEFAULT_SAVE_CHUNKS = True
DEFAULT_SAVE_CHUNK_FRAMES = True
DEFAULT_CREATE_COMPARISON_VIDEO = True

# Seeding
DEFAULT_SEED = 99
DEFAULT_RANDOM_SEED = False

# RIFE Interpolation
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

# FPS Decrease
DEFAULT_ENABLE_FPS_DECREASE = False
DEFAULT_FPS_DECREASE_MODE = "multiplier"
DEFAULT_FPS_MULTIPLIER = 0.5
DEFAULT_TARGET_FPS = 24.0
DEFAULT_FPS_INTERPOLATION_METHOD = "blend"

# Batch Processing
DEFAULT_BATCH_SKIP_EXISTING = True
DEFAULT_BATCH_SAVE_CAPTIONS = True
DEFAULT_BATCH_USE_PROMPT_FILES = True

# Image Upscaler
DEFAULT_ENABLE_IMAGE_UPSCALER = True  # Changed to True by default
DEFAULT_IMAGE_UPSCALER_MODEL = "2xLiveActionV1_SPAN_490000.pth"  # Set default model
DEFAULT_IMAGE_UPSCALER_BATCH_SIZE = 1
DEFAULT_IMAGE_UPSCALER_MIN_BATCH_SIZE = 1
DEFAULT_IMAGE_UPSCALER_MAX_BATCH_SIZE = 50
DEFAULT_IMAGE_UPSCALER_DEVICE = "cuda"
DEFAULT_IMAGE_UPSCALER_CACHE_MODELS = True

# Upscaler Type Selection
DEFAULT_UPSCALER_TYPE = "image_upscaler"  # Options: "star", "image_upscaler", "seedvr2"

# SeedVR2 Upscaler - now controlled by upscaler_type setting
DEFAULT_ENABLE_SEEDVR2 = False
DEFAULT_SEEDVR2_MODEL = None
DEFAULT_SEEDVR2_BATCH_SIZE = 5  # Minimum 5 for temporal consistency
DEFAULT_SEEDVR2_QUALITY_PRESET = "quality"  # Options: "fast", "balanced", "quality"
DEFAULT_SEEDVR2_USE_GPU = True

# SeedVR2 Advanced Settings
DEFAULT_SEEDVR2_PRESERVE_VRAM = True
DEFAULT_SEEDVR2_FLASH_ATTENTION = True
DEFAULT_SEEDVR2_COLOR_CORRECTION = True  # Wavelet reconstruction
DEFAULT_SEEDVR2_ENABLE_MULTI_GPU = False
DEFAULT_SEEDVR2_GPU_DEVICES = "0"  # Comma-separated list of GPU indices

# SeedVR2 Block Swap Configuration
DEFAULT_SEEDVR2_ENABLE_BLOCK_SWAP = False
DEFAULT_SEEDVR2_BLOCK_SWAP_COUNTER = 0  # 0 = disabled, 1-12+ = number of blocks to swap
DEFAULT_SEEDVR2_BLOCK_SWAP_OFFLOAD_IO = False  # I/O component offloading
DEFAULT_SEEDVR2_BLOCK_SWAP_MODEL_CACHING = False  # Keep model cached in RAM between runs

# SeedVR2 Temporal Consistency Settings
DEFAULT_SEEDVR2_TEMPORAL_OVERLAP = 2  # Frames of overlap between temporal chunks (minimum 2 for consistency)
DEFAULT_SEEDVR2_SCENE_AWARENESS = True  # Enable scene-aware temporal processing
DEFAULT_SEEDVR2_TEMPORAL_QUALITY = "balanced"  # fast, balanced, quality
DEFAULT_SEEDVR2_CONSISTENCY_VALIDATION = True  # Enable temporal consistency validation
DEFAULT_SEEDVR2_CHUNK_OPTIMIZATION = True  # Enable chunk boundary optimization
DEFAULT_SEEDVR2_ENABLE_TEMPORAL_CONSISTENCY = True  # Requires batch_size >= 5

# SeedVR2 Frame Processing
DEFAULT_SEEDVR2_ENABLE_FRAME_PADDING = True  # Automatic frame padding like STAR
DEFAULT_SEEDVR2_PAD_LAST_CHUNK = True  # Pad last chunk to same size as batch_size
DEFAULT_SEEDVR2_SKIP_FIRST_FRAMES = 0  # Number of frames to skip at start

# SeedVR2 Chunk Preview Configuration (same as STAR for consistency)
DEFAULT_SEEDVR2_ENABLE_CHUNK_PREVIEW = True  # Enable chunk preview by default 
DEFAULT_SEEDVR2_CHUNK_PREVIEW_FRAMES = 25  # Show last 125 frames as default
DEFAULT_SEEDVR2_KEEP_LAST_CHUNKS = 5  # Keep last 5 chunk videos in chunks folder

# SeedVR2 Model Configuration
DEFAULT_SEEDVR2_MODEL_PRECISION = "auto"  # Options: "auto", "fp16", "fp8"
DEFAULT_SEEDVR2_CFG_SCALE = 1.0  # Guidance scale for generation
DEFAULT_SEEDVR2_SEED = -1  # Use global seed if -1

# SeedVR2 Resolution Configuration
DEFAULT_SEEDVR2_UPSCALE_FACTOR = 2.0  # Default 2x upscale factor for SeedVR2
DEFAULT_SEEDVR2_DEFAULT_RESOLUTION = 1072  # Safe default resolution
DEFAULT_SEEDVR2_MIN_RESOLUTION = 256  # Minimum allowed resolution
DEFAULT_SEEDVR2_MAX_RESOLUTION = 4096  # Maximum allowed resolution


# Single Image Upscaling (SeedVR2)
DEFAULT_ENABLE_SINGLE_IMAGE_UPSCALE = False
DEFAULT_SINGLE_IMAGE_INPUT_PATH = None
DEFAULT_SINGLE_IMAGE_OUTPUT_FORMAT = "PNG"  # Options: "PNG", "JPEG", "WEBP"
DEFAULT_SINGLE_IMAGE_PRESERVE_METADATA = True
DEFAULT_SINGLE_IMAGE_CREATE_COMPARISON = True
DEFAULT_SINGLE_IMAGE_UPSCALER_TYPE = "Use SeedVR2 for Images"  # Options: "Use SeedVR2 for Images", "Use Image Based Upscalers"
DEFAULT_SINGLE_IMAGE_PRESERVE_ASPECT_RATIO = True
DEFAULT_SINGLE_IMAGE_QUALITY_LEVEL = 95  # Quality for JPEG/WEBP (70-100)
DEFAULT_SINGLE_IMAGE_CUSTOM_SUFFIX = "_upscaled"

# Face Restoration
DEFAULT_ENABLE_FACE_RESTORATION = False
DEFAULT_FACE_RESTORATION_FIDELITY = 0.7
DEFAULT_ENABLE_FACE_COLORIZATION = False
DEFAULT_FACE_RESTORATION_WHEN = "after"
DEFAULT_CODEFORMER_MODEL = None
DEFAULT_FACE_RESTORATION_BATCH_SIZE = 1
DEFAULT_FACE_RESTORATION_MIN_BATCH_SIZE = 1
DEFAULT_FACE_RESTORATION_MAX_BATCH_SIZE = 50

# Default constants for GPU configuration
DEFAULT_GPU_DEVICE = "0"

# Video comparison thresholds
DEFAULT_VIDEO_COUNT_THRESHOLD_3 = 3
DEFAULT_VIDEO_COUNT_THRESHOLD_4 = 4

# Progress mapping constants
DEFAULT_PROGRESS_OFFSET = 0.2
DEFAULT_PROGRESS_SCALE = 0.7
DEFAULT_BATCH_PROGRESS_OFFSET = 0.1
DEFAULT_BATCH_PROGRESS_SCALE = 0.8

# Preset system constants
DEFAULT_PRESET_LOAD_RETRIES = 3
DEFAULT_PRESET_SAVE_DELAY = 0.1
DEFAULT_PRESET_RETRY_DELAY = 0.2
DEFAULT_PRESET_LOAD_DELAY = 0.1
DEFAULT_CONDITIONAL_UPDATES_COUNT = 20  # Matches the number of conditional updates in load_preset_wrapper

# Standalone Face Restoration Settings
DEFAULT_STANDALONE_ENABLE_FACE_RESTORATION = True
DEFAULT_STANDALONE_FACE_RESTORATION_FIDELITY = 0.7
DEFAULT_STANDALONE_ENABLE_FACE_COLORIZATION = False
DEFAULT_STANDALONE_FACE_RESTORATION_BATCH_SIZE = 4
DEFAULT_STANDALONE_SAVE_FRAMES = False
DEFAULT_STANDALONE_CREATE_COMPARISON = True
DEFAULT_STANDALONE_PRESERVE_AUDIO = True

# Video Editing Settings
DEFAULT_PRECISE_CUTTING_MODE = "precise"
DEFAULT_PREVIEW_FIRST_SEGMENT = True

# Manual Comparison Settings
DEFAULT_MANUAL_VIDEO_COUNT = 2

# --- Dataclasses ---

@dataclass
class PathConfig:
    app_base_path: str = ""
    outputs_dir: str = "outputs"
    cogvlm_model_path: str = ""
    light_deg_model_path: str = ""
    heavy_deg_model_path: str = ""
    rife_model_path: str = ""
    upscale_models_dir: str = ""
    codeformer_base_path: str = ""
    face_restoration_models_dir: str = ""

@dataclass
class PromptConfig:
    user: str = ""
    positive: str = DEFAULT_POS_PROMPT
    negative: str = DEFAULT_NEG_PROMPT

@dataclass
class StarModelConfig:
    model_choice: str = DEFAULT_MODEL_CHOICE
    cfg_scale: float = DEFAULT_CFG_SCALE
    solver_mode: str = DEFAULT_SOLVER_MODE
    steps: int = DEFAULT_DIFFUSION_STEPS_FAST
    color_fix_method: str = DEFAULT_COLOR_FIX_METHOD

@dataclass
class PerformanceConfig:
    max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN
    vae_chunk: int = DEFAULT_VAE_CHUNK
    enable_chunk_optimization: bool = DEFAULT_ENABLE_CHUNK_OPTIMIZATION
    enable_vram_optimization: bool = DEFAULT_ENABLE_VRAM_OPTIMIZATION
    enable_fp16_processing: bool = DEFAULT_ENABLE_FP16_PROCESSING

@dataclass
class ResolutionConfig:
    enable_target_res: bool = DEFAULT_ENABLE_TARGET_RES
    target_res_mode: str = DEFAULT_TARGET_RES_MODE
    target_h: int = DEFAULT_TARGET_H
    target_w: int = DEFAULT_TARGET_W
    upscale_factor: float = DEFAULT_UPSCALE_FACTOR
    enable_auto_aspect_resolution: bool = DEFAULT_ENABLE_AUTO_ASPECT_RESOLUTION
    auto_resolution_status: str = DEFAULT_AUTO_RESOLUTION_STATUS
    pixel_budget: int = DEFAULT_PIXEL_BUDGET
    last_video_aspect_ratio: float = DEFAULT_LAST_VIDEO_ASPECT_RATIO
    auto_calculated_h: int = DEFAULT_AUTO_CALCULATED_H
    auto_calculated_w: int = DEFAULT_AUTO_CALCULATED_W

@dataclass
class ContextWindowConfig:
    enable: bool = DEFAULT_ENABLE_CONTEXT_WINDOW
    overlap: int = DEFAULT_CONTEXT_OVERLAP

@dataclass
class TilingConfig:
    enable: bool = DEFAULT_ENABLE_TILING
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP

@dataclass
class FfmpegConfig:
    use_gpu: bool = DEFAULT_FFMPEG_USE_GPU
    preset: str = DEFAULT_FFMPEG_PRESET
    quality: int = DEFAULT_FFMPEG_QUALITY_CPU

@dataclass
class FrameFolderConfig:
    enable: bool = False
    input_path: str = ""
    fps: float = DEFAULT_FRAME_FOLDER_FPS

@dataclass
class SceneSplitConfig:
    enable: bool = DEFAULT_ENABLE_SCENE_SPLIT
    mode: str = DEFAULT_SCENE_SPLIT_MODE
    min_scene_len: float = DEFAULT_SCENE_MIN_SCENE_LEN
    threshold: float = DEFAULT_SCENE_THRESHOLD
    drop_short: bool = DEFAULT_SCENE_DROP_SHORT
    merge_last: bool = DEFAULT_SCENE_MERGE_LAST
    frame_skip: int = DEFAULT_SCENE_FRAME_SKIP
    min_content_val: float = DEFAULT_SCENE_MIN_CONTENT_VAL
    frame_window: int = DEFAULT_SCENE_FRAME_WINDOW
    manual_split_type: str = DEFAULT_SCENE_MANUAL_SPLIT_TYPE
    manual_split_value: float = DEFAULT_SCENE_MANUAL_SPLIT_VALUE
    copy_streams: bool = DEFAULT_SCENE_COPY_STREAMS
    use_mkvmerge: bool = DEFAULT_SCENE_USE_MKVMERGE
    rate_factor: int = DEFAULT_SCENE_RATE_FACTOR
    encoding_preset: str = DEFAULT_SCENE_ENCODING_PRESET
    quiet_ffmpeg: bool = DEFAULT_SCENE_QUIET_FFMPEG

@dataclass
class CogVLMConfig:
    quant_display: str = DEFAULT_COGVLM_QUANT_DISPLAY_INT4 if UTIL_BITSANDBYTES_AVAILABLE else DEFAULT_COGVLM_QUANT_DISPLAY_FP16
    unload_after_use: str = DEFAULT_COGVLM_UNLOAD_AFTER_USE
    auto_caption_then_upscale: bool = DEFAULT_AUTO_CAPTION_THEN_UPSCALE
    enable_auto_caption_per_scene: bool = False
    quant_value: int = 4 # Default to INT4 if available

@dataclass
class OutputConfig:
    save_frames: bool = DEFAULT_SAVE_FRAMES
    save_metadata: bool = DEFAULT_SAVE_METADATA
    save_chunks: bool = DEFAULT_SAVE_CHUNKS
    save_chunk_frames: bool = DEFAULT_SAVE_CHUNK_FRAMES
    create_comparison_video: bool = DEFAULT_CREATE_COMPARISON_VIDEO

@dataclass
class SeedConfig:
    seed: int = DEFAULT_SEED
    use_random: bool = DEFAULT_RANDOM_SEED

@dataclass
class RifeConfig:
    enable: bool = DEFAULT_RIFE_ENABLE_INTERPOLATION
    multiplier: int = DEFAULT_RIFE_MULTIPLIER
    fp16: bool = DEFAULT_RIFE_FP16
    uhd: bool = DEFAULT_RIFE_UHD
    scale: float = DEFAULT_RIFE_SCALE
    skip_static: bool = DEFAULT_RIFE_SKIP_STATIC
    enable_fps_limit: bool = DEFAULT_RIFE_ENABLE_FPS_LIMIT
    max_fps_limit: int = DEFAULT_RIFE_MAX_FPS_LIMIT
    apply_to_chunks: bool = DEFAULT_RIFE_APPLY_TO_CHUNKS
    apply_to_scenes: bool = DEFAULT_RIFE_APPLY_TO_SCENES
    keep_original: bool = DEFAULT_RIFE_KEEP_ORIGINAL
    overwrite_original: bool = DEFAULT_RIFE_OVERWRITE_ORIGINAL

@dataclass
class FpsDecreaseConfig:
    enable: bool = DEFAULT_ENABLE_FPS_DECREASE
    mode: str = DEFAULT_FPS_DECREASE_MODE
    multiplier_preset: str = "1/2x (Half FPS)"
    multiplier_custom: float = DEFAULT_FPS_MULTIPLIER
    target_fps: float = DEFAULT_TARGET_FPS
    interpolation_method: str = DEFAULT_FPS_INTERPOLATION_METHOD

@dataclass
class BatchConfig:
    enable: bool = False
    input_folder: str = ""
    output_folder: str = ""
    skip_existing: bool = DEFAULT_BATCH_SKIP_EXISTING
    save_captions: bool = DEFAULT_BATCH_SAVE_CAPTIONS
    use_prompt_files: bool = DEFAULT_BATCH_USE_PROMPT_FILES
    enable_auto_caption: bool = True
    enable_frame_folders: bool = False
    enable_direct_image_upscaling: bool = False  # NEW: Process individual images directly with image upscaler
    original_filename: Optional[str] = None

@dataclass
class ImageUpscalerConfig:
    enable: bool = DEFAULT_ENABLE_IMAGE_UPSCALER
    model: Optional[str] = DEFAULT_IMAGE_UPSCALER_MODEL
    batch_size: int = DEFAULT_IMAGE_UPSCALER_BATCH_SIZE
    device: str = DEFAULT_IMAGE_UPSCALER_DEVICE
    cache_models: bool = DEFAULT_IMAGE_UPSCALER_CACHE_MODELS

@dataclass
class FaceRestorationConfig:
    enable: bool = DEFAULT_ENABLE_FACE_RESTORATION
    fidelity_weight: float = DEFAULT_FACE_RESTORATION_FIDELITY
    enable_colorization: bool = DEFAULT_ENABLE_FACE_COLORIZATION
    when: str = DEFAULT_FACE_RESTORATION_WHEN
    model: Optional[str] = DEFAULT_CODEFORMER_MODEL
    batch_size: int = DEFAULT_FACE_RESTORATION_BATCH_SIZE

@dataclass
class GpuConfig:
    device: str = DEFAULT_GPU_DEVICE

@dataclass
class UpscalerTypeConfig:
    """Configuration for upscaler type selection."""
    upscaler_type: str = DEFAULT_UPSCALER_TYPE  # "star", "image_upscaler", "seedvr2"

@dataclass
class SeedVR2Config:
    """Configuration for SeedVR2 upscaler with advanced features.
    Note: 'enable' is now controlled by the upscaler_type setting in Core Settings."""
    # Basic settings (enable is derived from upscaler_type == "seedvr2")
    enable: bool = DEFAULT_ENABLE_SEEDVR2
    model: Optional[str] = DEFAULT_SEEDVR2_MODEL
    batch_size: int = DEFAULT_SEEDVR2_BATCH_SIZE
    quality_preset: str = DEFAULT_SEEDVR2_QUALITY_PRESET
    use_gpu: bool = DEFAULT_SEEDVR2_USE_GPU
    
    # Advanced settings
    preserve_vram: bool = DEFAULT_SEEDVR2_PRESERVE_VRAM
    flash_attention: bool = DEFAULT_SEEDVR2_FLASH_ATTENTION
    color_correction: bool = DEFAULT_SEEDVR2_COLOR_CORRECTION
    enable_multi_gpu: bool = DEFAULT_SEEDVR2_ENABLE_MULTI_GPU
    gpu_devices: str = DEFAULT_SEEDVR2_GPU_DEVICES
    
    # Block swap configuration
    enable_block_swap: bool = DEFAULT_SEEDVR2_ENABLE_BLOCK_SWAP
    block_swap_counter: int = DEFAULT_SEEDVR2_BLOCK_SWAP_COUNTER
    block_swap_offload_io: bool = DEFAULT_SEEDVR2_BLOCK_SWAP_OFFLOAD_IO
    block_swap_model_caching: bool = DEFAULT_SEEDVR2_BLOCK_SWAP_MODEL_CACHING
    
    # Temporal consistency settings
    temporal_overlap: int = DEFAULT_SEEDVR2_TEMPORAL_OVERLAP
    scene_awareness: bool = DEFAULT_SEEDVR2_SCENE_AWARENESS
    temporal_quality: str = DEFAULT_SEEDVR2_TEMPORAL_QUALITY
    consistency_validation: bool = DEFAULT_SEEDVR2_CONSISTENCY_VALIDATION
    chunk_optimization: bool = DEFAULT_SEEDVR2_CHUNK_OPTIMIZATION
    enable_temporal_consistency: bool = DEFAULT_SEEDVR2_ENABLE_TEMPORAL_CONSISTENCY
    
    # Frame processing
    enable_frame_padding: bool = DEFAULT_SEEDVR2_ENABLE_FRAME_PADDING
    pad_last_chunk: bool = DEFAULT_SEEDVR2_PAD_LAST_CHUNK
    skip_first_frames: int = DEFAULT_SEEDVR2_SKIP_FIRST_FRAMES
    
    # Chunk preview configuration
    enable_chunk_preview: bool = DEFAULT_SEEDVR2_ENABLE_CHUNK_PREVIEW
    chunk_preview_frames: int = DEFAULT_SEEDVR2_CHUNK_PREVIEW_FRAMES
    keep_last_chunks: int = DEFAULT_SEEDVR2_KEEP_LAST_CHUNKS
    
    # Model configuration
    model_precision: str = DEFAULT_SEEDVR2_MODEL_PRECISION
    cfg_scale: float = DEFAULT_SEEDVR2_CFG_SCALE
    seed: int = DEFAULT_SEEDVR2_SEED

@dataclass
class SingleImageUpscaleConfig:
    """Configuration for single image upscaling feature."""
    enable: bool = DEFAULT_ENABLE_SINGLE_IMAGE_UPSCALE
    input_path: Optional[str] = DEFAULT_SINGLE_IMAGE_INPUT_PATH
    output_format: str = DEFAULT_SINGLE_IMAGE_OUTPUT_FORMAT
    preserve_metadata: bool = DEFAULT_SINGLE_IMAGE_PRESERVE_METADATA
    create_comparison: bool = DEFAULT_SINGLE_IMAGE_CREATE_COMPARISON
    upscaler_type: str = DEFAULT_SINGLE_IMAGE_UPSCALER_TYPE
    preserve_aspect_ratio: bool = DEFAULT_SINGLE_IMAGE_PRESERVE_ASPECT_RATIO
    quality_level: int = DEFAULT_SINGLE_IMAGE_QUALITY_LEVEL
    custom_suffix: str = DEFAULT_SINGLE_IMAGE_CUSTOM_SUFFIX

@dataclass
class StandaloneFaceRestorationConfig:
    """Configuration for standalone face restoration settings."""
    enable: bool = DEFAULT_STANDALONE_ENABLE_FACE_RESTORATION
    fidelity_weight: float = DEFAULT_STANDALONE_FACE_RESTORATION_FIDELITY
    enable_colorization: bool = DEFAULT_STANDALONE_ENABLE_FACE_COLORIZATION
    batch_size: int = DEFAULT_STANDALONE_FACE_RESTORATION_BATCH_SIZE
    save_frames: bool = DEFAULT_STANDALONE_SAVE_FRAMES
    create_comparison: bool = DEFAULT_STANDALONE_CREATE_COMPARISON
    preserve_audio: bool = DEFAULT_STANDALONE_PRESERVE_AUDIO
    input_video: Optional[str] = None
    mode: str = "Single Video"
    batch_input_folder: str = ""
    batch_output_folder: str = ""
    codeformer_model: Optional[str] = None

@dataclass
class VideoEditingConfig:
    """Configuration for video editing settings."""
    precise_cutting_mode: str = DEFAULT_PRECISE_CUTTING_MODE
    preview_first_segment: bool = DEFAULT_PREVIEW_FIRST_SEGMENT
    # Note: Video editing components like ranges_input are handled separately
    # as they are complex objects that don't fit well in the preset system

@dataclass
class PresetSystemConfig:
    """Configuration for preset system settings."""
    load_retries: int = DEFAULT_PRESET_LOAD_RETRIES
    save_delay: float = DEFAULT_PRESET_SAVE_DELAY
    retry_delay: float = DEFAULT_PRESET_RETRY_DELAY
    load_delay: float = DEFAULT_PRESET_LOAD_DELAY
    conditional_updates_count: int = DEFAULT_CONDITIONAL_UPDATES_COUNT

@dataclass
class ManualComparisonConfig:
    """Configuration for manual comparison settings."""
    video_count: int = DEFAULT_MANUAL_VIDEO_COUNT
    original_video: Optional[str] = None
    upscaled_video: Optional[str] = None
    third_video: Optional[str] = None
    fourth_video: Optional[str] = None
    layout: str = "auto"

@dataclass
class AppConfig:
    input_video_path: Optional[str] = None
    paths: PathConfig = field(default_factory=PathConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    star_model: StarModelConfig = field(default_factory=StarModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    context_window: ContextWindowConfig = field(default_factory=ContextWindowConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    frame_folder: FrameFolderConfig = field(default_factory=FrameFolderConfig)
    scene_split: SceneSplitConfig = field(default_factory=SceneSplitConfig)
    cogvlm: CogVLMConfig = field(default_factory=CogVLMConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    rife: RifeConfig = field(default_factory=RifeConfig)
    fps_decrease: FpsDecreaseConfig = field(default_factory=FpsDecreaseConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    image_upscaler: ImageUpscalerConfig = field(default_factory=ImageUpscalerConfig)
    face_restoration: FaceRestorationConfig = field(default_factory=FaceRestorationConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    upscaler_type: UpscalerTypeConfig = field(default_factory=UpscalerTypeConfig)
    seedvr2: SeedVR2Config = field(default_factory=SeedVR2Config)
    single_image_upscale: SingleImageUpscaleConfig = field(default_factory=SingleImageUpscaleConfig)
    standalone_face_restoration: StandaloneFaceRestorationConfig = field(default_factory=StandaloneFaceRestorationConfig)
    video_editing: VideoEditingConfig = field(default_factory=VideoEditingConfig)
    manual_comparison: ManualComparisonConfig = field(default_factory=ManualComparisonConfig)
    preset_system: PresetSystemConfig = field(default_factory=PresetSystemConfig)

def create_app_config(base_path: str, outputs_folder: str, star_cfg: Optional[Any]) -> AppConfig:
    """Factory function to create and initialize the main AppConfig object."""
    
    path_config = PathConfig(
        app_base_path=base_path,
        outputs_dir=os.path.abspath(outputs_folder),
        cogvlm_model_path=os.path.join(base_path, 'models', 'CogVLMFixed'),
        light_deg_model_path=os.path.join(base_path, 'pretrained_weight', 'light_deg.pt'),
        heavy_deg_model_path=os.path.join(base_path, 'pretrained_weight', 'heavy_deg.pt'),
        rife_model_path=os.path.join(base_path, '..', 'Practical-RIFE', 'train_log'),
        upscale_models_dir=os.path.join(base_path, 'upscale_models'),
        codeformer_base_path=os.path.join(os.path.dirname(base_path), 'CodeFormer_STAR'),
        face_restoration_models_dir=os.path.join(base_path, 'pretrained_weight')
    )

    prompt_config = PromptConfig()
    if star_cfg:
        prompt_config.positive = star_cfg.positive_prompt
        prompt_config.negative = star_cfg.negative_prompt
    else:
        print("Warning: star_cfg not provided to config factory. Using placeholder prompts.")

    # Create a complete AppConfig with all sections properly initialized
    app_config = AppConfig(
        paths=path_config,
        prompts=prompt_config
    )
    
    # Ensure all sections are properly initialized with default values
    # This is important for preset saving to include all sections
    if not hasattr(app_config, 'manual_comparison') or app_config.manual_comparison is None:
        app_config.manual_comparison = ManualComparisonConfig()
    
    if not hasattr(app_config, 'standalone_face_restoration') or app_config.standalone_face_restoration is None:
        app_config.standalone_face_restoration = StandaloneFaceRestorationConfig()
    
    if not hasattr(app_config, 'preset_system') or app_config.preset_system is None:
        app_config.preset_system = PresetSystemConfig()
    
    if not hasattr(app_config, 'video_editing') or app_config.video_editing is None:
        app_config.video_editing = VideoEditingConfig()
    
    if not hasattr(app_config, 'single_image_upscale') or app_config.single_image_upscale is None:
        app_config.single_image_upscale = SingleImageUpscaleConfig()
    
    return app_config

def get_cogvlm_quant_choices_map(torch_cuda_available: bool, bitsandbytes_available: bool) -> Dict[int, str]:
    choices_map = {0: DEFAULT_COGVLM_QUANT_DISPLAY_FP16}
    if torch_cuda_available and bitsandbytes_available:
        choices_map[4] = DEFAULT_COGVLM_QUANT_DISPLAY_INT4
        choices_map[8] = "INT8 (CUDA)"
    return choices_map

def get_default_cogvlm_quant_display(cogvlm_quant_choices_map: Dict[int, str]) -> str:
    return DEFAULT_COGVLM_QUANT_DISPLAY_INT4 if 4 in cogvlm_quant_choices_map else DEFAULT_COGVLM_QUANT_DISPLAY_FP16
