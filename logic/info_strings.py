"""
Info Strings Module - Contains all UI info strings, help text, and markdown content for the STAR app.
This module centralizes all large text content to reduce file size and improve maintainability.
"""

# CSS Styles
APP_CSS = """
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

# UI Component Info Strings
UPSCALER_TYPE_INFO = """Choose your preferred upscaling method:
• SeedVR2: Advanced AI upscaling with superior quality and detail preservation
• Image Based: Fast traditional upscalers (RealESRGAN, ESRGAN, etc.)"""

TARGET_RESOLUTION_MODE_INFO = """How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio. Ratio upscale doesn't work for Image Based Upscaler and yields very poor results for STAR model.
'Downscale then 4x': For STAR models, downscales towards Target H/W ÷ 4, then applies 4x upscale. For image upscalers, adapts to model scale (e.g., 2x model = Target H/W ÷ 2). Can clean noisy high-res input before upscaling."""

AUTO_ASPECT_RESOLUTION_INFO = """Automatically calculate optimal resolution that maintains input video aspect ratio within the pixel budget (Target H × Target W).
- Triggered whenever you change video or process batch videos
- Maintains exact aspect ratio while maximizing quality within pixel limits
- Prevents manual resolution adjustment for different aspect ratios
- Example: 1024×1024 budget + 360×640 input → auto-sets to 720×1280 (maintains 9:16 ratio, uses 921,600 pixels)"""

CAPTION_UNLOAD_STRATEGY_INFO = """Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""

FACE_RESTORATION_TIMING_INFO = """When to apply face restoration:
'before': Apply before upscaling (may be enhanced further)
'after': Apply after upscaling (final enhancement)"""

CODEFORMER_MODEL_SELECTION_INFO = """Choose the model based on input video quality.
'Light Degradation': Better for relatively clean inputs (e.g., downloaded web videos).
'Heavy Degradation': Better for inputs with significant compression artifacts, noise, or blur."""

SOLVER_MODE_INFO = """Diffusion solver type.
'fast': Fewer steps (default ~15), much faster, good quality usually.
'normal': More steps (default ~50), slower, potentially slightly better detail/coherence."""

COLOR_CORRECTION_INFO = """Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""

MAX_CHUNK_LEN_INFO = """IMPORTANT for VRAM. This is the standard way the application manages VRAM. It divides the entire sequence of video frames into sequential, non-overlapping chunks.
- Mechanism: The STAR model processes one complete chunk (of this many frames) at a time.
- 32 Frames is best quality and uses a lot of VRAM. So reduce frame count if you get Out of Memory Error"""

CONTEXT_OVERLAP_INFO = """Process extra frames for last chunks to ensure optimal quality. When the last chunk is not equal to the user-set chunk size, this processes additional frames but only keeps the necessary output.
- Example: For 70 frames with 32-frame chunks, instead of processing only 6 frames for the last chunk (poor quality), it processes 32 frames (38-69) but keeps only the last 6 (64-69).
- Quality Impact: Ensures all chunks have optimal processing conditions by always processing the full user-set chunk size."""

VAE_DECODE_BATCH_SIZE_INFO = """Controls max latent frames decoded back to pixels by VAE simultaneouslyç
- No quality impact.
- Higher may yield faster decoding but uses more VRAM."""

TILING_INFO = """Processes each frame in small spatial patches (tiles).
- Speed Impact: Extremely Slow.
- I didn't find use case for this but still implemented"""

COMPARISON_VIDEO_INFO = """Create a side-by-side or top-bottom comparison video showing original vs upscaled quality.
The layout is automatically chosen based on aspect ratio to stay within 1920x1080 bounds when possible.
This helps visualize the quality improvement from upscaling."""

PROMPT_USER_INFO = """Describe the main subject and action in the video. This guides the upscaling process.
Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens (STAR Model)."""

PROMPT_POSITIVE_INFO = """Appended to your 'Describe Video Content' prompt. Focuses on desired quality aspects (e.g., realism, detail).
The total combined prompt length is limited to 77 tokens."""

PROMPT_NEGATIVE_INFO = """Guides the model *away* from undesired aspects (e.g., bad quality, artifacts, specific styles). This does NOT count towards the 77 token limit for positive guidance."""

ENHANCED_INPUT_INFO = """Enter path to either a video file (mp4, avi, mov, etc.) or folder containing image frames (jpg, png, tiff, etc.). Automatically detected - works on Windows and Linux."""

SCENE_SPLIT_INFO = """Split video into scenes and process each scene individually.
- Quality Impact: Better temporal consistency within scenes, and excellent quality with auto-captioning per scene.
- You can also cancel and use processed scenes later, saves each scene individually as well
- Not useful for Image Based Upscalers since no prompt or temporal consistency is utilized."""

SCENE_SPLIT_MODE_INFO = """'automatic': Uses scene detection algorithms to find natural scene boundaries.
'manual': Splits video at fixed intervals (duration or frame count)."""

TARGET_WIDTH_INFO = """Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""

TARGET_HEIGHT_INFO = """Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""

# Markdown Help Blocks
IMAGE_UPSCALER_OPTIMIZATION_TIPS = """
**💡 Optimization Tips:**
- **Batch Size**: Start with 2-4, increase if you have more VRAM
- **2x Models**: Significantly faster and uses lesser VRAM
- **4x Models**: Better for very low resolution videos to upscale like 960p to 3840p
- **Popular Models**: 
  - 2xLiveActionV1_SPAN_490000: Excellent for live-action video
  - RealESRGAN_x4plus: Good general-purpose 4x upscaler
  - DAT-2 series: High quality but slower
"""

VIDEO_EDITING_HELP = """
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
"""

FACE_RESTORATION_HELP = """
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
"""

# Model Info Display Templates
SEEDVR2_MODEL_INFO_NO_MODELS = """📥 No Models Found

To use SeedVR2, you need to place model files in:
└── SeedVR2/models/

📋 Supported Models:
• SeedVR2_3B_FP8.safetensors (Recommended - Fast & Efficient)
• SeedVR2_7B_FP16.safetensors (High Quality)
• Custom SeedVR2 models (.safetensors format)

💡 Recommended Models:
• 3B FP8: Best speed/VRAM balance (~6GB VRAM)
• 7B FP16: Highest quality (~12GB VRAM)

🔄 Click 'Refresh Models' after adding files"""

SEEDVR2_MODEL_INFO_ERROR_TEMPLATE = """❌ Error Detected

{error_message}

Please check the logs for more details and ensure:
• SeedVR2 is properly installed
• Required dependencies are available
• File permissions are correct"""

# Processing Status Messages
DEFAULT_STATUS_MESSAGES = {
    'ready_video': '🎞️ Ready to edit videos. Upload a video and specify cut ranges to begin.',
    'ready_face_restoration': '🎭 Ready for face restoration processing. Upload a video and configure settings to begin.',
    'ready_processing_stats': '📊 Processing statistics will appear here during face restoration.',
    'ready_image_processing': 'Ready to process images...',
    'ready_image_details': 'Upload an image to see details...',
    'ready_processing_log': 'Processing log will appear here...',
    'validate_input': 'Enter a video file path or frames folder path above to validate',
    'cut_analysis': '✏️ Enter ranges above to see cut analysis',
    'time_estimate': '📊 Upload video and enter ranges to see time estimate',
    'expected_resolution': '📹 Upload a video and configure settings to see expected output resolution',
}

# Processing Success Messages
PROCESSING_SUCCESS_TEMPLATES = {
    'image_processing_complete': """✅ Image processing completed successfully!

📁 Output: {output_filename}
📊 Details: {width}×{height} pixels
⏱️ Time: {processing_time:.2f} seconds
🔍 Upscale: {upscale_factor:.2f}x""",
    
    'temp_cleanup_success': "✅ Temp folder cleared. Freed {freed_gb:.2f} GB. Remaining: {remaining_label}",
    'temp_cleanup_error': "⚠️ Temp folder cleanup encountered errors. Check logs."
}

# Direct Image Upscaling Info Messages
DIRECT_IMAGE_UPSCALING_WARNING = "⚠️ Direct Image Upscaling enabled: Automatically uses Image Based Upscalers. Switch to 'Use Image Based Upscalers' in Core Settings for optimal results."

DIRECT_IMAGE_UPSCALING_SUCCESS = "✅ Direct Image Upscaling enabled: Will process JPG, PNG, etc. files directly with the selected image upscaler model."

DIRECT_IMAGE_UPSCALING_DEFAULT = "Process individual image files (JPG, PNG, etc.) directly with selected image upscaler model. Ideal for batch upscaling photos/images."

# Block Swap Status Messages
BLOCK_SWAP_DISABLED = "Block swap disabled"

BLOCK_SWAP_ENABLED_TEMPLATE = """🔄 Advanced Block Swap Enabled

{status_icon} {status_msg}
💾 Current Memory: {memory_info}

📊 Block Swap Configuration:
• Blocks to swap: {block_swap_counter}
• Estimated VRAM savings: ~{estimated_savings_gb:.1f}GB
• Expected performance impact: ~{estimated_performance_impact:.1f}%

💡 Real-time optimization active"""

BLOCK_SWAP_FALLBACK_TEMPLATE = """🔄 Block Swap Enabled: {block_swap_counter} blocks

📉 Estimated VRAM reduction: ~{estimated_savings}%
⚡ Performance impact: ~{performance_impact}% slower
💡 Recommended for VRAM-limited systems

⚠️ Real-time monitoring unavailable"""

# GPU Status Messages
NO_CUDA_GPUS_DETECTED = """❌ No CUDA GPUs detected"""

SEEDVR2_DEPENDENCIES_SUCCESS = "✅ SeedVR2 ready to use!\n🚀 All dependencies available"

SEEDVR2_DEPENDENCIES_MISSING_TEMPLATE = "❌ Missing dependencies:\n{missing_deps}"

# Error Messages
GENERAL_ERROR_TEMPLATES = {
    'model_info_error': "Error loading model info: {error}",
    'video_info_error': "❌ Error reading video: {error}",
    'validation_error': "❌ Validation Error: {error}",
    'processing_error': "❌ Error: {error}",
    'dependency_check_error': "❌ Dependency check failed: {error}",
    'initialization_error': "❌ Initialization failed: {error}",
    'no_results_error': "❌ No results returned from image processing"
}

# Preset Status Messages
PRESET_STATUS_LOADED = "✅ Loaded on startup: {preset_name}"
PRESET_STATUS_NO_PRESET = "⚠️ No preset loaded on startup - using defaults"

# Auto-Caption Status Messages
AUTO_CAPTION_DISABLED_IMAGE_UPSCALER = "Image-based upscaling is enabled. Auto-captioning is disabled as image upscalers don't use prompts."
AUTO_CAPTION_UNAVAILABLE = "Auto-captioning requested but CogVLM2 is not available. Using original prompt."

# Video Information Display Template
VIDEO_INFO_NO_VIDEO = "📹 Upload a video to see detailed information"

# Batch Processing Info
BATCH_SAVE_CAPTIONS_INFO = "Save auto-generated captions as filename.txt in the input folder for future reuse. Never overwrites existing prompt files."

BATCH_USE_PROMPT_FILES_INFO = "Look for text files with same name as video (e.g., video.txt) to use as custom prompts. Takes priority over user prompt and auto-caption."

BATCH_ENABLE_AUTO_CAPTION_INFO = "Generate automatic captions for videos that don't have prompt files. Uses the same CogVLM2 settings as Core Settings tab."

# RIFE Interpolation Info
RIFE_INTERPOLATION_INFO = "Enable AI-powered frame interpolation to increase video FPS. When enabled, RIFE will be applied to the final upscaled video and can optionally be applied to intermediate chunks and scenes."

RIFE_DESCRIPTION = "**RIFE (Real-time Intermediate Flow Estimation)** uses AI to intelligently generate intermediate frames between existing frames, increasing video smoothness and frame rate."

# SeedVR2 Specific Info
SEEDVR2_BATCH_SIZE_INFO = "Frames processed simultaneously. Min 5 for temporal consistency. Higher = better quality but more VRAM."

SEEDVR2_TEMPORAL_OVERLAP_INFO = "Frame overlap between batches for smoother transitions. Higher = smoother but slower."

SEEDVR2_DESCRIPTION = "**🚀 Real-time Temporal Consistency & Superior Quality**"

# Image Processing Info
IMAGE_PRESERVE_ASPECT_RATIO_INFO = "Maintain the original image aspect ratio during upscaling"

IMAGE_OUTPUT_FORMAT_INFO = "Choose output image format. PNG for lossless, JPEG for smaller files"

IMAGE_QUALITY_INFO = "Quality level for JPEG/WEBP output (70-100, higher = better quality)"

IMAGE_PRESERVE_METADATA_INFO = "Keep original image EXIF data and metadata in the output"

IMAGE_CUSTOM_SUFFIX_INFO = "Custom suffix added to output filename (e.g., 'image_upscaled.png')"

IMAGE_ENABLE_COMPARISON_INFO = "Generate a side-by-side before/after comparison image"

# Face Restoration Detailed Info
FACE_RESTORATION_ENABLE_INFO = """Enhance faces in the video using CodeFormer. Works with all upscaler types.
- Detects and restores faces automatically
- Can be applied before or after upscaling
- Supports both face restoration and colorization
- Requires CodeFormer models in pretrained_weight/ directory"""

# Context and Memory Management Info
CONTEXT_WINDOW_INFO = """Include previous frames as context when processing each chunk (except the first). Similar to "Optimize Last Chunk Quality" but applied to all chunks.
- Mechanism: Each chunk (except first) includes N previous frames as context, but only outputs new frames. Provides temporal consistency without complex overlap logic.
- Quality Impact: Better temporal consistency and reduced flickering between chunks. More context = better consistency.
- VRAM Impact: Medium increase due to processing context frames (recommend 25-50% of Max Frames per Chunk).
- Speed Impact: Slower due to processing additional context frames, but simpler and more predictable than traditional sliding window."""

ADVANCED_MEMORY_MANAGEMENT_INFO = """Advanced memory management for STAR model components. Automatically moves models to CPU/unloads when not in use.
- Text Encoder: Completely unloaded after encoding, reloaded only for new prompts
- VAE: Moved to CPU when not actively encoding/decoding
- VRAM Impact: Slightly reduces peak VRAM
- Quality Impact: None - identical results with optimized memory usage"""

# Processing Control Messages
PROCESSING_CANCELLED_MESSAGES = {
    'auto_caption_cancelled': "⚠️ Process cancelled by user during auto-captioning",
    'main_process_cancelled': "⚠️ Process cancelled by user",
    'caption_generation_cancelled': "Caption generation failed or was cancelled. Using original prompt."
} 