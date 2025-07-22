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

# Main App Title and Headers
APP_TITLE = "# SECourses Video and Image Upscaler Pro V4 - https://www.patreon.com/posts/134405610"

ENHANCED_INPUT_HEADER = "### 📁 Enhanced Input: Video Files & Frame Folders"
ENHANCED_INPUT_DESCRIPTION = "*Auto-detects whether your input is a single video file or a folder containing frame sequences*"

CHOOSE_IMAGE_UPSCALING_METHOD = "### 🎯 Choose Your Image Upscaling Method"

PROCESSING_CONTROLS_HEADER = "### 🚀 Processing Controls"

IMAGE_UPSCALER_SUPPORT_NOTE = "📝 **Image Upscaler Support:** Target resolution now works with image upscalers! The system automatically adapts based on your selected model's scale factor (2x, 4x, etc.)."

AUTO_RESOLUTION_HEADER = "### 🎯 Auto-Resolution (Aspect Ratio Aware)"

EXPECTED_OUTPUT_RESOLUTION_HEADER = "### 📐 Expected Output Resolution Preview"
EXPECTED_OUTPUT_RESOLUTION_DESCRIPTION = "*Real-time preview of your final video resolution based on current settings*"

AUTOMATIC_SCENE_DETECTION_SETTINGS = "**Automatic Scene Detection Settings**"
MANUAL_SPLIT_SETTINGS = "**Manual Split Settings**"
ENCODING_SETTINGS_SCENE_SEGMENTS = "**Encoding Settings (for scene segments)**"

UPSCALER_SELECTION_HEADER = "### Upscaler Selection"

AUTO_CAPTIONING_DISABLED_NOTE = "_(Auto-captioning disabled as CogVLM2 components are not fully available.)_"

FACE_RESTORATION_HEADER = "### Face Restoration (CodeFormer)"

STAR_MODEL_SETTINGS_HEADER = "### STAR Model Settings - Temporal Upscaling"

IMAGE_BASED_UPSCALER_SETTINGS_HEADER = "### Image-Based Upscaler Settings - Spatial Upscaling"
IMAGE_BASED_UPSCALER_DESCRIPTION = "**High-speed deterministic upscaling using specialized image upscaler models**"
IMAGE_BASED_UPSCALER_NOTE = "*⚙️ Enable Image-Based Upscaling in the Core Settings tab first*"

QUICK_PREVIEW_HEADER = "### 🔍 Quick Preview & Model Testing"
QUICK_PREVIEW_DESCRIPTION = "**Test upscaler models on the first frame of your video**"

MODEL_INFORMATION_HEADER = "### Model Information & Performance"

SEEDVR2_VIDEO_UPSCALER_HEADER = "### SeedVR2 Video Upscaler - Advanced AI Video Enhancement"

BLOCK_SWAP_DESCRIPTION = "**🔄 Block Swap reduces VRAM usage by offloading transformer blocks to CPU**"

CHUNK_PREVIEW_DESCRIPTION = "**📹 Chunk Preview - Similar to STAR model chunk preview functionality**"

GENERATE_CUSTOM_COMPARISON_VIDEOS = "### Generate Custom Comparison Videos"
CUSTOM_COMPARISON_DESCRIPTION = "Upload 2-4 videos to create custom comparison videos with various layout options using the same FFmpeg settings as the automatic comparison feature."

CUSTOM_COMPARISON_STEP1 = "**Step 1:** Choose number of videos to compare"
CUSTOM_COMPARISON_STEP2 = "**Step 2:** Upload videos for comparison"
CUSTOM_COMPARISON_STEP3 = "**Step 3:** Choose comparison layout"
CUSTOM_COMPARISON_STEP4 = "**Step 4:** Generate the comparison video using current FFmpeg settings"

FRAME_INTERPOLATION_HEADER = "### Frame Interpolation (RIFE)"

APPLY_RIFE_TO_INTERMEDIATE = "**Apply RIFE to intermediate videos (recommended)**"

RIFE_NOTE = "**Note:** When RIFE is enabled, the system will return RIFE-interpolated versions to the interface instead of originals, ensuring you get the smoothest possible results throughout the process."

PRE_PROCESSING_FPS_REDUCTION_HEADER = "### Pre-Processing FPS Reduction"
PRE_PROCESSING_FPS_REDUCTION_DESCRIPTION = "**Reduce FPS before upscaling** to speed up processing and reduce VRAM usage. You can then use RIFE interpolation to restore smooth motion afterward."

WORKFLOW_TIP = "**💡 Workflow Tip:** Use FPS decrease (1/2x for balanced speed/quality) for faster upscaling, then enable RIFE 2x-4x to restore smooth 24-60 FPS output!"

VIDEO_EDITOR_HEADER = "# Video Editor - Cut and Extract Video Segments"
VIDEO_EDITOR_DESCRIPTION = "**Cut specific time ranges or frame ranges from your videos with precise FFmpeg encoding.**"

STANDALONE_FACE_RESTORATION_HEADER = "# Standalone Face Restoration - CodeFormer Processing"
STANDALONE_FACE_RESTORATION_DESCRIPTION = "**Apply face restoration to videos using CodeFormer without upscaling. Perfect for improving face quality in existing videos.**"

# Processing Status Messages
PROCESSING_COMPLETE_TEMPLATE = """📊 Processing Complete!

⏱️ Total Time: {total_time:.2f} seconds
🎬 Frames Processed: {frames_processed}
📁 Output: {output_filename}
✅ Status: Success"""

BATCH_PROCESSING_COMPLETE_TEMPLATE = """📊 Batch Processing Complete!

⏱️ Total Time: {total_time:.2f} seconds
🎬 Videos Processed: {videos_processed}
📁 Output Folder: {output_folder}
🔧 Batch Size: {face_restoration_batch_size_val}"""

VIDEO_CUTTING_SUCCESS_TEMPLATE = """✅ Video cutting completed successfully!

📁 Output: {output_filename}
⏱️ Processing Time: {processing_time:.2f} seconds
✂️ Cuts Applied: {cuts_applied}

{analysis_text}"""

VIDEO_CUTTING_INTEGRATION_TEMPLATE = """🎬✅ Video cutting completed successfully!

📁 Output: {output_filename}
⏱️ Processing Time: {processing_time:.2f} seconds
✂️ Cuts Applied: {cuts_applied}

{status_msg}"""

# SeedVR2 Status Messages
SEEDVR2_INSTALLATION_MISSING = """❌ SeedVR2 Installation Missing

Required components not found in:
└── SeedVR2/

Please ensure SeedVR2 is properly installed.
Expected location: SeedVR2/models/"""

SEEDVR2_NO_MODELS_FOUND = """📥 No Models Found

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

SEEDVR2_ERROR_DETECTED_TEMPLATE = """❌ Error Detected

{error_message}

Please check the logs for more details and ensure:
• SeedVR2 is properly installed
• Required dependencies are available
• File permissions are correct"""

# Block Swap Status Messages
BLOCK_SWAP_ADVANCED_ENABLED_TEMPLATE = """🔄 Advanced Block Swap Enabled

{status_icon} {status_msg}
💾 Current Memory: {memory_info}

📊 Block Swap Configuration:
• Blocks to swap: {block_swap_counter}
• Estimated VRAM savings: ~{estimated_savings_gb:.1f}GB
• Expected performance impact: ~{estimated_performance_impact:.1f}%

💡 Real-time optimization active"""

BLOCK_SWAP_ENABLED_FALLBACK_TEMPLATE = """🔄 Block Swap Enabled: {block_swap_counter} blocks

📉 Estimated VRAM reduction: ~{estimated_savings}%
⚡ Performance impact: ~{performance_impact}% slower
💡 Recommended for VRAM-limited systems

⚠️ Real-time monitoring unavailable"""

# GPU Status Messages
NO_CUDA_GPUS_DETECTED_DETAILED = """❌ No CUDA GPUs detected

Please ensure:
• NVIDIA GPU is installed and recognized
• CUDA drivers are properly installed
• GPU is not being used by other processes"""

# Intelligent Analysis Messages
INTELLIGENT_BLOCK_SWAP_ANALYSIS_TEMPLATE = """🧠 Intelligent Block Swap Analysis

📊 Model: {model_filename}
💾 VRAM Requirements: {vram_requirements}
⚙️ Recommended Settings:
{recommendations}

Click "Apply Optimal Settings" to use these recommendations."""

PROFESSIONAL_MULTI_GPU_ANALYSIS_TEMPLATE = """🚀 Professional Multi-GPU Analysis

📊 Model: {model_filename}
💾 Total VRAM: {total_vram:.1f}GB
⚙️ Recommended Configuration:
{recommendations}

⚡ Performance monitoring will activate during processing to track actual speedup."""

MODEL_VALIDATION_ENHANCED_TEMPLATE = """{current_info}

📊 Model Specifications:
• Architecture: {architecture}
• Scale Factor: {scale_factor}x
• Parameters: {parameters}
• VRAM Usage: {vram_usage}
• Block Swap: {'Enabled' if recommendations.get('enable_block_swap', False) else 'Disabled'}
• Multi-GPU: {'Enabled' if recommendations.get('enable_multi_gpu', False) else 'Disabled'}

📊 Available VRAM: {total_vram:.1f}GB"""

MODEL_VALIDATION_FAILED_TEMPLATE = """❌ Model Validation Failed

Error: {error_message}

Please ensure:
• Model file is not corrupted
• File permissions are correct
• Sufficient disk space available"""

BLOCK_SWAP_ACTIVE_STATUS_TEMPLATE = """🔄 Advanced Block Swap Active

📊 Current Status: {status_msg}
💾 Memory Usage: {memory_info}
⚙️ Configuration: {block_swap_counter} blocks
📈 Estimated Savings: ~{estimated_savings:.1f}GB

🔄 Auto-refreshing every 30 seconds"""

# Model Info Display Templates
MODEL_INFO_DISPLAY_TEMPLATE = """**{model_name}**
📊 Scale Factor: {scale_factor}x
💾 VRAM Usage: {vram_usage}
🔧 Architecture: {architecture}
📏 Input Size: {input_size}
🎯 Output Size: {output_size}
⚡ Supports BFloat16: {supports_bfloat16}"""

# Error Messages
MODEL_INFO_EXTRACTION_ERROR = "Could not extract model information from selection"
MODEL_SCAN_ERROR_STATUS = "Error scanning models - check upscale_models/ directory"
SEEDVR2_NO_MODELS_STATUS = "No SeedVR2 models found - Place .safetensors files in SeedVR2/models/"

# Processing Cancelled Messages
PROCESSING_CANCELLED_MESSAGES = {
    'auto_caption_cancelled': "⚠️ Process cancelled by user during auto-captioning",
    'main_process_cancelled': "⚠️ Process cancelled by user",
    'caption_generation_cancelled': "Caption generation failed or was cancelled. Using original prompt."
}

# Additional UI Component Info Strings
UPSCALER_TYPE_SELECTION_INFO = """Select the upscaling method:
• STAR Model: AI temporal upscaler with prompts and advanced settings
• Image Based: Fast deterministic spatial upscaling (RealESRGAN, etc.)  
• SeedVR2: Upcoming advanced video upscaler (coming soon)"""

COGVLM_QUANTIZATION_INFO = "Quantization for the CogVLM2 captioning model (uses less VRAM). INT4/8 require CUDA & bitsandbytes."

FACE_RESTORATION_FIDELITY_INFO = """Balance between quality and identity preservation:
- 0.0-0.3: Prioritize quality/detail (may change facial features)
- 0.4-0.6: Balanced approach
- 0.7-1.0: Prioritize identity preservation (may reduce enhancement)"""

AUTO_CAPTION_THEN_UPSCALE_INFO = "If checked, clicking 'Upscale Video' will first generate a caption and use it as the prompt."

GPU_SELECTOR_INFO = "Select which GPU to use for processing. Defaults to GPU 0."

PRESET_SELECTOR_INFO = "Select a preset to auto-load, or type a new name and click Save."

COMPARISON_IMAGE_GENERATION_INFO = "Generate side-by-side comparison image showing original vs upscaled"

TARGET_RESOLUTION_ENABLE_INFO = "Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."

AUTO_RESOLUTION_DISPLAY_INFO = "Shows current auto-calculated resolution and aspect ratio information"

OUTPUT_RESOLUTION_DISPLAY_INFO = "Shows the calculated final output resolution based on your current upscaler settings, target resolution, and input video"

# Scene Splitting Info Strings
SCENE_MIN_LENGTH_INFO = "Minimum duration for a scene. Shorter scenes will be merged or dropped."

SCENE_THRESHOLD_INFO = "Sensitivity of scene detection. Lower values detect more scenes."

SCENE_DROP_SHORT_INFO = "If enabled, scenes shorter than minimum length are dropped instead of merged."

SCENE_MERGE_LAST_INFO = "If the last scene is too short, merge it with the previous scene."

SCENE_FRAME_SKIP_INFO = "Skip frames during detection to speed up processing. 0 = analyze every frame."

SCENE_MIN_CONTENT_INFO = "Minimum content change required to detect a scene boundary."

SCENE_FRAME_WINDOW_INFO = "Number of frames to analyze for scene detection."

SCENE_MANUAL_SPLIT_TYPE_INFO = "'duration': Split every N seconds.\n'frame_count': Split every N frames."

SCENE_MANUAL_SPLIT_VALUE_INFO = "Duration in seconds or number of frames for manual splitting."

SCENE_COPY_STREAMS_INFO = "Copy video/audio streams without re-encoding during scene splitting (faster) but can generate inaccurate splits."

SCENE_USE_MKVMERGE_INFO = "Use mkvmerge instead of ffmpeg for splitting (if available)."

SCENE_RATE_FACTOR_INFO = "Quality setting for re-encoding (lower = better quality). Only used if Copy Streams is disabled."

SCENE_ENCODING_PRESET_INFO = "Encoding speed vs quality trade-off. Only used if Copy Streams is disabled."

SCENE_QUIET_FFMPEG_INFO = "Suppress ffmpeg output during scene splitting."

# Face Restoration Additional Info
FACE_COLORIZATION_INFO = "Apply colorization to grayscale faces (experimental feature)"

CODEFORMER_MODEL_SELECTION_DETAILED_INFO = "Select the CodeFormer model. 'Auto' uses the default model. Models should be in pretrained_weight/ directory."

FACE_RESTORATION_BATCH_SIZE_INFO = "Number of frames to process simultaneously for face restoration. Higher values = faster processing but more VRAM usage."

# Core Model Settings Info
UPSCALE_FACTOR_INFO = "Simple multiplication factor for output resolution if 'Enable Max Target Resolution' is OFF. E.g., 4.0 means 4x height and 4x width."

GUIDANCE_SCALE_INFO = "Controls how strongly the model follows your combined text prompt. Higher values mean stricter adherence, lower values allow more creativity. Typical values: 5.0-10.0."

DENOISING_STEPS_INFO = "Number of denoising steps. 'Fast' mode uses a fixed ~15 steps. 'Normal' mode uses the value set here."

CONTEXT_FRAMES_INFO = "Number of previous frames to include as context for each chunk (except first). 0 = disabled (same as normal chunking). Higher values = better consistency but more VRAM and slower processing. Recommended: 25-50% of Max Frames per Chunk."

# Tiling Settings Info
TILE_SIZE_INFO = "Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."

TILE_OVERLAP_INFO = "How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."

# Image Upscaler Info
IMAGE_UPSCALER_MODEL_INFO = "Select the image upscaler model. Models should be placed in the 'upscale_models/' directory. Recommended: 2xLiveActionV1_SPAN_490000.pth for video content."

IMAGE_UPSCALER_BATCH_SIZE_INFO = "Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage. Adjust based on your GPU memory."

MODEL_DETAILS_DISPLAY_INFO = "Shows architecture, scale factor, and other model details"

# SeedVR2 Info Strings
SEEDVR2_STATUS_DISPLAY_INFO = "Shows SeedVR2 installation and dependency status"

SEEDVR2_MODEL_SELECTION_INFO = "Select SeedVR2 model. 3B FP8 models offer best speed/VRAM balance, 7B models provide highest quality."

SEEDVR2_MODEL_INFO_DISPLAY_INFO = "Detailed model specifications and requirements"

SEEDVR2_QUALITY_PRESET_INFO = "Processing quality preset. Fast: prioritize speed, Balanced: good speed/quality balance, Quality: maximum quality."

SEEDVR2_VRAM_OPTIMIZATION_INFO = "Optimize VRAM usage. Recommended for most systems."

SEEDVR2_COLOR_FIX_INFO = "Fix color shifts using wavelet reconstruction. Recommended."

SEEDVR2_OPTIMIZE_LAST_CHUNK_INFO = "Optimize last chunk quality like STAR model. Recommended."

SEEDVR2_MEMORY_EFFICIENT_ATTENTION_INFO = "Memory-efficient attention mechanism. Default enabled."

SEEDVR2_SCENE_AWARE_PROCESSING_INFO = "Enable scene-aware temporal processing for better scene boundary handling."

SEEDVR2_TEMPORAL_VALIDATION_INFO = "Validate temporal consistency during processing and report quality metrics."

SEEDVR2_CHUNK_OPTIMIZATION_INFO = "Optimize chunk boundaries for temporal coherence. Recommended for longer videos."

SEEDVR2_TEMPORAL_CONSISTENCY_INFO = "Balance between processing speed and temporal consistency quality."

SEEDVR2_GPU_ACCELERATION_INFO = "Enable GPU acceleration for SeedVR2 processing."

SEEDVR2_MULTI_GPU_INFO = "Distribute processing across multiple GPUs for faster processing."

SEEDVR2_GPU_DEVICES_INFO = "Comma-separated GPU IDs (e.g., '0,1,2'). Single GPU: '0'"

SEEDVR2_GPU_STATUS_INFO = "Shows available GPUs and their VRAM status"

SEEDVR2_BLOCK_SWAP_INFO = "Enable block swapping for large models on limited VRAM systems."

SEEDVR2_BLOCK_SWAP_COUNTER_INFO = "Number of blocks to swap (0=disabled). Higher = more VRAM savings but slower."

SEEDVR2_BLOCK_SWAP_IO_INFO = "Offload input/output layers for maximum VRAM savings. Optional."

SEEDVR2_MODEL_CACHE_INFO = "Keep model cached in RAM between runs. Faster batch processing."

SEEDVR2_BLOCK_SWAP_STATUS_INFO = "Shows block swap configuration and estimated VRAM savings"

SEEDVR2_CHUNK_PREVIEW_INFO = "Enable chunk preview functionality to display processed chunks in main tab."

SEEDVR2_PREVIEW_FRAMES_INFO = "Number of frames to show in chunk preview (default: 125 frames)."

SEEDVR2_CHUNK_RETENTION_INFO = "Number of recent chunk videos to keep in chunks folder (default: 5)."

SEEDVR2_GUIDANCE_SCALE_INFO = "Guidance scale for generation. Usually 1.0 for SeedVR2."

# FFmpeg Settings Info
FFMPEG_GPU_ENCODING_INFO = "If checked, uses NVIDIA's NVENC for FFmpeg video encoding (downscaling and final video creation). Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."

FFMPEG_PRESET_INFO = "Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently from CPU presets."

FFMPEG_QUALITY_INFO = "For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality but larger files."

FRAMES_TO_VIDEO_FPS_INFO = "FPS to use when converting frame folders to videos. This setting only applies when processing input frame folders (not regular videos). Common values: 23.976, 24, 25, 29.97, 30, 60."

# Output Settings Info
SAVE_FRAMES_INFO = "If checked, saves the extracted input frames and the upscaled output frames into a subfolder named after the output video (e.g., '0001/input_frames' and '0001/processed_frames')."

SAVE_METADATA_INFO = "If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time."

SAVE_CHUNKS_INFO = "If checked, saves each processed chunk as a video file in a 'chunks' subfolder (e.g., '0001/chunks/chunk_0001.mp4'). Uses the same FFmpeg settings as the final video."

SAVE_CHUNK_FRAMES_INFO = "If checked, saves the input frames for each chunk before processing into a 'chunk_frames' subfolder (e.g., '0001/chunk_frames/chunk_01_frame_012.png'). Useful for debugging which frames are processed in each chunk."

# Seed Settings Info
SEED_INFO = "Seed for random number generation. Used for reproducibility. Set to -1 or check 'Random Seed' for a random seed. Value is ignored if 'Random Seed' is checked."

RANDOM_SEED_INFO = "If checked, a random seed will be generated and used, ignoring the 'Seed' value."

# Video Comparison Info
VIDEO_COUNT_SELECTOR_INFO = "Select how many videos you want to compare. Additional video inputs will appear based on your selection."

COMPARISON_LAYOUT_INFO = "Layout options will update based on number of videos selected."

# Batch Processing Additional Info
BATCH_INPUT_FOLDER_INFO = "Folder containing video files to process in batch mode."

BATCH_OUTPUT_FOLDER_INFO = "Folder where processed videos will be saved with organized structure."

BATCH_FRAME_FOLDERS_INFO = "Enable to process subfolders containing frame sequences instead of video files. Each subfolder with images will be converted to video first."

BATCH_DIRECT_IMAGE_UPSCALING_INFO = "Process individual image files (JPG, PNG, etc.) directly with selected image upscaler model. Ideal for batch upscaling photos/images."

BATCH_SKIP_EXISTING_INFO = "Skip processing if the output file already exists. Useful for resuming interrupted batch jobs."

# RIFE Processing Info
RIFE_MULTIPLIER_INFO = "Choose how much to increase the frame rate. 2x doubles FPS (e.g., 24→48), 4x quadruples FPS (e.g., 24→96)."

RIFE_FP16_INFO = "Use half-precision floating point for faster processing and lower VRAM usage. Recommended for most users."

RIFE_UHD_INFO = "Enable UHD mode for 4K+ videos. May improve quality for very high resolution content but requires more VRAM."

RIFE_SCALE_INFO = "Scale factor for RIFE processing. 1.0 = original size. Lower values use less VRAM but may reduce quality. Higher values may improve quality but use more VRAM."

RIFE_SKIP_STATIC_INFO = "Automatically detect and skip interpolating static (non-moving) frames to save processing time and avoid unnecessary interpolation."

RIFE_APPLY_TO_CHUNKS_INFO = "Apply RIFE interpolation to individual video chunks during processing. Enabled by default for smoother intermediate results."

RIFE_APPLY_TO_SCENES_INFO = "Apply RIFE interpolation to individual scene videos when scene splitting is enabled. Enabled by default for consistent results."

# FPS Control Info
FPS_DECREASE_ENABLE_INFO = "Reduce video FPS before upscaling to speed up processing. Fewer frames = faster upscaling and lower VRAM usage."

FPS_MODE_INFO = "Multiplier: Reduce by fraction (1/2x, 1/4x). Fixed: Set specific FPS value. Multiplier is recommended for automatic adaptation to input video."

FPS_MULTIPLIER_PRESET_INFO = "Choose common multiplier. 1/2x is recommended for good speed/quality balance."

FPS_MULTIPLIER_CUSTOM_INFO = "Custom multiplier value (0.1 to 1.0). Lower = fewer frames."

FPS_TARGET_INFO = "Target FPS for the reduced video. Lower FPS = faster upscaling. Common choices: 12-15 FPS for fast processing, 24 FPS for cinema standard. Supports precise values like 23.976."

FPS_REDUCTION_METHOD_INFO = "Drop: Faster, simply removes frames. Blend: Smoother, blends frames together (slower but may preserve motion better)."

# RIFE Output Control Info
RIFE_ENABLE_FPS_LIMIT_INFO = "Limit the output FPS to specific common values instead of unlimited interpolation. Useful for compatibility with displays and media players."

RIFE_MAX_FPS_LIMIT_INFO = "Maximum FPS when limiting is enabled. NTSC rates: 23.976/29.970/59.940 (film/TV), Standard: 24/25/30/50/60, Gaming: 120/144/240+. Choose based on your target format and display."

RIFE_KEEP_ORIGINAL_INFO = "Keep the original (non-interpolated) video files alongside the RIFE-processed versions. Recommended to compare results."

RIFE_REPLACE_OUTPUT_INFO = "Replace the original upscaled video with the RIFE version as the primary output. When disabled, both versions are available."

# Video Information Display Info
VIDEO_INFO_DISPLAY_INFO = "Shows duration, FPS, frame count, resolution"

# Video Editing Info
CUTTING_MODE_INFO = "Choose between time-based or frame-based cutting"

TIME_RANGES_INFO = "Format: start1-end1,start2-end2,... (supports decimal seconds and MM:SS format)"

FRAME_RANGES_INFO = "Format: start1-end1,start2-end2,... (frame numbers are 0-indexed)"

CUT_ANALYSIS_INFO = "Shows details about the cuts being made"

# Status Messages for Various Components
DEFAULT_PREVIEW_STATUS = "Upload a video and click a preview button to test upscaler models"

DEFAULT_TIME_ESTIMATE_STATUS = "📊 Upload video and enter ranges to see time estimate"

MODEL_SCAN_ERROR_STATUS = "Error scanning models - check upscale_models/ directory"

SEEDVR2_NO_MODELS_STATUS = "No SeedVR2 models found - Place .safetensors files in SeedVR2/models/"

MODEL_INFO_EXTRACTION_ERROR = "Could not extract model information from selection"

# Labels for Longer Components
DESCRIBE_VIDEO_CONTENT_LABEL = "Describe the Video Content (Prompt) (Useful only for STAR Model)"

DEFAULT_POSITIVE_PROMPT_LABEL = "Default Positive Prompt (Appended)"

DEFAULT_NEGATIVE_PROMPT_LABEL = "Default Negative Prompt (Appended)"

INPUT_VIDEO_OR_FRAMES_LABEL = "Input Video or Frames Folder Path"

OUTPUT_QUALITY_JPEG_LABEL = "Output Quality (JPEG/WEBP only)"

CREATE_BEFORE_AFTER_COMPARISON_LABEL = "Create Before/After Comparison"

ENABLE_SCENE_SPLITTING_LABEL = "Enable Scene Splitting - Recommended"

STAR_MODEL_TEMPORAL_UPSCALING_LABEL = "STAR Model - Temporal Upscaling"

UPSCALE_FACTOR_TARGET_RES_DISABLED_LABEL = "Upscale Factor (if Target Res disabled)"

ENABLE_ADVANCED_VRAM_OPTIMIZATION_LABEL = "Enable Advanced VRAM Optimization"

SELECT_UPSCALER_MODEL_SPATIAL_LABEL = "Select Upscaler Model - Spatial Upscaling"

BEFORE_AFTER_COMPARISON_SLIDER_LABEL = "Before/After Comparison (Original ← Slider → Upscaled)"

BATCH_SIZE_TEMPORAL_CONSISTENCY_LABEL = "Batch Size (Temporal Consistency)"

USE_NVIDIA_GPU_FFMPEG_LABEL = "Use NVIDIA GPU for FFmpeg (h264_nvenc)"

FFMPEG_QUALITY_CRF_CQ_LABEL = "FFmpeg Quality (CRF for libx264 / CQ for NVENC)"

SAVE_INPUT_PROCESSED_FRAMES_LABEL = "Save Input and Processed Frames"

SAVE_CHUNK_INPUT_FRAMES_DEBUG_LABEL = "Save Chunk Input Frames (Debug)"

PROCESS_FRAME_FOLDERS_BATCH_LABEL = "Process Frame Folders in Batch"

USE_PROMPT_FILES_FILENAME_LABEL = "Use Prompt Files (filename.txt)"

GENERATE_PREVIEW_FIRST_SEGMENT_LABEL = "Generate Preview of First Segment"

INPUT_VIDEO_FACE_RESTORATION_LABEL = "Input Video for Face Restoration"

FACE_RESTORATION_FIDELITY_WEIGHT_LABEL = "Face Restoration Fidelity Weight"

CREATE_BEFORE_AFTER_COMPARISON_VIDEO_LABEL = "Create Before/After Comparison Video"

FFMPEG_QUALITY_CRF_LIBX264_LABEL = "FFmpeg Quality (CRF for libx264)" 

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

# General Error Templates
GENERAL_ERROR_TEMPLATES = {
    'no_results_error': "❌ No results generated. Please check your input and try again.",
    'processing_failed': "❌ Processing failed. Please check the logs for details.",
    'invalid_input': "❌ Invalid input provided. Please check your settings.",
    'model_not_found': "❌ Model not found. Please check your model selection.",
    'gpu_error': "❌ GPU error occurred. Please check your GPU settings.",
    'memory_error': "❌ Out of memory. Please reduce batch size or model complexity.",
    'file_not_found': "❌ File not found. Please check your file paths.",
    'permission_error': "❌ Permission denied. Please check file permissions.",
    'timeout_error': "❌ Processing timed out. Please try with smaller input.",
    'unknown_error': "❌ An unknown error occurred. Please check the logs."
}

# Auto-Caption Labels and Info
AUTO_CAPTION_THEN_UPSCALE_LABEL = "Auto-Caption Then Upscale"
AUTO_CAPTION_THEN_UPSCALE_INFO = "Generate automatic captions before upscaling using CogVLM2. Useful for videos without existing prompts." 

# Parser and System Messages
PARSER_HELP_OUTPUTS_FOLDER = "Main folder for output videos and related files"

# Warning and Error Messages for Base Path Setup
VIDEO_TO_VIDEO_DIR_NOT_FOUND_WARNING = "Warning: 'video_to_video' directory not found in inferred base_path: {base_path}. Attempting to use parent directory."
BASE_PATH_AUTO_DETERMINE_ERROR = "Error: Could not auto-determine STAR repository root. Please set 'base_path' manually."
BASE_PATH_USAGE_INFO = "Using STAR repository base_path: {base_path}"
APP_PLACEMENT_ERROR = "Please ensure app.py is correctly placed or base_path is manually set."
DEPENDENCIES_INSTALL_ERROR = "Please ensure the STAR repository is correctly in the Python path (set by base_path) and all dependencies from 'requirements.txt' are installed."

# Logger Diagnostic Messages
STREAM_HANDLER_LEVEL_SET_INFO = "Diagnostic: Explicitly set StreamHandler level to INFO."
STREAM_HANDLER_ADDED_INFO = "Diagnostic: No StreamHandler found, added a new one with INFO level."

# Model Path Error Messages
LIGHT_DEG_MODEL_NOT_FOUND_ERROR = "FATAL: Light degradation model not found at {model_path}."
HEAVY_DEG_MODEL_NOT_FOUND_ERROR = "FATAL: Heavy degradation model not found at {model_path}."

# Preset Loading Messages
PRESET_LAST_USED_FOUND_INFO = "Found last used preset: '{preset_name}'. Attempting to load."
PRESET_NO_LAST_USED_INFO = "No last used preset found. Looking for 'Default' preset."
PRESET_SUCCESSFULLY_LOADED_INFO = "Successfully loaded initial preset '{preset_name}'."
PRESET_GPU_INDEX_OUT_OF_RANGE_WARNING = "GPU index {gpu_num} out of range. Defaulting to GPU {default_gpu}."
PRESET_INVALID_GPU_VALUE_WARNING = "Invalid GPU device value '{value}'. Defaulting to GPU {default_gpu}."
PRESET_CLEARING_FAILED_WARNING = "Clearing last used preset '{preset_name}' because it failed to load."
PRESET_FALLBACK_ATTEMPT_WARNING = "Could not load initial preset '{preset_name}'. Reason: {message}. Trying fallback preset 'Image_Upscaler_Fast_Low_VRAM'."
PRESET_FALLBACK_SUCCESS_INFO = "Successfully loaded fallback preset 'Image_Upscaler_Fast_Low_VRAM'."
PRESET_FALLBACK_FAILED_WARNING = "Could not load fallback preset 'Image_Upscaler_Fast_Low_VRAM'. Reason: {message}."
PRESET_TRYING_FIRST_AVAILABLE_INFO = "Trying to load first available preset: '{preset_name}'"
PRESET_LAST_RESORT_SUCCESS_INFO = "Successfully loaded last resort preset '{preset_name}'."
PRESET_LAST_RESORT_FAILED_WARNING = "Last resort preset '{preset_name}' also failed to load: {message}"
PRESET_NO_PRESETS_LOADED_WARNING = "No presets could be loaded. Starting with application defaults."
PRESET_INITIAL_FOR_DROPDOWN_INFO = "Initial preset for dropdown: '{preset_name}' (actually loaded)"
PRESET_NO_PRESET_SUCCESSFULLY_LOADED_INFO = "No preset was successfully loaded, dropdown will show default"

# UI Component Placeholders and Info
PANDA_PLAYING_GUITAR_PLACEHOLDER = "e.g., A panda playing guitar by a lake at sunset."
VIDEO_FRAMES_FOLDER_PLACEHOLDER = "C:/path/to/video.mp4 or C:/path/to/frames/folder/"
PRESET_SELECTION_INFO = "Select a preset to auto-load, or type a new name and click Save."
COMPARISON_IMAGE_GENERATION_DETAILED_INFO = "Generate side-by-side comparison image showing original vs upscaled"

# Button Labels
GENERATE_CAPTION_NO_UPSCALE_BUTTON = "Generate Caption with CogVLM2 (No Upscale)"
CANCEL_UPSCALING_BUTTON = "Cancel Upscaling"
CANCEL_PROCESSING_BUTTON = "Cancel Processing"

# Accordion Labels
PROMPT_SETTINGS_STAR_MODEL_ACCORDION = "Prompt Settings (Useful only for STAR Model)"
TARGET_RESOLUTION_ASPECT_RATIO_ACCORDION = "Target Resolution - Maintains Your Input Video Aspect Ratio"
AUTO_CAPTIONING_COGVLM2_STAR_ACCORDION = "Auto-Captioning Settings (CogVLM2) (Only for STAR Model)"
CONTEXT_WINDOW_EXPERIMENTAL_ACCORDION = "Context Window - Previous Frames for Better Consistency - Experimental Probably Not Needed"
PERFORMANCE_VRAM_32_FRAMES_ACCORDION = "Performance & VRAM Optimization - 32 Frames Yields Best Quality"
TILING_HIGH_RES_LOW_VRAM_ACCORDION = "Advanced: Tiling (Very High Res / Low VRAM)"

# Model and System Status Messages
NO_MODELS_FOUND_UPSCALE_MODELS = "No models found - place models in upscale_models/"
ERROR_SCANNING_UPSCALE_MODELS = "Error scanning models - check upscale_models/ directory"
ERROR_SCANNING_SEEDVR2_MODELS = "Error scanning models - check SeedVR2/models/ directory"
SELECT_MODEL_DETAILED_SPECIFICATIONS = "Select a model to see detailed specifications"

# SeedVR2 Info Messages
SEEDVR2_INSTALLATION_DEPENDENCY_STATUS_INFO = "Shows SeedVR2 installation and dependency status"
SEEDVR2_MODEL_SPEED_QUALITY_BALANCE_INFO = "Select SeedVR2 model. 3B FP8 models offer best speed/VRAM balance, 7B models provide highest quality."
SEEDVR2_MODEL_SPECIFICATIONS_INFO = "Detailed model specifications and requirements"
SEEDVR2_QUALITY_PRESET_DETAILED_INFO = "Processing quality preset. Fast: prioritize speed, Balanced: good speed/quality balance, Quality: maximum quality."
SEEDVR2_VRAM_OPTIMIZATION_SYSTEMS_INFO = "Optimize VRAM usage. Recommended for most systems."
SEEDVR2_COLOR_FIX_WAVELET_RECONSTRUCTION_INFO = "Fix color shifts using wavelet reconstruction. Recommended."
SEEDVR2_LAST_CHUNK_QUALITY_STAR_INFO = "Optimize last chunk quality like STAR model. Recommended."
SEEDVR2_MEMORY_EFFICIENT_ATTENTION_DEFAULT_INFO = "Memory-efficient attention mechanism. Default enabled."
SEEDVR2_SCENE_AWARE_BOUNDARY_HANDLING_INFO = "Enable scene-aware temporal processing for better scene boundary handling."
SEEDVR2_TEMPORAL_CONSISTENCY_QUALITY_METRICS_INFO = "Validate temporal consistency during processing and report quality metrics."
SEEDVR2_CHUNK_BOUNDARIES_LONGER_VIDEOS_INFO = "Optimize chunk boundaries for temporal coherence. Recommended for longer videos."
SEEDVR2_SPEED_CONSISTENCY_BALANCE_INFO = "Balance between processing speed and temporal consistency quality."
SEEDVR2_GPU_ACCELERATION_PROCESSING_INFO = "Enable GPU acceleration for SeedVR2 processing."
SEEDVR2_MULTI_GPU_FASTER_PROCESSING_INFO = "Distribute processing across multiple GPUs for faster processing."
SEEDVR2_GPU_DEVICES_COMMA_SEPARATED_INFO = "Comma-separated GPU IDs (e.g., '0,1,2'). Single GPU: '0'"
SEEDVR2_GPU_VRAM_STATUS_INFO = "Shows available GPUs and their VRAM status"
SEEDVR2_BLOCK_SWAP_LIMITED_VRAM_INFO = "Enable block swapping for large models on limited VRAM systems."
SEEDVR2_BLOCK_SWAP_COUNTER_SAVINGS_INFO = "Number of blocks to swap (0=disabled). Higher = more VRAM savings but slower."
SEEDVR2_IO_OFFLOAD_MAXIMUM_SAVINGS_INFO = "Offload input/output layers for maximum VRAM savings. Optional."
SEEDVR2_MODEL_CACHE_FASTER_BATCH_INFO = "Keep model cached in RAM between runs. Faster batch processing."
SEEDVR2_BLOCK_SWAP_CONFIG_SAVINGS_INFO = "Shows block swap configuration and estimated VRAM savings"
SEEDVR2_CHUNK_PREVIEW_MAIN_TAB_INFO = "Enable chunk preview functionality to display processed chunks in main tab."
SEEDVR2_PREVIEW_FRAME_COUNT_125_INFO = "Number of frames to show in chunk preview (default: 125 frames)."
SEEDVR2_CHUNKS_FOLDER_DEFAULT_5_INFO = "Number of recent chunk videos to keep in chunks folder (default: 5)."
SEEDVR2_GUIDANCE_SCALE_GENERATION_INFO = "Guidance scale for generation. Usually 1.0 for SeedVR2."

# FFmpeg Settings Info
FFMPEG_CONTROLS_ENCODING_NVENC_PRESETS_INFO = "Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently from CPU presets."
FFMPEG_QUALITY_CRF_LIBX264_CQ_NVENC_LABEL = "FFmpeg Quality (CRF for libx264 / CQ for NVENC)"
FFMPEG_QUALITY_DETAILED_CRF_CQ_INFO = "For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality but larger files."
FRAME_FOLDER_FPS_CONVERSION_SETTING_INFO = "FPS to use when converting frame folders to videos. This setting only applies when processing input frame folders (not regular videos). Common values: 23.976, 24, 25, 29.97, 30, 60."

# Output Settings Info
SAVE_PROCESSING_METADATA_TXT_INFO = "If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time."
SAVE_CHUNKS_SUBFOLDER_FFMPEG_INFO = "If checked, saves each processed chunk as a video file in a 'chunks' subfolder (e.g., '0001/chunks/chunk_0001.mp4'). Uses the same FFmpeg settings as the final video."

# Seed Settings Detailed Info
SEED_REPRODUCIBILITY_RANDOM_GENERATION_INFO = "Seed for random number generation. Used for reproducibility. Set to -1 or check 'Random Seed' for a random seed. Value is ignored if 'Random Seed' is checked."
RANDOM_SEED_GENERATION_IGNORING_VALUE_INFO = "If checked, a random seed will be generated and used, ignoring the 'Seed' value."

# Video Comparison Settings
VIDEO_COUNT_ADDITIONAL_INPUTS_SELECTION_INFO = "Select how many videos you want to compare. Additional video inputs will appear based on your selection."
COMPARISON_LAYOUT_OPTIONS_VIDEO_SELECTION_INFO = "Layout options will update based on number of videos selected."

# Batch Processing Settings
BATCH_INPUT_FOLDER_CONTAINING_VIDEOS_PLACEHOLDER = "Path to folder containing videos to process..."
BATCH_INPUT_FOLDER_PROCESS_MODE_INFO = "Folder containing video files to process in batch mode."
BATCH_OUTPUT_FOLDER_PROCESSED_VIDEOS_PLACEHOLDER = "Path to output folder for processed videos..."
BATCH_OUTPUT_FOLDER_ORGANIZED_STRUCTURE_INFO = "Folder where processed videos will be saved with organized structure."
BATCH_FRAME_FOLDERS_SUBFOLDERS_SEQUENCES_INFO = "Enable to process subfolders containing frame sequences instead of video files. Each subfolder with images will be converted to video first."
BATCH_DIRECT_IMAGE_UPSCALING_JPG_PNG_INFO = "Process individual image files (JPG, PNG, etc.) directly with selected image upscaler model. Ideal for batch upscaling photos/images."
BATCH_SKIP_EXISTING_INTERRUPTED_JOBS_INFO = "Skip processing if the output file already exists. Useful for resuming interrupted batch jobs."

# FPS Processing Settings
FPS_DECREASE_FASTER_UPSCALING_VRAM_INFO = "Reduce video FPS before upscaling to speed up processing. Fewer frames = faster upscaling and lower VRAM usage."
FPS_MODE_MULTIPLIER_FIXED_ADAPTATION_INFO = "Multiplier: Reduce by fraction (1/2x, 1/4x). Fixed: Set specific FPS value. Multiplier is recommended for automatic adaptation to input video."
FPS_MULTIPLIER_PRESET_SPEED_QUALITY_INFO = "Choose common multiplier. 1/2x is recommended for good speed/quality balance."
FPS_MULTIPLIER_CUSTOM_LOWER_FRAMES_INFO = "Custom multiplier value (0.1 to 1.0). Lower = fewer frames."
FPS_TARGET_CINEMA_STANDARD_23976_INFO = "Target FPS for the reduced video. Lower FPS = faster upscaling. Common choices: 12-15 FPS for fast processing, 24 FPS for cinema standard. Supports precise values like 23.976."

# Status and Control Info
TARGET_RESOLUTION_SIMPLE_UPSCALE_FACTOR_INFO = "Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
AUTO_CALCULATED_RESOLUTION_ASPECT_INFO = "Shows current auto-calculated resolution and aspect ratio information"
FINAL_OUTPUT_RESOLUTION_UPSCALER_SETTINGS_INFO = "Shows the calculated final output resolution based on your current upscaler settings, target resolution, and input video"

# Advanced Processing Info
FPS_REDUCTION_DROP_BLEND_SMOOTHER_INFO = "Drop: Faster, simply removes frames. Blend: Smoother, blends frames together (slower but may preserve motion better)."

# UI Resolution Control Info
TARGET_RESOLUTION_MAXIMUM_OUTPUT_CONTROL_INFO = "Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."

# Additional Status Messages for Various Use Cases
ENHANCED_INPUT_FRAMES_FOLDER_AUTO_DETECTED = "Auto-detects whether your input is a single video file or a folder containing frame sequences"
SEEDVR2_MODELS_SPEED_VRAM_BALANCE_3B_7B = "3B FP8 models offer best speed/VRAM balance, 7B models provide highest quality"

# Logger Configuration Messages 
LOGGER_CONFIGURED_LEVEL_HANDLERS_INFO = "Logger '{logger_name}' configured with level: {level}. Handlers: {handlers}"

# Video Processing Status
FRAME_FOLDER_PROCESSING_MODE_ENABLED_LOG = "Batch frame folder processing mode enabled"

# Image Processing Info
COMPARISON_IMAGE_ORIGINAL_UPSCALED_SHOWING = "Generate side-by-side comparison image showing original vs upscaled"

# Processing Time and Status Templates
PROCESSING_TIME_ESTIMATE_UPLOAD_VIDEO = "📊 Upload video and enter ranges to see time estimate"
CALCULATION_UPLOAD_VIDEO_FPS_REDUCTION = "**📊 Calculation:** Upload a video to see FPS reduction preview" 

# Additional Long Strings from secourses_app.py

# FFmpeg Quality Labels and Settings
FFMPEG_QUALITY_CRF_LIBX264_ONLY_LABEL = "FFmpeg Quality (CRF for libx264)"
FFMPEG_QUALITY_CQ_NVENC_ONLY_LABEL = "FFmpeg Quality (CQ for NVENC)"

# Advanced Processing Messages
SCENE_SPLITTING_ENABLED_AUTO_CAPTIONING_PER_SCENE = "Scene splitting enabled: Auto-captioning will be done per scene during upscaling."
IMAGE_BASED_UPSCALING_AUTO_CAPTIONING_DISABLED = "Image-based upscaling is enabled. Auto-captioning is disabled as image upscalers don't use prompts."
AUTO_CAPTIONING_UNAVAILABLE_ORIGINAL_PROMPT = "Auto-captioning requested but CogVLM2 is not available. Using original prompt."

# Validation and Analysis Messages
VALIDATION_ERROR_PREFIX = "❌ Validation Error: {error_message}"
CUT_ANALYSIS_ERROR_PREFIX = "❌ Error: {error_message}"
TIME_ESTIMATE_VALIDATION_ERROR = "❌ Cannot estimate due to validation error"
TIME_ESTIMATE_ANALYSIS_ERROR = "❌ Error during analysis"

# UI Loading and Status Messages
UPLOAD_VIDEO_DETAILED_INFORMATION = "📹 Upload a video to see detailed information"
UPLOAD_VIDEO_FIRST_MESSAGE = "✏️ Upload a video first"
ENTER_RANGES_CUT_ANALYSIS = "✏️ Enter ranges above to see cut analysis"
ENTER_RANGES_TIME_ESTIMATE = "📊 Upload video and enter ranges to see time estimate"
COULD_NOT_READ_VIDEO_INFORMATION = "❌ Could not read video information"
VIDEO_INFORMATION_ESTIMATE_ERROR = "❌ Cannot estimate without video info"

# Processing Control Messages
PLEASE_UPLOAD_VIDEO_FIRST = "❌ Please upload a video first"
PLEASE_ENTER_CUT_RANGES = "❌ Please enter cut ranges"
PROCESSING_CANCELLED_USER_REQUEST = "⚠️ Processing cancelled by user"
PROCESSING_FAILED_CHECK_LOGS = "❌ Processing failed. Please check the logs for details."

# Auto-Caption Status Messages  
STARTING_AUTO_CAPTIONING = "Starting auto-captioning..."
AUTO_CAPTIONING_CANCELLED_USER = "Auto-captioning cancelled by user"
CAPTION_GENERATION_FAILED_CANCELLED = "Caption generation failed or was cancelled. Using original prompt."

# GPU and System Messages
GPU_INDEX_OUT_OF_RANGE_DEFAULTING = "GPU index {gpu_index} out of range. Available GPUs: {gpu_count}. Defaulting to 0."
FAILED_PARSE_GPU_INDEX_DEFAULTING = "Failed to parse GPU index from '{gpu_choice}'. Defaulting to 0."
GPU_CHOICE_NOT_FOUND_DEFAULTING = "GPU choice '{gpu_choice}' not found in available GPUs. Defaulting to 0."
NO_CUDA_GPUS_DETECTED = "No CUDA GPUs detected"

# Video Information and Status
VIDEO_INFORMATION_UPLOADED_PREFIX = "Video uploaded: {filename}"
VIDEO_DETAILS_FRAMES_FPS_DURATION = "Video details: {frames} frames, {fps:.2f} FPS, {duration:.2f}s, {width}x{height}"
FAILED_GET_VIDEO_INFO_FOR = "Failed to get video info for: {video_path}"
EXCEPTION_VIDEO_INFO_PREFIX = "Exception in video info: {error}"
ERROR_READING_VIDEO_PREFIX = "❌ Error reading video: {error}"

# Model and Processing Status
SELECT_MODEL_SEE_INFORMATION = "Select a model to see its information"
MODEL_LOAD_INFO_ERROR_PREFIX = "Error loading model: {error}"
MODEL_INFO_LOAD_FAILED_PREFIX = "Could not load information for {model_name}"
FAILED_GET_MODEL_INFO_PREFIX = "Failed to get model info for {model_name}: {error}"
ERROR_LOADING_MODEL_INFO_PREFIX = "Error loading model info: {error}"

# Preset and Configuration Messages
EXPECTED_UI_ARGUMENTS_GOT_ADDING_DEFAULTS = "Expected {expected} UI arguments, got {got}. Adding default values for missing components."
EXPECTED_UI_ARGUMENTS_BATCH_GOT_ADDING = "Expected {expected} UI arguments for batch processing, got {got}. Adding default values for missing components."
EXPECTED_UI_ARGUMENTS_BATCH_GOT_TRIMMING = "Expected {expected} UI arguments for batch processing, got {got}. Trimming extra arguments."

# Frame Folder Processing
AUTO_DETECTED_FRAMES_FOLDER_PREFIX = "Auto-detected frames folder: {folder_path}"
AUTO_DETECTED_VIDEO_FILE_PREFIX = "Auto-detected video file: {video_path}"

# Processing Progress Messages
STARTING_NEW_UPSCALE_CANCELLATION_RESET = "Starting new upscale process - cancellation state reset"
AUTO_CAPTION_FIRST_USER_PROMPT_PREFIX = "In upscale_director_logic. Auto-caption first: {auto_caption}, User prompt: '{user_prompt}...'"

# Temp Folder and Cleanup
TEMP_CLEANUP_REQUESTED_UI_BUTTON = "Temp cleanup requested via UI button"
TEMP_CLEANUP_COMPLETED_FREED_REMAINING = "Temp cleanup completed. Freed {freed_gb:.2f} GB (Remaining: {remaining_gb:.2f} GB)"

# FFmpeg Quality Setting Updates
FFMPEG_CRF_LIBX264_LOWER_VALUES_QUALITY = "For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default)."
FFMPEG_CQ_NVENC_CONSTRAINED_QUALITY_RANGE = "For h24_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28."

# UI Component Default Values and States
DEFAULT_PREVIEW_STATUS_VALUE = "Upload a video and click a preview button to test upscaler models"
SELECT_PRESET_CUSTOM_PROMPT_INFO = "Select a preset to auto-load, or type a new name and click Save."

# Image Processing Status
UPLOAD_IMAGE_SEE_DETAILS = "Upload an image to see details..."
READY_PROCESS_IMAGES = "Ready to process images..."
IMAGE_LOADED_PIXELS_STATUS = "✅ Image loaded: {width}×{height} pixels"
ERROR_LOADING_IMAGE_PREFIX = "❌ Error loading image: {error}"
PLEASE_UPLOAD_IMAGE_FIRST = "❌ Please upload an image first"

# Advanced Settings and Configuration Messages
COMPONENT_WITHOUT_LABEL_ATTRIBUTE_SKIPPING = "Component without label attribute found, skipping"
SECTION_NOT_FOUND_DEFAULT_CONFIG = "Section '{section}' not found in default config, skipping component"
ERROR_ACCESSING_SECTION_KEY_DEFAULT = "Error accessing {section}.{key}: {error}. Using default value."
FALLBACK_VALUE_SECTION_KEY_PREFIX = "Using fallback value for {section}.{key}: {value}"
COMPONENT_LABEL_NOT_FOUND_MAPPING = "Component with label '{label}' not found in component_key_map. Skipping update."

# Video Processing and Conversion Messages
CONVERTING_FRAME_FOLDER_VIDEO_GLOBAL = "Converting frame folder to video using global encoding settings..."
SUCCESSFULLY_CONVERTED_FRAME_FOLDER = "✅ Successfully converted frame folder to video: {message}"
FAILED_CONVERT_FRAME_FOLDER = "❌ Failed to convert frame folder: {message}"
EXCEPTION_FRAME_FOLDER_CONVERSION = "❌ Exception during frame folder conversion: {error}"
FAILED_CONVERT_ANY_FRAME_FOLDERS = "❌ Failed to convert any frame folders to videos"
SUCCESSFULLY_CONVERTED_COUNT_FRAME_FOLDERS = "Successfully converted {count} frame folders to videos"

# Model Information Display
MODEL_INFO_DISPLAY_BASIC_TEMPLATE = "**{model_name}**\n📊 Scale Factor: {scale_factor}x\n💾 VRAM Usage: {vram_usage}\n🔧 Architecture: {architecture}\n📏 Input Size: {input_size}\n🎯 Output Size: {output_size}\n⚡ Supports BFloat16: {supports_bfloat16}"

# Resolution and Preview Messages
EXPECTED_OUTPUT_UPLOAD_VIDEO_SETTINGS = "📹 Upload a video and configure settings to see expected output resolution"
INVALID_VIDEO_DIMENSIONS_PREFIX = "❌ Invalid video dimensions: {width}x{height}"
ERROR_CALCULATING_RESOLUTION_PREFIX = "❌ Error calculating resolution: {error}"
ERROR_CALCULATING_TARGET_RESOLUTION = "❌ Error calculating target resolution: {error}"
ERROR_CALCULATING_PREVIEW_PREFIX = "❌ Error calculating preview ({error})"

# Processing Success and Error Templates
VIDEO_CUTTING_SUCCESS_BASIC_TEMPLATE = "✅ Video cutting completed successfully!\n\n📁 Output: {output_filename}\n⏱️ Processing Time: {processing_time:.2f} seconds\n✂️ Cuts Applied: {cuts_applied}\n\n{analysis_text}"

# Batch and Multi-GPU Processing
FOUND_SEEDVR2_MODELS_COUNT_LOG = "Found {count} SeedVR2 models"
SEEDVR2_MODELS_DIRECTORY_EXISTS_NO_MODELS = "SeedVR2 models directory exists but no models found"
FAILED_REFRESH_SEEDVR2_MODELS = "Failed to refresh SeedVR2 models: {error}"
REFRESH_PRESET_LIST_LOG = "Refreshing preset list: {preset_list}"

# Block Swap and GPU Configuration
BLOCK_SWAP_MONITORING_ERROR_PREFIX = "Block swap monitoring error: {error}"
FAILED_CHECK_SEEDVR2_DEPENDENCIES = "Failed to check SeedVR2 dependencies: {error}"
FAILED_INITIALIZE_SEEDVR2_TAB = "Failed to initialize SeedVR2 tab: {error}"
FAILED_UPDATE_PROFESSIONAL_GPU_STATUS = "Failed to update professional GPU status: {error}"
FAILED_VALIDATE_GPU_DEVICES = "Failed to validate GPU devices: {error}"

# Model and Processing Configuration
MODEL_FILENAME_EXTRACTED_FROM_DROPDOWN = "Displaying info for model: {model_filename}"
FAILED_GET_MODEL_INFO_FOR_CHOICE = "Failed to get model info for '{model_choice}': {error}"
APPLYING_OPTIMAL_SETTINGS_MODEL_VRAM = "Applying optimal settings for {model_filename} with {total_vram:.1f}GB VRAM"
APPLIED_SETTINGS_BATCH_BLOCK_MULTI = "Applied settings: batch_size={batch_size}, block_swap={enable_block_swap}, multi_gpu={enable_multi_gpu}"
FAILED_APPLY_OPTIMAL_SETTINGS = "Failed to apply optimal settings: {error}"

# Advanced Analysis and Recommendations
GENERATED_BLOCK_SWAP_RECOMMENDATIONS = "Generated block swap recommendations for {model_filename}"
FAILED_GET_BLOCK_SWAP_RECOMMENDATIONS = "Failed to get block swap recommendations: {error}"
GENERATED_PROFESSIONAL_MULTI_GPU_RECOMMENDATIONS = "Generated professional multi-GPU recommendations for {model_filename}"
FAILED_GET_MULTI_GPU_RECOMMENDATIONS = "Failed to get multi-GPU recommendations: {error}"
FAILED_VALIDATE_MODEL_CHOICE = "Failed to validate model '{model_choice}': {error}"

# System Startup and Configuration
GRADIO_APP_STARTING_OUTPUT_DEFAULT = "Gradio App Starting. Default output to: {output_path}"
STAR_MODELS_EXPECTED_PATHS = "STAR Models expected at: {light_path}, {heavy_path}"
COGVLM2_MODEL_EXPECTED_PATH = "CogVLM2 Model expected at: {cogvlm_path}"
INITIALIZED_DEFAULT_GPU_INFO = "Initialized with default GPU: {gpu_val} (GPU 0)"
NO_CUDA_GPUS_CPU_FALLBACK_INFO = "No CUDA GPUs detected, attempting to set to GPU 0 (CPU fallback)."

# Additional UI and Processing Messages
CANCELLATION_REQUESTED_STOP_CHECKPOINT = "🚨 CANCELLATION REQUESTED: Processing will stop safely at the next checkpoint. Please wait for current operation to complete..."
NO_ACTIVE_PROCESSING_CANCEL_ALREADY_REQUESTED = "⚠️ No active processing to cancel or cancellation already requested."
PROCESS_CANCELLED_USER_DURING_AUTO_CAPTIONING = "⚠️ Process cancelled by user during auto-captioning"
PROCESS_CANCELLED_MAIN_UPSCALING = "Process cancelled before main upscaling - stopping"

# Template and Format Strings
TEMP_FOLDER_CLEARED_FREED_REMAINING = "✅ Temp folder cleared. Freed {freed_gb:.2f} GB. Remaining: {remaining_label}"
TEMP_FOLDER_CLEANUP_ERRORS_CHECK_LOGS = "⚠️ Temp folder cleanup encountered errors. Check logs."

# Delete Temp Button Updates
DELETE_TEMP_FOLDER_WITH_SIZE = "Delete Temp Folder ({size_label})"

# Video and Processing Analysis
VIDEO_EDITOR_VALIDATION_VIDEO_UPLOADED = "Video uploaded - {filename}"
VIDEO_EDITOR_FAILED_VIDEO_INFO = "Video editor: Failed to get video info for: {video_path}"
VIDEO_EDITOR_EXCEPTION_VIDEO_INFO = "Video editor: Exception in video info: {error}"
VIDEO_EDITOR_ERROR_VALIDATION = "Video editor: Error in validation: {error}"

# FFmpeg Settings and Configuration
CURRENT_FFMPEG_SETTINGS_UNAVAILABLE = "Current FFmpeg settings unavailable, using defaults"

# Processing Results and Status Updates
CUT_COMPLETED_UPDATING_MAIN_TAB = "Video editor: Cut completed, updating main tab input: {output_path}"
PROCESSING_CANCELLED_USER_DURING_CAPTION = "Processing was cancelled by user during captioning."
PROCESSING_CANCELLED_USER_GENERAL = "Processing was cancelled by user."
CURRENT_OUTPUT_VIDEO_CANCELLATION = "Current output video value at cancellation: {output_video}"

# Partial Video Processing Messages
PARTIAL_VIDEO_ALREADY_SET_GENERATOR = "Partial video already set from generator: {video_path}"
PROCESSING_CANCELLED_PARTIAL_SAVED = "⚠️ Processing cancelled by user. Partial video saved: {filename}"
FOUND_PARTIAL_VIDEO_AFTER_CANCELLATION = "Found partial video after cancellation: {video_path}"
SEARCHING_PARTIAL_VIDEOS_PATTERN = "Searching for partial videos with pattern: {pattern}"
FOUND_PARTIAL_FILES_LIST = "Found partial files: {files}"

# Final Processing Messages
PROCESSING_COMPLETED_UI_STATE_RESET = "Processing completed - UI state reset"
ERROR_DURING_PROCESSING_PREFIX = "❌ Error during processing: {error}"
FINAL_CANCELLATION_YIELD_STATUS = "Final cancellation yield - output video: {video_path}, status: {status}"
FINAL_CANCELLATION_NO_PARTIAL_FOUND = "❌ Processing cancelled by user."

# Image Processing and Upload Messages
IMAGE_PROCESSING_FAILED_PREFIX = "❌ Image processing failed: {error}"
ERROR_LOADING_IMAGE_DETAILS = "❌ Error loading image: {error}" 