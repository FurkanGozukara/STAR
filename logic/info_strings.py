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