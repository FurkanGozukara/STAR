"""
Helper functions for creating common Gradio component patterns.
This module reduces code duplication by providing standardized ways to create Gradio components.
"""

import gradio as gr
from typing import List, Union, Optional, Any, Tuple


def create_checkbox(
    label: str,
    value: bool = False,
    info: str = "",
    interactive: bool = True,
    **kwargs
) -> gr.Checkbox:
    """
    Create a standardized Checkbox component.
    
    Args:
        label: The label for the checkbox
        value: Default value (True/False)
        info: Help text to display
        interactive: Whether the component is interactive
        **kwargs: Additional parameters to pass to gr.Checkbox
    
    Returns:
        gr.Checkbox: Configured checkbox component
    """
    return gr.Checkbox(
        label=label,
        value=value,
        info=info,
        interactive=interactive,
        **kwargs
    )


def create_slider(
    label: str,
    minimum: float,
    maximum: float,
    value: float,
    step: float = 1.0,
    info: str = "",
    interactive: bool = True,
    **kwargs
) -> gr.Slider:
    """
    Create a standardized Slider component.
    
    Args:
        label: The label for the slider
        minimum: Minimum value
        maximum: Maximum value
        value: Default value
        step: Step size for the slider
        info: Help text to display
        interactive: Whether the component is interactive
        **kwargs: Additional parameters to pass to gr.Slider
    
    Returns:
        gr.Slider: Configured slider component
    """
    return gr.Slider(
        label=label,
        minimum=minimum,
        maximum=maximum,
        value=value,
        step=step,
        info=info,
        interactive=interactive,
        **kwargs
    )


def create_dropdown(
    label: str,
    choices: List[str],
    value: str = None,
    info: str = "",
    interactive: bool = True,
    allow_custom_value: bool = False,
    **kwargs
) -> gr.Dropdown:
    """
    Create a standardized Dropdown component.
    
    Args:
        label: The label for the dropdown
        choices: List of available options
        value: Default selected value
        info: Help text to display
        interactive: Whether the component is interactive
        allow_custom_value: Allow user to input custom values
        **kwargs: Additional parameters to pass to gr.Dropdown
    
    Returns:
        gr.Dropdown: Configured dropdown component
    """
    if value is None and choices:
        value = choices[0]
    
    return gr.Dropdown(
        label=label,
        choices=choices,
        value=value,
        info=info,
        interactive=interactive,
        allow_custom_value=allow_custom_value,
        **kwargs
    )


def create_radio(
    label: str,
    choices: List[str],
    value: str = None,
    info: str = "",
    interactive: bool = True,
    **kwargs
) -> gr.Radio:
    """
    Create a standardized Radio component.
    
    Args:
        label: The label for the radio buttons
        choices: List of available options
        value: Default selected value
        info: Help text to display
        interactive: Whether the component is interactive
        **kwargs: Additional parameters to pass to gr.Radio
    
    Returns:
        gr.Radio: Configured radio component
    """
    if value is None and choices:
        value = choices[0]
    
    return gr.Radio(
        label=label,
        choices=choices,
        value=value,
        info=info,
        interactive=interactive,
        **kwargs
    )


def create_number(
    label: str,
    value: float = 0.0,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None,
    precision: Optional[int] = None,
    info: str = "",
    interactive: bool = True,
    **kwargs
) -> gr.Number:
    """
    Create a standardized Number input component.
    
    Args:
        label: The label for the number input
        value: Default value
        minimum: Minimum allowed value
        maximum: Maximum allowed value
        step: Step size for increment/decrement
        precision: Number of decimal places
        info: Help text to display
        interactive: Whether the component is interactive
        **kwargs: Additional parameters to pass to gr.Number
    
    Returns:
        gr.Number: Configured number input component
    """
    return gr.Number(
        label=label,
        value=value,
        minimum=minimum,
        maximum=maximum,
        step=step,
        precision=precision,
        info=info,
        interactive=interactive,
        **kwargs
    )


def create_textbox(
    label: str,
    value: str = "",
    placeholder: str = "",
    lines: int = 1,
    max_lines: Optional[int] = None,
    info: str = "",
    interactive: bool = True,
    **kwargs
) -> gr.Textbox:
    """
    Create a standardized Textbox component.
    
    Args:
        label: The label for the textbox
        value: Default text value
        placeholder: Placeholder text
        lines: Number of lines to display
        max_lines: Maximum number of lines
        info: Help text to display
        interactive: Whether the component is interactive
        **kwargs: Additional parameters to pass to gr.Textbox
    
    Returns:
        gr.Textbox: Configured textbox component
    """
    return gr.Textbox(
        label=label,
        value=value,
        placeholder=placeholder,
        lines=lines,
        max_lines=max_lines,
        info=info,
        interactive=interactive,
        **kwargs
    )


def create_button(
    value: str,
    variant: str = "secondary",
    scale: int = 1,
    icon: Optional[str] = None,
    **kwargs
) -> gr.Button:
    """
    Create a standardized Button component.
    
    Args:
        value: Button text
        variant: Button style ("primary", "secondary", "stop")
        scale: Relative width scale
        icon: Optional icon path
        **kwargs: Additional parameters to pass to gr.Button
    
    Returns:
        gr.Button: Configured button component
    """
    return gr.Button(
        value=value,
        variant=variant,
        scale=scale,
        icon=icon,
        **kwargs
    )


# Specialized helper functions for common patterns

def create_batch_size_slider(
    label: str = "Batch Size",
    value: int = 1,
    minimum: int = 1,
    maximum: int = 50,
    info: str = "Number of frames to process simultaneously. Higher values = faster processing but more VRAM usage.",
    **kwargs
) -> gr.Slider:
    """Create a standardized batch size slider."""
    return create_slider(
        label=label,
        minimum=minimum,
        maximum=maximum,
        value=value,
        step=1,
        info=info,
        **kwargs
    )


def create_fidelity_slider(
    label: str = "Fidelity Weight",
    value: float = 0.7,
    info: str = "Balance between quality (0.3) and identity preservation (0.8). 0.7 is recommended for most videos.",
    **kwargs
) -> gr.Slider:
    """Create a standardized fidelity weight slider."""
    return create_slider(
        label=label,
        minimum=0.0,
        maximum=1.0,
        value=value,
        step=0.05,
        info=info,
        **kwargs
    )


def create_gpu_dropdown(
    available_gpus: List[str],
    label: str = "GPU Selection",
    info: str = "Select which GPU to use for processing. Defaults to GPU 0.",
    **kwargs
) -> gr.Dropdown:
    """Create a standardized GPU selection dropdown."""
    choices = available_gpus if available_gpus else ["No CUDA GPUs detected"]
    default_value = available_gpus[0] if available_gpus else "No CUDA GPUs detected"
    
    return create_dropdown(
        label=label,
        choices=choices,
        value=default_value,
        info=info,
        **kwargs
    )


def create_model_dropdown(
    model_choices: List[str],
    default_choice: str,
    label: str = "Model Selection",
    info: str = "Select the model to use for processing.",
    **kwargs
) -> gr.Dropdown:
    """Create a standardized model selection dropdown."""
    return create_dropdown(
        label=label,
        choices=model_choices,
        value=default_choice,
        info=info,
        **kwargs
    )


def create_ffmpeg_preset_dropdown(
    label: str = "FFmpeg Preset",
    value: str = "medium",
    info: str = "Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression.",
    **kwargs
) -> gr.Dropdown:
    """Create a standardized FFmpeg preset dropdown."""
    choices = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
    return create_dropdown(
        label=label,
        choices=choices,
        value=value,
        info=info,
        **kwargs
    )


def create_quality_slider(
    label: str = "Quality",
    value: int = 23,
    minimum: int = 0,
    maximum: int = 51,
    info: str = "Quality setting. Lower values mean higher quality.",
    **kwargs
) -> gr.Slider:
    """Create a standardized quality slider."""
    return create_slider(
        label=label,
        minimum=minimum,
        maximum=maximum,
        value=value,
        step=1,
        info=info,
        **kwargs
    )


def create_save_output_checkbox(
    label: str,
    value: bool = False,
    save_type: str = "frames",
    **kwargs
) -> gr.Checkbox:
    """Create standardized save output checkboxes."""
    info_map = {
        "frames": "Save processed frames as individual image files",
        "metadata": "Save processing parameters and timing information", 
        "chunks": "Save each processed chunk as a separate video file",
        "comparison": "Create side-by-side comparison video showing original vs processed"
    }
    
    info = info_map.get(save_type, "Save additional output files")
    
    return create_checkbox(
        label=label,
        value=value,
        info=info,
        **kwargs
    ) 

# Group helper functions that create multiple components at once

def create_scene_detection_settings(config) -> Tuple[gr.Number, gr.Number, gr.Checkbox, gr.Checkbox, gr.Number, gr.Number, gr.Number]:
    """
    Create all scene detection number inputs and checkboxes at once.
    
    Returns:
        Tuple of (min_scene_len, threshold, drop_short_check, merge_last_check, 
                 frame_skip, min_content_val, frame_window)
    """
    min_scene_len = create_number(
        label="Min Scene Length (seconds)",
        value=config.scene_split.min_scene_len,
        minimum=0.1,
        step=0.1,
        info="Minimum duration for a scene. Shorter scenes will be merged or dropped."
    )
    
    threshold = create_number(
        label="Detection Threshold",
        value=config.scene_split.threshold,
        minimum=0.1,
        maximum=10.0,
        step=0.1,
        info="Sensitivity of scene detection. Lower values detect more scenes."
    )
    
    drop_short_check = create_checkbox(
        label="Drop Short Scenes",
        value=config.scene_split.drop_short,
        info="If enabled, scenes shorter than minimum length are dropped instead of merged."
    )
    
    merge_last_check = create_checkbox(
        label="Merge Last Scene",
        value=config.scene_split.merge_last,
        info="If the last scene is too short, merge it with the previous scene."
    )
    
    frame_skip = create_number(
        label="Frame Skip",
        value=config.scene_split.frame_skip,
        minimum=0,
        step=1,
        info="Skip frames during detection to speed up processing. 0 = analyze every frame."
    )
    
    min_content_val = create_number(
        label="Min Content Value",
        value=config.scene_split.min_content_val,
        minimum=0.0,
        step=1.0,
        info="Minimum content change required to detect a scene boundary."
    )
    
    frame_window = create_number(
        label="Frame Window",
        value=config.scene_split.frame_window,
        minimum=1,
        step=1,
        info="Number of frames to analyze for scene detection."
    )
    
    return min_scene_len, threshold, drop_short_check, merge_last_check, frame_skip, min_content_val, frame_window


def create_rife_processing_settings(config) -> Tuple[gr.Checkbox, gr.Checkbox, gr.Slider]:
    """
    Create RIFE processing checkboxes and scale slider.
    
    Returns:
        Tuple of (fp16_check, uhd_check, scale_slider)
    """
    fp16_check = create_checkbox(
        label="Use FP16 Precision",
        value=config.rife.fp16,
        info="Use half-precision floating point for faster processing and lower VRAM usage. Recommended for most users."
    )
    
    uhd_check = create_checkbox(
        label="UHD Mode",
        value=config.rife.uhd,
        info="Enable UHD mode for 4K+ videos. May improve quality for very high resolution content but requires more VRAM."
    )
    
    scale_slider = create_slider(
        label="Scale Factor",
        minimum=0.25,
        maximum=2.0,
        value=config.rife.scale,
        step=0.25,
        info="Scale factor for RIFE processing. 1.0 = original size. Lower values use less VRAM but may reduce quality."
    )
    
    return fp16_check, uhd_check, scale_slider


def create_all_save_output_checkboxes(config) -> Tuple[gr.Checkbox, gr.Checkbox, gr.Checkbox, gr.Checkbox, gr.Checkbox]:
    """
    Create all save output checkboxes at once.
    
    Returns:
        Tuple of (comparison_video, save_frames, save_metadata, save_chunks, save_chunk_frames)
    """
    comparison_video = create_save_output_checkbox(
        label="Generate Comparison Video",
        value=config.outputs.create_comparison_video,
        save_type="comparison"
    )
    
    save_frames = create_save_output_checkbox(
        label="Save Input and Processed Frames",
        value=config.outputs.save_frames,
        save_type="frames"
    )
    
    save_metadata = create_save_output_checkbox(
        label="Save Processing Metadata",
        value=config.outputs.save_metadata,
        save_type="metadata"
    )
    
    save_chunks = create_save_output_checkbox(
        label="Save Processed Chunks",
        value=config.outputs.save_chunks,
        save_type="chunks"
    )
    
    save_chunk_frames = create_checkbox(
        label="Save Chunk Input Frames (Debug)",
        value=config.outputs.save_chunk_frames,
        info="Save input frames for each chunk before processing. Useful for debugging which frames are processed in each chunk."
    )
    
    return comparison_video, save_frames, save_metadata, save_chunks, save_chunk_frames


def create_ffmpeg_settings(config) -> Tuple[gr.Checkbox, gr.Dropdown, gr.Slider]:
    """
    Create FFmpeg encoding settings components.
    
    Returns:
        Tuple of (use_gpu_check, preset_dropdown, quality_slider)
    """
    use_gpu_check = create_checkbox(
        label="Use NVIDIA GPU for FFmpeg (h264_nvenc)",
        value=config.ffmpeg.use_gpu,
        info="Use NVIDIA's NVENC for FFmpeg video encoding. Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."
    )
    
    preset_dropdown = create_ffmpeg_preset_dropdown(
        value=config.ffmpeg.preset
    )
    
    quality_slider = create_quality_slider(
        label="FFmpeg Quality (CRF for libx264 / CQ for NVENC)",
        value=config.ffmpeg.quality,
        info="For libx264 (CPU): CRF (lower = higher quality). For h264_nvenc (GPU): CQ (lower = better quality)."
    )
    
    return use_gpu_check, preset_dropdown, quality_slider


def create_performance_settings(config) -> Tuple[gr.Slider, gr.Checkbox, gr.Slider, gr.Checkbox]:
    """
    Create performance and VRAM optimization settings.
    
    Returns:
        Tuple of (max_chunk_len, enable_chunk_optimization, vae_chunk, enable_vram_optimization)
    """
    max_chunk_len = create_slider(
        label="Max Frames per Chunk (VRAM)",
        minimum=1,
        maximum=1000,
        value=config.performance.max_chunk_len,
        step=1,
        info="IMPORTANT for VRAM. Divides video into chunks. 32 frames = best quality, uses lots of VRAM."
    )
    
    enable_chunk_optimization = create_checkbox(
        label="Optimize Last Chunk Quality",
        value=config.performance.enable_chunk_optimization,
        info="Process extra frames for last chunks to ensure optimal quality."
    )
    
    vae_chunk = create_slider(
        label="VAE Decode Chunk (VRAM)",
        minimum=1,
        maximum=16,
        value=config.performance.vae_chunk,
        step=1,
        info="Controls max latent frames decoded by VAE simultaneously. Higher = faster but more VRAM."
    )
    
    enable_vram_optimization = create_checkbox(
        label="Enable Advanced VRAM Optimization",
        value=config.performance.enable_vram_optimization,
        info="Automatically moves models to CPU/unloads when not in use. Reduces peak VRAM with no quality impact."
    )
    
    return max_chunk_len, enable_chunk_optimization, vae_chunk, enable_vram_optimization


def create_resolution_settings(config) -> Tuple[gr.Checkbox, gr.Slider, gr.Slider, gr.Radio, gr.Checkbox]:
    """
    Create resolution and target settings.
    
    Returns:
        Tuple of (enable_target_res, target_w, target_h, target_res_mode, enable_auto_aspect)
    """
    enable_target_res = create_checkbox(
        label="Enable Max Target Resolution",
        value=config.resolution.enable_target_res,
        info="Enable to set maximum output resolution limits instead of using upscale factor."
    )
    
    target_w = create_slider(
        label="Target Width",
        minimum=64,
        maximum=7680,
        value=config.resolution.target_w,
        step=64,
        info="Maximum output width in pixels. Will be scaled down proportionally if exceeded."
    )
    
    target_h = create_slider(
        label="Target Height", 
        minimum=64,
        maximum=4320,
        value=config.resolution.target_h,
        step=64,
        info="Maximum output height in pixels. Will be scaled down proportionally if exceeded."
    )
    
    target_res_mode = create_radio(
        label="Target Resolution Mode",
        choices=['downscale_only', 'upscale_and_downscale'],
        value=config.resolution.target_res_mode,
        info="downscale_only: Only reduce resolution if it exceeds targets. upscale_and_downscale: Always scale to targets."
    )
    
    enable_auto_aspect = create_checkbox(
        label="Auto Aspect Resolution",
        value=config.resolution.enable_auto_aspect_resolution,
        info="Automatically calculate optimal resolution based on input video aspect ratio."
    )
    
    return enable_target_res, target_w, target_h, target_res_mode, enable_auto_aspect


def create_face_restoration_settings(config) -> Tuple[gr.Checkbox, gr.Slider, gr.Checkbox, gr.Radio, gr.Dropdown, gr.Slider]:
    """
    Create face restoration settings group.
    
    Returns:
        Tuple of (enable, fidelity, colorization, when, model, batch_size)
    """
    enable = create_checkbox(
        label="Enable Face Restoration",
        value=config.face_restoration.enable,
        info="Enhance faces using CodeFormer. Works with all upscaler types."
    )
    
    fidelity = create_fidelity_slider(
        value=config.face_restoration.fidelity_weight
    )
    
    colorization = create_checkbox(
        label="Enable Colorization",
        value=config.face_restoration.enable_colorization,
        info="Apply colorization to grayscale faces (experimental feature)"
    )
    
    when = create_radio(
        label="Apply Timing",
        choices=['before', 'after'],
        value=config.face_restoration.when,
        info="before: Apply before upscaling. after: Apply after upscaling."
    )
    
    model_choices = ["Auto (Default)", "codeformer.pth (359.2MB)"]
    default_choice = "codeformer.pth (359.2MB)" if config.face_restoration.model and "codeformer.pth" in config.face_restoration.model else "Auto (Default)"
    
    model = create_model_dropdown(
        model_choices=model_choices,
        default_choice=default_choice,
        label="CodeFormer Model",
        info="Select the CodeFormer model. 'Auto' uses the default model."
    )
    
    batch_size = create_batch_size_slider(
        label="Face Restoration Batch Size",
        value=config.face_restoration.batch_size
    )
    
    return enable, fidelity, colorization, when, model, batch_size 