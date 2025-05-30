import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging
from typing import Tuple, Optional

from .ffmpeg_utils import run_ffmpeg_command as util_run_ffmpeg_command
from .file_utils import get_video_resolution as util_get_video_resolution


def determine_comparison_layout(original_w: int, original_h: int, upscaled_w: int, upscaled_h: int) -> Tuple[str, int, int]:
    """
    Determine the best layout (side-by-side or top-bottom) and final dimensions
    for comparison video based on aspect ratio considerations.
    
    Target is to keep final resolution as close to 1920x1080 as possible.
    
    Args:
        original_w, original_h: Original video dimensions
        upscaled_w, upscaled_h: Upscaled video dimensions
        
    Returns:
        Tuple of (layout, final_width, final_height)
        layout: "side_by_side" or "top_bottom"
    """
    
    # Calculate what the final dimensions would be for each layout
    side_by_side_w = original_w + upscaled_w
    side_by_side_h = max(original_h, upscaled_h)
    
    top_bottom_w = max(original_w, upscaled_w)  
    top_bottom_h = original_h + upscaled_h
    
    # Check aspect ratios and prefer the one that fits better within 1920x1080
    side_by_side_area = side_by_side_w * side_by_side_h
    top_bottom_area = top_bottom_w * top_bottom_h
    
    target_area = 1920 * 1080
    
    # Calculate how much each layout exceeds the target
    side_by_side_excess = max(0, side_by_side_area - target_area)
    top_bottom_excess = max(0, top_bottom_area - target_area)
    
    # Also check if either dimension exceeds 1920x1080
    side_by_side_exceeds_bounds = side_by_side_w > 1920 or side_by_side_h > 1080
    top_bottom_exceeds_bounds = top_bottom_w > 1920 or top_bottom_h > 1080
    
    # Prefer the layout that doesn't exceed bounds, or has less excess
    if not side_by_side_exceeds_bounds and top_bottom_exceeds_bounds:
        return "side_by_side", side_by_side_w, side_by_side_h
    elif not top_bottom_exceeds_bounds and side_by_side_exceeds_bounds:
        return "top_bottom", top_bottom_w, top_bottom_h
    elif side_by_side_excess <= top_bottom_excess:
        return "side_by_side", side_by_side_w, side_by_side_h
    else:
        return "top_bottom", top_bottom_w, top_bottom_h


def create_comparison_video(
    original_video_path: str,
    upscaled_video_path: str, 
    output_path: str,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create a comparison video combining original and upscaled videos.
    
    Args:
        original_video_path: Path to the original video
        upscaled_video_path: Path to the upscaled video
        output_path: Path where comparison video will be saved
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting (CRF/CQ)
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        # Get video resolutions
        original_w, original_h = util_get_video_resolution(original_video_path)
        upscaled_w, upscaled_h = util_get_video_resolution(upscaled_video_path)
        
        logger.info(f"Original video resolution: {original_w}x{original_h}")
        logger.info(f"Upscaled video resolution: {upscaled_w}x{upscaled_h}")
        
        # Determine the best layout
        layout, final_w, final_h = determine_comparison_layout(original_w, original_h, upscaled_w, upscaled_h)
        
        logger.info(f"Comparison layout: {layout}, final resolution: {final_w}x{final_h}")
        
        # Prepare scaling parameters for both videos
        if layout == "side_by_side":
            # Scale both videos to the same height (max of both)
            target_height = max(original_h, upscaled_h)
            original_scale_w = int(original_w * target_height / original_h)
            original_scale_h = target_height
            upscaled_scale_w = int(upscaled_w * target_height / upscaled_h)  
            upscaled_scale_h = target_height
            
            # FFmpeg filter for side-by-side layout
            video_filter = (
                f"[0:v]scale={original_scale_w}:{original_scale_h}[left];"
                f"[1:v]scale={upscaled_scale_w}:{upscaled_scale_h}[right];"
                f"[left][right]hstack=inputs=2[output]"
            )
            
        else:  # top_bottom
            # Scale both videos to the same width (max of both)
            target_width = max(original_w, upscaled_w)
            original_scale_w = target_width
            original_scale_h = int(original_h * target_width / original_w)
            upscaled_scale_w = target_width
            upscaled_scale_h = int(upscaled_h * target_width / upscaled_w)
            
            # FFmpeg filter for top-bottom layout
            video_filter = (
                f"[0:v]scale={original_scale_w}:{original_scale_h}[top];"
                f"[1:v]scale={upscaled_scale_w}:{upscaled_scale_h}[bottom];"
                f"[top][bottom]vstack=inputs=2[output]"
            )
        
        # Build FFmpeg command
        codec = "h264_nvenc" if ffmpeg_use_gpu else "libx264"
        quality_param = f"-cq {ffmpeg_quality}" if ffmpeg_use_gpu else f"-crf {ffmpeg_quality}"
        
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{original_video_path}" -i "{upscaled_video_path}" '
            f'-filter_complex "{video_filter}" -map "[output]" '
            f'-map 0:a? -c:v {codec} {quality_param} -preset {ffmpeg_preset} '
            f'-c:a copy "{output_path}"'
        )
        
        logger.info(f"Creating comparison video: {layout} layout")
        util_run_ffmpeg_command(ffmpeg_cmd, "Comparison Video Creation", logger=logger)
        
        if os.path.exists(output_path):
            logger.info(f"Comparison video created successfully: {output_path}")
            return True
        else:
            logger.error("Comparison video creation failed - output file not found")
            return False
            
    except Exception as e:
        logger.error(f"Error creating comparison video: {e}")
        return False


def get_comparison_output_path(original_output_path: str) -> str:
    """
    Generate the output path for comparison video based on the original output path.
    
    Args:
        original_output_path: Path to the upscaled video
        
    Returns:
        str: Path for the comparison video
    """
    base_path, ext = os.path.splitext(original_output_path)
    return f"{base_path}_comparison{ext}" 