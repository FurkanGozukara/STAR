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
    for the combined comparison video.
    
    The function aims to choose a layout that fits well within a 1920x1080 target,
    or otherwise prefers the layout that results in a smaller overall area.
    
    Args:
        original_w, original_h: Actual original video width and height.
        upscaled_w, upscaled_h: Actual upscaled video width and height.
        
    Returns:
        Tuple of (layout_choice, combined_video_width, combined_video_height).
        'layout_choice' is "side_by_side" or "top_bottom".
        'combined_video_width' and 'combined_video_height' are the dimensions
        of the final stacked video, ensured to be even.
    """
    
    # --- Calculate dimensions if Side-by-Side (SBS) ---
    # For SBS, both videos are scaled to the maximum height of the two.
    sbs_common_h = max(original_h, upscaled_h)
    # Scale original video to this common height, maintaining aspect ratio
    sbs_orig_w_scaled = int(round(original_w * sbs_common_h / original_h)) if original_h > 0 else 0
    sbs_orig_h_scaled = sbs_common_h
    # Scale upscaled video to this common height, maintaining aspect ratio
    sbs_upscaled_w_scaled = int(round(upscaled_w * sbs_common_h / upscaled_h)) if upscaled_h > 0 else 0
    sbs_upscaled_h_scaled = sbs_common_h
    
    # Final dimensions for SBS layout
    sbs_final_w = sbs_orig_w_scaled + sbs_upscaled_w_scaled
    sbs_final_h = sbs_common_h

    # --- Calculate dimensions if Top-Bottom (TB) ---
    # For TB, both videos are scaled to the maximum width of the two.
    tb_common_w = max(original_w, upscaled_w)
    # Scale original video to this common width, maintaining aspect ratio
    tb_orig_w_scaled = tb_common_w
    tb_orig_h_scaled = int(round(original_h * tb_common_w / original_w)) if original_w > 0 else 0
    # Scale upscaled video to this common width, maintaining aspect ratio
    tb_upscaled_w_scaled = tb_common_w
    tb_upscaled_h_scaled = int(round(upscaled_h * tb_common_w / upscaled_w)) if upscaled_w > 0 else 0

    # Final dimensions for TB layout
    tb_final_w = tb_common_w
    tb_final_h = tb_orig_h_scaled + tb_upscaled_h_scaled

    # --- Decision Logic ---
    TARGET_W, TARGET_H = 1920, 1080 # Target reference resolution
    NVENC_MAX_WIDTH = 4096  # NVENC hardware encoder maximum width limitation
    
    sbs_exceeds_target = sbs_final_w > TARGET_W or sbs_final_h > TARGET_H
    tb_exceeds_target = tb_final_w > TARGET_W or tb_final_h > TARGET_H
    
    # Additional check for NVENC width limitations
    sbs_exceeds_nvenc = sbs_final_w > NVENC_MAX_WIDTH
    tb_exceeds_nvenc = tb_final_w > NVENC_MAX_WIDTH
    
    chosen_layout: str
    combined_w: int
    combined_h: int

    # Prioritize avoiding NVENC width limits to prevent encoding failures
    if sbs_exceeds_nvenc and not tb_exceeds_nvenc:
        # SBS exceeds NVENC width limit, TB does not. Choose TB to avoid encoding issues.
        chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
    elif not sbs_exceeds_nvenc and tb_exceeds_nvenc:
        # TB exceeds NVENC width limit, SBS does not. Choose SBS.
        chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
    elif not sbs_exceeds_target and tb_exceeds_target:
        # SBS fits within target, TB does not. Choose SBS.
        chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
    elif sbs_exceeds_target and not tb_exceeds_target:
        # TB fits within target, SBS does not. Choose TB.
        chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
    else:
        # Both fit, or both exceed target/NVENC limits. Choose based on smaller area.
        # If areas are very similar, side-by-side is often preferred visually for comparison.
        sbs_area = sbs_final_w * sbs_final_h
        tb_area = tb_final_w * tb_final_h
        
        # Prefer the layout with smaller area. If areas are equal, default to SBS.
        if tb_area < sbs_area:
            chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
        else: # sbs_area <= tb_area
            chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
            
    # Ensure final dimensions are even numbers for video codecs
    combined_w = (combined_w // 2) * 2
    combined_h = (combined_h // 2) * 2
    
    return chosen_layout, combined_w, combined_h


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
        original_video_path: Path to the original video.
        upscaled_video_path: Path to the upscaled video.
        output_path: Path where comparison video will be saved.
        ffmpeg_preset: FFmpeg encoding preset.
        ffmpeg_quality: FFmpeg quality setting (CRF/CQ).
        ffmpeg_use_gpu: Whether to use GPU encoding.
        logger: Logger instance.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    
    if logger is None:
        logger = logging.getLogger(__name__) # Basic fallback logger
        
    try:
        # Get video resolutions (util_get_video_resolution returns Height, Width)
        orig_h, orig_w = util_get_video_resolution(original_video_path, logger=logger)
        upscaled_h, upscaled_w = util_get_video_resolution(upscaled_video_path, logger=logger)
        
        if orig_w == 0 or orig_h == 0 or upscaled_w == 0 or upscaled_h == 0:
            logger.error("Failed to get valid dimensions for one or both videos. Cannot create comparison.")
            return False

        logger.info(f"Original video resolution (WxH): {orig_w}x{orig_h}")
        logger.info(f"Upscaled video resolution (WxH): {upscaled_w}x{upscaled_h}")
        
        # Determine the best layout and the final dimensions of the *combined* video
        layout_choice, combined_final_w, combined_final_h = determine_comparison_layout(
            orig_w, orig_h, upscaled_w, upscaled_h
        )
        
        logger.info(f"Chosen comparison layout: {layout_choice}, "
                    f"final combined video resolution (WxH): {combined_final_w}x{combined_final_h}")
        
        # Prepare scaling parameters for individual videos within the combined frame
        # These are the dimensions each video needs to be scaled to *before* stacking.
        # FFmpeg scale filter format is scale=width:height

        scaled_orig_w_ffmpeg: int
        scaled_orig_h_ffmpeg: int
        scaled_upscaled_w_ffmpeg: int
        scaled_upscaled_h_ffmpeg: int

        if layout_choice == "side_by_side":
            # For SBS, both videos are scaled to the height of the combined video.
            # The width of each scaled video is calculated to maintain its aspect ratio.
            common_h_for_stacking = combined_final_h 
            
            scaled_orig_w_ffmpeg = int(round(orig_w * common_h_for_stacking / orig_h)) if orig_h > 0 else 0
            scaled_orig_h_ffmpeg = common_h_for_stacking
            
            scaled_upscaled_w_ffmpeg = int(round(upscaled_w * common_h_for_stacking / upscaled_h)) if upscaled_h > 0 else 0
            scaled_upscaled_h_ffmpeg = common_h_for_stacking
            
            # Ensure widths are even for ffmpeg
            scaled_orig_w_ffmpeg = (scaled_orig_w_ffmpeg // 2) * 2
            scaled_upscaled_w_ffmpeg = (scaled_upscaled_w_ffmpeg // 2) * 2

            video_filter = (
                f"[0:v]scale={scaled_orig_w_ffmpeg}:{scaled_orig_h_ffmpeg},setsar=1[left];"
                f"[1:v]scale={scaled_upscaled_w_ffmpeg}:{scaled_upscaled_h_ffmpeg},setsar=1[right];"
                f"[left][right]hstack=inputs=2[output]"
            )
            
        else:  # top_bottom
            # For TB, both videos are scaled to the width of the combined video.
            # The height of each scaled video is calculated to maintain its aspect ratio.
            common_w_for_stacking = combined_final_w
            
            scaled_orig_w_ffmpeg = common_w_for_stacking
            scaled_orig_h_ffmpeg = int(round(orig_h * common_w_for_stacking / orig_w)) if orig_w > 0 else 0
            
            scaled_upscaled_w_ffmpeg = common_w_for_stacking
            scaled_upscaled_h_ffmpeg = int(round(upscaled_h * common_w_for_stacking / upscaled_w)) if upscaled_w > 0 else 0

            # Ensure heights are even for ffmpeg
            scaled_orig_h_ffmpeg = (scaled_orig_h_ffmpeg // 2) * 2
            scaled_upscaled_h_ffmpeg = (scaled_upscaled_h_ffmpeg // 2) * 2

            video_filter = (
                f"[0:v]scale={scaled_orig_w_ffmpeg}:{scaled_orig_h_ffmpeg},setsar=1[top];"
                f"[1:v]scale={scaled_upscaled_w_ffmpeg}:{scaled_upscaled_h_ffmpeg},setsar=1[bottom];"
                f"[top][bottom]vstack=inputs=2[output]"
            )
        
        # NVENC has a maximum width limitation of 4096 pixels
        NVENC_MAX_WIDTH = 4096
        
        # Check if NVENC width limit would be exceeded and fallback to CPU if needed
        use_gpu_final = ffmpeg_use_gpu
        if ffmpeg_use_gpu and combined_final_w > NVENC_MAX_WIDTH:
            use_gpu_final = False
            logger.warning(f"Comparison video width ({combined_final_w}px) exceeds NVENC maximum ({NVENC_MAX_WIDTH}px). Falling back to CPU encoding (libx264).")
        
        codec = "h264_nvenc" if use_gpu_final else "libx264"
        quality_param = f"-cq {ffmpeg_quality}" if use_gpu_final else f"-crf {ffmpeg_quality}"
        
        # The -s {combined_final_w}x{combined_final_h} is not strictly necessary if the filter complex
        # correctly produces the desired output dimensions. However, it can be added for explicitness.
        # The hstack/vstack filters implicitly define the output size from their inputs.
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{original_video_path}" -i "{upscaled_video_path}" '
            f'-filter_complex "{video_filter}" -map "[output]" '
            f'-map 0:a? -c:v {codec} {quality_param} -preset {ffmpeg_preset} '
            f'-c:a copy "{output_path}"'
            # Optional: f'-s {combined_final_w}x{combined_final_h} ' # To be explicit about output size
        )
        
        logger.info(f"FFmpeg command for comparison video: {ffmpeg_cmd}")
        util_run_ffmpeg_command(ffmpeg_cmd, "Comparison Video Creation", logger=logger)
        
        if os.path.exists(output_path):
            # Verify final resolution for debugging purposes
            try:
                final_comp_h, final_comp_w = util_get_video_resolution(output_path, logger=logger)
                logger.info(f"Comparison video created: {output_path} (Actual WxH: {final_comp_w}x{final_comp_h})")
            except Exception as e_res_check:
                 logger.info(f"Comparison video created: {output_path} (Could not verify final resolution via ffprobe: {e_res_check})")
            return True
        else:
            logger.error("Comparison video creation failed - output file not found after ffmpeg command.")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error during comparison video creation: {e}", exc_info=True)
        return False


def get_comparison_output_path(original_output_path: str) -> str:
    """
    Generate the output path for a comparison video based on the main upscaled video's path.
    Example: "video.mp4" -> "video_comparison.mp4"
    
    Args:
        original_output_path: Path to the main upscaled video.
        
    Returns:
        str: Path for the corresponding comparison video.
    """
    base_path, ext = os.path.splitext(original_output_path)
    return f"{base_path}_comparison{ext}"