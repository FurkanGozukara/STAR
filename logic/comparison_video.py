import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging
from typing import Tuple, Optional

from .ffmpeg_utils import run_ffmpeg_command as util_run_ffmpeg_command
from .file_utils import get_video_resolution as util_get_video_resolution


def determine_comparison_layout(original_w: int, original_h: int, upscaled_w: int, upscaled_h: int, max_dimension: int = 4096) -> Tuple[str, int, int, bool]:
    """
    Determine the best layout (side-by-side or top_bottom) and final dimensions
    for the combined comparison video.
    
    The function chooses layout based on target aspect ratio fit within 1920x1080:
    - If combined width exceeds 1920px → use top-bottom layout 
    - If combined height exceeds 1080px → use side-by-side layout
    - Otherwise choose the layout that fits better or has smaller area
    
    Args:
        original_w, original_h: Actual original video width and height.
        upscaled_w, upscaled_h: Actual upscaled video width and height.
        max_dimension: Maximum width or height allowed (for NVENC: 4096px)
        
    Returns:
        Tuple of (layout_choice, combined_video_width, combined_video_height, needs_downscaling).
        'layout_choice' is "side_by_side" or "top_bottom".
        'combined_video_width' and 'combined_video_height' are the dimensions
        of the final stacked video, ensured to be even.
        'needs_downscaling' indicates if the result exceeds max_dimension limits.
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
    
    # Check for hardware encoder limitations (both width AND height)
    sbs_exceeds_hw_limit = sbs_final_w > max_dimension or sbs_final_h > max_dimension
    tb_exceeds_hw_limit = tb_final_w > max_dimension or tb_final_h > max_dimension
    
    chosen_layout: str
    combined_w: int
    combined_h: int
    needs_downscaling: bool = False

    # Prioritize avoiding hardware encoder limits to prevent encoding failures
    if sbs_exceeds_hw_limit and not tb_exceeds_hw_limit:
        # SBS exceeds hardware limit, TB does not. Choose TB to avoid encoding issues.
        chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
    elif not sbs_exceeds_hw_limit and tb_exceeds_hw_limit:
        # TB exceeds hardware limit, SBS does not. Choose SBS.
        chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
    else:
        # Apply aspect ratio based decision logic:
        # Target aspect ratio is 16:9 (1920/1080 = 1.777)
        target_aspect_ratio = TARGET_W / TARGET_H  # 1.777
        
        # Calculate aspect ratios for both layouts
        sbs_aspect_ratio = sbs_final_w / sbs_final_h if sbs_final_h > 0 else float('inf')
        tb_aspect_ratio = tb_final_w / tb_final_h if tb_final_h > 0 else float('inf')
        
        # If side-by-side aspect ratio is too wide → use top-bottom
        # If top-bottom aspect ratio is too tall → use side-by-side
        
        if sbs_aspect_ratio > target_aspect_ratio and tb_aspect_ratio <= target_aspect_ratio:
            # Side-by-side is too wide, top-bottom fits better → use top-bottom
            chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
        elif tb_aspect_ratio < target_aspect_ratio and sbs_aspect_ratio >= target_aspect_ratio:
            # Top-bottom is too tall, side-by-side fits better → use side-by-side
            chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
        else:
            # Both fit within target aspect ratio or both exceed - choose based on which is closer to target
            sbs_aspect_diff = abs(sbs_aspect_ratio - target_aspect_ratio)
            tb_aspect_diff = abs(tb_aspect_ratio - target_aspect_ratio)
            
            if tb_aspect_diff < sbs_aspect_diff:
                chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
            else:
                chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
    
    # Check if the chosen layout still exceeds hardware limits
    if combined_w > max_dimension or combined_h > max_dimension:
        needs_downscaling = True
            
    # Ensure final dimensions are even numbers for video codecs
    combined_w = (combined_w // 2) * 2
    combined_h = (combined_h // 2) * 2
    
    return chosen_layout, combined_w, combined_h, needs_downscaling


def calculate_downscale_dimensions(width: int, height: int, max_dimension: int = 4096) -> Tuple[int, int, float]:
    """
    Calculate downscaled dimensions that fit within the maximum dimension constraint.
    
    Args:
        width: Original width
        height: Original height
        max_dimension: Maximum allowed width or height
        
    Returns:
        Tuple of (new_width, new_height, scale_factor)
    """
    if width <= max_dimension and height <= max_dimension:
        return width, height, 1.0
    
    # Calculate scale factor to fit within constraints
    scale_factor = min(max_dimension / width, max_dimension / height)
    
    # Apply scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Ensure even dimensions for video codecs
    new_width = (new_width // 2) * 2
    new_height = (new_height // 2) * 2
    
    return new_width, new_height, scale_factor


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
    Create a comparison video combining original and upscaled videos with robust error handling.
    
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
    
    # Hardware encoder limitations
    NVENC_MAX_DIMENSION = 4096  # NVENC maximum width or height
    
    attempts = []  # Track what we've tried for logging
    
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
        layout_choice, combined_final_w, combined_final_h, needs_downscaling = determine_comparison_layout(
            orig_w, orig_h, upscaled_w, upscaled_h, NVENC_MAX_DIMENSION
        )
        
        logger.info(f"Chosen comparison layout: {layout_choice}, "
                    f"final combined video resolution (WxH): {combined_final_w}x{combined_final_h}")
        
        # Handle downscaling if needed
        downscale_factor = 1.0
        if needs_downscaling:
            combined_final_w, combined_final_h, downscale_factor = calculate_downscale_dimensions(
                combined_final_w, combined_final_h, NVENC_MAX_DIMENSION
            )
            logger.warning(f"Video dimensions exceed hardware limits. Downscaling by {downscale_factor:.2f}x to: {combined_final_w}x{combined_final_h}")
        
        # Prepare scaling parameters for individual videos within the combined frame
        scaled_orig_w_ffmpeg: int
        scaled_orig_h_ffmpeg: int
        scaled_upscaled_w_ffmpeg: int
        scaled_upscaled_h_ffmpeg: int

        if layout_choice == "side_by_side":
            # For SBS, both videos are scaled to the height of the combined video.
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
        
        # Try different encoding approaches with fallbacks
        encoding_attempts = []
        
        # Attempt 1: GPU encoding if requested and dimensions are within limits
        use_gpu_final = ffmpeg_use_gpu
        if ffmpeg_use_gpu and (combined_final_w > NVENC_MAX_DIMENSION or combined_final_h > NVENC_MAX_DIMENSION):
            use_gpu_final = False
            logger.warning(f"Comparison video dimensions ({combined_final_w}x{combined_final_h}) exceed NVENC maximum ({NVENC_MAX_DIMENSION}px). Falling back to CPU encoding (libx264).")
        
        if use_gpu_final:
            encoding_attempts.append({
                'name': 'GPU (NVENC)',
                'codec': 'h264_nvenc',
                'quality_param': f'-cq {ffmpeg_quality}',
                'use_gpu': True
            })
        
        # Attempt 2: CPU encoding
        encoding_attempts.append({
            'name': 'CPU (libx264)',
            'codec': 'libx264', 
            'quality_param': f'-crf {ffmpeg_quality}',
            'use_gpu': False
        })
        
        # Try each encoding approach
        for attempt_idx, encoding_config in enumerate(encoding_attempts):
            try:
                # Map preset for h264_nvenc compatibility
                actual_preset = ffmpeg_preset
                if encoding_config["codec"] == "h264_nvenc":
                    # Map libx264 presets to h264_nvenc presets
                    if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
                        actual_preset = "fast"
                    elif ffmpeg_preset in ["slower", "veryslow"]:
                        actual_preset = "slow"
                    elif ffmpeg_preset == "medium":
                        actual_preset = "medium"  # medium is valid for both
                    else:  # "slow" and others
                        actual_preset = "slow"
                
                ffmpeg_cmd = (
                    f'ffmpeg -y -i "{original_video_path}" -i "{upscaled_video_path}" '
                    f'-filter_complex "{video_filter}" -map "[output]" '
                    f'-map 0:a? -c:v {encoding_config["codec"]} {encoding_config["quality_param"]} -preset {actual_preset} '
                    f'-c:a copy "{output_path}"'
                )
                
                logger.info(f"Attempt {attempt_idx + 1}: {encoding_config['name']} encoding")
                logger.info(f"FFmpeg command for comparison video: {ffmpeg_cmd}")
                
                attempts.append(f"{encoding_config['name']} encoding")
                
                # Try to run the command (don't raise error immediately, allow fallbacks)
                cmd_success = util_run_ffmpeg_command(ffmpeg_cmd, f"Comparison Video Creation ({encoding_config['name']})", logger=logger, raise_on_error=False)
                
                if not cmd_success:
                    logger.warning(f"❌ {encoding_config['name']} FFmpeg command failed")
                    continue
                
                # Check if output was created successfully
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # Verify final resolution for debugging purposes
                    try:
                        final_comp_h, final_comp_w = util_get_video_resolution(output_path, logger=logger)
                        logger.info(f"✅ Comparison video created successfully with {encoding_config['name']}: {output_path} (Actual WxH: {final_comp_w}x{final_comp_h})")
                        if downscale_factor < 1.0:
                            logger.info(f"Note: Video was downscaled by {downscale_factor:.2f}x to fit hardware encoder limitations")
                    except Exception as e_res_check:
                        logger.info(f"✅ Comparison video created successfully with {encoding_config['name']}: {output_path} (Could not verify final resolution: {e_res_check})")
                    return True
                else:
                    logger.warning(f"❌ {encoding_config['name']} encoding produced no output file")
                    
            except Exception as e_encoding:
                logger.warning(f"❌ {encoding_config['name']} encoding failed: {str(e_encoding)}")
                # Clean up any partial output file
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                continue
        
        # If all encoding attempts failed, try one last fallback with very conservative settings
        try:
            logger.warning("🔄 All standard encoding attempts failed. Trying conservative CPU fallback...")
            
            # Use smaller dimensions and basic settings
            fallback_w, fallback_h, fallback_scale = calculate_downscale_dimensions(
                combined_final_w, combined_final_h, max_dimension=2048  # Even more conservative
            )
            
            # Recalculate filter with smaller dimensions
            if layout_choice == "side_by_side":
                fallback_orig_w = int(round(orig_w * fallback_h / orig_h)) if orig_h > 0 else 0
                fallback_upscaled_w = int(round(upscaled_w * fallback_h / upscaled_h)) if upscaled_h > 0 else 0
                
                fallback_orig_w = (fallback_orig_w // 2) * 2
                fallback_upscaled_w = (fallback_upscaled_w // 2) * 2
                
                fallback_filter = (
                    f"[0:v]scale={fallback_orig_w}:{fallback_h},setsar=1[left];"
                    f"[1:v]scale={fallback_upscaled_w}:{fallback_h},setsar=1[right];"
                    f"[left][right]hstack=inputs=2[output]"
                )
            else:
                fallback_orig_h = int(round(orig_h * fallback_w / orig_w)) if orig_w > 0 else 0
                fallback_upscaled_h = int(round(upscaled_h * fallback_w / upscaled_w)) if upscaled_w > 0 else 0
                
                fallback_orig_h = (fallback_orig_h // 2) * 2
                fallback_upscaled_h = (fallback_upscaled_h // 2) * 2
                
                fallback_filter = (
                    f"[0:v]scale={fallback_w}:{fallback_orig_h},setsar=1[top];"
                    f"[1:v]scale={fallback_w}:{fallback_upscaled_h},setsar=1[bottom];"
                    f"[top][bottom]vstack=inputs=2[output]"
                )
            
            # Conservative fallback command
            fallback_cmd = (
                f'ffmpeg -y -i "{original_video_path}" -i "{upscaled_video_path}" '
                f'-filter_complex "{fallback_filter}" -map "[output]" '
                f'-c:v libx264 -crf 28 -preset ultrafast -pix_fmt yuv420p '
                f'"{output_path}"'
            )
            
            logger.info(f"Conservative fallback command: {fallback_cmd}")
            logger.info(f"Fallback dimensions: {fallback_w}x{fallback_h} (scale: {fallback_scale:.2f}x)")
            
            attempts.append("Conservative CPU fallback")
            fallback_success = util_run_ffmpeg_command(fallback_cmd, "Comparison Video Creation (Conservative Fallback)", logger=logger, raise_on_error=False)
            
            if not fallback_success:
                logger.error("❌ Conservative fallback FFmpeg command failed")
                return False
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"✅ Comparison video created with conservative fallback: {output_path}")
                logger.info(f"Note: Video was heavily downscaled to {fallback_w}x{fallback_h} for compatibility")
                return True
            
        except Exception as e_fallback:
            logger.error(f"❌ Even conservative fallback failed: {str(e_fallback)}")
        
        # If we reach here, everything failed
        logger.error(f"❌ Comparison video creation failed after trying all methods: {', '.join(attempts)}")
        return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error during comparison video creation: {e}", exc_info=True)
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