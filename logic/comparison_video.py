import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging
from typing import Tuple, Optional, List

from .ffmpeg_utils import run_ffmpeg_command as util_run_ffmpeg_command
from .file_utils import get_video_resolution as util_get_video_resolution


def determine_comparison_layout(original_w: int, original_h: int, upscaled_w: int, upscaled_h: int, max_dimension: int = 4096, force_layout: Optional[str] = None) -> Tuple[str, int, int, bool]:
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
        force_layout: Optional manual override - "side_by_side" or "top_bottom"
        
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

    # Handle manual layout override
    if force_layout in ["side_by_side", "top_bottom"]:
        if force_layout == "side_by_side" and not sbs_exceeds_hw_limit:
            chosen_layout, combined_w, combined_h = "side_by_side", sbs_final_w, sbs_final_h
        elif force_layout == "top_bottom" and not tb_exceeds_hw_limit:
            chosen_layout, combined_w, combined_h = "top_bottom", tb_final_w, tb_final_h
        elif force_layout == "side_by_side" and sbs_exceeds_hw_limit:
            # User wants side-by-side but it exceeds hardware limits, warn and fall back to auto
            chosen_layout = None  # Will trigger automatic selection below
        elif force_layout == "top_bottom" and tb_exceeds_hw_limit:
            # User wants top-bottom but it exceeds hardware limits, warn and fall back to auto
            chosen_layout = None  # Will trigger automatic selection below
    else:
        chosen_layout = None  # Use automatic selection
    
    # Automatic layout selection (when no force_layout or hardware limits exceeded)
    if chosen_layout is None:
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
    force_layout: Optional[str] = None,
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
        force_layout: Optional manual layout override ("side_by_side" or "top_bottom").
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
            orig_w, orig_h, upscaled_w, upscaled_h, NVENC_MAX_DIMENSION, force_layout=force_layout
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
        
        # Get encoding configuration with automatic NVENC fallback
        from .nvenc_utils import get_nvenc_fallback_encoding_config
        
        encoding_config = get_nvenc_fallback_encoding_config(
            use_gpu=ffmpeg_use_gpu,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality,
            width=combined_final_w,
            height=combined_final_h,
            logger=logger
        )
        
        # Primary encoding attempt
        encoding_attempts.append({
            'name': f"Primary ({encoding_config['codec'].upper()})",
            'codec': encoding_config['codec'],
            'preset': encoding_config['preset'],
            'quality_param': f"-{encoding_config['quality_param']} {encoding_config['quality_value']}",
            'use_gpu': encoding_config['codec'] == 'h264_nvenc'
        })
        
        # Fallback to CPU if primary was GPU
        if encoding_config['codec'] == 'h264_nvenc':
            encoding_attempts.append({
                'name': 'CPU Fallback (libx264)',
                'codec': 'libx264',
                'preset': ffmpeg_preset,
                'quality_param': f'-crf {ffmpeg_quality}',
                'use_gpu': False
            })
        
        # Try each encoding approach
        for attempt_idx, encoding_config in enumerate(encoding_attempts):
            try:
                # Use the preset from the encoding config if available, otherwise use the actual preset handling
                actual_preset = encoding_config.get('preset', ffmpeg_preset)
                
                # For h264_nvenc, we need to add the :v suffix
                if encoding_config["codec"] == "h264_nvenc":
                    preset_param = f"-preset:v {actual_preset}"
                else:
                    preset_param = f"-preset {actual_preset}"
                
                # Set GOP size based on encoder type
                if encoding_config["codec"] == "h264_nvenc":
                    gop_size = "2"
                    keyint_min = "2"
                else:
                    gop_size = "1"
                    keyint_min = "1"
                
                ffmpeg_cmd = (
                    f'ffmpeg -y -i "{original_video_path}" -i "{upscaled_video_path}" '
                    f'-filter_complex "{video_filter}" -map "[output]" '
                    f'-map 0:a? -c:v {encoding_config["codec"]} {encoding_config["quality_param"]} {preset_param} '
                    f'-bf 0 -g {gop_size} -keyint_min {keyint_min} -c:a copy "{output_path}"'
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
            
            # Conservative fallback command (uses CPU encoding, so GOP size 1 is fine)
            fallback_cmd = (
                f'ffmpeg -y -i "{original_video_path}" -i "{upscaled_video_path}" '
                f'-filter_complex "{fallback_filter}" -map "[output]" '
                f'-c:v libx264 -crf 28 -preset ultrafast -pix_fmt yuv420p -bf 0 -g 1 -keyint_min 1 '
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


def determine_multi_video_layout(video_paths: List[str], max_dimension: int = 4096, force_layout: Optional[str] = None, logger: Optional[logging.Logger] = None) -> Tuple[str, int, int, bool, str]:
    """
    Determine the best layout for multiple videos (2-4 videos).
    
    Args:
        video_paths: List of video file paths (2-4 videos)
        max_dimension: Maximum width or height allowed (for NVENC: 4096px)  
        force_layout: Optional manual override layout choice
        logger: Logger instance
        
    Returns:
        Tuple of (layout_choice, combined_video_width, combined_video_height, needs_downscaling, filter_complex).
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    num_videos = len(video_paths)
    if num_videos < 2 or num_videos > 4:
        raise ValueError(f"Multi-video comparison supports 2-4 videos, got {num_videos}")
    
    # Get video resolutions
    video_resolutions = []
    for path in video_paths:
        try:
            h, w = util_get_video_resolution(path, logger=logger)
            if w == 0 or h == 0:
                raise ValueError(f"Invalid resolution for video: {path}")
            video_resolutions.append((w, h))
        except Exception as e:
            logger.error(f"Failed to get resolution for {path}: {e}")
            raise
    
    logger.info(f"Video resolutions: {[f'{w}x{h}' for w, h in video_resolutions]}")
    
    # Handle 2-video case with existing logic
    if num_videos == 2:
        layout_choice, combined_w, combined_h, needs_downscaling = determine_comparison_layout(
            video_resolutions[0][0], video_resolutions[0][1],
            video_resolutions[1][0], video_resolutions[1][1],
            max_dimension, force_layout
        )
        filter_complex = _create_2_video_filter(video_resolutions, layout_choice, combined_w, combined_h)
        return layout_choice, combined_w, combined_h, needs_downscaling, filter_complex
    
    # Define layout options for 3 and 4 videos
    layout_options = []
    
    if num_videos == 3:
        layout_options = [
            ("3x1_horizontal", _calculate_3x1_horizontal),
            ("1x3_vertical", _calculate_1x3_vertical), 
            ("L_shape", _calculate_L_shape)
        ]
    elif num_videos == 4:
        layout_options = [
            ("2x2_grid", _calculate_2x2_grid),
            ("4x1_horizontal", _calculate_4x1_horizontal),
            ("1x4_vertical", _calculate_1x4_vertical)
        ]
    
    # If force_layout is specified, try that first
    if force_layout:
        for layout_name, calc_func in layout_options:
            if layout_name == force_layout:
                try:
                    combined_w, combined_h, filter_complex = calc_func(video_resolutions, max_dimension)
                    needs_downscaling = combined_w > max_dimension or combined_h > max_dimension
                    if needs_downscaling:
                        combined_w, combined_h, _ = calculate_downscale_dimensions(combined_w, combined_h, max_dimension)
                    return force_layout, combined_w, combined_h, needs_downscaling, filter_complex
                except Exception as e:
                    logger.warning(f"Failed to use forced layout {force_layout}: {e}")
                    break
    
    # Auto-select best layout
    best_layout = None
    best_combined_w, best_combined_h = 0, 0
    best_filter = ""
    min_area = float('inf')
    
    for layout_name, calc_func in layout_options:
        try:
            combined_w, combined_h, filter_complex = calc_func(video_resolutions, max_dimension)
            area = combined_w * combined_h
            
            # Prefer layouts that don't exceed hardware limits
            exceeds_limit = combined_w > max_dimension or combined_h > max_dimension
            if not exceeds_limit and area < min_area:
                min_area = area
                best_layout = layout_name
                best_combined_w, best_combined_h = combined_w, combined_h
                best_filter = filter_complex
                
        except Exception as e:
            logger.warning(f"Failed to calculate layout {layout_name}: {e}")
            continue
    
    if best_layout is None:
        # Fallback to first available layout with downscaling
        layout_name, calc_func = layout_options[0]
        combined_w, combined_h, filter_complex = calc_func(video_resolutions, max_dimension)
        best_layout = layout_name
        best_combined_w, best_combined_h = combined_w, combined_h
        best_filter = filter_complex
    
    needs_downscaling = best_combined_w > max_dimension or best_combined_h > max_dimension
    if needs_downscaling:
        best_combined_w, best_combined_h, _ = calculate_downscale_dimensions(best_combined_w, best_combined_h, max_dimension)
    
    # Ensure even dimensions
    best_combined_w = (best_combined_w // 2) * 2
    best_combined_h = (best_combined_h // 2) * 2
    
    logger.info(f"Selected layout: {best_layout}, final dimensions: {best_combined_w}x{best_combined_h}")
    
    return best_layout, best_combined_w, best_combined_h, needs_downscaling, best_filter


def _create_2_video_filter(video_resolutions: List[Tuple[int, int]], layout_choice: str, combined_w: int, combined_h: int) -> str:
    """Create FFmpeg filter for 2-video comparison."""
    orig_w, orig_h = video_resolutions[0]
    upscaled_w, upscaled_h = video_resolutions[1]
    
    if layout_choice == "side_by_side":
        # Calculate individual video dimensions for side-by-side
        common_h = combined_h
        scaled_orig_w = int(round(orig_w * common_h / orig_h)) if orig_h > 0 else 0
        scaled_upscaled_w = int(round(upscaled_w * common_h / upscaled_h)) if upscaled_h > 0 else 0
        
        scaled_orig_w = (scaled_orig_w // 2) * 2
        scaled_upscaled_w = (scaled_upscaled_w // 2) * 2
        
        return (
            f"[0:v]scale={scaled_orig_w}:{common_h},setsar=1[left];"
            f"[1:v]scale={scaled_upscaled_w}:{common_h},setsar=1[right];"
            f"[left][right]hstack=inputs=2[output]"
        )
    else:  # top_bottom
        # Calculate individual video dimensions for top-bottom
        common_w = combined_w
        scaled_orig_h = int(round(orig_h * common_w / orig_w)) if orig_w > 0 else 0
        scaled_upscaled_h = int(round(upscaled_h * common_w / upscaled_w)) if upscaled_w > 0 else 0
        
        scaled_orig_h = (scaled_orig_h // 2) * 2
        scaled_upscaled_h = (scaled_upscaled_h // 2) * 2
        
        return (
            f"[0:v]scale={common_w}:{scaled_orig_h},setsar=1[top];"
            f"[1:v]scale={common_w}:{scaled_upscaled_h},setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2[output]"
        )


def _calculate_3x1_horizontal(video_resolutions: List[Tuple[int, int]], max_dimension: int) -> Tuple[int, int, str]:
    """Calculate dimensions and filter for 3x1 horizontal layout."""
    # Find common height (max of all heights)
    common_h = max(h for w, h in video_resolutions)
    
    # Scale all videos to common height and calculate total width
    scaled_videos = []
    total_width = 0
    
    for i, (w, h) in enumerate(video_resolutions):
        scaled_w = int(round(w * common_h / h)) if h > 0 else 0
        scaled_w = (scaled_w // 2) * 2
        scaled_videos.append((scaled_w, common_h))
        total_width += scaled_w
    
    # Create filter complex
    filter_parts = []
    input_labels = []
    
    for i, (scaled_w, scaled_h) in enumerate(scaled_videos):
        label = f"v{i}"
        filter_parts.append(f"[{i}:v]scale={scaled_w}:{scaled_h},setsar=1[{label}]")
        input_labels.append(f"[{label}]")
    
    filter_complex = ";".join(filter_parts) + ";" + "".join(input_labels) + f"hstack=inputs=3[output]"
    
    return total_width, common_h, filter_complex


def _calculate_1x3_vertical(video_resolutions: List[Tuple[int, int]], max_dimension: int) -> Tuple[int, int, str]:
    """Calculate dimensions and filter for 1x3 vertical layout."""
    # Find common width (max of all widths)
    common_w = max(w for w, h in video_resolutions)
    
    # Scale all videos to common width and calculate total height
    scaled_videos = []
    total_height = 0
    
    for i, (w, h) in enumerate(video_resolutions):
        scaled_h = int(round(h * common_w / w)) if w > 0 else 0
        scaled_h = (scaled_h // 2) * 2
        scaled_videos.append((common_w, scaled_h))
        total_height += scaled_h
    
    # Create filter complex
    filter_parts = []
    input_labels = []
    
    for i, (scaled_w, scaled_h) in enumerate(scaled_videos):
        label = f"v{i}"
        filter_parts.append(f"[{i}:v]scale={scaled_w}:{scaled_h},setsar=1[{label}]")
        input_labels.append(f"[{label}]")
    
    filter_complex = ";".join(filter_parts) + ";" + "".join(input_labels) + f"vstack=inputs=3[output]"
    
    return common_w, total_height, filter_complex


def _calculate_L_shape(video_resolutions: List[Tuple[int, int]], max_dimension: int) -> Tuple[int, int, str]:
    """Calculate dimensions and filter for L-shape layout (2 videos on top, 1 on bottom full width)."""
    # For L-shape: top two videos side by side, bottom video full width
    # Calculate dimensions for top row (first 2 videos)
    top_common_h = max(video_resolutions[0][1], video_resolutions[1][1])
    
    top_v1_w = int(round(video_resolutions[0][0] * top_common_h / video_resolutions[0][1])) if video_resolutions[0][1] > 0 else 0
    top_v2_w = int(round(video_resolutions[1][0] * top_common_h / video_resolutions[1][1])) if video_resolutions[1][1] > 0 else 0
    
    top_v1_w = (top_v1_w // 2) * 2
    top_v2_w = (top_v2_w // 2) * 2
    
    top_row_width = top_v1_w + top_v2_w
    
    # Bottom video uses the full width of top row
    bottom_v3_w = top_row_width
    bottom_v3_h = int(round(video_resolutions[2][1] * bottom_v3_w / video_resolutions[2][0])) if video_resolutions[2][0] > 0 else 0
    bottom_v3_h = (bottom_v3_h // 2) * 2
    
    total_width = top_row_width
    total_height = top_common_h + bottom_v3_h
    
    # Create filter complex
    filter_complex = (
        f"[0:v]scale={top_v1_w}:{top_common_h},setsar=1[v1];"
        f"[1:v]scale={top_v2_w}:{top_common_h},setsar=1[v2];"
        f"[2:v]scale={bottom_v3_w}:{bottom_v3_h},setsar=1[v3];"
        f"[v1][v2]hstack=inputs=2[top_row];"
        f"[top_row][v3]vstack=inputs=2[output]"
    )
    
    return total_width, total_height, filter_complex


def _calculate_2x2_grid(video_resolutions: List[Tuple[int, int]], max_dimension: int) -> Tuple[int, int, str]:
    """Calculate dimensions and filter for 2x2 grid layout."""
    # For 2x2 grid: arrange 4 videos in a 2x2 grid
    # Calculate common dimensions for each row
    
    # Top row (videos 0, 1)
    top_common_h = max(video_resolutions[0][1], video_resolutions[1][1])
    top_v1_w = int(round(video_resolutions[0][0] * top_common_h / video_resolutions[0][1])) if video_resolutions[0][1] > 0 else 0
    top_v2_w = int(round(video_resolutions[1][0] * top_common_h / video_resolutions[1][1])) if video_resolutions[1][1] > 0 else 0
    
    # Bottom row (videos 2, 3)
    bottom_common_h = max(video_resolutions[2][1], video_resolutions[3][1])
    bottom_v3_w = int(round(video_resolutions[2][0] * bottom_common_h / video_resolutions[2][1])) if video_resolutions[2][1] > 0 else 0
    bottom_v4_w = int(round(video_resolutions[3][0] * bottom_common_h / video_resolutions[3][1])) if video_resolutions[3][1] > 0 else 0
    
    # Make all widths even
    top_v1_w = (top_v1_w // 2) * 2
    top_v2_w = (top_v2_w // 2) * 2  
    bottom_v3_w = (bottom_v3_w // 2) * 2
    bottom_v4_w = (bottom_v4_w // 2) * 2
    
    # Calculate final grid dimensions
    top_row_width = top_v1_w + top_v2_w
    bottom_row_width = bottom_v3_w + bottom_v4_w
    total_width = max(top_row_width, bottom_row_width)
    total_height = top_common_h + bottom_common_h
    
    # Adjust individual video widths to match grid alignment
    if top_row_width < total_width:
        # Scale top row videos proportionally to fill total width
        scale_factor = total_width / top_row_width
        top_v1_w = int(top_v1_w * scale_factor)
        top_v2_w = int(top_v2_w * scale_factor)
        top_v1_w = (top_v1_w // 2) * 2
        top_v2_w = (top_v2_w // 2) * 2
    
    if bottom_row_width < total_width:
        # Scale bottom row videos proportionally to fill total width
        scale_factor = total_width / bottom_row_width
        bottom_v3_w = int(bottom_v3_w * scale_factor)
        bottom_v4_w = int(bottom_v4_w * scale_factor)
        bottom_v3_w = (bottom_v3_w // 2) * 2
        bottom_v4_w = (bottom_v4_w // 2) * 2
    
    # Create filter complex
    filter_complex = (
        f"[0:v]scale={top_v1_w}:{top_common_h},setsar=1[v1];"
        f"[1:v]scale={top_v2_w}:{top_common_h},setsar=1[v2];"
        f"[2:v]scale={bottom_v3_w}:{bottom_common_h},setsar=1[v3];"
        f"[3:v]scale={bottom_v4_w}:{bottom_common_h},setsar=1[v4];"
        f"[v1][v2]hstack=inputs=2[top_row];"
        f"[v3][v4]hstack=inputs=2[bottom_row];"
        f"[top_row][bottom_row]vstack=inputs=2[output]"
    )
    
    return total_width, total_height, filter_complex


def _calculate_4x1_horizontal(video_resolutions: List[Tuple[int, int]], max_dimension: int) -> Tuple[int, int, str]:
    """Calculate dimensions and filter for 4x1 horizontal layout."""
    # Find common height (max of all heights)
    common_h = max(h for w, h in video_resolutions)
    
    # Scale all videos to common height and calculate total width
    scaled_videos = []
    total_width = 0
    
    for i, (w, h) in enumerate(video_resolutions):
        scaled_w = int(round(w * common_h / h)) if h > 0 else 0
        scaled_w = (scaled_w // 2) * 2
        scaled_videos.append((scaled_w, common_h))
        total_width += scaled_w
    
    # Create filter complex
    filter_parts = []
    input_labels = []
    
    for i, (scaled_w, scaled_h) in enumerate(scaled_videos):
        label = f"v{i}"
        filter_parts.append(f"[{i}:v]scale={scaled_w}:{scaled_h},setsar=1[{label}]")
        input_labels.append(f"[{label}]")
    
    filter_complex = ";".join(filter_parts) + ";" + "".join(input_labels) + f"hstack=inputs=4[output]"
    
    return total_width, common_h, filter_complex


def _calculate_1x4_vertical(video_resolutions: List[Tuple[int, int]], max_dimension: int) -> Tuple[int, int, str]:
    """Calculate dimensions and filter for 1x4 vertical layout."""
    # Find common width (max of all widths)
    common_w = max(w for w, h in video_resolutions)
    
    # Scale all videos to common width and calculate total height
    scaled_videos = []
    total_height = 0
    
    for i, (w, h) in enumerate(video_resolutions):
        scaled_h = int(round(h * common_w / w)) if w > 0 else 0
        scaled_h = (scaled_h // 2) * 2
        scaled_videos.append((common_w, scaled_h))
        total_height += scaled_h
    
    # Create filter complex
    filter_parts = []
    input_labels = []
    
    for i, (scaled_w, scaled_h) in enumerate(scaled_videos):
        label = f"v{i}"
        filter_parts.append(f"[{i}:v]scale={scaled_w}:{scaled_h},setsar=1[{label}]")
        input_labels.append(f"[{label}]")
    
    filter_complex = ";".join(filter_parts) + ";" + "".join(input_labels) + f"vstack=inputs=4[output]"
    
    return common_w, total_height, filter_complex


def create_multi_video_comparison(
    video_paths: List[str],
    output_path: str,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    force_layout: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create a comparison video from multiple videos (2-4 videos).
    
    Args:
        video_paths: List of video file paths (2-4 videos)
        output_path: Path where comparison video will be saved
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting (CRF/CQ)
        ffmpeg_use_gpu: Whether to use GPU encoding
        force_layout: Optional manual layout override
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(video_paths) < 2 or len(video_paths) > 4:
        logger.error(f"Multi-video comparison supports 2-4 videos, got {len(video_paths)}")
        return False
    
    # Validate all input videos exist
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            logger.error(f"Video {i+1} not found: {path}")
            return False
    
    NVENC_MAX_DIMENSION = 4096
    
    try:
        # Determine layout and get filter complex
        layout_choice, combined_w, combined_h, needs_downscaling, filter_complex = determine_multi_video_layout(
            video_paths, NVENC_MAX_DIMENSION, force_layout, logger
        )
        
        logger.info(f"Using layout: {layout_choice}, dimensions: {combined_w}x{combined_h}")
        if needs_downscaling:
            logger.warning(f"Video will be downscaled to fit hardware limits")
        
        # Build FFmpeg command with multiple inputs
        input_args = []
        for path in video_paths:
            input_args.extend(["-i", f'"{path}"'])
        
        # Get encoding configuration with automatic NVENC fallback
        from .nvenc_utils import get_nvenc_fallback_encoding_config
        
        encoding_config = get_nvenc_fallback_encoding_config(
            use_gpu=ffmpeg_use_gpu,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality,
            width=combined_w,
            height=combined_h,
            logger=logger
        )
        
        # Try different encoding approaches
        encoding_attempts = []
        
        # Primary encoding attempt
        encoding_attempts.append({
            'name': f"Primary ({encoding_config['codec'].upper()})",
            'codec': encoding_config['codec'],
            'preset': encoding_config['preset'],
            'quality_param': f"-{encoding_config['quality_param']} {encoding_config['quality_value']}",
        })
        
        # Fallback to CPU if primary was GPU
        if encoding_config['codec'] == 'h264_nvenc':
            encoding_attempts.append({
                'name': 'CPU Fallback (libx264)',
                'codec': 'libx264',
                'preset': ffmpeg_preset,
                'quality_param': f'-crf {ffmpeg_quality}',
            })
        
        # Try each encoding approach
        for attempt_idx, encoding_config in enumerate(encoding_attempts):
            try:
                # Use the preset from the encoding config if available, otherwise use the actual preset handling
                actual_preset = encoding_config.get('preset', ffmpeg_preset)
                
                # For h264_nvenc, we need to add the :v suffix
                if encoding_config["codec"] == "h264_nvenc":
                    preset_param = f"-preset:v {actual_preset}"
                else:
                    preset_param = f"-preset {actual_preset}"
                
                ffmpeg_cmd = (
                    f'ffmpeg -y {" ".join(input_args)} '
                    f'-filter_complex "{filter_complex}" -map "[output]" '
                    f'-map 0:a? -c:v {encoding_config["codec"]} {encoding_config["quality_param"]} {preset_param} '
                    f'-c:a copy "{output_path}"'
                )
                
                logger.info(f"Attempt {attempt_idx + 1}: {encoding_config['name']} encoding")
                logger.info(f"FFmpeg command: {ffmpeg_cmd}")
                
                # Try to run the command
                cmd_success = util_run_ffmpeg_command(ffmpeg_cmd, f"Multi-Video Comparison ({encoding_config['name']})", logger=logger, raise_on_error=False)
                
                if not cmd_success:
                    logger.warning(f"❌ {encoding_config['name']} FFmpeg command failed")
                    continue
                
                # Check if output was created successfully
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"✅ Multi-video comparison created successfully with {encoding_config['name']}: {output_path}")
                    return True
                else:
                    logger.warning(f"❌ {encoding_config['name']} encoding produced no output file")
                    
            except Exception as e_encoding:
                logger.warning(f"❌ {encoding_config['name']} encoding failed: {str(e_encoding)}")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                continue
        
        logger.error("❌ All encoding attempts failed for multi-video comparison")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during multi-video comparison creation: {e}", exc_info=True)
        return False