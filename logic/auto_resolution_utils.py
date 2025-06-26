"""
Auto-Resolution Utilities for STAR Video Upscaler

This module provides functions to automatically calculate optimal target resolutions
that maintain the input video's aspect ratio while staying within a specified pixel budget.
"""

import os
import math
from typing import Tuple, Optional
import logging

from .ffmpeg_utils import get_video_info
from .dataclasses import ResolutionConfig


def calculate_optimal_resolution(
    video_width: int, 
    video_height: int, 
    pixel_budget: int,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, int, str]:
    """
    Calculate optimal resolution maintaining aspect ratio within pixel budget.
    
    Args:
        video_width: Input video width in pixels
        video_height: Input video height in pixels  
        pixel_budget: Maximum total pixels allowed (width × height)
        logger: Optional logger instance
        
    Returns:
        Tuple of (optimal_width, optimal_height, status_message)
        
    The calculation finds the largest resolution that:
    1. Maintains the exact aspect ratio of the input video
    2. Does not exceed the pixel budget
    3. Has even dimensions (required for video codecs)
    """
    if logger:
        logger.debug(f"Calculating optimal resolution for {video_width}x{video_height} within {pixel_budget:,} pixel budget")
    
    # Validate inputs
    if video_width <= 0 or video_height <= 0:
        error_msg = f"Invalid video dimensions: {video_width}x{video_height}"
        if logger:
            logger.error(error_msg)
        return video_width, video_height, f"❌ {error_msg}"
    
    if pixel_budget <= 0:
        error_msg = f"Invalid pixel budget: {pixel_budget}"
        if logger:
            logger.error(error_msg)
        return video_width, video_height, f"❌ {error_msg}"
    
    # Calculate aspect ratio
    aspect_ratio = video_width / video_height
    
    # Always calculate optimal dimensions to use the full pixel budget while maintaining aspect ratio
    # Mathematical solution:
    # For aspect_ratio = w/h and pixel_budget = w*h
    # Substitute w = h * aspect_ratio into pixel_budget equation:
    # pixel_budget = (h * aspect_ratio) * h = h² * aspect_ratio
    # Therefore: h = sqrt(pixel_budget / aspect_ratio)
    optimal_height = math.sqrt(pixel_budget / aspect_ratio)
    optimal_width = optimal_height * aspect_ratio
    
    # Round to integers
    optimal_width = int(round(optimal_width))
    optimal_height = int(round(optimal_height))
    
    # Ensure even dimensions for video codec compatibility
    optimal_width = (optimal_width // 2) * 2
    optimal_height = (optimal_height // 2) * 2
    
    # Verify we're still within budget after rounding
    actual_pixels = optimal_width * optimal_height
    if actual_pixels > pixel_budget:
        # If rounding up exceeded budget, round down
        optimal_width = ((optimal_width - 2) // 2) * 2
        optimal_height = ((optimal_height - 2) // 2) * 2
        actual_pixels = optimal_width * optimal_height
    
    # Final validation
    if optimal_width <= 0 or optimal_height <= 0:
        error_msg = f"Calculated invalid dimensions: {optimal_width}x{optimal_height}"
        if logger:
            logger.error(error_msg)
        return video_width, video_height, f"❌ {error_msg}"
    
    # Calculate final aspect ratio and check how close it is to original
    final_aspect_ratio = optimal_width / optimal_height
    aspect_ratio_error = abs(final_aspect_ratio - aspect_ratio) / aspect_ratio * 100
    
    # Success message with detailed information
    input_pixels = video_width * video_height
    pixel_usage_percent = (actual_pixels / pixel_budget) * 100
    
    if input_pixels == actual_pixels:
        status_msg = (f"✅ Optimal matches input: {optimal_width}x{optimal_height} "
                     f"({actual_pixels:,} pixels, {pixel_usage_percent:.1f}% of budget)")
    else:
        status_msg = (f"✅ Auto-calculated: {optimal_width}x{optimal_height} "
                     f"({actual_pixels:,} pixels, {pixel_usage_percent:.1f}% of budget, "
                     f"aspect ratio error: {aspect_ratio_error:.2f}%)")
    
    if logger:
        logger.info(f"Auto-resolution calculation: {video_width}x{video_height} → {optimal_width}x{optimal_height}")
        logger.info(f"Aspect ratio: {aspect_ratio:.3f} → {final_aspect_ratio:.3f} (error: {aspect_ratio_error:.2f}%)")
        logger.info(f"Pixel usage: {actual_pixels:,} / {pixel_budget:,} ({actual_pixels/pixel_budget*100:.1f}%)")
    
    return optimal_width, optimal_height, status_msg


def update_auto_resolution_if_enabled(
    video_path: str, 
    current_resolution_config: ResolutionConfig,
    logger: Optional[logging.Logger] = None
) -> Tuple[ResolutionConfig, str]:
    """
    Main function to update resolution config when video changes.
    
    Args:
        video_path: Path to the video file
        current_resolution_config: Current ResolutionConfig object
        logger: Optional logger instance
        
    Returns:
        Tuple of (updated_ResolutionConfig, status_message)
        
    This function:
    1. Checks if auto-resolution is enabled
    2. Gets video information if a valid video is provided
    3. Calculates optimal resolution if possible
    4. Updates the ResolutionConfig with new values
    5. Returns status message for UI display
    """
    # Create a copy to avoid modifying the original
    updated_config = ResolutionConfig(
        enable_target_res=current_resolution_config.enable_target_res,
        target_res_mode=current_resolution_config.target_res_mode,
        target_h=current_resolution_config.target_h,
        target_w=current_resolution_config.target_w,
        upscale_factor=current_resolution_config.upscale_factor,
        enable_auto_aspect_resolution=current_resolution_config.enable_auto_aspect_resolution,
        auto_resolution_status=current_resolution_config.auto_resolution_status,
        pixel_budget=current_resolution_config.pixel_budget,
        last_video_aspect_ratio=current_resolution_config.last_video_aspect_ratio,
        auto_calculated_h=current_resolution_config.auto_calculated_h,
        auto_calculated_w=current_resolution_config.auto_calculated_w
    )
    
    # If auto-resolution is disabled, return unchanged config
    if not updated_config.enable_auto_aspect_resolution:
        updated_config.auto_resolution_status = "Auto-resolution disabled"
        return updated_config, updated_config.auto_resolution_status
    
    # If no video provided, reset to defaults
    if not video_path or not video_path.strip():
        updated_config.auto_resolution_status = "No video loaded"
        updated_config.last_video_aspect_ratio = 1.0
        updated_config.auto_calculated_h = updated_config.target_h
        updated_config.auto_calculated_w = updated_config.target_w
        return updated_config, updated_config.auto_resolution_status
    
    # Check if video file exists
    if not os.path.exists(video_path):
        error_msg = f"Video file not found: {os.path.basename(video_path)}"
        updated_config.auto_resolution_status = f"❌ {error_msg}"
        if logger:
            logger.warning(f"Auto-resolution: {error_msg}")
        return updated_config, updated_config.auto_resolution_status
    
    try:
        # Get video information
        video_info = get_video_info(video_path, logger=logger)
        
        if not video_info:
            error_msg = f"Could not read video info: {os.path.basename(video_path)}"
            updated_config.auto_resolution_status = f"❌ {error_msg}"
            if logger:
                logger.warning(f"Auto-resolution: {error_msg}")
            return updated_config, updated_config.auto_resolution_status
        
        video_width = video_info.get('width', 0)
        video_height = video_info.get('height', 0)
        
        if video_width <= 0 or video_height <= 0:
            error_msg = f"Invalid video dimensions: {video_width}x{video_height}"
            updated_config.auto_resolution_status = f"❌ {error_msg}"
            if logger:
                logger.warning(f"Auto-resolution: {error_msg}")
            return updated_config, updated_config.auto_resolution_status
        
        # Update pixel budget based on current target resolution
        updated_config.pixel_budget = updated_config.target_h * updated_config.target_w
        
        # Calculate optimal resolution
        optimal_w, optimal_h, status_msg = calculate_optimal_resolution(
            video_width, video_height, updated_config.pixel_budget, logger
        )
        
        # Update config with calculated values
        updated_config.auto_calculated_w = optimal_w
        updated_config.auto_calculated_h = optimal_h
        updated_config.last_video_aspect_ratio = video_width / video_height
        updated_config.auto_resolution_status = status_msg
        
        if logger:
            logger.info(f"Auto-resolution updated: {video_width}x{video_height} → {optimal_w}x{optimal_h}")
            logger.info(f"Pixel budget: {updated_config.pixel_budget:,} pixels")
            logger.info(f"Video aspect ratio: {updated_config.last_video_aspect_ratio:.3f}")
        
        return updated_config, status_msg
        
    except Exception as e:
        error_msg = f"Error calculating auto-resolution: {str(e)}"
        updated_config.auto_resolution_status = f"❌ {error_msg}"
        if logger:
            logger.error(f"Auto-resolution error: {e}", exc_info=True)
        return updated_config, updated_config.auto_resolution_status


def get_effective_resolution(resolution_config: ResolutionConfig) -> Tuple[int, int]:
    """
    Get the effective resolution that should be used for processing.
    
    Args:
        resolution_config: ResolutionConfig object
        
    Returns:
        Tuple of (effective_width, effective_height)
        
    This function determines which resolution values to use:
    - If auto-resolution is enabled, use auto-calculated values
    - Otherwise, use the manually set target values
    """
    if (resolution_config.enable_auto_aspect_resolution and 
        resolution_config.auto_calculated_w > 0 and 
        resolution_config.auto_calculated_h > 0):
        return resolution_config.auto_calculated_w, resolution_config.auto_calculated_h
    else:
        return resolution_config.target_w, resolution_config.target_h


def format_resolution_info(resolution_config: ResolutionConfig) -> str:
    """
    Format resolution information for display in UI.
    
    Args:
        resolution_config: ResolutionConfig object
        
    Returns:
        Formatted string with resolution information
    """
    if not resolution_config.enable_auto_aspect_resolution:
        return "Auto-resolution disabled"
    
    effective_w, effective_h = get_effective_resolution(resolution_config)
    effective_pixels = effective_w * effective_h
    
    info_lines = [
        f"📐 Auto-calculated: {effective_w}x{effective_h}",
        f"🎯 Pixel budget: {resolution_config.pixel_budget:,} pixels",
        f"📊 Usage: {effective_pixels:,} pixels ({effective_pixels/resolution_config.pixel_budget*100:.1f}%)",
        f"📹 Video aspect ratio: {resolution_config.last_video_aspect_ratio:.3f}"
    ]
    
    return "\n".join(info_lines)


def validate_auto_resolution_config(resolution_config: ResolutionConfig) -> Tuple[bool, str]:
    """
    Validate that auto-resolution configuration is consistent and reasonable.
    
    Args:
        resolution_config: ResolutionConfig object to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not resolution_config.enable_auto_aspect_resolution:
        return True, "Auto-resolution disabled"
    
    if resolution_config.pixel_budget <= 0:
        return False, "Pixel budget must be positive"
    
    if resolution_config.target_h <= 0 or resolution_config.target_w <= 0:
        return False, "Target resolution must be positive"
    
    # Check if pixel budget matches target resolution
    expected_budget = resolution_config.target_h * resolution_config.target_w
    if abs(resolution_config.pixel_budget - expected_budget) > 1:
        return False, f"Pixel budget ({resolution_config.pixel_budget:,}) doesn't match target resolution ({expected_budget:,})"
    
    return True, "Configuration valid"


def update_resolution_from_video(
    video_path: str,
    pixel_budget: int,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Simple wrapper function to calculate optimal resolution from video path and pixel budget.
    
    Args:
        video_path: Path to the video file
        pixel_budget: Maximum total pixels allowed (width × height)
        logger: Optional logger instance
        
    Returns:
        Dictionary with keys:
        - success: bool - Whether calculation succeeded
        - optimal_width: int - Calculated optimal width
        - optimal_height: int - Calculated optimal height
        - status_message: str - Status message for UI display
        - error: str - Error message if success is False
    """
    try:
        # Check if video file exists
        if not video_path or not os.path.exists(video_path):
            return {
                'success': False,
                'error': f"Video file not found: {os.path.basename(video_path) if video_path else 'None'}",
                'optimal_width': 0,
                'optimal_height': 0,
                'status_message': "❌ Video file not found"
            }
        
        # Get video information
        video_info = get_video_info(video_path, logger=logger)
        
        if not video_info:
            return {
                'success': False,
                'error': f"Could not read video info: {os.path.basename(video_path)}",
                'optimal_width': 0,
                'optimal_height': 0,
                'status_message': "❌ Could not read video information"
            }
        
        video_width = video_info.get('width', 0)
        video_height = video_info.get('height', 0)
        
        if video_width <= 0 or video_height <= 0:
            return {
                'success': False,
                'error': f"Invalid video dimensions: {video_width}x{video_height}",
                'optimal_width': 0,
                'optimal_height': 0,
                'status_message': f"❌ Invalid video dimensions: {video_width}x{video_height}"
            }
        
        # Calculate optimal resolution
        optimal_width, optimal_height, status_message = calculate_optimal_resolution(
            video_width=video_width,
            video_height=video_height,
            pixel_budget=pixel_budget,
            logger=logger
        )
        
        return {
            'success': True,
            'optimal_width': optimal_width,
            'optimal_height': optimal_height,
            'status_message': status_message,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Exception in auto-resolution calculation: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        
        return {
            'success': False,
            'error': error_msg,
            'optimal_width': 0,
            'optimal_height': 0,
            'status_message': f"❌ {error_msg}"
        } 