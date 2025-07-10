"""
Auto-Resolution Utilities for STAR Video Upscaler

This module provides functions to automatically calculate optimal target resolutions
that maintain the input video's aspect ratio while staying within a specified constraint box.
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
    constraint_width: int,
    constraint_height: int,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, int, str]:
    """
    Calculate optimal resolution maintaining aspect ratio within constraint box.
    
    Args:
        video_width: Input video width in pixels
        video_height: Input video height in pixels  
        constraint_width: Maximum allowed width
        constraint_height: Maximum allowed height
        logger: Optional logger instance
        
    Returns:
        Tuple of (optimal_width, optimal_height, status_message)
        
    The calculation finds the largest resolution that:
    1. Maintains the exact aspect ratio of the input video
    2. Fits within the constraint box (width × height)
    3. Has even dimensions (required for video codecs)
    """
    if logger:
        logger.debug(f"Calculating optimal resolution for {video_width}x{video_height} within {constraint_width}x{constraint_height} constraint")
    
    # Validate inputs
    if video_width <= 0 or video_height <= 0:
        error_msg = f"Invalid video dimensions: {video_width}x{video_height}"
        if logger:
            logger.error(error_msg)
        return video_width, video_height, f"❌ {error_msg}"
    
    if constraint_width <= 0 or constraint_height <= 0:
        error_msg = f"Invalid constraint dimensions: {constraint_width}x{constraint_height}"
        if logger:
            logger.error(error_msg)
        return video_width, video_height, f"❌ {error_msg}"
    
    # Calculate aspect ratio
    aspect_ratio = video_width / video_height
    
    # Find the largest resolution that fits within the constraint box
    # while maintaining the aspect ratio
    if aspect_ratio >= 1.0:
        # Wide or square aspect ratio - try width-first approach
        optimal_width = constraint_width
        optimal_height = constraint_width / aspect_ratio
        if optimal_height > constraint_height:
            # Height exceeds constraint, use height as limiting factor
            optimal_height = constraint_height
            optimal_width = constraint_height * aspect_ratio
    else:
        # Tall aspect ratio - try height-first approach
        optimal_height = constraint_height
        optimal_width = constraint_height * aspect_ratio
        if optimal_width > constraint_width:
            # Width exceeds constraint, use width as limiting factor
            optimal_width = constraint_width
            optimal_height = constraint_width / aspect_ratio
    
    # Round to even dimensions
    optimal_width = int(round(optimal_width / 2) * 2)
    optimal_height = int(round(optimal_height / 2) * 2)
    
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
    output_pixels = optimal_width * optimal_height
    max_pixels = constraint_width * constraint_height
    
    if input_pixels == output_pixels:
        status_msg = (f"✅ Optimal matches input: {optimal_width}x{optimal_height} "
                     f"({output_pixels:,} pixels, {output_pixels/max_pixels*100:.1f}% of constraint)")
    elif output_pixels == max_pixels:
        status_msg = (f"✅ Optimal uses full constraint: {optimal_width}x{optimal_height} "
                     f"({output_pixels:,} pixels, 100.0% of constraint)")
    else:
        status_msg = (f"✅ Auto-calculated: {optimal_width}x{optimal_height} "
                     f"({output_pixels:,} pixels, {output_pixels/max_pixels*100:.1f}% of constraint, "
                     f"aspect ratio error: {aspect_ratio_error:.2f}%)")
    
    if logger:
        logger.info(f"Auto-resolution calculation: {video_width}x{video_height} → {optimal_width}x{optimal_height}")
        logger.info(f"Aspect ratio: {aspect_ratio:.3f} → {final_aspect_ratio:.3f} (error: {aspect_ratio_error:.2f}%)")
        logger.info(f"Constraint usage: {output_pixels:,} / {max_pixels:,} ({output_pixels/max_pixels*100:.1f}%)")
    
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
        
        # Calculate optimal resolution using constraint box approach
        optimal_w, optimal_h, status_msg = calculate_optimal_resolution(
            video_width, video_height, 
            updated_config.target_w, updated_config.target_h, 
            logger
        )
        
        # Update config with calculated values
        updated_config.auto_calculated_w = optimal_w
        updated_config.auto_calculated_h = optimal_h
        updated_config.last_video_aspect_ratio = video_width / video_height
        updated_config.pixel_budget = optimal_w * optimal_h
        updated_config.auto_resolution_status = status_msg
        
        if logger:
            logger.info(f"Auto-resolution updated: {video_width}x{video_height} → {optimal_w}x{optimal_h}")
            logger.info(f"Constraint: {updated_config.target_w}x{updated_config.target_h}")
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
    constraint_pixels = resolution_config.target_w * resolution_config.target_h
    
    info_lines = [
        f"📐 Auto-calculated: {effective_w}x{effective_h}",
        f"📦 Constraint: {resolution_config.target_w}x{resolution_config.target_h}",
        f"📊 Usage: {effective_pixels:,} pixels ({effective_pixels/constraint_pixels*100:.1f}% of constraint)",
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
    
    if resolution_config.target_h <= 0 or resolution_config.target_w <= 0:
        return False, "Target resolution must be positive"
    
    # Check if auto-calculated values are reasonable
    if (resolution_config.auto_calculated_w > 0 and 
        resolution_config.auto_calculated_h > 0):
        if (resolution_config.auto_calculated_w > resolution_config.target_w or
            resolution_config.auto_calculated_h > resolution_config.target_h):
            return False, "Auto-calculated resolution exceeds constraint"
    
    return True, "Configuration valid"


def update_resolution_from_video(
    video_path: str,
    constraint_width: int,
    constraint_height: int,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Simple wrapper function to calculate optimal resolution from video path and constraint box.
    
    Args:
        video_path: Path to the video file
        constraint_width: Maximum width allowed
        constraint_height: Maximum height allowed
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
        
        # Calculate optimal resolution using constraint box approach
        optimal_width, optimal_height, status_message = calculate_optimal_resolution(
            video_width=video_width,
            video_height=video_height,
            constraint_width=constraint_width,
            constraint_height=constraint_height,
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