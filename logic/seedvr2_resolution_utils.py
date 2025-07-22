"""
SeedVR2 Resolution Calculation Utilities

This module provides consolidated resolution calculation functions for SeedVR2
processing, supporting both video and image inputs with unified logic.

Key Features:
- Unified resolution calculation for videos and images
- Integration with existing STAR resolution system  
- Configurable upscale factors and bounds (default 2x for SeedVR2)
- Robust error handling and fallbacks
- Input validation for file types and parameters
- Performance optimization with intelligent caching
- Cache management functions for debugging and memory control

Performance:
- Caches resolution calculations to avoid redundant computations
- Cache keys include file modification time for cache invalidation
- Safe cache implementation that doesn't break functionality on cache errors

Constants Used:
- DEFAULT_SEEDVR2_UPSCALE_FACTOR: 2.0x (vs 4x for STAR models)
- DEFAULT_SEEDVR2_DEFAULT_RESOLUTION: 1072 (safe fallback)
- Resolution bounds: 256-4096 pixels (ensures compatibility)
"""

import os
import logging
import hashlib
from typing import Optional, Tuple, Union, Dict, Any
from PIL import Image

# Import constants
from .dataclasses import (
    DEFAULT_SEEDVR2_UPSCALE_FACTOR,
    DEFAULT_SEEDVR2_DEFAULT_RESOLUTION,
    DEFAULT_SEEDVR2_MIN_RESOLUTION,
    DEFAULT_SEEDVR2_MAX_RESOLUTION
)

# Simple cache for resolution calculations to avoid redundant calculations
_RESOLUTION_CACHE: Dict[str, int] = {}


def calculate_seedvr2_resolution(
    input_path: str, 
    enable_target_res: bool = False, 
    target_h: int = 1080, 
    target_w: int = 1920, 
    target_res_mode: str = "Ratio Upscale",
    upscale_factor: float = None,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Calculate the target resolution for SeedVR2 based on input media and settings.
    
    This function works with both video and image inputs, automatically detecting
    the input type and applying appropriate resolution calculation logic.
    
    Args:
        input_path: Path to input video or image file
        enable_target_res: Whether target resolution is enabled in UI
        target_h: Target height from UI settings
        target_w: Target width from UI settings  
        target_res_mode: Resolution mode (Ratio Upscale, Downscale then Upscale)
        upscale_factor: Custom upscale factor (defaults to 2.0x for SeedVR2)
        logger: Optional logger instance
        
    Returns:
        Target resolution (short side) for SeedVR2 processing
        
    Raises:
        ValueError: If input file doesn't exist or has invalid dimensions
    """
    # Validate and set defaults
    if upscale_factor is None:
        upscale_factor = DEFAULT_SEEDVR2_UPSCALE_FACTOR
    
    if upscale_factor <= 0:
        raise ValueError(f"Invalid upscale factor: {upscale_factor}. Must be positive.")
    
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}. Must be positive.")
    
    # Try to use cache (but don't fail if caching fails)
    cache_key = None
    try:
        cache_key = _create_cache_key(input_path, enable_target_res, target_h, target_w, target_res_mode, upscale_factor)
        
        # Check cache first
        if cache_key in _RESOLUTION_CACHE:
            if logger:
                logger.debug(f"Using cached resolution for {os.path.basename(input_path)}: {_RESOLUTION_CACHE[cache_key]}")
            return _RESOLUTION_CACHE[cache_key]
    except Exception as cache_error:
        if logger:
            logger.debug(f"Cache lookup failed (continuing without cache): {cache_error}")
        cache_key = None
    
    try:
        # Validate input file exists
        if not input_path or not os.path.exists(input_path):
            raise ValueError(f"Input file not found: {input_path}")
        
        # Additional file type validation
        if not _is_supported_file_type(input_path):
            raise ValueError(f"Unsupported file type: {os.path.splitext(input_path)[1]}")
        
        # Get input dimensions based on file type
        orig_w, orig_h = _get_input_dimensions(input_path, logger)
        
        if logger:
            logger.debug(f"SeedVR2 resolution calculation for {os.path.basename(input_path)}: {orig_w}x{orig_h}")
        
        # Calculate target resolution based on settings
        if enable_target_res:
            target_resolution = _calculate_with_target_constraints(
                orig_w, orig_h, target_h, target_w, target_res_mode, upscale_factor, logger
            )
        else:
            # Simple 2x upscale of the shorter side
            target_resolution = min(orig_w, orig_h) * upscale_factor
            if logger:
                logger.info(f"SeedVR2 using {upscale_factor}x default upscale: {orig_w}x{orig_h} -> target resolution {target_resolution} (short side)")
        
        # Apply bounds and ensure even number
        target_resolution = _apply_resolution_constraints(target_resolution, logger)
        
        # Cache the result (if cache key was created successfully)
        if cache_key:
            try:
                _RESOLUTION_CACHE[cache_key] = target_resolution
            except Exception as cache_error:
                if logger:
                    logger.debug(f"Failed to cache result (continuing): {cache_error}")
        
        return target_resolution
        
    except Exception as e:
        error_msg = f"Failed to calculate SeedVR2 resolution: {e}"
        if logger:
            logger.error(f"{error_msg}, using default {DEFAULT_SEEDVR2_DEFAULT_RESOLUTION}")
        else:
            print(f"Warning: {error_msg}, using default {DEFAULT_SEEDVR2_DEFAULT_RESOLUTION}")
        return DEFAULT_SEEDVR2_DEFAULT_RESOLUTION


def _create_cache_key(input_path: str, enable_target_res: bool, target_h: int, target_w: int, 
                     target_res_mode: str, upscale_factor: float) -> str:
    """Create a unique cache key for resolution calculation parameters."""
    # Include file modification time to invalidate cache if file changes
    try:
        mtime = os.path.getmtime(input_path)
    except OSError:
        mtime = 0
    
    # Create hash of all parameters
    key_data = f"{input_path}_{mtime}_{enable_target_res}_{target_h}_{target_w}_{target_res_mode}_{upscale_factor}"
    return hashlib.md5(key_data.encode()).hexdigest()


def clear_resolution_cache():
    """Clear the resolution calculation cache. Useful for testing or memory management."""
    global _RESOLUTION_CACHE
    _RESOLUTION_CACHE.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for debugging and monitoring."""
    return {
        "cache_size": len(_RESOLUTION_CACHE),
        "cached_entries": list(_RESOLUTION_CACHE.keys())[:5],  # Show first 5 for debugging
        "total_cached_entries": len(_RESOLUTION_CACHE)
    }


def _is_supported_file_type(input_path: str) -> bool:
    """Check if the file type is supported for SeedVR2 processing."""
    file_ext = os.path.splitext(input_path)[1].lower()
    
    # Supported image formats
    image_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    # Supported video formats (common ones)
    video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    
    return file_ext in image_formats or file_ext in video_formats


def _get_input_dimensions(input_path: str, logger: Optional[logging.Logger] = None) -> Tuple[int, int]:
    """
    Get dimensions from video or image file.
    
    Returns:
        Tuple of (width, height)
    """
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
        # Image file
        try:
            with Image.open(input_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            raise ValueError(f"Failed to read image dimensions: {e}")
    
    else:
        # Assume video file
        try:
            from .file_utils import get_video_resolution
            orig_h, orig_w = get_video_resolution(input_path, logger=logger)
            return orig_w, orig_h  # Convert from (height, width) to (width, height)
        except Exception as e:
            raise ValueError(f"Failed to read video dimensions: {e}")


def _calculate_with_target_constraints(
    orig_w: int, orig_h: int, target_h: int, target_w: int, 
    target_res_mode: str, upscale_factor: float, 
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Calculate resolution using the existing STAR resolution calculation system.
    """
    try:
        from .upscaling_utils import calculate_upscale_params
        
        needs_downscale, ds_h, ds_w, upscale_factor_calc, final_h_calc, final_w_calc = calculate_upscale_params(
            orig_h, orig_w, target_h, target_w, target_res_mode, logger=logger, image_upscaler_model=None
        )
        
        # For SeedVR2, use the shorter side as resolution
        target_resolution = min(final_w_calc, final_h_calc)
        
        if logger:
            logger.info(f"SeedVR2 resolution with target constraints: {orig_w}x{orig_h} -> target resolution {target_resolution} (short side)")
            
        return target_resolution
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to use target resolution calculation, falling back to default {upscale_factor}x: {e}")
        # Fallback to simple upscale
        return min(orig_w, orig_h) * upscale_factor


def _apply_resolution_constraints(target_resolution: float, logger: Optional[logging.Logger] = None) -> int:
    """
    Apply resolution bounds and ensure even number for video codec compatibility.
    """
    # Ensure reasonable bounds
    target_resolution = max(DEFAULT_SEEDVR2_MIN_RESOLUTION, min(DEFAULT_SEEDVR2_MAX_RESOLUTION, int(target_resolution)))
    
    # Ensure even number for video codec compatibility
    target_resolution = int(target_resolution // 2) * 2
    
    if logger:
        logger.debug(f"Applied resolution constraints: final resolution {target_resolution}")
    
    return target_resolution


# Legacy compatibility functions for existing code
def calculate_seedvr2_video_resolution(*args, **kwargs) -> int:
    """Legacy compatibility wrapper for video resolution calculation."""
    return calculate_seedvr2_resolution(*args, **kwargs)


def calculate_seedvr2_image_resolution(input_image_path: str, upscale_factor: float = None, logger: Optional[logging.Logger] = None) -> int:
    """Legacy compatibility wrapper for image resolution calculation."""
    if upscale_factor is None:
        upscale_factor = DEFAULT_SEEDVR2_UPSCALE_FACTOR
    
    return calculate_seedvr2_resolution(
        input_path=input_image_path,
        enable_target_res=False,  # Images use simple 2x upscale by default
        upscale_factor=upscale_factor,
        logger=logger
    ) 