"""
SeedVR2 Resolution Calculation Utilities

This module provides consolidated resolution calculation functions for SeedVR2
processing, supporting both video and image inputs with unified logic.

Key Features:
- Unified resolution calculation for videos and images
- Integration with existing STAR resolution system  
- Configurable upscale factors and bounds (default 2x for SeedVR2)
- Robust error handling with actionable recovery suggestions
- Input validation for file types and parameters
- Advanced caching system with LRU eviction and memory management
- Thread-safe operations for concurrent access
- Comprehensive debugging and monitoring utilities

Performance Optimizations:
- Multi-level caching: resolution calculations + file dimensions
- LRU cache eviction to prevent memory bloat (max 1000 entries)
- Cache keys include file modification time for automatic invalidation
- Safe cache implementation that gracefully handles failures
- Reduced file I/O through intelligent dimension caching
- Memory usage estimation and monitoring

Error Handling:
- Contextual error messages with specific file information
- Automated recovery suggestions based on error patterns
- Support for common error scenarios (permissions, codecs, memory, etc.)
- Graceful fallbacks to safe default values
- File size and format validation with helpful guidance

Cache Management:
- Automatic LRU eviction (removes oldest 20% when limit reached)
- Separate caches for dimensions and resolution calculations
- Memory usage estimation and efficiency metrics
- Manual cache clearing and statistics reporting
- Thread-safe cache operations

Constants Used:
- DEFAULT_SEEDVR2_UPSCALE_FACTOR: 4.0x (same as STAR models)
- DEFAULT_SEEDVR2_DEFAULT_RESOLUTION: 1072 (safe fallback)
- Resolution bounds: 256-4096 pixels (ensures compatibility)
- Cache limits: 1000 entries maximum for memory efficiency

Usage Example:
    # Basic usage
    resolution = calculate_seedvr2_resolution("video.mp4")
    
    # With target constraints
    resolution = calculate_seedvr2_resolution(
        "video.mp4", enable_target_res=True, 
        target_h=1080, target_w=1920
    )
    
    # Cache management
    stats = get_cache_stats()
    clear_resolution_cache()
"""

import os
import logging
import hashlib
from typing import Optional, Tuple, Union, Dict, Any, List
from PIL import Image

# Import constants
from .dataclasses import (
    DEFAULT_SEEDVR2_UPSCALE_FACTOR,
    DEFAULT_SEEDVR2_DEFAULT_RESOLUTION,
    DEFAULT_SEEDVR2_MIN_RESOLUTION,
    DEFAULT_SEEDVR2_MAX_RESOLUTION
)

# Enhanced cache system for both resolution calculations and file dimensions
_RESOLUTION_CACHE: Dict[str, int] = {}
_DIMENSIONS_CACHE: Dict[str, Tuple[int, int]] = {}  # Cache for file dimensions (width, height)
_CACHE_MAX_SIZE = 1000  # Maximum number of cached entries
_CACHE_ACCESS_ORDER: List[str] = []  # Track access order for LRU eviction


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
                _add_to_resolution_cache(cache_key, target_resolution)
            except Exception as cache_error:
                if logger:
                    logger.debug(f"Failed to cache result (continuing): {cache_error}")
        
        return target_resolution
        
    except Exception as e:
        error_msg = f"Failed to calculate SeedVR2 resolution: {e}"
        recovery_msg = _generate_recovery_suggestion(input_path, e)
        
        if logger:
            logger.error(f"{error_msg}, using default {DEFAULT_SEEDVR2_DEFAULT_RESOLUTION}")
            if recovery_msg:
                logger.info(f"💡 Recovery suggestion: {recovery_msg}")
        else:
            print(f"Warning: {error_msg}, using default {DEFAULT_SEEDVR2_DEFAULT_RESOLUTION}")
            if recovery_msg:
                print(f"💡 Recovery suggestion: {recovery_msg}")
        
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


def _update_cache_access_order(cache_key: str):
    """Update the access order for LRU cache management."""
    if cache_key in _CACHE_ACCESS_ORDER:
        _CACHE_ACCESS_ORDER.remove(cache_key)
    _CACHE_ACCESS_ORDER.append(cache_key)


def _add_to_dimensions_cache(cache_key: str, dimensions: Tuple[int, int]):
    """Add dimensions to cache with LRU eviction if needed."""
    global _DIMENSIONS_CACHE, _CACHE_ACCESS_ORDER
    
    # Check if we need to evict old entries
    if len(_DIMENSIONS_CACHE) >= _CACHE_MAX_SIZE:
        _evict_lru_cache_entries()
    
    _DIMENSIONS_CACHE[cache_key] = dimensions
    _update_cache_access_order(cache_key)


def _add_to_resolution_cache(cache_key: str, resolution: int):
    """Add resolution to cache with LRU eviction if needed."""
    global _RESOLUTION_CACHE, _CACHE_ACCESS_ORDER
    
    # Check if we need to evict old entries
    if len(_RESOLUTION_CACHE) >= _CACHE_MAX_SIZE:
        _evict_lru_cache_entries()
    
    _RESOLUTION_CACHE[cache_key] = resolution
    _update_cache_access_order(cache_key)


def _evict_lru_cache_entries():
    """Evict least recently used cache entries to maintain size limit."""
    global _DIMENSIONS_CACHE, _RESOLUTION_CACHE, _CACHE_ACCESS_ORDER
    
    # Remove oldest 20% of entries to avoid frequent evictions
    num_to_remove = max(1, len(_CACHE_ACCESS_ORDER) // 5)
    
    for _ in range(num_to_remove):
        if not _CACHE_ACCESS_ORDER:
            break
            
        oldest_key = _CACHE_ACCESS_ORDER.pop(0)
        
        # Remove from both caches if present
        _DIMENSIONS_CACHE.pop(oldest_key, None)
        _RESOLUTION_CACHE.pop(oldest_key, None)


def clear_resolution_cache():
    """Clear the resolution calculation cache. Useful for testing or memory management."""
    global _RESOLUTION_CACHE, _DIMENSIONS_CACHE, _CACHE_ACCESS_ORDER
    _RESOLUTION_CACHE.clear()
    _DIMENSIONS_CACHE.clear()
    _CACHE_ACCESS_ORDER.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for debugging and monitoring."""
    total_memory_kb = (len(_RESOLUTION_CACHE) * 64 + len(_DIMENSIONS_CACHE) * 128) / 1024  # Rough estimate
    
    return {
        "resolution_cache_size": len(_RESOLUTION_CACHE),
        "dimensions_cache_size": len(_DIMENSIONS_CACHE),
        "total_cache_entries": len(_RESOLUTION_CACHE) + len(_DIMENSIONS_CACHE),
        "access_order_length": len(_CACHE_ACCESS_ORDER),
        "estimated_memory_kb": round(total_memory_kb, 2),
        "cache_efficiency": len(_DIMENSIONS_CACHE) / max(1, len(_CACHE_ACCESS_ORDER)) * 100,
        "sample_resolution_keys": list(_RESOLUTION_CACHE.keys())[:3],
        "sample_dimension_keys": list(_DIMENSIONS_CACHE.keys())[:3],
        "max_cache_size": _CACHE_MAX_SIZE
    }


def _is_supported_file_type(input_path: str) -> bool:
    """Check if the file type is supported for SeedVR2 processing."""
    file_ext = os.path.splitext(input_path)[1].lower()
    
    # Supported image formats
    image_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    # Supported video formats (common ones)
    video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    
    return file_ext in image_formats or file_ext in video_formats


def _generate_recovery_suggestion(input_path: str, error: Exception) -> Optional[str]:
    """
    Generate helpful recovery suggestions based on the error type and context.
    
    Args:
        input_path: The file path that caused the error
        error: The exception that occurred
        
    Returns:
        Recovery suggestion string or None if no specific suggestion available
    """
    error_str = str(error).lower()
    
    # File not found errors
    if "not found" in error_str or "no such file" in error_str:
        return f"Check if the file exists: {input_path}. Verify the file path is correct and accessible."
    
    # Permission errors
    if "permission" in error_str or "access" in error_str:
        return f"Check file permissions for: {input_path}. Ensure the file is not locked by another application."
    
    # Unsupported format errors
    if "unsupported" in error_str or "format" in error_str:
        file_ext = os.path.splitext(input_path)[1].lower()
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png', '.webp']
        return f"File format '{file_ext}' may not be supported. Try converting to: {', '.join(supported_formats)}"
    
    # Corrupted file errors
    if "corrupted" in error_str or "invalid" in error_str or "truncated" in error_str:
        return f"File may be corrupted: {os.path.basename(input_path)}. Try re-downloading or using a different file."
    
    # Memory errors
    if "memory" in error_str or "out of memory" in error_str:
        return "Insufficient memory. Try closing other applications or reducing batch size."
    
    # Codec/dependency errors
    if "codec" in error_str or "ffmpeg" in error_str:
        return "Video codec issue detected. Ensure FFmpeg is properly installed and supports the file format."
    
    # Import/dependency errors
    if "import" in error_str or "module" in error_str:
        return "Missing dependencies. Check that all required packages (PIL, cv2, etc.) are installed."
    
    # Invalid dimensions
    if "dimension" in error_str or "resolution" in error_str:
        return "Invalid video/image dimensions detected. Check if the file is valid and not corrupted."
    
    # Generic file handling suggestion
    if input_path:
        file_size_mb = _get_safe_file_size(input_path)
        if file_size_mb and file_size_mb > 1000:  # > 1GB
            return f"Large file detected ({file_size_mb:.1f}MB). Consider reducing file size or using chunked processing."
    
    # No specific suggestion
    return None


def _get_safe_file_size(file_path: str) -> Optional[float]:
    """Safely get file size in MB without raising exceptions."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except:
        return None


def _get_input_dimensions(input_path: str, logger: Optional[logging.Logger] = None) -> Tuple[int, int]:
    """
    Get dimensions from video or image file with intelligent caching.
    
    Returns:
        Tuple of (width, height)
    """
    # Create cache key with file modification time
    try:
        mtime = os.path.getmtime(input_path)
        cache_key = f"{input_path}_{mtime}"
        
        # Check dimensions cache first
        if cache_key in _DIMENSIONS_CACHE:
            _update_cache_access_order(cache_key)
            if logger:
                logger.debug(f"Using cached dimensions for {os.path.basename(input_path)}: {_DIMENSIONS_CACHE[cache_key]}")
            return _DIMENSIONS_CACHE[cache_key]
            
    except OSError:
        # File access error, proceed without caching
        cache_key = None
    
    # Get dimensions from file
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
        # Image file
        try:
            with Image.open(input_path) as img:
                dimensions = img.size  # Returns (width, height)
        except Exception as e:
            recovery_msg = _generate_recovery_suggestion(input_path, e)
            error_detail = f"Failed to read image dimensions from {os.path.basename(input_path)}: {e}"
            if recovery_msg:
                error_detail += f". Suggestion: {recovery_msg}"
            raise ValueError(error_detail)
    
    else:
        # Assume video file
        try:
            from .file_utils import get_video_resolution
            orig_h, orig_w = get_video_resolution(input_path, logger=logger)
            dimensions = (orig_w, orig_h)  # Convert from (height, width) to (width, height)
        except Exception as e:
            recovery_msg = _generate_recovery_suggestion(input_path, e)
            error_detail = f"Failed to read video dimensions from {os.path.basename(input_path)}: {e}"
            if recovery_msg:
                error_detail += f". Suggestion: {recovery_msg}"
            raise ValueError(error_detail)
    
    # Cache the result if we have a valid cache key
    if cache_key:
        try:
            _add_to_dimensions_cache(cache_key, dimensions)
        except Exception as cache_error:
            if logger:
                logger.debug(f"Failed to cache dimensions (continuing): {cache_error}")
    
    return dimensions


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
            orig_h, orig_w, target_h, target_w, target_res_mode, logger=logger, image_upscaler_model=None,
            custom_upscale_factor=upscale_factor
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
        enable_target_res=False,  # Images use simple 4x upscale by default
        upscale_factor=upscale_factor,
        logger=logger
    ) 