"""
NVENC (NVIDIA Video Encoder) utility functions.
"""

import logging


def is_resolution_too_small_for_nvenc(width, height, logger=None):
    """
    Check if the given resolution is too small for NVENC hardware encoding.
    
    NVENC has minimum resolution requirements. If the resolution is below
    these minimums, we need to fall back to CPU encoding.
    
    Args:
        width: Video width in pixels
        height: Video height in pixels
        logger: Optional logger instance for warnings
        
    Returns:
        bool: True if resolution is too small for NVENC, False otherwise
    """
    min_width, min_height = 145, 96
    too_small = width < min_width or height < min_height
    if too_small and logger:
        logger.warning(f"Resolution {width}x{height} is below NVENC minimum ({min_width}x{min_height}), will fallback to CPU encoding")
    return too_small 