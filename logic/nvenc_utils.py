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


def is_resolution_too_large_for_nvenc(width, height, logger=None):
    """
    Check if the given resolution is too large for NVENC hardware encoding.
    
    NVENC has maximum dimension limits. If the width or height exceeds
    these maximums, we need to fall back to CPU encoding.
    
    Args:
        width: Video width in pixels
        height: Video height in pixels
        logger: Optional logger instance for warnings
        
    Returns:
        bool: True if resolution is too large for NVENC, False otherwise
    """
    max_dimension = 4096  # NVENC maximum width or height
    too_large = width > max_dimension or height > max_dimension
    if too_large and logger:
        logger.warning(f"Resolution {width}x{height} exceeds NVENC maximum ({max_dimension}px), will fallback to CPU encoding")
    return too_large


def should_fallback_to_cpu_encoding(width, height, logger=None):
    """
    Check if NVENC encoding should fallback to CPU encoding due to resolution constraints.
    
    Combines both minimum and maximum resolution checks for NVENC.
    
    Args:
        width: Video width in pixels
        height: Video height in pixels
        logger: Optional logger instance for warnings
        
    Returns:
        bool: True if should fallback to CPU encoding, False if NVENC can be used
    """
    return is_resolution_too_small_for_nvenc(width, height, logger) or is_resolution_too_large_for_nvenc(width, height, logger) 