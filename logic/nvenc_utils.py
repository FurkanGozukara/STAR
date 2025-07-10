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


def test_nvenc_availability(logger=None):
    """
    Test if NVENC is available by running a quick ffmpeg test command.
    
    This function tests NVENC availability by encoding a small test video.
    It's designed to be fast and detect common NVENC issues like:
    - NVENC not supported on current hardware
    - NVENC driver issues
    - NVENC already in use by another process
    
    Args:
        logger: Optional logger instance for status messages
        
    Returns:
        bool: True if NVENC is available, False if should fallback to CPU
    """
    try:
        from .ffmpeg_utils import run_ffmpeg_command
        
        # Test if NVENC is available by running a quick ffmpeg command
        # Use 512x512 resolution to ensure compatibility with all NVENC hardware generations
        test_cmd = 'ffmpeg -loglevel error -f lavfi -i color=c=black:s=512x512:d=0.1:r=1 -c:v h264_nvenc -preset fast -f null -'
        test_result = run_ffmpeg_command(test_cmd, "NVENC Test", logger, raise_on_error=False)
        
        if logger:
            logger.info(f"NVENC availability test: {'PASSED' if test_result else 'FAILED'}")
        
        return test_result
        
    except Exception as e:
        if logger:
            logger.warning(f"NVENC availability test failed: {e}")
        return False


def get_nvenc_fallback_encoding_config(use_gpu, ffmpeg_preset, ffmpeg_quality, width=None, height=None, logger=None):
    """
    Get encoding configuration with automatic NVENC fallback to CPU if needed.
    
    This function determines whether to use NVENC or CPU encoding based on:
    1. User preference (use_gpu parameter)
    2. Resolution constraints
    3. NVENC hardware availability
    
    Args:
        use_gpu: User preference for GPU encoding
        ffmpeg_preset: FFmpeg preset (e.g., 'medium', 'slower')
        ffmpeg_quality: Quality setting (CQ for NVENC, CRF for CPU)
        width: Video width in pixels (optional, for resolution checks)
        height: Video height in pixels (optional, for resolution checks)
        logger: Optional logger instance for status messages
        
    Returns:
        dict: Encoding configuration with 'codec', 'preset', 'quality_param', 'quality_value'
    """
    # Start with user preference
    should_use_gpu = use_gpu
    
    # Check resolution constraints if dimensions provided
    if should_use_gpu and width is not None and height is not None:
        if should_fallback_to_cpu_encoding(width, height, logger):
            should_use_gpu = False
    
    # Test NVENC availability if user wants GPU and resolution is OK
    if should_use_gpu:
        if not test_nvenc_availability(logger):
            should_use_gpu = False
            if logger:
                logger.info("Falling back to CPU encoding due to NVENC not available")
    
    # Build encoding configuration
    if should_use_gpu:
        # Map libx264 presets to h264_nvenc presets
        nvenc_preset = ffmpeg_preset
        if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
            nvenc_preset = "fast"
        elif ffmpeg_preset in ["slower", "veryslow"]:
            nvenc_preset = "slow"
        
        return {
            'codec': 'h264_nvenc',
            'preset': nvenc_preset,
            'quality_param': 'cq:v',
            'quality_value': ffmpeg_quality
        }
    else:
        return {
            'codec': 'libx264',
            'preset': ffmpeg_preset,
            'quality_param': 'crf',
            'quality_value': ffmpeg_quality
        }


def build_ffmpeg_video_encoding_args(encoding_config):
    """
    Build FFmpeg video encoding arguments from encoding configuration.
    
    Args:
        encoding_config: Dict with 'codec', 'preset', 'quality_param', 'quality_value'
        
    Returns:
        str: FFmpeg video encoding arguments
    """
    codec = encoding_config['codec']
    preset = encoding_config['preset']
    quality_param = encoding_config['quality_param']
    quality_value = encoding_config['quality_value']
    
    if codec == 'h264_nvenc':
        return f'-c:v {codec} -preset:v {preset} -{quality_param} {quality_value} -pix_fmt yuv420p'
    else:
        return f'-c:v {codec} -preset {preset} -{quality_param} {quality_value} -pix_fmt yuv420p' 