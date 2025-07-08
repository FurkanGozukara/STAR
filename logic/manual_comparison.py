import os
import time
import tempfile
import logging
from typing import Optional

from .comparison_video import create_comparison_video, get_comparison_output_path
from .common_utils import format_time


def generate_manual_comparison_video(
    original_video_path: str,
    upscaled_video_path: str,
    ffmpeg_preset: str,
    ffmpeg_quality: int,
    ffmpeg_use_gpu: bool,
    output_dir: str,
    comparison_layout: str = "auto",
    seed_value: int = 99,
    logger: Optional[logging.Logger] = None,
    progress=None
) -> tuple[Optional[str], str]:
    """
    Generate a manual comparison video from two uploaded videos.
    
    Args:
        original_video_path: Path to the original/reference video
        upscaled_video_path: Path to the upscaled/enhanced video  
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting (CRF/CQ)
        ffmpeg_use_gpu: Whether to use GPU encoding
        output_dir: Directory to save the output comparison video
        comparison_layout: Layout choice ("auto", "side_by_side", or "top_bottom")
        seed_value: Seed value for metadata (following user rules)
        logger: Logger instance
        progress: Gradio progress callback
        
    Returns:
        tuple: (output_path or None if failed, status_message)
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        # Validate inputs
        if not os.path.exists(original_video_path):
            error_msg = f"Original video file not found: {original_video_path}"
            logger.error(error_msg)
            return None, error_msg
            
        if not os.path.exists(upscaled_video_path):
            error_msg = f"Upscaled video file not found: {upscaled_video_path}"
            logger.error(error_msg)
            return None, error_msg
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if progress is not None:
            progress(0.1, desc="Preparing comparison video generation...")
        
        # Generate output filename based on input files
        original_name = os.path.splitext(os.path.basename(original_video_path))[0]
        upscaled_name = os.path.splitext(os.path.basename(upscaled_video_path))[0]
        
        # Create a unique filename for manual comparison
        output_filename = f"manual_comparison_{original_name}_vs_{upscaled_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Ensure unique filename if file already exists
        counter = 1
        base_output_path = output_path
        while os.path.exists(output_path):
            name_without_ext = os.path.splitext(base_output_path)[0]
            output_path = f"{name_without_ext}_{counter:03d}.mp4"
            counter += 1
        
        if progress is not None:
            progress(0.3, desc="Creating comparison video...")
        
        logger.info(f"Generating manual comparison video: {output_path}")
        logger.info(f"Original video: {original_video_path}")
        logger.info(f"Upscaled video: {upscaled_video_path}")
        logger.info(f"FFmpeg settings - Preset: {ffmpeg_preset}, Quality: {ffmpeg_quality}, GPU: {ffmpeg_use_gpu}")
        logger.info(f"Layout choice: {comparison_layout}")
        logger.info(f"Seed value: {seed_value}")  # Include seed in logs following user rules
        
        # Convert layout choice to force_layout parameter
        force_layout = None if comparison_layout == "auto" else comparison_layout
        
        # Use the existing comparison video creation logic
        success = create_comparison_video(
            original_video_path=original_video_path,
            upscaled_video_path=upscaled_video_path,
            output_path=output_path,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality,
            ffmpeg_use_gpu=ffmpeg_use_gpu,
            force_layout=force_layout,
            logger=logger
        )
        
        if progress is not None:
            progress(0.9, desc="Finalizing comparison video...")
        
        elapsed_time = time.time() - start_time
        
        if success and os.path.exists(output_path):
            success_msg = f"Manual comparison video created successfully: {output_path}. Time: {format_time(elapsed_time)}"
            logger.info(success_msg)
            
            if progress is not None:
                progress(1.0, desc="Comparison video complete!")
            
            return output_path, success_msg
        else:
            error_msg = f"Failed to create manual comparison video. Time: {format_time(elapsed_time)}"
            logger.error(error_msg)
            return None, error_msg
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Error generating manual comparison video: {str(e)}. Time: {format_time(elapsed_time)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg 