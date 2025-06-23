"""
Frame folder utilities for processing input frame directories.
Handles converting frame sequences to videos for further processing.
"""

import os
import tempfile
from typing import Optional, List, Tuple
import logging
from .ffmpeg_utils import create_video_from_input_frames, get_supported_image_extensions, natural_sort_key

def find_frame_folders_in_directory(input_dir: str, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Find all subdirectories that contain frame sequences in the input directory.
    
    Args:
        input_dir: Directory to search for frame folders
        logger: Logger instance
        
    Returns:
        List of paths to directories containing frame sequences
    """
    if not os.path.exists(input_dir):
        if logger:
            logger.warning(f"Input directory does not exist: {input_dir}")
        return []
    
    frame_folders = []
    supported_extensions = get_supported_image_extensions()
    
    try:
        # Check if the main directory itself contains frames
        main_dir_frames = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                main_dir_frames.append(file)
        
        if main_dir_frames:
            frame_folders.append(input_dir)
            if logger:
                logger.info(f"Found {len(main_dir_frames)} frames in main directory: {input_dir}")
        
        # Check subdirectories for frame sequences
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                frame_files = []
                try:
                    for file in os.listdir(item_path):
                        if any(file.lower().endswith(ext) for ext in supported_extensions):
                            frame_files.append(file)
                    
                    if frame_files:
                        frame_folders.append(item_path)
                        if logger:
                            logger.info(f"Found {len(frame_files)} frames in subdirectory: {item}")
                            
                except (PermissionError, OSError) as e:
                    if logger:
                        logger.warning(f"Could not access subdirectory {item_path}: {e}")
                    continue
        
        return frame_folders
        
    except Exception as e:
        if logger:
            logger.error(f"Error scanning for frame folders in {input_dir}: {e}")
        return []

def process_frame_folder_to_video(
    frames_dir: str, 
    output_video_path: str, 
    fps: float = 30.0,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality_value: int = 23,
    ffmpeg_use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Process a frame folder into a video file.
    
    Args:
        frames_dir: Directory containing frame sequence
        output_video_path: Output path for generated video
        fps: Target FPS for the video
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality_value: Quality value for encoding
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
        
    Returns:
        Tuple of (success, message)
    """
    try:
        if not os.path.exists(frames_dir):
            error_msg = f"Frame directory does not exist: {frames_dir}"
            if logger:
                logger.error(error_msg)
            return False, error_msg
        
        # Count frames first
        supported_extensions = get_supported_image_extensions()
        frame_files = []
        for file in os.listdir(frames_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                frame_files.append(file)
        
        if not frame_files:
            error_msg = f"No supported image files found in {frames_dir}"
            if logger:
                logger.warning(error_msg)
            return False, error_msg
        
        # Apply natural sorting
        frame_files = sorted(frame_files, key=natural_sort_key)
        
        if logger:
            logger.info(f"Processing {len(frame_files)} frames from {frames_dir} to video at {fps} FPS")
            logger.info(f"First few frames: {frame_files[:5]}{'...' if len(frame_files) > 5 else ''}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # Use existing function to create video from frames
        success = create_video_from_input_frames(
            frames_dir, output_video_path, fps,
            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger
        )
        
        if success:
            success_msg = f"Successfully created video from {len(frame_files)} frames: {output_video_path}"
            if logger:
                logger.info(success_msg)
            return True, success_msg
        else:
            error_msg = f"Failed to create video from frames in {frames_dir}"
            if logger:
                logger.error(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error processing frame folder {frames_dir}: {e}"
        if logger:
            logger.error(error_msg, exc_info=True)
        return False, error_msg

def is_frame_folder(folder_path: str, min_frames: int = 2, logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a folder contains a frame sequence (minimum number of supported image files).
    
    Args:
        folder_path: Path to folder to check
        min_frames: Minimum number of frames required
        logger: Logger instance
        
    Returns:
        True if folder contains frame sequence
    """
    try:
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False
        
        supported_extensions = get_supported_image_extensions()
        frame_count = 0
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                frame_count += 1
                if frame_count >= min_frames:
                    return True
        
        return False
        
    except Exception as e:
        if logger:
            logger.warning(f"Error checking if {folder_path} is frame folder: {e}")
        return False

def validate_frame_folder_input(
    frames_dir: str, 
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str, int]:
    """
    Validate frame folder input and return information.
    
    Args:
        frames_dir: Directory containing frames
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, message, frame_count)
    """
    try:
        if not frames_dir or not frames_dir.strip():
            return False, "No frame folder specified", 0
        
        if not os.path.exists(frames_dir):
            return False, f"Frame folder does not exist: {frames_dir}", 0
        
        if not os.path.isdir(frames_dir):
            return False, f"Path is not a directory: {frames_dir}", 0
        
        # Count supported image files
        supported_extensions = get_supported_image_extensions()
        frame_files = []
        for file in os.listdir(frames_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                frame_files.append(file)
        
        frame_count = len(frame_files)
        
        if frame_count == 0:
            return False, f"No supported image files found in {frames_dir}. Supported formats: {', '.join(supported_extensions)}", 0
        
        if frame_count < 2:
            return False, f"Not enough frames found (need at least 2): {frame_count}", frame_count
        
        # Apply natural sorting to check naming
        sorted_files = sorted(frame_files, key=natural_sort_key)
        
        success_msg = f"Valid frame folder: {frame_count} frames found ({sorted_files[0]} to {sorted_files[-1]})"
        if logger:
            logger.info(success_msg)
        
        return True, success_msg, frame_count
        
    except Exception as e:
        error_msg = f"Error validating frame folder {frames_dir}: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg, 0 