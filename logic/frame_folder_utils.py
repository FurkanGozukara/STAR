"""
Frame folder utilities for processing input frame directories.
Handles converting frame sequences to videos for further processing.
"""

import os
import tempfile
import time
from typing import Optional, List, Tuple, Dict
import logging
from .ffmpeg_utils import create_video_from_input_frames, get_supported_image_extensions, natural_sort_key


def get_supported_video_extensions() -> List[str]:
    """Return list of supported video file extensions (lowercase, with dots)."""
    return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts', '.mts', '.m2ts']


def detect_input_type(input_path: str, logger: Optional[logging.Logger] = None) -> Tuple[str, str, Dict]:
    """
    Auto-detect if input is a single video file or frames folder.
    Works cross-platform with Windows and Linux path handling.
    
    Args:
        input_path: Path to analyze (can be file or folder)
        logger: Optional logger instance
        
    Returns:
        Tuple of (input_type, validated_path, metadata)
        - input_type: "video_file", "frames_folder", or "invalid" 
        - validated_path: Normalized cross-platform path
        - metadata: Dict with additional info (frame_count, duration, etc.)
    """
    metadata = {}
    
    try:
        if not input_path or not input_path.strip():
            return "invalid", "", {"error": "No input path specified"}
        
        # Normalize path for cross-platform compatibility
        validated_path = os.path.normpath(input_path.strip())
        
        if not os.path.exists(validated_path):
            return "invalid", validated_path, {"error": f"Path does not exist: {validated_path}"}
        
        # Check if it's a file
        if os.path.isfile(validated_path):
            file_ext = os.path.splitext(validated_path)[1].lower()
            supported_video_exts = get_supported_video_extensions()
            
            if file_ext in supported_video_exts:
                # It's a video file
                try:
                    # Try to get basic video info
                    from .ffmpeg_utils import get_video_info
                    video_info = get_video_info(validated_path, logger)
                    if video_info:
                        metadata.update({
                            "file_extension": file_ext,
                            "duration": video_info.get("duration", 0),
                            "fps": video_info.get("fps", 0),
                            "width": video_info.get("width", 0),
                            "height": video_info.get("height", 0),
                            "frames": video_info.get("frames", 0)
                        })
                        return "video_file", validated_path, metadata
                    else:
                        return "invalid", validated_path, {"error": "Could not read video file information"}
                except ImportError:
                    # Fallback if video info functions not available
                    metadata["file_extension"] = file_ext
                    return "video_file", validated_path, metadata
            else:
                return "invalid", validated_path, {"error": f"Unsupported file format: {file_ext}. Supported: {', '.join(supported_video_exts)}"}
        
        # Check if it's a directory (frames folder)
        elif os.path.isdir(validated_path):
            supported_image_exts = get_supported_image_extensions()
            frame_files = []
            
            try:
                for file in os.listdir(validated_path):
                    file_path = os.path.join(validated_path, file)
                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in supported_image_exts:
                            frame_files.append(file)
                
                frame_count = len(frame_files)
                
                if frame_count == 0:
                    return "invalid", validated_path, {
                        "error": f"No supported image files found. Supported formats: {', '.join(supported_image_exts)}"
                    }
                
                if frame_count < 2:
                    return "invalid", validated_path, {
                        "error": f"Not enough frames found (need at least 2): {frame_count}"
                    }
                
                # Apply natural sorting to get proper frame sequence
                sorted_files = sorted(frame_files, key=natural_sort_key)
                
                metadata.update({
                    "frame_count": frame_count,
                    "first_frame": sorted_files[0],
                    "last_frame": sorted_files[-1],
                    "supported_formats": list(set(os.path.splitext(f)[1].lower() for f in frame_files))
                })
                
                if logger:
                    logger.info(f"Detected frames folder: {frame_count} frames ({sorted_files[0]} to {sorted_files[-1]})")
                
                return "frames_folder", validated_path, metadata
                
            except (PermissionError, OSError) as e:
                return "invalid", validated_path, {"error": f"Cannot access directory: {e}"}
        
        else:
            return "invalid", validated_path, {"error": "Path is neither a file nor a directory"}
            
    except Exception as e:
        error_msg = f"Error detecting input type for {input_path}: {e}"
        if logger:
            logger.error(error_msg)
        return "invalid", input_path, {"error": error_msg}

def validate_input_path(input_path: str, logger: Optional[logging.Logger] = None) -> Tuple[bool, str, Dict]:
    """
    Enhanced validation function that handles both video files and frame folders.
    
    Args:
        input_path: Path to validate (video file or frames folder)
        logger: Optional logger instance
        
    Returns:
        Tuple of (is_valid, message, metadata)
        - is_valid: Boolean indicating if input is valid
        - message: Human-readable status message  
        - metadata: Dict with detailed information about the input
    """
    if not input_path or not input_path.strip():
        return False, "No input path specified", {}
    
    input_type, validated_path, metadata = detect_input_type(input_path, logger)
    
    if input_type == "invalid":
        error_msg = metadata.get("error", "Unknown validation error")
        return False, f"❌ {error_msg}", metadata
    
    elif input_type == "video_file":
        # Format video file success message
        if "duration" in metadata and "fps" in metadata:
            duration = metadata["duration"]
            fps = metadata["fps"]
            width = metadata.get("width", "unknown")
            height = metadata.get("height", "unknown")
            frames = metadata.get("frames", "unknown")
            
            success_msg = f"✅ Valid video file: {duration:.1f}s, {fps:.2f} FPS, {width}x{height}, {frames} frames"
        else:
            success_msg = f"✅ Valid video file: {os.path.basename(validated_path)}"
        
        if logger:
            logger.info(f"Video file validated: {validated_path}")
        
        return True, success_msg, metadata
    
    elif input_type == "frames_folder":
        # Format frames folder success message
        frame_count = metadata.get("frame_count", 0)
        first_frame = metadata.get("first_frame", "")
        last_frame = metadata.get("last_frame", "")
        
        success_msg = f"✅ Valid frames folder: {frame_count} frames ({first_frame} to {last_frame})"
        
        if logger:
            logger.info(f"Frames folder validated: {validated_path}")
        
        return True, success_msg, metadata
    
    else:
        return False, "❌ Unknown input type", metadata

def validate_frame_folder_input(
    frames_dir: str, 
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str, int]:
    """
    Legacy function - validates only frame folders.
    Use validate_input_path() for enhanced validation of both video files and frame folders.
    
    Args:
        frames_dir: Directory containing frames
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, message, frame_count)
    """
    is_valid, message, metadata = validate_input_path(frames_dir, logger)
    
    # Extract frame count for backward compatibility
    frame_count = 0
    if is_valid and metadata.get("frame_count"):
        frame_count = metadata["frame_count"]
    elif is_valid and metadata.get("frames"):
        frame_count = metadata["frames"]  # For video files
    
    return is_valid, message, frame_count

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