"""
Video Editor Module - Core Logic for Video Cutting and Editing
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import cv2

from .ffmpeg_utils import run_ffmpeg_command, get_video_info
from .file_utils import sanitize_filename, get_next_filename


def safe_progress_update(progress, value, desc=None):
    """Safely update progress without causing IndexError."""
    if progress is None:
        return
    try:
        if desc:
            progress(value, desc=desc)
        else:
            progress(value)
    except (IndexError, AttributeError, TypeError):
        pass


def parse_time_ranges(time_ranges_str: str, video_duration: float) -> List[Tuple[float, float]]:
    """Parse time ranges from user input string."""
    if not time_ranges_str or not time_ranges_str.strip():
        raise ValueError("Time ranges cannot be empty")
    
    def parse_time_to_seconds(time_str: str) -> float:
        time_str = time_str.strip()
        if re.match(r'^\d+\.?\d*$', time_str):
            return float(time_str)
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError(f"Invalid time format: {time_str}")
    
    ranges = []
    range_parts = [part.strip() for part in time_ranges_str.split(',')]
    
    for range_part in range_parts:
        if not range_part or '-' not in range_part:
            continue
        start_str, end_str = range_part.split('-', 1)
        start_time = parse_time_to_seconds(start_str)
        end_time = parse_time_to_seconds(end_str)
        
        if start_time >= end_time:
            raise ValueError(f"Start time must be less than end time: {start_time} >= {end_time}")
        if end_time > video_duration:
            raise ValueError(f"End time {end_time}s exceeds video duration {video_duration}s")
            
        ranges.append((start_time, end_time))
    
    return ranges


def parse_frame_ranges(frame_ranges_str: str, total_frames: int) -> List[Tuple[int, int]]:
    """Parse frame ranges from user input string."""
    if not frame_ranges_str or not frame_ranges_str.strip():
        raise ValueError("Frame ranges cannot be empty")
    
    ranges = []
    range_parts = [part.strip() for part in frame_ranges_str.split(',')]
    
    for range_part in range_parts:
        if not range_part or '-' not in range_part:
            continue
        start_str, end_str = range_part.split('-', 1)
        start_frame = int(start_str)
        end_frame = int(end_str)
        
        if start_frame >= end_frame:
            raise ValueError(f"Start frame must be less than end frame")
        if end_frame > total_frames:
            raise ValueError(f"End frame exceeds total frames")
            
        ranges.append((start_frame, end_frame))
    
    return ranges


def validate_ranges(ranges: List[Tuple], max_value: float, range_type: str = "time") -> Dict[str, Any]:
    """Validate parsed ranges for overlaps and provide analysis."""
    if not ranges:
        return {"status": "No ranges", "analysis_text": "No ranges to validate"}
    
    total_duration = sum(end - start for start, end in ranges)
    coverage_percent = (total_duration / max_value) * 100 if max_value > 0 else 0
    unit = "seconds" if range_type == "time" else "frames"
    
    analysis_text = f"✅ {len(ranges)} segments, {total_duration:.2f} {unit} total ({coverage_percent:.1f}% of original)"
    
    return {
        "status": "✅ Valid",
        "analysis_text": analysis_text,
        "total_duration": total_duration,
        "coverage_percent": coverage_percent
    }


def get_video_detailed_info(video_path: str, logger) -> Optional[Dict[str, Any]]:
    """Get detailed video information."""
    if not video_path or not os.path.exists(video_path):
        return None
    
    try:
        basic_info = get_video_info(video_path, logger)
        if not basic_info:
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return basic_info
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            accurate_duration = total_frames / fps if fps > 0 else 0
            
            enhanced_info = {
                **basic_info,
                'total_frames': total_frames,
                'accurate_duration': accurate_duration,
                'width': width,
                'height': height,
                'fps': fps,
                'file_size': os.path.getsize(video_path),
                'filename': os.path.basename(video_path)
            }
            return enhanced_info
        finally:
            cap.release()
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return basic_info


def format_video_info_for_display(video_info: Dict[str, Any]) -> str:
    """Format video information for display in UI."""
    if not video_info:
        return "No video information available"
    
    file_size = video_info.get('file_size', 0)
    if file_size > 1024 * 1024 * 1024:
        size_str = f"{file_size / (1024**3):.1f} GB"
    elif file_size > 1024 * 1024:
        size_str = f"{file_size / (1024**2):.1f} MB"
    else:
        size_str = f"{file_size / 1024:.1f} KB"
    
    duration = video_info.get('accurate_duration') or video_info.get('duration', 0)
    duration_str = f"{duration:.2f}s"
    if duration >= 60:
        minutes = int(duration // 60)
        seconds = duration % 60
        duration_str = f"{minutes}m {seconds:.1f}s"
    
    info_lines = [
        f"📁 File: {video_info.get('filename', 'Unknown')}",
        f"⏱️ Duration: {duration_str}",
        f"🎬 Frames: {video_info.get('total_frames', 'Unknown')}",
        f"📺 FPS: {video_info.get('fps', 'Unknown'):.2f}",
        f"📐 Resolution: {video_info.get('width', '?')}x{video_info.get('height', '?')}",
        f"💾 Size: {size_str}"
    ]
    
    return "\n".join(info_lines)


def cut_video_segments(
    video_path: str,
    ranges: List[Tuple],
    range_type: str,
    output_dir: str,
    ffmpeg_settings: Dict[str, Any],
    logger,
    progress=None,
    seed: int = None
) -> Dict[str, Any]:
    """Cut video into segments based on provided ranges."""
    try:
        if not ranges:
            return {"success": False, "error": "No ranges provided"}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(video_path).stem
        session_dir = os.path.join(output_dir, f"{base_name}_cut_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        logger.info(f"Starting video cutting: {len(ranges)} segments from {video_path}")
        
        video_info = get_video_detailed_info(video_path, logger)
        if not video_info:
            return {"success": False, "error": "Could not get video information"}
        
        fps = video_info.get('fps', 30)
        
        if range_type == "frame":
            time_ranges = [(start/fps, end/fps) for start, end in ranges]
        else:
            time_ranges = ranges
        
        segment_paths = []
        total_segments = len(time_ranges)
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            progress_value = (i / total_segments) * 0.8
            safe_progress_update(progress, progress_value, f"Cutting segment {i+1}/{total_segments}")
            
            duration = end_time - start_time
            segment_filename = f"segment_{i+1:03d}_{start_time:.3f}s_to_{end_time:.3f}s.mp4"
            segment_path = os.path.join(session_dir, segment_filename)
            
            cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path, "-t", str(duration)]
            
            # Get encoding configuration with automatic NVENC fallback
            from .nvenc_utils import get_nvenc_fallback_encoding_config
            
            encoding_config = get_nvenc_fallback_encoding_config(
                use_gpu=ffmpeg_settings.get("use_gpu", False),
                ffmpeg_preset=ffmpeg_settings.get("preset", "medium"),
                ffmpeg_quality=ffmpeg_settings.get("quality", 18),
                width=video_info.get('width'),
                height=video_info.get('height'),
                logger=logger
            )
            
            cmd.extend(["-c:v", encoding_config['codec']])
            
            if encoding_config['codec'] == 'h264_nvenc':
                cmd.extend(["-preset:v", encoding_config['preset']])
                cmd.extend(["-cq:v", str(encoding_config['quality_value'])])
            else:
                cmd.extend(["-preset", encoding_config['preset']])
                cmd.extend(["-crf", str(encoding_config['quality_value'])])
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            cmd.append(segment_path)
            
            logger.info(f"Cutting segment {i+1}: {start_time:.3f}s - {end_time:.3f}s")
            try:
                result = run_ffmpeg_command(cmd, f"Video Segment Cutting ({i+1}/{total_segments})", logger)
                # run_ffmpeg_command returns True on success, False on failure
                if result:
                    segment_paths.append(segment_path)
                    logger.info(f"Segment {i+1} completed: {segment_path}")
                else:
                    logger.error(f"Failed to cut segment {i+1}")
                    return {"success": False, "error": f"Failed to cut segment {i+1}"}
            except Exception as e:
                logger.error(f"Failed to cut segment {i+1}: {e}")
                return {"success": False, "error": f"Failed to cut segment {i+1}: {str(e)}"}
        
        safe_progress_update(progress, 0.8, "Concatenating segments...")
        
        if len(segment_paths) == 1:
            final_output = segment_paths[0]
        else:
            final_output = os.path.join(session_dir, f"{base_name}_cut_combined.mp4")
            concat_file = final_output + ".concat.txt"
            
            with open(concat_file, 'w') as f:
                for segment_path in segment_paths:
                    f.write(f"file '{os.path.abspath(segment_path)}'\n")
            
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", final_output]
            try:
                result = run_ffmpeg_command(cmd, "Video Segment Concatenation", logger)
                
                try:
                    os.remove(concat_file)
                except:
                    pass
                
                # run_ffmpeg_command returns True on success, False on failure
                if not result:
                    return {"success": False, "error": "Failed to concatenate"}
            except Exception as e:
                try:
                    os.remove(concat_file)
                except:
                    pass
                return {"success": False, "error": f"Failed to concatenate: {str(e)}"}
        
        safe_progress_update(progress, 0.9, "Generating metadata...")
        
        metadata = {
            "timestamp": timestamp,
            "input_video": video_path,
            "ranges": ranges,
            "range_type": range_type,
            "segments": len(ranges),
            "output_file": final_output,
            "session_directory": session_dir,
            "seed": seed
        }
        
        metadata_path = os.path.join(session_dir, "cut_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        safe_progress_update(progress, 1.0, "Video cutting completed!")
        
        logger.info(f"Video cutting completed successfully: {final_output}")
        
        return {
            "success": True,
            "final_output": final_output,
            "segment_paths": segment_paths,
            "session_dir": session_dir,
            "metadata_path": metadata_path,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error during video cutting: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def create_preview_segment(video_path: str, first_range: Tuple, range_type: str, output_dir: str, ffmpeg_settings: Dict[str, Any], logger) -> Optional[str]:
    """Create a preview of the first segment."""
    try:
        if range_type == "frame":
            video_info = get_video_detailed_info(video_path, logger)
            if not video_info:
                return None
            fps = video_info.get('fps', 30)
            start_time, end_time = first_range[0]/fps, first_range[1]/fps
        else:
            start_time, end_time = first_range
        
        base_name = Path(video_path).stem
        preview_filename = f"{base_name}_preview_{start_time:.3f}s_to_{end_time:.3f}s.mp4"
        preview_path = os.path.join(output_dir, preview_filename)
        
        duration = end_time - start_time
        cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", video_path, "-t", str(duration), "-c", "copy", preview_path]
        
        try:
            result = run_ffmpeg_command(cmd, "Video Preview Creation", logger)
            # run_ffmpeg_command returns True on success, False on failure
            return preview_path if result else None
        except Exception as e:
            logger.error(f"Error running preview ffmpeg command: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        return None


def estimate_processing_time(ranges: List[Tuple], video_info: Dict[str, Any], ffmpeg_settings: Dict[str, Any] = None) -> Dict[str, Any]:
    """Estimate processing time for cutting operations."""
    if not ranges or not video_info:
        return {"time_estimate_text": "Could not estimate time", "total_seconds": 0, "formatted": "Unknown"}
    
    total_duration = sum(end - start for start, end in ranges)
    
    # Adjust estimates based on ffmpeg settings if provided
    if ffmpeg_settings:
        # Fast cutting (stream copy) vs precise cutting (re-encoding)
        if ffmpeg_settings.get("stream_copy", False):
            base_multiplier = 0.05  # Very fast for stream copy
        elif ffmpeg_settings.get("use_gpu", False):
            base_multiplier = 0.15  # GPU encoding is faster
        else:
            base_multiplier = 0.3   # CPU encoding is slower
    else:
        base_multiplier = 0.2  # Default estimate
    
    estimated_seconds = total_duration * base_multiplier
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        else:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    
    time_estimate_text = f"📊 Estimated time: ~{format_time(estimated_seconds)} for {len(ranges)} segments ({total_duration:.1f}s total)"
    
    return {
        "time_estimate_text": time_estimate_text,
        "estimated_seconds": estimated_seconds,
        "total_duration": total_duration,
        "segments_count": len(ranges),
        "formatted": format_time(estimated_seconds)
    } 