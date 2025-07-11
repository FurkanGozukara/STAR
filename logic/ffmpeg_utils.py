import os
import subprocess
import gradio as gr
import cv2
import re
from .nvenc_utils import should_fallback_to_cpu_encoding
from .cancellation_manager import cancellation_manager

def run_ffmpeg_command(cmd, desc="ffmpeg command", logger=None, raise_on_error=True):
    """
    Run an FFmpeg command with error handling and optional graceful failure.
    Supports cancellation via the global cancellation manager.
    
    Args:
        cmd: FFmpeg command to run (string or list)
        desc: Description for logging
        logger: Logger instance  
        raise_on_error: If True, raises gr.Error on failure. If False, returns False on failure.
        
    Returns:
        bool: True if successful, False if failed (when raise_on_error=False)
        
    Raises:
        gr.Error: When command fails and raise_on_error=True
    """
    # Determine if cmd is a list or string and set shell accordingly
    if isinstance(cmd, list):
        use_shell = False
        cmd_for_logging = ' '.join(cmd)  # For logging purposes
    else:
        use_shell = True
        cmd_for_logging = cmd
    
    if logger:
        logger.info(f"Running {desc}: {cmd_for_logging}")
    try:
        # Use Popen for better process control and cancellation support
        process = subprocess.Popen(cmd, shell=use_shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, encoding='utf-8', errors='ignore')
        
        # Register the process with the cancellation manager
        cancellation_manager.set_active_process(process)
        
        try:
            stdout, stderr = process.communicate()
            returncode = process.returncode
        except Exception:
            # If communication is interrupted (e.g., by cancellation), clean up
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        finally:
            # Clear the active process
            cancellation_manager.clear_active_process()
            
        if stdout and logger:
            logger.info(f"{desc} stdout: {stdout.strip()}")

        if returncode != 0:
            if stderr and logger:
                logger.error(f"{desc} stderr: {stderr.strip()}")
            raise subprocess.CalledProcessError(returncode, cmd_for_logging, stdout, stderr)
        elif stderr and ('error' in stderr.lower() or 'warning' in stderr.lower()):
            if logger:
                logger.info(f"{desc} stderr: {stderr.strip()}")
                
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Error running {desc}:")
            logger.error(f"  Command: {cmd_for_logging}")
            logger.error(f"  Return code: {e.returncode}")
            if e.stdout:
                logger.error(f"  Stdout: {e.stdout.strip()}")
            if e.stderr:
                logger.error(f"  Stderr: {e.stderr.strip()}")
        
        if raise_on_error:
            raise gr.Error(f"ffmpeg command failed (see console for details): {e.stderr.strip()[:500] if e.stderr else 'Unknown ffmpeg error'}")
        else:
            return False
            
    except Exception as e_gen:
        if logger:
            logger.error(f"Unexpected error preparing/running {desc} for command '{cmd_for_logging}': {e_gen}")
        
        if raise_on_error:
            raise gr.Error(f"ffmpeg command failed: {e_gen}")
        else:
            return False

def extract_frames(video_path, temp_dir, logger=None):
    """Extract frames from video using FFmpeg."""
    if logger:
        logger.info(f"Extracting frames from '{video_path}' to '{temp_dir}'")
    os.makedirs(temp_dir, exist_ok=True)
    fps = 30.0
    try:
        probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        process = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        rate_str = process.stdout.strip()
        if '/' in rate_str:
            num, den = map(int, rate_str.split('/'))
            if den != 0:
                fps = num / den
        elif rate_str:
            fps = float(rate_str)
        if logger:
            logger.info(f"Detected FPS: {fps}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not get FPS using ffprobe for '{video_path}': {e}. Using default {fps} FPS.")

    cmd = f'ffmpeg -i "{video_path}" -vsync vfr -qscale:v 2 "{os.path.join(temp_dir, "frame_%06d.png")}"'
    run_ffmpeg_command(cmd, "Frame Extraction", logger)

    frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')], key=natural_sort_key)
    frame_count = len(frame_files)
    if logger:
        logger.info(f"Extracted {frame_count} frames.")
    if frame_count == 0:
        raise gr.Error("Failed to extract any frames. Check video file and ffmpeg installation.")
    return frame_count, fps, frame_files

def get_video_fps(video_path, logger=None):
    """Get the FPS of a video file using FFprobe."""
    fps = 30.0  # Default fallback
    try:
        probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        process = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        rate_str = process.stdout.strip()
        if '/' in rate_str:
            num, den = map(int, rate_str.split('/'))
            if den != 0:
                fps = num / den
        elif rate_str:
            fps = float(rate_str)
        if logger:
            logger.info(f"Detected FPS: {fps}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not get FPS using ffprobe for '{video_path}': {e}. Using default {fps} FPS.")
    return fps

def calculate_target_fps_from_multiplier(input_video_path, multiplier, logger=None):
    """
    Calculate target FPS by applying a multiplier to the input video's FPS.
    
    Args:
        input_video_path: Path to input video
        multiplier: FPS multiplier (e.g., 0.5 for half FPS, 0.25 for quarter FPS)
        logger: Logger instance
    
    Returns:
        tuple: (target_fps: float, input_fps: float, multiplier_applied: float)
    """
    if not os.path.exists(input_video_path):
        error_msg = f"Input video not found: {input_video_path}"
        if logger:
            logger.error(error_msg)
        return 24.0, 30.0, 0.8  # Fallback values
    
    # Get input video FPS
    input_fps = get_video_fps(input_video_path, logger)
    
    # Validate multiplier
    if multiplier <= 0 or multiplier > 1:
        if logger:
            logger.warning(f"Invalid multiplier {multiplier}, using 0.5 as fallback")
        multiplier = 0.5
    
    # Calculate target FPS
    target_fps = input_fps * multiplier
    
    # Ensure minimum FPS of 1.0
    if target_fps < 1.0:
        if logger:
            logger.warning(f"Calculated target FPS ({target_fps:.2f}) is too low, using 1.0 FPS minimum")
        target_fps = 1.0
        multiplier = target_fps / input_fps
    
    if logger:
        logger.info(f"FPS calculation: {input_fps:.2f} × {multiplier:.3f} = {target_fps:.2f} FPS")
    
    return target_fps, input_fps, multiplier

def get_common_fps_multipliers():
    """
    Get common FPS multipliers with descriptive names.
    
    Returns:
        dict: Mapping of multiplier values to descriptive names
    """
    return {
        0.5: "1/2x (Half FPS)",
        0.25: "1/4x (Quarter FPS)", 
        0.33: "1/3x (Third FPS)",
        0.67: "2/3x (Two-thirds FPS)",
        0.75: "3/4x (Three-quarters FPS)",
        0.125: "1/8x (Eighth FPS)"
    }

def decrease_fps(input_video_path, output_video_path, target_fps=None, interpolation_method="drop", ffmpeg_preset="medium", ffmpeg_quality_value=23, ffmpeg_use_gpu=False, logger=None, fps_mode="fixed", fps_multiplier=0.5):
    """
    Decrease video FPS using FFmpeg with support for both fixed FPS and multiplier modes.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video
        target_fps: Target FPS (float) - used in fixed mode, ignored in multiplier mode
        interpolation_method: "drop" for dropping frames, "blend" for blending frames
        ffmpeg_preset: FFmpeg preset for encoding
        ffmpeg_quality_value: Quality value for encoding
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
        fps_mode: "fixed" for absolute FPS value, "multiplier" for relative FPS reduction
        fps_multiplier: Multiplier for input FPS (e.g., 0.5 for half FPS) - used in multiplier mode
    
    Returns:
        tuple: (success: bool, output_fps: float, message: str)
    """
    if not os.path.exists(input_video_path):
        error_msg = f"Input video not found: {input_video_path}"
        if logger:
            logger.error(error_msg)
        return False, 0.0, error_msg
    
    # Determine target FPS based on mode
    if fps_mode == "multiplier":
        calculated_target_fps, input_fps, actual_multiplier = calculate_target_fps_from_multiplier(
            input_video_path, fps_multiplier, logger
        )
        target_fps = calculated_target_fps
        current_fps = input_fps
        mode_description = f"multiplier mode ({actual_multiplier:.3f}x)"
    else:  # fixed mode
        current_fps = get_video_fps(input_video_path, logger)
        if target_fps is None:
            target_fps = 24.0  # Default fallback
            if logger:
                logger.warning(f"No target FPS specified in fixed mode, using default {target_fps} FPS")
        mode_description = "fixed FPS mode"
    
    # Check if FPS decrease is necessary
    if current_fps <= target_fps:
        msg = f"Input FPS ({current_fps:.2f}) is already at or below target FPS ({target_fps:.2f}) in {mode_description}. No FPS decrease needed."
        if logger:
            logger.info(msg)
        # Copy the file instead of processing
        try:
            import shutil
            shutil.copy2(input_video_path, output_video_path)
            return True, current_fps, msg
        except Exception as e:
            error_msg = f"Error copying file: {e}"
            if logger:
                logger.error(error_msg)
            return False, 0.0, error_msg
    
    if logger:
        logger.info(f"Decreasing FPS from {current_fps:.2f} to {target_fps:.2f} using {mode_description} and '{interpolation_method}' method")
    
    try:
        # Prepare output directory
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # Build filter based on interpolation method
        if interpolation_method == "blend":
            # Use minterpolate filter for blending frames
            fps_filter = f"minterpolate=fps={target_fps}:mi_mode=blend"
        else:  # "drop" method (default)
            # Use fps filter to drop frames
            fps_filter = f"fps={target_fps}"
        
        # Detect video resolution for NVENC minimum size check
        frame_width, frame_height = None, None
        if ffmpeg_use_gpu:
            try:
                probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of default=noprint_wrappers=1:nokey=1 "{input_video_path}"'
                process = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                lines = process.stdout.strip().split('\n')
                if len(lines) >= 2:
                    frame_width = int(lines[0])
                    frame_height = int(lines[1])
                    if logger:
                        logger.info(f"Detected video resolution: {frame_width}x{frame_height}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not detect video resolution: {e}")
        
        # Get encoding configuration with automatic NVENC fallback
        from .nvenc_utils import get_nvenc_fallback_encoding_config, build_ffmpeg_video_encoding_args
        
        encoding_config = get_nvenc_fallback_encoding_config(
            use_gpu=ffmpeg_use_gpu,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality_value,
            width=frame_width,
            height=frame_height,
            logger=logger
        )
        
        video_codec_opts = build_ffmpeg_video_encoding_args(encoding_config)
        
        if logger:
            codec_info = f"Using {encoding_config['codec']} for FPS decrease with preset {encoding_config['preset']} and {encoding_config['quality_param'].upper()} {encoding_config['quality_value']}."
            logger.info(codec_info)
        
        # Build FFmpeg command
        cmd = f'ffmpeg -y -i "{input_video_path}" -filter:v "{fps_filter}" {video_codec_opts} -c:a copy "{output_video_path}"'
        
        # Run the command
        run_ffmpeg_command(cmd, f"FPS Decrease ({interpolation_method})", logger)
        
        # Verify output and get final FPS
        if os.path.exists(output_video_path):
            final_fps = get_video_fps(output_video_path, logger)
            success_msg = f"FPS decreased successfully from {current_fps:.2f} to {final_fps:.2f} FPS using {mode_description} and {interpolation_method} method"
            if logger:
                logger.info(success_msg)
            return True, final_fps, success_msg
        else:
            error_msg = "Output video file was not created"
            if logger:
                logger.error(error_msg)
            return False, 0.0, error_msg
            
    except Exception as e:
        error_msg = f"Error during FPS decrease: {str(e)}"
        if logger:
            logger.error(error_msg)
        return False, 0.0, error_msg

def decrease_fps_with_multiplier(input_video_path, output_video_path, multiplier, interpolation_method="drop", ffmpeg_preset="medium", ffmpeg_quality_value=23, ffmpeg_use_gpu=False, logger=None):
    """
    Convenience function to decrease FPS using a multiplier.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video  
        multiplier: FPS multiplier (e.g., 0.5 for half FPS, 0.25 for quarter FPS)
        interpolation_method: "drop" for dropping frames, "blend" for blending frames
        ffmpeg_preset: FFmpeg preset for encoding
        ffmpeg_quality_value: Quality value for encoding
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
    
    Returns:
        tuple: (success: bool, output_fps: float, message: str)
    """
    return decrease_fps(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        target_fps=None,  # Not used in multiplier mode
        interpolation_method=interpolation_method,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_quality_value=ffmpeg_quality_value,
        ffmpeg_use_gpu=ffmpeg_use_gpu,
        logger=logger,
        fps_mode="multiplier",
        fps_multiplier=multiplier
    )

def create_video_from_frames(frame_dir, output_path, fps, ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=None):
    """Create video from frames using FFmpeg."""
    if logger:
        logger.info(f"Creating video from frames in '{frame_dir}' to '{output_path}' at {fps} FPS with preset: {ffmpeg_preset}, quality: {ffmpeg_quality_value}, GPU: {ffmpeg_use_gpu}")
    input_pattern = os.path.join(frame_dir, "frame_%06d.png")

    # Detect frame resolution for NVENC resolution compatibility check
    frame_width, frame_height = None, None
    if ffmpeg_use_gpu:
        # Find first frame to check resolution
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')], key=natural_sort_key)
        if frame_files:
            first_frame_path = os.path.join(frame_dir, frame_files[0])
            try:
                frame = cv2.imread(first_frame_path)
                if frame is not None:
                    frame_height, frame_width = frame.shape[:2]
                    if logger:
                        logger.info(f"Detected frame resolution: {frame_width}x{frame_height}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not detect frame resolution: {e}")

    # Get encoding configuration with automatic NVENC fallback
    from .nvenc_utils import get_nvenc_fallback_encoding_config, build_ffmpeg_video_encoding_args
    
    encoding_config = get_nvenc_fallback_encoding_config(
        use_gpu=ffmpeg_use_gpu,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_quality=ffmpeg_quality_value,
        width=frame_width,
        height=frame_height,
        logger=logger
    )
    
    video_codec_opts = build_ffmpeg_video_encoding_args(encoding_config)
    
    if logger:
        codec_info = f"Using {encoding_config['codec']} with preset {encoding_config['preset']} and {encoding_config['quality_param'].upper()} {encoding_config['quality_value']}."
        logger.info(codec_info)

    cmd = f'ffmpeg -y -framerate {fps} -i "{input_pattern}" {video_codec_opts} "{output_path}"'
    
    try:
        success = run_ffmpeg_command(cmd, "Video Reassembly (silent)", logger, raise_on_error=False)
        if success and os.path.exists(output_path):
            return True
        else:
            if logger:
                logger.error(f"Video creation failed or output file not created: {output_path}")
            return False
    except Exception as e:
        if logger:
            logger.error(f"Exception during video creation: {e}")
        return False

def get_video_info(video_path, logger=None):
    """
    Get comprehensive video information including frames, FPS, duration, and resolution.
    
    Args:
        video_path: Path to the video file
        logger: Logger instance for logging
    
    Returns:
        dict: Dictionary containing video information with keys:
              'frames', 'fps', 'duration', 'width', 'height', 'format', 'bitrate'
              Returns None if video cannot be read
    """
    if not os.path.exists(video_path):
        if logger:
            logger.error(f"Video file not found: {video_path}")
        return None
    
    video_info = {
        'frames': 0,
        'fps': 0.0,
        'duration': 0.0,
        'width': 0,
        'height': 0,
        'format': 'Unknown',
        'bitrate': 'Unknown'
    }
    
    try:
        # First, get the actual frame count directly using ffprobe
        frame_count_cmd = (
            f'ffprobe -v error -select_streams v:0 '
            f'-count_frames -show_entries stream=nb_read_frames '
            f'-of csv=p=0 "{video_path}"'
        )
        
        try:
            frame_count_process = subprocess.run(frame_count_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            actual_frame_count = frame_count_process.stdout.strip()
            if actual_frame_count and actual_frame_count.isdigit():
                video_info['frames'] = int(actual_frame_count)
                if logger:
                    logger.debug(f"Got actual frame count from ffprobe: {video_info['frames']}")
        except subprocess.CalledProcessError:
            if logger:
                logger.debug("Direct frame count failed, will use alternative methods")
        
        # Use ffprobe to get comprehensive video information
        probe_cmd = (
            f'ffprobe -v error -select_streams v:0 '
            f'-show_entries stream=width,height,r_frame_rate,duration,bit_rate '
            f'-show_entries format=format_name,duration,bit_rate '
            f'-of csv=s=,:p=0 "{video_path}"'
        )
        
        process = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        lines = process.stdout.strip().split('\n')
        
        # Parse stream information (first line)
        if len(lines) >= 1 and lines[0]:
            stream_parts = lines[0].split(',')
            if len(stream_parts) >= 4:
                try:
                    video_info['width'] = int(stream_parts[0]) if stream_parts[0] else 0
                    video_info['height'] = int(stream_parts[1]) if stream_parts[1] else 0
                    
                    # Parse frame rate
                    if '/' in stream_parts[2]:
                        num, den = map(int, stream_parts[2].split('/'))
                        video_info['fps'] = num / den if den != 0 else 0.0
                    elif stream_parts[2]:
                        video_info['fps'] = float(stream_parts[2])
                    
                    # Parse duration from stream
                    if stream_parts[3]:
                        video_info['duration'] = float(stream_parts[3])
                    
                    # Parse bitrate from stream
                    if len(stream_parts) > 4 and stream_parts[4]:
                        bitrate_bps = int(stream_parts[4])
                        video_info['bitrate'] = f"{bitrate_bps // 1000} kbps"
                except (ValueError, IndexError) as e:
                    if logger:
                        logger.warning(f"Error parsing stream info: {e}")
        
        # Parse format information (second line) - fallback for duration and bitrate
        if len(lines) >= 2 and lines[1]:
            format_parts = lines[1].split(',')
            if len(format_parts) >= 1:
                video_info['format'] = format_parts[0]
                
                # Use format duration if stream duration not available
                if video_info['duration'] == 0.0 and len(format_parts) > 1 and format_parts[1]:
                    try:
                        video_info['duration'] = float(format_parts[1])
                    except ValueError:
                        pass
                
                # Use format bitrate if stream bitrate not available
                if video_info['bitrate'] == 'Unknown' and len(format_parts) > 2 and format_parts[2]:
                    try:
                        bitrate_bps = int(format_parts[2])
                        video_info['bitrate'] = f"{bitrate_bps // 1000} kbps"
                    except ValueError:
                        pass
        
        # Fallback to OpenCV if ffprobe didn't get all info
        if video_info['width'] == 0 or video_info['height'] == 0 or video_info['frames'] == 0:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    if video_info['width'] == 0:
                        video_info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    if video_info['height'] == 0:
                        video_info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if video_info['frames'] == 0:
                        video_info['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if logger:
                            logger.debug(f"Got frame count from OpenCV: {video_info['frames']}")
                    if video_info['fps'] == 0.0:
                        video_info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                    
                    cap.release()
            except Exception as cv_e:
                if logger:
                    logger.warning(f"OpenCV fallback failed: {cv_e}")
        
        # Final fallback: calculate frame count from FPS and duration if still not available
        if video_info['frames'] == 0 and video_info['fps'] > 0 and video_info['duration'] > 0:
            calculated_frames = int(video_info['fps'] * video_info['duration'])
            video_info['frames'] = calculated_frames
            if logger:
                logger.debug(f"Calculated frame count from FPS×duration: {calculated_frames}")
        
        # Recalculate duration if we have accurate frame count and FPS but no duration
        if video_info['duration'] == 0.0 and video_info['frames'] > 0 and video_info['fps'] > 0:
            video_info['duration'] = video_info['frames'] / video_info['fps']
        
        if logger:
            logger.info(f"Video info for '{os.path.basename(video_path)}': "
                       f"{video_info['frames']} frames, {video_info['fps']:.2f} FPS, "
                       f"{video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']}, "
                       f"{video_info['format']}, {video_info['bitrate']}")
        
        return video_info
        
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"ffprobe failed for '{video_path}': {e}")
    except Exception as e:
        if logger:
            logger.error(f"Error getting video info for '{video_path}': {e}")
    
    return None

def format_video_info_message(video_info, filename=None):
    """
    Format video information into a readable message.
    
    Args:
        video_info: Dictionary from get_video_info()
        filename: Optional filename to include in message
    
    Returns:
        str: Formatted message string
    """
    if not video_info:
        return "❌ Could not read video information"
    
    filename_part = f" for '{filename}'" if filename else ""
    
    # Format duration as MM:SS
    duration_minutes = int(video_info['duration'] // 60)
    duration_seconds = int(video_info['duration'] % 60)
    duration_str = f"{duration_minutes}:{duration_seconds:02d}"
    
    # Format file size info if available
    resolution_str = f"{video_info['width']}x{video_info['height']}"
    
    message = (
        f"📹 Video Information{filename_part}:\n"
        f"   • Frames: {video_info['frames']:,}\n"
        f"   • FPS: {video_info['fps']:.2f}\n"
        f"   • Duration: {duration_str} ({video_info['duration']:.2f}s)\n"
        f"   • Resolution: {resolution_str}\n"
        f"   • Format: {video_info['format']}\n"
        f"   • Bitrate: {video_info['bitrate']}"
    )
    
    return message 

def natural_sort_key(text):
    """
    Natural sorting key function to handle numeric sorting properly.
    Converts '2.png' to come before '12.png' instead of lexicographic sorting.
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', text)]

def get_supported_image_extensions():
    """Get list of supported image extensions."""
    return ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.jp2', '.dpx', '.bmp', '.webp']

def create_video_from_input_frames(
    input_frames_dir, output_path, fps=30.0, 
    ffmpeg_preset="medium", ffmpeg_quality_value=23, ffmpeg_use_gpu=False, logger=None
):
    """
    Create video from input frame directory with natural sorting.
    Supports multiple image formats: jpg, png, tiff, jp2, dpx, etc.
    
    Args:
        input_frames_dir: Directory containing input frames
        output_path: Path for output video file
        fps: Target FPS for output video
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality_value: Quality value for encoding
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
    
    Returns:
        bool: Success status
    """
    if logger:
        logger.info(f"Creating video from input frames in '{input_frames_dir}' to '{output_path}' at {fps} FPS")
    
    try:
        # Get all supported image files
        supported_extensions = get_supported_image_extensions()
        frame_files = []
        
        for file in os.listdir(input_frames_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                frame_files.append(file)
        
        if not frame_files:
            if logger:
                logger.error(f"No supported image files found in {input_frames_dir}")
            return False
        
        # Apply natural sorting to fix 2.png vs 12.png issue
        frame_files = sorted(frame_files, key=natural_sort_key)
        
        if logger:
            logger.info(f"Found {len(frame_files)} frames: {frame_files[:5]}{'...' if len(frame_files) > 5 else ''}")
        
        # Create temporary symlinks with sequential naming for ffmpeg
        import tempfile
        import shutil
        
        # Create a temporary directory that we manage manually for better control
        temp_dir = tempfile.mkdtemp(prefix="frame_conversion_")
        
        try:
            # Create sequential frame links (1-based indexing to match FFmpeg pattern)
            for i, frame_file in enumerate(frame_files):
                src_path = os.path.join(input_frames_dir, frame_file)
                dst_path = os.path.join(temp_dir, f"frame_{i+1:06d}.png")
                
                # Copy or convert to PNG if needed
                if frame_file.lower().endswith('.png'):
                    # For PNG files, always copy (symlinks can cause issues on Windows)
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        if logger:
                            logger.warning(f"Could not copy {frame_file}: {e}")
                        continue
                else:
                    # Convert other formats to PNG using OpenCV
                    try:
                        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            cv2.imwrite(dst_path, img)
                        else:
                            if logger:
                                logger.warning(f"Could not read image: {frame_file}")
                            continue
                    except Exception as e:
                        if logger:
                            logger.warning(f"Could not convert {frame_file}: {e}")
                        continue
            
            # Verify frames were copied
            copied_frames = [f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')]
            if logger:
                logger.info(f"Copied {len(copied_frames)} frames to temporary directory")
            
            if len(copied_frames) == 0:
                if logger:
                    logger.error("No frames were successfully copied to temporary directory")
                return False
            
            # Now use the existing create_video_from_frames function
            result = create_video_from_frames(
                temp_dir, output_path, fps, 
                ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger
            )
            
            return result
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                if logger:
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not clean up temporary directory {temp_dir}: {e}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error creating video from input frames: {e}")
        return False 