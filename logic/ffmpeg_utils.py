import os
import subprocess
import gradio as gr
import cv2

def run_ffmpeg_command(cmd, desc="ffmpeg command", logger=None):
    """Run an FFmpeg command with error handling."""
    if logger:
        logger.info(f"Running {desc}: {cmd}")
    try:
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if process.stdout and logger:
            logger.info(f"{desc} stdout: {process.stdout.strip()}")

        if process.returncode != 0 or (process.stderr and ('error' in process.stderr.lower() or 'warning' in process.stderr.lower())):
            if process.stderr and logger:
                logger.info(f"{desc} stderr: {process.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Error running {desc}:")
            logger.error(f"  Command: {e.cmd}")
            logger.error(f"  Return code: {e.returncode}")
            if e.stdout:
                logger.error(f"  Stdout: {e.stdout.strip()}")
            if e.stderr:
                logger.error(f"  Stderr: {e.stderr.strip()}")
        raise gr.Error(f"ffmpeg command failed (see console for details): {e.stderr.strip()[:500] if e.stderr else 'Unknown ffmpeg error'}")
    except Exception as e_gen:
        if logger:
            logger.error(f"Unexpected error preparing/running {desc} for command '{cmd}': {e_gen}")
        raise gr.Error(f"ffmpeg command failed: {e_gen}")

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

    frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')])
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

def decrease_fps(input_video_path, output_video_path, target_fps, interpolation_method="drop", ffmpeg_preset="medium", ffmpeg_quality_value=23, ffmpeg_use_gpu=False, logger=None):
    """
    Decrease video FPS using FFmpeg.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video
        target_fps: Target FPS (float)
        interpolation_method: "drop" for dropping frames, "blend" for blending frames
        ffmpeg_preset: FFmpeg preset for encoding
        ffmpeg_quality_value: Quality value for encoding
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
    
    Returns:
        tuple: (success: bool, output_fps: float, message: str)
    """
    if not os.path.exists(input_video_path):
        error_msg = f"Input video not found: {input_video_path}"
        if logger:
            logger.error(error_msg)
        return False, 0.0, error_msg
    
    # Get current FPS
    current_fps = get_video_fps(input_video_path, logger)
    
    # Check if FPS decrease is necessary
    if current_fps <= target_fps:
        msg = f"Input FPS ({current_fps:.2f}) is already at or below target FPS ({target_fps:.2f}). No FPS decrease needed."
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
        logger.info(f"Decreasing FPS from {current_fps:.2f} to {target_fps:.2f} using method '{interpolation_method}'")
    
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
        
        # Check if we need to fallback to CPU due to small resolution
        use_cpu_fallback = (ffmpeg_use_gpu and frame_width is not None and frame_height is not None and 
                           is_resolution_too_small_for_nvenc(frame_width, frame_height, logger))
        
        # Build encoding options
        video_codec_opts = ""
        if ffmpeg_use_gpu and not use_cpu_fallback:
            nvenc_preset = ffmpeg_preset
            if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
                nvenc_preset = "fast"
            elif ffmpeg_preset in ["slower", "veryslow"]:
                nvenc_preset = "slow"
            
            video_codec_opts = f'-c:v h264_nvenc -preset:v {nvenc_preset} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
            if logger:
                logger.info(f"Using NVIDIA NVENC for FPS decrease with preset {nvenc_preset} and CQ {ffmpeg_quality_value}.")
        else:
            if use_cpu_fallback and logger:
                logger.info(f"Falling back to CPU encoding for FPS decrease due to small video resolution: {frame_width}x{frame_height}")
            video_codec_opts = f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'
            if logger:
                logger.info(f"Using libx264 for FPS decrease with preset {ffmpeg_preset} and CRF {ffmpeg_quality_value}.")
        
        # Build FFmpeg command
        cmd = f'ffmpeg -y -i "{input_video_path}" -filter:v "{fps_filter}" {video_codec_opts} -c:a copy "{output_video_path}"'
        
        # Run the command
        run_ffmpeg_command(cmd, f"FPS Decrease ({interpolation_method})", logger)
        
        # Verify output and get final FPS
        if os.path.exists(output_video_path):
            final_fps = get_video_fps(output_video_path, logger)
            success_msg = f"FPS decreased successfully from {current_fps:.2f} to {final_fps:.2f} FPS using {interpolation_method} method"
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

def is_resolution_too_small_for_nvenc(width, height, logger=None):
    """Check if resolution is too small for NVENC (minimum 145x96)."""
    min_width, min_height = 145, 96
    too_small = width < min_width or height < min_height
    if too_small and logger:
        logger.warning(f"Resolution {width}x{height} is below NVENC minimum ({min_width}x{min_height}), will fallback to CPU encoding")
    return too_small

def create_video_from_frames(frame_dir, output_path, fps, ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=None):
    """Create video from frames using FFmpeg."""
    if logger:
        logger.info(f"Creating video from frames in '{frame_dir}' to '{output_path}' at {fps} FPS with preset: {ffmpeg_preset}, quality: {ffmpeg_quality_value}, GPU: {ffmpeg_use_gpu}")
    input_pattern = os.path.join(frame_dir, "frame_%06d.png")

    # Detect frame resolution for NVENC minimum size check
    frame_width, frame_height = None, None
    if ffmpeg_use_gpu:
        # Find first frame to check resolution
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
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

    # Check if we need to fallback to CPU due to small resolution
    use_cpu_fallback = (ffmpeg_use_gpu and frame_width is not None and frame_height is not None and 
                       is_resolution_too_small_for_nvenc(frame_width, frame_height, logger))

    video_codec_opts = ""
    if ffmpeg_use_gpu and not use_cpu_fallback:
        nvenc_preset = ffmpeg_preset
        if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
            nvenc_preset = "fast"
        elif ffmpeg_preset in ["slower", "veryslow"]:
            nvenc_preset = "slow"

        video_codec_opts = f'-c:v h264_nvenc -preset:v {nvenc_preset} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
        if logger:
            logger.info(f"Using NVIDIA NVENC with preset {nvenc_preset} and CQ {ffmpeg_quality_value}.")
    else:
        if use_cpu_fallback and logger:
            logger.info(f"Falling back to CPU encoding for video creation due to small frame resolution: {frame_width}x{frame_height}")
        video_codec_opts = f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'
        if logger:
            logger.info(f"Using libx264 with preset {ffmpeg_preset} and CRF {ffmpeg_quality_value}.")

    cmd = f'ffmpeg -y -framerate {fps} -i "{input_pattern}" {video_codec_opts} "{output_path}"'
    run_ffmpeg_command(cmd, "Video Reassembly (silent)", logger) 