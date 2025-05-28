import os
import subprocess
import gradio as gr

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

def create_video_from_frames(frame_dir, output_path, fps, ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=None):
    """Create video from frames using FFmpeg."""
    if logger:
        logger.info(f"Creating video from frames in '{frame_dir}' to '{output_path}' at {fps} FPS with preset: {ffmpeg_preset}, quality: {ffmpeg_quality_value}, GPU: {ffmpeg_use_gpu}")
    input_pattern = os.path.join(frame_dir, "frame_%06d.png")

    video_codec_opts = ""
    if ffmpeg_use_gpu:
        nvenc_preset = ffmpeg_preset
        if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
            nvenc_preset = "fast"
        elif ffmpeg_preset in ["slower", "veryslow"]:
            nvenc_preset = "slow"

        video_codec_opts = f'-c:v h264_nvenc -preset:v {nvenc_preset} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
        if logger:
            logger.info(f"Using NVIDIA NVENC with preset {nvenc_preset} and CQ {ffmpeg_quality_value}.")
    else:
        video_codec_opts = f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'
        if logger:
            logger.info(f"Using libx264 with preset {ffmpeg_preset} and CRF {ffmpeg_quality_value}.")

    cmd = f'ffmpeg -y -framerate {fps} -i "{input_pattern}" {video_codec_opts} "{output_path}"'
    run_ffmpeg_command(cmd, "Video Reassembly (silent)", logger) 