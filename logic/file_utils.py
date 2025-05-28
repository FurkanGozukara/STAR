import os
import sys
import platform
import subprocess
import shutil
import re
import cv2
import gradio as gr
from pathlib import Path
import string
from ctypes import windll

def sanitize_filename(filename):
    """Sanitize filename by removing invalid characters."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    if len(filename) > 200:
        filename = filename[:200]
    return filename if filename else "unnamed"

def get_batch_filename(output_dir, original_filename):
    """Generate batch filename and paths."""
    base_name = Path(original_filename).stem
    sanitized_name = sanitize_filename(base_name)
    
    batch_output_path = os.path.join(output_dir, sanitized_name)
    os.makedirs(batch_output_path, exist_ok=True)
    
    output_video_path = os.path.join(batch_output_path, f"{sanitized_name}.mp4")
    
    return sanitized_name, output_video_path, batch_output_path

def open_folder(folder_path, logger=None):
    """Open folder in system file explorer."""
    if logger:
        logger.info(f"Attempting to open folder: {folder_path}")
    if not os.path.isdir(folder_path):
        if logger:
            logger.warning(f"Folder does not exist or is not a directory: {folder_path}")
        gr.Warning(f"Output folder '{folder_path}' does not exist yet. Please run an upscale first.")
        return
    try:
        if sys.platform == "win32":
            os.startfile(os.path.normpath(folder_path))
        elif sys.platform == "darwin":
            subprocess.run(['open', folder_path], check=True)
        else:
            subprocess.run(['xdg-open', folder_path], check=True)
        if logger:
            logger.info(f"Successfully requested to open folder: {folder_path}")
    except FileNotFoundError:
        if logger:
            logger.error(f"File explorer command (e.g., xdg-open, open) not found for platform {sys.platform}. Cannot open folder.")
        gr.Error(f"Could not find a file explorer utility for your system ({sys.platform}).")
    except Exception as e:
        if logger:
            logger.error(f"Failed to open folder '{folder_path}': {e}")
        gr.Error(f"Failed to open folder: {e}")

def get_next_filename(output_dir, logger=None):
    """Get next available filename in output directory."""
    os.makedirs(output_dir, exist_ok=True)
    max_num = 0
    existing_mp4_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    for f_name in existing_mp4_files:
        try:
            num = int(os.path.splitext(f_name)[0])
            if num > max_num:
                max_num = num
        except ValueError:
            continue

    current_num_to_try = max_num + 1
    while True:
        base_filename_no_ext = f"{current_num_to_try:04d}"
        tmp_lock_file_path = os.path.join(output_dir, f"{base_filename_no_ext}.tmp")
        full_output_path = os.path.join(output_dir, f"{base_filename_no_ext}.mp4")

        try:
            fd = os.open(tmp_lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return base_filename_no_ext, full_output_path
        except FileExistsError:
            current_num_to_try += 1
        except Exception as e:
            if logger:
                logger.error(f"Error trying to create lock file {tmp_lock_file_path}: {e}")
            current_num_to_try += 1
            if current_num_to_try > max_num + 1000:
                if logger:
                    logger.error("Failed to secure a lock file after many attempts. Aborting get_next_filename.")
                raise IOError("Could not secure a unique filename lock.")

def cleanup_temp_dir(temp_dir, logger=None):
    """Clean up temporary directory."""
    if temp_dir and os.path.exists(temp_dir) and os.path.isdir(temp_dir):
        if logger:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            if logger:
                logger.error(f"Error removing temporary directory '{temp_dir}': {e}")

def get_video_resolution(video_path, logger=None):
    """Get video resolution using ffprobe or OpenCV."""
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "{video_path}"'
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        w_str, h_str = process.stdout.strip().split('x')
        w, h = int(w_str), int(h_str)
        if logger:
            logger.info(f"Video resolution (wxh) from ffprobe for '{video_path}': {w}x{h}")
        return h, w
    except Exception as e:
        if logger:
            logger.warning(f"ffprobe failed for '{video_path}' ({e}), trying OpenCV...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise gr.Error(f"Cannot open video file: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if h > 0 and w > 0:
            if logger:
                logger.info(f"Video resolution (wxh) from OpenCV for '{video_path}': {w}x{h}")
            return h, w
        raise gr.Error(f"Could not determine resolution for video: {video_path}")

def get_available_drives(default_output_dir, base_path, logger=None):
    """Get available drives/paths for file browser."""
    available_paths = []
    if platform.system() == "Windows":
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin":
        available_paths = ["/", "/Volumes"]
    else:
        available_paths = ["/", "/mnt", "/media"]

        home_dir = os.path.expanduser("~")
        if home_dir not in available_paths:
            available_paths.append(home_dir)

    existing_paths = [p for p in available_paths if os.path.exists(p) and os.path.isdir(p)]

    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if cwd not in existing_paths:
        existing_paths.append(cwd)
    if script_dir not in existing_paths:
        existing_paths.append(script_dir)

    abs_default_output_dir = os.path.abspath(default_output_dir)
    if not any(abs_default_output_dir.startswith(os.path.abspath(p)) for p in existing_paths):
        parent_default_output_dir = os.path.dirname(abs_default_output_dir)
        if parent_default_output_dir not in existing_paths:
            existing_paths.append(parent_default_output_dir)

    if base_path not in existing_paths and os.path.isdir(base_path):
        existing_paths.append(base_path)

    unique_paths = sorted(list(set(os.path.abspath(p) for p in existing_paths if os.path.isdir(p))))
    if logger:
        logger.info(f"Effective Gradio allowed_paths: {unique_paths}")
    return unique_paths 