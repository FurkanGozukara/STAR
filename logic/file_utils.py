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
    """Get next available filename in output directory.
    Returns: (base_filename_no_ext, full_output_path, tmp_lock_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    max_num = 0
    
    # Check for existing MP4 files AND existing subdirectories with numeric names
    existing_entries = os.listdir(output_dir)
    for entry in existing_entries:
        # Check MP4 files
        if entry.endswith('.mp4'):
            try:
                num = int(os.path.splitext(entry)[0])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
        # Check directories with numeric names (e.g., "0215")
        elif os.path.isdir(os.path.join(output_dir, entry)):
            try:
                num = int(entry)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    current_num_to_try = max_num + 1
    while True:
        base_filename_no_ext = f"{current_num_to_try:04d}"
        tmp_lock_file_path = os.path.join(output_dir, f"{base_filename_no_ext}.tmp")
        full_output_path = os.path.join(output_dir, f"{base_filename_no_ext}.mp4")
        session_dir_path = os.path.join(output_dir, base_filename_no_ext)

        # Check if either the tmp file, the output directory, or the mp4 file already exists
        if os.path.exists(tmp_lock_file_path) or os.path.exists(session_dir_path) or os.path.exists(full_output_path):
            current_num_to_try += 1
            continue

        try:
            fd = os.open(tmp_lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return base_filename_no_ext, full_output_path, tmp_lock_file_path
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
        # Cross-platform Windows drive detection
        drives = []
        for letter in string.ascii_uppercase:
            drive_path = f"{letter}:\\"
            try:
                # Check if drive exists and is accessible
                if os.path.exists(drive_path) and os.path.isdir(drive_path):
                    drives.append(drive_path)
            except (OSError, PermissionError):
                # Skip drives that can't be accessed
                continue
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

# RIFE-specific file management utilities

def get_disk_space_info(path, logger=None):
    """Get disk space information for a given path."""
    try:
        if platform.system() == "Windows":
            free_bytes = shutil.disk_usage(path).free
        else:
            stat = os.statvfs(path)
            free_bytes = stat.f_bavail * stat.f_frsize
        
        # Convert to GB
        free_gb = free_bytes / (1024 ** 3)
        return free_gb
    except Exception as e:
        if logger:
            logger.warning(f"Could not get disk space for {path}: {e}")
        return None

def estimate_rife_output_size(input_video_path, multiplier=2, logger=None):
    """Estimate the size of RIFE output based on input video."""
    try:
        input_size = os.path.getsize(input_video_path)
        # RIFE typically increases file size by 1.5-2.5x the multiplier due to increased frames
        # This is a conservative estimate
        estimated_size = input_size * multiplier * 2.0
        return estimated_size
    except Exception as e:
        if logger:
            logger.warning(f"Could not estimate RIFE output size for {input_video_path}: {e}")
        return None

def check_disk_space_for_rife(input_video_path, output_path, multiplier=2, logger=None):
    """Check if there's sufficient disk space for RIFE processing."""
    try:
        estimated_size = estimate_rife_output_size(input_video_path, multiplier, logger)
        if estimated_size is None:
            return True, "Could not estimate space requirements"
        
        output_dir = os.path.dirname(output_path)
        free_space = get_disk_space_info(output_dir, logger)
        if free_space is None:
            return True, "Could not check available disk space"
        
        required_gb = estimated_size / (1024 ** 3)
        
        if free_space < required_gb:
            message = f"Insufficient disk space. Required: {required_gb:.2f}GB, Available: {free_space:.2f}GB"
            return False, message
        else:
            message = f"Disk space check passed. Required: {required_gb:.2f}GB, Available: {free_space:.2f}GB"
            return True, message
    except Exception as e:
        if logger:
            logger.warning(f"Error checking disk space: {e}")
        return True, f"Could not verify disk space: {e}"

def validate_video_file(video_path, check_playable=True, logger=None):
    """Validate that a video file exists and is readable."""
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return False, f"Video file does not exist: {video_path}"
        
        # Check if file has content
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return False, f"Video file is empty: {video_path}"
        
        # Check if file is readable with OpenCV
        if check_playable:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return False, f"Video file cannot be opened: {video_path}"
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False, f"Video file cannot be read (no frames): {video_path}"
        
        if logger:
            logger.info(f"Video validation passed: {video_path} ({file_size / (1024*1024):.2f}MB)")
        
        return True, f"Video validation successful ({file_size / (1024*1024):.2f}MB)"
    
    except Exception as e:
        error_msg = f"Error validating video file {video_path}: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def create_backup_file(file_path, backup_suffix=".backup", logger=None):
    """Create a backup copy of a file."""
    try:
        backup_path = f"{file_path}{backup_suffix}"
        
        # If backup already exists, create numbered backup
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{file_path}{backup_suffix}.{counter}"
            counter += 1
        
        shutil.copy2(file_path, backup_path)
        
        if logger:
            logger.info(f"Created backup: {backup_path}")
        
        return True, backup_path
    
    except Exception as e:
        error_msg = f"Failed to create backup of {file_path}: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def restore_from_backup(original_path, backup_path, logger=None):
    """Restore a file from its backup."""
    try:
        if not os.path.exists(backup_path):
            return False, f"Backup file does not exist: {backup_path}"
        
        # Remove original if it exists
        if os.path.exists(original_path):
            os.remove(original_path)
        
        # Restore from backup
        shutil.copy2(backup_path, original_path)
        
        if logger:
            logger.info(f"Restored {original_path} from backup {backup_path}")
        
        return True, f"Successfully restored from backup"
    
    except Exception as e:
        error_msg = f"Failed to restore {original_path} from backup {backup_path}: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def cleanup_backup_files(directory, backup_pattern=".backup", max_age_hours=24, logger=None):
    """Clean up old backup files in a directory."""
    try:
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if backup_pattern in file:
                    file_path = os.path.join(root, file)
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            cleaned_files.append(file_path)
                            if logger:
                                logger.info(f"Cleaned up old backup: {file_path}")
                    except Exception as e:
                        if logger:
                            logger.warning(f"Could not clean backup file {file_path}: {e}")
        
        return True, f"Cleaned up {len(cleaned_files)} backup files"
    
    except Exception as e:
        error_msg = f"Error during backup cleanup: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def safe_file_replace(source_path, target_path, create_backup=True, logger=None):
    """Safely replace a file with another, with optional backup."""
    try:
        backup_path = None
        
        # Validate source file
        valid, msg = validate_video_file(source_path, check_playable=True, logger=logger)
        if not valid:
            return False, f"Source file validation failed: {msg}", None
        
        # Create backup of target if it exists and backup is requested
        if create_backup and os.path.exists(target_path):
            backup_success, backup_result = create_backup_file(target_path, logger=logger)
            if backup_success:
                backup_path = backup_result
            else:
                if logger:
                    logger.warning(f"Could not create backup: {backup_result}")
        
        # Remove target if it exists
        if os.path.exists(target_path):
            os.remove(target_path)
        
        # Move source to target
        shutil.move(source_path, target_path)
        
        # Validate the moved file
        valid, msg = validate_video_file(target_path, check_playable=True, logger=logger)
        if not valid:
            # Restore from backup if validation fails
            if backup_path and os.path.exists(backup_path):
                if logger:
                    logger.warning(f"Replaced file failed validation, restoring backup: {msg}")
                restore_success, restore_msg = restore_from_backup(target_path, backup_path, logger)
                if not restore_success:
                    return False, f"File replacement failed and backup restore failed: {restore_msg}", backup_path
                return False, f"File replacement failed validation, restored from backup: {msg}", backup_path
            else:
                return False, f"File replacement failed validation and no backup available: {msg}", None
        
        if logger:
            logger.info(f"Successfully replaced {target_path} with {source_path}")
        
        return True, "File replacement successful", backup_path
    
    except Exception as e:
        error_msg = f"Error during file replacement: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg, backup_path

def cleanup_rife_temp_files(directory, logger=None):
    """Clean up temporary RIFE processing files."""
    try:
        import glob
        
        temp_patterns = [
            "*.temp", 
            "*_temp.mp4",
            "*_processing.mp4",
            "rife_*.mp4"
        ]
        
        cleaned_files = []
        
        for pattern in temp_patterns:
            temp_files = glob.glob(os.path.join(directory, pattern))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    cleaned_files.append(temp_file)
                    if logger:
                        logger.info(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not clean temp file {temp_file}: {e}")
        
        return True, f"Cleaned up {len(cleaned_files)} temporary files"
    
    except Exception as e:
        error_msg = f"Error during temp file cleanup: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg 