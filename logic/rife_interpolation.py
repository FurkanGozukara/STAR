import os
import sys
import time
import subprocess
import cv2
import torch
import glob
import gradio as gr
from pathlib import Path
import re

from .ffmpeg_utils import run_ffmpeg_command
from .file_utils import (
    sanitize_filename, get_video_resolution, cleanup_temp_dir,
    check_disk_space_for_rife, validate_video_file, safe_file_replace,
    cleanup_rife_temp_files, create_backup_file, restore_from_backup
)
from .metadata_handler import save_rife_metadata


def get_video_fps(video_path, logger=None):
    """Get video FPS using ffprobe."""
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        rate_str = process.stdout.strip()
        if '/' in rate_str:
            num, den = map(int, rate_str.split('/'))
            if den != 0:
                fps = num / den
            else:
                fps = 30.0
        elif rate_str:
            fps = float(rate_str)
        else:
            fps = 30.0
        if logger:
            logger.info(f"Detected FPS: {fps}")
        return fps
    except Exception as e:
        if logger:
            logger.warning(f"Could not get FPS using ffprobe for '{video_path}': {e}. Using default 30.0 FPS.")
        return 30.0


def video_has_audio(video_path, logger=None):
    """Check if video has audio track."""
    try:
        cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        has_audio = bool(process.stdout.strip())
        if logger:
            logger.info(f"Audio detected in '{video_path}': {has_audio}")
        return has_audio
    except Exception as e:
        if logger:
            logger.warning(f"Could not check audio for '{video_path}': {e}. Assuming no audio.")
        return False


def limit_fps_if_needed(target_fps, max_fps_limit, enable_fps_limit, logger=None):
    """Limit FPS if enabled and target exceeds limit."""
    if not enable_fps_limit:
        return target_fps
    
    if target_fps > max_fps_limit:
        if logger:
            logger.info(f"Limiting FPS from {target_fps} to {max_fps_limit}")
        return max_fps_limit
    
    return target_fps


def generate_rife_filename(input_path, output_dir, multiplier, overwrite_original=False, logger=None):
    """Generate appropriate filename for RIFE output."""
    input_name = Path(input_path).stem
    sanitized_name = sanitize_filename(input_name)
    
    if overwrite_original:
        # Use same name as input
        output_path = os.path.join(output_dir, f"{sanitized_name}.mp4")
    else:
        # Add suffix based on multiplier
        suffix = f"_{multiplier}x_FPS"
        output_path = os.path.join(output_dir, f"{sanitized_name}{suffix}.mp4")
    
    if logger:
        logger.info(f"Generated RIFE filename: {output_path}")
    
    return output_path


def increase_fps_single(video_path, output_path=None, multiplier=2, fp16=True, uhd=False, scale=1.0, 
                       skip_static=False, enable_fps_limit=False, max_fps_limit=60,
                       ffmpeg_preset="medium", ffmpeg_quality_value=18, ffmpeg_use_gpu=True,
                       overwrite_original=False, keep_original=True, output_dir=None,
                       seed=99, logger=None, progress=None):
    """Process a single video with RIFE to increase FPS with enhanced file management."""
    
    if logger:
        logger.info(f"Starting RIFE processing for: {os.path.basename(video_path)}")
    
    # Step 1: Validate input video
    valid, validation_msg = validate_video_file(video_path, check_playable=True, logger=logger)
    if not valid:
        return None, f"Input validation failed: {validation_msg}"
    
    # Step 2: Set up output path
    if output_path is None:
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        output_path = generate_rife_filename(video_path, output_dir, multiplier, overwrite_original, logger)
    
    if logger:
        logger.info(f"Output: {os.path.basename(output_path)}")
    
    # Step 3: Check disk space requirements
    space_ok, space_msg = check_disk_space_for_rife(video_path, output_path, multiplier, logger)
    if not space_ok:
        return None, f"Disk space check failed: {space_msg}"
    elif logger:
        logger.info(f"Disk space check: {space_msg}")
    
    # Step 4: Clean up any existing temp files in output directory
    output_dir_for_cleanup = os.path.dirname(output_path)
    cleanup_success, cleanup_msg = cleanup_rife_temp_files(output_dir_for_cleanup, logger)
    if logger:
        logger.info(f"Temp file cleanup: {cleanup_msg}")
    
    # Check for audio
    skip_audio = not video_has_audio(video_path, logger)
    if skip_audio and logger:
        logger.info("No audio track detected in source video")
    
    # Get original FPS and calculate target FPS
    original_fps = get_video_fps(video_path, logger)
    target_fps = original_fps * multiplier
    
    # Apply FPS limiting if enabled
    final_fps = limit_fps_if_needed(target_fps, max_fps_limit, enable_fps_limit, logger)
    
    # Determine actual multiplier based on final FPS
    actual_multiplier = final_fps / original_fps if original_fps > 0 else multiplier
    
    # Set up RIFE model directory
    model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
    if not os.path.exists(model_dir):
        raise gr.Error(f"RIFE model directory not found: {model_dir}")
    
    # Build RIFE command
    rife_cmd = [
        f'"{sys.executable}" "Practical-RIFE/inference_video.py"',
        f'--video "{video_path}"',
        f'--output "{output_path}"',
        f'--model "{model_dir}"',
        f'--multi {int(actual_multiplier)}'
    ]
    
    if fp16:
        rife_cmd.append("--fp16")
    if uhd:
        rife_cmd.append("--UHD")
    if scale != 1.0:
        rife_cmd.append(f"--scale {scale}")
    if skip_static:
        rife_cmd.append("--skip")
    if skip_audio:
        rife_cmd.append("--no-audio")
    if not ffmpeg_use_gpu:
        rife_cmd.append("--show-ffmpeg")
    
    # Execute RIFE processing
    start_time = time.time()
    cmd = " ".join(rife_cmd)
    
    if logger:
        logger.info(f"Starting RIFE processing with command: {cmd}")
    
    try:
        # Create process with line buffering for immediate output
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time with progress updates
        output_lines = []
        last_progress_line = ""
        
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if line.startswith("\r[") and "%" in line:
                    try:
                        # Extract progress information
                        frame_info = line.split(']')[1].strip()
                        current_frame = int(frame_info.split('/')[0])
                        total_frames = int(frame_info.split('/')[1].split()[0])
                        
                        if progress:
                            progress_percent = current_frame / total_frames
                            progress(progress_percent, f"Processing frame {current_frame}/{total_frames}")
                        
                        if current_frame == 1 or current_frame % 100 == 0 or "100.0%" in line:
                            if logger:
                                logger.info(f"RIFE Progress: {line}")
                            last_progress_line = line
                    except (IndexError, ValueError):
                        if logger:
                            logger.info(f"RIFE Progress: {line}")
                        last_progress_line = line
                elif "Initializing" in line or ("Processing" in line and "complete" in line):
                    if logger:
                        logger.info(f"RIFE: {line}")
                elif "frames loaded" in line and "Pre-loading: Complete" in line:
                    if logger:
                        logger.info(f"RIFE: {line}")
                elif "[SUCCESS]" in line or "[DONE]" in line:
                    if logger:
                        logger.info(f"RIFE: {line}")
                elif not any(skip_term in line for skip_term in [
                    "frame=", "fps=", "Input #0", "Output #0", "Stream #", 
                    "Press [q]", "configuration:", "libavutil", "built with", 
                    "encoder:", "ffmpeg version", "compatible_brands", "creation_time"
                ]):
                    if line and "[INFO]" not in line and "Processing:" not in line:
                        if logger:
                            logger.info(f"RIFE: {line}")
                output_lines.append(line)
        
        return_code = process.wait()
        end_time = time.time()
        
        if return_code != 0:
            error_message = "Error processing video with RIFE. Check the output for details."
            if logger:
                logger.error(error_message)
            raise gr.Error(error_message)
        
        # Verify output file
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            error_message = "RIFE output file was not created or is empty"
            if logger:
                logger.error(error_message)
            raise gr.Error(error_message)
        
        processing_time = end_time - start_time
        if logger:
            logger.info(f"RIFE processing completed in {processing_time:.2f} seconds")
        
        # Verify video is readable
        try:
            check_cap = cv2.VideoCapture(output_path)
            if check_cap.isOpened():
                ret, _ = check_cap.read()
                check_cap.release()
                if not ret:
                    if logger:
                        logger.warning(f"Output video exists but cannot be read: {output_path}")
            else:
                if logger:
                    logger.warning(f"Output video exists but cannot be opened: {output_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Error verifying output video: {str(e)}")
        
        # Save metadata using the proper metadata handler
        rife_params = {
            "multiplier": multiplier,
            "actual_multiplier": actual_multiplier,
            "fp16": fp16,
            "uhd": uhd,
            "scale": scale,
            "skip_static": skip_static,
            "original_fps": original_fps,
            "target_fps": target_fps,
            "final_fps": final_fps,
            "fps_limit_enabled": enable_fps_limit,
            "fps_limit": max_fps_limit if enable_fps_limit else None,
            "apply_to_chunks": False,  # Single video processing
            "apply_to_scenes": False,  # Single video processing
            "keep_original": keep_original,
            "overwrite_original": overwrite_original,
            "ffmpeg_preset": ffmpeg_preset,
            "ffmpeg_quality_value": ffmpeg_quality_value,
            "ffmpeg_use_gpu": ffmpeg_use_gpu
        }
        
        try:
            success, message = save_rife_metadata(output_path, video_path, rife_params, processing_time, seed, logger)
            if success and logger:
                logger.info(f"RIFE metadata saved successfully: {message}")
        except Exception as e:
            if logger:
                logger.warning(f"Could not save RIFE metadata: {str(e)}")
        
        # Step 6: Enhanced file handling with validation and backup
        if overwrite_original:
            try:
                # Use safe file replacement with automatic backup and validation
                replace_success, replace_msg, backup_path = safe_file_replace(
                    source_path=output_path,
                    target_path=video_path,
                    create_backup=keep_original,
                    logger=logger
                )
                
                if replace_success:
                    if logger:
                        logger.info(f"Successfully overwrote original file: {video_path}")
                        if backup_path and keep_original:
                            logger.info(f"Original backed up to: {backup_path}")
                    
                    return video_path, "RIFE processing completed successfully (original overwritten with validation)"
                else:
                    if logger:
                        logger.error(f"Safe file replacement failed: {replace_msg}")
                    
                    # Return the RIFE output path as fallback
                    return output_path, f"RIFE processing completed but safe replacement failed: {replace_msg}"
                    
            except Exception as e:
                if logger:
                    logger.error(f"Error during safe file replacement: {str(e)}")
                # Return the interpolated file path as fallback
                return output_path, f"RIFE processing completed but file replacement error: {str(e)}"
        
        # Step 7: Final validation of output file
        final_valid, final_msg = validate_video_file(output_path, check_playable=True, logger=logger)
        if not final_valid:
            if logger:
                logger.error(f"Final output validation failed: {final_msg}")
            return None, f"RIFE processing completed but output validation failed: {final_msg}"
        
        if logger:
            logger.info(f"RIFE processing completed successfully with validation: {output_path}")
        
        return output_path, "RIFE processing completed successfully with validation"
        
    except subprocess.CalledProcessError as e:
        error_message = f"RIFE process failed with return code {e.returncode}"
        if logger:
            logger.error(f"{error_message}: {e.stderr if e.stderr else 'No error details'}")
        raise gr.Error(error_message)
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error during RIFE processing: {str(e)}")
        raise gr.Error(f"RIFE processing failed: {str(e)}")


def batch_process_fps_increase(input_folder, output_folder, multiplier=2, fp16=True, uhd=False, 
                              scale=1.0, skip_static=False, enable_fps_limit=False, max_fps_limit=60,
                              ffmpeg_preset="medium", ffmpeg_quality_value=18, ffmpeg_use_gpu=True,
                              overwrite_original=False, keep_original=True, skip_existing=True, 
                              include_subfolders=True, seed=99, logger=None, progress=None):
    """Batch process multiple videos with RIFE to increase FPS."""
    
    if logger:
        logger.info(f"Starting batch RIFE processing from '{input_folder}' to '{output_folder}'")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    
    if logger:
        logger.info("Scanning for video files...")
    
    if include_subfolders:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, '**', f'*{ext}'), recursive=True))
    else:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
    
    if logger:
        logger.info(f"Found {len(video_files)} videos to process")
    
    if len(video_files) == 0:
        return "No video files found to process"
    
    # Process statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, video_file in enumerate(video_files):
        rel_path = os.path.relpath(video_file, input_folder)
        
        # Create output path maintaining directory structure
        if overwrite_original:
            output_path = video_file  # Overwrite in place
        else:
            output_path = generate_rife_filename(
                video_file, 
                os.path.join(output_folder, os.path.dirname(rel_path)), 
                multiplier, 
                overwrite_original, 
                logger
            )
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Update progress
        batch_progress = (i + 1) / len(video_files)
        if progress:
            progress(batch_progress, f"Processing video {i+1}/{len(video_files)}: {os.path.basename(rel_path)}")
        
        if logger:
            logger.info(f"\n[{i+1}/{len(video_files)}] Processing: {rel_path}")
        
        # Skip if output already exists and skip_existing is True
        if skip_existing and os.path.exists(output_path) and not overwrite_original:
            if logger:
                logger.info(f"Skipping {os.path.basename(rel_path)} (output already exists)")
            skipped_count += 1
            continue
        
        try:
            result, message = increase_fps_single(
                video_file, 
                output_path if not overwrite_original else None,
                multiplier, fp16, uhd, scale, skip_static,
                enable_fps_limit, max_fps_limit,
                ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu,
                overwrite_original, keep_original,
                os.path.dirname(output_path) if not overwrite_original else None,
                seed, logger, None  # No individual progress for batch items
            )
            
            if result:
                processed_count += 1
                if logger:
                    logger.info(f"Successfully processed {os.path.basename(rel_path)}")
            else:
                failed_count += 1
                if logger:
                    logger.error(f"Failed to process {os.path.basename(rel_path)}: {message}")
        
        except Exception as e:
            failed_count += 1
            if logger:
                logger.error(f"Error processing {os.path.basename(rel_path)}: {str(e)}")
    
    # Final summary
    summary = f"Batch RIFE processing completed: {processed_count} processed, {skipped_count} skipped, {failed_count} failed"
    if logger:
        logger.info(summary)
    
    return summary


def apply_rife_to_chunks(chunk_paths, multiplier=2, fp16=True, uhd=False, scale=1.0,
                        skip_static=False, enable_fps_limit=False, max_fps_limit=60,
                        ffmpeg_preset="medium", ffmpeg_quality_value=18, ffmpeg_use_gpu=True,
                        keep_original=True, seed=99, logger=None, progress=None):
    """Apply RIFE interpolation to video chunks."""
    
    if not chunk_paths:
        return []
    
    if logger:
        logger.info(f"Applying RIFE to {len(chunk_paths)} chunks")
    
    rife_chunk_paths = []
    
    for i, chunk_path in enumerate(chunk_paths):
        if progress:
            chunk_progress = i / len(chunk_paths)
            progress(chunk_progress, f"Applying RIFE to chunk {i+1}/{len(chunk_paths)}")
        
        try:
            # Generate RIFE chunk path
            chunk_dir = os.path.dirname(chunk_path)
            chunk_name = Path(chunk_path).stem
            rife_chunk_path = os.path.join(chunk_dir, f"{chunk_name}_{multiplier}x_FPS.mp4")
            
            result, message = increase_fps_single(
                chunk_path, rife_chunk_path, multiplier, fp16, uhd, scale,
                skip_static, enable_fps_limit, max_fps_limit,
                ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu,
                False, keep_original, chunk_dir, seed, logger, None
            )
            
            if result:
                rife_chunk_paths.append(result)
                if logger:
                    logger.info(f"Successfully applied RIFE to chunk: {os.path.basename(chunk_path)}")
            else:
                if logger:
                    logger.error(f"Failed to apply RIFE to chunk {chunk_path}: {message}")
                # Use original chunk if RIFE fails
                rife_chunk_paths.append(chunk_path)
        
        except Exception as e:
            if logger:
                logger.error(f"Error applying RIFE to chunk {chunk_path}: {str(e)}")
            # Use original chunk if RIFE fails
            rife_chunk_paths.append(chunk_path)
    
    if logger:
        logger.info(f"RIFE applied to chunks: {len([p for p in rife_chunk_paths if '_FPS' in p])} successful, {len([p for p in rife_chunk_paths if '_FPS' not in p])} failed")
    
    return rife_chunk_paths


def apply_rife_to_scenes(scene_paths, multiplier=2, fp16=True, uhd=False, scale=1.0,
                        skip_static=False, enable_fps_limit=False, max_fps_limit=60,
                        ffmpeg_preset="medium", ffmpeg_quality_value=18, ffmpeg_use_gpu=True,
                        keep_original=True, seed=99, logger=None, progress=None):
    """Apply RIFE interpolation to scene videos."""
    
    if not scene_paths:
        return []
    
    if logger:
        logger.info(f"Applying RIFE to {len(scene_paths)} scenes")
    
    rife_scene_paths = []
    
    for i, scene_path in enumerate(scene_paths):
        if progress:
            scene_progress = i / len(scene_paths)
            progress(scene_progress, f"Applying RIFE to scene {i+1}/{len(scene_paths)}")
        
        try:
            # Generate RIFE scene path
            scene_dir = os.path.dirname(scene_path)
            scene_name = Path(scene_path).stem
            rife_scene_path = os.path.join(scene_dir, f"{scene_name}_{multiplier}x_FPS.mp4")
            
            result, message = increase_fps_single(
                scene_path, rife_scene_path, multiplier, fp16, uhd, scale,
                skip_static, enable_fps_limit, max_fps_limit,
                ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu,
                False, keep_original, scene_dir, seed, logger, None
            )
            
            if result:
                rife_scene_paths.append(result)
                if logger:
                    logger.info(f"Successfully applied RIFE to scene: {os.path.basename(scene_path)}")
            else:
                if logger:
                    logger.error(f"Failed to apply RIFE to scene {scene_path}: {message}")
                # Use original scene if RIFE fails
                rife_scene_paths.append(scene_path)
        
        except Exception as e:
            if logger:
                logger.error(f"Error applying RIFE to scene {scene_path}: {str(e)}")
            # Use original scene if RIFE fails
            rife_scene_paths.append(scene_path)
    
    if logger:
        logger.info(f"RIFE applied to scenes: {len([p for p in rife_scene_paths if '_FPS' in p])} successful, {len([p for p in rife_scene_paths if '_FPS' not in p])} failed")
    
    return rife_scene_paths 