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
import numpy as np

from .ffmpeg_utils import run_ffmpeg_command
from .file_utils import (
    sanitize_filename, get_video_resolution, cleanup_temp_dir,
    check_disk_space_for_rife, validate_video_file, safe_file_replace,
    cleanup_rife_temp_files, create_backup_file, restore_from_backup
)
from .metadata_handler import save_rife_metadata
from . import config


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
    
    # Always use the original multiplier for RIFE processing to generate intermediate frames
    # FPS limiting will be applied during the final ffmpeg re-encoding step
    actual_multiplier = multiplier
    
    # Set up RIFE model directory using configured path
    if config.RIFE_MODEL_PATH and os.path.exists(config.RIFE_MODEL_PATH):
        model_dir = config.RIFE_MODEL_PATH
    else:
        # Fallback to relative path if config not available or path doesn't exist
        model_dir = os.path.abspath(os.path.join("..", "Practical-RIFE", "train_log"))
        if not os.path.exists(model_dir):
            # Try current directory structure (for backwards compatibility)
            model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
    
    if not os.path.exists(model_dir):
        raise gr.Error(f"RIFE model directory not found: {model_dir}")
    
    # Set up RIFE script path
    if config.RIFE_MODEL_PATH:
        # Use the configured path to determine script location
        rife_base_dir = os.path.dirname(config.RIFE_MODEL_PATH)
        rife_script_path = os.path.join(rife_base_dir, "inference_video.py")
    else:
        # Fallback paths
        rife_script_path = os.path.abspath(os.path.join("..", "Practical-RIFE", "inference_video.py"))
        if not os.path.exists(rife_script_path):
            rife_script_path = os.path.abspath(os.path.join("Practical-RIFE", "inference_video.py"))
    
    if not os.path.exists(rife_script_path):
        raise gr.Error(f"RIFE script not found: {rife_script_path}")
    
    # Build RIFE command
    rife_cmd = [
        f'"{sys.executable}" "{rife_script_path}"',
        f'--video "{video_path}"',
        f'--output "{output_path}"',
        f'--model "{model_dir}"',
        f'--multi {int(actual_multiplier)}'
    ]
    
    # When FPS limiting is enabled, pass the limited FPS to RIFE for proper ffmpeg setup
    if enable_fps_limit and final_fps != target_fps:
        rife_cmd.append(f"--fps {final_fps:.3f}")
    
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


def increase_fps_standalone(
    input_video_path, 
    output_dir=None, 
    multiplier=2, 
    fp16=True, 
    uhd=False, 
    scale=1.0,
    skip_static=False, 
    enable_fps_limit=False, 
    max_fps_limit=60,
    ffmpeg_preset="medium", 
    ffmpeg_quality_value=18, 
    ffmpeg_use_gpu=True,
    keep_original=True,
    overwrite_original=False,
    seed=99,
    logger=None, 
    progress=None
):
    """
    Standalone RIFE FPS increase function for use in Gradio app.
    
    This function is designed to be called independently without upscaling,
    similar to the auto-caption feature. It uses the app-provided seed value
    and includes metadata generation as per user rules.
    
    Args:
        input_video_path: Path to input video file
        output_dir: Directory for output video (defaults to same as input)
        multiplier: FPS multiplication factor
        fp16: Use FP16 precision
        uhd: Use UHD mode
        scale: Scale factor for processing
        skip_static: Skip static frames
        enable_fps_limit: Enable FPS limiting
        max_fps_limit: Maximum FPS limit
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality_value: FFmpeg quality setting
        ffmpeg_use_gpu: Use GPU acceleration for FFmpeg
        keep_original: Keep original video file
        overwrite_original: Overwrite original file with output
        seed: Seed value from app (no hard-coded seeds as per user rules)
        logger: Logger instance
        progress: Progress callback for Gradio
        
    Returns:
        tuple: (output_video_path, status_message)
    """
    
    if logger:
        logger.info(f"Starting standalone RIFE FPS increase for: {os.path.basename(input_video_path)}")
        logger.info(f"Using seed: {seed} (from app, not hard-coded)")
    
    # Validate input video
    if not input_video_path or not os.path.exists(input_video_path):
        error_msg = "Please provide a valid input video file"
        if logger:
            logger.error(error_msg)
        return None, error_msg
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(input_video_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Call the existing increase_fps_single function
        output_path, status_message = increase_fps_single(
            video_path=input_video_path,
            output_path=None,  # Let function generate path
            multiplier=multiplier,
            fp16=fp16,
            uhd=uhd,
            scale=scale,
            skip_static=skip_static,
            enable_fps_limit=enable_fps_limit,
            max_fps_limit=max_fps_limit,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality_value=ffmpeg_quality_value,
            ffmpeg_use_gpu=ffmpeg_use_gpu,
            overwrite_original=overwrite_original,
            keep_original=keep_original,
            output_dir=output_dir,
            seed=seed,  # Use app-provided seed
            logger=logger,
            progress=progress
        )
        
        if output_path:
            success_msg = f"RIFE FPS increase completed successfully. Output: {os.path.basename(output_path)}"
            if logger:
                logger.info(success_msg)
            return output_path, success_msg
        else:
            error_msg = f"RIFE processing failed: {status_message}"
            if logger:
                logger.error(error_msg)
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error during standalone RIFE processing: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        return None, error_msg


def rife_fps_only_wrapper(
    input_video_val,
    rife_multiplier_val=2,
    rife_fp16_val=True,
    rife_uhd_val=False,
    rife_scale_val=1.0,
    rife_skip_static_val=False,
    rife_enable_fps_limit_val=False,
    rife_max_fps_limit_val=60,
    ffmpeg_preset_dropdown_val="medium",
    ffmpeg_quality_slider_val=18,
    ffmpeg_use_gpu_check_val=True,
    seed_num_val=99,
    random_seed_check_val=False,
    output_dir=None,
    logger=None,
    progress=None
):
    """
    Wrapper function for RIFE FPS increase in Gradio app.
    
    This function handles the business logic for standalone RIFE processing,
    following user rules for moving logic out of secourses_app.py and 
    keeping the app file minimal.
    
    Args:
        input_video_val: Input video from Gradio
        rife_multiplier_val: FPS multiplier from UI
        rife_fp16_val: FP16 setting from UI
        rife_uhd_val: UHD setting from UI
        rife_scale_val: Scale setting from UI
        rife_skip_static_val: Skip static frames setting from UI
        rife_enable_fps_limit_val: Enable FPS limit setting from UI
        rife_max_fps_limit_val: Max FPS limit from UI
        ffmpeg_preset_dropdown_val: FFmpeg preset from UI
        ffmpeg_quality_slider_val: FFmpeg quality from UI
        ffmpeg_use_gpu_check_val: GPU usage setting from UI
        seed_num_val: Seed value from UI (following user rules)
        random_seed_check_val: Random seed checkbox from UI
        output_dir: Output directory (defaults to app config)
        logger: Logger instance
        progress: Progress callback
        
    Returns:
        tuple: (output_video_path, status_message) for Gradio display
    """
    
    # Input validation
    if input_video_val is None:
        error_msg = "Please upload a video file first"
        if logger:
            logger.warning(error_msg)
        return None, error_msg
    
    # Handle seed logic following user rules (no hard-coded seeds)
    actual_seed_to_use = seed_num_val
    if random_seed_check_val:
        actual_seed_to_use = np.random.randint(0, 2**31)
        if logger:
            logger.info(f"Random seed checkbox is checked. Using generated seed: {actual_seed_to_use}")
    elif seed_num_val == -1:
        actual_seed_to_use = np.random.randint(0, 2**31)
        if logger:
            logger.info(f"Seed input is -1. Using generated seed: {actual_seed_to_use}")
    else:
        if logger:
            logger.info(f"Using provided seed: {actual_seed_to_use}")
    
    # Set default output directory if not provided
    if output_dir is None:
        # Import config to get default output directory
        from . import config
        output_dir = getattr(config, 'DEFAULT_OUTPUT_DIR', os.path.dirname(input_video_val))
    
    if logger:
        logger.info(f"Starting RIFE FPS only processing with multiplier: {rife_multiplier_val}x")
    
    try:
        # Call the standalone RIFE function
        output_path, status_message = increase_fps_standalone(
            input_video_path=input_video_val,
            output_dir=output_dir,
            multiplier=rife_multiplier_val,
            fp16=rife_fp16_val,
            uhd=rife_uhd_val,
            scale=rife_scale_val,
            skip_static=rife_skip_static_val,
            enable_fps_limit=rife_enable_fps_limit_val,
            max_fps_limit=rife_max_fps_limit_val,
            ffmpeg_preset=ffmpeg_preset_dropdown_val,
            ffmpeg_quality_value=ffmpeg_quality_slider_val,
            ffmpeg_use_gpu=ffmpeg_use_gpu_check_val,
            keep_original=True,  # Always keep original for standalone processing
            overwrite_original=False,  # Don't overwrite for standalone processing
            seed=actual_seed_to_use,  # Use app-generated seed (following user rules)
            logger=logger,
            progress=progress
        )
        
        return output_path, status_message
        
    except Exception as e:
        error_msg = f"RIFE FPS processing failed: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        return None, error_msg 