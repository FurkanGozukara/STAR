"""
Core image upscaling processing logic.
Handles the main upscaling workflow using image-based upscaler models.
"""

import os
import time
import tempfile
import shutil
import math
import numpy as np
import cv2
import gc
import logging
from typing import List, Dict, Tuple, Optional, Any, Generator

from .image_upscaler_utils import (
    scan_for_models, load_model, unload_model, clear_model_cache,
    get_model_info, get_model_scale_factor, get_model_architecture,
    process_frames_batch, estimate_output_resolution,
    validate_model_compatibility, get_recommended_batch_size,
    extract_model_filename_from_dropdown
)

def process_video_with_image_upscaler(
    input_video_path: str,
    selected_model_filename: str,
    batch_size: int,
    upscale_models_dir: str,
    
    # Video processing parameters
    enable_target_res: bool = False,
    target_h: int = 1080,
    target_w: int = 1920,
    target_res_mode: str = "Ratio Upscale",
    
    # Frame saving and output parameters
    save_frames: bool = False,
    save_metadata: bool = False,
    save_chunks: bool = False,
    save_chunk_frames: bool = False,
    
    # Scene processing parameters
    enable_scene_split: bool = False,
    scene_video_paths: List[str] = None,
    
    # Directory and file management
    temp_dir: str = None,
    output_dir: str = None,
    base_output_filename_no_ext: str = None,
    
    # FPS parameters
    input_fps: float = 30.0,
    
    # FFmpeg parameters
    ffmpeg_preset: str = "medium",
    ffmpeg_quality_value: int = 23,
    ffmpeg_use_gpu: bool = False,
    
    # Dependencies
    logger: logging.Logger = None,
    progress = None,
    
    # Utility functions (injected dependencies)
    util_extract_frames = None,
    util_create_video_from_frames = None,
    util_get_gpu_device = None,
    format_time = None,
    
    # Metadata parameters
    params_for_metadata: Dict = None,
    metadata_handler_module = None,
    
    # Status tracking
    status_log: List[str] = None,
    current_seed: int = 99
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """
    Process video using image-based upscaler models.
    
    This function integrates with the existing pipeline but uses image upscalers
    instead of STAR models for frame processing.
    
    Args:
        input_video_path: Path to input video
        selected_model_filename: Filename of the selected upscaler model
        batch_size: Number of frames to process in each batch
        upscale_models_dir: Directory containing upscaler models
        ... (other parameters as in original run_upscale)
        
    Yields:
        Tuple of (output_video_path, status_message, chunk_video_path, chunk_status, comparison_video_path)
    """
    
    if status_log is None:
        status_log = []
    
    if params_for_metadata is None:
        params_for_metadata = {}
    
    # Initialize variables
    output_video_path = None
    last_chunk_video_path = None
    last_chunk_status = "Initializing image upscaler..."
    comparison_video_path = None
    
    # Validate model selection and extract filename
    actual_model_filename = extract_model_filename_from_dropdown(selected_model_filename)
    if not actual_model_filename:
        error_msg = "No valid image upscaler model selected"
        status_log.append(f"Error: {error_msg}")
        yield None, "\n".join(status_log), None, error_msg, None
        return
    
    model_path = os.path.join(upscale_models_dir, actual_model_filename)
    
    # Validate model file
    is_valid, validation_message = validate_model_compatibility(model_path, logger)
    if not is_valid:
        error_msg = f"Model validation failed: {validation_message}"
        status_log.append(f"Error: {error_msg}")
        if logger:
            logger.error(error_msg)
        yield None, "\n".join(status_log), None, error_msg, None
        return
    
    try:
        # Get model information
        model_info = get_model_info(model_path, logger)
        if "error" in model_info:
            error_msg = f"Failed to get model info: {model_info['error']}"
            status_log.append(f"Error: {error_msg}")
            yield None, "\n".join(status_log), None, error_msg, None
            return
        
        # Determine device first
        device = "cuda" if util_get_gpu_device and util_get_gpu_device(logger=logger) != "cpu" else "cpu"
        if logger:
            logger.info(f"Using device: {device}")
        
        # Update metadata with image upscaler info
        params_for_metadata.update({
            "upscaler_type": "image_upscaler",
            "image_upscaler_enabled": True,
            "image_upscaler_model": actual_model_filename,
            "image_upscaler_model_display": selected_model_filename,
            "image_upscaler_architecture": model_info.get("architecture_name", "Unknown"),
            "image_upscaler_scale_factor": model_info.get("scale", "Unknown"),
            "image_upscaler_batch_size": batch_size,
            "image_upscaler_device": device,
            "seed": current_seed
        })
        
        # Log model information
        model_scale = model_info.get("scale", "Unknown")
        model_arch = model_info.get("architecture_name", "Unknown")
        init_msg = f"Initializing image upscaler: {actual_model_filename} (Scale: {model_scale}x, Architecture: {model_arch})"
        status_log.append(init_msg)
        if logger:
            logger.info(init_msg)
        
        yield None, "\n".join(status_log), None, "Loading image upscaler model...", None
        
        # Load the model
        model_load_start = time.time()
        model = load_model(model_path, device=device, logger=logger)
        
        if model is None:
            error_msg = f"Failed to load model: {actual_model_filename}"
            status_log.append(f"Error: {error_msg}")
            yield None, "\n".join(status_log), None, error_msg, None
            return
        
        model_load_time = time.time() - model_load_start
        load_msg = f"Image upscaler loaded in {format_time(model_load_time) if format_time else f'{model_load_time:.2f}s'}"
        status_log.append(load_msg)
        if logger:
            logger.info(load_msg)
        
        # Get actual model scale for resolution calculation
        actual_scale = get_model_scale_factor(model)
        params_for_metadata["actual_model_scale"] = actual_scale
        
        yield None, "\n".join(status_log), None, "Model loaded, processing frames...", None
        
        # Process based on scene splitting
        if enable_scene_split and scene_video_paths:
            # Process each scene separately
            total_scenes = len(scene_video_paths)
            processed_scenes = []
            
            for scene_idx, scene_video_path in enumerate(scene_video_paths):
                scene_start_time = time.time()
                scene_name = f"scene_{scene_idx + 1:04d}"
                
                scene_msg = f"Processing scene {scene_idx + 1}/{total_scenes}: {scene_name}"
                status_log.append(scene_msg)
                if logger:
                    logger.info(scene_msg)
                
                yield None, "\n".join(status_log), None, f"Processing {scene_name}...", None
                
                # Process single scene
                try:
                    scene_output_path = None
                    for result_tuple in process_single_scene_image_upscaler(
                        scene_video_path=scene_video_path,
                        scene_index=scene_idx,
                        total_scenes=total_scenes,
                        model=model,
                        batch_size=batch_size,
                        device=device,
                        temp_dir=temp_dir,
                        output_dir=output_dir,
                        base_output_filename_no_ext=base_output_filename_no_ext,
                        save_frames=save_frames,
                        input_fps=input_fps,
                        ffmpeg_preset=ffmpeg_preset,
                        ffmpeg_quality_value=ffmpeg_quality_value,
                        ffmpeg_use_gpu=ffmpeg_use_gpu,
                        logger=logger,
                        util_extract_frames=util_extract_frames,
                        util_create_video_from_frames=util_create_video_from_frames,
                        format_time=format_time,
                        status_log=status_log
                    ):
                        # Forward intermediate yields and extract final output path
                        output_video_path_inner, status_message_inner, chunk_video_path_inner, chunk_status_inner, comparison_video_path_inner = result_tuple
                        
                        if output_video_path_inner is not None:
                            scene_output_path = output_video_path_inner
                            
                        # Forward the yield to caller
                        yield result_tuple
                    
                    # Handle scene completion
                    if scene_output_path:
                        processed_scenes.append(scene_output_path)
                        last_chunk_video_path = scene_output_path
                        
                        scene_duration = time.time() - scene_start_time
                        scene_complete_msg = f"Scene {scene_idx + 1} complete in {format_time(scene_duration) if format_time else f'{scene_duration:.2f}s'}"
                        status_log.append(scene_complete_msg)
                        last_chunk_status = f"Scene {scene_idx + 1}/{total_scenes} complete"
                        
                        yield None, "\n".join(status_log), last_chunk_video_path, last_chunk_status, None
                    
                except Exception as e:
                    error_msg = f"Error processing scene {scene_idx + 1}: {str(e)}"
                    status_log.append(f"Error: {error_msg}")
                    if logger:
                        logger.error(error_msg, exc_info=True)
                    yield None, "\n".join(status_log), None, error_msg, None
                    continue
            
            # Merge scenes if we have multiple successful scenes
            if len(processed_scenes) > 1:
                merge_msg = f"Merging {len(processed_scenes)} processed scenes..."
                status_log.append(merge_msg)
                yield None, "\n".join(status_log), None, "Merging scenes...", None
                
                # Use existing scene merging functionality
                # This would need to be implemented or imported from scene_utils
                # For now, we'll use the first scene as output
                output_video_path = processed_scenes[0]
                
            elif len(processed_scenes) == 1:
                output_video_path = processed_scenes[0]
            else:
                error_msg = "No scenes were processed successfully"
                status_log.append(f"Error: {error_msg}")
                yield None, "\n".join(status_log), None, error_msg, None
                return
        
        else:
            # Process entire video directly
            output_video_path = None
            for result_tuple in process_single_video_image_upscaler(
                input_video_path=input_video_path,
                model=model,
                batch_size=batch_size,
                device=device,
                temp_dir=temp_dir,
                output_dir=output_dir,
                base_output_filename_no_ext=base_output_filename_no_ext,
                save_frames=save_frames,
                input_fps=input_fps,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality_value=ffmpeg_quality_value,
                ffmpeg_use_gpu=ffmpeg_use_gpu,
                logger=logger,
                util_extract_frames=util_extract_frames,
                util_create_video_from_frames=util_create_video_from_frames,
                format_time=format_time,
                status_log=status_log,
                progress=progress
            ):
                # Forward intermediate yields and extract final output path
                output_video_path_inner, status_message_inner, chunk_video_path_inner, chunk_status_inner, comparison_video_path_inner = result_tuple
                
                if output_video_path_inner is not None:
                    output_video_path = output_video_path_inner
                    
                # Forward the yield to caller
                yield result_tuple
        
        # Save metadata if enabled
        if save_metadata and metadata_handler_module and output_video_path:
            metadata_msg = "Saving processing metadata..."
            status_log.append(metadata_msg)
            yield None, "\n".join(status_log), last_chunk_video_path, "Saving metadata...", None
            
            try:
                processing_time_total = time.time() - model_load_start
                
                # Update metadata with final processing results
                # Extract processing stats from status log if available
                processed_count = 0
                failed_count = 0
                for log_line in status_log:
                    if "Frame processing complete:" in log_line:
                        # Parse: "Frame processing complete: X processed, Y failed in Z"
                        try:
                            parts = log_line.split(":")
                            if len(parts) > 1:
                                stats_part = parts[1].strip()
                                if "processed" in stats_part and "failed" in stats_part:
                                    # Extract numbers
                                    import re
                                    numbers = re.findall(r'\d+', stats_part)
                                    if len(numbers) >= 2:
                                        processed_count = int(numbers[0])
                                        failed_count = int(numbers[1])
                        except:
                            pass  # Use defaults if parsing fails
                        break
                
                # Update metadata with processing results
                params_for_metadata.update({
                    "image_upscaler_processing_time": processing_time_total,
                    "image_upscaler_frames_processed": processed_count,
                    "image_upscaler_frames_failed": failed_count
                })
                
                final_status_info = {"processing_time_total": processing_time_total}
                
                success, message = metadata_handler_module.save_metadata(
                    save_flag=True,
                    output_dir=output_dir,
                    base_filename_no_ext=base_output_filename_no_ext,
                    params_dict=params_for_metadata,
                    status_info=final_status_info,
                    logger=logger
                )
                
                if success:
                    meta_msg = f"Metadata saved successfully: {message.split(': ')[-1] if ': ' in message else message}"
                    status_log.append(meta_msg)
                else:
                    status_log.append(f"Metadata save warning: {message}")
                
            except Exception as e:
                error_msg = f"Error saving metadata: {str(e)}"
                status_log.append(f"Warning: {error_msg}")
                if logger:
                    logger.warning(error_msg)
        
        # Final success message
        if output_video_path:
            success_msg = f"Image upscaling complete! Output: {os.path.basename(output_video_path)}"
            status_log.append(success_msg)
            if logger:
                logger.info(success_msg)
            
            last_chunk_status = "Processing complete!"
            yield output_video_path, "\n".join(status_log), last_chunk_video_path, last_chunk_status, comparison_video_path
        else:
            error_msg = "Image upscaling failed - no output video generated"
            status_log.append(f"Error: {error_msg}")
            yield None, "\n".join(status_log), None, error_msg, None
        
    except Exception as e:
        error_msg = f"Unexpected error in image upscaling: {str(e)}"
        status_log.append(f"Critical Error: {error_msg}")
        if logger:
            logger.error(error_msg, exc_info=True)
        yield None, "\n".join(status_log), None, error_msg, None
    
    finally:
        # Clean up model from memory
        if 'model' in locals() and model is not None:
            try:
                unload_model(model_path, device=device, logger=logger)
                if logger:
                    logger.info("Image upscaler model unloaded from memory")
            except Exception as e:
                if logger:
                    logger.warning(f"Error unloading model: {e}")

def process_single_video_image_upscaler(
    input_video_path: str,
    model: Any,
    batch_size: int,
    device: str,
    temp_dir: str,
    output_dir: str,
    base_output_filename_no_ext: str,
    save_frames: bool,
    input_fps: float,
    ffmpeg_preset: str,
    ffmpeg_quality_value: int,
    ffmpeg_use_gpu: bool,
    logger: logging.Logger,
    util_extract_frames,
    util_create_video_from_frames,
    format_time,
    status_log: List[str],
    progress = None
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """Process a single video with image upscaler."""
    
    # Create temporary directories
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # NEW: Set up permanent frame save directories *before* processing so that
    # processed frames can be written there in real-time.
    permanent_frames_subdir = None
    permanent_processed_frames_dir = None
    permanent_input_frames_dir = None

    if save_frames and base_output_filename_no_ext:
        permanent_frames_subdir = os.path.join(output_dir, base_output_filename_no_ext)
        permanent_processed_frames_dir = os.path.join(permanent_frames_subdir, "processed_frames")
        permanent_input_frames_dir = os.path.join(permanent_frames_subdir, "input_frames")

        # Ensure the directories exist now so that individual frames can be
        # copied during processing without race conditions.
        os.makedirs(permanent_processed_frames_dir, exist_ok=True)
        os.makedirs(permanent_input_frames_dir, exist_ok=True)
    
    try:
        # Extract frames
        extract_start = time.time()
        extract_msg = "Extracting frames from video..."
        status_log.append(extract_msg)
        yield None, "\n".join(status_log), None, "Extracting frames...", None
        
        frame_count, actual_fps, frame_files = util_extract_frames(
            input_video_path, input_frames_dir, logger=logger
        )
        
        # Immediately copy input frames to their final location if requested so
        # the user can inspect them while the upscale runs.
        if save_frames and permanent_input_frames_dir:
            try:
                shutil.copytree(input_frames_dir, permanent_input_frames_dir, dirs_exist_ok=True)
            except Exception as copy_in_e:
                if logger:
                    logger.warning(f"Could not copy input frames early: {copy_in_e}")
        
        if actual_fps:
            input_fps = actual_fps
        
        extract_time = time.time() - extract_start
        extract_complete_msg = f"Extracted {frame_count} frames at {input_fps:.2f} FPS in {format_time(extract_time) if format_time else f'{extract_time:.2f}s'}"
        status_log.append(extract_complete_msg)
        if logger:
            logger.info(extract_complete_msg)
        
        yield None, "\n".join(status_log), None, f"Extracted {frame_count} frames", None
        
        # Process frames with image upscaler
        process_start = time.time()
        process_msg = f"Processing {frame_count} frames with image upscaler (batch size: {batch_size})..."
        status_log.append(process_msg)
        
        def progress_callback(progress_val, desc):
            if progress:
                progress(progress_val, desc=desc)
        
        yield None, "\n".join(status_log), None, "Processing frames...", None
        
        # Process frames in batches
        processed_count, failed_count = process_frames_batch(
            frame_files=frame_files,
            input_dir=input_frames_dir,
            output_dir=output_frames_dir,
            model=model,
            batch_size=batch_size,
            device=device,
            progress_callback=progress_callback,
            secondary_output_dir=permanent_processed_frames_dir,
            logger=logger
        )
        
        process_time = time.time() - process_start
        process_complete_msg = f"Frame processing complete: {processed_count} processed, {failed_count} failed in {format_time(process_time) if format_time else f'{process_time:.2f}s'}"
        status_log.append(process_complete_msg)
        if logger:
            logger.info(process_complete_msg)
        
        if processed_count == 0:
            raise Exception("No frames were processed successfully")
        
        yield None, "\n".join(status_log), None, f"Processed {processed_count} frames", None
        
        # Create output video
        video_start = time.time()
        video_msg = "Creating output video from processed frames..."
        status_log.append(video_msg)
        
        # Generate output path
        output_video_path = os.path.join(output_dir, f"{base_output_filename_no_ext}.mp4")
        
        yield None, "\n".join(status_log), None, "Creating video...", None
        
        # Create video from frames
        util_create_video_from_frames(
            output_frames_dir,
            output_video_path,
            input_fps,
            ffmpeg_preset,
            ffmpeg_quality_value,
            ffmpeg_use_gpu,
            logger=logger
        )
        
        video_time = time.time() - video_start
        video_complete_msg = f"Output video created in {format_time(video_time) if format_time else f'{video_time:.2f}s'}"
        status_log.append(video_complete_msg)
        if logger:
            logger.info(video_complete_msg)
        
        # Handle frame saving – final sync/copy for any remaining files that
        # might not have been flushed yet or if save_frames was disabled.
        if save_frames and permanent_frames_subdir:
            frame_save_msg = "Synchronising processed frames to output directory..."
            status_log.append(frame_save_msg)

            try:
                # Ensure any frames still only in the temp directory are copied
                # over (e.g. if secondary save failed for some reason).
                shutil.copytree(output_frames_dir, permanent_processed_frames_dir, dirs_exist_ok=True)

                frame_save_complete_msg = f"Frames saved to: {permanent_frames_subdir}"
                status_log.append(frame_save_complete_msg)
            except Exception as final_copy_e:
                warn_msg = f"Warning: could not final-sync frames: {final_copy_e}"
                status_log.append(warn_msg)
                if logger:
                    logger.warning(warn_msg)
        
        # Final yield with the output video path
        yield output_video_path, "\n".join(status_log), output_video_path, "Video processing complete", None
        
    except Exception as e:
        error_msg = f"Error in single video processing: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        yield None, "\n".join(status_log), None, f"Error: {error_msg}", None
        raise

def process_single_scene_image_upscaler(
    scene_video_path: str,
    scene_index: int,
    total_scenes: int,
    model: Any,
    batch_size: int,
    device: str,
    temp_dir: str,
    output_dir: str,
    base_output_filename_no_ext: str,
    save_frames: bool,
    input_fps: float,
    ffmpeg_preset: str,
    ffmpeg_quality_value: int,
    ffmpeg_use_gpu: bool,
    logger: logging.Logger,
    util_extract_frames,
    util_create_video_from_frames,
    format_time,
    status_log: List[str]
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """Process a single scene with image upscaler."""
    
    scene_name = f"scene_{scene_index + 1:04d}"
    scene_temp_dir = os.path.join(temp_dir, scene_name)
    scene_input_frames_dir = os.path.join(scene_temp_dir, "input_frames")
    scene_output_frames_dir = os.path.join(scene_temp_dir, "output_frames")
    
    os.makedirs(scene_temp_dir, exist_ok=True)
    os.makedirs(scene_input_frames_dir, exist_ok=True)
    os.makedirs(scene_output_frames_dir, exist_ok=True)
    
    # NEW: Prepare permanent scene frame dirs upfront if requested so frames are
    # saved immediately.
    permanent_scene_frames_dir = None
    permanent_scene_processed_dir = None
    permanent_scene_input_dir = None

    if save_frames and base_output_filename_no_ext:
        permanent_scene_frames_dir = os.path.join(output_dir, base_output_filename_no_ext, "scenes", scene_name)
        permanent_scene_processed_dir = os.path.join(permanent_scene_frames_dir, "processed_frames")
        permanent_scene_input_dir = os.path.join(permanent_scene_frames_dir, "input_frames")

        os.makedirs(permanent_scene_processed_dir, exist_ok=True)
        os.makedirs(permanent_scene_input_dir, exist_ok=True)
    
    try:
        # Extract frames from scene
        frame_count, scene_fps, frame_files = util_extract_frames(
            scene_video_path, scene_input_frames_dir, logger=logger
        )
        
        # Immediately copy input frames for the scene if requested
        if save_frames and permanent_scene_input_dir:
            try:
                shutil.copytree(scene_input_frames_dir, permanent_scene_input_dir, dirs_exist_ok=True)
            except Exception as copy_scene_in_e:
                if logger:
                    logger.warning(f"Could not copy scene input frames early: {copy_scene_in_e}")
        
        if scene_fps:
            input_fps = scene_fps
        
        scene_extract_msg = f"Scene {scene_index + 1}: Extracted {frame_count} frames"
        status_log.append(scene_extract_msg)
        if logger:
            logger.info(scene_extract_msg)
        
        yield None, "\n".join(status_log), None, f"Extracting frames from {scene_name}...", None
        
        # Process frames
        processed_count, failed_count = process_frames_batch(
            frame_files=frame_files,
            input_dir=scene_input_frames_dir,
            output_dir=scene_output_frames_dir,
            model=model,
            batch_size=batch_size,
            device=device,
            progress_callback=None,
            secondary_output_dir=permanent_scene_processed_dir,
            logger=logger
        )
        
        if processed_count == 0:
            raise Exception(f"No frames processed for scene {scene_index + 1}")
        
        yield None, "\n".join(status_log), None, f"Processing {scene_name} frames...", None
        
        # Create scene output video
        scene_output_path = os.path.join(temp_dir, f"{scene_name}_upscaled.mp4")
        
        util_create_video_from_frames(
            scene_output_frames_dir,
            scene_output_path,
            input_fps,
            ffmpeg_preset,
            ffmpeg_quality_value,
            ffmpeg_use_gpu,
            logger=logger
        )
        
        # Final sync for any frames still only in temp dir
        if save_frames and permanent_scene_frames_dir:
            try:
                shutil.copytree(scene_output_frames_dir, permanent_scene_processed_dir, dirs_exist_ok=True)
            except Exception as final_scene_copy_e:
                if logger:
                    logger.warning(f"Could not final-sync scene frames: {final_scene_copy_e}")
        
        # Final yield with the scene output path
        yield scene_output_path, "\n".join(status_log), scene_output_path, f"Scene {scene_index + 1} complete", None
        
    except Exception as e:
        error_msg = f"Error processing scene {scene_index + 1}: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        yield None, "\n".join(status_log), None, error_msg, None
        raise

def get_available_image_upscaler_models(upscale_models_dir: str, logger: logging.Logger = None) -> List[Dict[str, Any]]:
    """
    Get list of available image upscaler models with their information.
    
    Args:
        upscale_models_dir: Directory containing upscaler models
        logger: Logger instance
        
    Returns:
        List of dictionaries containing model information
    """
    models = []
    model_files = scan_for_models(upscale_models_dir, logger)
    
    for model_file in model_files:
        model_path = os.path.join(upscale_models_dir, model_file)
        model_info = get_model_info(model_path, logger)
        
        model_data = {
            "filename": model_file,
            "path": model_path,
            "scale": model_info.get("scale", "Unknown"),
            "architecture": model_info.get("architecture_name", "Unknown"),
            "error": model_info.get("error", None)
        }
        
        models.append(model_data)
    
    return models 