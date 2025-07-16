"""
Preview utilities for upscaling the first frame of videos.
Provides functionality to quickly test upscaler models on single frames.
"""

import os
import time
import tempfile
import cv2
import numpy as np
import shutil
import logging
from typing import List, Dict, Tuple, Optional, Any
import subprocess

from .image_upscaler_utils import (
    scan_for_models, load_model, get_model_info,
    prepare_frame_tensor, tensor_to_frame,
    extract_model_filename_from_dropdown
)
from .upscaling_utils import calculate_upscale_params
from .file_utils import get_next_filename, sanitize_filename
from .ffmpeg_utils import natural_sort_key


def extract_first_frame(video_path: str, output_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Extract the first frame from a video file.
    
    Args:
        video_path: Path to the input video
        output_path: Path where the first frame should be saved
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(video_path):
            if logger:
                logger.error(f"Video file not found: {video_path}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use FFmpeg to extract the first frame
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'select=eq(n\\,0)',  # Select first frame (frame number 0)
            '-frames:v', '1',  # Extract only 1 frame
            '-y',  # Overwrite output file
            output_path
        ]
        
        if logger:
            logger.info(f"Extracting first frame from {os.path.basename(video_path)}")
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if process.returncode != 0:
            error_msg = f"FFmpeg failed to extract first frame: {process.stderr}"
            if logger:
                logger.error(error_msg)
            return False
        
        # Verify the frame was extracted
        if not os.path.exists(output_path):
            if logger:
                logger.error(f"First frame extraction failed - output file not created: {output_path}")
            return False
        
        if logger:
            logger.info(f"Successfully extracted first frame to: {output_path}")
        
        return True
        
    except subprocess.TimeoutExpired:
        error_msg = "Frame extraction timed out"
        if logger:
            logger.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"Error extracting first frame: {e}"
        if logger:
            logger.error(error_msg)
        return False


def upscale_single_frame(
    frame_path: str,
    model_path: str,
    output_path: str,
    device: str = "cuda",
    apply_resolution_constraints: bool = True,
    target_h: int = 1080,
    target_w: int = 1920,
    target_res_mode: str = "Downscale then Upscale",
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Upscale a single frame using an image upscaler model.
    
    Args:
        frame_path: Path to the input frame
        model_path: Path to the upscaler model
        output_path: Path where upscaled frame should be saved
        device: Device to use for processing
        apply_resolution_constraints: Whether to apply resolution constraints
        target_h: Target height constraint
        target_w: Target width constraint  
        target_res_mode: Resolution constraint mode
        logger: Logger instance
        
    Returns:
        Dictionary with result information
    """
    result = {
        'success': False,
        'output_path': None,
        'model_info': {},
        'original_resolution': None,
        'output_resolution': None,
        'processing_time': 0.0,
        'message': '',
        'error': None
    }
    
    start_time = time.time()
    
    try:
        if not os.path.exists(frame_path):
            result['error'] = f"Frame file not found: {frame_path}"
            return result
        
        if not os.path.exists(model_path):
            result['error'] = f"Model file not found: {model_path}"
            return result
        
        # Load model
        model = load_model(model_path, device=device, logger=logger)
        if model is None:
            result['error'] = f"Failed to load model: {os.path.basename(model_path)}"
            return result
        
        # Get model info
        model_info = get_model_info(model_path, logger)
        result['model_info'] = model_info
        
        model_scale = model_info.get("scale", 4)
        
        # Load frame
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            result['error'] = f"Could not read frame: {frame_path}"
            return result
        
        orig_h, orig_w = frame_bgr.shape[:2]
        result['original_resolution'] = (orig_w, orig_h)
        
        if logger:
            logger.info(f"Processing frame {orig_w}x{orig_h} with {os.path.basename(model_path)} ({model_scale}x)")
        
        # Apply resolution constraints if enabled
        input_frame = frame_bgr
        if apply_resolution_constraints:
            try:
                # Calculate optimal resolution with constraints
                upscale_params = calculate_upscale_params(
                    orig_h, orig_w, target_h, target_w, target_res_mode,
                    logger=logger, image_upscaler_model=os.path.basename(model_path)
                )
                
                calculated_h = upscale_params['final_h']
                calculated_w = upscale_params['final_w']
                downscale_h = upscale_params.get('downscale_h')
                downscale_w = upscale_params.get('downscale_w')
                
                # Apply downscaling if needed
                if downscale_h and downscale_w and (downscale_h != orig_h or downscale_w != orig_w):
                    if logger:
                        logger.info(f"Downscaling frame from {orig_w}x{orig_h} to {downscale_w}x{downscale_h} before {model_scale}x upscaling")
                    input_frame = cv2.resize(frame_bgr, (downscale_w, downscale_h), interpolation=cv2.INTER_AREA)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Could not apply resolution constraints: {e}, proceeding without constraints")
        
        # Prepare frame tensor
        frame_tensor = prepare_frame_tensor(input_frame, device)
        
        # Upscale frame
        import torch
        with torch.no_grad():
            upscaled_tensor = model(frame_tensor.unsqueeze(0))  # Add batch dimension
        
        # Convert back to frame
        upscaled_frame = tensor_to_frame(upscaled_tensor.squeeze(0))  # Remove batch dimension
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save upscaled frame
        success = cv2.imwrite(output_path, upscaled_frame)
        if not success:
            result['error'] = f"Failed to save upscaled frame to: {output_path}"
            return result
        
        result['success'] = True
        result['output_path'] = output_path
        result['output_resolution'] = (upscaled_frame.shape[1], upscaled_frame.shape[0])
        result['processing_time'] = time.time() - start_time
        result['message'] = f"Successfully upscaled frame with {os.path.basename(model_path)}"
        
        if logger:
            logger.info(f"Frame upscaled from {result['original_resolution']} to {result['output_resolution']} in {result['processing_time']:.2f}s")
        
        return result
        
    except Exception as e:
        result['error'] = f"Error during frame upscaling: {e}"
        result['processing_time'] = time.time() - start_time
        if logger:
            logger.error(f"Frame upscaling failed: {e}")
        return result


def preview_single_model(
    video_path: str,
    model_name: str,
    upscale_models_dir: str,
    temp_dir: str,
    device: str = "cuda",
    apply_resolution_constraints: bool = True,
    target_h: int = 1080,
    target_w: int = 1920,
    target_res_mode: str = "Downscale then Upscale",
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Preview upscaling with a single model on the first frame of a video.
    
    Args:
        video_path: Path to the input video
        model_name: Name of the upscaler model to use
        upscale_models_dir: Directory containing upscaler models
        temp_dir: Temporary directory for processing
        device: Device to use for processing
        apply_resolution_constraints: Whether to apply resolution constraints
        target_h: Target height constraint
        target_w: Target width constraint
        target_res_mode: Resolution constraint mode
        logger: Logger instance
        
    Returns:
        Dictionary with preview result
    """
    result = {
        'success': False,
        'preview_image_path': None,
        'original_image_path': None,
        'model_info': {},
        'original_resolution': None,
        'output_resolution': None,
        'processing_time': 0.0,
        'message': '',
        'error': None
    }
    
    start_time = time.time()
    
    try:
        if not video_path or not os.path.exists(video_path):
            result['error'] = "Please upload a video first"
            return result
        
        # Extract actual model filename
        actual_model_filename = extract_model_filename_from_dropdown(model_name)
        if not actual_model_filename:
            result['error'] = f"Invalid model selection: {model_name}"
            return result
        
        model_path = os.path.join(upscale_models_dir, actual_model_filename)
        if not os.path.exists(model_path):
            result['error'] = f"Model file not found: {actual_model_filename}"
            return result
        
        # Create temp directory for processing
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract first frame
        first_frame_path = os.path.join(temp_dir, "first_frame.png")
        if not extract_first_frame(video_path, first_frame_path, logger):
            result['error'] = "Failed to extract first frame from video"
            return result
        
        # Upscale the frame
        preview_output_path = os.path.join(temp_dir, f"preview_{sanitize_filename(actual_model_filename)}.png")
        upscale_result = upscale_single_frame(
            first_frame_path, model_path, preview_output_path, device,
            apply_resolution_constraints, target_h, target_w, target_res_mode, logger
        )
        
        if not upscale_result['success']:
            result['error'] = upscale_result['error']
            return result
        
        result['success'] = True
        result['preview_image_path'] = preview_output_path
        result['original_image_path'] = first_frame_path
        result['model_info'] = upscale_result['model_info']
        result['original_resolution'] = upscale_result['original_resolution']
        result['output_resolution'] = upscale_result['output_resolution']
        result['processing_time'] = time.time() - start_time
        result['message'] = f"Preview generated with {actual_model_filename}"
        
        return result
        
    except Exception as e:
        result['error'] = f"Preview generation failed: {e}"
        result['processing_time'] = time.time() - start_time
        if logger:
            logger.error(f"Preview generation error: {e}")
        return result


def preview_all_models(
    video_path: str,
    upscale_models_dir: str,
    output_dir: str,
    device: str = "cuda",
    apply_resolution_constraints: bool = True,
    target_h: int = 1080,
    target_w: int = 1920,
    target_res_mode: str = "Downscale then Upscale",
    logger: Optional[logging.Logger] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Generate upscale previews with all available models and save them to outputs folder.
    
    Args:
        video_path: Path to the input video
        upscale_models_dir: Directory containing upscaler models
        output_dir: Output directory for saving test results
        device: Device to use for processing
        apply_resolution_constraints: Whether to apply resolution constraints
        target_h: Target height constraint
        target_w: Target width constraint
        target_res_mode: Resolution constraint mode
        logger: Logger instance
        progress_callback: Optional progress callback function
        
    Returns:
        Dictionary with results for all models
    """
    result = {
        'success': False,
        'output_folder': None,
        'processed_models': [],
        'failed_models': [],
        'total_models': 0,
        'processing_time': 0.0,
        'message': '',
        'error': None
    }
    
    start_time = time.time()
    
    try:
        if not video_path or not os.path.exists(video_path):
            result['error'] = "Please upload a video first"
            return result
        
        # Scan for available models
        available_models = scan_for_models(upscale_models_dir, logger)
        if not available_models:
            result['error'] = "No upscaler models found in upscale_models directory"
            return result
        
        result['total_models'] = len(available_models)
        
        if logger:
            logger.info(f"Found {len(available_models)} upscaler models for comparison")
        
        # Create output folder with incremental naming
        base_folder_name = "upscale_models_test"
        output_folder = get_next_filename(output_dir, base_folder_name, "", is_folder=True)
        os.makedirs(output_folder, exist_ok=True)
        result['output_folder'] = output_folder
        
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp(prefix="preview_all_")
        
        try:
            # Extract first frame
            first_frame_path = os.path.join(temp_dir, "first_frame.png")
            if not extract_first_frame(video_path, first_frame_path, logger):
                result['error'] = "Failed to extract first frame from video"
                return result
            
            # Save original frame to output folder
            original_frame_output = os.path.join(output_folder, "00_original.png")
            shutil.copy2(first_frame_path, original_frame_output)
            
            # Process each model
            for i, model_filename in enumerate(available_models):
                try:
                    if progress_callback:
                        progress = (i + 1) / len(available_models)
                        progress_callback(progress, f"Processing model {i+1}/{len(available_models)}: {model_filename}")
                    
                    if logger:
                        logger.info(f"Processing model {i+1}/{len(available_models)}: {model_filename}")
                    
                    model_path = os.path.join(upscale_models_dir, model_filename)
                    
                    # Create output filename using model name
                    model_base_name = os.path.splitext(model_filename)[0]
                    sanitized_name = sanitize_filename(model_base_name)
                    output_filename = f"{i+1:02d}_{sanitized_name}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Upscale frame
                    upscale_result = upscale_single_frame(
                        first_frame_path, model_path, output_path, device,
                        apply_resolution_constraints, target_h, target_w, target_res_mode, logger
                    )
                    
                    if upscale_result['success']:
                        result['processed_models'].append({
                            'model_name': model_filename,
                            'output_path': output_path,
                            'model_info': upscale_result['model_info'],
                            'original_resolution': upscale_result['original_resolution'],
                            'output_resolution': upscale_result['output_resolution'],
                            'processing_time': upscale_result['processing_time']
                        })
                        if logger:
                            logger.info(f"✅ Successfully processed {model_filename}")
                    else:
                        result['failed_models'].append({
                            'model_name': model_filename,
                            'error': upscale_result['error']
                        })
                        if logger:
                            logger.warning(f"❌ Failed to process {model_filename}: {upscale_result['error']}")
                
                except Exception as model_error:
                    result['failed_models'].append({
                        'model_name': model_filename,
                        'error': str(model_error)
                    })
                    if logger:
                        logger.error(f"❌ Error processing {model_filename}: {model_error}")
        
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                if logger:
                    logger.warning(f"Could not clean up temp directory: {cleanup_error}")
        
        # Generate summary
        successful_count = len(result['processed_models'])
        failed_count = len(result['failed_models'])
        
        result['success'] = successful_count > 0
        result['processing_time'] = time.time() - start_time
        
        if result['success']:
            result['message'] = f"Generated comparison with {successful_count}/{result['total_models']} models in {output_folder}"
            if failed_count > 0:
                result['message'] += f" ({failed_count} models failed)"
        else:
            result['error'] = f"All {result['total_models']} models failed to process"
        
        if logger:
            logger.info(f"Model comparison completed: {successful_count} succeeded, {failed_count} failed")
            logger.info(f"Results saved to: {output_folder}")
        
        return result
        
    except Exception as e:
        result['error'] = f"Comparison generation failed: {e}"
        result['processing_time'] = time.time() - start_time
        if logger:
            logger.error(f"Model comparison error: {e}")
        return result 