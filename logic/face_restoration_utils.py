import os
import sys
import cv2
import numpy as np
import tempfile
import shutil
import subprocess
import glob
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from .ffmpeg_utils import natural_sort_key

# Global variables to store CodeFormer components after import
codeformer_available = False
codeformer_modules = {}
codeformer_model_cache = {}

def scan_codeformer_models(pretrained_weight_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Scan the pretrained_weight directory for CodeFormer models.
    
    Args:
        pretrained_weight_dir: Path to the pretrained weights directory
        logger: Logger instance for logging
        
    Returns:
        Dictionary containing available models and their info
    """
    try:
        models_info = {
            'available': False,
            'codeformer_models': [],
            'error': None
        }
        
        if not os.path.exists(pretrained_weight_dir):
            logger.warning(f"Pretrained weights directory not found: {pretrained_weight_dir}")
            models_info['error'] = f"Directory not found: {pretrained_weight_dir}"
            return models_info
        
        # Look for CodeFormer model files
        codeformer_patterns = [
            "**/*codeformer*.pth",
            "**/*CodeFormer*.pth", 
            "**/CodeFormer/**/*.pth",
            "**/codeformer/**/*.pth"
        ]
        
        found_models = []
        for pattern in codeformer_patterns:
            model_files = glob.glob(os.path.join(pretrained_weight_dir, pattern), recursive=True)
            for model_file in model_files:
                if os.path.isfile(model_file):
                    model_info = {
                        'path': model_file,
                        'name': os.path.basename(model_file),
                        'size_mb': os.path.getsize(model_file) / (1024 * 1024),
                        'relative_path': os.path.relpath(model_file, pretrained_weight_dir)
                    }
                    found_models.append(model_info)
                    logger.info(f"Found CodeFormer model: {model_info['name']} ({model_info['size_mb']:.1f} MB)")
        
        # Remove duplicates based on file path
        unique_models = []
        seen_paths = set()
        for model in found_models:
            if model['path'] not in seen_paths:
                unique_models.append(model)
                seen_paths.add(model['path'])
        
        models_info['codeformer_models'] = unique_models
        models_info['available'] = len(unique_models) > 0
        
        if not models_info['available']:
            logger.warning("No CodeFormer models found in pretrained weights directory")
            models_info['error'] = "No CodeFormer models found"
        else:
            logger.info(f"Found {len(unique_models)} unique CodeFormer model(s)")
            
        return models_info
        
    except Exception as e:
        logger.error(f"Error scanning CodeFormer models: {e}", exc_info=True)
        return {
            'available': False,
            'codeformer_models': [],
            'error': str(e)
        }

def find_codeformer_model_path(model_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], bool]:
    """
    Find the CodeFormer model path, checking pretrained_weight directory first.
    
    Args:
        model_path: Optional specific model path provided by user
        logger: Logger instance
        
    Returns:
        Tuple of (model_path, needs_download) where:
        - model_path: Path to the model file or None if not found
        - needs_download: True if model needs to be downloaded to weights folder
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # If specific model path provided and exists, use it
    if model_path and os.path.exists(model_path):
        logger.info(f"Using provided CodeFormer model: {model_path}")
        return model_path, False
    
    # Try to find model in pretrained_weight directory first
    try:
        # Get the configured pretrained_weight directory
        from . import config
        pretrained_weight_dir = getattr(config, 'FACE_RESTORATION_MODELS_DIR', None)
        
        if pretrained_weight_dir and os.path.exists(pretrained_weight_dir):
            # Look for CodeFormer models in pretrained_weight directory
            potential_paths = [
                os.path.join(pretrained_weight_dir, 'codeformer.pth'),
                os.path.join(pretrained_weight_dir, 'CodeFormer', 'codeformer.pth'),
                os.path.join(pretrained_weight_dir, 'CodeFormer_STAR', 'codeformer.pth')
            ]
            
            for potential_path in potential_paths:
                if os.path.exists(potential_path):
                    logger.info(f"Found CodeFormer model in pretrained_weight directory: {potential_path}")
                    return potential_path, False
        
        # Also check if model already exists in weights folder
        weights_path = os.path.join('weights', 'CodeFormer', 'codeformer.pth')
        if os.path.exists(weights_path):
            logger.info(f"Found existing CodeFormer model in weights directory: {weights_path}")
            return weights_path, False
            
        # Model not found locally, will need to download
        logger.info("CodeFormer model not found locally, will download to weights/CodeFormer/")
        return None, True
        
    except Exception as e:
        logger.warning(f"Error checking for CodeFormer model: {e}")
        # Fall back to download
        return None, True

def setup_codeformer_environment(codeformer_base_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Setup the CodeFormer environment by adding paths and importing necessary modules.
    
    Args:
        codeformer_base_path: Path to the CodeFormer_STAR directory
        logger: Logger instance
        
    Returns:
        Dictionary containing setup results and imported modules
    """
    global codeformer_available, codeformer_modules
    
    setup_result = {
        'success': False,
        'modules': {},
        'error': None,
        'missing_dependencies': []
    }
    
    try:
        if not os.path.exists(codeformer_base_path):
            error_msg = f"CodeFormer base path not found: {codeformer_base_path}"
            logger.error(error_msg)
            setup_result['error'] = error_msg
            return setup_result
        
        # Add CodeFormer paths to sys.path if not already present
        paths_to_add = [
            codeformer_base_path,
            os.path.join(codeformer_base_path, 'basicsr'),
            os.path.join(codeformer_base_path, 'facelib')
        ]
        
        for path in paths_to_add:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
                logger.info(f"Added to sys.path: {path}")
        
        # Try to import required CodeFormer modules
        try:
            # Import face detection and alignment
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
            from facelib.detection import init_detection_model
            from facelib.parsing import init_parsing_model
            
            # Import basicsr components
            from basicsr.utils import imwrite, img2tensor, tensor2img
            from basicsr.models import build_model
            from basicsr.utils.options import dict2str
            
            # Store imported modules
            setup_result['modules'] = {
                'FaceRestoreHelper': FaceRestoreHelper,
                'init_detection_model': init_detection_model,
                'init_parsing_model': init_parsing_model,
                'imwrite': imwrite,
                'img2tensor': img2tensor,
                'tensor2img': tensor2img,
                'build_model': build_model,
                'dict2str': dict2str
            }
            
            codeformer_modules = setup_result['modules']
            logger.info("Successfully imported CodeFormer modules")
            
        except ImportError as e:
            error_msg = f"Failed to import CodeFormer modules: {e}"
            logger.error(error_msg)
            setup_result['error'] = error_msg
            setup_result['missing_dependencies'].append(str(e))
            return setup_result
        
        # Test basic functionality
        try:
            # Try to create a simple face restoration helper to verify everything works
            # This is a lightweight test without loading heavy models
            logger.info("CodeFormer environment setup completed successfully")
            setup_result['success'] = True
            codeformer_available = True
            
        except Exception as e:
            error_msg = f"CodeFormer environment test failed: {e}"
            logger.error(error_msg)
            setup_result['error'] = error_msg
            return setup_result
            
    except Exception as e:
        error_msg = f"Unexpected error during CodeFormer setup: {e}"
        logger.error(error_msg, exc_info=True)
        setup_result['error'] = error_msg
        
    return setup_result

def ensure_codeformer_available(logger: Optional[logging.Logger] = None) -> bool:
    """
    Ensure CodeFormer environment is available, initializing if needed.
    
    Args:
        logger: Logger instance for logging
        
    Returns:
        bool: True if CodeFormer is available, False otherwise
    """
    global codeformer_available
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # If already available, return True
    if codeformer_available:
        return True
    
    # Try to initialize CodeFormer environment
    try:
        # Find CodeFormer_STAR directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_codeformer_paths = [
            os.path.join(current_dir, "..", "..", "CodeFormer_STAR"),
            os.path.join(current_dir, "..", "CodeFormer_STAR"),
            os.path.join(current_dir, "CodeFormer_STAR"),
            os.path.join(os.path.dirname(current_dir), "CodeFormer_STAR"),
        ]
        
        codeformer_base_path = None
        for path in possible_codeformer_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                codeformer_base_path = abs_path
                break
        
        if not codeformer_base_path:
            logger.warning("CodeFormer_STAR directory not found. Face restoration will use subprocess approach.")
            # Set to True anyway since we can still use subprocess approach
            codeformer_available = True
            return True
        
        # Try to setup environment (but don't fail if imports fail)
        setup_result = setup_codeformer_environment(codeformer_base_path, logger)
        if setup_result['success']:
            logger.info("CodeFormer environment initialized successfully")
            codeformer_available = True
            return True
        else:
            logger.warning(f"CodeFormer environment setup failed: {setup_result.get('error', 'Unknown error')}. Using subprocess approach.")
            # Set to True anyway since we can still use subprocess approach
            codeformer_available = True
            return True
        
    except Exception as e:
        logger.warning(f"Failed to initialize CodeFormer environment: {e}. Using subprocess approach.")
        # Set to True anyway since we can still use subprocess approach
        codeformer_available = True
        return True

def restore_single_image(
    image_path: str,
    output_path: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    model_path: Optional[str] = None,
    suppress_individual_logging: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Restore a single image using CodeFormer.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image
        fidelity_weight: Fidelity weight (0.0-1.0), higher values preserve identity better
        enable_colorization: Whether to apply colorization for grayscale images
        model_path: Path to CodeFormer model (optional)
        suppress_individual_logging: If True, suppress individual frame completion logging (for batch processing)
        logger: Logger instance
        
    Returns:
        Dictionary containing restoration results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Ensure CodeFormer is available before proceeding
    if not ensure_codeformer_available(logger):
        return {
            'success': False,
            'output_path': None,
            'error': "CodeFormer environment not available",
            'faces_detected': 0,
            'processing_time': 0.0
        }
        
    result = {
        'success': False,
        'output_path': None,
        'error': None,
        'faces_detected': 0,
        'processing_time': 0.0
    }
    
    start_time = time.time()
    
    try:
        if not codeformer_available:
            result['error'] = "CodeFormer environment not available"
            return result
            
        if not os.path.exists(image_path):
            result['error'] = f"Input image not found: {image_path}"
            return result
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            result['error'] = f"Failed to load image: {image_path}"
            return result
            
        # Check if image is grayscale and colorization is requested
        is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]))
        
        if is_grayscale and enable_colorization:
            if not suppress_individual_logging:
                logger.info(f"Applying colorization to grayscale image: {os.path.basename(image_path)}")
            # Apply colorization first, then face restoration
            result = _apply_colorization(img, image_path, output_path, fidelity_weight, model_path, logger)
        else:
            # Apply face restoration
            result = _apply_face_restoration(img, image_path, output_path, fidelity_weight, model_path, logger)
            
        result['processing_time'] = time.time() - start_time
        
        if result['success']:
            if not suppress_individual_logging:
                logger.info(f"Face restoration completed: {os.path.basename(image_path)} -> {os.path.basename(output_path)} ({result['processing_time']:.2f}s)")
            else:
                logger.debug(f"Face restoration completed: {os.path.basename(image_path)} -> {os.path.basename(output_path)} ({result['processing_time']:.2f}s)")
        else:
            if not suppress_individual_logging:
                logger.warning(f"Face restoration failed for {os.path.basename(image_path)}: {result['error']}")
            else:
                logger.debug(f"Face restoration failed for {os.path.basename(image_path)}: {result['error']}")
            
    except Exception as e:
        result['error'] = f"Unexpected error during face restoration: {e}"
        if not suppress_individual_logging:
            logger.error(result['error'], exc_info=True)
        else:
            logger.debug(result['error'], exc_info=True)
        
    return result

def _apply_face_restoration(
    img: np.ndarray,
    image_path: str,
    output_path: str,
    fidelity_weight: float,
    model_path: Optional[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Apply face restoration to an image using CodeFormer.
    
    Args:
        img: Input image as numpy array
        image_path: Original image path (for logging)
        output_path: Output path for restored image  
        fidelity_weight: Fidelity weight for restoration
        model_path: Path to CodeFormer model
        logger: Logger instance
        
    Returns:
        Dictionary containing restoration results
    """
    result = {
        'success': False,
        'output_path': None,
        'error': None,
        'faces_detected': 0
    }
    
    try:
        # Use CodeFormer's command line interface for now
        # This is more reliable than trying to replicate the complex model loading
        codeformer_script = None
        
        # Find the inference_codeformer.py script
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "CodeFormer_STAR", "inference_codeformer.py"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", "CodeFormer_STAR", "inference_codeformer.py")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                codeformer_script = path
                break
                
        if not codeformer_script:
            result['error'] = "CodeFormer inference script not found"
            return result
            
        # Create temporary input file if needed
        temp_input = None
        if not image_path.endswith(('.jpg', '.jpeg', '.png')):
            temp_input = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(temp_input.name, img)
            input_path = temp_input.name
        else:
            input_path = image_path
            
        try:
            # Build command with correct arguments
            cmd = [
                sys.executable,
                codeformer_script,
                '--input_path', input_path,
                '--output_path', os.path.dirname(output_path),
                '--fidelity_weight', str(fidelity_weight),
                '--upscale', '1',  # Don't upscale, we just want face restoration
                '--detection_model', 'retinaface_resnet50'
            ]
                
            # Run CodeFormer
            logger.debug(f"Running CodeFormer command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode == 0:
                # CodeFormer saves final results in 'final_results' subfolder
                input_basename = os.path.splitext(os.path.basename(input_path))[0]
                expected_output = os.path.join(os.path.dirname(output_path), 'final_results', f"{input_basename}.png")
                
                if os.path.exists(expected_output):
                    # Move to desired output location
                    shutil.move(expected_output, output_path)
                    result['success'] = True
                    result['output_path'] = output_path
                    result['faces_detected'] = 1  # Assume at least one face if successful
                else:
                    # Look for any output files in the final_results directory
                    final_results_dir = os.path.join(os.path.dirname(output_path), 'final_results')
                    if os.path.exists(final_results_dir):
                        output_files = glob.glob(os.path.join(final_results_dir, "*.png")) + glob.glob(os.path.join(final_results_dir, "*.jpg"))
                        
                        if output_files:
                            # Use the most recent file
                            latest_file = max(output_files, key=os.path.getctime)
                            shutil.move(latest_file, output_path)
                            result['success'] = True
                            result['output_path'] = output_path
                            result['faces_detected'] = 1
                            
                            # Clean up CodeFormer temporary folders
                            try:
                                output_base_dir = os.path.dirname(output_path)
                                cleanup_dirs = ['final_results', 'cropped_faces', 'restored_faces']
                                for cleanup_dir in cleanup_dirs:
                                    cleanup_path = os.path.join(output_base_dir, cleanup_dir)
                                    if os.path.exists(cleanup_path) and not os.listdir(cleanup_path):
                                        os.rmdir(cleanup_path)
                            except:
                                pass
                        else:
                            result['error'] = "CodeFormer completed but no output file found in final_results"
                    else:
                        result['error'] = "CodeFormer completed but final_results directory not found"
            else:
                result['error'] = f"CodeFormer failed: {process.stderr}"
                
        finally:
            # Clean up temporary file
            if temp_input:
                try:
                    os.unlink(temp_input.name)
                except:
                    pass
                    
    except subprocess.TimeoutExpired:
        result['error'] = "CodeFormer timed out"
    except Exception as e:
        result['error'] = f"Error running CodeFormer: {e}"
        
    return result

def _apply_colorization(
    img: np.ndarray,
    image_path: str,
    output_path: str,
    fidelity_weight: float,
    model_path: Optional[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Apply colorization to a grayscale image using CodeFormer.
    
    Args:
        img: Input grayscale image as numpy array
        image_path: Original image path
        output_path: Output path for colorized image
        fidelity_weight: Fidelity weight (passed to subsequent face restoration)
        model_path: Path to CodeFormer model
        logger: Logger instance
        
    Returns:
        Dictionary containing colorization results
    """
    result = {
        'success': False,
        'output_path': None,
        'error': None,
        'faces_detected': 0
    }
    
    try:
        # Find the colorization script
        colorization_script = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "CodeFormer_STAR", "inference_colorization.py"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", "CodeFormer_STAR", "inference_colorization.py")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                colorization_script = path
                break
                
        if not colorization_script:
            result['error'] = "CodeFormer colorization script not found"
            return result
            
        # Create temporary files
        temp_input = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_output = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        
        try:
            # Save grayscale image
            cv2.imwrite(temp_input.name, img)
            
            # Run colorization
            cmd = [
                sys.executable,
                colorization_script,
                '--input_path', temp_input.name,
                '--output_path', os.path.dirname(temp_output.name)
            ]
            
            logger.debug(f"Running colorization command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode == 0:
                # Find colorized output
                colorized_files = glob.glob(os.path.join(os.path.dirname(temp_output.name), "colorized_*.png"))
                
                if colorized_files:
                    colorized_img = cv2.imread(colorized_files[0])
                    if colorized_img is not None:
                        # Now apply face restoration to the colorized image
                        restoration_result = _apply_face_restoration(
                            colorized_img, colorized_files[0], output_path, 
                            fidelity_weight, model_path, logger
                        )
                        result.update(restoration_result)
                    else:
                        result['error'] = "Failed to load colorized image"
                else:
                    result['error'] = "Colorization completed but no output found"
            else:
                result['error'] = f"Colorization failed: {process.stderr}"
                
        finally:
            # Clean up temporary files
            for temp_file in [temp_input.name, temp_output.name]:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except subprocess.TimeoutExpired:
        result['error'] = "Colorization timed out"
    except Exception as e:
        result['error'] = f"Error during colorization: {e}"
        
    return result

def restore_frames_batch(
    frame_paths: List[str],
    output_dir: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    model_path: Optional[str] = None,
    batch_size: int = 4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Batch process multiple frames for face restoration.
    
    Args:
        frame_paths: List of paths to input frames
        output_dir: Directory for output frames
        fidelity_weight: Fidelity weight for restoration
        enable_colorization: Whether to apply colorization
        model_path: Path to CodeFormer model
        batch_size: Number of frames to process simultaneously
        progress_callback: Optional callback for progress updates (current, total, status)
        logger: Logger instance
        
    Returns:
        Dictionary containing batch processing results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    result = {
        'success': False,
        'processed_count': 0,
        'failed_count': 0,
        'total_count': len(frame_paths),
        'output_paths': [],
        'errors': [],
        'total_processing_time': 0.0,
        'faces_detected_total': 0
    }
    
    start_time = time.time()
    
    try:
        # Ensure CodeFormer is available before proceeding
        if not ensure_codeformer_available(logger):
            result['errors'].append("CodeFormer environment not available")
            return result
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Process frames in batches
        total_frames = len(frame_paths)
        processed_frames = 0
        
        logger.info(f"Processing {total_frames} frames for face restoration with batch size: {batch_size}")
        
        # Split frames into batches
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            current_batch = frame_paths[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_frames + batch_size - 1) // batch_size
            
            if progress_callback:
                progress_callback(processed_frames, total_frames, f"Processing batch {batch_num}/{total_batches} ({len(current_batch)} frames)")
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: frames {batch_start+1}-{batch_end} ({len(current_batch)} frames)")
            batch_start_time = time.time()
            
            # Process all frames in current batch
            batch_success_count = 0
            batch_failed_count = 0
            
            for i, frame_path in enumerate(current_batch):
                try:
                    # Generate output path
                    frame_name = os.path.basename(frame_path)
                    name_without_ext = os.path.splitext(frame_name)[0]
                    output_path = os.path.join(output_dir, f"{name_without_ext}_restored.png")
                    
                    # Restore single frame (suppress individual logging during batch processing)
                    frame_result = restore_single_image(
                        frame_path, output_path, fidelity_weight, 
                        enable_colorization, model_path, True, logger
                    )
                    
                    if frame_result['success']:
                        result['processed_count'] += 1
                        batch_success_count += 1
                        result['output_paths'].append(frame_result['output_path'])
                        result['faces_detected_total'] += frame_result['faces_detected']
                        logger.debug(f"Successfully processed frame {processed_frames + i + 1}/{total_frames}: {frame_name}")
                    else:
                        result['failed_count'] += 1
                        batch_failed_count += 1
                        error_msg = f"Frame {frame_name}: {frame_result['error']}"
                        result['errors'].append(error_msg)
                        logger.warning(error_msg)
                        
                except Exception as e:
                    result['failed_count'] += 1
                    batch_failed_count += 1
                    error_msg = f"Unexpected error processing frame {frame_path}: {e}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            
            batch_time = time.time() - batch_start_time
            processed_frames += len(current_batch)
            
            # Calculate average time per frame in this batch
            avg_time_per_frame = batch_time / len(current_batch) if len(current_batch) > 0 else 0
            logger.info(f"Batch {batch_num}/{total_batches} completed: {batch_success_count} successful, {batch_failed_count} failed in {batch_time:.2f}s (avg {avg_time_per_frame:.2f}s/frame)")
            
        result['total_processing_time'] = time.time() - start_time
        result['success'] = result['processed_count'] > 0
        
        if progress_callback:
            progress_callback(total_frames, total_frames, "Batch processing completed")
            
        logger.info(f"Batch face restoration completed: {result['processed_count']}/{result['total_count']} frames processed successfully in {result['total_processing_time']:.1f}s")
        
    except Exception as e:
        error_msg = f"Unexpected error during batch processing: {e}"
        result['errors'].append(error_msg)
        logger.error(error_msg, exc_info=True)
        
    return result

def restore_video_frames(
    video_path: str,
    output_dir: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    timing_mode: str = "after_upscale",
    model_path: Optional[str] = None,
    batch_size: int = 4,
    save_frames: bool = False,
    create_comparison: bool = True,
    preserve_audio: bool = True,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract frames from video, apply face restoration, and reassemble into video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for processed frames and output video
        fidelity_weight: Fidelity weight for restoration
        enable_colorization: Whether to apply colorization
        timing_mode: When to apply restoration (before_upscale, after_upscale, per_frame)
        model_path: Path to CodeFormer model
        batch_size: Number of frames to process simultaneously
        save_frames: Whether to save individual processed frames
        create_comparison: Whether to create before/after comparison video
        preserve_audio: Whether to preserve original audio track
        ffmpeg_preset: FFmpeg encoding preset for comparison video
        ffmpeg_quality: FFmpeg quality setting (CRF/CQ) for comparison video
        ffmpeg_use_gpu: Whether to use GPU encoding for comparison video
        progress_callback: Optional callback for progress updates (progress, status)
        logger: Logger instance
        
    Returns:
        Dictionary containing video processing results with output_video_path and comparison_video_path
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    result = {
        'success': False,
        'extracted_frames': 0,
        'processed_frames': 0,
        'output_frame_dir': None,
        'frame_paths': [],
        'output_video_path': None,
        'comparison_video_path': None,
        'message': '',
        'error': None,
        'processing_time': 0.0,
        'faces_detected_total': 0
    }
    
    start_time = time.time()
    
    try:
        if not os.path.exists(video_path):
            result['error'] = f"Video file not found: {video_path}"
            return result
        
        # Extract video metadata FIRST to get the correct FPS
        if progress_callback:
            progress_callback(0.05, "Extracting video metadata...")
        
        video_metadata = _get_video_metadata(video_path, logger)
        video_fps = video_metadata.get('fps', 30.0)
        
        # Create organized output directory structure like regular upscaling
        # Import the get_next_filename utility to create numbered session folders
        from .file_utils import get_next_filename
        
        # Get next numbered folder (e.g., "0003") for this face restoration session
        base_output_filename_no_ext, output_video_path, tmp_lock_file_path = get_next_filename(output_dir, logger=logger)
        session_output_dir = os.path.join(output_dir, base_output_filename_no_ext)
        os.makedirs(session_output_dir, exist_ok=True)
        
        # Create organized subdirectories for frames
        extracted_frames_dir = os.path.join(session_output_dir, "extracted_frames")
        processed_frames_dir = os.path.join(session_output_dir, "processed_frames")
        os.makedirs(extracted_frames_dir, exist_ok=True)
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        # Create temp directory for processing (will be cleaned up later)
        import tempfile
        import random
        run_id = f"face_restore_{int(time.time())}_{random.randint(1000, 9999)}"
        temp_dir_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_dir_base, run_id)
        frames_dir = os.path.join(temp_dir, "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        logger.info(f"Face restoration session folder: {session_output_dir}")
        logger.info(f"Extracted frames will be saved to: {extracted_frames_dir}")
        logger.info(f"Processed frames will be saved to: {processed_frames_dir}")
        logger.info(f"Save frames setting: {save_frames}")
        
        if progress_callback:
            progress_callback(0.1, "Extracting frames from video...")
            
        # Extract frames using ffmpeg WITHOUT FPS filtering to preserve all frames
        # This fixes the frame loss issue by ensuring every frame from the original video is extracted
        logger.info(f"Extracting all frames from video (original FPS: {video_fps:.6f})")
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        
        try:
            # Extract ALL frames without FPS filtering to avoid frame loss
            cmd = [
                'ffmpeg', '-i', video_path,
                '-y',  # Overwrite output files
                frame_pattern
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if process.returncode != 0:
                result['error'] = f"Frame extraction failed: {process.stderr}"
                return result
                
        except subprocess.TimeoutExpired:
            result['error'] = "Frame extraction timed out"
            return result
        except Exception as e:
            result['error'] = f"Error during frame extraction: {e}"
            return result
            
        # Get list of extracted frames
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")), key=lambda x: natural_sort_key(os.path.basename(x)))
        result['extracted_frames'] = len(frame_files)
        
        if not frame_files:
            result['error'] = "No frames were extracted from video"
            return result
            
        logger.info(f"Extracted {len(frame_files)} frames from video")
        
        # Copy extracted frames to permanent storage immediately so user can see them
        try:
            for frame_file in frame_files:
                frame_name = os.path.basename(frame_file)
                src_path = frame_file
                dst_path = os.path.join(extracted_frames_dir, frame_name)
                shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {len(frame_files)} extracted frames to permanent storage: {extracted_frames_dir}")
        except Exception as e:
            logger.warning(f"Failed to copy extracted frames to permanent storage: {e}")
        
        if progress_callback:
            progress_callback(0.2, f"Processing {len(frame_files)} frames for face restoration...")
            
        # Create temp output directory for restored frames (will copy to permanent later)
        restored_frames_dir = os.path.join(temp_dir, "restored_frames")
        os.makedirs(restored_frames_dir, exist_ok=True)
        
        # Process frames in batches
        def batch_progress_callback(current, total, status):
            overall_progress = 0.2 + (current / total) * 0.7
            if progress_callback:
                progress_callback(overall_progress, f"Face restoration: {status}")
                
        batch_result = restore_frames_batch_true(
            frame_files, restored_frames_dir, fidelity_weight,
            enable_colorization, model_path, batch_size, batch_progress_callback, 
            processed_frames_dir, logger  # Pass processed_frames_dir for live saving
        )
        
        # Log final frame saving results
        if batch_result['success']:
            logger.info(f"Face restoration batch completed: {batch_result['processed_count']} frames processed")
            logger.info(f"All processed frames saved to: {processed_frames_dir}")
        
        result['processed_frames'] = batch_result['processed_count']
        result['faces_detected_total'] = batch_result['faces_detected_total']
        result['frame_paths'] = batch_result['output_paths']
        result['output_frame_dir'] = restored_frames_dir
        
        if batch_result['success'] and result['processed_frames'] > 0:
            if progress_callback:
                progress_callback(0.9, "Reassembling video from processed frames...")
            
            # Generate output video path in the session folder
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path_final = os.path.join(session_output_dir, f"{video_name}_face_restored.mp4")
            
            # Reassemble video from processed frames using the correct FPS
            # Using vsync vfr and exact FPS to maintain original timing
            logger.info(f"Reassembling video with exact FPS: {video_fps:.6f}")
            
            # FIXED: Face restoration video reassembly issue
            # Previously used _reassemble_video_from_frames which used ffmpeg concat without frame durations,
            # causing only 3 frames to be output instead of the full video (e.g., 1,257 frames -> 3 frames).
            # Now using create_video_from_frames_with_duration_preservation which calculates exact FPS
            # and preserves input video duration, ensuring all frames are included in the output.
            
            # Use the same duration preservation method as the working upscaling pipeline
            from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
            
            # Create a temporary directory with frames for the duration preservation method
            import tempfile
            import shutil
            temp_frames_dir_for_video = os.path.join(temp_dir, "frames_for_video_creation")
            os.makedirs(temp_frames_dir_for_video, exist_ok=True)
            
            # Copy processed frames to the temp directory with sequential naming
            sorted_frame_paths = sorted(batch_result['output_paths'])
            for i, frame_path in enumerate(sorted_frame_paths):
                frame_name = f"frame_{i+1:06d}.png"
                dst_path = os.path.join(temp_frames_dir_for_video, frame_name)
                shutil.copy2(frame_path, dst_path)
            
            # Use duration-preserved video creation to ensure exact frame count and FPS
            video_creation_success = create_video_from_frames_with_duration_preservation(
                temp_frames_dir_for_video,
                output_video_path_final,
                video_path,  # Original video for duration reference
                ffmpeg_preset,
                ffmpeg_quality,
                ffmpeg_use_gpu,
                logger=logger
            )
            
            # Clean up temp frames directory
            try:
                shutil.rmtree(temp_frames_dir_for_video)
            except Exception as cleanup_e:
                logger.warning(f"Failed to clean up temp frames directory: {cleanup_e}")
            
            if video_creation_success:
                # Add audio from original video if requested
                if preserve_audio and video_path and os.path.exists(video_path):
                    # Check if original video has audio stream
                    has_audio = _check_video_has_audio(video_path, logger)
                    
                    if has_audio:
                        # Create final video with audio
                        temp_silent_video = output_video_path_final + "_temp_silent.mp4"
                        shutil.move(output_video_path_final, temp_silent_video)
                        
                        # Merge audio from original video
                        audio_merge_cmd = f'ffmpeg -y -i "{temp_silent_video}" -i "{video_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -avoid_negative_ts make_zero "{output_video_path_final}"'
                        
                        from .ffmpeg_utils import run_ffmpeg_command
                        audio_success = run_ffmpeg_command(audio_merge_cmd, "Face Restoration Audio Merge", logger, raise_on_error=False)
                        
                        if audio_success and os.path.exists(output_video_path_final):
                            logger.info("Successfully merged audio from original video")
                            # Clean up temporary silent video
                            try:
                                os.remove(temp_silent_video)
                            except Exception as temp_cleanup_e:
                                logger.warning(f"Failed to clean up temp silent video: {temp_cleanup_e}")
                        else:
                            # Audio merge failed, restore silent video
                            logger.warning("Audio merge failed, using silent video")
                            if os.path.exists(temp_silent_video):
                                shutil.move(temp_silent_video, output_video_path_final)
                    else:
                        logger.info("Original video has no audio track, keeping silent video")
                
                result['output_video_path'] = output_video_path_final
                result['success'] = True
                logger.info(f"Successfully reassembled video: {output_video_path_final}")
                
                # Verify frame count in output video matches input
                output_metadata = _get_video_metadata(output_video_path_final, logger)
                input_frame_count = video_metadata.get('total_frames', result['extracted_frames'])
                output_frame_count = output_metadata.get('total_frames', 0)
                
                if output_frame_count != input_frame_count and abs(output_frame_count - input_frame_count) > 1:
                    logger.warning(f"Frame count mismatch: Input {input_frame_count} frames, Output {output_frame_count} frames")
                else:
                    logger.info(f"Frame count preserved: {output_frame_count} frames (input: {input_frame_count})")
                
                result['message'] = f"Face restoration completed successfully! Processed {result['processed_frames']} frames with {result['faces_detected_total']} faces detected."
                
                # Create comparison video if requested
                if create_comparison:
                    if progress_callback:
                        progress_callback(0.95, "Creating before/after comparison video...")
                    
                    comparison_path = os.path.join(session_output_dir, f"{video_name}_comparison.mp4")
                    comparison_result = _create_comparison_video(
                        original_video_path=video_path,
                        restored_video_path=output_video_path_final,
                        output_path=comparison_path,
                        ffmpeg_preset=ffmpeg_preset,
                        ffmpeg_quality=ffmpeg_quality,
                        ffmpeg_use_gpu=ffmpeg_use_gpu,
                        logger=logger
                    )
                    
                    if comparison_result['success']:
                        result['comparison_video_path'] = comparison_path
                        logger.info(f"Comparison video created: {comparison_path}")
                    else:
                        logger.warning(f"Failed to create comparison video: {comparison_result['error']}")
                
                # Clean up temporary directories
                try:
                    shutil.rmtree(temp_dir)
                    logger.info("Cleaned up temporary processing directories")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directories: {e}")
                    
                # Add session folder info to result for user reference
                result['session_folder'] = session_output_dir
                result['session_name'] = base_output_filename_no_ext
                
                if progress_callback:
                    progress_callback(1.0, "Face restoration completed successfully!")
                    
            else:
                result['error'] = "Video reassembly failed using duration preservation method"
                result['message'] = f"Face restoration processed {result['processed_frames']} frames but failed to create output video."
        else:
            result['error'] = f"Face restoration failed: {len(batch_result['errors'])} errors occurred"
            result['message'] = f"Face restoration failed with {len(batch_result['errors'])} errors."
            
        result['processing_time'] = time.time() - start_time
        
        logger.info(f"Video face restoration completed: {result['processed_frames']}/{result['extracted_frames']} frames processed in {result['processing_time']:.1f}s")
        
    except Exception as e:
        result['error'] = f"Unexpected error during video face restoration: {e}"
        result['message'] = f"Processing failed due to unexpected error: {str(e)}"
        logger.error(result['error'], exc_info=True)
        
    return result

def _get_video_metadata(video_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe.
    
    Args:
        video_path: Path to video file
        logger: Logger instance
        
    Returns:
        Dictionary containing video metadata
    """
    metadata = {
        'fps': 30.0,
        'duration': 0.0,
        'width': 1920,
        'height': 1080,
        'total_frames': 0
    }
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            '-select_streams', 'v:0', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            probe_data = json.loads(result.stdout)
            
            if 'streams' in probe_data and len(probe_data['streams']) > 0:
                stream = probe_data['streams'][0]
                
                # Extract FPS
                if 'r_frame_rate' in stream:
                    fps_parts = stream['r_frame_rate'].split('/')
                    if len(fps_parts) == 2 and fps_parts[1] != '0':
                        metadata['fps'] = float(fps_parts[0]) / float(fps_parts[1])
                elif 'avg_frame_rate' in stream:
                    fps_parts = stream['avg_frame_rate'].split('/')
                    if len(fps_parts) == 2 and fps_parts[1] != '0':
                        metadata['fps'] = float(fps_parts[0]) / float(fps_parts[1])
                
                # Extract dimensions
                if 'width' in stream:
                    metadata['width'] = int(stream['width'])
                if 'height' in stream:
                    metadata['height'] = int(stream['height'])
                
                # Extract duration
                if 'duration' in stream:
                    metadata['duration'] = float(stream['duration'])
                    metadata['total_frames'] = int(metadata['duration'] * metadata['fps'])
                
                logger.info(f"Video metadata: {metadata['width']}x{metadata['height']} @ {metadata['fps']:.2f}fps, {metadata['duration']:.1f}s")
        
    except Exception as e:
        logger.warning(f"Failed to extract video metadata: {e}, using defaults")
    
    return metadata

def _check_video_has_audio(video_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a video file has an audio stream.
    
    Args:
        video_path: Path to video file
        logger: Logger instance
        
    Returns:
        True if video has audio stream, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            '-select_streams', 'a', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            probe_data = json.loads(result.stdout)
            streams = probe_data.get('streams', [])
            has_audio = len(streams) > 0
            
            if has_audio:
                logger.debug(f"Video {video_path} has audio stream")
            else:
                logger.debug(f"Video {video_path} has no audio stream")
                
            return has_audio
        else:
            logger.warning(f"Failed to probe audio streams for {video_path}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout while probing audio streams for {video_path}")
        return False
    except Exception as e:
        logger.warning(f"Error checking audio streams for {video_path}: {e}")
        return False

def _create_comparison_video(
    original_video_path: str,
    restored_video_path: str,
    output_path: str,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Create a dynamic comparison video using the same logic as the main upscaling comparison.
    Automatically chooses side-by-side or top-bottom layout and handles NVENC width limitations.
    
    Args:
        original_video_path: Path to original video
        restored_video_path: Path to face-restored video
        output_path: Path for comparison video output
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting (CRF/CQ)
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
        
    Returns:
        Dictionary containing comparison results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    result = {
        'success': False,
        'output_path': output_path,
        'error': None
    }
    
    try:
        if not os.path.exists(original_video_path):
            result['error'] = f"Original video not found: {original_video_path}"
            return result
            
        if not os.path.exists(restored_video_path):
            result['error'] = f"Restored video not found: {restored_video_path}"
            return result
        
        # Import the comparison video logic from the main module
        from .comparison_video import create_comparison_video
        
        # Use the same sophisticated comparison video creation logic as the main upscaling
        comparison_success = create_comparison_video(
            original_video_path=original_video_path,
            upscaled_video_path=restored_video_path,  # Use restored video as "upscaled" for comparison
            output_path=output_path,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality,
            ffmpeg_use_gpu=ffmpeg_use_gpu,
            logger=logger
        )
        
        if comparison_success:
            result['success'] = True
            logger.info(f"Successfully created face restoration comparison video: {output_path}")
        else:
            result['error'] = "Failed to create comparison video using sophisticated comparison logic"
            logger.error(f"Face restoration comparison video creation failed: {output_path}")
    
    except Exception as e:
        result['error'] = f"Unexpected error during comparison video creation: {e}"
        logger.error(f"Face restoration comparison video error: {e}", exc_info=True)
    
    return result

def apply_face_restoration_to_frames(
    input_frames_dir: str,
    output_frames_dir: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    model_path: Optional[str] = None,
    batch_size: int = 4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Apply face restoration to a directory of frame images.
    This function is used by the upscaling and scene processing integration.
    
    Args:
        input_frames_dir: Directory containing input frames
        output_frames_dir: Directory for output frames
        fidelity_weight: CodeFormer fidelity weight
        enable_colorization: Whether to apply colorization
        model_path: Path to CodeFormer model
        batch_size: Processing batch size
        progress_callback: Progress callback function
        logger: Logger instance
        
    Returns:
        Dictionary containing processing results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    result = {
        'success': False,
        'processed_count': 0,
        'total_count': 0,
        'output_dir': output_frames_dir,
        'error': None,
        'processing_time': 0.0
    }
    
    start_time = time.time()
    
    try:
        if not os.path.exists(input_frames_dir):
            result['error'] = f"Input frames directory not found: {input_frames_dir}"
            return result
        
        # Find all frame files
        frame_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        frame_files = []
        for ext in frame_extensions:
            frame_files.extend(glob.glob(os.path.join(input_frames_dir, f"*{ext}")))
            frame_files.extend(glob.glob(os.path.join(input_frames_dir, f"*{ext.upper()}")))
        
        frame_files = sorted(frame_files)
        result['total_count'] = len(frame_files)
        
        if not frame_files:
            result['error'] = f"No frame files found in: {input_frames_dir}"
            return result
        
        logger.info(f"Processing {len(frame_files)} frames for face restoration")
        
        # Create output directory
        os.makedirs(output_frames_dir, exist_ok=True)
        
        # Process frames in batches
        def batch_progress_callback(current, total, status):
            if progress_callback:
                progress = current / total
                progress_callback(progress, status)
        
        batch_result = restore_frames_batch_true(
            frame_files, output_frames_dir, fidelity_weight,
            enable_colorization, model_path, batch_size, batch_progress_callback, None, logger
        )
        
        result['processed_count'] = batch_result['processed_count']
        result['success'] = batch_result['success']
        result['processing_time'] = time.time() - start_time
        
        if not batch_result['success']:
            result['error'] = f"Frame processing failed: {len(batch_result['errors'])} errors"
        
        logger.info(f"Frame face restoration completed: {result['processed_count']}/{result['total_count']} frames processed in {result['processing_time']:.1f}s")
        
    except Exception as e:
        result['error'] = f"Unexpected error during frame face restoration: {e}"
        logger.error(result['error'], exc_info=True)
    
    return result

def apply_face_restoration_to_scene_frames(
    scene_frames_dir: str,
    output_frames_dir: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    model_path: Optional[str] = None,
    batch_size: int = 4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Apply face restoration to scene frames.
    This is a wrapper around apply_face_restoration_to_frames for scene processing.
    
    Args:
        scene_frames_dir: Directory containing scene frames
        output_frames_dir: Directory for output frames
        fidelity_weight: CodeFormer fidelity weight
        enable_colorization: Whether to apply colorization
        model_path: Path to CodeFormer model
        batch_size: Processing batch size
        progress_callback: Progress callback function
        logger: Logger instance
        
    Returns:
        Dictionary containing processing results
    """
    return apply_face_restoration_to_frames(
        input_frames_dir=scene_frames_dir,
        output_frames_dir=output_frames_dir,
        fidelity_weight=fidelity_weight,
        enable_colorization=enable_colorization,
        model_path=model_path,
        batch_size=batch_size,
        progress_callback=progress_callback,
        logger=logger
    )

def restore_frames_batch_true(
    frame_paths: List[str],
    output_dir: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    model_path: Optional[str] = None,
    batch_size: int = 4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    permanent_frames_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Process frames using TRUE batch processing with CodeFormer.
    
    This function processes multiple face restoration frames simultaneously in real batches,
    maximizing GPU utilization and processing speed. Unlike the regular batch processing
    which processes frames one by one, this processes multiple faces in a single forward pass.
    
    Args:
        frame_paths: List of input frame file paths
        output_dir: Directory to save processed frames
        fidelity_weight: Fidelity weight (0.0-1.0), higher values preserve identity better
        enable_colorization: Whether to apply colorization for grayscale images
        model_path: Path to CodeFormer model (optional, uses default if None)
        batch_size: Number of frames to process simultaneously 
        progress_callback: Optional callback for progress updates (current, total, status)
        logger: Logger instance
        
    Returns:
        Dictionary containing processing results and statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    result = {
        'success': False,
        'processed_count': 0,
        'failed_count': 0,
        'faces_detected_total': 0,
        'output_paths': [],
        'total_processing_time': 0.0,
        'average_time_per_frame': 0.0,
        'errors': []
    }
    
    if logger and permanent_frames_dir:
        logger.info(f"Face restoration batch processing: permanent frames will be saved to {permanent_frames_dir}")
        # Ensure the permanent directory exists
        os.makedirs(permanent_frames_dir, exist_ok=True)
    
    try:
        # Ensure CodeFormer is available before proceeding
        if not ensure_codeformer_available(logger):
            result['errors'].append("CodeFormer environment not available")
            return result
            
        # Import required modules
        import torch
        import cv2
        import numpy as np
        from torchvision.transforms.functional import normalize
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.utils.registry import ARCH_REGISTRY
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils.misc import get_device
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from facelib.utils.misc import is_gray
        
        device = get_device()
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CodeFormer network
        logger.info("Initializing CodeFormer network for true batch processing...")
        net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
            connect_list=['32', '64', '128', '256']
        ).to(device)
        
        # Load model weights - check local path first, then download if needed
        ckpt_path, needs_download = find_codeformer_model_path(model_path, logger)
        
        if ckpt_path:
            # Load the checkpoint
            checkpoint = torch.load(ckpt_path)['params_ema']
            net.load_state_dict(checkpoint)
            net.eval()
        else:
            # Need to download the model
            if needs_download:
                logger.info("Downloading CodeFormer model to weights/CodeFormer/...")
                pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
                ckpt_path = load_file_from_url(
                    url=pretrain_model_url, 
                    model_dir='weights/CodeFormer', 
                    progress=True, 
                    file_name=None
                )
                checkpoint = torch.load(ckpt_path)['params_ema']
                net.load_state_dict(checkpoint)
                net.eval()
            else:
                result['errors'].append("CodeFormer model not found and download failed")
                return result
        
        # Initialize face detection helper
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device
        )
        
        logger.info(f"Processing {len(frame_paths)} frames with TRUE batch processing (batch_size: {batch_size})")
        
        # Step 1: Extract all faces from all frames
        all_face_data = []  # List of face data with frame info
        frames_without_faces = []  # Track frames with no faces for fallback
        
        for frame_idx, frame_path in enumerate(frame_paths):
            if progress_callback:
                progress_callback(frame_idx, len(frame_paths), f"Extracting faces from frame {frame_idx+1}")
                
            try:
                # Load image
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"Failed to load frame: {frame_path}")
                    result['failed_count'] += 1
                    # Save original frame as fallback
                    frames_without_faces.append((frame_path, None))
                    continue
                    
                # Clean face helper for new image
                face_helper.clean_all()
                face_helper.read_image(img)
                
                # Detect and align faces
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=False, resize=640, eye_dist_threshold=5
                )
                
                if num_det_faces > 0:
                    face_helper.align_warp_face()
                    
                    # Store all detected faces
                    for face_idx, cropped_face in enumerate(face_helper.cropped_faces):
                        all_face_data.append({
                            'frame_path': frame_path,
                            'frame_idx': frame_idx,
                            'face_idx': face_idx,
                            'cropped_face': cropped_face,
                            'original_img': img.copy(),
                            'face_helper_state': {
                                'input_img': face_helper.input_img.copy(),
                                'all_landmarks_5': [lm.copy() for lm in face_helper.all_landmarks_5],
                                'det_faces': [face.copy() for face in face_helper.det_faces],
                                'affine_matrices': [mat.copy() for mat in face_helper.affine_matrices],
                                'inverse_affine_matrices': [mat.copy() for mat in face_helper.inverse_affine_matrices]
                            }
                        })
                        
                    result['faces_detected_total'] += num_det_faces
                else:
                    # No faces detected, save original frame
                    frames_without_faces.append((frame_path, img.copy()))
                        
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {e}")
                result['failed_count'] += 1
                # Save original frame as fallback
                frames_without_faces.append((frame_path, None))
                continue
        
        # Save frames without faces as original (fallback)
        for frame_path, original_img in frames_without_faces:
            try:
                frame_name = os.path.splitext(os.path.basename(frame_path))[0]
                
                # Load original if not provided
                if original_img is None:
                    original_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                    if original_img is None:
                        logger.error(f"Could not load original frame for fallback: {frame_path}")
                        continue
                
                # Save to temp output
                output_path = os.path.join(output_dir, f"{frame_name}_restored.png")
                cv2.imwrite(output_path, original_img)
                result['output_paths'].append(output_path)
                
                # Save to permanent directory if provided
                if permanent_frames_dir:
                    permanent_frame_name = f"{frame_name}.png"
                    permanent_path = os.path.join(permanent_frames_dir, permanent_frame_name)
                    cv2.imwrite(permanent_path, original_img)
                    logger.debug(f"Saved original frame (no faces detected): {permanent_frame_name}")
                
                result['processed_count'] += 1
                
            except Exception as e:
                logger.error(f"Failed to save original frame {frame_path}: {e}")
                result['failed_count'] += 1
        
        if not all_face_data and not frames_without_faces:
            result['errors'].append("No faces detected and no frames could be processed")
            return result
            
        logger.info(f"Extracted {len(all_face_data)} faces from {len(frame_paths)} frames, {len(frames_without_faces)} frames without faces saved as originals")
        
        # Step 2: Process faces in true batches (only if we have faces to process)
        total_faces = len(all_face_data)
        processed_faces = 0
        
        if total_faces > 0:
            logger.info(f"Processing {total_faces} faces in TRUE batch mode")
        
        for batch_start in range(0, total_faces, batch_size):
            batch_end = min(batch_start + batch_size, total_faces)
            current_batch = all_face_data[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_faces + batch_size - 1) // batch_size
            
            logger.info(f"Processing TRUE batch {batch_num}/{total_batches}: {len(current_batch)} faces simultaneously")
            
            if progress_callback:
                progress_callback(processed_faces, total_faces, f"Processing batch {batch_num}/{total_batches}")
            
            batch_start_time = time.time()
            
            try:
                # Prepare batch tensor
                batch_faces = []
                for face_data in current_batch:
                    cropped_face = face_data['cropped_face']
                    # Convert to tensor
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    batch_faces.append(cropped_face_t)
                
                # Stack into batch tensor [batch_size, 3, 512, 512]
                batch_tensor = torch.stack(batch_faces, dim=0).to(device)
                
                # TRUE BATCH PROCESSING - Process all faces at once!
                with torch.no_grad():
                    batch_outputs = net(batch_tensor, w=fidelity_weight, adain=True)[0]
                    
                # Convert batch outputs back to individual images
                restored_faces = []
                for i in range(batch_outputs.shape[0]):
                    restored_face = tensor2img(batch_outputs[i], rgb2bgr=True, min_max=(-1, 1))
                    restored_face = restored_face.astype('uint8')
                    restored_faces.append(restored_face)
                
                # Clean up GPU memory
                del batch_tensor, batch_outputs
                torch.cuda.empty_cache()
                
                # Step 3: Paste restored faces back to original images and save
                for face_data, restored_face in zip(current_batch, restored_faces):
                    frame_name = os.path.splitext(os.path.basename(face_data['frame_path']))[0]
                    output_path = os.path.join(output_dir, f"{frame_name}_restored.png")
                    
                    try:
                        # Restore face helper state
                        face_helper.clean_all()
                        face_helper.input_img = face_data['face_helper_state']['input_img']
                        face_helper.all_landmarks_5 = face_data['face_helper_state']['all_landmarks_5']
                        face_helper.det_faces = face_data['face_helper_state']['det_faces']
                        face_helper.affine_matrices = face_data['face_helper_state']['affine_matrices']
                        face_helper.inverse_affine_matrices = face_data['face_helper_state']['inverse_affine_matrices']
                        face_helper.cropped_faces = [face_data['cropped_face']]
                        
                        # Add restored face
                        face_helper.add_restored_face(restored_face, face_data['cropped_face'])
                        
                        # Paste back to original image
                        face_helper.get_inverse_affine(None)
                        restored_img = face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)
                        
                        # Save result
                        cv2.imwrite(output_path, restored_img)
                        
                        # Save to permanent directory immediately if provided
                        if permanent_frames_dir:
                            permanent_frame_name = f"{frame_name}.png"  # Use original naming convention
                            permanent_path = os.path.join(permanent_frames_dir, permanent_frame_name)
                            cv2.imwrite(permanent_path, restored_img)
                            if (result['processed_count'] + 1) % 25 == 0 or (result['processed_count'] + 1) % 100 == 0:
                                logger.info(f"Live frame saving: {result['processed_count'] + 1} frames saved to {permanent_frames_dir}")
                            else:
                                logger.debug(f"Saved restored frame to permanent storage: {permanent_frame_name}")
                        
                        result['output_paths'].append(output_path)
                        result['processed_count'] += 1
                        
                    except Exception as e:
                        # If face restoration fails, use the original frame as fallback
                        logger.warning(f"Face restoration failed for {face_data['frame_path']}: {e}. Using original frame.")
                        
                        try:
                            # Use original frame as fallback
                            original_img = face_data['original_img']
                            cv2.imwrite(output_path, original_img)
                            
                            # Save original to permanent directory if provided
                            if permanent_frames_dir:
                                permanent_frame_name = f"{frame_name}.png"
                                permanent_path = os.path.join(permanent_frames_dir, permanent_frame_name)
                                cv2.imwrite(permanent_path, original_img)
                                if (result['processed_count'] + 1) % 25 == 0 or (result['processed_count'] + 1) % 100 == 0:
                                    logger.info(f"Live frame saving (fallback): {result['processed_count'] + 1} frames saved to {permanent_frames_dir}")
                                else:
                                    logger.debug(f"Saved original frame to permanent storage (fallback): {permanent_frame_name}")
                            
                            result['output_paths'].append(output_path)
                            result['processed_count'] += 1  # Count as processed even though it's original
                            
                        except Exception as fallback_e:
                            logger.error(f"Even fallback to original frame failed for {face_data['frame_path']}: {fallback_e}")
                            result['failed_count'] += 1
                            result['errors'].append(f"Complete failure for {face_data['frame_path']}: {e}, fallback: {fallback_e}")
                
                batch_time = time.time() - batch_start_time
                processed_faces += len(current_batch)
                
                avg_time_per_face = batch_time / len(current_batch) if len(current_batch) > 0 else 0
                logger.info(f"TRUE batch {batch_num}/{total_batches} completed: {len(current_batch)} faces in {batch_time:.2f}s (avg {avg_time_per_face:.2f}s/face)")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                result['failed_count'] += len(current_batch)
                result['errors'].append(f"Batch {batch_num} error: {e}")
                continue
        
        result['total_processing_time'] = time.time() - start_time
        result['success'] = result['processed_count'] > 0
        
        if result['success']:
            faces_processed = len(all_face_data) if 'all_face_data' in locals() else 0
            originals_saved = len(frames_without_faces) if 'frames_without_faces' in locals() else 0
            logger.info(f"TRUE batch processing completed: {result['processed_count']} frames processed ({faces_processed} with face restoration, {originals_saved} originals), {result['failed_count']} failed in {result['total_processing_time']:.2f}s")
            if permanent_frames_dir:
                logger.info(f"All frames (restored + originals) saved to: {permanent_frames_dir}")
        else:
            logger.error(f"TRUE batch processing failed: {result['failed_count']} frames failed")
            
    except Exception as e:
        result['error'] = f"Critical error in TRUE batch processing: {e}"
        logger.error(result['error'], exc_info=True)
        
    return result 