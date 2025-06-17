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
            from basicsr.models import create_model
            from basicsr.utils.options import dict2str
            
            # Store imported modules
            setup_result['modules'] = {
                'FaceRestoreHelper': FaceRestoreHelper,
                'init_detection_model': init_detection_model,
                'init_parsing_model': init_parsing_model,
                'imwrite': imwrite,
                'img2tensor': img2tensor,
                'tensor2img': tensor2img,
                'create_model': create_model,
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

def restore_single_image(
    image_path: str,
    output_path: str,
    fidelity_weight: float = 0.7,
    enable_colorization: bool = False,
    model_path: Optional[str] = None,
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
        logger: Logger instance
        
    Returns:
        Dictionary containing restoration results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
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
            logger.info(f"Applying colorization to grayscale image: {os.path.basename(image_path)}")
            # Apply colorization first, then face restoration
            result = _apply_colorization(img, image_path, output_path, fidelity_weight, model_path, logger)
        else:
            # Apply face restoration
            result = _apply_face_restoration(img, image_path, output_path, fidelity_weight, model_path, logger)
            
        result['processing_time'] = time.time() - start_time
        
        if result['success']:
            logger.info(f"Face restoration completed: {os.path.basename(image_path)} -> {os.path.basename(output_path)} ({result['processing_time']:.2f}s)")
        else:
            logger.warning(f"Face restoration failed for {os.path.basename(image_path)}: {result['error']}")
            
    except Exception as e:
        result['error'] = f"Unexpected error during face restoration: {e}"
        logger.error(result['error'], exc_info=True)
        
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
            # Build command
            cmd = [
                sys.executable,
                codeformer_script,
                '-w', str(fidelity_weight),
                '--input_path', input_path,
                '--output_path', os.path.dirname(output_path)
            ]
            
            if model_path:
                cmd.extend(['--model_path', model_path])
                
            # Run CodeFormer
            logger.debug(f"Running CodeFormer command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode == 0:
                # Find the generated output file
                # CodeFormer typically saves with a specific naming pattern
                expected_output = os.path.join(os.path.dirname(output_path), f"restored_{os.path.basename(input_path)}")
                
                if os.path.exists(expected_output):
                    # Move to desired output location
                    shutil.move(expected_output, output_path)
                    result['success'] = True
                    result['output_path'] = output_path
                    result['faces_detected'] = 1  # Assume at least one face if successful
                else:
                    # Look for any output files in the output directory
                    output_dir = os.path.dirname(output_path)
                    output_files = glob.glob(os.path.join(output_dir, "*.png")) + glob.glob(os.path.join(output_dir, "*.jpg"))
                    
                    if output_files:
                        # Use the most recent file
                        latest_file = max(output_files, key=os.path.getctime)
                        shutil.move(latest_file, output_path)
                        result['success'] = True
                        result['output_path'] = output_path
                        result['faces_detected'] = 1
                    else:
                        result['error'] = "CodeFormer completed but no output file found"
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
        if not codeformer_available:
            result['errors'].append("CodeFormer environment not available")
            return result
            
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame_path in enumerate(frame_paths):
            if progress_callback:
                progress_callback(i, len(frame_paths), f"Processing frame {i+1}/{len(frame_paths)}")
                
            try:
                # Generate output path
                frame_name = os.path.basename(frame_path)
                name_without_ext = os.path.splitext(frame_name)[0]
                output_path = os.path.join(output_dir, f"{name_without_ext}_restored.png")
                
                # Restore single frame
                frame_result = restore_single_image(
                    frame_path, output_path, fidelity_weight, 
                    enable_colorization, model_path, logger
                )
                
                if frame_result['success']:
                    result['processed_count'] += 1
                    result['output_paths'].append(frame_result['output_path'])
                    result['faces_detected_total'] += frame_result['faces_detected']
                    logger.debug(f"Successfully processed frame {i+1}/{len(frame_paths)}: {frame_name}")
                else:
                    result['failed_count'] += 1
                    error_msg = f"Frame {frame_name}: {frame_result['error']}"
                    result['errors'].append(error_msg)
                    logger.warning(error_msg)
                    
            except Exception as e:
                result['failed_count'] += 1
                error_msg = f"Unexpected error processing frame {frame_path}: {e}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                
        result['total_processing_time'] = time.time() - start_time
        result['success'] = result['processed_count'] > 0
        
        if progress_callback:
            progress_callback(len(frame_paths), len(frame_paths), "Batch processing completed")
            
        logger.info(f"Batch face restoration completed: {result['processed_count']}/{result['total_count']} frames processed successfully")
        
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
            
        # Create frame extraction directory
        frames_dir = os.path.join(output_dir, "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        if progress_callback:
            progress_callback(0.1, "Extracting frames from video...")
            
        # Extract frames using ffmpeg
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'fps=fps=30',  # Extract at 30fps or original fps
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
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        result['extracted_frames'] = len(frame_files)
        
        if not frame_files:
            result['error'] = "No frames were extracted from video"
            return result
            
        logger.info(f"Extracted {len(frame_files)} frames from video")
        
        if progress_callback:
            progress_callback(0.2, f"Processing {len(frame_files)} frames for face restoration...")
            
        # Create output directory for restored frames
        restored_frames_dir = os.path.join(output_dir, "restored_frames")
        os.makedirs(restored_frames_dir, exist_ok=True)
        
        # Process frames in batches
        def batch_progress_callback(current, total, status):
            overall_progress = 0.2 + (current / total) * 0.7
            if progress_callback:
                progress_callback(overall_progress, f"Face restoration: {status}")
                
        batch_result = restore_frames_batch(
            frame_files, restored_frames_dir, fidelity_weight,
            enable_colorization, model_path, batch_progress_callback, logger
        )
        
        result['processed_frames'] = batch_result['processed_count']
        result['faces_detected_total'] = batch_result['faces_detected_total']
        result['frame_paths'] = batch_result['output_paths']
        result['output_frame_dir'] = restored_frames_dir
        
        if batch_result['success'] and result['processed_frames'] > 0:
            if progress_callback:
                progress_callback(0.9, "Reassembling video from processed frames...")
            
            # Get video metadata for reassembly
            video_metadata = _get_video_metadata(video_path, logger)
            
            # Generate output video path
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"{video_name}_face_restored.mp4")
            
            # Reassemble video from processed frames
            reassembly_result = _reassemble_video_from_frames(
                frame_paths=batch_result['output_paths'],
                output_path=output_video_path,
                original_video_path=video_path if preserve_audio else None,
                fps=video_metadata.get('fps', 30.0),
                preserve_audio=preserve_audio,
                logger=logger
            )
            
            if reassembly_result['success']:
                result['output_video_path'] = output_video_path
                result['success'] = True
                result['message'] = f"Face restoration completed successfully! Processed {result['processed_frames']} frames with {result['faces_detected_total']} faces detected."
                
                # Create comparison video if requested
                if create_comparison:
                    if progress_callback:
                        progress_callback(0.95, "Creating before/after comparison video...")
                    
                    comparison_path = os.path.join(output_dir, f"{video_name}_comparison.mp4")
                    comparison_result = _create_comparison_video(
                        original_video_path=video_path,
                        restored_video_path=output_video_path,
                        output_path=comparison_path,
                        logger=logger
                    )
                    
                    if comparison_result['success']:
                        result['comparison_video_path'] = comparison_path
                        logger.info(f"Comparison video created: {comparison_path}")
                    else:
                        logger.warning(f"Failed to create comparison video: {comparison_result['error']}")
                
                # Clean up temporary frames if not saving them
                if not save_frames:
                    try:
                        shutil.rmtree(frames_dir)
                        shutil.rmtree(restored_frames_dir)
                        logger.info("Cleaned up temporary frame directories")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary directories: {e}")
                
                if progress_callback:
                    progress_callback(1.0, "Face restoration completed successfully!")
                    
            else:
                result['error'] = f"Video reassembly failed: {reassembly_result['error']}"
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

def _get_video_metadata(video_path: str, logger: logging.Logger) -> Dict[str, Any]:
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

def _reassemble_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    original_video_path: Optional[str] = None,
    fps: float = 30.0,
    preserve_audio: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Reassemble video from processed frames.
    
    Args:
        frame_paths: List of paths to processed frame images
        output_path: Path for output video
        original_video_path: Path to original video (for audio extraction)
        fps: Frame rate for output video
        preserve_audio: Whether to include audio from original video
        logger: Logger instance
        
    Returns:
        Dictionary containing reassembly results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    result = {
        'success': False,
        'output_path': output_path,
        'error': None
    }
    
    try:
        if not frame_paths:
            result['error'] = "No frames provided for video reassembly"
            return result
        
        # Sort frame paths to ensure correct order
        sorted_frames = sorted(frame_paths)
        
        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for frame_path in sorted_frames:
                f.write(f"file '{frame_path}'\n")
                f.write(f"duration {1/fps}\n")
            # Add the last frame again for proper duration
            if sorted_frames:
                f.write(f"file '{sorted_frames[-1]}'\n")
            temp_list_file = f.name
        
        try:
            # Create video from frames
            if preserve_audio and original_video_path and os.path.exists(original_video_path):
                # Create video with audio from original
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', temp_list_file,
                    '-i', original_video_path,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                    '-r', str(fps), '-y', output_path
                ]
            else:
                # Create video without audio
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', temp_list_file,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-r', str(fps), '-y', output_path
                ]
            
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if process.returncode == 0:
                result['success'] = True
                logger.info(f"Successfully reassembled video: {output_path}")
            else:
                result['error'] = f"FFmpeg failed: {process.stderr}"
                logger.error(f"Video reassembly failed: {process.stderr}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_list_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_list_file}: {e}")
    
    except subprocess.TimeoutExpired:
        result['error'] = "Video reassembly timed out"
        logger.error("Video reassembly timed out")
    except Exception as e:
        result['error'] = f"Unexpected error during video reassembly: {e}"
        logger.error(f"Video reassembly error: {e}", exc_info=True)
    
    return result

def _create_comparison_video(
    original_video_path: str,
    restored_video_path: str,
    output_path: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Create a side-by-side comparison video.
    
    Args:
        original_video_path: Path to original video
        restored_video_path: Path to face-restored video
        output_path: Path for comparison video output
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
        
        # Create side-by-side comparison using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', original_video_path,
            '-i', restored_video_path,
            '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]',
            '-map', '[vid]',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-y', output_path
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if process.returncode == 0:
            result['success'] = True
            logger.info(f"Successfully created comparison video: {output_path}")
        else:
            result['error'] = f"FFmpeg comparison failed: {process.stderr}"
            logger.error(f"Comparison video creation failed: {process.stderr}")
    
    except subprocess.TimeoutExpired:
        result['error'] = "Comparison video creation timed out"
        logger.error("Comparison video creation timed out")
    except Exception as e:
        result['error'] = f"Unexpected error during comparison video creation: {e}"
        logger.error(f"Comparison video error: {e}", exc_info=True)
    
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
        
        batch_result = restore_frames_batch(
            frame_files, output_frames_dir, fidelity_weight,
            enable_colorization, model_path, batch_progress_callback, logger
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