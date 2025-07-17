"""
Image upscaler utilities for handling various AI upscaling models.
Provides model scanning, loading, caching, and batch processing functionality.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
import shutil
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import logging
import math

# Supported model file extensions
SUPPORTED_EXTENSIONS = ['.pth', '.safetensors', '.pt', '.bin', '.onnx']

# Global cache for loaded models to avoid reloading
_loaded_models_cache = {}

# Import cancellation manager
from .cancellation_manager import cancellation_manager, CancelledError

def get_upscale_models_dir(base_path: str) -> str:
    """Get the upscale models directory path."""
    return os.path.join(base_path, "upscale_models")

def scan_for_models(models_dir: str, logger: logging.Logger = None) -> List[str]:
    """
    Scans the specified directory for model files with supported extensions.
    
    Args:
        models_dir: Directory path to scan for models
        logger: Logger instance for logging
        
    Returns:
        List of model filenames found
    """
    if logger:
        logger.info(f"Scanning for image upscaler models in: {models_dir}")
    
    if not os.path.exists(models_dir):
        if logger:
            logger.info(f"Creating upscale models directory: {models_dir}")
        try:
            os.makedirs(models_dir, exist_ok=True)
        except Exception as e:
            if logger:
                logger.error(f"Failed to create models directory: {e}")
            return []
        return []
    
    found_models = []
    try:
        for filename in os.listdir(models_dir):
            if any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                found_models.append(filename)
        
        found_models.sort()
        
        if logger:
            logger.info(f"Found {len(found_models)} image upscaler model(s): {found_models}")
        
    except Exception as e:
        if logger:
            logger.error(f"Error scanning models directory: {e}")
        return []
    
    return found_models

def get_model_info(model_path: str, logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Get information about a model without fully loading it.
    
    Args:
        model_path: Path to the model file
        logger: Logger instance
        
    Returns:
        Dictionary with model information
    """
    try:
        # Try to import spandrel
        try:
            import spandrel
        except ImportError:
            if logger:
                logger.warning("Spandrel library not available. Install with: pip install spandrel")
            return {"error": "Spandrel library not available"}
        
        # Load model to get info
        model = spandrel.ModelLoader().load_from_file(model_path)
        
        info = {
            "scale": getattr(model, 'scale', 'Unknown'),
            "architecture": getattr(model, 'architecture', 'Unknown'),
            "architecture_name": getattr(model.architecture, 'name', 'Unknown') if hasattr(model, 'architecture') else 'Unknown',
            "input_channels": getattr(model, 'input_channels', 'Unknown'),
            "output_channels": getattr(model, 'output_channels', 'Unknown'),
            "supports_half": getattr(model, 'supports_half', False),
            "supports_bfloat16": getattr(model, 'supports_bfloat16', False),
        }
        
        # Clean up model from memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return info
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to get model info for {model_path}: {e}")
        return {"error": str(e)}

def load_model(model_path: str, device: str = "cuda", logger: logging.Logger = None) -> Optional[Any]:
    """
    Load and cache an upscaler model.
    
    Args:
        model_path: Path to the model file
        device: Device to load model on ('cuda' or 'cpu')
        logger: Logger instance
        
    Returns:
        Loaded model instance or None if failed
    """
    global _loaded_models_cache
    
    if not os.path.exists(model_path):
        if logger:
            logger.error(f"Model file not found: {model_path}")
        return None
    
    # Check cache first
    cache_key = f"{model_path}_{device}"
    if cache_key in _loaded_models_cache:
        if logger:
            logger.info(f"Using cached model: {os.path.basename(model_path)}")
        return _loaded_models_cache[cache_key]
    
    try:
        # Import spandrel
        try:
            import spandrel
        except ImportError:
            if logger:
                logger.error("Spandrel library not available. Install with: pip install spandrel")
            return None
        
        if logger:
            logger.info(f"Loading image upscaler model: {os.path.basename(model_path)} on {device}")
        
        load_start_time = time.time()
        
        # Load model
        model = spandrel.ModelLoader().load_from_file(model_path)
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
        elif device == "cpu":
            model = model.to("cpu")
        else:
            if logger:
                logger.warning(f"Requested device {device} not available, using CPU")
            model = model.to("cpu")
        
        model.eval()
        
        # Cache the model
        _loaded_models_cache[cache_key] = model
        
        load_time = time.time() - load_start_time
        
        if logger:
            logger.info(f"Model loaded successfully in {load_time:.2f}s. Scale: {getattr(model, 'scale', 'Unknown')}, Architecture: {getattr(model.architecture, 'name', 'Unknown') if hasattr(model, 'architecture') else 'Unknown'}")
        
        return model
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load model {model_path}: {e}")
        return None

def unload_model(model_path: str, device: str = "cuda", logger: logging.Logger = None):
    """
    Remove a model from cache and free memory.
    
    Args:
        model_path: Path to the model file
        device: Device the model was loaded on
        logger: Logger instance
    """
    global _loaded_models_cache
    
    cache_key = f"{model_path}_{device}"
    if cache_key in _loaded_models_cache:
        if logger:
            logger.info(f"Unloading model from cache: {os.path.basename(model_path)}")
        
        del _loaded_models_cache[cache_key]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def clear_model_cache(logger: logging.Logger = None):
    """
    Clear all cached models and free memory.
    
    Args:
        logger: Logger instance
    """
    global _loaded_models_cache
    
    if logger:
        logger.info(f"Clearing {len(_loaded_models_cache)} cached models")
    
    _loaded_models_cache.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def prepare_image_tensor(image_path: str, device: str = "cuda") -> torch.Tensor:
    """
    Load an image and prepare it as a tensor.
    
    Args:
        image_path: Path to the image file
        device: Device to put tensor on
        
    Returns:
        Image tensor in format CHW, normalized to [0,1], on specified device
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img) / 255.0).float().permute(2, 0, 1)
    return img_tensor.to(device)

def prepare_frame_tensor(frame_bgr: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Prepare a BGR frame as a tensor.
    
    Args:
        frame_bgr: BGR frame as numpy array (HWC)
        device: Device to put tensor on
        
    Returns:
        Image tensor in format CHW, normalized to [0,1], on specified device
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame_rgb / 255.0).float().permute(2, 0, 1)
    return img_tensor.to(device)

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert an output tensor to a BGR frame.
    
    Args:
        tensor: Output tensor from model (CHW format, with or without batch dimension)
        
    Returns:
        BGR frame as numpy array (HWC, uint8)
    """
    try:
        # Handle different tensor shapes
        if tensor.dim() == 4:  # NCHW format (batch dimension)
            # Take the first (and should be only) item from batch
            if tensor.shape[0] > 0:
                img_tensor = tensor[0]  # CHW
            else:
                raise ValueError("Empty batch tensor")
        elif tensor.dim() == 3:  # CHW format (no batch dimension)
            img_tensor = tensor
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # Move to CPU and convert to numpy
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        # Clip and convert to uint8
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return frame_bgr
        
    except Exception as e:
        # Create a debug frame if conversion fails
        raise ValueError(f"Failed to convert tensor to frame: {e}, tensor shape: {tensor.shape}")

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert an output tensor to a PIL Image.
    
    Args:
        tensor: Output tensor from model (CHW format, with or without batch dimension)
        
    Returns:
        PIL Image in RGB format
    """
    try:
        # Handle different tensor shapes
        if tensor.dim() == 4:  # NCHW format (batch dimension)
            # Take the first (and should be only) item from batch
            if tensor.shape[0] > 0:
                img_tensor = tensor[0]  # CHW
            else:
                raise ValueError("Empty batch tensor")
        elif tensor.dim() == 3:  # CHW format (no batch dimension)
            img_tensor = tensor
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
        
    except Exception as e:
        raise ValueError(f"Failed to convert tensor to PIL: {e}, tensor shape: {tensor.shape}")

def group_frames_by_size(frame_files: List[str], frames_dir: str, logger: logging.Logger = None) -> Dict[Tuple[int, int], List[str]]:
    """
    Group frame files by their dimensions for efficient batch processing.
    
    Args:
        frame_files: List of frame filenames
        frames_dir: Directory containing the frames
        logger: Logger instance
        
    Returns:
        Dictionary mapping (width, height) to list of frame files
    """
    frames_by_size = defaultdict(list)
    
    if logger:
        logger.info(f"Analyzing dimensions of {len(frame_files)} frames...")
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        try:
            # Read image to get dimensions
            frame = cv2.imread(frame_path)
            if frame is not None:
                h, w = frame.shape[:2]
                frames_by_size[(w, h)].append(frame_file)
            else:
                if logger:
                    logger.warning(f"Could not read frame {frame_file}, skipping")
        except Exception as e:
            if logger:
                logger.warning(f"Error reading frame {frame_file}: {e}")
    
    if logger:
        logger.info(f"Found {len(frames_by_size)} unique frame dimension(s)")
        for size, files in frames_by_size.items():
            logger.info(f"  {size[0]}x{size[1]}: {len(files)} frames")
    
    return dict(frames_by_size)

def upscale_frame_batch(
    frame_tensors: List[torch.Tensor], 
    model: Any, 
    device: str = "cuda",
    logger: logging.Logger = None
) -> List[torch.Tensor]:
    """
    Upscale a batch of frame tensors.
    
    Args:
        frame_tensors: List of frame tensors (CHW format)
        model: Loaded upscaler model
        device: Device to process on
        logger: Logger instance
        
    Returns:
        List of upscaled frame tensors (same length as input or empty on failure)
    """
    if not frame_tensors:
        return []
    
    try:
        if logger:
            logger.debug(f"Processing batch of {len(frame_tensors)} tensors")
            for i, tensor in enumerate(frame_tensors):
                logger.debug(f"  Input tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Stack tensors into batch (NCHW)
        input_batch = torch.stack(frame_tensors)
        
        if logger:
            logger.debug(f"Stacked input batch shape: {input_batch.shape}")
        
        # Run inference
        with torch.no_grad():
            output_batch = model(input_batch)
        
        if logger:
            logger.debug(f"Model output batch shape: {output_batch.shape if output_batch is not None else 'None'}")
        
        # Validate output batch
        if output_batch is None:
            if logger:
                logger.error(f"Model returned None output for batch of {len(frame_tensors)} frames")
            return []
            
        if output_batch.shape[0] != input_batch.shape[0]:
            if logger:
                logger.error(f"Invalid output batch: expected {input_batch.shape[0]} frames, got {output_batch.shape[0]}")
            return []
        
        # Split back into individual tensors
        output_tensors = []
        for i in range(output_batch.shape[0]):
            try:
                individual_tensor = output_batch[i]
                if logger:
                    logger.debug(f"  Output tensor {i}: shape={individual_tensor.shape}, dtype={individual_tensor.dtype}")
                output_tensors.append(individual_tensor)
            except IndexError as e:
                if logger:
                    logger.error(f"Index error extracting frame {i} from output batch: {e}")
                    logger.error(f"  Output batch shape: {output_batch.shape}")
                    logger.error(f"  Trying to access index {i}")
                # Return empty list to indicate complete failure
                return []
        
        # Final validation of output tensors list
        if len(output_tensors) != len(frame_tensors):
            if logger:
                logger.error(f"Output tensor count mismatch: expected {len(frame_tensors)}, got {len(output_tensors)}")
            return []
        
        if logger:
            logger.debug(f"Successfully processed batch: {len(output_tensors)} output tensors")
        
        return output_tensors
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if logger:
                logger.error(f"GPU out of memory during batch upscaling: {e}")
            # Try to clear cache and return empty
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if logger:
                logger.error(f"Runtime error in batch upscaling: {e}")
        return []
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error in batch upscaling: {e}")
        return []

def process_frames_batch(
    frame_files: List[str],
    input_dir: str,
    output_dir: str,
    model: Any,
    batch_size: int = 4,
    device: str = "cuda",
    progress_callback = None,
    secondary_output_dir: str = None,
    logger: logging.Logger = None
) -> Tuple[int, int]:
    """
    Process a batch of frames with the upscaler model.
    
    Args:
        frame_files: List of frame filenames to process
        input_dir: Input frames directory
        output_dir: Output frames directory
        model: Loaded upscaler model
        batch_size: Number of frames to process at once
        device: Device to process on
        progress_callback: Optional callback for progress updates
        secondary_output_dir: Optional secondary output directory for immediate saving
        logger: Logger instance
        
    Returns:
        Tuple of (processed_count, failed_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If a secondary directory is provided, create it as well so we can copy
    # frames there in real-time.
    if secondary_output_dir:
        os.makedirs(secondary_output_dir, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    total_frames = len(frame_files)
    
    # Calculate total number of processing steps (batches) so we can report detailed progress
    # Guard against division by zero if batch_size is somehow 0
    total_steps = math.ceil(total_frames / batch_size) if batch_size else 1
    current_step = 0  # Will be incremented at the start of each batch
    
    # Timing tracking for ETA calculation
    processing_start_time = time.time()
    batch_times = []  # Store time for each batch
    
    # Group frames by size for efficient batching
    frames_by_size = group_frames_by_size(frame_files, input_dir, logger)
    
    for size, size_frame_files in frames_by_size.items():
        # Check for cancellation before processing each size group
        try:
            cancellation_manager.check_cancel("image upscaler - before size group")
        except CancelledError:
            if logger:
                logger.info(f"Image upscaling cancelled before processing size group {size[0]}x{size[1]}")
            # Return current progress with partial results
            return processed_count, failed_count + (total_frames - processed_count - failed_count)
        
        if logger:
            logger.info(f"Processing {len(size_frame_files)} frames of size {size[0]}x{size[1]}")
        
        # Process frames in batches within each size group
        for i in range(0, len(size_frame_files), batch_size):
            # Check for cancellation before each batch
            try:
                cancellation_manager.check_cancel("image upscaler - before batch")
            except CancelledError:
                if logger:
                    logger.info(f"Image upscaling cancelled during batch processing (step {current_step + 1})")
                # Return current progress with partial results
                return processed_count, failed_count + (total_frames - processed_count - failed_count)
                
            # Increment the global step counter *before* processing so that step numbers start at 1
            current_step += 1
            
            # Start timing this batch
            batch_start_time = time.time()

            batch_files = size_frame_files[i:i + batch_size]
            
            # Track how many frames get successfully written in this particular batch so we
            # can accurately update failure counts even if an unexpected exception occurs
            processed_in_current_batch = 0
            
            try:
                # Load frame tensors
                frame_tensors = []
                valid_files = []
                
                for frame_file in batch_files:
                    frame_path = os.path.join(input_dir, frame_file)
                    try:
                        frame_bgr = cv2.imread(frame_path)
                        if frame_bgr is not None:
                            frame_tensor = prepare_frame_tensor(frame_bgr, device)
                            frame_tensors.append(frame_tensor)
                            valid_files.append(frame_file)
                        else:
                            if logger:
                                logger.warning(f"Could not read frame {frame_file}")
                            failed_count += 1
                    except Exception as e:
                        if logger:
                            logger.warning(f"Error loading frame {frame_file}: {e}")
                        failed_count += 1
                
                if not frame_tensors:
                    continue
                
                # Upscale batch
                if logger:
                    # Temporarily increase log level for debugging
                    original_level = logger.level
                    logger.setLevel(logging.DEBUG)
                
                output_tensors = upscale_frame_batch(frame_tensors, model, device, logger)
                
                if logger:
                    # Restore original log level
                    logger.setLevel(original_level)
                
                # Handle batch processing failure
                if not output_tensors:
                    if logger:
                        logger.error(f"Batch processing completely failed for {len(valid_files)} frames")
                    failed_count += len(valid_files)
                    continue
                
                if len(output_tensors) != len(valid_files):
                    if logger:
                        logger.error(f"Batch processing failed: expected {len(valid_files)} outputs, got {len(output_tensors)}")
                    failed_count += len(valid_files)
                    continue
                
                # Save output frames - ensure we don't exceed the shorter list
                actual_outputs = min(len(valid_files), len(output_tensors))
                for idx in range(actual_outputs):
                    frame_file = valid_files[idx]
                    try:
                        # Get output tensor with bounds checking
                        if idx >= len(output_tensors):
                            raise IndexError(f"Index {idx} exceeds output tensor list length {len(output_tensors)}")
                        
                        output_tensor = output_tensors[idx]
                        
                        # Debug tensor info
                        if logger:
                            logger.debug(f"Processing tensor {idx}: shape={output_tensor.shape}, dtype={output_tensor.dtype}")
                        
                        # Convert tensor to frame
                        output_frame = tensor_to_frame(output_tensor)
                        
                        # Save frame to the primary (temporary) output dir
                        output_path = os.path.join(output_dir, frame_file)
                        success = cv2.imwrite(output_path, output_frame)
                        
                        if not success:
                            raise RuntimeError(f"cv2.imwrite failed for {frame_file}")
                        
                        processed_count += 1
                        processed_in_current_batch += 1
                        
                        # Additionally save to the secondary permanent folder
                        if secondary_output_dir:
                            try:
                                secondary_path = os.path.join(secondary_output_dir, frame_file)
                                # Use cv2.imwrite again – this is usually fast because
                                # the data is already resident in memory.
                                cv2.imwrite(secondary_path, output_frame)
                            except Exception as sec_e:
                                if logger:
                                    logger.warning(f"Could not write frame {frame_file} to secondary dir: {sec_e}")
                        
                    except Exception as e:
                        if logger:
                            logger.error(f"Error saving frame {frame_file} (index {idx}): {e}")
                            logger.error(f"  Tensor info: {type(output_tensor) if 'output_tensor' in locals() else 'not available'}")
                            if 'output_tensor' in locals():
                                logger.error(f"  Tensor shape: {output_tensor.shape}")
                        failed_count += 1
                
                # Count any remaining frames that couldn't be processed due to batch failure
                if actual_outputs < len(valid_files):
                    unprocessed_count = len(valid_files) - actual_outputs
                    failed_count += unprocessed_count
                    if logger:
                        logger.warning(f"Could not process {unprocessed_count} frames due to batch processing failure")
                
                # Record batch processing time
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                batch_times.append(batch_duration)
                
                # Calculate ETA after the first 2 steps for more stable timing
                eta_str = ""
                if current_step >= 2:
                    # Use average of last 2 batches for more accurate estimation
                    recent_times = batch_times[-2:]
                    avg_batch_time = sum(recent_times) / len(recent_times)
                    remaining_batches = total_steps - current_step
                    eta_seconds = remaining_batches * avg_batch_time
                    
                    # Format ETA time
                    if eta_seconds >= 3600:  # More than 1 hour
                        eta_str = f" | ETA: {int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                    elif eta_seconds >= 60:  # More than 1 minute
                        eta_str = f" | ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    else:  # Less than 1 minute
                        eta_str = f" | ETA: {int(eta_seconds)}s"
                
                # Update progress
                if progress_callback:
                    # Guard against progress values going slightly over 1.0 due to rounding issues.
                    progress_value = (processed_count + failed_count) / total_frames if total_frames else 1.0
                    progress_value = min(progress_value, 1.0)

                    # Build a richer description including step/frame counts and ETA
                    progress_desc = (
                        f"Step {current_step}/{total_steps} | "
                        f"Frames {processed_count}/{total_frames} processed"
                        f"{eta_str}"
                    )

                    try:
                        progress_callback(progress_value, progress_desc)
                    except Exception as prog_e:
                        if logger:
                            logger.debug(f"Non-critical progress callback error ignored: {prog_e}")
                
                # Console logging for CLI users
                if logger:
                    remaining_steps = max(total_steps - current_step, 0)
                    remaining_frames = max(total_frames - processed_count, 0)
                    batch_time_str = f"({batch_duration:.2f}s/batch)"
                    
                    log_msg = (
                        f"Image Upscaler Progress — Step {current_step}/{total_steps} complete {batch_time_str}. "
                        f"Processed frames: {processed_count}/{total_frames}. "
                        f"Remaining steps: {remaining_steps}, Remaining frames: {remaining_frames}."
                    )
                    
                    # Add ETA to log message as well
                    if eta_str:
                        log_msg += f" {eta_str.strip(' |')}"
                    
                    logger.info(log_msg)
                
                # Check for cancellation after each batch completion
                try:
                    cancellation_manager.check_cancel("image upscaler - after batch")
                except CancelledError:
                    if logger:
                        logger.info(f"Image upscaling cancelled after completing step {current_step}")
                    # Return current progress with partial results
                    return processed_count, failed_count + (total_frames - processed_count - failed_count)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if logger:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Only the frames that have *not* already been marked as processed in this
                # batch should be added to the failed counter to avoid double-counting.
                failed_in_this_batch = len(batch_files) - processed_in_current_batch
                if failed_in_this_batch < 0:
                    failed_in_this_batch = 0  # sanity safeguard
                failed_count += failed_in_this_batch
    
    if logger:
        total_processing_time = time.time() - processing_start_time
        logger.info(f"Frame processing complete: {processed_count} processed, {failed_count} failed in {total_processing_time:.2f}s")
    
    return processed_count, failed_count

def get_model_scale_factor(model: Any) -> int:
    """
    Get the scale factor of a loaded model.
    
    Args:
        model: Loaded upscaler model
        
    Returns:
        Scale factor (e.g., 2, 4) or 1 if unknown
    """
    try:
        return getattr(model, 'scale', 1)
    except:
        return 1

def get_model_architecture(model: Any) -> str:
    """
    Get the architecture name of a loaded model.
    
    Args:
        model: Loaded upscaler model
        
    Returns:
        Architecture name or 'Unknown'
    """
    try:
        if hasattr(model, 'architecture') and hasattr(model.architecture, 'name'):
            return model.architecture.name
        return getattr(model, 'architecture', 'Unknown')
    except:
        return 'Unknown'

def estimate_output_resolution(input_width: int, input_height: int, model: Any) -> Tuple[int, int]:
    """
    Estimate the output resolution after upscaling.
    
    Args:
        input_width: Input width
        input_height: Input height
        model: Loaded upscaler model
        
    Returns:
        Tuple of (output_width, output_height)
    """
    scale_factor = get_model_scale_factor(model)
    return input_width * scale_factor, input_height * scale_factor

def validate_model_compatibility(model_path: str, logger: logging.Logger = None) -> Tuple[bool, str]:
    """
    Validate if a model file is compatible with the upscaler system.
    
    Args:
        model_path: Path to the model file
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    
    # Check file extension
    if not any(model_path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        return False, f"Unsupported file format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    
    try:
        # Try to load model info
        info = get_model_info(model_path, logger)
        if "error" in info:
            return False, f"Model loading failed: {info['error']}"
        
        # Check if scale factor is valid
        scale = info.get('scale', 'Unknown')
        if scale == 'Unknown' or not isinstance(scale, (int, float)) or scale <= 1:
            if logger:
                logger.warning(f"Model {os.path.basename(model_path)} has unusual scale factor: {scale}")
        
        return True, "Model is compatible"
        
    except Exception as e:
        return False, f"Model validation failed: {str(e)}"

def get_recommended_batch_size(model: Any, device: str = "cuda") -> int:
    """
    Get a recommended batch size based on model and device capabilities.
    
    Args:
        model: Loaded upscaler model
        device: Device being used
        
    Returns:
        Recommended batch size
    """
    if device == "cpu":
        return 1  # CPU processing is typically single-threaded
    
    if not torch.cuda.is_available():
        return 1
    
    try:
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / (1024**3)
        
        # Basic heuristic based on memory
        if memory_gb >= 24:
            return 8
        elif memory_gb >= 16:
            return 6
        elif memory_gb >= 12:
            return 4
        elif memory_gb >= 8:
            return 2
        else:
            return 1
            
    except:
        return 4  # Safe default 

def process_single_image_direct(
    image_path: str,
    output_path: str,
    model_name: str,
    upscale_models_dir: str,
    apply_target_resolution: bool = False,
    target_h: int = 1024,
    target_w: int = 1024,
    target_res_mode: str = "Ratio Upscale",
    device: str = "cuda",
    logger: logging.Logger = None
) -> Tuple[bool, Optional[str], float]:
    """
    Process a single image file directly with an image upscaler model.
    
    Args:
        image_path: Path to the input image file
        output_path: Path where the upscaled image should be saved
        model_name: Name of the upscaler model to use
        upscale_models_dir: Directory containing upscaler models
        apply_target_resolution: Whether to apply target resolution constraints
        target_h: Target height (if apply_target_resolution=True)
        target_w: Target width (if apply_target_resolution=True)
        target_res_mode: How to apply target resolution ("Ratio Upscale" or "Downscale then Upscale")
        device: Device to use for processing
        logger: Logger instance
        
    Returns:
        Tuple of (success, result_path, processing_time)
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not os.path.exists(image_path):
            if logger:
                logger.error(f"Input image not found: {image_path}")
            return False, None, 0.0
        
        if not model_name:
            if logger:
                logger.error("No model name provided")
            return False, None, 0.0
        
        # Build model path
        model_path = os.path.join(upscale_models_dir, model_name)
        if not os.path.exists(model_path):
            if logger:
                logger.error(f"Model not found: {model_path}")
            return False, None, 0.0
        
        # Load the model
        model = load_model(model_path, device, logger)
        if model is None:
            if logger:
                logger.error(f"Failed to load model: {model_name}")
            return False, None, 0.0
        
        # Load the input image
        try:
            input_image = Image.open(image_path).convert("RGB")
            input_w, input_h = input_image.size
            
            if logger:
                logger.info(f"Processing image {os.path.basename(image_path)}: {input_w}x{input_h}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to load image {image_path}: {e}")
            return False, None, time.time() - start_time
        
        # Convert to tensor
        input_tensor = prepare_image_tensor(image_path, device)
        
        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0)  # CHW -> NCHW
        
        # Run upscaling
        with torch.no_grad():
            output_batch = model(input_batch)
        
        if output_batch is None:
            if logger:
                logger.error("Model returned None output")
            return False, None, time.time() - start_time
        
        # Convert back to image
        output_image = tensor_to_pil(output_batch)
        output_w, output_h = output_image.size
        
        # Apply target resolution constraints if enabled
        final_image = output_image
        if apply_target_resolution:
            if target_res_mode == "Ratio Upscale":
                # Scale down proportionally if output exceeds target
                if output_w > target_w or output_h > target_h:
                    # Calculate scale to fit within target bounds
                    scale_w = target_w / output_w
                    scale_h = target_h / output_h
                    scale = min(scale_w, scale_h)
                    
                    new_w = int(output_w * scale)
                    new_h = int(output_h * scale)
                    
                    final_image = output_image.resize((new_w, new_h), Image.LANCZOS)
                    
                    if logger:
                        logger.info(f"Applied ratio upscale: {output_w}x{output_h} -> {new_w}x{new_h}")
            
            elif target_res_mode == "Downscale then Upscale":
                # Resize to target resolution exactly
                final_image = output_image.resize((target_w, target_h), Image.LANCZOS)
                
                if logger:
                    logger.info(f"Applied downscale then upscale: {output_w}x{output_h} -> {target_w}x{target_h}")
        
        # Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_image.save(output_path, quality=95, optimize=True)
        
        processing_time = time.time() - start_time
        final_w, final_h = final_image.size
        
        if logger:
            logger.info(f"Successfully processed {os.path.basename(image_path)}: {input_w}x{input_h} -> {final_w}x{final_h} in {processing_time:.2f}s")
        
        return True, output_path, processing_time
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing image {image_path}: {e}")
        return False, None, time.time() - start_time

def extract_model_filename_from_dropdown(dropdown_value: str) -> str:
    """
    Extract the actual model filename from the dropdown display value.
    
    Args:
        dropdown_value: The display value from the dropdown (e.g., "4x-UltraSharp (ESRGAN - 4x)")
        
    Returns:
        The actual model filename (e.g., "4x-UltraSharp") or None if invalid
    """
    if not dropdown_value or "No models found" in dropdown_value or "Error scanning" in dropdown_value:
        return None
    
    # Extract filename before the first parenthesis
    if "(" in dropdown_value:
        return dropdown_value.split("(")[0].strip()
    
    return dropdown_value.strip()