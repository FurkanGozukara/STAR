"""
SeedVR2 Image Core Processing Module

This module handles single image upscaling using both SeedVR2 models 
and traditional image-based upscalers, providing a unified interface
for the image upscaling tab.

Key Features:
- Single image processing with SeedVR2 models
- Integration with existing image upscaler pipeline  
- Before/after comparison generation
- Multiple output formats (PNG, JPEG, WEBP)
- Metadata preservation
- Progress tracking and cancellation support
"""

import os
import sys
import time
import tempfile
import shutil
import math
import numpy as np
import cv2
import gc
import logging
from typing import List, Dict, Tuple, Optional, Any, Generator
from pathlib import Path
import torch
from PIL import Image, ImageOps
import json

# Import cancellation manager
from .cancellation_manager import cancellation_manager, CancelledError
from .common_utils import format_time

# Import the unified resolution calculation function
from .seedvr2_resolution_utils import calculate_seedvr2_resolution

def process_single_image(
    input_image_path: str,
    upscaler_type: str,  # "seedvr2" or "image_upscaler"
    
    # SeedVR2 configuration (if using SeedVR2)
    seedvr2_config = None,
    
    # Image upscaler configuration (if using image upscalers)
    image_upscaler_model: str = None,
    
    # Output settings
    output_format: str = "PNG",
    output_quality: int = 95,
    preserve_aspect_ratio: bool = True,
    preserve_metadata: bool = True,
    custom_suffix: str = "_upscaled",
    
    # Comparison settings
    create_comparison: bool = True,
    
    # Processing settings
    output_dir: str = None,
    
    # Dependencies
    logger: logging.Logger = None,
    progress = None,
    
    # Utility functions (injected dependencies)
    util_get_gpu_device = None,
    format_time = None,
    
    # Status tracking
    current_seed: int = 99
) -> Generator[Tuple[Optional[str], Optional[str], str, Dict[str, Any]], None, None]:
    """
    Process a single image using SeedVR2 or image-based upscalers.
    
    Args:
        input_image_path: Path to input image
        upscaler_type: Type of upscaler ("seedvr2" or "image_upscaler")
        seedvr2_config: SeedVR2Config object (if using SeedVR2)
        image_upscaler_model: Image upscaler model name (if using image upscalers)
        output_format: Output format (PNG, JPEG, WEBP)
        output_quality: Quality for JPEG/WEBP (70-100)
        preserve_aspect_ratio: Whether to preserve aspect ratio
        preserve_metadata: Whether to preserve original metadata
        custom_suffix: Custom suffix for output filename
        create_comparison: Whether to create before/after comparison
        output_dir: Output directory (uses default if None)
        logger: Logger instance
        progress: Progress callback
        util_get_gpu_device: GPU utility function
        format_time: Time formatting function
        current_seed: Seed for processing
        
    Yields:
        Tuple of (output_image_path, comparison_image_path, status_message, image_info)
    """
    
    # Check for cancellation at start
    cancellation_manager.check_cancel()
    
    if not input_image_path or not os.path.exists(input_image_path):
        raise ValueError("Please select a valid input image file.")
    
    process_start_time = time.time()
    if logger:
        logger.info(f"Starting image upscaling: {upscaler_type}")
        logger.info(f"Input image: {input_image_path}")
    
    # Validate input image
    try:
        with Image.open(input_image_path) as img:
            original_width, original_height = img.size
            original_format = img.format
            original_mode = img.mode
            
        if logger:
            logger.info(f"Input image: {original_width}x{original_height}, format: {original_format}, mode: {original_mode}")
    except Exception as e:
        error_msg = f"Invalid image file: {e}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs_images')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    input_path = Path(input_image_path)
    base_name = input_path.stem
    
    # Determine output extension
    if output_format.upper() == "PNG":
        output_ext = ".png"
    elif output_format.upper() == "JPEG":
        output_ext = ".jpg"
    elif output_format.upper() == "WEBP":
        output_ext = ".webp"
    else:
        output_ext = ".png"  # Default fallback
    
    # Generate output filename with auto-increment if file exists
    counter = 0
    output_filename = f"{base_name}{custom_suffix}{output_ext}"
    output_image_path = os.path.join(output_dir, output_filename)
    
    while os.path.exists(output_image_path):
        counter += 1
        output_filename = f"{base_name}{custom_suffix}_{counter:04d}{output_ext}"
        output_image_path = os.path.join(output_dir, output_filename)
    
    # Generate comparison filename if needed
    comparison_image_path = None
    if create_comparison:
        comparison_filename = f"{base_name}_comparison{output_ext}"
        comparison_image_path = os.path.join(output_dir, comparison_filename)
        
        # Also apply auto-increment to comparison file
        comparison_counter = 0
        while os.path.exists(comparison_image_path):
            comparison_counter += 1
            comparison_filename = f"{base_name}_comparison_{comparison_counter:04d}{output_ext}"
            comparison_image_path = os.path.join(output_dir, comparison_filename)
    
    if progress:
        progress(0.1, "🔍 Analyzing input image...")
    
    # Extract image information
    image_info = _extract_image_info(input_image_path, logger)
    
    cancellation_manager.check_cancel()
    
    if progress:
        progress(0.2, "🚀 Initializing upscaler...")
    
    # Process based on upscaler type
    try:
        if upscaler_type == "seedvr2":
            if not seedvr2_config:
                raise ValueError("SeedVR2 configuration required for SeedVR2 upscaling")
            
            processed_image_array = _process_with_seedvr2(
                input_image_path, seedvr2_config, logger, progress, current_seed
            )
        else:  # image_upscaler
            if not image_upscaler_model:
                raise ValueError("Image upscaler model required for image-based upscaling")
            
            processed_image_array = _process_with_image_upscaler(
                input_image_path, image_upscaler_model, logger, progress
            )
        
        if logger:
            logger.info(f"Image processing completed successfully")
            
    except Exception as e:
        error_msg = f"Image processing failed: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    cancellation_manager.check_cancel()
    
    if progress:
        progress(0.8, "💾 Saving processed image...")
    
    # Save processed image
    try:
        _save_processed_image(
            processed_image_array,
            output_image_path,
            output_format,
            output_quality,
            preserve_metadata,
            input_image_path,
            logger
        )
        
        if logger:
            logger.info(f"Processed image saved: {output_image_path}")
    
    except Exception as e:
        error_msg = f"Failed to save processed image: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    cancellation_manager.check_cancel()
    
    # Create comparison image if requested
    if create_comparison:
        if progress:
            progress(0.9, "📊 Creating before/after comparison...")
        
        try:
            _create_comparison_image(
                input_image_path,
                output_image_path,
                comparison_image_path,
                output_format,
                output_quality,
                logger
            )
            
            if logger:
                logger.info(f"Comparison image created: {comparison_image_path}")
        
        except Exception as e:
            if logger:
                logger.warning(f"Failed to create comparison image: {e}")
            comparison_image_path = None
    
    # Update image info with output details
    try:
        with Image.open(output_image_path) as output_img:
            output_width, output_height = output_img.size
            output_file_size = os.path.getsize(output_image_path)
            
        image_info.update({
            'output_width': output_width,
            'output_height': output_height,
            'output_file_size': output_file_size,
            'output_file_size_mb': round(output_file_size / (1024 * 1024), 2),
            'upscale_factor_x': output_width / image_info['width'],
            'upscale_factor_y': output_height / image_info['height'],
            'output_format': output_format,
            'processing_time': time.time() - process_start_time
        })
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to analyze output image: {e}")
    
    # Calculate total processing time
    total_time = time.time() - process_start_time
    final_status = f"✅ Image upscaling completed in {format_time(total_time) if format_time else f'{total_time:.1f}s'}"
    
    if logger:
        logger.info(final_status)
    
    # Final progress update
    if progress:
        progress(1.0, final_status)
    
    # Yield final result
    yield (output_image_path, comparison_image_path, final_status, image_info)


def _extract_image_info(image_path: str, logger=None) -> Dict[str, Any]:
    """Extract comprehensive information about the input image."""
    
    info = {
        'filename': os.path.basename(image_path),
        'file_size': 0,
        'file_size_mb': 0,
        'width': 0,
        'height': 0,
        'format': 'Unknown',
        'mode': 'Unknown',
        'has_transparency': False,
        'color_profile': None,
        'exif_data': {},
        'creation_time': None
    }
    
    try:
        # Get file size
        info['file_size'] = os.path.getsize(image_path)
        info['file_size_mb'] = round(info['file_size'] / (1024 * 1024), 2)
        
        # Get file creation time
        info['creation_time'] = time.ctime(os.path.getctime(image_path))
        
        # Open and analyze image
        with Image.open(image_path) as img:
            info['width'], info['height'] = img.size
            info['format'] = img.format or 'Unknown'
            info['mode'] = img.mode or 'Unknown'
            info['has_transparency'] = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            
            # Get color profile if available
            if hasattr(img, 'info') and 'icc_profile' in img.info:
                info['color_profile'] = 'ICC Profile Present'
            
            # Extract EXIF data if available
            if hasattr(img, '_getexif') and img._getexif is not None:
                exif_dict = img._getexif()
                if exif_dict:
                    info['exif_data'] = {str(k): str(v) for k, v in exif_dict.items()}
    
    except Exception as e:
        if logger:
            logger.warning(f"Error extracting image info: {e}")
    
    return info


def _process_with_seedvr2(
    input_image_path: str, 
    seedvr2_config, 
    logger=None, 
    progress=None, 
    current_seed=99
) -> np.ndarray:
    """Process image using SeedVR2 model."""
    
    # Add SeedVR2 project root to Python path for imports
    seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
    if seedvr2_base_path not in sys.path:
        sys.path.insert(0, seedvr2_base_path)
    
    try:
        # Import SeedVR2 CLI processing to avoid ComfyUI dependencies
        from .seedvr2_cli_core import _process_batch_with_seedvr2_model, SeedVR2BlockSwap, cleanup_global_session, force_cleanup_gpu_memory, destroy_global_session_completely, cleanup_vram_only
        
        if logger:
            logger.info("SeedVR2 CLI modules imported successfully")
            
    except ImportError as e:
        error_msg = f"Failed to import SeedVR2 CLI modules: {e}. Using fallback processing."
        if logger:
            logger.warning(error_msg)
        raise RuntimeError(error_msg)
    
    # Setup model path
    models_dir = os.path.join(seedvr2_base_path, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # Validate and download model if needed
    if not seedvr2_config.model:
        seedvr2_config.model = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    
    model_path = os.path.join(models_dir, seedvr2_config.model)
    if not os.path.exists(model_path):
        if logger:
            logger.info(f"Downloading SeedVR2 model: {seedvr2_config.model}")
        try:
            # Import download_weight from SeedVR2
            from src.utils.downloads import download_weight
            download_weight(seedvr2_config.model, models_dir)
        except ImportError as e:
            error_msg = f"Failed to import SeedVR2 download_weight: {e}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Failed to download model {seedvr2_config.model}: {e}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    if progress:
        progress(0.3, "📥 Loading image...")
    
    # Load and preprocess image
    image_tensor = _load_image_to_tensor(input_image_path, logger)
    
    if progress:
        progress(0.4, "🔧 Configuring SeedVR2 model...")
    
    # Setup CLI-based SeedVR2 processing parameters (matching video pipeline)
    processing_args = {
        "model": seedvr2_config.model,
        "model_dir": models_dir,
        "preserve_vram": getattr(seedvr2_config, 'preserve_vram', True),
        "debug": logger.level <= logging.DEBUG if logger else False,
        "cfg_scale": getattr(seedvr2_config, 'cfg_scale', 7.5),
        "seed": getattr(seedvr2_config, 'seed', current_seed) if getattr(seedvr2_config, 'seed', -1) >= 0 else current_seed,
        # Calculate resolution using the main function
        "res_w": calculate_seedvr2_resolution(
            input_path=input_image_path,
            enable_target_res=getattr(seedvr2_config, 'enable_target_res', False),
            target_h=getattr(seedvr2_config, 'target_h', 2048),
            target_w=getattr(seedvr2_config, 'target_w', 2048),
            target_res_mode=getattr(seedvr2_config, 'target_res_mode', 'Ratio Upscale'),
            upscale_factor=4.0,  # SeedVR2 default
            logger=logger
        ),
        "batch_size": getattr(seedvr2_config, 'batch_size', 1),
        "temporal_overlap": 0,  # No temporal processing for single image
        # Add missing features from video pipeline
        "flash_attention": getattr(seedvr2_config, 'flash_attention', False),
        "color_correction": getattr(seedvr2_config, 'color_correction', True),
        "quality_preset": getattr(seedvr2_config, 'quality_preset', 'high'),
        "model_precision": getattr(seedvr2_config, 'model_precision', 'auto'),
        # Multi-GPU support
        "enable_multi_gpu": getattr(seedvr2_config, 'enable_multi_gpu', False),
        "gpu_devices": getattr(seedvr2_config, 'gpu_devices', '0'),
        # Additional temporal consistency settings (for compatibility)
        "scene_awareness": getattr(seedvr2_config, 'scene_awareness', False),  # False for single image
        "temporal_quality": getattr(seedvr2_config, 'temporal_quality', 'balanced'),
        "consistency_validation": getattr(seedvr2_config, 'consistency_validation', False),  # False for single image
        "chunk_optimization": getattr(seedvr2_config, 'chunk_optimization', False),  # False for single image
        "enable_temporal_consistency": getattr(seedvr2_config, 'enable_temporal_consistency', False),  # False for single image
        "enable_frame_padding": getattr(seedvr2_config, 'enable_frame_padding', False),  # False for single image
        "pad_last_chunk": getattr(seedvr2_config, 'pad_last_chunk', False),  # False for single image
        # GPU/processing settings
        "use_gpu": getattr(seedvr2_config, 'use_gpu', True),
    }
    
    # Setup block swap if enabled (matching video pipeline)
    block_swap = None
    if getattr(seedvr2_config, 'enable_block_swap', False) and getattr(seedvr2_config, 'block_swap_counter', 0) > 0:
        # Pass force_enable=True to respect user's choice
        # Always enable debug for BlockSwap when it's active to show swap messages
        block_swap_debug = True  # Show BlockSwap messages when enabled
        block_swap = SeedVR2BlockSwap(
            enable_debug=block_swap_debug,
            force_enable=True  # Enable because user explicitly requested it
        )
        # Add block swap config to processing args
        processing_args["block_swap_config"] = {
            "blocks_to_swap": seedvr2_config.block_swap_counter,
            "offload_io_components": getattr(seedvr2_config, 'block_swap_offload_io', False),
            "use_non_blocking": True,
            "enable_debug": block_swap_debug  # Use same debug setting
        }
        if logger:
            logger.info(f"Block swap enabled for image processing: {seedvr2_config.block_swap_counter} blocks")
    
    if progress:
        progress(0.5, "🚀 Processing with SeedVR2...")
    
    try:
        # Add batch dimension for processing: [H, W, C] -> [1, H, W, C]
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Process with SeedVR2 CLI (pass processing_args which now includes block_swap_config)
        result_tensor = _process_batch_with_seedvr2_model(
            image_tensor,
            processing_args,
            seedvr2_config,
            block_swap
        )
        
        # Convert tensor back to numpy array
        result_array = result_tensor.cpu().numpy()
        
        # Handle single image result (remove batch dimension if present)
        if result_array.ndim == 4 and result_array.shape[0] == 1:
            result_array = result_array[0]  # Remove batch dimension: [1, H, W, C] -> [H, W, C]
        
        # Convert from 0-1 range to 0-255 range
        if result_array.max() <= 1.0:
            result_array = (result_array * 255).astype(np.uint8)
        
        if logger:
            logger.info(f"SeedVR2 image processing completed: {result_array.shape}")
        
        # Cleanup based on preserve_vram setting
        # When preserve_vram is True, we want to save VRAM by cleaning up after processing
        if getattr(seedvr2_config, 'preserve_vram', True):
            try:
                if logger:
                    logger.info("🧹 Cleaning up VRAM after image processing (preserve_vram=True)...")
                # Use VRAM-only cleanup to keep models for reuse
                cleanup_vram_only()
                force_cleanup_gpu_memory()
                if logger:
                    logger.info("✅ VRAM cleaned, models kept for reuse")
            except Exception as cleanup_error:
                if logger:
                    logger.warning(f"Failed to cleanup VRAM: {cleanup_error}")
        else:
            if logger:
                logger.info("💾 Keeping SeedVR2 session loaded (preserve_vram=False)")
        
        return result_array
        
    except Exception as e:
        if logger:
            logger.error(f"SeedVR2 image processing failed: {e}")
        # Try to cleanup on error
        try:
            cleanup_global_session()
            force_cleanup_gpu_memory()
        except:
            pass
        raise


def _process_with_image_upscaler(
    input_image_path: str, 
    model_name: str, 
    logger=None, 
    progress=None
) -> np.ndarray:
    """Process image using traditional image upscaler."""
    
    # Import image upscaler utilities
    from .image_upscaler_utils import (
        load_model, process_frames_batch, extract_model_filename_from_dropdown
    )
    
    if progress:
        progress(0.3, "🔧 Loading image upscaler model...")
    
    # Extract actual model filename if needed
    actual_model_filename = extract_model_filename_from_dropdown(model_name)
    if not actual_model_filename:
        actual_model_filename = model_name
    
    # Get upscale models directory
    upscale_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'upscale_models')
    
    try:
        # Load the upscaler model
        model, device = load_model(actual_model_filename, upscale_models_dir, logger=logger)
        
        if logger:
            logger.info(f"Image upscaler model loaded: {actual_model_filename}")
    
    except Exception as e:
        error_msg = f"Failed to load image upscaler model: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if progress:
        progress(0.4, "📥 Loading and preprocessing image...")
    
    # Load image
    try:
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_image_path}")
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare batch (single image)
        image_batch = [image_rgb]
        
    except Exception as e:
        error_msg = f"Failed to load input image: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if progress:
        progress(0.5, "🚀 Processing with image upscaler...")
    
    try:
        # Process the image batch
        processed_batch = process_frames_batch(
            image_batch, model, device, batch_size=1, logger=logger
        )
        
        if not processed_batch:
            raise ValueError("No processed images returned from upscaler")
        
        # Get the processed image (first and only item in batch)
        processed_image = processed_batch[0]
        
        return processed_image
        
    except Exception as e:
        if logger:
            logger.error(f"Image upscaler processing failed: {e}")
        raise
    
    finally:
        # Clean up model to free VRAM
        try:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
        except:
            pass


def _load_image_to_tensor(image_path: str, logger=None) -> torch.Tensor:
    """Load image file into tensor format compatible with SeedVR2."""
    
    try:
        # Load image using OpenCV (BGR format)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor format expected by SeedVR2: [T, H, W, C] with values normalized to 0-1
        # For single image, T=1
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch/time dimension: [H, W, C] -> [1, H, W, C]
        
        # Convert to FP16 for memory efficiency
        image_tensor = image_tensor.to(torch.float16)
        
        return image_tensor
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to load image to tensor: {e}")
        raise


def _save_processed_image(
    image_array: np.ndarray, 
    output_path: str, 
    output_format: str, 
    output_quality: int,
    preserve_metadata: bool,
    original_image_path: str,
    logger=None
):
    """Save processed image array to file with specified format and quality."""
    
    try:
        # Ensure image array is in correct format
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert numpy array to PIL Image
        if image_array.ndim == 3:
            if image_array.shape[2] == 3:  # RGB
                pil_image = Image.fromarray(image_array, 'RGB')
            elif image_array.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(image_array, 'RGBA')
            else:
                raise ValueError(f"Unsupported image shape: {image_array.shape}")
        else:
            raise ValueError(f"Unsupported image dimensions: {image_array.ndim}")
        
        # Preserve metadata if requested
        save_kwargs = {}
        if preserve_metadata:
            try:
                with Image.open(original_image_path) as original_img:
                    if hasattr(original_img, 'info'):
                        save_kwargs['exif'] = original_img.info.get('exif')
                        save_kwargs['icc_profile'] = original_img.info.get('icc_profile')
            except Exception as e:
                if logger:
                    logger.warning(f"Could not preserve metadata: {e}")
        
        # Set format-specific parameters
        if output_format.upper() == "JPEG":
            save_kwargs['quality'] = output_quality
            save_kwargs['optimize'] = True
            # Convert RGBA to RGB for JPEG
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
        elif output_format.upper() == "WEBP":
            save_kwargs['quality'] = output_quality
            save_kwargs['method'] = 6  # Best compression method
        elif output_format.upper() == "PNG":
            save_kwargs['optimize'] = True
            save_kwargs['compress_level'] = 6  # Good balance of speed vs compression
        
        # Save the image
        pil_image.save(output_path, format=output_format.upper(), **save_kwargs)
        
        if logger:
            logger.info(f"Image saved successfully: {output_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to save processed image: {e}")
        raise


def _create_comparison_image(
    original_path: str,
    processed_path: str,
    comparison_path: str,
    output_format: str,
    output_quality: int,
    logger=None
):
    """Create a side-by-side before/after comparison image."""
    
    try:
        # Load both images
        with Image.open(original_path) as original_img:
            with Image.open(processed_path) as processed_img:
                
                # Get dimensions
                orig_width, orig_height = original_img.size
                proc_width, proc_height = processed_img.size
                
                # Calculate scale factor to make heights match
                if orig_height != proc_height:
                    scale_factor = orig_height / proc_height
                    new_proc_width = int(proc_width * scale_factor)
                    processed_img = processed_img.resize((new_proc_width, orig_height), Image.Resampling.LANCZOS)
                    proc_width = new_proc_width
                
                # Create comparison image (side by side)
                comparison_width = orig_width + proc_width + 20  # 20px gap
                comparison_height = max(orig_height, proc_height)
                
                # Create new image with white background
                comparison_img = Image.new('RGB', (comparison_width, comparison_height), 'white')
                
                # Paste original image (left side)
                comparison_img.paste(original_img, (0, 0))
                
                # Paste processed image (right side)
                comparison_img.paste(processed_img, (orig_width + 20, 0))
                
                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comparison_img)
                
                # Try to use a nice font, fall back to default if not available
                try:
                    font_size = max(20, min(orig_height // 20, 40))
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                if font:
                    # Add "Original" label
                    draw.text((10, 10), "Original", fill='black', font=font)
                    
                    # Add "Upscaled" label
                    draw.text((orig_width + 30, 10), "Upscaled", fill='black', font=font)
                
                # Save comparison image
                save_kwargs = {}
                if output_format.upper() == "JPEG":
                    save_kwargs['quality'] = output_quality
                    save_kwargs['optimize'] = True
                elif output_format.upper() == "WEBP":
                    save_kwargs['quality'] = output_quality
                    save_kwargs['method'] = 6
                elif output_format.upper() == "PNG":
                    save_kwargs['optimize'] = True
                    save_kwargs['compress_level'] = 6
                
                comparison_img.save(comparison_path, format=output_format.upper(), **save_kwargs)
                
                if logger:
                    logger.info(f"Comparison image created: {comparison_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to create comparison image: {e}")
        raise


def format_image_info_display(image_info: Dict[str, Any]) -> str:
    """Format image information for display in UI."""
    
    if not image_info:
        return "No image information available"
    
    display_text = f"""**📁 File Information:**
• Filename: {image_info.get('filename', 'Unknown')}
• File Size: {image_info.get('file_size_mb', 0):.2f} MB
• Created: {image_info.get('creation_time', 'Unknown')}

**📊 Image Properties:**
• Dimensions: {image_info.get('width', 0)} × {image_info.get('height', 0)} pixels
• Format: {image_info.get('format', 'Unknown')}
• Color Mode: {image_info.get('mode', 'Unknown')}
• Transparency: {'Yes' if image_info.get('has_transparency', False) else 'No'}
• Color Profile: {image_info.get('color_profile', 'None')}"""

    # Add output information if available
    if 'output_width' in image_info:
        display_text += f"""

**🚀 Processing Results:**
• Output Dimensions: {image_info.get('output_width', 0)} × {image_info.get('output_height', 0)} pixels
• Upscale Factor: {image_info.get('upscale_factor_x', 1):.2f}x × {image_info.get('upscale_factor_y', 1):.2f}x
• Output Format: {image_info.get('output_format', 'Unknown')}
• Output Size: {image_info.get('output_file_size_mb', 0):.2f} MB
• Processing Time: {image_info.get('processing_time', 0):.2f} seconds"""

    # Add EXIF data if available
    if image_info.get('exif_data'):
        display_text += f"""

**📷 EXIF Data:**
• {len(image_info['exif_data'])} metadata entries found"""

    return display_text 