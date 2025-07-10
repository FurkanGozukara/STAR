import os
import time
import math
import shutil
import cv2
import torch
import numpy as np
import gradio as gr
from .gpu_utils import get_gpu_device
from .ffmpeg_utils import extract_frames, create_video_from_frames
from .common_utils import format_time
from .cogvlm_utils import auto_caption, COG_VLM_AVAILABLE

def calculate_upscale_params(orig_h, orig_w, target_h, target_w, target_res_mode, logger=None, image_upscaler_model=None):
    """Calculate upscaling parameters based on target resolution mode."""
    final_h = int(target_h)
    final_w = int(target_w)
    needs_downscale = False
    downscale_h, downscale_w = orig_h, orig_w

    if target_res_mode == 'Downscale then 4x':
        # Determine actual upscale factor based on model type
        if image_upscaler_model is not None:
            # Get scale factor from image upscaler model
            from .image_upscaler_utils import get_model_scale_factor, get_model_info, scan_for_models
            from .config import UPSCALE_MODELS_DIR
            
            # Get model path and info
            models_list = scan_for_models(UPSCALE_MODELS_DIR, logger=logger)
            model_path = None
            for model_file in models_list:
                if model_file == image_upscaler_model:
                    model_path = os.path.join(UPSCALE_MODELS_DIR, model_file)
                    break
            
            if model_path and os.path.exists(model_path):
                model_info = get_model_info(model_path, logger)
                actual_upscale_factor = float(model_info.get("scale", 4))
                model_type = "image upscaler"
            else:
                actual_upscale_factor = 4.0  # Fallback
                model_type = "STAR (fallback)"
                if logger:
                    logger.warning(f"Could not determine image upscaler scale, using 4x fallback")
        else:
            actual_upscale_factor = 4.0  # STAR model default
            model_type = "STAR"
        
        # Calculate optimal final resolution that fits within the constraint box
        # while maintaining the original aspect ratio
        if orig_h == 0 or orig_w == 0:
            raise ValueError("Original dimensions cannot be zero.")
        
        aspect_ratio = orig_w / orig_h
        
        # Find the largest resolution that fits within the constraint box (final_w x final_h)
        # while maintaining the aspect ratio
        if aspect_ratio >= 1.0:
            # Wide or square aspect ratio - width is the limiting factor
            optimal_final_w = final_w
            optimal_final_h = final_w / aspect_ratio
            if optimal_final_h > final_h:
                # Height exceeds constraint, use height as limiting factor
                optimal_final_h = final_h
                optimal_final_w = final_h * aspect_ratio
        else:
            # Tall aspect ratio - height is the limiting factor
            optimal_final_h = final_h
            optimal_final_w = final_h * aspect_ratio
            if optimal_final_w > final_w:
                # Width exceeds constraint, use width as limiting factor
                optimal_final_w = final_w
                optimal_final_h = final_w / aspect_ratio
        
        # Round to even dimensions
        optimal_final_h = int(round(optimal_final_h / 2) * 2)
        optimal_final_w = int(round(optimal_final_w / 2) * 2)
        
        # **FIX: Always respect the constraint box (final_w x final_h)**
        # Calculate what the direct upscale would be
        direct_upscale_h = int(round(orig_h * actual_upscale_factor / 2) * 2)
        direct_upscale_w = int(round(orig_w * actual_upscale_factor / 2) * 2)
        
        # Check if direct upscale fits within the constraint box
        if direct_upscale_w <= optimal_final_w and direct_upscale_h <= optimal_final_h:
            # Direct upscale fits within constraint box - use it
            needs_downscale = False
            downscale_h, downscale_w = orig_h, orig_w
            final_h = direct_upscale_h
            final_w = direct_upscale_w
            
            if logger:
                logger.info(f"Original video {orig_w}x{orig_h} can be directly upscaled {actual_upscale_factor}x to {final_w}x{final_h} within constraint {optimal_final_w}x{optimal_final_h}.")
                logger.info(f"Target resolution mode: Direct {actual_upscale_factor}x upscale. Calculated upscale: {actual_upscale_factor:.2f}x. Target output: {final_w}x{final_h}")
        else:
            # Direct upscale exceeds constraint box - use optimal resolution within constraint
            # Calculate what the intermediate (pre-upscale) resolution should be
            intermediate_h = optimal_final_h / actual_upscale_factor
            intermediate_w = optimal_final_w / actual_upscale_factor
            
            # Check if we need to downscale the original to reach the intermediate resolution
            if orig_h > intermediate_h or orig_w > intermediate_w:
                needs_downscale = True
                # Calculate the ratio needed to fit within intermediate resolution while maintaining aspect ratio
                ratio = min(intermediate_h / orig_h, intermediate_w / orig_w)
                downscale_h = int(round(orig_h * ratio / 2) * 2)
                downscale_w = int(round(orig_w * ratio / 2) * 2)
                if logger:
                    logger.info(f"Downscaling required: {orig_w}x{orig_h} -> {downscale_w}x{downscale_h} for {actual_upscale_factor}x {model_type} target.")
            else:
                # No downscaling needed - use original dimensions
                needs_downscale = False
                downscale_h, downscale_w = orig_h, orig_w
                if logger:
                    logger.info(f"No downscaling needed for 'Downscale then {actual_upscale_factor}x' mode with {model_type}. Using original resolution: {orig_w}x{orig_h}.")
            
            # Use the optimal resolution that fits within the constraint box
            final_h = optimal_final_h
            final_w = optimal_final_w
            
            if logger:
                logger.info(f"Direct {actual_upscale_factor}x upscale would exceed constraint box. Using optimal resolution within constraint.")
                logger.info(f"Target resolution mode: Downscale then {actual_upscale_factor}x. Calculated upscale: {actual_upscale_factor:.2f}x. Target output: {final_w}x{final_h}")
        
        final_upscale_factor = actual_upscale_factor
        
        if logger:
            orig_pixels = orig_h * orig_w
            final_pixels = final_h * final_w
            max_pixel_budget = target_h * target_w
            logger.info(f"Pixel budget: {max_pixel_budget:,} pixels, Final output: {final_pixels:,} pixels ({final_pixels/max_pixel_budget*100:.1f}% of budget)")
            logger.info(f"Aspect ratio preserved: {orig_w/orig_h:.3f} -> {final_w/final_h:.3f}")
            if target_w == target_h:
                logger.info(f"Square target resolution {target_w}x{target_h} detected. Aspect ratio maintained by using larger dimension as constraint.")

    elif target_res_mode == 'Ratio Upscale':
        if orig_h == 0 or orig_w == 0:
            raise ValueError("Original dimensions cannot be zero.")
        ratio_h = final_h / orig_h
        ratio_w = final_w / orig_w
        final_upscale_factor = min(ratio_h, ratio_w)
        final_h = int(round(orig_h * final_upscale_factor / 2) * 2)
        final_w = int(round(orig_w * final_upscale_factor / 2) * 2)
        if logger:
            logger.info(f"Ratio Upscale mode: Using upscale factor {final_upscale_factor:.2f}")
    else:
        raise ValueError(f"Invalid target_res_mode: {target_res_mode}")

    if logger:
        logger.info(f"Calculated final target resolution: {final_w}x{final_h} with upscale {final_upscale_factor:.2f}")
    return needs_downscale, downscale_h, downscale_w, final_upscale_factor, final_h, final_w 