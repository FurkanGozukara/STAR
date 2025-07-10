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
        
        # Calculate pixel budget from target resolution
        pixel_budget = final_h * final_w
        
        # Calculate what the final resolution should be to fit the pixel budget while maintaining aspect ratio
        # This is the same logic as "Ratio Upscale" but with pixel budget constraint
        if orig_h == 0 or orig_w == 0:
            raise ValueError("Original dimensions cannot be zero.")
        
        aspect_ratio = orig_w / orig_h
        
        # Calculate optimal final resolution within pixel budget
        optimal_final_h = math.sqrt(pixel_budget / aspect_ratio)
        optimal_final_w = optimal_final_h * aspect_ratio
        
        # Round to even dimensions
        optimal_final_h = int(round(optimal_final_h / 2) * 2)
        optimal_final_w = int(round(optimal_final_w / 2) * 2)
        
        # Verify we're still within pixel budget after rounding
        if optimal_final_h * optimal_final_w > pixel_budget:
            # If rounding up exceeded budget, try rounding down
            optimal_final_h = int(optimal_final_h / 2) * 2
            optimal_final_w = int(optimal_final_w / 2) * 2
        
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
                logger.info(f"Downscaling required: {orig_h}x{orig_w} -> {downscale_w}x{downscale_h} for {actual_upscale_factor}x {model_type} target.")
        else:
            if logger:
                logger.info(f"No downscaling needed for 'Downscale then {actual_upscale_factor}x' mode with {model_type}.")
        
        final_upscale_factor = actual_upscale_factor

        # Calculate final dimensions from the downscaled/original dimensions
        final_h = int(round(downscale_h * final_upscale_factor / 2) * 2)
        final_w = int(round(downscale_w * final_upscale_factor / 2) * 2)
        
        if logger:
            orig_pixels = orig_h * orig_w
            final_pixels = final_h * final_w
            logger.info(f"Pixel budget: {pixel_budget:,} pixels, Final output: {final_pixels:,} pixels ({final_pixels/pixel_budget*100:.1f}% of budget)")
            logger.info(f"Aspect ratio preserved: {orig_w/orig_h:.3f} -> {final_w/final_h:.3f}")

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