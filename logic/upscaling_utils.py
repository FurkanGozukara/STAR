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
        
        intermediate_h = final_h / actual_upscale_factor
        intermediate_w = final_w / actual_upscale_factor
        if orig_h > intermediate_h or orig_w > intermediate_w:
            needs_downscale = True
            ratio = min(intermediate_h / orig_h, intermediate_w / orig_w)
            downscale_h = int(round(orig_h * ratio / 2) * 2)
            downscale_w = int(round(orig_w * ratio / 2) * 2)
            if logger:
                logger.info(f"Downscaling required: {orig_h}x{orig_w} -> {downscale_w}x{downscale_h} for {actual_upscale_factor}x {model_type} target.")
        else:
            if logger:
                logger.info(f"No downscaling needed for 'Downscale then {actual_upscale_factor}x' mode with {model_type}.")
        final_upscale_factor = actual_upscale_factor

        final_h = int(round(downscale_h * final_upscale_factor / 2) * 2)
        final_w = int(round(downscale_w * final_upscale_factor / 2) * 2)

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