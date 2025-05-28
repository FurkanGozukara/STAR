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

def calculate_upscale_params(orig_h, orig_w, target_h, target_w, target_res_mode, logger=None):
    """Calculate upscaling parameters based on target resolution mode."""
    final_h = int(target_h)
    final_w = int(target_w)
    needs_downscale = False
    downscale_h, downscale_w = orig_h, orig_w

    if target_res_mode == 'Downscale then 4x':
        intermediate_h = final_h / 4.0
        intermediate_w = final_w / 4.0
        if orig_h > intermediate_h or orig_w > intermediate_w:
            needs_downscale = True
            ratio = min(intermediate_h / orig_h, intermediate_w / orig_w)
            downscale_h = int(round(orig_h * ratio / 2) * 2)
            downscale_w = int(round(orig_w * ratio / 2) * 2)
            if logger:
                logger.info(f"Downscaling required: {orig_h}x{orig_w} -> {downscale_w}x{downscale_h} for 4x target.")
        else:
            if logger:
                logger.info("No downscaling needed for 'Downscale then 4x' mode.")
        final_upscale_factor = 4.0

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