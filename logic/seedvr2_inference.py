"""
SeedVR2 Inference Engine for STAR Application

This module handles SeedVR2 video and image upscaling without ComfyUI dependencies.
It provides CLI-compatible inference integrated with STAR's infrastructure.

Key Features:
- Video and image upscaling using SeedVR2 models
- CLI-compatible inference without ComfyUI
- Integration with STAR's progress tracking and configuration
- Temporal consistency and scene-aware processing
- Memory-efficient batch processing
"""

import os
import sys
import cv2
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

# Add SeedVR2 to path for imports
seedvr2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'SeedVR2')
if seedvr2_path not in sys.path:
    sys.path.insert(0, seedvr2_path)

# Import STAR utilities
from .seedvr2_model_manager import get_seedvr2_model_manager, SEEDVR2_AVAILABLE
from .ffmpeg_utils import get_video_info
from .star_dataclasses import AppConfig, DEFAULT_SEEDVR2_DEFAULT_RESOLUTION

# Import SeedVR2 components (now with fixed ComfyUI dependencies)
if SEEDVR2_AVAILABLE:
    try:
        from src.core.generation import generation_loop
        from src.utils.color_fix import wavelet_reconstruction
        import torch
        import torchvision.transforms as transforms
    except ImportError as e:
        print(f"Warning: Some SeedVR2 components not available: {e}")


class SeedVR2InferenceEngine:
    """
    SeedVR2 Inference Engine for STAR Application
    
    Handles video and image upscaling using SeedVR2 models with CLI compatibility.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_manager = get_seedvr2_model_manager(logger)
        self.current_config = None
        
    def extract_frames_from_video(self, video_path: str, skip_first_frames: int = 0, 
                                 load_cap: Optional[int] = None, debug: bool = False) -> torch.Tensor:
        """
        Extract frames from video and convert to tensor format
        
        Args:
            video_path: Path to input video
            skip_first_frames: Number of frames to skip from start
            load_cap: Maximum number of frames to load
            debug: Enable debug logging
            
        Returns:
            Tensor of video frames in format (T, C, H, W)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if debug:
            self.logger.info(f"Extracting frames from {video_path}, total frames: {total_frames}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip initial frames if specified
                if frame_count < skip_first_frames:
                    frame_count += 1
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor and normalize to [0, 1]
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
                
                frames.append(frame_tensor)
                frame_count += 1
                
                # Check load cap
                if load_cap and len(frames) >= load_cap:
                    break
                    
        finally:
            cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Stack frames into tensor (T, C, H, W)
        video_tensor = torch.stack(frames, dim=0)
        
        if debug:
            self.logger.info(f"Extracted {len(frames)} frames, tensor shape: {video_tensor.shape}")
        
        return video_tensor
    
    def extract_frames_from_image(self, image_path: str, debug: bool = False) -> torch.Tensor:
        """
        Extract single frame from image for upscaling
        
        Args:
            image_path: Path to input image
            debug: Enable debug logging
            
        Returns:
            Tensor of image in format (1, C, H, W) for consistency with video processing
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        # Add batch dimension for consistency with video processing
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        
        if debug:
            self.logger.info(f"Extracted image, tensor shape: {image_tensor.shape}")
        
        return image_tensor
    
    def upscale_video(self, video_path: str, output_path: str, config: AppConfig,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        Upscale a video using SeedVR2
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            config: Application configuration
            progress_callback: Optional progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        if not SEEDVR2_AVAILABLE:
            self.logger.error("SeedVR2 not available for video upscaling")
            return False
        
        try:
            # Extract configuration
            model_name = getattr(config, 'seedvr2_model', 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
            target_resolution = getattr(config, 'seedvr2_resolution', DEFAULT_SEEDVR2_DEFAULT_RESOLUTION)
            batch_size = getattr(config, 'seedvr2_batch_size', 5)
            preserve_vram = getattr(config, 'seedvr2_preserve_vram', True)
            seed = getattr(config, 'seed_num', 100)
            temporal_overlap = getattr(config, 'seedvr2_temporal_overlap', 0)
            
            # Block swap configuration
            block_swap_config = None
            blocks_to_swap = getattr(config, 'seedvr2_blocks_to_swap', 0)
            if blocks_to_swap > 0:
                block_swap_config = {
                    'blocks_to_swap': blocks_to_swap,
                    'offload_io_components': getattr(config, 'seedvr2_offload_io', False),
                    'cache_model': getattr(config, 'seedvr2_cache_model', False),
                    'use_non_blocking': True,
                    'enable_debug': getattr(config, 'debug_mode', False)
                }
            
            # Load model
            if not self.model_manager.load_model(model_name, preserve_vram, block_swap_config):
                return False
            
            # Extract frames
            self.logger.info("Extracting frames from video...")
            if progress_callback:
                progress_callback(0.1, "Extracting frames...")
            
            video_frames = self.extract_frames_from_video(
                video_path,
                skip_first_frames=0,
                load_cap=None,
                debug=getattr(config, 'debug_mode', False)
            )
            
            # Ensure batch size meets temporal consistency requirements
            if batch_size < 5:
                self.logger.warning("Batch size < 5, temporal consistency will be disabled")
            
            # Progress callback for generation
            def generation_progress_callback(batch_idx, total_batches, current_batch_frames, message=""):
                if progress_callback:
                    progress = 0.1 + 0.8 * (batch_idx / total_batches) if total_batches > 0 else 0.5
                    status_msg = f"Processing batch {batch_idx+1}/{total_batches}"
                    if message:
                        status_msg += f" - {message}"
                    progress_callback(progress, status_msg)
            
            # Perform upscaling
            self.logger.info("Starting SeedVR2 upscaling...")
            if progress_callback:
                progress_callback(0.1, "Starting upscaling...")
            
            # Extract tiled VAE settings
            tiled_vae = getattr(config, 'seedvr2_tiled_vae', False)
            tile_size = getattr(config, 'seedvr2_tile_size', (64, 64))
            tile_stride = getattr(config, 'seedvr2_tile_stride', (32, 32))
            
            upscaled_frames = generation_loop(
                self.model_manager.runner,
                video_frames,
                cfg_scale=1.0,
                seed=seed,
                res_w=target_resolution,
                batch_size=batch_size,
                preserve_vram=preserve_vram,
                temporal_overlap=temporal_overlap,
                debug=getattr(config, 'debug_mode', False),
                block_swap_config=block_swap_config,
                progress_callback=generation_progress_callback,
                tiled_vae=tiled_vae,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            
            # Apply color correction if enabled
            if getattr(config, 'seedvr2_color_correction', True):
                if progress_callback:
                    progress_callback(0.9, "Applying color correction...")
                upscaled_frames = wavelet_reconstruction(upscaled_frames, video_frames)
            
            # Save result
            if progress_callback:
                progress_callback(0.95, "Saving upscaled video...")
            
            success = self._save_video_result(
                upscaled_frames, 
                output_path, 
                video_path,
                config
            )
            
            if progress_callback:
                progress_callback(1.0, "Video upscaling completed!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to upscale video: {e}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            return False
    
    def upscale_image(self, image_path: str, output_path: str, config: AppConfig,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        Upscale a single image using SeedVR2
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            config: Application configuration
            progress_callback: Optional progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        if not SEEDVR2_AVAILABLE:
            self.logger.error("SeedVR2 not available for image upscaling")
            return False
        
        try:
            # Extract configuration
            model_name = getattr(config, 'seedvr2_model', 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
            target_resolution = getattr(config, 'seedvr2_resolution', DEFAULT_SEEDVR2_DEFAULT_RESOLUTION)
            preserve_vram = getattr(config, 'seedvr2_preserve_vram', True)
            seed = getattr(config, 'seed_num', 100)
            
            # For images, use batch_size=1 and no temporal features
            batch_size = 1
            temporal_overlap = 0
            
            # Block swap configuration
            block_swap_config = None
            blocks_to_swap = getattr(config, 'seedvr2_blocks_to_swap', 0)
            if blocks_to_swap > 0:
                block_swap_config = {
                    'blocks_to_swap': blocks_to_swap,
                    'offload_io_components': getattr(config, 'seedvr2_offload_io', False),
                    'cache_model': getattr(config, 'seedvr2_cache_model', False),
                    'use_non_blocking': True,
                    'enable_debug': getattr(config, 'debug_mode', False)
                }
            
            # Load model
            if not self.model_manager.load_model(model_name, preserve_vram, block_swap_config):
                return False
            
            # Extract image frame
            if progress_callback:
                progress_callback(0.1, "Loading image...")
            
            image_frame = self.extract_frames_from_image(
                image_path,
                debug=getattr(config, 'debug_mode', False)
            )
            
            # Perform upscaling
            self.logger.info("Starting SeedVR2 image upscaling...")
            if progress_callback:
                progress_callback(0.2, "Starting upscaling...")
            
            # Extract tiled VAE settings
            tiled_vae = getattr(config, 'seedvr2_tiled_vae', False)
            tile_size = getattr(config, 'seedvr2_tile_size', (64, 64))
            tile_stride = getattr(config, 'seedvr2_tile_stride', (32, 32))
            
            upscaled_frame = generation_loop(
                self.model_manager.runner,
                image_frame,
                cfg_scale=1.0,
                seed=seed,
                res_w=target_resolution,
                batch_size=batch_size,
                preserve_vram=preserve_vram,
                temporal_overlap=temporal_overlap,
                debug=getattr(config, 'debug_mode', False),
                block_swap_config=block_swap_config,
                progress_callback=lambda *args: None,  # No detailed progress for single image
                tiled_vae=tiled_vae,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            
            # Apply color correction if enabled
            if getattr(config, 'seedvr2_color_correction', True):
                if progress_callback:
                    progress_callback(0.9, "Applying color correction...")
                upscaled_frame = wavelet_reconstruction(upscaled_frame, image_frame)
            
            # Save result
            if progress_callback:
                progress_callback(0.95, "Saving upscaled image...")
            
            success = self._save_image_result(
                upscaled_frame,
                output_path,
                config
            )
            
            if progress_callback:
                progress_callback(1.0, "Image upscaling completed!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to upscale image: {e}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            return False
    
    def _save_video_result(self, upscaled_frames: torch.Tensor, output_path: str, 
                          original_video_path: str, config: AppConfig) -> bool:
        """
        Save upscaled video frames to output file
        
        Args:
            upscaled_frames: Upscaled video tensor
            output_path: Output video path
            original_video_path: Original video path (for metadata)
            config: Application configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get original video info for framerate and metadata
            video_info = get_video_info(original_video_path)
            fps = video_info.get('fps', 30.0)
            
            # Convert tensor to numpy and denormalize
            if isinstance(upscaled_frames, torch.Tensor):
                frames_np = upscaled_frames.detach().cpu().numpy()
            else:
                frames_np = upscaled_frames
            
            # Ensure values are in [0, 1] range then convert to [0, 255]
            frames_np = np.clip(frames_np, 0, 1)
            frames_np = (frames_np * 255).astype(np.uint8)
            
            # Rearrange from (T, C, H, W) to (T, H, W, C)
            if frames_np.ndim == 4:
                frames_np = frames_np.transpose(0, 2, 3, 1)
            
            # Get video output settings
            codec = getattr(config, 'ffmpeg_encoder', 'libx264')
            quality = getattr(config, 'ffmpeg_quality', 23)
            
            # Create video writer
            height, width = frames_np.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Cannot create video writer for {output_path}")
            
            # Write frames
            for frame in frames_np:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            self.logger.info(f"Saved upscaled video: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save video: {e}")
            return False
    
    def _save_image_result(self, upscaled_frame: torch.Tensor, output_path: str, 
                          config: AppConfig) -> bool:
        """
        Save upscaled image frame to output file
        
        Args:
            upscaled_frame: Upscaled image tensor
            output_path: Output image path
            config: Application configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert tensor to numpy and denormalize
            if isinstance(upscaled_frame, torch.Tensor):
                frame_np = upscaled_frame.detach().cpu().numpy()
            else:
                frame_np = upscaled_frame
            
            # Remove batch dimension if present and get first frame
            if frame_np.ndim == 4:
                frame_np = frame_np[0]  # Take first frame
            
            # Ensure values are in [0, 1] range then convert to [0, 255]
            frame_np = np.clip(frame_np, 0, 1)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Rearrange from (C, H, W) to (H, W, C)
            if frame_np.ndim == 3:
                frame_np = frame_np.transpose(1, 2, 0)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image as PNG (lossless)
            success = cv2.imwrite(output_path, frame_bgr)
            
            if success:
                self.logger.info(f"Saved upscaled image: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to write image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            return False
    
    def cleanup(self):
        """Cleanup inference engine and free resources"""
        if self.model_manager:
            self.model_manager.cleanup()


# Global instance for easy access
_inference_engine = None

def get_seedvr2_inference_engine(logger: Optional[logging.Logger] = None) -> SeedVR2InferenceEngine:
    """Get global SeedVR2 inference engine instance"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = SeedVR2InferenceEngine(logger)
    return _inference_engine 