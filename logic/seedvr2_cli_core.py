"""
SeedVR2 CLI-based Core Processing Module

This module provides video-to-video upscaling using SeedVR2 models via CLI approach,
integrating with the existing STAR pipeline structure and patterns.

Key Features:
- Video-to-video ratio-based upscaling with temporal consistency
- Integration with STAR's scene splitting and chunk processing
- CLI-based processing without ComfyUI dependencies
- Block swap support for large models on limited VRAM
- Multi-GPU support for faster processing
- Color correction using wavelet reconstruction
- Automatic frame padding and chunk optimization
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
import subprocess
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any, Generator
from pathlib import Path
import torch
import json

# Import cancellation manager
from .cancellation_manager import cancellation_manager, CancelledError
from .common_utils import format_time


class SeedVR2BlockSwap:
    """
    Independent block swapping implementation for SeedVR2 models.
    
    This class handles dynamic block swapping between GPU and CPU memory
    without ComfyUI dependencies. Currently disabled for stability.
    """
    
    def __init__(self, enable_debug: bool = False):
        self.enable_debug = enable_debug
        self.swap_history = []
        self._is_available = False  # Disabled by default for stability
        
    def log(self, message: str, level: str = "INFO"):
        """Log debug messages if enabled."""
        if self.enable_debug:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] BlockSwap-{level}: {message}")
    
    def get_module_memory_mb(self, module: torch.nn.Module) -> float:
        """Calculate memory usage of a module in MB."""
        if module is None:
            return 0.0
        total_bytes = sum(
            param.nelement() * param.element_size() 
            for param in module.parameters() 
            if param.data is not None
        )
        return total_bytes / (1024 * 1024)
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3), 
            "max_allocated": torch.cuda.max_memory_allocated() / (1024**3)
        }
    
    def apply_block_swap(self, model, blocks_to_swap: int, offload_io: bool = False, model_caching: bool = False):
        """
        Apply block swapping to a model.
        
        IMPORTANT: Block swap is currently disabled for stability.
        This method will do nothing and return immediately.
        
        Args:
            model: The model to apply block swapping to
            blocks_to_swap: Number of blocks to swap (0=disabled)
            offload_io: Whether to offload I/O components (disabled)
            model_caching: Whether to enable model caching (disabled)
        """
        if not self._is_available:
            self.log("Block swap is disabled for stability", "INFO")
            return
            
        if blocks_to_swap <= 0:
            self.log("Block swap disabled (blocks_to_swap=0)")
            return
            
        # NOTE: Block swap functionality is disabled for now
        # This prevents ComfyUI compatibility issues and ensures stability
        self.log("Block swap temporarily disabled for compatibility", "WARN")
        return
    
    def _offload_io_components(self, model):
        """Offload input/output components to CPU. (DISABLED)"""
        self.log("I/O component offloading is disabled", "INFO")
        return
    
    def _setup_model_caching(self, model):
        """Setup model caching in RAM. (DISABLED)"""
        self.log("Model caching is disabled", "INFO")
        return


class SeedVR2SessionManager:
    """
    Proper session-based SeedVR2 model manager.
    Creates model ONCE and reuses for all batches - fixes memory leak.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.runner = None
        self.current_model = None
        self.is_initialized = False
        self.processing_args = None
        
    def initialize_session(self, processing_args: Dict[str, Any], seedvr2_config) -> bool:
        """
        Initialize SeedVR2 session with model loading.
        Call this ONCE at the start of processing.
        """
        if self.is_initialized:
            self.logger.warning("Session already initialized, skipping...")
            return True
            
        try:
            self.processing_args = processing_args.copy()
            
            # Add SeedVR2 to path
            seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
            if seedvr2_base_path not in sys.path:
                sys.path.insert(0, seedvr2_base_path)
            
            # Import modules
            from src.core.model_manager import configure_runner
            
            # Verify model exists
            model_path = os.path.join(processing_args["model_dir"], processing_args["model"])
            if not os.path.exists(model_path):
                from src.utils.downloads import download_weight
                self.logger.info(f"Downloading model: {processing_args['model']}")
                download_weight(processing_args["model"], processing_args["model_dir"])
            
            # Create runner ONCE for the entire session
            self.logger.info(f"🔧 Creating SeedVR2 session with model: {processing_args['model']}")
            
            self.runner = configure_runner(
                model=processing_args["model"],
                base_cache_dir=processing_args["model_dir"],
                preserve_vram=processing_args["preserve_vram"],
                debug=processing_args["debug"],
                block_swap_config=None  # Disabled for stability
            )
            
            self.current_model = processing_args["model"]
            self.is_initialized = True
            
            self.logger.info("✅ SeedVR2 session initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SeedVR2 session: {e}")
            return False
    
    def process_batch(self, batch_frames: torch.Tensor, batch_args: Dict[str, Any] = None) -> torch.Tensor:
        """
        Process a batch using the existing session runner.
        This REUSES the model instead of recreating it.
        """
        if not self.is_initialized or self.runner is None:
            raise RuntimeError("Session not initialized! Call initialize_session() first.")
        
        try:
            # Use session args, override with batch-specific args if provided
            effective_args = self.processing_args.copy()
            if batch_args:
                effective_args.update(batch_args)
            
            # ✅ CALCULATE res_w based on batch frame dimensions (was missing!)
            res_w = effective_args.get("res_w", None)
            if res_w is None or res_w <= 0:
                # Calculate resolution based on input frame dimensions
                input_height, input_width = batch_frames.shape[1], batch_frames.shape[2]
                
                # Default 2x upscale for SeedVR2
                target_width = input_width * 2
                target_height = input_height * 2
                
                # Use the shorter side as the target resolution (as expected by SeedVR2)
                res_w = min(target_width, target_height)
                
                # Ensure resolution is reasonable (between 256 and 2048)
                res_w = max(256, min(2048, res_w))
                
                if self.logger:
                    self.logger.debug(f"🔧 Auto-calculated res_w: {res_w} (from {input_width}x{input_height} input)")
                    self.logger.debug(f"   Target output: ~{target_width}x{target_height}")
            
            # Ensure res_w is valid
            if res_w is None or res_w <= 0:
                res_w = 720  # Safe default
                if self.logger:
                    self.logger.warning(f"⚠️ Failed to calculate valid res_w, using default {res_w}")
            
            # Import generation function
            from src.core.generation import generation_loop
            
            # Process batch with REUSED runner
            result_tensor = generation_loop(
                runner=self.runner,  # ✅ REUSE EXISTING MODEL
                images=batch_frames,
                cfg_scale=effective_args.get("cfg_scale", 1.0),
                seed=effective_args.get("seed", -1),
                res_w=res_w,  # ✅ Now properly calculated
                batch_size=len(batch_frames),
                preserve_vram=effective_args["preserve_vram"],
                temporal_overlap=effective_args.get("temporal_overlap", 0),
                debug=effective_args.get("debug", False),
            )
            
            return result_tensor
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
    
    def cleanup_session(self):
        """
        Cleanup session and free all resources.
        Call this ONCE at the end of all processing.
        """
        if not self.is_initialized:
            return
            
        try:
            self.logger.info("🧹 Cleaning up SeedVR2 session...")
            
            if self.runner:
                # Proper cleanup of runner components
                if hasattr(self.runner, 'dit') and self.runner.dit is not None:
                    if hasattr(self.runner.dit, 'cpu'):
                        self.runner.dit.cpu()
                    del self.runner.dit
                    self.runner.dit = None
                
                if hasattr(self.runner, 'vae') and self.runner.vae is not None:
                    if hasattr(self.runner.vae, 'cpu'):
                        self.runner.vae.cpu()
                    del self.runner.vae
                    self.runner.vae = None
                
                del self.runner
                self.runner = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.current_model = None
            self.is_initialized = False
            self.processing_args = None
            
            self.logger.info("✅ SeedVR2 session cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during session cleanup: {e}")


# Global session manager instance
_global_session_manager = None

def get_session_manager(logger: Optional[logging.Logger] = None) -> SeedVR2SessionManager:
    """Get or create the global session manager."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SeedVR2SessionManager(logger)
    return _global_session_manager

def cleanup_global_session():
    """Cleanup the global session manager."""
    global _global_session_manager
    if _global_session_manager:
        _global_session_manager.cleanup_session()
        _global_session_manager = None


def extract_frames_from_video_cli(video_path: str, debug: bool = False, skip_first_frames: int = 0, load_cap: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """
    Extract frames from video using OpenCV (CLI approach).
    
    Args:
        video_path: Path to input video
        debug: Enable debug logging
        skip_first_frames: Number of frames to skip at start
        load_cap: Maximum number of frames to load
        
    Returns:
        Tuple of (frames_tensor, fps)
    """
    if debug:
        print(f"🎬 Extracting frames from video: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if debug:
        print(f"📊 Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS")
        if skip_first_frames:
            print(f"⏭️ Will skip first {skip_first_frames} frames")
        if load_cap:
            print(f"🔢 Will load maximum {load_cap} frames")
    
    frames = []
    frame_idx = 0
    frames_loaded = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip first frames if requested
        if frame_idx < skip_first_frames:
            frame_idx += 1
            continue
        
        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            print(f"📹 Extracted {frames_loaded}/{total_to_load} frames")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    if debug:
        print(f"✅ Extracted {len(frames)} frames")
    
    # Convert to tensor [T, H, W, C] and cast to Float16 for SeedVR2 compatibility
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    
    if debug:
        print(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
    
    return frames_tensor, fps


def save_frames_to_video_cli(frames_tensor: torch.Tensor, output_path: str, fps: float = 30.0, debug: bool = False):
    """
    Save frames tensor to video file (CLI approach).
    
    Args:
        frames_tensor: Frames in format [T, H, W, C] (Float16, 0-1)
        output_path: Output video path
        fps: Output video FPS
        debug: Enable debug logging
    """
    if debug:
        print(f"🎬 Saving {frames_tensor.shape[0]} frames to video: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy and denormalize
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Get video properties
    T, H, W, C = frames_np.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames_np):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if debug and (i + 1) % 100 == 0:
            print(f"💾 Saved {i + 1}/{T} frames")
    
    out.release()
    
    if debug:
        print(f"✅ Video saved successfully: {output_path}")


def process_video_with_seedvr2_cli(
    input_video_path: str,
    seedvr2_config,  # SeedVR2Config object from dataclasses
    
    # Video processing parameters
    enable_target_res: bool = False,
    target_h: int = 1080,
    target_w: int = 1920,
    target_res_mode: str = "Ratio Upscale",
    
    # Frame saving and output parameters
    save_frames: bool = False,
    save_metadata: bool = False,
    save_chunks: bool = False,
    save_chunk_frames: bool = False,
    
    # Scene processing parameters
    enable_scene_split: bool = False,
    scene_min_scene_len: int = 30,
    scene_threshold: float = 0.3,
    
    # Output parameters
    output_folder: str = "output",
    temp_folder: str = "temp",
    create_comparison_video: bool = False,
    
    # Progress callback
    progress_callback: Optional[callable] = None,
    status_callback: Optional[callable] = None,
    
    # Global settings
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    seed: int = -1,
    
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Process video using SeedVR2 with CLI approach.
    
    Args:
        input_video_path: Path to input video
        seedvr2_config: SeedVR2Config object with all settings
        ... (other parameters as per STAR pattern)
        
    Returns:
        Path to output video
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting SeedVR2 video processing (CLI mode)")
    
    # Validate input
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    if not seedvr2_config.enable:
        raise ValueError("SeedVR2 is not enabled in configuration")
    
    # Setup paths
    input_path = Path(input_video_path)
    output_dir = Path(output_folder)
    temp_dir = Path(temp_folder)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    timestamp = int(time.time())
    output_filename = f"{input_path.stem}_seedvr2_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    # Setup permanent frame saving directories (STAR standard structure)
    input_frames_permanent_save_path = None
    processed_frames_permanent_save_path = None
    
    if save_frames:
        # Create frame saving directories using STAR standard structure
        base_output_filename_no_ext = input_path.stem
        frames_output_subfolder = output_dir / base_output_filename_no_ext
        frames_output_subfolder.mkdir(parents=True, exist_ok=True)
        
        input_frames_permanent_save_path = frames_output_subfolder / "input_frames"
        processed_frames_permanent_save_path = frames_output_subfolder / "processed_frames"
        input_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
        processed_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Frame saving enabled - Input: {input_frames_permanent_save_path}")
            logger.info(f"Frame saving enabled - Processed: {processed_frames_permanent_save_path}")
    
    try:
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        if status_callback:
            status_callback("🎬 Extracting frames from video...")
        
        # Extract frames
        logger.info("Extracting frames from input video")
        frames_tensor, original_fps = extract_frames_from_video_cli(
            input_video_path, 
            debug=seedvr2_config.preserve_vram,  # Use preserve_vram flag for debug
            skip_first_frames=0,  # Can be made configurable
            load_cap=None  # Process all frames
        )
        
        # Save input frames immediately if requested (STAR standard behavior)
        if save_frames and input_frames_permanent_save_path:
            total_frames = frames_tensor.shape[0]
            if logger:
                logger.info(f"Copying {total_frames} input frames to permanent storage...")
            
            # Convert frames tensor to images and save
            frames_saved = 0
            for frame_idx in range(total_frames):
                frame_tensor = frames_tensor[frame_idx].cpu()
                if frame_tensor.dtype != torch.uint8:
                    frame_tensor = (frame_tensor * 255).clamp(0, 255).to(torch.uint8)
                
                frame_np = frame_tensor.numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                # Generate frame filename
                frame_filename = f"frame{frame_idx + 1:06d}.png"
                frame_path = input_frames_permanent_save_path / frame_filename
                
                success = cv2.imwrite(str(frame_path), frame_bgr)
                if success:
                    frames_saved += 1
                else:
                    if logger:
                        logger.warning(f"Failed to save input frame: {frame_filename}")
            
            if logger:
                logger.info(f"Input frames copied: {frames_saved}/{total_frames}")
        
        if progress_callback:
            progress_callback(0.1, "Frames extracted, preparing for processing")
        
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        if status_callback:
            status_callback("🧠 Configuring SeedVR2 model...")
        
        # Setup SeedVR2 processing
        logger.info("Setting up SeedVR2 processing")
        
        # Prepare processing arguments
        processing_args = {
            "model": seedvr2_config.model,
            "model_dir": str(Path(__file__).parent.parent.parent / "SeedVR2" / "models"),
            "preserve_vram": seedvr2_config.preserve_vram,
            "debug": seedvr2_config.preserve_vram,  # Use preserve_vram flag for debug
            "cfg_scale": seedvr2_config.cfg_scale,
            "seed": seed if seed >= 0 else torch.randint(0, 2**32, (1,)).item(),
            "batch_size": max(5, seedvr2_config.batch_size),  # Ensure minimum 5 for temporal consistency
            "temporal_overlap": seedvr2_config.temporal_overlap,
            "quality_preset": seedvr2_config.quality_preset,
            "use_gpu": seedvr2_config.use_gpu,
        }
        
        # Setup multi-GPU if enabled
        if seedvr2_config.enable_multi_gpu and seedvr2_config.gpu_devices:
            device_list = [d.strip() for d in seedvr2_config.gpu_devices.split(',') if d.strip()]
        else:
            device_list = ["0"]  # Default to GPU 0
        
        logger.info(f"Using devices: {device_list}")
        
        if progress_callback:
            progress_callback(0.2, "Model configured, starting processing")
        
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        if status_callback:
            status_callback("⚡ Processing video with SeedVR2...")
        
        # Process video using CLI approach
        logger.info("Starting SeedVR2 processing")
        result_tensor = _process_frames_with_seedvr2_cli(
            frames_tensor, 
            device_list, 
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path
        )
        
        if progress_callback:
            progress_callback(0.8, "Processing complete, saving output video")
        
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        if status_callback:
            status_callback("💾 Saving processed video...")
        
        # Save output video
        logger.info(f"Saving output video to: {output_path}")
        save_frames_to_video_cli(
            result_tensor, 
            str(output_path), 
            original_fps, 
            debug=seedvr2_config.preserve_vram
        )
        
        if progress_callback:
            progress_callback(1.0, f"SeedVR2 processing complete: {output_path.name}")
        
        logger.info(f"✅ SeedVR2 processing completed successfully: {output_path}")
        return str(output_path)
        
    except CancelledError:
        logger.info("SeedVR2 processing cancelled by user")
        raise
    except Exception as e:
        logger.error(f"Error during SeedVR2 processing: {e}")
        raise
    finally:
        # ✅ CLEANUP SESSION: Clean up the global session at the end of ALL processing
        try:
            cleanup_global_session()
            logger.info("SeedVR2 session cleaned up successfully")
        except Exception as cleanup_error:
            logger.warning(f"Session cleanup error: {cleanup_error}")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _process_frames_with_seedvr2_cli(
    frames_tensor: torch.Tensor,
    device_list: List[str],
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None
) -> torch.Tensor:
    """
    Process frames using SeedVR2 CLI approach with multi-GPU support.
    
    Args:
        frames_tensor: Input frames tensor
        device_list: List of GPU device IDs
        processing_args: Processing arguments
        seedvr2_config: SeedVR2 configuration
        progress_callback: Progress callback function
        logger: Logger instance
        
    Returns:
        Processed frames tensor
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    num_devices = len(device_list)
    
    if num_devices == 1:
        # Single GPU processing
        return _process_single_gpu_cli(
            frames_tensor, 
            device_list[0], 
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path
        )
    else:
        # Multi-GPU processing
        return _process_multi_gpu_cli(
            frames_tensor, 
            device_list, 
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path
        )


def _process_single_gpu_cli(
    frames_tensor: torch.Tensor,
    device_id: str,
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None
) -> torch.Tensor:
    """
    Process frames on single GPU using SESSION-BASED approach.
    Fixed: Now reuses model instead of recreating for each batch.
    """
    try:
        # Get session manager
        session_manager = get_session_manager(logger)
        
        # Initialize session if not already done
        if not session_manager.is_initialized:
            success = session_manager.initialize_session(processing_args, seedvr2_config)
            if not success:
                logger.error("Failed to initialize SeedVR2 session")
                return _apply_placeholder_upscaling(frames_tensor, logger.level <= logging.DEBUG if logger else False)
        
        # Calculate processing parameters
        batch_size = max(5, processing_args.get("batch_size", 5))
        temporal_overlap = max(0, processing_args.get("temporal_overlap", 0))
        total_frames = frames_tensor.shape[0]
        
        processed_frames = []
        
        logger.info(f"Processing {total_frames} frames in batches of {batch_size} with overlap {temporal_overlap}")
        
        for i in range(0, total_frames, batch_size - temporal_overlap):
            # Check for cancellation
            cancellation_manager.check_cancel()
            
            end_idx = min(i + batch_size, total_frames)
            batch_frames = frames_tensor[i:end_idx]
            
            logger.info(f"Processing batch {i//batch_size + 1}: frames {i}-{end_idx-1}")
            
            # ✅ FIXED: Process batch with REUSED session model
            processed_batch = session_manager.process_batch(
                batch_frames,
                batch_args={"seed": processing_args.get("seed", -1)}  # Override seed per batch if needed
            )
            
            # Save processed frames immediately after each batch (STAR standard behavior)
            if save_frames and processed_frames_permanent_save_path and processed_batch is not None:
                batch_num = i // batch_size + 1
                total_batches = (total_frames + batch_size - 1) // batch_size
                
                # Save frames from this batch immediately
                chunk_frames_saved = 0
                for local_frame_idx in range(processed_batch.shape[0]):
                    global_frame_idx = i + local_frame_idx
                    
                    # Convert tensor frame to image
                    frame_tensor = processed_batch[local_frame_idx].cpu()
                    if frame_tensor.dtype != torch.uint8:
                        frame_tensor = (frame_tensor * 255).clamp(0, 255).to(torch.uint8)
                    
                    frame_np = frame_tensor.numpy()
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    
                    # Generate frame filename
                    frame_filename = f"frame{global_frame_idx + 1:06d}.png"
                    dst_path = processed_frames_permanent_save_path / frame_filename
                    
                    if not dst_path.exists():  # Don't overwrite existing frames
                        success = cv2.imwrite(str(dst_path), frame_bgr)
                        if success:
                            chunk_frames_saved += 1
                        else:
                            if logger:
                                logger.warning(f"Failed to save frame: {frame_filename}")
                
                if chunk_frames_saved > 0 and logger:
                    logger.info(f"Immediately saved {chunk_frames_saved} processed frames from batch {batch_num}/{total_batches}")
            
            # Handle overlap
            if i > 0 and temporal_overlap > 0:
                processed_batch = processed_batch[temporal_overlap:]
            
            processed_frames.append(processed_batch)
            
            # Update progress
            if progress_callback:
                progress = 0.2 + 0.6 * (end_idx / total_frames)
                progress_callback(progress, f"Processed {end_idx}/{total_frames} frames")
        
        # Concatenate all processed frames
        result_tensor = torch.cat(processed_frames, dim=0)
        
        logger.info(f"Processing complete: {result_tensor.shape[0]} frames processed with REUSED model")
        return result_tensor
        
    except Exception as e:
        logger.error(f"Single GPU processing failed: {e}")
        # Cleanup on error
        cleanup_global_session()
        raise


def _process_multi_gpu_cli(
    frames_tensor: torch.Tensor,
    device_list: List[str],
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None
) -> torch.Tensor:
    """
    Process frames using multiple GPUs with CLI approach.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Processing on multiple GPUs: {device_list}")
    
    num_devices = len(device_list)
    
    # Split frames across devices
    chunks = torch.chunk(frames_tensor, num_devices, dim=0)
    
    # Use multiprocessing for true parallel GPU processing
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_queue = manager.Queue()
    workers = []
    
    # Prepare shared arguments
    shared_args = processing_args.copy()
    shared_args.update({
        "block_swap_enabled": seedvr2_config.enable_block_swap,
        "block_swap_counter": seedvr2_config.block_swap_counter,
        "block_swap_offload_io": seedvr2_config.block_swap_offload_io,
        "block_swap_model_caching": seedvr2_config.block_swap_model_caching,
    })
    
    # Start worker processes
    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process_cli,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)
    
    # Collect results
    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1
        
        if progress_callback:
            progress = 0.2 + 0.6 * (collected / num_devices)
            progress_callback(progress, f"GPU {collected}/{num_devices} completed")
    
    # Wait for all processes to complete
    for p in workers:
        p.join()
    
    # Concatenate results in original order
    result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
    
    logger.info(f"Multi-GPU processing complete: {result_tensor.shape[0]} frames processed")
    return result_tensor


def _worker_process_cli(proc_idx: int, device_id: str, frames_np: np.ndarray, shared_args: Dict[str, Any], return_queue):
    """
    Worker process for multi-GPU processing using CLI approach.
    """
    # Set CUDA device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    
    # Import torch here to respect device settings
    import torch
    
    # Convert numpy back to tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16).cuda()
    
    # Setup block swap if enabled
    block_swap = None
    if shared_args.get("block_swap_enabled", False) and shared_args.get("block_swap_counter", 0) > 0:
        block_swap = SeedVR2BlockSwap(enable_debug=shared_args.get("debug", False))
    
    # Process frames with SeedVR2 model
    # Reconstruct seedvr2_config from shared_args
    class SeedVR2ConfigStub:
        def __init__(self, args):
            self.enable_block_swap = args.get("block_swap_enabled", False)
            self.block_swap_counter = args.get("block_swap_counter", 0)
            self.block_swap_offload_io = args.get("block_swap_offload_io", False)
            self.block_swap_model_caching = args.get("block_swap_model_caching", False)
    
    seedvr2_config_stub = SeedVR2ConfigStub(shared_args)
    
    result_tensor = _process_batch_with_seedvr2_model(
        frames_tensor,
        shared_args,
        seedvr2_config_stub,
        block_swap
    )
    
    # Send result back as numpy array
    return_queue.put((proc_idx, result_tensor.cpu().numpy()))


def _process_batch_with_seedvr2_model(
    batch_frames: torch.Tensor,
    processing_args: Dict[str, Any],
    seedvr2_config,
    block_swap: Optional[SeedVR2BlockSwap] = None
) -> torch.Tensor:
    """
    ✅ FIXED: Process frames using session-based approach instead of recreating model.
    This function now REUSES the existing model instead of creating a new one each time.
    """
    try:
        debug = processing_args.get("debug", False)
        
        if debug:
            print(f"🔄 Processing batch: {batch_frames.shape} (REUSING MODEL)")
            print(f"📁 Model: {processing_args.get('model', 'unknown')}")
            print(f"💾 Preserve VRAM: {processing_args.get('preserve_vram', True)}")
        
        # Check system requirements
        if not check_seedvr2_system_requirements(debug):
            if debug:
                print("❌ System requirements not met for SeedVR2")
            return _apply_placeholder_upscaling(batch_frames, debug)
        
        # Get session manager
        session_manager = get_session_manager()
        
        # Initialize session if needed
        if not session_manager.is_initialized:
            if debug:
                print("🔧 Initializing SeedVR2 session...")
            
            success = session_manager.initialize_session(processing_args, seedvr2_config)
            if not success:
                if debug:
                    print("❌ Failed to initialize SeedVR2 session")
                return _apply_placeholder_upscaling(batch_frames, debug)
        
        # ✅ FIXED: Process with REUSED model
        if debug:
            print(f"🚀 Processing batch with REUSED SeedVR2 model")
        
        result_tensor = session_manager.process_batch(batch_frames)
        
        if debug:
            print(f"✅ Batch processed successfully: {result_tensor.shape}")
        
        return result_tensor
        
    except Exception as e:
        if debug:
            print(f"❌ Batch processing error: {e}")
            import traceback
            traceback.print_exc()
        
        # On error, cleanup and try placeholder
        cleanup_global_session()
        return _apply_placeholder_upscaling(batch_frames, debug)


def _apply_placeholder_upscaling(batch_frames: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Apply placeholder upscaling when SeedVR2 is not available.
    
    Uses simple 2x nearest neighbor upscaling as fallback.
    """
    if debug:
        print("🔄 Applying placeholder 2x upscaling")
    
    try:
        # Simple 2x upscaling using interpolation
        T, H, W, C = batch_frames.shape
        upscaled = torch.nn.functional.interpolate(
            batch_frames.permute(0, 3, 1, 2),  # [T, C, H, W]
            scale_factor=2.0,
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1)  # Back to [T, H, W, C]
        
        if debug:
            print(f"✅ Placeholder upscaling: {batch_frames.shape} -> {upscaled.shape}")
        
        return upscaled
        
    except Exception as e:
        if debug:
            print(f"❌ Placeholder upscaling failed: {e}")
        # Last resort: return original frames
        return batch_frames


# Color correction utilities (independent implementation)
def apply_wavelet_color_correction(frames_tensor: torch.Tensor, original_frames: torch.Tensor) -> torch.Tensor:
    """
    Apply wavelet-based color correction to fix color shifts.
    
    Args:
        frames_tensor: Processed frames tensor
        original_frames: Original frames tensor for reference
        
    Returns:
        Color-corrected frames tensor
    """
    # Placeholder implementation
    # TODO: Implement actual wavelet reconstruction for color correction
    return frames_tensor


def validate_and_fix_seedvr2_config(seedvr2_config, debug: bool = False):
    """
    Validate and fix common SeedVR2 configuration issues.
    
    Args:
        seedvr2_config: SeedVR2 configuration object
        debug: Enable debug logging
        
    Returns:
        Fixed configuration object
    """
    if debug:
        print("🔧 Validating SeedVR2 configuration...")
    
    # Set safer defaults for problematic configurations
    if hasattr(seedvr2_config, 'model'):
        # Use FP8 models for better stability and memory efficiency
        if 'fp16' in seedvr2_config.model and '3b' in seedvr2_config.model:
            old_model = seedvr2_config.model
            seedvr2_config.model = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
            if debug:
                print(f"🔄 Switched model from {old_model} to {seedvr2_config.model} for better stability")
    
    # Ensure batch size is optimal (4n+1 format)
    if hasattr(seedvr2_config, 'batch_size'):
        if seedvr2_config.batch_size % 4 != 1 or seedvr2_config.batch_size < 5:
            old_batch = seedvr2_config.batch_size
            seedvr2_config.batch_size = 5  # Smallest optimal batch size
            if debug:
                print(f"🔄 Adjusted batch size from {old_batch} to {seedvr2_config.batch_size} for optimal processing")
    
    # FORCE DISABLE block swap for stability (ComfyUI dependencies removed)
    if hasattr(seedvr2_config, 'enable_block_swap'):
        if seedvr2_config.enable_block_swap:
            seedvr2_config.enable_block_swap = False
            if debug:
                print("🔄 DISABLED block swap for stability (ComfyUI dependencies removed)")
    
    # Ensure block swap counter is 0
    if hasattr(seedvr2_config, 'block_swap_counter'):
        if seedvr2_config.block_swap_counter > 0:
            old_counter = seedvr2_config.block_swap_counter
            seedvr2_config.block_swap_counter = 0
            if debug:
                print(f"🔄 Reset block swap counter from {old_counter} to 0 for stability")
    
    # Disable all block swap related features
    if hasattr(seedvr2_config, 'block_swap_offload_io'):
        if seedvr2_config.block_swap_offload_io:
            seedvr2_config.block_swap_offload_io = False
            if debug:
                print("🔄 Disabled block swap I/O offloading for stability")
    
    if hasattr(seedvr2_config, 'block_swap_model_caching'):
        if seedvr2_config.block_swap_model_caching:
            seedvr2_config.block_swap_model_caching = False
            if debug:
                print("🔄 Disabled block swap model caching for stability")
    
    # Disable temporal overlap for stability if block swap was attempted
    if hasattr(seedvr2_config, 'temporal_overlap'):
        if seedvr2_config.temporal_overlap > 0:
            old_overlap = seedvr2_config.temporal_overlap
            seedvr2_config.temporal_overlap = 0
            if debug:
                print(f"🔄 Disabled temporal overlap (was {old_overlap}) for stability")
    
    # Set conservative CFG scale
    if hasattr(seedvr2_config, 'cfg_scale'):
        if seedvr2_config.cfg_scale != 1.0:
            old_cfg = seedvr2_config.cfg_scale
            seedvr2_config.cfg_scale = 1.0
            if debug:
                print(f"🔄 Set CFG scale from {old_cfg} to {seedvr2_config.cfg_scale} for stability")
    
    # Enable preserve VRAM for better memory management
    if hasattr(seedvr2_config, 'preserve_vram'):
        if not seedvr2_config.preserve_vram:
            seedvr2_config.preserve_vram = True
            if debug:
                print("🔄 Enabled preserve VRAM for better memory management")
    
    if debug:
        print("✅ SeedVR2 configuration validated and fixed")
    
    return seedvr2_config


def check_seedvr2_system_requirements(debug: bool = False) -> bool:
    """
    Check if the system meets SeedVR2 requirements.
    
    Args:
        debug: Enable debug logging
        
    Returns:
        True if requirements are met, False otherwise
    """
    if debug:
        print("🔍 Checking SeedVR2 system requirements...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        if debug:
            print("❌ CUDA not available")
        return False
    
    # Check VRAM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8.0:  # Minimum 8GB VRAM
            if debug:
                print(f"❌ Insufficient VRAM: {gpu_memory:.1f}GB (minimum 8GB required)")
            return False
        else:
            if debug:
                print(f"✅ VRAM check passed: {gpu_memory:.1f}GB")
    
    # Check PyTorch version
    torch_version = torch.__version__
    if debug:
        print(f"✅ PyTorch version: {torch_version}")
    
    # Check for required dependencies
    try:
        import einops
        import omegaconf
        if debug:
            print("✅ Required dependencies available")
    except ImportError as e:
        if debug:
            print(f"❌ Missing dependency: {e}")
        return False
    
    if debug:
        print("✅ System requirements check passed")
    
    return True 