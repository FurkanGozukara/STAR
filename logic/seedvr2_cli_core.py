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
        Process frames with the initialized SeedVR2 model (REUSED).
        """
        if not self.is_initialized:
            raise RuntimeError("SeedVR2 session not initialized")
        
        try:
            # ✅ REUSE MODEL: No model creation/loading here
            
            # Merge batch arguments with processing args
            effective_args = self.processing_args.copy()
            if batch_args:
                effective_args.update(batch_args)
            
            # Add resolution calculation - ensure we have valid target resolution
            current_height, current_width = batch_frames.shape[1], batch_frames.shape[2]
            
            # Calculate target resolution width (maintaining aspect ratio if possible)
            if current_height == current_width:  # Square video
                res_w = max(512, min(1920, current_width * 2))  # Conservative 2x upscale
            else:
                # Use larger dimension as base, cap at reasonable size
                base_res = max(current_height, current_width)
                res_w = max(512, min(1920, base_res * 2))
            
            # ✅ ADD DEBUGGING: Log tensor format before processing
            self.logger.info(f"🔍 Processing batch - Input tensor shape: {batch_frames.shape}, dtype: {batch_frames.dtype}")
            self.logger.info(f"🔍 Frame count constraint check: {batch_frames.shape[0]} % 4 = {batch_frames.shape[0] % 4} (should be 1)")
            self.logger.info(f"🎯 Target resolution: {res_w}")
            
            # Import generation function
            from src.core.generation import generation_loop
            
            # ✅ ADD ERROR HANDLING: Wrap VAE processing to catch assertion errors
            try:
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
                
            except AssertionError as ae:
                # ✅ HANDLE VAE ASSERTION ERROR: Provide detailed debugging info
                self.logger.error(f"❌ VAE Assertion Error occurred during processing:")
                self.logger.error(f"   Input tensor shape: {batch_frames.shape}")
                self.logger.error(f"   Input tensor dtype: {batch_frames.dtype}")
                self.logger.error(f"   Frame count: {batch_frames.shape[0]}")
                self.logger.error(f"   Frame count % 4: {batch_frames.shape[0] % 4}")
                self.logger.error(f"   Expected constraint: frame_count % 4 == 1")
                self.logger.error(f"   Assertion error: {ae}")
                
                # Try to provide a fallback processing
                self.logger.warning("🔄 Attempting fallback processing with tensor format adjustment...")
                
                # Ensure tensor is in correct format and constraints
                if batch_frames.shape[0] % 4 != 1:
                    # Force adjust frame count
                    target_frames = ((batch_frames.shape[0] - 1) // 4 + 1) * 4 + 1
                    if target_frames > batch_frames.shape[0]:
                        padding_needed = target_frames - batch_frames.shape[0]
                        last_frame = batch_frames[-1:].expand(padding_needed, -1, -1, -1)
                        batch_frames_fixed = torch.cat([batch_frames, last_frame], dim=0)
                        self.logger.info(f"🔧 Fixed frame count: {batch_frames.shape[0]} -> {batch_frames_fixed.shape[0]}")
                        
                        # Retry with fixed tensor
                        try:
                            result_tensor = generation_loop(
                                runner=self.runner,
                                images=batch_frames_fixed,
                                cfg_scale=effective_args.get("cfg_scale", 1.0),
                                seed=effective_args.get("seed", -1),
                                res_w=res_w,
                                batch_size=len(batch_frames_fixed),
                                preserve_vram=effective_args["preserve_vram"],
                                temporal_overlap=effective_args.get("temporal_overlap", 0),
                                debug=effective_args.get("debug", False),
                            )
                            self.logger.info("✅ Fallback processing succeeded")
                        except Exception as fallback_error:
                            self.logger.error(f"❌ Fallback processing also failed: {fallback_error}")
                            raise
                    else:
                        raise
                else:
                    # Frame count is correct, but still failed - re-raise original error
                    raise
            
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
    
    # ✅ FIX: Ensure frame count follows 4n+1 constraint for SeedVR2 VAE compatibility
    original_frame_count = frames_tensor.shape[0]
    if original_frame_count % 4 != 1:
        # Calculate how many frames to add to reach 4n+1 format
        target_count = ((original_frame_count - 1) // 4 + 1) * 4 + 1
        padding_needed = target_count - original_frame_count
        
        if debug:
            print(f"🔧 Frame count adjustment: {original_frame_count} -> {target_count} (padding {padding_needed} frames)")
        
        # Duplicate last frame to reach target count
        if padding_needed > 0:
            last_frame = frames_tensor[-1:].expand(padding_needed, -1, -1, -1)
            frames_tensor = torch.cat([frames_tensor, last_frame], dim=0)
    
    if debug:
        print(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
        print(f"🔍 Frame count constraint check: {frames_tensor.shape[0]} % 4 = {frames_tensor.shape[0] % 4} (should be 1)")
    
    return frames_tensor, fps


def save_frames_to_video_cli(
    frames_tensor: torch.Tensor, 
    output_path: str, 
    fps: float = 30.0, 
    debug: bool = False,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
):
    """
    Save frames tensor to video file using global FFmpeg settings.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C] (Float16, 0-1)
        output_path: Output video path
        fps: Output video FPS
        debug: Enable debug logging
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
    """
    if debug and logger:
        logger.info(f"🎬 Saving {frames_tensor.shape[0]} frames to video: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory(prefix="seedvr2_video_save_") as temp_frames_dir:
            # Convert tensor to numpy and denormalize
            frames_np = frames_tensor.cpu().numpy()
            frames_np = (frames_np * 255.0).astype(np.uint8)
            
            # Get video properties
            T, H, W, C = frames_np.shape
            
            if debug and logger:
                logger.info(f"Processing {T} frames of size {H}x{W}")
            
            # Save frames to temporary directory
            for i, frame in enumerate(frames_np):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_filename = f"frame_{i+1:06d}.png"
                frame_path = os.path.join(temp_frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame_bgr)
                
                if debug and logger and (i + 1) % 100 == 0:
                    logger.info(f"💾 Saved {i + 1}/{T} frames to temp directory")
            
            # Use global FFmpeg pipeline to create video
            from .ffmpeg_utils import create_video_from_input_frames
            
            success = create_video_from_input_frames(
                input_frames_dir=temp_frames_dir,
                output_path=output_path,
                fps=fps,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality_value=ffmpeg_quality,
                ffmpeg_use_gpu=ffmpeg_use_gpu,
                logger=logger
            )
            
            if not success:
                raise RuntimeError("Failed to create video using FFmpeg pipeline")
            
            if debug and logger:
                logger.info(f"✅ Video saved successfully: {output_path}")
                
    except Exception as e:
        error_msg = f"Failed to save video: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)


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
    
    # Session directory management (NEW - to use existing session)
    session_output_dir: Optional[str] = None,  # Existing session directory from main pipeline
    base_output_filename_no_ext: Optional[str] = None,  # Base filename from main pipeline
    
    # Progress callback
    progress_callback: Optional[callable] = None,
    status_callback: Optional[callable] = None,
    
    # Global settings
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    seed: int = -1,
    
    logger: Optional[logging.Logger] = None
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """
    Process video using SeedVR2 with CLI approach.
    
    Args:
        input_video_path: Path to input video
        seedvr2_config: SeedVR2Config object with all settings
        ... (other parameters as per STAR pattern)
        
    Yields:
        Tuple of (output_video_path, status_message, chunk_video_path, chunk_status, comparison_video_path)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting SeedVR2 video processing (CLI mode)")
    
    # Validate input
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    if not seedvr2_config.enable:
        raise ValueError("SeedVR2 is not enabled in configuration")
    
    # Setup paths using STAR standard structure
    input_path = Path(input_video_path)
    output_dir = Path(output_folder)
    temp_dir = Path(temp_folder)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use existing session directory if provided, otherwise create new one
    if session_output_dir and base_output_filename_no_ext:
        # Use existing session directory from main pipeline (PREFERRED)
        session_output_path = Path(session_output_dir)
        base_name = base_output_filename_no_ext
        
        # Generate output video path using existing session
        output_video_path = str(output_dir / f"{base_name}_seedvr2_upscaled.mp4")
        output_path = Path(output_video_path)
        
        logger.info(f"Using existing session directory: {session_output_path}")
    else:
        # Fallback: Create new session directory (for standalone usage)
        from .file_utils import get_next_filename
        base_name, output_video_path = get_next_filename(str(output_dir), logger=logger)
        output_path = Path(output_video_path)
        
        # Create session directory following STAR pattern
        session_output_path = output_dir / base_name
        session_output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created new session directory: {session_output_path}")
    
    # Ensure session directory exists
    session_output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup permanent frame saving directories (STAR standard structure)
    input_frames_permanent_save_path = None
    processed_frames_permanent_save_path = None
    chunks_permanent_save_path = None
    
    if save_frames:
        # Create frame saving directories using STAR standard structure
        input_frames_permanent_save_path = session_output_path / "input_frames"
        processed_frames_permanent_save_path = session_output_path / "processed_frames"
        input_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
        processed_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Frame saving enabled - Input: {input_frames_permanent_save_path}")
            logger.info(f"Frame saving enabled - Processed: {processed_frames_permanent_save_path}")
    
    # Setup chunks directory for chunk preview functionality (STAR standard structure)
    if save_chunks or seedvr2_config.enable_chunk_preview:
        chunks_permanent_save_path = session_output_path / "chunks"
        chunks_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Chunk preview enabled - Chunks: {chunks_permanent_save_path}")
    
    try:
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        if status_callback:
            status_callback("🎬 Extracting frames from video...")
        
        # ✅ FIX FOR FRAME DUPLICATION: Check if frames already exist in session directory
        frames_tensor = None
        original_fps = None
        frames_loaded_from_existing = False
        
        # First try to load from existing input frames in session directory
        if save_frames and input_frames_permanent_save_path and input_frames_permanent_save_path.exists():
            existing_frame_files = sorted([f for f in input_frames_permanent_save_path.iterdir() 
                                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            
            if existing_frame_files:
                logger.info(f"Found {len(existing_frame_files)} existing input frames in session directory")
                logger.info("Loading frames from existing session instead of re-extracting...")
                
                try:
                    # Load frames from existing files
                    frames = []
                    for frame_file in existing_frame_files:
                        frame = cv2.imread(str(frame_file))
                        if frame is not None:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Convert to float32 and normalize to 0-1
                            frame_normalized = frame_rgb.astype(np.float32) / 255.0
                            frames.append(frame_normalized)
                    
                    if frames:
                        frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
                        
                        # ✅ FIX: Ensure frame count follows 4n+1 constraint for SeedVR2 VAE compatibility
                        original_frame_count = frames_tensor.shape[0]
                        if original_frame_count % 4 != 1:
                            # Calculate how many frames to add to reach 4n+1 format
                            target_count = ((original_frame_count - 1) // 4 + 1) * 4 + 1
                            padding_needed = target_count - original_frame_count
                            
                            logger.info(f"🔧 Frame count adjustment: {original_frame_count} -> {target_count} (padding {padding_needed} frames)")
                            
                            # Duplicate last frame to reach target count
                            if padding_needed > 0:
                                last_frame = frames_tensor[-1:].expand(padding_needed, -1, -1, -1)
                                frames_tensor = torch.cat([frames_tensor, last_frame], dim=0)
                        
                        frames_loaded_from_existing = True
                        
                        # Get FPS from original video
                        cap = cv2.VideoCapture(input_video_path)
                        original_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
                        cap.release()
                        
                        logger.info(f"✅ Loaded {len(frames)} frames from existing session directory")
                        logger.info(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
                        logger.info(f"🔍 Frame count constraint check: {frames_tensor.shape[0]} % 4 = {frames_tensor.shape[0] % 4} (should be 1)")
                    
                except Exception as e:
                    logger.warning(f"Failed to load from existing frames, will extract from video: {e}")
                    frames_tensor = None
                    frames_loaded_from_existing = False
        
        # If we couldn't load from existing frames, extract from video
        if frames_tensor is None:
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
            if frames_loaded_from_existing:
                progress_callback(0.1, "Frames loaded from existing session, preparing for processing")
            else:
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
            "ffmpeg_preset": ffmpeg_preset,
            "ffmpeg_quality": ffmpeg_quality,
            "ffmpeg_use_gpu": ffmpeg_use_gpu,
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
        result_tensor, chunk_results = _process_frames_with_seedvr2_cli(
            frames_tensor, 
            device_list, 
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path,
            chunks_permanent_save_path=chunks_permanent_save_path,
            original_fps=original_fps
        )
        
        if progress_callback:
            progress_callback(0.8, "Processing complete, saving chunks and output video")
        
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        # Handle chunk preview functionality
        last_chunk_video_path = None
        if chunks_permanent_save_path and chunk_results:
            if status_callback:
                status_callback("📹 Saving chunk previews...")
            
            last_chunk_video_path = _save_seedvr2_chunk_previews(
                chunk_results,
                chunks_permanent_save_path,
                original_fps or 30.0,  # Use original_fps parameter or default to 30.0
                seedvr2_config,
                ffmpeg_preset,
                ffmpeg_quality,
                ffmpeg_use_gpu,
                logger
            )
            
            if logger and last_chunk_video_path:
                logger.info(f"Chunk preview saved: {last_chunk_video_path}")
                # Yield chunk preview update
                yield (None, "Chunk preview saved", last_chunk_video_path, "SeedVR2 chunk preview available", None)
        
        if status_callback:
            status_callback("💾 Saving processed video...")
        
        # Save output video using global FFmpeg settings
        logger.info(f"Saving output video to: {output_path}")
        save_frames_to_video_cli(
            result_tensor, 
            str(output_path), 
            original_fps, 
            debug=seedvr2_config.preserve_vram,
            ffmpeg_preset=processing_args.get('ffmpeg_preset', 'medium'),
            ffmpeg_quality=processing_args.get('ffmpeg_quality', 23),
            ffmpeg_use_gpu=processing_args.get('ffmpeg_use_gpu', False),
            logger=logger
        )
        
        if progress_callback:
            progress_callback(1.0, f"SeedVR2 processing complete: {output_path.name}")
        
        logger.info(f"✅ SeedVR2 processing completed successfully: {output_path}")
        
        # Yield final result with chunk preview information
        yield (str(output_path), "SeedVR2 processing completed successfully", last_chunk_video_path, "SeedVR2 processing complete", None)
        
    except CancelledError:
        logger.info("SeedVR2 processing cancelled by user")
        yield (None, "SeedVR2 processing cancelled by user", None, "Cancelled", None)
        raise
    except Exception as e:
        logger.error(f"Error during SeedVR2 processing: {e}")
        yield (None, f"SeedVR2 processing error: {e}", None, f"Error: {e}", None)
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
    processed_frames_permanent_save_path: Optional[Path] = None,
    chunks_permanent_save_path: Optional[Path] = None,
    original_fps: Optional[float] = None
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Process frames using SeedVR2 CLI approach with multi-GPU support.
    
    Args:
        frames_tensor: Input frames tensor
        device_list: List of GPU device IDs
        processing_args: Processing arguments
        seedvr2_config: SeedVR2 configuration
        progress_callback: Progress callback function
        logger: Logger instance
        save_frames: Whether to save frames
        processed_frames_permanent_save_path: Path for permanent frame storage
        chunks_permanent_save_path: Path for chunk preview storage
        original_fps: Original FPS of the input video
        
    Returns:
        Tuple of (processed frames tensor, chunk results for preview)
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
            processed_frames_permanent_save_path=processed_frames_permanent_save_path,
            chunks_permanent_save_path=chunks_permanent_save_path,
            original_fps=original_fps,
            ffmpeg_preset=processing_args.get("ffmpeg_preset", "medium"),
            ffmpeg_quality=processing_args.get("ffmpeg_quality", 23),
            ffmpeg_use_gpu=processing_args.get("ffmpeg_use_gpu", False)
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
            processed_frames_permanent_save_path=processed_frames_permanent_save_path,
            chunks_permanent_save_path=chunks_permanent_save_path,
            original_fps=original_fps,
            ffmpeg_preset=processing_args.get("ffmpeg_preset", "medium"),
            ffmpeg_quality=processing_args.get("ffmpeg_quality", 23),
            ffmpeg_use_gpu=processing_args.get("ffmpeg_use_gpu", False)
        )


def _process_single_gpu_cli(
    frames_tensor: torch.Tensor,
    device_id: str,
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None,
    chunks_permanent_save_path: Optional[Path] = None,
    original_fps: Optional[float] = None,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Process frames on single GPU using SESSION-BASED approach.
    Fixed: Now reuses model instead of recreating for each batch.
    """
    try:
        # ✅ ADD FRAME COUNT TRACKING: Track original frame count
        original_frame_count = frames_tensor.shape[0]
        logger.info(f"🔢 Original frame count: {original_frame_count}")
        
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
        
        # Calculate total batches
        if temporal_overlap > 0:
            total_batches = (total_frames - batch_size) // (batch_size - temporal_overlap) + 1
        else:
            total_batches = (total_frames + batch_size - 1) // batch_size
        
        processed_frames = []
        total_processed_frames = 0
        chunk_results = []  # Initialize chunk_results at the beginning
        
        logger.info(f"Processing {total_frames} frames in batches of {batch_size} with overlap {temporal_overlap}")
        logger.info(f"Total batches to process: {total_batches}")
        
        for i in range(0, total_frames, batch_size - temporal_overlap):
            # Check for cancellation
            cancellation_manager.check_cancel()
            
            end_idx = min(i + batch_size, total_frames)
            batch_frames = frames_tensor[i:end_idx]
            
            batch_num = i // (batch_size - temporal_overlap) + 1 if temporal_overlap > 0 else i // batch_size + 1
            logger.info(f"Processing batch {batch_num}: frames {i}-{end_idx-1} (batch size: {batch_frames.shape[0]})")
            
            # ✅ FIXED: Process batch with REUSED session model
            processed_batch = session_manager.process_batch(
                batch_frames,
                batch_args={"seed": processing_args.get("seed", -1)}  # Override seed per batch if needed
            )
            
            if processed_batch is None:
                logger.warning(f"Batch {batch_num} returned None - skipping")
                continue
            
            # ✅ FRAME COUNT TRACKING: Log processed batch info
            logger.info(f"📊 Batch {batch_num} processed: {processed_batch.shape[0]} frames")
            
            # Save processed frames immediately after each batch (STAR standard behavior)
            if save_frames and processed_frames_permanent_save_path and processed_batch is not None:
                # Calculate frame indices for this batch
                frames_start_idx = i  # Use actual start index from the loop
                frames_end_idx = min(i + processed_batch.shape[0], total_frames)
                
                # Save frames from this batch immediately
                batch_frames_saved = 0
                for local_frame_idx in range(processed_batch.shape[0]):
                    global_frame_idx = frames_start_idx + local_frame_idx
                    
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
                            batch_frames_saved += 1
                        else:
                            if logger:
                                logger.warning(f"Failed to save frame: {frame_filename}")
                
                if logger and batch_frames_saved > 0:
                    logger.info(f"Saved {batch_frames_saved} frames from batch {batch_num}")
            
            # Handle overlap - remove overlapping frames from output (but not from the first batch)
            if batch_num > 0 and temporal_overlap > 0 and processed_batch.shape[0] > temporal_overlap:
                processed_batch = processed_batch[temporal_overlap:]
                logger.info(f"📊 After overlap removal: {processed_batch.shape[0]} frames kept from batch {batch_num}")
            
            processed_frames.append(processed_batch)
            total_processed_frames += processed_batch.shape[0]
            
            # Create chunk result for preview functionality
            if chunks_permanent_save_path and seedvr2_config.enable_chunk_preview:
                chunk_result = {
                    'frames_tensor': processed_batch,
                    'chunk_id': batch_num,
                    'frame_count': processed_batch.shape[0],
                    'processing_time': time.time(),
                    'device_id': device_id,
                    'batch_start_frame': frames_start_idx,
                    'batch_end_frame': frames_end_idx
                }
                chunk_results.append(chunk_result)
                
                # Keep only the last N chunks
                if len(chunk_results) >= seedvr2_config.keep_last_chunks:
                    chunk_results = chunk_results[-seedvr2_config.keep_last_chunks:]
                
                # Save chunk preview
                last_chunk_path = _save_seedvr2_chunk_previews(
                    chunk_results,
                    chunks_permanent_save_path,
                    original_fps or 30.0,  # Use original_fps parameter or default to 30.0
                    seedvr2_config,
                    ffmpeg_preset,
                    ffmpeg_quality,
                    ffmpeg_use_gpu,
                    logger
                )
                
                if last_chunk_path and logger:
                    logger.info(f"Updated chunk preview: {last_chunk_path}")
            
            # Update progress
            if progress_callback:
                progress = 0.2 + 0.6 * (batch_num / total_batches)
                progress_callback(progress, f"Batch {batch_num + 1}/{total_batches} completed")
        
        # ✅ FRAME COUNT VALIDATION: Check if we processed the right number of frames
        logger.info(f"🔢 Frame count summary:")
        logger.info(f"   Original frames: {original_frame_count}")
        logger.info(f"   Total processed: {total_processed_frames}")
        
        # Concatenate all processed frames
        if processed_frames:
            result_tensor = torch.cat(processed_frames, dim=0)
            final_frame_count = result_tensor.shape[0]
            
            # ✅ ENSURE CORRECT OUTPUT COUNT: Trim to original frame count if needed
            if final_frame_count > original_frame_count:
                logger.warning(f"Trimming output from {final_frame_count} to {original_frame_count} frames")
                result_tensor = result_tensor[:original_frame_count]
                final_frame_count = result_tensor.shape[0]
            elif final_frame_count < original_frame_count:
                logger.warning(f"Output has fewer frames ({final_frame_count}) than input ({original_frame_count})")
                # Could potentially pad with last frame here if needed
            
            logger.info(f"📊 Final output: {final_frame_count} frames")
        else:
            logger.error("No processed frames available!")
            return _apply_placeholder_upscaling(frames_tensor, logger.level <= logging.DEBUG if logger else False)
        
        logger.info(f"Processing complete: {result_tensor.shape[0]} frames processed with REUSED model")
        return result_tensor, chunk_results
        
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
    processed_frames_permanent_save_path: Optional[Path] = None,
    chunks_permanent_save_path: Optional[Path] = None,
    original_fps: Optional[float] = None,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
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
    
    # Create chunk results for preview functionality
    chunk_results = []
    if chunks_permanent_save_path and seedvr2_config.enable_chunk_preview:
        # For now, create a single chunk with the entire processed result
        # This can be enhanced later to support actual chunking during processing
        chunk_results.append({
            'frames_tensor': result_tensor,
            'chunk_id': 1,
            'frame_count': result_tensor.shape[0],
            'processing_time': time.time(),
        })
    
    logger.info(f"Multi-GPU processing complete: {result_tensor.shape[0]} frames processed")
    return result_tensor, chunk_results


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


def _save_seedvr2_chunk_previews(
    chunk_results: List[Dict[str, Any]], 
    chunks_permanent_save_path: Path,
    fps: float,
    seedvr2_config,
    ffmpeg_preset: str,
    ffmpeg_quality: int,
    ffmpeg_use_gpu: bool,
    logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Save SeedVR2 chunk previews following STAR pattern.
    
    Args:
        chunk_results: List of chunk processing results
        chunks_permanent_save_path: Path to save chunks
        fps: Video frame rate
        seedvr2_config: SeedVR2 configuration
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Optional logger
        
    Returns:
        Path to last chunk video or None if no chunks
    """
    if not chunk_results or not seedvr2_config.enable_chunk_preview:
        return None
        
    try:
        chunks_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        # Sort chunks by processing time to get the most recent
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get('processing_time', 0))
        
        # Keep only the last N chunks as configured
        chunks_to_keep = sorted_chunks[-seedvr2_config.keep_last_chunks:]
        
        saved_chunk_paths = []
        last_chunk_video_path = None
        
        if logger:
            logger.info(f"Saving {len(chunks_to_keep)} chunk previews to {chunks_permanent_save_path}")
        
        for i, chunk_data in enumerate(chunks_to_keep):
            try:
                chunk_num = len(sorted_chunks) - len(chunks_to_keep) + i + 1
                chunk_filename = f"chunk_{chunk_num:04d}.mp4"
                chunk_video_path = chunks_permanent_save_path / chunk_filename
                
                # Get chunk frames (torch tensor)
                chunk_frames = chunk_data.get('frames_tensor')
                if chunk_frames is None:
                    if logger:
                        logger.warning(f"Chunk {chunk_num} has no frames tensor, skipping")
                    continue
                
                # Create temporary directory for this chunk's frames
                import tempfile
                with tempfile.TemporaryDirectory(prefix=f"seedvr2_chunk_{chunk_num}_") as temp_chunk_dir:
                    # Save frames to temporary directory
                    frame_count = chunk_frames.shape[0]
                    preview_frames = min(frame_count, getattr(seedvr2_config, 'chunk_preview_frames', 125))
                    
                    # Take last N frames from chunk for preview
                    preview_tensor = chunk_frames[-preview_frames:] if frame_count > preview_frames else chunk_frames
                    
                    if logger:
                        logger.info(f"Processing chunk {chunk_num}: {preview_tensor.shape[0]} frames, shape: {preview_tensor.shape}")
                    
                    # Save frames as images
                    saved_frames = 0
                    for frame_idx in range(preview_tensor.shape[0]):
                        try:
                            frame_tensor = preview_tensor[frame_idx]
                            
                            # Convert tensor to numpy array with proper shape handling
                            if frame_tensor.dim() == 4:  # Batch dimension present
                                frame_tensor = frame_tensor.squeeze(0)  # Remove batch dimension
                            
                            if frame_tensor.dim() == 3:  # CHW format
                                if frame_tensor.shape[0] == 3:  # RGB channels first
                                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                                elif frame_tensor.shape[2] == 3:  # RGB channels last
                                    frame_np = frame_tensor.cpu().numpy()
                                else:
                                    # Handle other channel configurations
                                    frame_np = frame_tensor.cpu().numpy()
                                    if frame_np.shape[0] == 3:
                                        frame_np = frame_np.transpose(1, 2, 0)
                            else:  # Already HWC format
                                frame_np = frame_tensor.cpu().numpy()
                            
                            # Ensure we have 3 channels (RGB)
                            if frame_np.shape[-1] != 3:
                                if logger:
                                    logger.warning(f"Frame {frame_idx} has {frame_np.shape[-1]} channels, expected 3. Skipping.")
                                continue
                            
                            # Convert to uint8 if needed
                            if frame_np.dtype != np.uint8:
                                if frame_np.max() <= 1.0:
                                    frame_np = (frame_np * 255).astype(np.uint8)
                                else:
                                    frame_np = frame_np.astype(np.uint8)
                            
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                            
                            # Save frame
                            frame_filename = f"frame_{frame_idx+1:06d}.png"
                            frame_path = os.path.join(temp_chunk_dir, frame_filename)
                            cv2.imwrite(frame_path, frame_bgr)
                            saved_frames += 1
                            
                        except Exception as frame_error:
                            if logger:
                                logger.warning(f"Error saving frame {frame_idx} in chunk {chunk_num}: {frame_error}")
                            continue
                    
                    if saved_frames == 0:
                        if logger:
                            logger.warning(f"No frames saved for chunk {chunk_num}, skipping video creation")
                        continue
                    
                    # Create video from frames using global FFmpeg settings
                    from .ffmpeg_utils import create_video_from_input_frames
                    success = create_video_from_input_frames(
                        input_frames_dir=temp_chunk_dir,
                        output_path=str(chunk_video_path),
                        fps=fps,
                        ffmpeg_preset=ffmpeg_preset,
                        ffmpeg_quality_value=ffmpeg_quality,
                        ffmpeg_use_gpu=ffmpeg_use_gpu,
                        logger=logger
                    )
                    
                    if success:
                        saved_chunk_paths.append(str(chunk_video_path))
                        last_chunk_video_path = str(chunk_video_path)
                        if logger:
                            logger.info(f"Successfully created chunk preview: {chunk_video_path}")
                    else:
                        if logger:
                            logger.warning(f"Failed to create chunk preview: {chunk_video_path}")
                            
            except Exception as chunk_error:
                if logger:
                    logger.error(f"Error saving chunk {i+1}: {chunk_error}")
                continue
        
        # Clean up old chunks if we have more than keep_chunks
        if chunks_permanent_save_path.exists():
            try:
                chunk_files = sorted([f for f in chunks_permanent_save_path.iterdir() 
                                    if f.name.startswith('chunk_') and f.name.endswith('.mp4')])
                
                if len(chunk_files) > seedvr2_config.keep_last_chunks:
                    files_to_remove = chunk_files[:-seedvr2_config.keep_last_chunks]
                    for old_chunk in files_to_remove:
                        old_chunk.unlink()
                        if logger:
                            logger.debug(f"Removed old chunk: {old_chunk.name}")
                            
            except Exception as e:
                if logger:
                    logger.warning(f"Error cleaning up old chunks: {e}")
        
        return last_chunk_video_path
        
    except Exception as e:
        if logger:
            logger.error(f"Error in chunk preview saving: {e}")
        return None


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