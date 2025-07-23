"""
SeedVR2 Core Processing Module

This module provides video-to-video upscaling using SeedVR2 models,
integrating with the existing STAR pipeline structure and patterns.

Key Features:
- Video-to-video ratio-based upscaling with temporal consistency
- Integration with STAR's scene splitting and chunk processing
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
from typing import List, Dict, Tuple, Optional, Any, Generator
from pathlib import Path
import torch

# Import cancellation manager
from .cancellation_manager import cancellation_manager, CancelledError
from .common_utils import format_time

# Import the unified resolution calculation function
from .seedvr2_resolution_utils import calculate_seedvr2_resolution

def process_video_with_seedvr2(
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
    scene_video_paths: List[str] = None,
    
    # Directory and file management
    temp_dir: str = None,
    output_dir: str = None,
    base_output_filename_no_ext: str = None,
    
    # FPS parameters
    input_fps: float = 30.0,
    
    # FFmpeg parameters
    ffmpeg_preset: str = "medium",
    ffmpeg_quality_value: int = 23,
    ffmpeg_use_gpu: bool = False,
    
    # Dependencies
    logger: logging.Logger = None,
    progress = None,
    
    # Utility functions (injected dependencies)
    util_extract_frames = None,
    util_create_video_from_frames = None,
    util_get_gpu_device = None,
    format_time = None,
    
    # Metadata parameters
    params_for_metadata: Dict = None,
    metadata_handler_module = None,
    
    # Status tracking
    status_log: List[str] = None,
    current_seed: int = 99,
    
    # Batch mode parameters
    is_batch_mode: bool = False,
    batch_output_dir: str = None,
    original_filename: str = None
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """
    Process video using SeedVR2 models with temporal consistency.
    
    This function integrates SeedVR2 processing with the existing STAR pipeline,
    supporting scene splitting, chunk processing, and all global features.
    
    Args:
        input_video_path: Path to input video
        seedvr2_config: SeedVR2Config object with all settings
        ... (other parameters follow STAR pipeline conventions)
        
    Yields:
        Tuple of (output_video_path, status_message, chunk_video_path, chunk_status, comparison_video_path)
    """
    
    if status_log is None:
        status_log = []
    
    # Add SeedVR2 project root to Python path for imports
    seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
    if seedvr2_base_path not in sys.path:
        sys.path.insert(0, seedvr2_base_path)
    
    try:
        # Import SeedVR2 modules
        from src.core.model_manager import configure_runner
        from src.core.generation import generation_loop
        from src.utils.downloads import download_weight
        from src.optimization.blockswap import apply_block_swap_to_dit
        from src.utils.color_fix import wavelet_reconstruction
        
        if logger:
            logger.info("SeedVR2 modules imported successfully")
            
    except ImportError as e:
        error_msg = f"Failed to import SeedVR2 modules: {e}. Ensure SeedVR2 is properly installed."
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Check for cancellation at start
    try:
        cancellation_manager.check_cancel()
    except CancelledError:
        if logger:
            logger.info("SeedVR2 processing cancelled before start")
        yield (None, "SeedVR2 processing cancelled", None, "Cancelled", None)
        return
    
    if not input_video_path or not os.path.exists(input_video_path):
        error_msg = "Please select a valid input video file."
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    process_start_time = time.time()
    if logger:
        logger.info(f"Starting SeedVR2 processing with model: {seedvr2_config.model}")
        logger.info(f"Input video: {os.path.basename(input_video_path)}")
        logger.info(f"Using seed: {current_seed}")
        if enable_scene_split and scene_video_paths:
            logger.info(f"Scene splitting enabled with {len(scene_video_paths)} scenes")
        if is_batch_mode:
            logger.info(f"Batch mode enabled: {original_filename}")
    
    # Validate temporal consistency requirements (ensuring global standard)
    batch_size = max(5, seedvr2_config.batch_size)  # Min 5 for temporal consistency
    if batch_size != seedvr2_config.batch_size:
        if logger:
            logger.warning(f"Adjusted batch size from {seedvr2_config.batch_size} to {batch_size} for temporal consistency")
        seedvr2_config.batch_size = batch_size
    
    # Setup model path
    models_dir = os.path.join(seedvr2_base_path, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # Validate and download model if needed
    if not seedvr2_config.model:
        # Default to 3B FP8 model for better VRAM efficiency
        seedvr2_config.model = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    
    model_path = os.path.join(models_dir, seedvr2_config.model)
    if not os.path.exists(model_path):
        if logger:
            logger.info(f"Downloading SeedVR2 model: {seedvr2_config.model}")
        try:
            download_weight(seedvr2_config.model, models_dir)
        except Exception as e:
            error_msg = f"Failed to download model {seedvr2_config.model}: {e}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Extract frames from video
    status_log.append("Extracting frames from video...")
    if progress:
        progress(0.05, "🎬 Extracting frames from video...")
    
    frame_extraction_start = time.time()
    
    # Setup temporary directories
    input_frames_dir = os.path.join(temp_dir, "input_frames")
    output_frames_dir = os.path.join(temp_dir, "output_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # Setup permanent frame saving directories (STAR standard structure)
    input_frames_permanent_save_path = None
    processed_frames_permanent_save_path = None
    
    if save_frames:
        if is_batch_mode and batch_output_dir:
            frames_output_subfolder = batch_output_dir
        else:
            frames_output_subfolder = os.path.join(output_dir, base_output_filename_no_ext)
        
        os.makedirs(frames_output_subfolder, exist_ok=True)
        input_frames_permanent_save_path = os.path.join(frames_output_subfolder, "input_frames")
        processed_frames_permanent_save_path = os.path.join(frames_output_subfolder, "processed_frames")
        os.makedirs(input_frames_permanent_save_path, exist_ok=True)
        os.makedirs(processed_frames_permanent_save_path, exist_ok=True)
        
        if logger:
            logger.info(f"Frame saving enabled - Input: {input_frames_permanent_save_path}")
            logger.info(f"Frame saving enabled - Processed: {processed_frames_permanent_save_path}")
    
    # Extract frames using STAR's utility
    try:
        frame_files = util_extract_frames(
            input_video_path,
            input_frames_dir,
            logger=logger
        )
        
        if not frame_files:
            raise ValueError("No frames could be extracted from the video")
        
        total_frames = len(frame_files)
        if logger:
            logger.info(f"Extracted {total_frames} frames in {time.time() - frame_extraction_start:.2f}s")
        
        # Save input frames immediately if requested (STAR standard behavior)
        if save_frames and input_frames_permanent_save_path:
            if logger:
                logger.info(f"Copying {total_frames} input frames to permanent storage...")
            
            frames_copied = 0
            for frame_file in frame_files:
                src_path = os.path.join(input_frames_dir, frame_file)
                dst_path = os.path.join(input_frames_permanent_save_path, frame_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    frames_copied += 1
            
            if logger:
                logger.info(f"Input frames copied: {frames_copied}/{total_frames}")
            
    except Exception as e:
        error_msg = f"Frame extraction failed: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    cancellation_manager.check_cancel()
    
    # Load frames into tensor format
    status_log.append("Loading frames into memory...")
    if progress:
        progress(0.15, "📥 Loading frames into memory...")
    
    frames_tensor = _load_frames_to_tensor(
        frame_files, 
        input_frames_dir, 
        skip_first_frames=getattr(seedvr2_config, 'skip_first_frames', 0),
        logger=logger
    )
    
    if logger:
        logger.info(f"Loaded {frames_tensor.shape[0]} frames, shape: {frames_tensor.shape}")
    
    # Apply frame padding if enabled (global setting inheritance)
    if seedvr2_config.enable_frame_padding:
        status_log.append("Applying automatic frame padding...")
        if progress:
            progress(0.18, "🔧 Applying frame padding...")
        
        frames_tensor = _apply_frame_padding(frames_tensor, seedvr2_config.batch_size, logger)
        
        if logger:
            logger.info(f"Frame padding applied, final frame count: {frames_tensor.shape[0]}")
    
    cancellation_manager.check_cancel()
    
    # Setup GPU configuration
    gpu_devices = _setup_gpu_configuration(seedvr2_config, logger)
    
    # Configure SeedVR2 runner
    status_log.append("Configuring SeedVR2 model...")
    if progress:
        progress(0.25, "🔧 Configuring SeedVR2 model...")
    
    # Initialize block swap manager for advanced memory management
    from .block_swap_manager import create_block_swap_manager
    
    block_swap_manager = create_block_swap_manager(logger)
    block_swap_manager.start_session()
    
    # Configure advanced block swap with intelligent recommendations
    enhanced_block_swap_config = block_swap_manager.configure_block_swap(seedvr2_config)
    
    # Prepare block swap configuration for SeedVR2
    block_swap_config = None
    if enhanced_block_swap_config.get("enable_block_swap", False) and enhanced_block_swap_config.get("blocks_to_swap", 0) > 0:
        block_swap_config = {
            "blocks_to_swap": enhanced_block_swap_config["blocks_to_swap"],
            "offload_io_components": enhanced_block_swap_config["offload_io"], 
            "use_non_blocking": enhanced_block_swap_config.get("use_non_blocking", True),
            "enable_debug": enhanced_block_swap_config.get("debug_enabled", False)
        }
        
        if logger:
            perf_est = enhanced_block_swap_config.get("performance_estimate", {})
            logger.info(f"Advanced block swap enabled: {block_swap_config['blocks_to_swap']} blocks")
            logger.info(f"I/O offload: {block_swap_config['offload_io_components']}")
            logger.info(f"Expected performance impact: {perf_est.get('performance_impact_percent', 0):.1f}%")
            logger.info(f"Expected memory savings: {perf_est.get('memory_savings_gb', 0):.1f}GB")
    
    # Store block swap manager for later use
    params_for_metadata["block_swap_manager"] = block_swap_manager
    
    # Initialize temporal consistency manager for scene-aware processing
    from .temporal_consistency_manager import create_temporal_consistency_manager
    
    temporal_manager = create_temporal_consistency_manager(logger)
    
    # Configure temporal consistency with scene awareness
    video_info = {
        "frame_count": frames_tensor.shape[0] if frames_tensor is not None else 0,
        "available_vram_gb": enhanced_block_swap_config.get("current_vram_usage", {}).get("vram_allocated_gb", 8.0),
        "target_quality": "balanced"  # Could be made configurable
    }
    
    temporal_success, temporal_config, temporal_chunks = temporal_manager.configure_temporal_processing(
        seedvr2_config=seedvr2_config,
        video_info=video_info,
        scene_video_paths=scene_video_paths if enable_scene_split else None,
        enable_scene_awareness=enable_scene_split,
        util_extract_frames=util_extract_frames
    )
    
    if temporal_success:
        if logger:
            logger.info(f"🎬 Temporal consistency configured successfully:")
            logger.info(f"   📊 Batch size: {temporal_config['batch_size']} (original: {seedvr2_config.batch_size})")
            logger.info(f"   🔄 Temporal overlap: {temporal_config['temporal_overlap']} frames")
            logger.info(f"   🎭 Scene-aware chunks: {temporal_config['total_chunks']}")
            logger.info(f"   ⚡ Processing efficiency: {temporal_config.get('efficiency_score', 1.0):.2f}")
        
        # Update seedvr2_config with temporal corrections
        if temporal_config.get("corrections_applied"):
            seedvr2_config.batch_size = temporal_config["batch_size"]
            seedvr2_config.temporal_overlap = temporal_config["temporal_overlap"]
            status_log.append(f"Applied temporal consistency corrections: batch_size={temporal_config['batch_size']}, overlap={temporal_config['temporal_overlap']}")
    else:
        if logger:
            logger.warning("⚠️ Temporal consistency configuration failed, using original settings")
        temporal_config = {}
        temporal_chunks = []
    
    # Store temporal information for metadata
    params_for_metadata["temporal_consistency"] = {
        "enabled": temporal_success,
        "config": temporal_config,
        "chunk_count": len(temporal_chunks),
        "scene_awareness": enable_scene_split,
        "batch_size_corrected": temporal_config.get("batch_size") != seedvr2_config.batch_size if temporal_config else False
    }
    
    try:
        runner = configure_runner(
            model=seedvr2_config.model,
            base_cache_dir=models_dir,
            preserve_vram=seedvr2_config.preserve_vram,
            debug=logger.level <= logging.DEBUG if logger else False,
            block_swap_config=block_swap_config,
            flash_attention=seedvr2_config.flash_attention,
            cfg_scale=seedvr2_config.cfg_scale,
            seed=current_seed  # Ensure consistent seed usage (global setting)
        )
        
        if logger:
            logger.info("SeedVR2 runner configured successfully")
            logger.info(f"Flash Attention: {seedvr2_config.flash_attention}")
            logger.info(f"CFG Scale: {seedvr2_config.cfg_scale}")
            logger.info(f"Using seed: {current_seed}")
            
    except Exception as e:
        error_msg = f"Failed to configure SeedVR2 runner: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    cancellation_manager.check_cancel()
    
    # Enhanced temporal processing with scene awareness
    if temporal_success and temporal_chunks:
        status_log.append(f"Creating {len(temporal_chunks)} scene-aware temporal chunks...")
        if progress:
            progress(0.30, "🎬 Creating scene-aware temporal chunks...")
        
        frame_chunks = []
        frame_chunk_info = []
        
        for i, temporal_chunk in enumerate(temporal_chunks):
            start_idx = temporal_chunk.start_frame
            end_idx = min(temporal_chunk.end_frame, frames_tensor.shape[0])
            
            if start_idx < end_idx:
                chunk_tensor = frames_tensor[start_idx:end_idx]
                frame_chunks.append(chunk_tensor)
                frame_chunk_info.append(temporal_chunk)
                
                if logger:
                    scene_info = f" (scene {temporal_chunk.scene_id})" if temporal_chunk.scene_id is not None else ""
                    boundary_info = " [scene boundary]" if temporal_chunk.scene_boundary else ""
                    logger.debug(f"Temporal chunk {i}: frames {start_idx}-{end_idx-1}{scene_info}{boundary_info}")
        
        if logger:
            logger.info(f"Created {len(frame_chunks)} scene-aware temporal chunks with optimized boundaries")
            
    elif seedvr2_config.temporal_overlap > 0:
        # Fallback to standard temporal overlap
        status_log.append(f"Creating temporal chunks with {seedvr2_config.temporal_overlap} frame overlap...")
        if progress:
            progress(0.30, "🎬 Creating temporal chunks...")
        
        frame_chunks = _apply_temporal_overlap(
            frames_tensor, 
            seedvr2_config.batch_size, 
            seedvr2_config.temporal_overlap, 
            logger
        )
        frame_chunk_info = []  # No enhanced chunk info for fallback
    else:
        # Process as single batch or split into non-overlapping chunks
        total_frames = frames_tensor.shape[0]
        if total_frames <= seedvr2_config.batch_size:
            frame_chunks = [frames_tensor]
        else:
            # Split into batch-sized chunks
            frame_chunks = []
            for i in range(0, total_frames, seedvr2_config.batch_size):
                chunk = frames_tensor[i:i + seedvr2_config.batch_size]
                frame_chunks.append(chunk)
            
            if logger:
                logger.info(f"Split into {len(frame_chunks)} non-overlapping chunks")
        
        frame_chunk_info = []  # No enhanced chunk info for simple processing
    
    # Calculate resolution for SeedVR2 (using 4x upscale by default)
    calculated_resolution = calculate_seedvr2_resolution(
        input_video_path, enable_target_res, target_h, target_w, target_res_mode, 
        logger=logger
    )
    
    if logger:
        logger.info(f"SeedVR2 target resolution calculated: {calculated_resolution}")
    
    # Process frames with SeedVR2
    status_log.append(f"Processing {len(frame_chunks)} chunks with SeedVR2...")
    if progress:
        progress(0.35, "🚀 Processing frames with SeedVR2...")
    
    processing_start = time.time()
    processed_chunks = []
    
    try:
        for chunk_idx, chunk_tensor in enumerate(frame_chunks):
            try:
                # Check for cancellation before each chunk
                cancellation_manager.check_cancel()
                
                chunk_status = f"Processing chunk {chunk_idx + 1}/{len(frame_chunks)} ({chunk_tensor.shape[0]} frames)"
                
                if progress:
                    chunk_progress = 0.35 + (chunk_idx / len(frame_chunks)) * 0.45  # 35% to 80%
                    progress(chunk_progress, f"🚀 {chunk_status}")
                
                # Yield intermediate progress for chunk processing
                yield (None, "\n".join(status_log + [chunk_status]), None, chunk_status, None)
                
                # Handle multi-GPU processing for this chunk
                if seedvr2_config.enable_multi_gpu and len(gpu_devices) > 1:
                    chunk_result = _process_multi_gpu(
                        chunk_tensor, gpu_devices, seedvr2_config, runner, calculated_resolution, logger
                    )
                else:
                    # Single GPU processing for this chunk
                    chunk_result = _process_single_gpu(
                        chunk_tensor, seedvr2_config, runner, calculated_resolution, logger, None  # Don't pass progress for individual chunks
                    )
                
                processed_chunks.append(chunk_result)
                
                # Save processed frames immediately after each batch (STAR standard behavior)
                if save_frames and processed_frames_permanent_save_path and chunk_result is not None:
                    # Calculate frame indices for this chunk
                    frames_start_idx = chunk_idx * seedvr2_config.batch_size
                    frames_end_idx = min(frames_start_idx + chunk_result.shape[0], total_frames)
                    
                    # Save frames from this chunk immediately
                    chunk_frames_saved = 0
                    for local_frame_idx in range(chunk_result.shape[0]):
                        global_frame_idx = frames_start_idx + local_frame_idx
                        if global_frame_idx < len(frame_files):
                            frame_name = frame_files[global_frame_idx]
                            
                            # Convert tensor frame to image
                            frame_tensor = chunk_result[local_frame_idx].cpu()
                            if frame_tensor.dtype != torch.uint8:
                                frame_tensor = (frame_tensor * 255).clamp(0, 255).to(torch.uint8)
                            
                            frame_np = frame_tensor.numpy()
                            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                            
                            # Save to permanent location
                            dst_path = os.path.join(processed_frames_permanent_save_path, frame_name)
                            if not os.path.exists(dst_path):  # Don't overwrite existing frames
                                success = cv2.imwrite(dst_path, frame_bgr)
                                if success:
                                    chunk_frames_saved += 1
                                else:
                                    if logger:
                                        logger.warning(f"Failed to save frame: {frame_name}")
                    
                    if chunk_frames_saved > 0 and logger:
                        logger.info(f"Immediately saved {chunk_frames_saved} processed frames from batch {chunk_idx + 1}/{len(frame_chunks)}")
                
                if logger:
                    logger.info(f"Chunk {chunk_idx + 1}/{len(frame_chunks)} processed successfully")
                    
            except CancelledError:
                if logger:
                    logger.info(f"SeedVR2 processing cancelled at chunk {chunk_idx + 1}/{len(frame_chunks)}")
                
                # Handle partial results - create video from processed chunks so far
                if processed_chunks:
                    if logger:
                        logger.info(f"Creating partial video from {len(processed_chunks)} processed chunks")
                    
                    try:
                        # Combine processed chunks
                        if len(processed_chunks) == 1:
                            partial_result = processed_chunks[0]
                        else:
                            if seedvr2_config.temporal_overlap > 0:
                                partial_result = _combine_overlapping_chunks(processed_chunks, seedvr2_config.temporal_overlap, logger)
                            else:
                                partial_result = torch.cat(processed_chunks, dim=0)
                        
                        # Save partial frames and create partial video
                        partial_frames_dir = os.path.join(temp_dir, "partial_output_frames")
                        os.makedirs(partial_frames_dir, exist_ok=True)
                        
                        _save_frames_from_tensor(partial_result, partial_frames_dir, logger=logger)
                        
                        # Create partial output video
                        partial_output_path = output_video_path.replace('.mp4', '_partial_cancelled.mp4')
                        
                        util_create_video_from_frames(
                            input_frames_dir=partial_frames_dir,
                            output_video_path=partial_output_path,
                            fps=input_fps,
                            preset=ffmpeg_preset,
                            quality_value=ffmpeg_quality_value,
                            use_gpu=ffmpeg_use_gpu,
                            logger=logger
                        )
                        
                        partial_status = f"⚠️ SeedVR2 processing cancelled. Partial video saved: {len(processed_chunks)}/{len(frame_chunks)} chunks completed"
                        
                        if logger:
                            logger.info(f"Partial video saved: {partial_output_path}")
                        
                        yield (partial_output_path, partial_status, None, "Cancelled - Partial result saved", None)
                        return
                        
                    except Exception as partial_error:
                        if logger:
                            logger.error(f"Failed to create partial video: {partial_error}")
                
                # No partial results to save
                yield (None, "❌ SeedVR2 processing cancelled", None, "Cancelled", None)
                return
                
            except Exception as chunk_error:
                if logger:
                    logger.error(f"Error processing chunk {chunk_idx + 1}: {chunk_error}")
                
                # Continue with next chunk or fail depending on severity
                error_msg = f"Failed to process chunk {chunk_idx + 1}/{len(frame_chunks)}: {chunk_error}"
                yield (None, error_msg, None, f"Chunk {chunk_idx + 1} failed", None)
                
                # For now, fail the entire process if any chunk fails
                # Could be enhanced to skip failed chunks in the future
                raise RuntimeError(error_msg)
        
        # Enhanced temporal chunk combination with consistency validation
        if len(processed_chunks) == 1:
            result_tensor = processed_chunks[0]
        else:
            # Validate temporal consistency before combination
            if temporal_success and frame_chunk_info:
                if progress:
                    progress(0.85, "🎯 Validating temporal consistency...")
                
                try:
                    consistency_analysis = temporal_manager.validate_temporal_consistency_during_processing(
                        chunk_results=processed_chunks,
                        temporal_chunks=frame_chunk_info,
                        logger=logger
                    )
                    
                    # Store consistency analysis in metadata
                    params_for_metadata["temporal_consistency"]["analysis"] = consistency_analysis
                    
                    if logger:
                        quality = consistency_analysis.get("temporal_quality", "unknown")
                        score = consistency_analysis.get("overall_consistency_score", 0)
                        logger.info(f"🎯 Temporal consistency validation: {quality} (score: {score:.3f})")
                        
                    status_log.append(f"Temporal consistency: {quality} (score: {score:.3f})")
                    
                except Exception as e:
                    if logger:
                        logger.warning(f"Temporal consistency validation failed: {e}")
                    params_for_metadata["temporal_consistency"]["analysis"] = {"error": str(e)}
            
            # Combine chunks using appropriate method
            if progress:
                progress(0.87, "🔗 Combining temporal chunks...")
                
            if seedvr2_config.temporal_overlap > 0:
                result_tensor = _combine_overlapping_chunks(processed_chunks, seedvr2_config.temporal_overlap, logger)
                status_log.append(f"Combined {len(processed_chunks)} temporal chunks with overlap blending")
            else:
                result_tensor = torch.cat(processed_chunks, dim=0)
                status_log.append(f"Concatenated {len(processed_chunks)} temporal chunks")
        
        processing_time = time.time() - processing_start
        
        # Remove padding frames if they were added
        original_frame_count = len(frame_files)
        if result_tensor.shape[0] > original_frame_count:
            result_tensor = result_tensor[:original_frame_count]
            if logger:
                logger.info(f"Removed padding, final frame count: {result_tensor.shape[0]}")
        
        if logger:
            logger.info(f"SeedVR2 processing completed in {processing_time:.2f}s")
            logger.info(f"Average FPS: {result_tensor.shape[0] / processing_time:.2f}")
    
    except Exception as e:
        error_msg = f"SeedVR2 processing failed: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    cancellation_manager.check_cancel()
    
    # Apply color correction if enabled
    if seedvr2_config.color_correction:
        status_log.append("Applying color correction (wavelet reconstruction)...")
        if progress:
            progress(0.80, "🎨 Applying color correction...")
        
        try:
            # Apply wavelet reconstruction for color correction
            result_tensor = _apply_color_correction(result_tensor, frames_tensor, logger)
            if logger:
                logger.info("Color correction applied successfully")
        except Exception as e:
            if logger:
                logger.warning(f"Color correction failed, skipping: {e}")
    
    cancellation_manager.check_cancel()
    
    # Save processed frames to output directory
    status_log.append("Saving processed frames...")
    if progress:
        progress(0.85, "💾 Saving processed frames...")
    
    try:
        _save_frames_from_tensor(
            result_tensor, 
            output_frames_dir, 
            logger=logger
        )
        
        processed_frames_count = result_tensor.shape[0]
        if logger:
            logger.info(f"Saved {processed_frames_count} processed frames")
    
    except Exception as e:
        error_msg = f"Failed to save processed frames: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Save processed frames for preservation if requested (STAR standard structure)
    if save_frames and processed_frames_permanent_save_path:
        try:
            # Copy processed frames to permanent location
            frame_files_processed = sorted([f for f in os.listdir(output_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            frames_saved = 0
            
            for frame_file in frame_files_processed:
                src_path = os.path.join(output_frames_dir, frame_file)
                dst_path = os.path.join(processed_frames_permanent_save_path, frame_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    frames_saved += 1
            
            if logger:
                logger.info(f"Processed frames saved: {frames_saved}/{len(frame_files_processed)} to {processed_frames_permanent_save_path}")
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save processed frames permanently: {e}")
    
    cancellation_manager.check_cancel()
    
    # Create output video using global FFmpeg settings
    status_log.append("Creating output video...")
    if progress:
        progress(0.90, "🎬 Creating output video...")
    
    # Determine output path using global file naming conventions
    if is_batch_mode and batch_output_dir and original_filename:
        from .file_utils import get_batch_filename
        _, output_video_path, _ = get_batch_filename(
            batch_output_dir, original_filename, suffix="_seedvr2_upscaled"
        )
    else:
        from .file_utils import get_next_filename
        _, output_video_path = get_next_filename(
            output_dir, base_output_filename_no_ext, suffix="_seedvr2_upscaled", logger=logger
        )
    
    try:
        # Create video using global FFmpeg settings
        util_create_video_from_frames(
            input_frames_dir=output_frames_dir,
            output_video_path=output_video_path,
            fps=input_fps,
            preset=ffmpeg_preset,
            quality_value=ffmpeg_quality_value,
            use_gpu=ffmpeg_use_gpu,
            logger=logger
        )
        
        if logger:
            logger.info(f"Output video created: {output_video_path}")
    
    except Exception as e:
        error_msg = f"Failed to create output video: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Save metadata using global metadata handling
    if save_metadata and metadata_handler_module and params_for_metadata:
        try:
            # Add comprehensive SeedVR2 metadata
            seedvr2_metadata = {
                "upscaler_type": "seedvr2",
                "seedvr2_model": seedvr2_config.model,
                "seedvr2_batch_size": seedvr2_config.batch_size,
                "seedvr2_temporal_overlap": seedvr2_config.temporal_overlap,
                "seedvr2_preserve_vram": seedvr2_config.preserve_vram,
                "seedvr2_color_correction": seedvr2_config.color_correction,
                "seedvr2_enable_frame_padding": seedvr2_config.enable_frame_padding,
                "seedvr2_flash_attention": seedvr2_config.flash_attention,
                "seedvr2_enable_multi_gpu": seedvr2_config.enable_multi_gpu,
                "seedvr2_gpu_devices": seedvr2_config.gpu_devices,
                "seedvr2_enable_block_swap": seedvr2_config.enable_block_swap,
                "seedvr2_block_swap_counter": seedvr2_config.block_swap_counter,
                "seedvr2_block_swap_offload_io": seedvr2_config.block_swap_offload_io,
                "seedvr2_block_swap_model_caching": seedvr2_config.block_swap_model_caching,
                "seedvr2_cfg_scale": seedvr2_config.cfg_scale,
                "seedvr2_processing_time": processing_time,
                "seedvr2_avg_fps": len(frames_tensor) / processing_time if processing_time > 0 else 0,
                "seedvr2_total_frames": len(frames_tensor),
                "current_seed": current_seed
            }
            
            # Merge with global metadata
            params_for_metadata.update(seedvr2_metadata)
            
            # Save metadata using global handler
            output_dir = os.path.dirname(output_video_path)
            base_filename = os.path.splitext(os.path.basename(output_video_path))[0]
            status_info = {"processing_time_total": time.time() - overall_process_start_time}
            
            metadata_handler_module.save_metadata(
                save_flag=True,
                output_dir=output_dir,
                base_filename_no_ext=base_filename,
                params_dict=params_for_metadata,
                status_info=status_info,
                logger=logger
            )
            
            if logger:
                logger.info("Metadata saved successfully")
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save metadata: {e}")
    
    # Create comparison video if requested (global setting)
    comparison_video_path = None
    create_comparison_enabled = False
    
    # Check for comparison video setting in multiple ways (global settings inheritance)
    if params_for_metadata:
        create_comparison_enabled = params_for_metadata.get('create_comparison_video', False)
    
    # Also check if it's explicitly passed as a global parameter
    if not create_comparison_enabled and hasattr(seedvr2_config, 'create_comparison_video'):
        create_comparison_enabled = seedvr2_config.create_comparison_video
    
    if create_comparison_enabled:
        try:
            status_log.append("Creating comparison video...")
            if progress:
                progress(0.95, "📊 Creating comparison video...")
            
            from .comparison_video import create_comparison_video, get_comparison_output_path
            
            comparison_output_path = get_comparison_output_path(output_video_path)
            
            comparison_video_path = create_comparison_video(
                original_video=input_video_path,
                upscaled_video=output_video_path,
                output_path=comparison_output_path,
                logger=logger
            )
            
            if logger:
                logger.info(f"Comparison video created: {comparison_video_path}")
                
            # Yield progress update for comparison video
            yield (None, "\n".join(status_log), None, "Comparison video created", comparison_video_path)
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to create comparison video: {e}")
            # Don't fail the entire process if comparison video creation fails
    
    # Save chunks if requested (global setting)
    if save_chunks:
        try:
            chunks_save_dir = os.path.join(output_dir, f"{base_output_filename_no_ext}_chunks")
            os.makedirs(chunks_save_dir, exist_ok=True)
            
            # Save the complete video as a chunk for consistency
            chunk_path = os.path.join(chunks_save_dir, f"chunk_001_seedvr2.mp4")
            shutil.copy2(output_video_path, chunk_path)
            
            if logger:
                logger.info(f"Chunk saved: {chunk_path}")
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save chunks: {e}")
    
    # Calculate total processing time
    total_time = time.time() - process_start_time
    
    # Format final status message
    final_status = f"✅ SeedVR2 processing completed in {format_time(total_time) if format_time else f'{total_time:.1f}s'}"
    final_status += f" | {processed_frames_count} frames | Avg: {processed_frames_count/total_time:.1f} FPS"
    
    if logger:
        logger.info(final_status)
        logger.info(f"Final output: {output_video_path}")
        if comparison_video_path:
            logger.info(f"Comparison video: {comparison_video_path}")
    
    status_log.append(final_status)
    
    # Clean up block swap manager and gather performance statistics
    try:
        if 'block_swap_manager' in params_for_metadata:
            block_swap_manager = params_for_metadata['block_swap_manager']
            
            # Get final memory statistics
            final_status_info = block_swap_manager.get_real_time_status()
            block_swap_manager.end_session()
            
            # Add block swap performance to metadata
            if save_metadata and metadata_handler_module:
                block_swap_stats = {
                    "block_swap_enabled": enhanced_block_swap_config.get("enable_block_swap", False),
                    "blocks_swapped": enhanced_block_swap_config.get("blocks_to_swap", 0),
                    "io_offloading": enhanced_block_swap_config.get("offload_io", False),
                    "model_caching": enhanced_block_swap_config.get("model_caching", False),
                    "final_memory_usage": final_status_info.get("memory_usage", {}),
                    "performance_stats": block_swap_manager.performance_stats
                }
                params_for_metadata.update(block_swap_stats)
            
            # Update final status with memory info
            memory_info = final_status_info.get("memory_usage", {})
            if memory_info:
                from .block_swap_manager import format_memory_info
                memory_summary = format_memory_info(memory_info)
                final_status += f" | {memory_summary}"
            
            if logger:
                logger.info("Block swap session completed successfully")
                
        # Remove block swap manager from metadata to avoid serialization issues
        if 'block_swap_manager' in params_for_metadata:
            del params_for_metadata['block_swap_manager']
            
    except Exception as e:
        if logger:
            logger.warning(f"Block swap cleanup error: {e}")
    
    # Final progress update
    if progress:
        progress(1.0, final_status)
    
    # Yield final result following STAR pipeline pattern
    yield (output_video_path, "\n".join(status_log), None, "SeedVR2 processing complete", comparison_video_path)


def _apply_color_correction(result_tensor: torch.Tensor, original_tensor: torch.Tensor, logger=None) -> torch.Tensor:
    """
    Apply color correction using wavelet reconstruction to fix color shifts.
    
    Args:
        result_tensor: Processed frames tensor
        original_tensor: Original frames tensor  
        logger: Logger instance
        
    Returns:
        Color-corrected result tensor
    """
    try:
        # Import SeedVR2's color correction module
        from src.utils.color_fix import wavelet_reconstruction
        
        # Apply wavelet reconstruction for color correction
        corrected_tensor = wavelet_reconstruction(result_tensor, original_tensor)
        
        if logger:
            logger.info("Wavelet color correction applied successfully")
        
        return corrected_tensor
        
    except ImportError:
        if logger:
            logger.warning("SeedVR2 color correction module not available, skipping")
        return result_tensor
    except Exception as e:
        if logger:
            logger.warning(f"Color correction failed: {e}")
        return result_tensor


def _apply_frame_padding(frames_tensor: torch.Tensor, target_batch_size: int, logger=None) -> torch.Tensor:
    """
    Apply automatic frame padding for optimal chunk quality (like STAR's chunk optimization).
    
    Args:
        frames_tensor: Input frames tensor
        target_batch_size: Target batch size for processing
        logger: Logger instance
        
    Returns:
        Padded frames tensor
    """
    
    total_frames = frames_tensor.shape[0]
    
    if total_frames <= target_batch_size:
        # No padding needed for small videos
        return frames_tensor
    
    # Calculate if padding is needed for the last chunk
    remainder = total_frames % target_batch_size
    
    if remainder == 0:
        # Perfect divisible, no padding needed
        return frames_tensor
    
    # Pad with repeated frames to ensure optimal processing
    padding_needed = target_batch_size - remainder
    
    # Repeat the last frame for padding
    last_frame = frames_tensor[-1:].repeat(padding_needed, 1, 1, 1)
    padded_tensor = torch.cat([frames_tensor, last_frame], dim=0)
    
    if logger:
        logger.info(f"Applied frame padding: {total_frames} -> {padded_tensor.shape[0]} frames (added {padding_needed} frames)")
    
    return padded_tensor


def _combine_overlapping_chunks(chunk_results: List[torch.Tensor], overlap: int, logger=None) -> torch.Tensor:
    """
    Combine overlapping chunks by blending the overlapped regions.
    
    Args:
        chunk_results: List of processed chunk tensors
        overlap: Number of overlapping frames
        logger: Logger instance
        
    Returns:
        Combined tensor with blended overlaps
    """
    
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    combined_frames = []
    
    for i, chunk in enumerate(chunk_results):
        if i == 0:
            # First chunk: take all frames
            combined_frames.append(chunk)
        else:
            # Subsequent chunks: blend the overlap region and take the rest
            prev_chunk_end = combined_frames[-1][-overlap:]
            current_chunk_start = chunk[:overlap]
            current_chunk_rest = chunk[overlap:]
            
            # Blend overlapping frames with linear interpolation
            blended_overlap = []
            for j in range(overlap):
                weight = (j + 1) / (overlap + 1)  # Gradual transition
                blended_frame = (1 - weight) * prev_chunk_end[j] + weight * current_chunk_start[j]
                blended_overlap.append(blended_frame)
            
            # Replace the overlap region in the previous chunk
            combined_frames[-1] = torch.cat([combined_frames[-1][:-overlap], torch.stack(blended_overlap)], dim=0)
            
            # Add the non-overlapping part of the current chunk
            if current_chunk_rest.shape[0] > 0:
                combined_frames.append(current_chunk_rest)
    
    # Concatenate all combined frames
    result = torch.cat(combined_frames, dim=0)
    
    if logger:
        logger.info(f"Combined {len(chunk_results)} overlapping chunks into {result.shape[0]} frames")
    
    return result


def _apply_temporal_overlap(frames_tensor: torch.Tensor, batch_size: int, overlap: int, logger=None) -> List[torch.Tensor]:
    """
    Split frames into overlapping chunks for temporal consistency.
    
    Args:
        frames_tensor: Input frames tensor
        batch_size: Batch size for processing
        overlap: Number of overlapping frames
        logger: Logger instance
        
    Returns:
        List of overlapping frame chunks
    """
    
    total_frames = frames_tensor.shape[0]
    chunks = []
    
    if total_frames <= batch_size:
        # Single chunk, no overlap needed
        return [frames_tensor]
    
    step_size = batch_size - overlap
    start_idx = 0
    
    while start_idx < total_frames:
        end_idx = min(start_idx + batch_size, total_frames)
        chunk = frames_tensor[start_idx:end_idx]
        
        # Ensure minimum chunk size
        if chunk.shape[0] < overlap and len(chunks) > 0:
            # Merge small end chunk with previous chunk
            prev_chunk = chunks[-1]
            combined_chunk = torch.cat([prev_chunk, chunk], dim=0)
            # Keep only the last batch_size frames to avoid overly large chunks
            if combined_chunk.shape[0] > batch_size:
                combined_chunk = combined_chunk[-batch_size:]
            chunks[-1] = combined_chunk
            break
        else:
            chunks.append(chunk)
        
        start_idx += step_size
        
        # Break if we've reached the end
        if end_idx >= total_frames:
            break
    
    if logger:
        logger.info(f"Created {len(chunks)} temporal chunks with {overlap} frame overlap")
    
    return chunks


def _load_frames_to_tensor(frame_files: List[str], frames_dir: str, skip_first_frames: int = 0, logger=None) -> torch.Tensor:
    """Load frame files into a tensor format compatible with SeedVR2."""
    
    # Skip first frames if requested
    if skip_first_frames > 0:
        frame_files = frame_files[skip_first_frames:]
        if logger:
            logger.info(f"Skipped first {skip_first_frames} frames")
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        if os.path.exists(frame_path):
            # Load frame using OpenCV (BGR format)
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                if logger:
                    logger.warning(f"Failed to load frame: {frame_path}")
        else:
            if logger:
                logger.warning(f"Frame file not found: {frame_path}")
    
    if not frames:
        raise ValueError("No valid frames could be loaded")
    
    # Convert to numpy array and then to tensor
    frames_np = np.stack(frames, axis=0)
    
    # Convert to tensor format expected by SeedVR2: [T, H, W, C] with values normalized to 0-1
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    
    # Convert to FP16 for memory efficiency
    frames_tensor = frames_tensor.to(torch.float16)
    
    return frames_tensor


def _setup_gpu_configuration(seedvr2_config, logger=None) -> List[str]:
    """Setup GPU configuration for SeedVR2 processing."""
    
    if not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available, falling back to CPU")
        return ["cpu"]
    
    if seedvr2_config.enable_multi_gpu:
        # Parse GPU devices
        gpu_devices = [d.strip() for d in seedvr2_config.gpu_devices.split(',') if d.strip()]
        
        # Validate GPU devices
        available_gpus = list(range(torch.cuda.device_count()))
        valid_devices = []
        
        for device_str in gpu_devices:
            try:
                device_id = int(device_str)
                if device_id in available_gpus:
                    valid_devices.append(device_str)
                else:
                    if logger:
                        logger.warning(f"GPU {device_id} not available, skipping")
            except ValueError:
                if logger:
                    logger.warning(f"Invalid GPU device ID: {device_str}")
        
        if valid_devices:
            if logger:
                logger.info(f"Multi-GPU enabled with devices: {valid_devices}")
            return valid_devices
        else:
            if logger:
                logger.warning("No valid GPU devices found, falling back to single GPU")
    
    # Single GPU or fallback
    gpu_devices = [seedvr2_config.gpu_devices.split(',')[0].strip() if seedvr2_config.gpu_devices else "0"]
    
    try:
        device_id = int(gpu_devices[0])
        if device_id < torch.cuda.device_count():
            if logger:
                logger.info(f"Using single GPU: {device_id}")
            return gpu_devices
    except ValueError:
        pass
    
    # Fallback to GPU 0
    if logger:
        logger.info("Using default GPU: 0")
    return ["0"]


def _process_single_gpu(frames_tensor: torch.Tensor, seedvr2_config, runner, calculated_resolution: int, logger=None, progress=None) -> torch.Tensor:
    """Process frames using a single GPU."""
    
    # Move to GPU
    device = f"cuda:{seedvr2_config.gpu_devices.split(',')[0].strip()}" if seedvr2_config.gpu_devices else "cuda:0"
    frames_tensor = frames_tensor.to(device)
    
    # Setup generation parameters
    generation_params = {
        "cfg_scale": seedvr2_config.cfg_scale,
        "seed": seedvr2_config.seed if seedvr2_config.seed >= 0 else None,
        "res_w": _calculate_seedvr2_resolution(input_video_path, enable_target_res, target_h, target_w, target_res_mode, upscale_factor=2.0, logger=logger),
        "batch_size": seedvr2_config.batch_size,
        "preserve_vram": seedvr2_config.preserve_vram,
        "temporal_overlap": seedvr2_config.temporal_overlap,
        "debug": logger.level <= logging.DEBUG if logger else False
    }
    
    # Add progress callback if available
    if progress:
        def progress_callback(batch_num, total_batches, current_frame, status):
            # Map to progress range 0.35 to 0.8 (processing phase)
            batch_progress = batch_num / total_batches if total_batches > 0 else 0
            overall_progress = 0.35 + (batch_progress * 0.45)
            progress(overall_progress, f"🚀 {status} (Batch {batch_num}/{total_batches})")
        
        generation_params["progress_callback"] = progress_callback
    
    try:
        # Import generation loop
        from src.core.generation import generation_loop
        
        # Run generation
        result_tensor = generation_loop(
            runner=runner,
            images=frames_tensor,
            **generation_params
        )
        
        return result_tensor
        
    except Exception as e:
        if logger:
            logger.error(f"Single GPU processing failed: {e}")
        raise


def _process_multi_gpu(frames_tensor: torch.Tensor, gpu_devices: List[str], seedvr2_config, runner, calculated_resolution: int, logger=None) -> torch.Tensor:
    """Process frames using multiple GPUs with professional optimization."""
    
    import multiprocessing as mp
    import numpy as np
    
    # Ensure spawn method for CUDA compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    num_devices = len(gpu_devices)
    
    if logger:
        logger.info(f"🚀 Professional Multi-GPU processing starting with {num_devices} devices: {gpu_devices}")
        logger.info(f"📊 Input tensor shape: {frames_tensor.shape}")
    
    # Intelligent frame distribution across GPUs
    chunks = torch.chunk(frames_tensor, num_devices, dim=0)
    
    if logger:
        for i, chunk in enumerate(chunks):
            logger.info(f"🔄 GPU {gpu_devices[i]} assigned {chunk.shape[0]} frames")
    
    # Enhanced shared arguments with all features
    shared_args = {
        "model": seedvr2_config.model,
        "model_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2', 'models'),
        "preserve_vram": seedvr2_config.preserve_vram,
        "debug": logger.level <= logging.DEBUG if logger else False,
        "cfg_scale": seedvr2_config.cfg_scale,
        "seed": seedvr2_config.seed if seedvr2_config.seed >= 0 else None,
        "res_w": calculated_resolution,
        "batch_size": seedvr2_config.batch_size,
        "temporal_overlap": seedvr2_config.temporal_overlap,
        "flash_attention": seedvr2_config.flash_attention,
        "color_correction": seedvr2_config.color_correction,
        "enable_frame_padding": seedvr2_config.enable_frame_padding,
        "block_swap_config": {
            "blocks_to_swap": seedvr2_config.block_swap_counter if seedvr2_config.enable_block_swap else 0,
            "offload_io_components": seedvr2_config.block_swap_offload_io,
            "use_non_blocking": True,
            "enable_debug": logger.level <= logging.DEBUG if logger else False
        } if seedvr2_config.enable_block_swap else None
    }
    
    # Setup multiprocessing with enhanced error handling
    manager = mp.Manager()
    return_queue = manager.Queue()
    workers = []
    
    # Launch worker processes with monitoring
    start_time = time.time()
    
    for idx, (device_id, chunk_tensor) in enumerate(zip(gpu_devices, chunks)):
        if logger:
            logger.info(f"🚀 Launching professional worker {idx} on GPU {device_id}")
        
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)
    
    # Enhanced result collection with timeout and error handling
    results_np = [None] * num_devices
    collected = 0
    
    if logger:
        logger.info("⏳ Collecting results from professional worker processes...")
    
    while collected < num_devices:
        try:
            proc_idx, res_np = return_queue.get(timeout=600)  # 10 minute timeout per result
            
            if res_np is not None:
                results_np[proc_idx] = res_np
                collected += 1
                
                if logger:
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Received professional results from worker {proc_idx} ({collected}/{num_devices}) - {elapsed:.1f}s elapsed")
            else:
                if logger:
                    logger.error(f"❌ Worker {proc_idx} returned None result")
                raise RuntimeError(f"Worker {proc_idx} failed to process frames")
                
        except Exception as e:
            if logger:
                logger.error(f"❌ Error collecting results from worker processes: {e}")
            
            # Terminate remaining workers
            for p in workers:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"Multi-GPU processing failed: {e}")
    
    # Wait for all processes to complete with timeout
    for i, p in enumerate(workers):
        try:
            p.join(timeout=60)  # 1 minute timeout for cleanup
            if logger:
                logger.info(f"✅ Professional worker process {i} completed")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Worker process {i} cleanup error: {e}")
            if p.is_alive():
                p.terminate()
    
    # Verify all results collected
    if None in results_np:
        missing_indices = [i for i, result in enumerate(results_np) if result is None]
        error_msg = f"Failed to collect results from workers: {missing_indices}"
        if logger:
            logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Professional result concatenation with validation
    if logger:
        logger.info("🔄 Concatenating professional multi-GPU results...")
    
    try:
        result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
        
        # Validate result
        if result_tensor.shape[0] != frames_tensor.shape[0]:
            if logger:
                logger.warning(f"⚠️ Frame count mismatch: input {frames_tensor.shape[0]}, output {result_tensor.shape[0]}")
        
        total_time = time.time() - start_time
        if logger:
            logger.info(f"✅ Professional Multi-GPU processing complete in {total_time:.1f}s")
            logger.info(f"📊 Output tensor shape: {result_tensor.shape}")
            logger.info(f"⚡ Processing speed: {result_tensor.shape[0]/total_time:.1f} frames/second")
        
        return result_tensor
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to concatenate multi-GPU results: {e}")
        raise RuntimeError(f"Multi-GPU result concatenation failed: {e}")


def _worker_process(proc_idx: int, device_id: str, frames_np: np.ndarray, shared_args: dict, return_queue):
    """Professional worker process for multi-GPU processing with enhanced CUDA context management."""
    
    import logging
    import time
    
    start_time = time.time()
    
    try:
        # 1. CRITICAL: Set CUDA visibility BEFORE importing any CUDA-dependent modules
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
        
        # 2. Add SeedVR2 to path for imports
        seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeedVR2")
        if seedvr2_base_path not in sys.path:
            sys.path.insert(0, seedvr2_base_path)
        
        # 3. Import SeedVR2 modules AFTER setting CUDA environment
        import torch
        from src.core.model_manager import configure_runner
        from src.core.generation import generation_loop
        
        # 4. Setup worker logging
        worker_logger = logging.getLogger(f"seedvr2_worker_{proc_idx}")
        worker_logger.setLevel(logging.INFO if shared_args.get("debug", False) else logging.WARNING)
        
        if shared_args.get("debug", False):
            print(f"🔄 Professional Worker {proc_idx} starting on GPU {device_id}")
            print(f"🖥️ CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"📊 Processing {frames_np.shape[0]} frames")
        
        # 5. Validate input frames
        if frames_np.size == 0:
            raise ValueError(f"Worker {proc_idx} received empty frames array")
        
        # 6. Reconstruct frames tensor with proper dtype
        frames_tensor = torch.from_numpy(frames_np).to(torch.float16)
        
        if shared_args.get("debug", False):
            print(f"🔄 Worker {proc_idx}: Frames tensor reconstructed - shape {frames_tensor.shape}")
        
        # 7. Configure SeedVR2 runner with all features
        configure_start = time.time()
        
        runner = configure_runner(
            model=shared_args["model"],
            base_cache_dir=shared_args["model_dir"],
            preserve_vram=shared_args["preserve_vram"],
            debug=shared_args["debug"],
            block_swap_config=shared_args.get("block_swap_config", None)
        )
        
        configure_time = time.time() - configure_start
        
        if shared_args.get("debug", False):
            print(f"✅ Worker {proc_idx}: Runner configured in {configure_time:.1f}s")
        
        # 8. Run professional generation with enhanced parameters
        generation_start = time.time()
        
        result_tensor = generation_loop(
            runner=runner,
            images=frames_tensor,
            cfg_scale=shared_args["cfg_scale"],
            seed=shared_args["seed"],
            res_w=shared_args["res_w"],
            batch_size=shared_args["batch_size"],
            preserve_vram=shared_args["preserve_vram"],
            temporal_overlap=shared_args["temporal_overlap"],
            debug=shared_args["debug"],
            block_swap_config=shared_args.get("block_swap_config", None)
        )
        
        generation_time = time.time() - generation_start
        
        if shared_args.get("debug", False):
            print(f"✅ Worker {proc_idx}: Generation complete in {generation_time:.1f}s")
            print(f"📊 Result tensor shape: {result_tensor.shape}")
        
        # 9. Validate result
        if result_tensor is None:
            raise RuntimeError(f"Worker {proc_idx}: Generation returned None")
        
        if result_tensor.shape[0] == 0:
            raise RuntimeError(f"Worker {proc_idx}: Generation returned empty result")
        
        # 10. Move result to CPU and send via queue (prevents CUDA context issues)
        result_cpu = result_tensor.cpu().numpy()
        
        # Validate CPU result
        if result_cpu is None or result_cpu.size == 0:
            raise RuntimeError(f"Worker {proc_idx}: Failed to convert result to CPU")
        
        return_queue.put((proc_idx, result_cpu))
        
        total_time = time.time() - start_time
        if shared_args.get("debug", False):
            frames_per_second = frames_np.shape[0] / total_time
            print(f"✅ Worker {proc_idx}: Complete success in {total_time:.1f}s ({frames_per_second:.1f} FPS)")
        
    except Exception as e:
        error_msg = f"Professional Worker {proc_idx} (GPU {device_id}) failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Print detailed traceback for debugging
        import traceback
        traceback.print_exc()
        
        # Send None result to signal failure (main process will handle this)
        try:
            return_queue.put((proc_idx, None))
        except:
            pass  # If we can't even send failure signal, main process will timeout
        
        # Don't re-raise here as it would just crash the worker process
        # Main process will detect the None result and handle the error appropriately


def _save_frames_from_tensor(result_tensor: torch.Tensor, output_frames_dir: str, logger=None):
    """Save processed frames from tensor to individual image files."""
    
    # Convert tensor back to numpy format
    # Expected tensor format: [T, H, W, C] with values 0-1
    frames_np = result_tensor.cpu().numpy()
    
    # Convert to 0-255 range and uint8
    frames_np = (frames_np * 255).astype(np.uint8)
    
    total_frames = frames_np.shape[0]
    digits = len(str(total_frames))
    
    for i, frame in enumerate(frames_np):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Generate filename
        filename = f"frame{i+1:0{digits}d}.png"
        output_path = os.path.join(output_frames_dir, filename)
        
        # Save frame
        success = cv2.imwrite(output_path, frame_bgr)
        
        if not success and logger:
            logger.warning(f"Failed to save frame: {filename}")
    
    if logger:
        logger.info(f"Saved {total_frames} processed frames to {output_frames_dir}")


def get_available_seedvr2_models(models_dir: str = None) -> List[str]:
    """
    Get list of available SeedVR2 models.
    
    Args:
        models_dir: Path to models directory (uses default if None)
        
    Returns:
        List of available model filenames
    """
    
    if models_dir is None:
        # Get default SeedVR2 models directory
        seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
        models_dir = os.path.join(seedvr2_base_path, 'models')
    
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.safetensors') and 'seedvr2' in filename.lower():
            models.append(filename)
    
    # Sort models by size/type preference
    model_priority = [
        'seedvr2_ema_3b_fp8_e4m3fn.safetensors',  # Best balance
        'seedvr2_ema_3b_fp16.safetensors',
        'seedvr2_ema_7b_fp8_e4m3fn.safetensors',
        'seedvr2_ema_7b_fp16.safetensors'
    ]
    
    # Sort by priority, then alphabetically
    sorted_models = []
    for priority_model in model_priority:
        if priority_model in models:
            sorted_models.append(priority_model)
            models.remove(priority_model)
    
    # Add remaining models
    sorted_models.extend(sorted(models))
    
    return sorted_models


def get_model_info(model_filename: str) -> Dict[str, Any]:
    """
    Get information about a SeedVR2 model.
    
    Args:
        model_filename: Name of the model file
        
    Returns:
        Dictionary with model information
    """
    
    info = {
        'name': model_filename,
        'size': 'Unknown',
        'precision': 'Unknown',
        'variant': 'Standard',
        'recommended_batch_size': 5,
        'vram_requirement': 'Unknown'
    }
    
    # Parse model information from filename
    if '3b' in model_filename.lower():
        info['size'] = '3B Parameters'
        info['vram_requirement'] = '6-8 GB'
        info['recommended_batch_size'] = 8
    elif '7b' in model_filename.lower():
        info['size'] = '7B Parameters'
        info['vram_requirement'] = '12-16 GB'
        info['recommended_batch_size'] = 5
    
    if 'fp8' in model_filename.lower():
        info['precision'] = 'FP8 (Fastest, Lowest VRAM)'
    elif 'fp16' in model_filename.lower():
        info['precision'] = 'FP16 (Standard)'
    
    if 'sharp' in model_filename.lower():
        info['variant'] = 'Sharp (Enhanced Detail)'
    
    return info 