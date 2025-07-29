"""
SeedVR2 Scene-based Processing Module

This module implements scene-based processing for SeedVR2, similar to STAR's scene processing.
It splits videos into scenes and processes each scene individually for better quality.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Generator, Any
import logging

from .cancellation_manager import cancellation_manager, CancelledError
from .scene_utils import split_video_into_scenes as util_split_video_into_scenes, merge_scene_videos as util_merge_scene_videos
from .seedvr2_cli_core import process_video_with_seedvr2_cli
from .common_utils import format_time


def process_seedvr2_with_scenes(
    input_video_path: str,
    seedvr2_config,
    
    # Scene split parameters
    scene_split_params: Dict[str, Any],
    
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
    
    # Output parameters
    output_folder: str = "output",
    temp_folder: str = "temp",
    create_comparison_video: bool = False,
    
    # Session directory management
    session_output_dir: Optional[str] = None,
    base_output_filename_no_ext: Optional[str] = None,
    
    # Chunk settings
    max_chunk_len: int = 25,
    
    # Progress callback
    progress_callback: Optional[callable] = None,
    status_callback: Optional[callable] = None,
    scene_progress_callback: Optional[callable] = None,
    
    # Global settings
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    seed: int = -1,
    
    logger: Optional[logging.Logger] = None
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """
    Process video with SeedVR2 using scene-based approach.
    
    This function:
    1. Splits the input video into scenes
    2. Processes each scene individually with SeedVR2
    3. Merges the processed scenes back together
    4. Handles cancellation and partial results properly
    """
    if logger:
        logger.info("Starting SeedVR2 scene-based processing")
    
    # Create temp directory for scene processing
    scene_temp_dir = os.path.join(temp_folder, f"seedvr2_scenes_{int(time.time())}")
    os.makedirs(scene_temp_dir, exist_ok=True)
    
    # Initialize tracking variables
    processed_scene_videos = []
    total_scenes = 0
    current_scene = 0
    scene_start_time = time.time()
    
    try:
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        # Step 1: Split video into scenes
        yield (None, "🎬 Analyzing and splitting video into scenes...", None, "Scene detection...", None)
        
        if scene_progress_callback:
            scene_progress_callback(0.0, "Detecting scenes...")
        
        scene_video_paths = util_split_video_into_scenes(
            input_video_path,
            scene_temp_dir,
            scene_split_params,
            scene_progress_callback,
            logger=logger
        )
        
        if not scene_video_paths:
            if logger:
                logger.warning("No scenes detected, falling back to direct processing")
            # Fall back to direct processing without scenes
            yield from process_video_with_seedvr2_cli(
                input_video_path=input_video_path,
                seedvr2_config=seedvr2_config,
                enable_target_res=enable_target_res,
                target_h=target_h,
                target_w=target_w,
                target_res_mode=target_res_mode,
                save_frames=save_frames,
                save_metadata=save_metadata,
                save_chunks=save_chunks,
                save_chunk_frames=save_chunk_frames,
                enable_scene_split=False,  # Disable scene split for fallback
                output_folder=output_folder,
                temp_folder=temp_folder,
                create_comparison_video=create_comparison_video,
                session_output_dir=session_output_dir,
                base_output_filename_no_ext=base_output_filename_no_ext,
                max_chunk_len=max_chunk_len,
                progress_callback=progress_callback,
                status_callback=status_callback,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality=ffmpeg_quality,
                ffmpeg_use_gpu=ffmpeg_use_gpu,
                seed=seed,
                logger=logger
            )
            return
        
        total_scenes = len(scene_video_paths)
        if logger:
            logger.info(f"Video split into {total_scenes} scenes")
        
        yield (None, f"✂️ Video split into {total_scenes} scenes", None, f"{total_scenes} scenes detected", None)
        
        # Step 2: Process each scene individually
        for scene_idx, scene_video_path in enumerate(scene_video_paths):
            current_scene = scene_idx + 1
            scene_name = os.path.basename(scene_video_path)
            
            # Check for cancellation before processing each scene
            cancellation_manager.check_cancel()
            
            yield (None, f"🎬 Processing scene {current_scene}/{total_scenes}: {scene_name}", 
                   None, f"Scene {current_scene}/{total_scenes}", None)
            
            if logger:
                logger.info(f"Processing scene {current_scene}/{total_scenes}: {scene_video_path}")
            
            # Create scene-specific output directory
            scene_output_dir = os.path.join(session_output_dir if session_output_dir else output_folder, 
                                          f"scene_{scene_idx + 1:03d}")
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # Process this scene with SeedVR2
            scene_output_path = None
            scene_generator = process_video_with_seedvr2_cli(
                input_video_path=scene_video_path,
                seedvr2_config=seedvr2_config,
                enable_target_res=enable_target_res,
                target_h=target_h,
                target_w=target_w,
                target_res_mode=target_res_mode,
                save_frames=save_frames and (scene_idx == 0),  # Only save frames for first scene
                save_metadata=save_metadata,
                save_chunks=save_chunks,
                save_chunk_frames=save_chunk_frames,
                enable_scene_split=False,  # Don't recursively split scenes
                output_folder=scene_output_dir,
                temp_folder=scene_temp_dir,
                create_comparison_video=False,  # Create comparison only for final video
                session_output_dir=scene_output_dir,
                base_output_filename_no_ext=f"scene_{scene_idx + 1:03d}",
                max_chunk_len=max_chunk_len,
                progress_callback=lambda p, d: progress_callback(
                    (scene_idx + p) / total_scenes, 
                    f"Scene {current_scene}/{total_scenes}: {d}"
                ) if progress_callback else None,
                status_callback=status_callback,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality=ffmpeg_quality,
                ffmpeg_use_gpu=ffmpeg_use_gpu,
                seed=seed,
                logger=logger
            )
            
            # Process scene generator output
            for scene_result in scene_generator:
                scene_video, scene_status, chunk_video, chunk_status, comparison_video = scene_result
                
                if scene_video:
                    scene_output_path = scene_video
                
                # Yield progress updates with scene context
                yield (None, 
                       f"Scene {current_scene}/{total_scenes}: {scene_status}" if scene_status else None,
                       chunk_video,
                       f"Scene {current_scene}/{total_scenes}: {chunk_status}" if chunk_status else None,
                       None)  # Don't yield comparison videos for individual scenes
            
            if scene_output_path and os.path.exists(scene_output_path):
                processed_scene_videos.append(scene_output_path)
                if logger:
                    logger.info(f"Scene {current_scene} processed successfully: {scene_output_path}")
            else:
                if logger:
                    logger.warning(f"Scene {current_scene} processing failed or was cancelled")
        
        # Step 3: Merge processed scenes
        if processed_scene_videos:
            yield (None, f"🎬 Merging {len(processed_scene_videos)} processed scenes...", 
                   None, "Merging scenes...", None)
            
            # Determine final output path
            if session_output_dir and base_output_filename_no_ext:
                final_output_path = os.path.join(output_folder, f"{base_output_filename_no_ext}.mp4")
            else:
                final_output_path = os.path.join(output_folder, "seedvr2_merged_output.mp4")
            
            # Check if this is a partial result (not all scenes processed)
            is_partial = len(processed_scene_videos) < total_scenes
            if is_partial:
                # Add partial indicator to filename
                base_name = os.path.splitext(os.path.basename(final_output_path))[0]
                final_output_path = os.path.join(
                    os.path.dirname(final_output_path),
                    f"{base_name}_partial_cancelled_{len(processed_scene_videos)}_of_{total_scenes}_scenes.mp4"
                )
                
                if logger:
                    logger.info(f"Creating partial result with {len(processed_scene_videos)}/{total_scenes} scenes")
            
            # Merge the processed scenes
            success = util_merge_scene_videos(
                processed_scene_videos,
                final_output_path,
                temp_dir=scene_temp_dir,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality=ffmpeg_quality,
                use_gpu=ffmpeg_use_gpu,
                logger=logger,
                allow_partial_merge=is_partial
            )
            
            if success and os.path.exists(final_output_path):
                processing_time = time.time() - scene_start_time
                status_msg = (f"✅ {'Partial' if is_partial else 'Complete'} scene-based processing finished! "
                             f"Processed {len(processed_scene_videos)}/{total_scenes} scenes in {format_time(processing_time)}")
                
                if logger:
                    logger.info(status_msg)
                
                # Create comparison video if requested and not partial
                comparison_video_path = None
                if create_comparison_video and not is_partial:
                    comparison_video_path = os.path.join(
                        os.path.dirname(final_output_path),
                        f"{os.path.splitext(os.path.basename(final_output_path))[0]}_comparison.mp4"
                    )
                    # TODO: Implement comparison video generation for SeedVR2
                    # For now, we'll skip this
                    comparison_video_path = None
                
                yield (final_output_path, status_msg, None, "Processing complete", comparison_video_path)
            else:
                error_msg = "Failed to merge processed scenes"
                if logger:
                    logger.error(error_msg)
                yield (None, f"❌ {error_msg}", None, "Merge failed", None)
        else:
            # No scenes were processed successfully
            if logger:
                logger.warning("No scenes were processed successfully")
            yield (None, "❌ No scenes were processed successfully", None, "Processing failed", None)
    
    except CancelledError:
        # Handle cancellation - try to create partial result
        if logger:
            logger.info("Scene processing cancelled by user, attempting to create partial result")
        
        if processed_scene_videos:
            yield (None, f"⚠️ Processing cancelled, merging {len(processed_scene_videos)} completed scenes...", 
                   None, "Creating partial result...", None)
            
            # Create partial result
            base_name = base_output_filename_no_ext or "seedvr2_output"
            partial_output_path = os.path.join(
                output_folder,
                f"{base_name}_partial_cancelled_{len(processed_scene_videos)}_of_{total_scenes}_scenes.mp4"
            )
            
            success = util_merge_scene_videos(
                processed_scene_videos,
                partial_output_path,
                temp_dir=scene_temp_dir,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality=ffmpeg_quality,
                use_gpu=ffmpeg_use_gpu,
                logger=logger,
                allow_partial_merge=True
            )
            
            if success and os.path.exists(partial_output_path):
                processing_time = time.time() - scene_start_time
                status_msg = (f"⚠️ Partial result created: {len(processed_scene_videos)}/{total_scenes} scenes "
                             f"in {format_time(processing_time)}")
                yield (partial_output_path, status_msg, None, "Partial result ready", None)
            else:
                yield (None, "❌ Cancelled - failed to create partial result", None, "Cancelled", None)
        else:
            yield (None, "❌ Processing cancelled - no scenes completed", None, "Cancelled", None)
        
        raise  # Re-raise to maintain cancellation flow
    
    except Exception as e:
        if logger:
            logger.error(f"Error in scene-based processing: {e}", exc_info=True)
        yield (None, f"❌ Error: {str(e)}", None, "Error", None)
        raise
    
    finally:
        # Cleanup temp directory
        try:
            if os.path.exists(scene_temp_dir):
                shutil.rmtree(scene_temp_dir)
                if logger:
                    logger.info(f"Cleaned up temp directory: {scene_temp_dir}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup temp directory: {e}")