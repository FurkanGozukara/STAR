import os
import time
import tempfile
import shutil
import numpy as np
import gradio as gr
from pathlib import Path
from .ffmpeg_utils import run_ffmpeg_command
from .nvenc_utils import should_fallback_to_cpu_encoding
from .file_utils import get_next_filename, cleanup_temp_dir
from .cogvlm_utils import auto_caption, COG_VLM_AVAILABLE
from .cancellation_manager import cancellation_manager, CancelledError

def split_video_fallback(input_video_path, scene_list, output_dir, video_name, ffmpeg_args, logger, formatter):
    """
    Fallback method for splitting video when PySceneDetect fails.
    Uses our own ffmpeg commands with proper path escaping.
    """
    try:
        if logger:
            logger.info(f"Using fallback method to split {len(scene_list)} scenes")
        
        # Import required components for scene metadata
        from scenedetect.output import SceneMetadata, VideoMetadata
        
        # Create a dummy video metadata object with correct constructor signature
        # VideoMetadata requires path and total_scenes parameters
        video_metadata = VideoMetadata(path=input_video_path, total_scenes=len(scene_list))
        
        success_count = 0
        
        for scene_idx, (start_time, end_time) in enumerate(scene_list):
            # Check for cancellation
            cancellation_manager.check_cancel()
            
            # Create scene metadata for the formatter
            scene_metadata = SceneMetadata(start=start_time, end=end_time)
            
            # Get the output filename using the formatter
            output_filename = formatter(video_metadata, scene_metadata)
            output_path = os.path.join(output_dir, output_filename)
            
            # Calculate duration in seconds
            start_seconds = start_time.get_seconds()
            end_seconds = end_time.get_seconds()
            duration = end_seconds - start_seconds
            
            # Build ffmpeg command with proper path escaping and Linux-friendly options
            # Use more compatible audio/subtitle mapping for Linux
            cmd = f'ffmpeg -y -ss {start_seconds:.6f} -i "{input_video_path}" -t {duration:.6f} {ffmpeg_args} "{output_path}"'
            
            if logger:
                logger.info(f"Splitting scene {scene_idx + 1}/{len(scene_list)}: {start_time.get_timecode()} - {end_time.get_timecode()}")
            
            try:
                # Use our existing run_ffmpeg_command function
                success = run_ffmpeg_command(cmd, f"Scene {scene_idx + 1} Split", logger, raise_on_error=False)
                
                if success and os.path.exists(output_path):
                    success_count += 1
                    if logger:
                        logger.debug(f"Successfully created scene {scene_idx + 1}: {output_path}")
                else:
                    if logger:
                        logger.error(f"Failed to create scene {scene_idx + 1}: {output_path}")
                    return 1  # Return non-zero for failure
                        
            except Exception as e:
                if logger:
                    logger.error(f"Error splitting scene {scene_idx + 1}: {e}")
                return 1  # Return non-zero for failure
        
        if logger:
            logger.info(f"Fallback scene splitting completed: {success_count}/{len(scene_list)} scenes created")
        
        return 0 if success_count == len(scene_list) else 1
        
    except Exception as e:
        if logger:
            logger.error(f"Error in fallback scene splitting: {e}")
        return 1

def split_video_into_scenes(input_video_path, temp_dir, scene_split_params, progress_callback=None, logger=None):
    """Split video into scenes using PySceneDetect."""
    try:
        # Import PySceneDetect components
        from scenedetect import (
            AdaptiveDetector, ContentDetector, CropRegion, FrameTimecode,
            Interpolation, SceneManager, open_video
        )
        from scenedetect.backends import AVAILABLE_BACKENDS
        from scenedetect.output import (
            SceneMetadata, VideoMetadata, is_ffmpeg_available, is_mkvmerge_available,
            split_video_ffmpeg, split_video_mkvmerge
        )
        import scenedetect

        if logger:
            logger.info(f"Starting scene detection for: {input_video_path}")

        scenes_dir = os.path.join(temp_dir, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)

        video = open_video(str(input_video_path), backend="opencv")

        stats_manager = scenedetect.StatsManager() if scene_split_params.get('save_stats') else None
        scene_manager = SceneManager(stats_manager=stats_manager)

        if scene_split_params['split_mode'] == 'automatic':
            # Check for cancellation at start of automatic detection
            cancellation_manager.check_cancel()
            
            # Automatic scene detection
            detector_weights = ContentDetector.Components(
                delta_hue=scene_split_params['weights'][0],
                delta_sat=scene_split_params['weights'][1],
                delta_lum=scene_split_params['weights'][2],
                delta_edges=scene_split_params['weights'][3]
            )

            min_scene_len_ft = FrameTimecode(scene_split_params['min_scene_len'], video.frame_rate)

            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=scene_split_params['threshold'],
                    min_content_val=scene_split_params['min_content_val'],
                    window_width=scene_split_params['frame_window'],
                    weights=detector_weights,
                    kernel_size=scene_split_params.get('kernel_size'),
                    min_scene_len=min_scene_len_ft
                )
            )

            if progress_callback:
                progress_callback(0.3, "Detecting scenes...")

            num_frames_processed = scene_manager.detect_scenes(
                video=video,
                frame_skip=scene_split_params['frame_skip'],
                show_progress=False
            )

            scene_list = scene_manager.get_scene_list(start_in_scene=True)

            global_min_scene_len_ft = FrameTimecode(scene_split_params['min_scene_len'], video.frame_rate)
            filtered_scene_list = []

            if scene_list:
                for i, (start, end) in enumerate(scene_list):
                    # Check for cancellation during scene filtering
                    cancellation_manager.check_cancel()
                    
                    current_scene_duration = end - start
                    if current_scene_duration < global_min_scene_len_ft:
                        if scene_split_params['drop_short_scenes']:
                            if logger:
                                logger.info(f"Dropping short scene ({i+1}): {start.get_timecode()} - {end.get_timecode()}")
                            continue
                        elif filtered_scene_list and not (scene_split_params['merge_last_scene'] and i == len(scene_list) - 1):
                            prev_start, _ = filtered_scene_list.pop()
                            filtered_scene_list.append((prev_start, end))
                            if logger:
                                logger.info(f"Merging short scene ({i+1}) with previous")
                            continue
                    filtered_scene_list.append((start, end))

                scene_list = filtered_scene_list

                if scene_split_params['merge_last_scene'] and len(scene_list) > 1:
                    last_start, last_end = scene_list[-1]
                    if (last_end - last_start) < global_min_scene_len_ft:
                        prev_start, _ = scene_list[-2]
                        scene_list = scene_list[:-2] + [(prev_start, last_end)]
        else:
            # Check for cancellation at start of manual splitting
            cancellation_manager.check_cancel()
            
            # Manual scene splitting
            if progress_callback:
                progress_callback(0.3, "Calculating manual split points...")

            total_frames = int(video.frame_rate * video.duration.get_seconds())
            scene_list = []

            if scene_split_params['manual_split_type'] == 'duration':
                split_duration_seconds = scene_split_params['manual_split_value']
                split_duration_frames = int(split_duration_seconds * video.frame_rate)
            else:
                split_duration_frames = scene_split_params['manual_split_value']

            current_frame = 0
            scene_num = 1
            while current_frame < total_frames:
                start_frame = current_frame
                end_frame = min(current_frame + split_duration_frames, total_frames)

                start_tc = FrameTimecode(start_frame, video.frame_rate)
                end_tc = FrameTimecode(end_frame, video.frame_rate)

                scene_list.append((start_tc, end_tc))
                current_frame = end_frame
                scene_num += 1

        if not scene_list:
            if logger:
                logger.warning("No scenes detected, using entire video as single scene")
            scene_list = [(FrameTimecode(0, video.frame_rate), video.duration)]

        if logger:
            logger.info(f"Found {len(scene_list)} scenes to split")
            # Add logging for each scene's start and end frames/times
            if logger and scene_list:
                logger.info("Detected scene list (start_frame, end_frame, start_time, end_time):")
                for i, (start, end) in enumerate(scene_list):
                    logger.info(
                        f"Scene {i+1}: "
                        f"Frames {start.get_frames()} - {end.get_frames()} | "
                        f"Time {start.get_timecode()} - {end.get_timecode()}"
                    )

        if progress_callback:
            progress_callback(0.6, f"Splitting video into {len(scene_list)} scenes...")

        def scene_filename_formatter(video, scene):
            scene_num = scene_list.index((scene.start, scene.end)) + 1
            return f"scene_{scene_num:04d}.mp4"

        # Sanitize video name to prevent issues with spaces and special characters in filenames
        raw_video_name = Path(input_video_path).stem
        # Use the existing sanitize_filename function for proper cross-platform filename handling
        from .file_utils import sanitize_filename
        sanitized_video_name = sanitize_filename(raw_video_name)
        
        if scene_split_params['use_mkvmerge'] and is_mkvmerge_available():
            return_code = split_video_mkvmerge(
                input_video_path=str(input_video_path),
                scene_list=scene_list,
                output_dir=scenes_dir,
                video_name=sanitized_video_name,
                show_output=not scene_split_params['quiet_ffmpeg'],
                formatter=scene_filename_formatter
            )
        else:
            # Use FFmpeg for splitting
            if scene_split_params['copy_streams']:
                # Use more compatible audio/subtitle mapping for cross-platform compatibility
                # Try to include audio but don't fail if not present
                ffmpeg_args = "-map 0:v:0 -map 0:a? -map 0:s? -c:v copy -c:a copy -avoid_negative_ts make_zero"
            else:
                # Check if GPU encoding is requested and available
                if scene_split_params.get('use_gpu', False):
                    # Get video resolution to check if it's too small for NVENC
                    try:
                        from .file_utils import get_video_resolution
                        orig_h, orig_w = get_video_resolution(input_video_path, logger)
                        use_cpu_fallback = should_fallback_to_cpu_encoding(orig_w, orig_h, logger)
                    except Exception as e:
                        if logger:
                            logger.warning(f"Could not get video resolution for NVENC check: {e}, using CPU fallback")
                        use_cpu_fallback = True
                    
                    # Additional NVENC availability check for Linux compatibility
                    nvenc_available = False
                    if not use_cpu_fallback:
                        try:
                            # Test if NVENC is available by running a quick ffmpeg command
                            # Use a more robust test that works on both Windows and Linux
                            test_cmd = 'ffmpeg -loglevel error -f lavfi -i color=c=black:s=64x64:d=0.1:r=1 -c:v h264_nvenc -preset fast -f null -'
                            test_result = run_ffmpeg_command(test_cmd, "NVENC Test", logger, raise_on_error=False)
                            nvenc_available = test_result
                            if logger:
                                logger.info(f"NVENC availability test: {'PASSED' if nvenc_available else 'FAILED'}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"NVENC availability test failed: {e}")
                            nvenc_available = False
                    
                    if not use_cpu_fallback and nvenc_available:
                        nvenc_preset = scene_split_params['preset']
                        if scene_split_params['preset'] in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
                            nvenc_preset = "fast"
                        elif scene_split_params['preset'] in ["slower", "veryslow"]:
                            nvenc_preset = "slow"
                        
                        # Use safer audio mapping with ? suffix for cross-platform compatibility
                        ffmpeg_args = f"-map 0:v:0 -map 0:a? -map 0:s? -c:v h264_nvenc -preset:v {nvenc_preset} -cq:v {scene_split_params['rate_factor']} -pix_fmt yuv420p -c:a aac -avoid_negative_ts make_zero"
                    else:
                        if logger:
                            reason = "resolution constraints" if use_cpu_fallback else "NVENC not available"
                            logger.info(f"Falling back to CPU encoding for scene splitting due to {reason}: {orig_w}x{orig_h}")
                        # Use safer audio mapping with ? suffix for cross-platform compatibility
                        ffmpeg_args = f"-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset {scene_split_params['preset']} -crf {scene_split_params['rate_factor']} -c:a aac -avoid_negative_ts make_zero"
                else:
                    # Use safer audio mapping with ? suffix for cross-platform compatibility
                    ffmpeg_args = f"-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset {scene_split_params['preset']} -crf {scene_split_params['rate_factor']} -c:a aac -avoid_negative_ts make_zero"

            # Debug: Log the parameters being passed to PySceneDetect
            if logger:
                logger.info(f"PySceneDetect parameters:")
                logger.info(f"  input_video_path: {input_video_path}")
                logger.info(f"  scenes_dir: {scenes_dir}")
                logger.info(f"  sanitized_video_name: {sanitized_video_name}")
                logger.info(f"  ffmpeg_args: {ffmpeg_args}")
                logger.info(f"  scene_count: {len(scene_list)}")
            
            return_code = split_video_ffmpeg(
                input_video_path=str(input_video_path),
                scene_list=scene_list,
                output_dir=scenes_dir,
                video_name=sanitized_video_name,
                arg_override=ffmpeg_args,
                show_progress=scene_split_params['show_progress'],
                show_output=not scene_split_params['quiet_ffmpeg'],
                formatter=scene_filename_formatter
            )
            
            # Debug: Log the return code
            if logger:
                logger.info(f"PySceneDetect split_video_ffmpeg returned code: {return_code}")

        if return_code != 0:
            if logger:
                logger.warning(f"PySceneDetect failed with return code {return_code}, trying fallback method")
            
            # Fallback: Use our own ffmpeg commands to split the video
            try:
                return_code = split_video_fallback(
                    input_video_path=str(input_video_path),
                    scene_list=scene_list,
                    output_dir=scenes_dir,
                    video_name=sanitized_video_name,
                    ffmpeg_args=ffmpeg_args,
                    logger=logger,
                    formatter=scene_filename_formatter
                )
                
                if return_code != 0:
                    # If fallback also failed and we were using GPU encoding, try CPU encoding
                    if 'h264_nvenc' in ffmpeg_args:
                        if logger:
                            logger.warning("Fallback with GPU encoding failed, trying CPU encoding")
                        
                        # Force CPU encoding for fallback
                        cpu_ffmpeg_args = ffmpeg_args.replace('h264_nvenc', 'libx264').replace('-preset:v', '-preset').replace('-cq:v', '-crf')
                        
                        return_code = split_video_fallback(
                            input_video_path=str(input_video_path),
                            scene_list=scene_list,
                            output_dir=scenes_dir,
                            video_name=sanitized_video_name,
                            ffmpeg_args=cpu_ffmpeg_args,
                            logger=logger,
                            formatter=scene_filename_formatter
                        )
                        
                        if return_code != 0:
                            raise Exception(f"Scene splitting failed with return code: {return_code}")
                        else:
                            if logger:
                                logger.info("Scene splitting succeeded using CPU encoding fallback")
                    else:
                        raise Exception(f"Scene splitting failed with return code: {return_code}")
                else:
                    if logger:
                        logger.info("Scene splitting succeeded using fallback method")
                        
            except Exception as fallback_error:
                if logger:
                    logger.error(f"Fallback scene splitting also failed: {fallback_error}")
                raise Exception(f"Scene splitting failed with return code: {return_code}")

        scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith('.mp4')])
        scene_paths = [os.path.join(scenes_dir, f) for f in scene_files]

        if logger:
            logger.info(f"Successfully split video into {len(scene_paths)} scene files")

        if progress_callback:
            progress_callback(1.0, f"Scene splitting complete: {len(scene_paths)} scenes")

        return scene_paths

    except Exception as e:
        if logger:
            logger.error(f"Error during scene splitting: {e}")
        raise gr.Error(f"Scene splitting failed: {e}")
    finally:
        if 'video' in locals():
            del video

def merge_scene_videos(scene_video_paths, output_path, temp_dir, ffmpeg_preset="medium", ffmpeg_quality=23, use_gpu=False, logger=None, allow_partial_merge=False):
    """Merge multiple scene videos into a single output video."""
    # Check for cancellation at start of merge process, but allow partial merges to proceed
    if not allow_partial_merge:
        cancellation_manager.check_cancel()
    
    if not scene_video_paths:
        if logger:
            logger.warning("No scene videos to merge")
        return False

    try:
        if logger:
            logger.info(f"Merging {len(scene_video_paths)} scene videos into: {output_path}")

        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for scene_path in scene_video_paths:
                # Normalize path for FFmpeg
                scene_path_normalized = scene_path.replace('\\', '/')
                f.write(f"file '{scene_path_normalized}'\n")

        # Check if we need to fallback to CPU due to small resolution
        use_cpu_fallback = False
        if use_gpu and scene_video_paths:
            try:
                from .file_utils import get_video_resolution
                scene_h, scene_w = get_video_resolution(scene_video_paths[0], logger)
                use_cpu_fallback = should_fallback_to_cpu_encoding(scene_w, scene_h, logger)
            except Exception as e:
                if logger:
                    logger.warning(f"Could not get scene video resolution for NVENC check: {e}, using CPU fallback")
                use_cpu_fallback = True

        if use_gpu and not use_cpu_fallback:
            nvenc_preset = ffmpeg_preset
            if ffmpeg_preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
                nvenc_preset = "fast"
            elif ffmpeg_preset in ["slower", "veryslow"]:
                nvenc_preset = "slow"

            ffmpeg_opts = f'-c:v h264_nvenc -preset:v {nvenc_preset} -cq:v {ffmpeg_quality} -pix_fmt yuv420p'
        else:
            if use_cpu_fallback and logger:
                logger.info(f"Falling back to CPU encoding for scene merging due to resolution constraints: {scene_w}x{scene_h}")
            ffmpeg_opts = f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality} -pix_fmt yuv420p'

        cmd = f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" {ffmpeg_opts} -c:a copy "{output_path}"'

        run_ffmpeg_command(cmd, "Scene Video Merging", logger)

        if logger:
            logger.info(f"Successfully merged scene videos into: {output_path}")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Error merging scene videos: {e}")
        raise gr.Error(f"Failed to merge scene videos: {e}")

def split_video_only(
    input_video_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
    scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
    scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
    scene_manual_split_type_radio_val, scene_manual_split_value_num_val,
    default_output_dir, logger=None, progress=gr.Progress(track_tqdm=True)
):
    """Split video into scenes only (no upscaling)."""
    if not input_video_val or not os.path.exists(input_video_val):
        raise gr.Error("Please upload a valid input video.")

    try:
        progress(0.1, desc="Initializing scene splitting...")

        run_id = f"split_only_{int(time.time())}_{np.random.randint(1000, 9999)}"
        temp_dir_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_dir_base, run_id)
        os.makedirs(temp_dir, exist_ok=True)

        scene_split_params = {
            'split_mode': scene_split_mode_radio_val,
            'min_scene_len': scene_min_scene_len_num_val,
            'drop_short_scenes': scene_drop_short_check_val,
            'merge_last_scene': scene_merge_last_check_val,
            'frame_skip': scene_frame_skip_num_val,
            'threshold': scene_threshold_num_val,
            'min_content_val': scene_min_content_val_num_val,
            'frame_window': scene_frame_window_num_val,
            'weights': [1.0, 1.0, 1.0, 0.0],
            'copy_streams': scene_copy_streams_check_val,
            'use_mkvmerge': scene_use_mkvmerge_check_val,
            'rate_factor': scene_rate_factor_num_val,
            'preset': scene_preset_dropdown_val,
            'quiet_ffmpeg': scene_quiet_ffmpeg_check_val,
            'show_progress': True,
            'manual_split_type': scene_manual_split_type_radio_val,
            'manual_split_value': scene_manual_split_value_num_val,
            'use_gpu': False  # Default to False for split-only functionality
        }

        def scene_progress_callback(progress_val, desc):
            progress(0.1 + (progress_val * 0.8), desc=desc)

        scene_video_paths = split_video_into_scenes(
            input_video_val,
            temp_dir,
            scene_split_params,
            scene_progress_callback,
            logger
        )

        base_output_filename_no_ext, _ = get_next_filename(default_output_dir, logger)
        split_output_dir = os.path.join(default_output_dir, f"{base_output_filename_no_ext}_scenes")
        os.makedirs(split_output_dir, exist_ok=True)

        progress(0.9, desc="Copying scene files...")
        copied_scenes = []
        for i, scene_path in enumerate(scene_video_paths):
            scene_filename = f"scene_{i+1:04d}.mp4"
            output_scene_path = os.path.join(split_output_dir, scene_filename)
            shutil.copy2(scene_path, output_scene_path)
            copied_scenes.append(output_scene_path)

        progress(1.0, desc="Scene splitting complete!")

        cleanup_temp_dir(temp_dir, logger)

        status_msg = f"Successfully split video into {len(copied_scenes)} scenes.\nOutput folder: {split_output_dir}"
        return None, status_msg

    except Exception as e:
        if logger:
            logger.error(f"Error during split-only operation: {e}")
        if 'temp_dir' in locals():
            cleanup_temp_dir(temp_dir, logger)
        raise gr.Error(f"Scene splitting failed: {e}") 