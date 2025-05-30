"""
Batch video processing operations.
Handles processing multiple videos in sequence.
"""

import os
import gradio as gr
from pathlib import Path


def process_batch_videos(
    batch_input_folder_val, batch_output_folder_val,
    user_prompt_val, pos_prompt_val, neg_prompt_val, model_selector_val,
    upscale_factor_slider_val, cfg_slider_val, steps_slider_val, solver_mode_radio_val,
    max_chunk_len_slider_val, vae_chunk_slider_val, color_fix_dropdown_val,
    enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
    enable_sliding_window_check_val, window_size_num_val, window_step_num_val,
    enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
    ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
    save_frames_checkbox_val, save_metadata_checkbox_val, save_chunks_checkbox_val,
    create_comparison_video_check_val,

    enable_scene_split_check_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
    scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
    scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
    scene_manual_split_type_radio_val, scene_manual_split_value_num_val,

    run_upscale_func,  # Pass the run_upscale function as a parameter
    logger,           # Pass the logger as a parameter
    progress=gr.Progress(track_tqdm=True)
):
    """
    Process multiple videos in batch mode.
    
    This function takes a folder of videos and processes each one using the same settings.
    Progress is tracked and reported for the entire batch.
    
    Args:
        batch_input_folder_val: Path to folder containing input videos
        batch_output_folder_val: Path to folder where output videos will be saved
        (... all other parameters are the same as single video processing ...)
        run_upscale_func: The run_upscale function to use for processing each video
        logger: Logger instance for reporting progress and errors
        progress: Gradio progress tracker
        
    Returns:
        tuple: (None, status_message) - None for video output, status message for UI
        
    Raises:
        gr.Error: If input validation fails or processing encounters errors
    """
    if not batch_input_folder_val or not os.path.exists(batch_input_folder_val):
        raise gr.Error("Please provide a valid input folder.")

    if not batch_output_folder_val:
        raise gr.Error("Please provide an output folder.")

    try:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        video_files = []

        for file in os.listdir(batch_input_folder_val):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(batch_input_folder_val, file))

        if not video_files:
            raise gr.Error(f"No video files found in: {batch_input_folder_val}")

        os.makedirs(batch_output_folder_val, exist_ok=True)

        processed_files = []
        failed_files = []

        for i, video_file in enumerate(video_files):
            try:
                progress((i / len(video_files)) * 0.9, desc=f"Processing {i + 1}/{len(video_files)}: {Path(video_file).name}")

                upscale_generator = run_upscale_func(
                    video_file, user_prompt_val, pos_prompt_val, neg_prompt_val, model_selector_val,
                    upscale_factor_slider_val, cfg_slider_val, steps_slider_val, solver_mode_radio_val,
                    max_chunk_len_slider_val, vae_chunk_slider_val, color_fix_dropdown_val,
                    enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
                    enable_sliding_window_check_val, window_size_num_val, window_step_num_val,
                    enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
                    ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
                    save_frames_checkbox_val, save_metadata_checkbox_val, save_chunks_checkbox_val,

                    enable_scene_split_check_val, scene_split_mode_radio_val, scene_min_scene_len_num_val, scene_drop_short_check_val, scene_merge_last_check_val,
                    scene_frame_skip_num_val, scene_threshold_num_val, scene_min_content_val_num_val, scene_frame_window_num_val,
                    scene_copy_streams_check_val, scene_use_mkvmerge_check_val, scene_rate_factor_num_val, scene_preset_dropdown_val, scene_quiet_ffmpeg_check_val,
                    scene_manual_split_type_radio_val, scene_manual_split_value_num_val,

                    create_comparison_video_enabled=create_comparison_video_check_val,

                    is_batch_mode=True, batch_output_dir=batch_output_folder_val, original_filename=video_file,

                    enable_auto_caption_per_scene=False,
                    cogvlm_quant=None,
                    cogvlm_unload=None,
                    progress=progress
                )

                final_output = None
                for output_val, status_val, _, _ in upscale_generator:
                    final_output = output_val

                if final_output and os.path.exists(final_output):
                    processed_files.append({"input": video_file, "output": final_output, "status": "success"})
                else:
                    failed_files.append((video_file, "Output file not created"))

            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                failed_files.append((video_file, str(e)))

        progress(1.0, desc=f"Batch processing complete: {len(processed_files)} successful, {len(failed_files)} failed")

        status_msg = f"Batch processing complete!\n"
        status_msg += f"Successfully processed: {len(processed_files)} videos\n"
        status_msg += f"Failed: {len(failed_files)} videos\n"
        status_msg += f"Output folder: {batch_output_folder_val}"

        if failed_files:
            status_msg += f"\n\nFailed files:\n"
            for file_path, error in failed_files[:5]:
                status_msg += f"- {Path(file_path).name}: {error}\n"
            if len(failed_files) > 5:
                status_msg += f"... and {len(failed_files) - 5} more"

        return None, status_msg

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise gr.Error(f"Batch processing failed: {e}") 