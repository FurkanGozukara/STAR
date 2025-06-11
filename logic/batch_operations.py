"""
Batch video processing operations.
Handles processing multiple videos in sequence.
"""

import os
import gradio as gr
from pathlib import Path


def get_prompt_file_path(video_file_path):
    """Get the path to the corresponding .txt file for a video file."""
    video_path = Path(video_file_path)
    return video_path.parent / f"{video_path.stem}.txt"


def load_prompt_from_file(prompt_file_path, logger=None):
    """Load prompt from a text file. Returns None if file doesn't exist or can't be read."""
    try:
        if prompt_file_path.exists():
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                if logger:
                    logger.info(f"Loaded prompt from file: {prompt_file_path}")
                return prompt
        return None
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read prompt file {prompt_file_path}: {e}")
        return None


def save_caption_to_file(caption_text, prompt_file_path, overwrite=False, logger=None):
    """Save caption to a text file. Returns True if saved, False if skipped."""
    try:
        if prompt_file_path.exists() and not overwrite:
            if logger:
                logger.info(f"Prompt file already exists, not overwriting: {prompt_file_path}")
            return False
        
        # Create directory if it doesn't exist
        prompt_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(prompt_file_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        
        if logger:
            logger.info(f"Saved caption to file: {prompt_file_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to save caption to file {prompt_file_path}: {e}")
        return False


def get_expected_output_path(video_file, batch_output_folder):
    """Get the expected output path for a video file."""
    from .file_utils import sanitize_filename
    base_name = Path(video_file).stem
    sanitized_name = sanitize_filename(base_name)
    expected_output = os.path.join(batch_output_folder, sanitized_name, f"{sanitized_name}.mp4")
    return expected_output


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

    # FPS decrease parameters for batch processing
    enable_fps_decrease_val, target_fps_val, fps_interpolation_method_val,
    
    # RIFE interpolation parameters for batch processing
    enable_rife_interpolation_val, rife_multiplier_val, rife_fp16_val, rife_uhd_val, rife_scale_val,
    rife_skip_static_val, rife_enable_fps_limit_val, rife_max_fps_limit_val,
    rife_apply_to_chunks_val, rife_apply_to_scenes_val, rife_keep_original_val, rife_overwrite_original_val,

    run_upscale_func,  # Pass the run_upscale function as a parameter
    logger,           # Pass the logger as a parameter

    # NEW BATCH-SPECIFIC PARAMETERS
    batch_skip_existing_val=True,
    batch_use_prompt_files_val=True, 
    batch_save_captions_val=True,
    batch_enable_auto_caption_val=False,
    batch_cogvlm_quant_val=None,
    batch_cogvlm_unload_val=None,
    current_seed=99,

    progress=gr.Progress(track_tqdm=True)
):
    """
    Process multiple videos in batch mode with enhanced features.
    
    This function takes a folder of videos and processes each one using the same settings.
    Features include skip existing, prompt file loading, and caption saving.
    
    Args:
        batch_input_folder_val: Path to folder containing input videos
        batch_output_folder_val: Path to folder where output videos will be saved
        (... core upscaling parameters are the same as single video processing ...)
        run_upscale_func: The run_upscale function to use for processing each video
        logger: Logger instance for reporting progress and errors
        batch_skip_existing_val: Skip processing if output file already exists
        batch_use_prompt_files_val: Look for filename.txt files to use as prompts
        batch_save_captions_val: Save auto-generated captions as filename.txt in input folder
        batch_enable_auto_caption_val: Enable auto-captioning for videos without prompt files
        batch_cogvlm_quant_val: CogVLM quantization for batch auto-captioning
        batch_cogvlm_unload_val: CogVLM memory management for batch auto-captioning
        progress: Gradio progress tracker
        
    Returns:
        tuple: (None, status_message) - None for video output, comprehensive status message for UI
        
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
        skipped_files = []
        caption_stats = {"saved": 0, "skipped": 0, "failed": 0}

        for i, video_file in enumerate(video_files):
            try:
                video_name = Path(video_file).name
                progress((i / len(video_files)) * 0.9, desc=f"Processing {i + 1}/{len(video_files)}: {video_name}")

                # 1. Check if output exists and skip if requested
                if batch_skip_existing_val:
                    expected_output = get_expected_output_path(video_file, batch_output_folder_val)
                    if os.path.exists(expected_output):
                        logger.info(f"Skipping {video_name} - output already exists: {expected_output}")
                        skipped_files.append(video_file)
                        continue

                # 2. Determine prompt to use (priority: prompt file > auto-caption > user prompt)
                effective_prompt = user_prompt_val
                prompt_source = "user_input"
                
                if batch_use_prompt_files_val:
                    prompt_file_path = get_prompt_file_path(video_file)
                    file_prompt = load_prompt_from_file(prompt_file_path, logger)
                    if file_prompt:
                        effective_prompt = file_prompt
                        prompt_source = "prompt_file"
                        logger.info(f"Using prompt from file for {video_name}: '{effective_prompt[:50]}...'")

                # 3. Handle auto-captioning if enabled and no prompt file exists
                enable_auto_caption_for_this_video = False
                if batch_enable_auto_caption_val and prompt_source != "prompt_file":
                    enable_auto_caption_for_this_video = True
                    prompt_source = "auto_caption"
                    logger.info(f"Will auto-caption {video_name} (no prompt file found)")

                # 4. Process the video
                upscale_generator = run_upscale_func(
                    input_video_path=video_file, 
                    user_prompt=effective_prompt, 
                    positive_prompt=pos_prompt_val, 
                    negative_prompt=neg_prompt_val, 
                    model_choice=model_selector_val,
                    upscale_factor_slider=upscale_factor_slider_val, 
                    cfg_scale=cfg_slider_val, 
                    steps=steps_slider_val, 
                    solver_mode=solver_mode_radio_val,
                    max_chunk_len=max_chunk_len_slider_val, 
                    vae_chunk=vae_chunk_slider_val, 
                    color_fix_method=color_fix_dropdown_val,
                    enable_tiling=enable_tiling_check_val, 
                    tile_size=tile_size_num_val, 
                    tile_overlap=tile_overlap_num_val,
                    enable_sliding_window=enable_sliding_window_check_val, 
                    window_size=window_size_num_val, 
                    window_step=window_step_num_val,
                    enable_target_res=enable_target_res_check_val, 
                    target_h=target_h_num_val, 
                    target_w=target_w_num_val, 
                    target_res_mode=target_res_mode_radio_val,
                    ffmpeg_preset=ffmpeg_preset_dropdown_val, 
                    ffmpeg_quality_value=ffmpeg_quality_slider_val, 
                    ffmpeg_use_gpu=ffmpeg_use_gpu_check_val,
                    save_frames=save_frames_checkbox_val, 
                    save_metadata=save_metadata_checkbox_val, 
                    save_chunks=save_chunks_checkbox_val,

                    enable_scene_split=enable_scene_split_check_val, 
                    scene_split_mode=scene_split_mode_radio_val, 
                    scene_min_scene_len=scene_min_scene_len_num_val, 
                    scene_drop_short=scene_drop_short_check_val, 
                    scene_merge_last=scene_merge_last_check_val,
                    scene_frame_skip=scene_frame_skip_num_val, 
                    scene_threshold=scene_threshold_num_val, 
                    scene_min_content_val=scene_min_content_val_num_val, 
                    scene_frame_window=scene_frame_window_num_val,
                    scene_copy_streams=scene_copy_streams_check_val, 
                    scene_use_mkvmerge=scene_use_mkvmerge_check_val, 
                    scene_rate_factor=scene_rate_factor_num_val, 
                    scene_preset=scene_preset_dropdown_val, 
                    scene_quiet_ffmpeg=scene_quiet_ffmpeg_check_val,
                    scene_manual_split_type=scene_manual_split_type_radio_val, 
                    scene_manual_split_value=scene_manual_split_value_num_val,

                    create_comparison_video_enabled=create_comparison_video_check_val, # Passed here

                    # FPS decrease parameters for batch processing
                    enable_fps_decrease=enable_fps_decrease_val,
                    target_fps=target_fps_val,
                    fps_interpolation_method=fps_interpolation_method_val,

                    # RIFE interpolation parameters for batch processing
                    enable_rife_interpolation=enable_rife_interpolation_val,
                    rife_multiplier=rife_multiplier_val, 
                    rife_fp16=rife_fp16_val, 
                    rife_uhd=rife_uhd_val, 
                    rife_scale=rife_scale_val,
                    rife_skip_static=rife_skip_static_val, 
                    rife_enable_fps_limit=rife_enable_fps_limit_val, 
                    rife_max_fps_limit=rife_max_fps_limit_val,
                    rife_apply_to_chunks=rife_apply_to_chunks_val, 
                    rife_apply_to_scenes=rife_apply_to_scenes_val, 
                    rife_keep_original=rife_keep_original_val, 
                    rife_overwrite_original=rife_overwrite_original_val,

                    is_batch_mode=True,
                    batch_output_dir=batch_output_folder_val,
                    original_filename=video_name,
                    
                    enable_auto_caption_per_scene=enable_auto_caption_for_this_video,
                    cogvlm_quant=batch_cogvlm_quant_val if enable_auto_caption_for_this_video else 0,
                    cogvlm_unload=batch_cogvlm_unload_val if enable_auto_caption_for_this_video else 'full',

                    current_seed=current_seed,
                    progress=progress
                )

                final_output = None
                generated_caption = None
                
                # 5. Consume generator and extract caption if auto-captioning was used
                for output_val, status_val, chunk_vid_val, chunk_status_val, comparison_vid_val in upscale_generator:
                    final_output = output_val
                    
                    # Extract auto-generated caption from status if available
                    if enable_auto_caption_for_this_video and status_val and "Generated caption:" in status_val:
                        # Parse caption from status message
                        try:
                            caption_start = status_val.find("Generated caption:") + len("Generated caption:")
                            caption_line = status_val[caption_start:].split('\n')[0].strip()
                            if caption_line:
                                generated_caption = caption_line
                        except Exception as e:
                            logger.warning(f"Failed to extract caption from status for {video_name}: {e}")

                # 6. Save caption to input folder if requested and available
                if batch_save_captions_val and generated_caption and prompt_source == "auto_caption":
                    prompt_file_path = get_prompt_file_path(video_file)
                    if save_caption_to_file(generated_caption, prompt_file_path, overwrite=False, logger=logger):
                        caption_stats["saved"] += 1
                    else:
                        caption_stats["skipped"] += 1

                # 7. Record results
                if final_output and os.path.exists(final_output):
                    processed_files.append({
                        "input": video_file, 
                        "output": final_output, 
                        "status": "success",
                        "prompt_source": prompt_source,
                        "caption_saved": batch_save_captions_val and generated_caption and prompt_source == "auto_caption"
                    })
                else:
                    failed_files.append((video_file, "Output file not created"))
                    logger.warning(f"Failed to get final output path for {video_file} or file does not exist.")


            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}", exc_info=True)
                failed_files.append((video_file, str(e)))

        progress(1.0, desc=f"Batch complete: {len(processed_files)} processed, {len(skipped_files)} skipped, {len(failed_files)} failed")

        # Enhanced status reporting
        status_msg = f"🎉 Batch processing complete!\n\n"
        status_msg += f"📊 **Summary:**\n"
        status_msg += f"  ✅ Successfully processed: {len(processed_files)} videos\n"
        status_msg += f"  ⏩ Skipped (existing): {len(skipped_files)} videos\n"
        status_msg += f"  ❌ Failed: {len(failed_files)} videos\n"
        status_msg += f"  📁 Output folder: {batch_output_folder_val}\n\n"

        if batch_save_captions_val and any(caption_stats.values()):
            status_msg += f"📝 **Caption Management:**\n"
            status_msg += f"  💾 Captions saved: {caption_stats['saved']}\n"
            status_msg += f"  ⏩ Captions skipped (file exists): {caption_stats['skipped']}\n"
            if caption_stats['failed'] > 0:
                status_msg += f"  ❌ Caption save failed: {caption_stats['failed']}\n"
            status_msg += "\n"

        # Show prompt source breakdown
        prompt_sources = {}
        for file_info in processed_files:
            source = file_info.get('prompt_source', 'unknown')
            prompt_sources[source] = prompt_sources.get(source, 0) + 1

        if prompt_sources:
            status_msg += f"📋 **Prompt Sources Used:**\n"
            for source, count in prompt_sources.items():
                source_label = {
                    'prompt_file': '📄 Prompt files',
                    'auto_caption': '🤖 Auto-generated',
                    'user_input': '👤 User prompt'
                }.get(source, source)
                status_msg += f"  {source_label}: {count} videos\n"
            status_msg += "\n"

        if failed_files:
            status_msg += f"❌ **Failed Files:**\n"
            for file_path, error in failed_files[:5]:
                status_msg += f"  • {Path(file_path).name}: {error[:100]}...\n"
            if len(failed_files) > 5:
                status_msg += f"  • ... and {len(failed_files) - 5} more (check logs)\n"

        return None, status_msg

    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise gr.Error(f"Batch processing failed: {e}")