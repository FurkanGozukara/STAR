"""
Batch video processing operations.
Handles processing multiple videos in sequence.
"""

import os
import gradio as gr
from pathlib import Path
from .cancellation_manager import cancellation_manager, CancelledError


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
    max_chunk_len_slider_val, enable_chunk_optimization_check_val, vae_chunk_slider_val, color_fix_dropdown_val,
    enable_tiling_check_val, tile_size_num_val, tile_overlap_num_val,
    enable_context_window_check_val, context_overlap_num_val,
    enable_target_res_check_val, target_h_num_val, target_w_num_val, target_res_mode_radio_val,
    enable_auto_aspect_resolution_check_val, auto_resolution_status_display_val,
    ffmpeg_preset_dropdown_val, ffmpeg_quality_slider_val, ffmpeg_use_gpu_check_val,
    save_frames_checkbox_val, save_metadata_checkbox_val, save_chunks_checkbox_val, save_chunk_frames_checkbox_val,
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
    
    # Image upscaler parameters for batch processing
    enable_image_upscaler_val=False,
    image_upscaler_model_val=None,
    image_upscaler_batch_size_val=4,

    # Face restoration parameters for batch processing
    enable_face_restoration_val=False,
    face_restoration_fidelity_val=0.7,
    enable_face_colorization_val=False,
    face_restoration_timing_val="after_upscale",
    face_restoration_when_val="after",
    codeformer_model_val=None,
    face_restoration_batch_size_val=4,

    # NEW: Direct image upscaling parameter
    enable_direct_image_upscaling_val=False,

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
        enable_image_upscaler_val: Enable image-based upscaling instead of STAR
        image_upscaler_model_val: Selected image upscaler model filename
        image_upscaler_batch_size_val: Batch size for image upscaler processing
        enable_face_restoration_val: Enable CodeFormer face restoration
        face_restoration_fidelity_val: CodeFormer fidelity weight (0.0-1.0)
        enable_face_colorization_val: Enable grayscale face colorization
        face_restoration_timing_val: When to apply face restoration
        face_restoration_when_val: Before or after upscaling
        codeformer_model_val: Path to CodeFormer model
        face_restoration_batch_size_val: Batch size for face restoration processing
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
        # Define file extensions based on processing mode
        if enable_direct_image_upscaling_val:
            # Process image files directly
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
            input_files = []
            
            for file in os.listdir(batch_input_folder_val):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    input_files.append(os.path.join(batch_input_folder_val, file))
            
            if not input_files:
                raise gr.Error(f"No image files found in: {batch_input_folder_val}")
            
            file_type = "image"
            logger.info(f"Direct image upscaling mode: Found {len(input_files)} image files")
        else:
            # Process video files (existing logic)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            input_files = []

            for file in os.listdir(batch_input_folder_val):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    input_files.append(os.path.join(batch_input_folder_val, file))

            if not input_files:
                raise gr.Error(f"No video files found in: {batch_input_folder_val}")
            
            file_type = "video"
            logger.info(f"Video processing mode: Found {len(input_files)} video files")

        os.makedirs(batch_output_folder_val, exist_ok=True)

        processed_files = []
        failed_files = []
        skipped_files = []
        caption_stats = {"saved": 0, "skipped": 0, "failed": 0}

        for i, input_file in enumerate(input_files):
            try:
                # Check for cancellation at the start of each file
                cancellation_manager.check_cancel()
                
                file_name = Path(input_file).name
                progress((i / len(input_files)) * 0.9, desc=f"Processing {i + 1}/{len(input_files)}: {file_name}")
                
                # Handle direct image upscaling
                if enable_direct_image_upscaling_val and file_type == "image":
                    # Process image file directly with image upscaler
                    try:
                        from .image_upscaler_utils import process_single_image_direct
                        
                        # Generate output path for image
                        input_stem = Path(input_file).stem
                        input_ext = Path(input_file).suffix
                        output_image_path = os.path.join(batch_output_folder_val, f"{input_stem}_upscaled{input_ext}")
                        
                        # Check if output exists and skip if requested
                        if batch_skip_existing_val and os.path.exists(output_image_path):
                            logger.info(f"Skipping {file_name} - output already exists: {output_image_path}")
                            skipped_files.append(input_file)
                            continue
                        
                        # Force image upscaler mode for direct image processing
                        if not enable_image_upscaler_val:
                            logger.info(f"Auto-enabling image upscaler for direct image processing of {file_name}")
                        
                        # Determine upscale models directory
                        # Look for upscale_models in parent directories
                        potential_models_dirs = [
                            os.path.join(os.path.dirname(batch_input_folder_val), '..', 'upscale_models'),
                            os.path.join(os.path.dirname(batch_input_folder_val), 'upscale_models'),
                            'upscale_models'
                        ]
                        models_dir = None
                        for potential_dir in potential_models_dirs:
                            if os.path.exists(potential_dir):
                                models_dir = potential_dir
                                break
                        
                        if not models_dir:
                            models_dir = potential_models_dirs[0]  # Use first option as fallback
                        
                        # Process the image directly
                        success, result_path, processing_time = process_single_image_direct(
                            image_path=input_file,
                            output_path=output_image_path,
                            model_name=image_upscaler_model_val,
                            upscale_models_dir=models_dir,
                            apply_target_resolution=enable_target_res_check_val,
                            target_h=target_h_num_val,
                            target_w=target_w_num_val,
                            target_res_mode=target_res_mode_radio_val,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            logger=logger
                        )
                        
                        if success and result_path and os.path.exists(result_path):
                            processed_files.append({
                                "input": input_file,
                                "output": result_path,
                                "status": "success",
                                "prompt_source": "not_applicable",
                                "caption_saved": False,
                                "auto_resolution_used": False,
                                "effective_resolution": None,
                                "processing_time": processing_time
                            })
                            logger.info(f"Successfully processed image {file_name} in {processing_time:.2f}s")
                        else:
                            failed_files.append((input_file, "Image processing failed or output not created"))
                            logger.error(f"Failed to process image {file_name}")
                        
                        continue  # Skip to next file
                        
                    except Exception as e:
                        logger.error(f"Error processing image {file_name}: {e}")
                        failed_files.append((input_file, f"Image processing error: {str(e)}"))
                        continue

                # 1. Check if output exists and skip if requested (for video files)
                if batch_skip_existing_val:
                    expected_output = get_expected_output_path(input_file, batch_output_folder_val)
                    if os.path.exists(expected_output):
                        logger.info(f"Skipping {file_name} - output already exists: {expected_output}")
                        skipped_files.append(input_file)
                        continue

                # 2. Determine prompt to use (priority: prompt file > auto-caption > user prompt) - for video files only
                effective_prompt = user_prompt_val
                prompt_source = "user_input"
                
                if batch_use_prompt_files_val and file_type == "video":
                    prompt_file_path = get_prompt_file_path(input_file)
                    file_prompt = load_prompt_from_file(prompt_file_path, logger)
                    if file_prompt:
                        effective_prompt = file_prompt
                        prompt_source = "prompt_file"
                        logger.info(f"Using prompt from file for {file_name}: '{effective_prompt[:50]}...'")

                # 3. Handle auto-captioning if enabled and no prompt file exists (for video files only)
                # Note: Auto-captioning is disabled when using image upscaler since it doesn't use prompts
                enable_auto_caption_for_this_video = False
                if batch_enable_auto_caption_val and prompt_source != "prompt_file" and not enable_image_upscaler_val and file_type == "video":
                    enable_auto_caption_for_this_video = True
                    prompt_source = "auto_caption"
                    logger.info(f"Will auto-caption {file_name} (no prompt file found)")
                elif enable_image_upscaler_val and batch_enable_auto_caption_val:
                    logger.info(f"Auto-captioning disabled for {file_name} (using image upscaler - prompts not used)")

                # 3.5. Calculate auto-resolution for this file if enabled (for video files only)
                effective_target_h = target_h_num_val
                effective_target_w = target_w_num_val
                auto_resolution_used = False
                
                if enable_auto_aspect_resolution_check_val and file_type == "video":
                    try:
                        from .auto_resolution_utils import update_resolution_from_video
                        result = update_resolution_from_video(
                            input_file, target_w_num_val, target_h_num_val, logger
                        )
                        if result['success']:
                            effective_target_h = result['optimal_height']
                            effective_target_w = result['optimal_width']
                            auto_status = result['status_message']
                            
                            if effective_target_h != target_h_num_val or effective_target_w != target_w_num_val:
                                auto_resolution_used = True
                                logger.info(f"Auto-resolution for {file_name}: {target_w_num_val}x{target_h_num_val} → {effective_target_w}x{effective_target_h}")
                                logger.info(f"Auto-resolution status: {auto_status}")
                            else:
                                logger.info(f"Auto-resolution for {file_name}: No change needed ({auto_status})")
                        else:
                            logger.warning(f"Auto-resolution calculation failed for {file_name}: {result.get('error', 'Unknown error')}")
                            # Fall back to original target resolution
                            effective_target_h = target_h_num_val
                            effective_target_w = target_w_num_val
                    except Exception as e:
                        logger.warning(f"Auto-resolution calculation failed for {file_name}: {e}")
                        # Fall back to original target resolution
                        effective_target_h = target_h_num_val
                        effective_target_w = target_w_num_val

                # 4. Process the file (video processing)
                upscale_generator = run_upscale_func(
                    input_video_path=input_file, 
                    user_prompt=effective_prompt, 
                    positive_prompt=pos_prompt_val, 
                    negative_prompt=neg_prompt_val, 
                    model_choice=model_selector_val,
                    upscale_factor_slider=upscale_factor_slider_val, 
                    cfg_scale=cfg_slider_val, 
                    steps=steps_slider_val, 
                    solver_mode=solver_mode_radio_val,
                    max_chunk_len=max_chunk_len_slider_val, 
                    enable_chunk_optimization=enable_chunk_optimization_check_val,
                    vae_chunk=vae_chunk_slider_val, 
                    enable_vram_optimization=enable_vram_optimization_check_val,  # FIX: Added missing parameter
                    color_fix_method=color_fix_dropdown_val,
                    enable_tiling=enable_tiling_check_val, 
                    tile_size=tile_size_num_val, 
                    tile_overlap=tile_overlap_num_val,
                    enable_context_window=enable_context_window_check_val, 
                    context_overlap=context_overlap_num_val,
                    enable_target_res=enable_target_res_check_val, 
                    target_h=effective_target_h, 
                    target_w=effective_target_w, 
                    target_res_mode=target_res_mode_radio_val,
                    ffmpeg_preset=ffmpeg_preset_dropdown_val, 
                    ffmpeg_quality_value=ffmpeg_quality_slider_val, 
                    ffmpeg_use_gpu=ffmpeg_use_gpu_check_val,
                    save_frames=save_frames_checkbox_val, 
                    save_metadata=save_metadata_checkbox_val, 
                    save_chunks=save_chunks_checkbox_val,
                    save_chunk_frames=save_chunk_frames_checkbox_val,

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
                    fps_decrease_mode=fps_decrease_mode_val,
                    fps_multiplier_preset=fps_multiplier_preset_val,
                    fps_multiplier_custom=fps_multiplier_custom_val,
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
                    original_filename=file_name,
                    
                    enable_auto_caption_per_scene=enable_auto_caption_for_this_video,
                    cogvlm_quant=batch_cogvlm_quant_val if enable_auto_caption_for_this_video else 0,
                    cogvlm_unload=batch_cogvlm_unload_val if enable_auto_caption_for_this_video else 'full',
                    
                    # Image upscaler parameters for batch processing
                    enable_image_upscaler=enable_image_upscaler_val,
                    image_upscaler_model=image_upscaler_model_val,
                    image_upscaler_batch_size=image_upscaler_batch_size_val,

                    # Face restoration parameters for batch processing
                    enable_face_restoration=enable_face_restoration_val,
                    face_restoration_fidelity=face_restoration_fidelity_val,
                    enable_face_colorization=enable_face_colorization_val,
                    face_restoration_timing=face_restoration_timing_val,
                    face_restoration_when=face_restoration_when_val,
                    codeformer_model=codeformer_model_val,
                    face_restoration_batch_size=face_restoration_batch_size_val,

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
                            logger.warning(f"Failed to extract caption from status for {file_name}: {e}")

                # 6. Save caption to input folder if requested and available
                if batch_save_captions_val and generated_caption and prompt_source == "auto_caption":
                    prompt_file_path = get_prompt_file_path(input_file)
                    if save_caption_to_file(generated_caption, prompt_file_path, overwrite=False, logger=logger):
                        caption_stats["saved"] += 1
                    else:
                        caption_stats["skipped"] += 1

                # 7. Record results
                if final_output and os.path.exists(final_output):
                    processed_files.append({
                        "input": input_file, 
                        "output": final_output, 
                        "status": "success",
                        "prompt_source": prompt_source,
                        "caption_saved": batch_save_captions_val and generated_caption and prompt_source == "auto_caption",
                        "auto_resolution_used": auto_resolution_used,
                        "effective_resolution": f"{effective_target_w}x{effective_target_h}" if auto_resolution_used else None
                    })
                else:
                    failed_files.append((input_file, "Output file not created"))
                    logger.warning(f"Failed to get final output path for {input_file} or file does not exist.")


            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}", exc_info=True)
                failed_files.append((input_file, str(e)))

        progress(1.0, desc=f"Batch complete: {len(processed_files)} processed, {len(skipped_files)} skipped, {len(failed_files)} failed")

        # Enhanced status reporting
        file_type_display = "images" if file_type == "image" else "videos"
        status_msg = f"🎉 Batch processing complete!\n\n"
        status_msg += f"📊 **Summary:**\n"
        status_msg += f"  ✅ Successfully processed: {len(processed_files)} {file_type_display}\n"
        status_msg += f"  ⏩ Skipped (existing): {len(skipped_files)} {file_type_display}\n"
        status_msg += f"  ❌ Failed: {len(failed_files)} {file_type_display}\n"
        status_msg += f"  📁 Output folder: {batch_output_folder_val}\n"
        
        # Add upscaler mode information
        if enable_image_upscaler_val:
            from .image_upscaler_utils import extract_model_filename_from_dropdown
            actual_model_name = extract_model_filename_from_dropdown(image_upscaler_model_val) if image_upscaler_model_val else "Unknown"
            status_msg += f"  🖼️ Upscaler: Image-based ({actual_model_name}, batch size: {image_upscaler_batch_size_val})\n"
        else:
            status_msg += f"  ⭐ Upscaler: STAR Model\n"
        
        # Add face restoration information
        if enable_face_restoration_val:
            colorization_text = " + Colorization" if enable_face_colorization_val else ""
            timing_text = f"{face_restoration_when_val} upscaling"
            status_msg += f"  👤 Face Restoration: Enabled (fidelity: {face_restoration_fidelity_val:.1f}, {timing_text}{colorization_text})\n"
        else:
            status_msg += f"  👤 Face Restoration: Disabled\n"
        
        # Add auto-resolution information
        if enable_auto_aspect_resolution_check_val:
            auto_resolution_count = sum(1 for file_info in processed_files if file_info.get('auto_resolution_used', False))
            status_msg += f"  🎯 Auto-Resolution: Enabled ({auto_resolution_count}/{len(processed_files)} videos adjusted)\n"
            if auto_resolution_count > 0:
                status_msg += f"      📐 Pixel budget: {target_w_num_val}x{target_h_num_val} ({target_w_num_val * target_h_num_val:,} pixels)\n"
        else:
            status_msg += f"  🎯 Auto-Resolution: Disabled\n"
        status_msg += "\n"

        if batch_save_captions_val and any(caption_stats.values()):
            status_msg += f"📝 **Caption Management:**\n"
            status_msg += f"  💾 Captions saved: {caption_stats['saved']}\n"
            status_msg += f"  ⏩ Captions skipped (file exists): {caption_stats['skipped']}\n"
            if caption_stats['failed'] > 0:
                status_msg += f"  ❌ Caption save failed: {caption_stats['failed']}\n"
            status_msg += "\n"
        elif enable_image_upscaler_val and batch_enable_auto_caption_val:
            status_msg += f"📝 **Caption Management:**\n"
            status_msg += f"  ℹ️ Auto-captioning was disabled (image upscaler doesn't use prompts)\n\n"

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


def process_batch_videos_from_app_config(app_config, run_upscale_func, logger, progress=gr.Progress(track_tqdm=True)):
    """
    Adapter function that extracts parameters from AppConfig and calls the main batch processing function.
    
    Args:
        app_config: AppConfig object containing all configuration settings
        run_upscale_func: The upscaling function to use for processing
        logger: Logger instance
        progress: Gradio progress tracker
        
    Returns:
        tuple: (None, status_message) from batch processing
    """
    return process_batch_videos(
        batch_input_folder_val=app_config.batch.input_folder,
        batch_output_folder_val=app_config.batch.output_folder,
        user_prompt_val=app_config.prompts.user,
        pos_prompt_val=app_config.prompts.positive,
        neg_prompt_val=app_config.prompts.negative,
        model_selector_val=app_config.star_model.model_choice,
        upscale_factor_slider_val=app_config.resolution.upscale_factor,
        cfg_slider_val=app_config.star_model.cfg_scale,
        steps_slider_val=app_config.star_model.steps,
        solver_mode_radio_val=app_config.star_model.solver_mode,
        max_chunk_len_slider_val=app_config.performance.max_chunk_len,
        enable_chunk_optimization_check_val=app_config.performance.enable_chunk_optimization,
        vae_chunk_slider_val=app_config.performance.vae_chunk,
        enable_vram_optimization_check_val=app_config.performance.enable_vram_optimization,  # FIX: Added missing parameter
        color_fix_dropdown_val=app_config.star_model.color_fix_method,
        enable_tiling_check_val=app_config.tiling.enable,
        tile_size_num_val=app_config.tiling.tile_size,
        tile_overlap_num_val=app_config.tiling.tile_overlap,
        enable_context_window_check_val=app_config.context_window.enable,
        context_overlap_num_val=app_config.context_window.overlap,
        enable_target_res_check_val=app_config.resolution.enable_target_res,
        target_h_num_val=app_config.resolution.target_h,
        target_w_num_val=app_config.resolution.target_w,
        target_res_mode_radio_val=app_config.resolution.target_res_mode,
        enable_auto_aspect_resolution_check_val=app_config.resolution.enable_auto_aspect_resolution,
        auto_resolution_status_display_val=app_config.resolution.auto_resolution_status,
        ffmpeg_preset_dropdown_val=app_config.ffmpeg.preset,
        ffmpeg_quality_slider_val=app_config.ffmpeg.quality,
        ffmpeg_use_gpu_check_val=app_config.ffmpeg.use_gpu,
        save_frames_checkbox_val=app_config.outputs.save_frames,
        save_metadata_checkbox_val=app_config.outputs.save_metadata,
        save_chunks_checkbox_val=app_config.outputs.save_chunks,
        save_chunk_frames_checkbox_val=app_config.outputs.save_chunk_frames,
        create_comparison_video_check_val=app_config.outputs.create_comparison_video,
        enable_scene_split_check_val=app_config.scene_split.enable,
        scene_split_mode_radio_val=app_config.scene_split.mode,
        scene_min_scene_len_num_val=app_config.scene_split.min_scene_len,
        scene_drop_short_check_val=app_config.scene_split.drop_short,
        scene_merge_last_check_val=app_config.scene_split.merge_last,
        scene_frame_skip_num_val=app_config.scene_split.frame_skip,
        scene_threshold_num_val=app_config.scene_split.threshold,
        scene_min_content_val_num_val=app_config.scene_split.min_content_val,
        scene_frame_window_num_val=app_config.scene_split.frame_window,
        scene_copy_streams_check_val=app_config.scene_split.copy_streams,
        scene_use_mkvmerge_check_val=app_config.scene_split.use_mkvmerge,
        scene_rate_factor_num_val=app_config.scene_split.rate_factor,
        scene_preset_dropdown_val=app_config.scene_split.encoding_preset,
        scene_quiet_ffmpeg_check_val=app_config.scene_split.quiet_ffmpeg,
        scene_manual_split_type_radio_val=app_config.scene_split.manual_split_type,
        scene_manual_split_value_num_val=app_config.scene_split.manual_split_value,
        enable_fps_decrease_val=app_config.fps_decrease.enable,
        fps_decrease_mode_val=app_config.fps_decrease.mode,  # FIX: Added missing parameter
        fps_multiplier_preset_val=app_config.fps_decrease.multiplier_preset,  # FIX: Added missing parameter
        fps_multiplier_custom_val=app_config.fps_decrease.multiplier_custom,  # FIX: Added missing parameter
        target_fps_val=app_config.fps_decrease.target_fps,
        fps_interpolation_method_val=app_config.fps_decrease.interpolation_method,
        enable_rife_interpolation_val=app_config.rife.enable,
        rife_multiplier_val=app_config.rife.multiplier,
        rife_fp16_val=app_config.rife.fp16,
        rife_uhd_val=app_config.rife.uhd,
        rife_scale_val=app_config.rife.scale,
        rife_skip_static_val=app_config.rife.skip_static,
        rife_enable_fps_limit_val=app_config.rife.enable_fps_limit,
        rife_max_fps_limit_val=app_config.rife.max_fps_limit,
        rife_apply_to_chunks_val=app_config.rife.apply_to_chunks,
        rife_apply_to_scenes_val=app_config.rife.apply_to_scenes,
        rife_keep_original_val=app_config.rife.keep_original,
        rife_overwrite_original_val=app_config.rife.overwrite_original,
        run_upscale_func=run_upscale_func,
        logger=logger,
        batch_skip_existing_val=app_config.batch.skip_existing,
        batch_use_prompt_files_val=app_config.batch.use_prompt_files,
        batch_save_captions_val=app_config.batch.save_captions,
        batch_enable_auto_caption_val=app_config.batch.enable_auto_caption,
        batch_cogvlm_quant_val=app_config.cogvlm.quant_value,
        batch_cogvlm_unload_val=app_config.cogvlm.unload_after_use,
        current_seed=app_config.seed.seed,
        enable_image_upscaler_val=app_config.image_upscaler.enable,
        image_upscaler_model_val=app_config.image_upscaler.model,
        image_upscaler_batch_size_val=app_config.image_upscaler.batch_size,
        enable_face_restoration_val=app_config.face_restoration.enable,
        face_restoration_fidelity_val=app_config.face_restoration.fidelity_weight,
        enable_face_colorization_val=app_config.face_restoration.enable_colorization,
        face_restoration_timing_val="after_upscale",  # Default value
        face_restoration_when_val=app_config.face_restoration.when,
        codeformer_model_val=app_config.face_restoration.model,
        face_restoration_batch_size_val=app_config.face_restoration.batch_size,
        enable_direct_image_upscaling_val=app_config.batch.enable_direct_image_upscaling,  # NEW: Direct image upscaling parameter
        progress=progress
    )