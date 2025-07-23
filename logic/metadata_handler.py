import os
import time
import logging
from .common_utils import format_time

def _prepare_metadata_dict(params_dict: dict, status_info: dict = None) -> dict:
    """
    Creates the structured dictionary for metadata from a flat params_dict and status_info.
    Only includes fields relevant to the specific upscaler type being used.
    """
    upscaler_type = params_dict.get("upscaler_type", "STAR")
    
    # Start with common metadata fields used by all upscalers
    metadata = {
        "upscaler_type": upscaler_type,
        "input_video_path": os.path.abspath(params_dict.get("input_video_path", "")) if params_dict.get("input_video_path") else "N/A",
        "final_output_video_path": os.path.abspath(params_dict.get("final_output_path", "")) if params_dict.get("final_output_path") else "N/A",
        "current_seed": params_dict.get("current_seed", "N/A"),
    }
    
    # Add resolution and upscale information (common to all)
    if params_dict.get("orig_w") is not None and params_dict.get("orig_h") is not None:
        metadata["original_video_resolution_wh"] = (params_dict.get("orig_w"), params_dict.get("orig_h"))
    if params_dict.get("final_w") is not None and params_dict.get("final_h") is not None:
        metadata["final_output_resolution_wh"] = (params_dict.get("final_w"), params_dict.get("final_h"))
    if params_dict.get("upscale_factor") is not None:
        metadata["calculated_upscale_factor"] = f"{params_dict.get('upscale_factor'):.2f}"
    if params_dict.get("input_fps") is not None:
        metadata["effective_input_fps"] = f"{params_dict.get('input_fps'):.2f}"
    
    # Target resolution settings (common to all)
    if params_dict.get("enable_target_res"):
        metadata["enable_target_res"] = True
        metadata["target_h"] = params_dict.get("target_h", "N/A")
        metadata["target_w"] = params_dict.get("target_w", "N/A")
        metadata["target_res_mode"] = params_dict.get("target_res_mode", "N/A")
    elif not params_dict.get("enable_target_res") and params_dict.get("upscale_factor_slider") is not None:
        metadata["upscale_factor_slider"] = params_dict.get("upscale_factor_slider")
    
    # FFmpeg settings (common to all)
    metadata["ffmpeg_preset"] = params_dict.get("ffmpeg_preset", "N/A")
    metadata["ffmpeg_quality_value"] = params_dict.get("ffmpeg_quality_value", "N/A")
    if params_dict.get("ffmpeg_use_gpu"):
        metadata["ffmpeg_use_gpu"] = True
    
    # Batch mode (common to all)
    if params_dict.get("is_batch_mode"):
        metadata["is_batch_mode"] = True
        metadata["batch_output_dir"] = params_dict.get("batch_output_dir", "N/A")
        if params_dict.get("original_filename"):
            metadata["original_filename"] = params_dict.get("original_filename")
    
    # Output settings (common to all)
    if params_dict.get("save_frames"):
        metadata["save_frames"] = True
    if params_dict.get("save_chunks"):
        metadata["save_chunks"] = True
    if params_dict.get("save_chunk_frames"):
        metadata["save_chunk_frames"] = True
    if params_dict.get("create_comparison_video_enabled"):
        metadata["create_comparison_video_enabled"] = True
    
    # Add upscaler-specific fields based on type
    if upscaler_type.lower() == "star":
        # STAR-specific fields only
        metadata.update({
            "user_prompt": params_dict.get("user_prompt", "N/A"),
            "positive_prompt": params_dict.get("positive_prompt", "N/A"),
            "negative_prompt": params_dict.get("negative_prompt", "N/A"),
            "model_choice": params_dict.get("model_choice", "N/A"),
            "cfg_scale": params_dict.get("cfg_scale", "N/A"),
            "steps": params_dict.get("ui_total_diffusion_steps", "N/A"),
            "solver_mode": params_dict.get("solver_mode", "N/A"),
            "max_chunk_len": params_dict.get("max_chunk_len", "N/A"),
            "vae_chunk": params_dict.get("vae_chunk", "N/A"),
            "color_fix_method": params_dict.get("color_fix_method", "N/A"),
        })
        
        # Additional STAR optimization settings
        if params_dict.get("enable_chunk_optimization"):
            metadata["enable_chunk_optimization"] = True
        if params_dict.get("enable_vram_optimization"):
            metadata["enable_vram_optimization"] = True
        if params_dict.get("enable_fp16_processing"):
            metadata["enable_fp16_processing"] = True
        
        # STAR tiling settings
        if params_dict.get("enable_tiling"):
            metadata["enable_tiling"] = True
            metadata["tile_size"] = params_dict.get("tile_size", "N/A")
            metadata["tile_overlap"] = params_dict.get("tile_overlap", "N/A")
        
        # STAR context window settings
        if params_dict.get("enable_context_window"):
            metadata["enable_context_window"] = True
            metadata["context_overlap"] = params_dict.get("context_overlap", "N/A")
        
        # Scene split settings (STAR uses this for prompts)
        if params_dict.get("enable_scene_split"):
            metadata["enable_scene_split"] = True
            metadata["scene_split_mode"] = params_dict.get("scene_split_mode", "N/A")
            metadata["scene_min_scene_len"] = params_dict.get("scene_min_scene_len", "N/A")
            metadata["scene_threshold"] = params_dict.get("scene_threshold", "N/A")
            if params_dict.get("scene_split_mode") == "Manual":
                metadata["scene_manual_split_type"] = params_dict.get("scene_manual_split_type", "N/A")
                metadata["scene_manual_split_value"] = params_dict.get("scene_manual_split_value", "N/A")
            # Additional scene split parameters for STAR
            metadata["scene_drop_short"] = params_dict.get("scene_drop_short", False)
            metadata["scene_merge_last"] = params_dict.get("scene_merge_last", False)
            metadata["scene_frame_skip"] = params_dict.get("scene_frame_skip", "N/A")
            metadata["scene_min_content_val"] = params_dict.get("scene_min_content_val", "N/A")
        
        # Scene-specific prompt if used
        if params_dict.get("scene_prompt_used_for_chunk") and params_dict.get("scene_prompt_used_for_chunk") != "N/A":
            metadata["scene_prompt_used_for_chunk"] = params_dict.get("scene_prompt_used_for_chunk")
        
        # Face restoration settings for STAR
        if params_dict.get("face_restoration_enabled"):
            metadata["face_restoration_enabled"] = True
            metadata["face_restoration_fidelity"] = params_dict.get("face_restoration_fidelity", "N/A")
            metadata["face_colorization_enabled"] = params_dict.get("face_colorization_enabled", False)
            metadata["face_restoration_timing"] = params_dict.get("face_restoration_timing", "N/A")
            metadata["codeformer_model"] = params_dict.get("codeformer_model", "N/A")
            metadata["face_restoration_batch_size"] = params_dict.get("face_restoration_batch_size", "N/A")
        
        # Auto-caption settings for STAR
        if params_dict.get("enable_auto_caption_per_scene"):
            metadata["enable_auto_caption_per_scene"] = True
            metadata["cogvlm_quant"] = params_dict.get("cogvlm_quant", "N/A")
            metadata["cogvlm_unload"] = params_dict.get("cogvlm_unload", "N/A")
            
    elif upscaler_type.lower() == "seedvr2":
        # SeedVR2-specific fields only
        seedvr2_fields = {
            "seedvr2_model": params_dict.get("seedvr2_model"),
            "seedvr2_batch_size": params_dict.get("seedvr2_batch_size"),
            "seedvr2_device": params_dict.get("seedvr2_device"),
            "seedvr2_frame_overlap": params_dict.get("seedvr2_frame_overlap"),
            "seedvr2_preserve_vram": params_dict.get("seedvr2_preserve_vram"),
            "seedvr2_color_correction": params_dict.get("seedvr2_color_correction"),
            "seedvr2_enable_frame_padding": params_dict.get("seedvr2_enable_frame_padding"),
            "seedvr2_flash_attention": params_dict.get("seedvr2_flash_attention"),
            "seedvr2_enable_multi_gpu": params_dict.get("seedvr2_enable_multi_gpu"),
            "seedvr2_gpu_devices": params_dict.get("seedvr2_gpu_devices"),
            "seedvr2_cfg_scale": params_dict.get("seedvr2_cfg_scale"),
            "seedvr2_frames_processed": params_dict.get("seedvr2_frames_processed"),
            "seedvr2_frames_failed": params_dict.get("seedvr2_frames_failed"),
            "seedvr2_avg_fps": params_dict.get("seedvr2_avg_fps"),
            # Additional SeedVR2 fields that were missing
            "seedvr2_quality_preset": params_dict.get("seedvr2_quality_preset"),
            "seedvr2_temporal_quality": params_dict.get("seedvr2_temporal_quality"),
            "seedvr2_enable_temporal_consistency": params_dict.get("seedvr2_enable_temporal_consistency"),
            "seedvr2_scene_awareness": params_dict.get("seedvr2_scene_awareness"),
            "seedvr2_consistency_validation": params_dict.get("seedvr2_consistency_validation"),
            "seedvr2_chunk_optimization": params_dict.get("seedvr2_chunk_optimization"),
            "seedvr2_pad_last_chunk": params_dict.get("seedvr2_pad_last_chunk"),
            "seedvr2_skip_first_frames": params_dict.get("seedvr2_skip_first_frames"),
            "seedvr2_enable_chunk_preview": params_dict.get("seedvr2_enable_chunk_preview"),
            "seedvr2_chunk_preview_frames": params_dict.get("seedvr2_chunk_preview_frames"),
            "seedvr2_keep_last_chunks": params_dict.get("seedvr2_keep_last_chunks"),
            "seedvr2_model_precision": params_dict.get("seedvr2_model_precision"),
            "seedvr2_seed": params_dict.get("seedvr2_seed"),
            "seedvr2_upscale_factor": params_dict.get("seedvr2_upscale_factor"),
            "seedvr2_total_frames": params_dict.get("seedvr2_total_frames"),
        }
        
        # Add processing time if available
        if params_dict.get("seedvr2_processing_time") is not None:
            seedvr2_fields["seedvr2_processing_time"] = f"{params_dict.get('seedvr2_processing_time'):.2f}"
        
        # Block swap settings if enabled
        if params_dict.get("seedvr2_enable_block_swap"):
            seedvr2_fields.update({
                "seedvr2_enable_block_swap": True,
                "seedvr2_block_swap_counter": params_dict.get("seedvr2_block_swap_counter"),
                "seedvr2_block_swap_offload_io": params_dict.get("seedvr2_block_swap_offload_io"),
                "seedvr2_block_swap_model_caching": params_dict.get("seedvr2_block_swap_model_caching"),
            })
        
        # Only add non-None values
        for key, value in seedvr2_fields.items():
            if value is not None and value != "N/A":
                metadata[key] = value
        
        # Add temporal consistency nested metadata if available
        if params_dict.get("temporal_consistency"):
            tc_data = params_dict.get("temporal_consistency")
            if isinstance(tc_data, dict):
                tc_metadata = {}
                if tc_data.get("enabled") is not None:
                    tc_metadata["temporal_consistency_enabled"] = tc_data.get("enabled")
                if tc_data.get("chunk_count"):
                    tc_metadata["temporal_consistency_chunk_count"] = tc_data.get("chunk_count")
                if tc_data.get("scene_awareness") is not None:
                    tc_metadata["temporal_consistency_scene_awareness"] = tc_data.get("scene_awareness")
                if tc_data.get("batch_size_corrected") is not None:
                    tc_metadata["temporal_consistency_batch_corrected"] = tc_data.get("batch_size_corrected")
                # Add to metadata
                metadata.update(tc_metadata)
                
    elif upscaler_type.lower() == "image_upscaler":
        # Image upscaler-specific fields only
        metadata["image_upscaler_enabled"] = True
        image_upscaler_fields = {
            "image_upscaler_model": params_dict.get("image_upscaler_model"),
            "image_upscaler_model_display": params_dict.get("image_upscaler_model_display"),
            "image_upscaler_architecture": params_dict.get("image_upscaler_architecture"),
            "image_upscaler_scale_factor": params_dict.get("image_upscaler_scale_factor"),
            "image_upscaler_batch_size": params_dict.get("image_upscaler_batch_size"),
            "image_upscaler_device": params_dict.get("image_upscaler_device"),
            "image_upscaler_frames_processed": params_dict.get("image_upscaler_frames_processed"),
            "image_upscaler_frames_failed": params_dict.get("image_upscaler_frames_failed"),
            # Additional image upscaler fields
            "image_upscaler_cache_models": params_dict.get("image_upscaler_cache_models"),
            "actual_model_scale": params_dict.get("actual_model_scale"),
            "image_upscaler_supports_half": params_dict.get("supports_half"),
            "image_upscaler_supports_bfloat16": params_dict.get("supports_bfloat16"),
            "image_upscaler_input_channels": params_dict.get("input_channels"),
            "image_upscaler_output_channels": params_dict.get("output_channels"),
        }
        
        # Add processing time if available
        if params_dict.get("image_upscaler_processing_time") is not None:
            image_upscaler_fields["image_upscaler_processing_time"] = f"{params_dict.get('image_upscaler_processing_time'):.2f}"
        
        # Only add non-None values
        for key, value in image_upscaler_fields.items():
            if value is not None and value != "N/A":
                metadata[key] = value
    
    # RIFE interpolation metadata (applies to any upscaler output)
    if params_dict.get("rife_enabled"):
        metadata["rife_enabled"] = True
        rife_fields = {
            "rife_multiplier": params_dict.get("rife_multiplier"),
            "rife_actual_multiplier": params_dict.get("rife_actual_multiplier"),
            "rife_fp16": params_dict.get("rife_fp16"),
            "rife_uhd": params_dict.get("rife_uhd"),
            "rife_scale": params_dict.get("rife_scale"),
            "rife_skip_static": params_dict.get("rife_skip_static"),
            "rife_apply_to_chunks": params_dict.get("rife_apply_to_chunks"),
            "rife_apply_to_scenes": params_dict.get("rife_apply_to_scenes"),
            "rife_keep_original": params_dict.get("rife_keep_original"),
            "rife_overwrite_original": params_dict.get("rife_overwrite_original"),
        }
        
        # FPS values
        if params_dict.get("rife_original_fps") is not None:
            rife_fields["rife_original_fps"] = f"{params_dict.get('rife_original_fps'):.2f}"
        if params_dict.get("rife_target_fps") is not None:
            rife_fields["rife_target_fps"] = f"{params_dict.get('rife_target_fps'):.2f}"
        if params_dict.get("rife_final_fps") is not None:
            rife_fields["rife_final_fps"] = f"{params_dict.get('rife_final_fps'):.2f}"
        
        # FPS limit if enabled
        if params_dict.get("rife_fps_limit_enabled"):
            rife_fields["rife_fps_limit_enabled"] = True
            rife_fields["rife_fps_limit"] = params_dict.get("rife_fps_limit")
        
        # Processing time and output path
        if params_dict.get("rife_processing_time") is not None:
            rife_fields["rife_processing_time"] = f"{params_dict.get('rife_processing_time'):.2f}"
        if params_dict.get("rife_output_path"):
            rife_fields["rife_output_path"] = os.path.abspath(params_dict.get("rife_output_path"))
        
        # Only add non-None values
        for key, value in rife_fields.items():
            if value is not None and value != "N/A":
                metadata[key] = value
    
    # FPS decrease metadata (applies to any upscaler)
    if params_dict.get("fps_decrease_applied"):
        metadata["fps_decrease_applied"] = True
        fps_fields = {
            "original_fps_before_decrease": params_dict.get("original_fps"),
            "fps_decrease_mode": params_dict.get("fps_decrease_actual_mode"),
            "fps_decrease_multiplier": params_dict.get("fps_decrease_actual_multiplier"),
            "fps_decrease_target": params_dict.get("fps_decrease_actual_target"),
            "fps_decrease_calculated_fps": params_dict.get("fps_decrease_calculated_fps"),
            "fps_decrease_saved_path": params_dict.get("fps_decrease_saved_path"),
        }
        # Only add non-None values
        for key, value in fps_fields.items():
            if value is not None and value != "N/A" and value != "Failed to save":
                metadata[key] = value
    
    # Add frame range information and processing status
    if status_info:
        processing_time_total = status_info.get("processing_time_total")
        overall_process_start_time = status_info.get("overall_process_start_time")
        current_chunk = status_info.get("current_chunk")
        total_chunks = status_info.get("total_chunks")
        scene_specific_data = status_info.get("scene_specific_data")
        
        # Calculate processing time
        current_processing_time = None
        if overall_process_start_time is not None and processing_time_total is None:
            current_processing_time = time.time() - overall_process_start_time
        
        effective_processing_time = processing_time_total if processing_time_total is not None else current_processing_time
        
        if effective_processing_time is not None:
            metadata["processing_time_seconds"] = f"{effective_processing_time:.2f}"
            metadata["processing_time_formatted"] = format_time(effective_processing_time)
        
        # Processing status
        if current_chunk is not None and total_chunks is not None:
            metadata["current_chunk_progress"] = f"{current_chunk}/{total_chunks}"
            metadata["processing_status"] = f"In progress - chunk {current_chunk} of {total_chunks} completed"
        elif processing_time_total is not None:
            metadata["processing_status"] = "Completed"
        else:
            metadata["processing_status"] = "In progress"
        
        # Video frame range
        video_frame_range = status_info.get("video_frame_range")
        if video_frame_range:
            metadata["overall_video_frame_range"] = f"frames {video_frame_range[0]}-{video_frame_range[1]}"
            metadata["overall_video_frame_count"] = video_frame_range[1] - video_frame_range[0] + 1
        
        # Chunk frame range
        chunk_frame_range = status_info.get("chunk_frame_range")
        if chunk_frame_range:
            metadata["current_chunk_frame_range"] = f"frames {chunk_frame_range[0]}-{chunk_frame_range[1]}"
            metadata["current_chunk_frame_count"] = chunk_frame_range[1] - chunk_frame_range[0] + 1
        
        # Scene-specific data (mainly for STAR)
        if scene_specific_data and upscaler_type.lower() == "star":
            scene_fields = {
                "scene_index": scene_specific_data.get("scene_index"),
                "scene_name": scene_specific_data.get("scene_name"),
                "scene_prompt_used": scene_specific_data.get("scene_prompt"),
                "scene_frame_count": scene_specific_data.get("scene_frame_count"),
            }
            
            if scene_specific_data.get("scene_fps"):
                scene_fields["scene_fps"] = f"{scene_specific_data.get('scene_fps'):.2f}"
            if scene_specific_data.get("scene_processing_time"):
                scene_fields["scene_processing_time_seconds"] = f"{scene_specific_data.get('scene_processing_time'):.2f}"
                scene_fields["scene_processing_time_formatted"] = format_time(scene_specific_data.get("scene_processing_time"))
            if scene_specific_data.get("scene_video_path"):
                scene_fields["scene_video_path"] = os.path.abspath(scene_specific_data.get("scene_video_path"))
            
            # Scene frame range
            scene_frame_range_data = scene_specific_data.get("scene_frame_range")
            if scene_frame_range_data:
                scene_fields["scene_frame_range"] = f"frames {scene_frame_range_data[0]}-{scene_frame_range_data[1]}"
                scene_fields["scene_total_frames"] = scene_frame_range_data[1] - scene_frame_range_data[0] + 1
            
            # Only add non-None values
            for key, value in scene_fields.items():
                if value is not None and value != "N/A":
                    metadata[key] = value
    
    return metadata

def _save_metadata_to_file_internal(metadata_params: dict, filepath: str, logger: logging.Logger = None) -> tuple[bool, str]:
    """Writes the metadata dictionary to a file, excluding N/A values."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in metadata_params.items():
                # Skip N/A values to reduce clutter
                if value != "N/A":
                    f.write(f"{key}: {value}\n")
        if logger:
            logger.info(f"Metadata saved to: {filepath}")
        return True, f"Metadata saved to: {filepath}"
    except Exception as e:
        error_msg = f"Error saving metadata to {filepath}: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg


def save_image_upscaler_metadata(
    output_video_path: str,
    input_video_path: str,
    model_filename: str,
    model_display_name: str,
    model_architecture: str,
    model_scale_factor: str,
    batch_size: int,
    device: str,
    frames_processed: int,
    frames_failed: int,
    processing_time: float,
    seed: int = 99,
    logger: logging.Logger = None
) -> tuple[bool, str]:
    """
    Save metadata specifically for image upscaler processing.
    
    Args:
        output_video_path: Path to the output video
        input_video_path: Path to the input video
        model_filename: Actual model filename
        model_display_name: Display name from dropdown
        model_architecture: Model architecture (e.g., DAT-2, ESRGAN)
        model_scale_factor: Upscaling factor (e.g., "4x")
        batch_size: Batch size used for processing
        device: Device used (cuda/cpu)
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        processing_time: Total processing time in seconds
        seed: Random seed used
        logger: Logger instance
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Create metadata dictionary
        metadata_dict = {
            "upscaler_type": "image_upscaler",
            "image_upscaler_enabled": True,
            "image_upscaler_model": model_filename,
            "image_upscaler_model_display": model_display_name,
            "image_upscaler_architecture": model_architecture,
            "image_upscaler_scale_factor": model_scale_factor,
            "image_upscaler_batch_size": batch_size,
            "image_upscaler_device": device,
            "image_upscaler_frames_processed": frames_processed,
            "image_upscaler_frames_failed": frames_failed,
            "image_upscaler_processing_time": processing_time,
            "input_video_path": os.path.abspath(input_video_path),
            "final_output_path": os.path.abspath(output_video_path),
            "current_seed": seed,
            "processing_time_seconds": f"{processing_time:.2f}",
            "processing_time_formatted": format_time(processing_time),
            "processing_status": "Completed"
        }
        
        # Generate metadata file path
        output_dir = os.path.dirname(output_video_path)
        base_filename = os.path.splitext(os.path.basename(output_video_path))[0]
        metadata_filepath = os.path.join(output_dir, f"{base_filename}_image_upscaler_metadata.txt")
        
        # Save metadata
        success, message = _save_metadata_to_file_internal(metadata_dict, metadata_filepath, logger)
        
        if success:
            return True, f"Image upscaler metadata saved to: {metadata_filepath}"
        else:
            return False, f"Failed to save image upscaler metadata: {message}"
            
    except Exception as e:
        error_msg = f"Error creating image upscaler metadata: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def save_rife_metadata(
    output_video_path: str,
    input_video_path: str,
    rife_params: dict,
    processing_time: float,
    seed: int = 99,
    logger: logging.Logger = None
) -> tuple[bool, str]:
    """
    Helper function to save RIFE-specific metadata.
    
    Args:
        output_video_path: Path to the output RIFE video
        input_video_path: Path to the input video
        rife_params: Dictionary containing RIFE processing parameters
        processing_time: Processing time in seconds
        seed: Seed value used for processing
        logger: Logger instance
        
    Returns:
        A tuple (success_boolean, message_string).
    """
    try:
        # Prepare metadata file path
        output_dir = os.path.dirname(output_video_path)
        base_filename = os.path.splitext(os.path.basename(output_video_path))[0]
        
        # Create params_dict in the expected format
        params_dict = {
            "input_video_path": input_video_path,
            "final_output_path": output_video_path,
            "current_seed": seed,
            "rife_enabled": True,
            "rife_multiplier": rife_params.get("multiplier", 2),
            "rife_actual_multiplier": rife_params.get("actual_multiplier", rife_params.get("multiplier", 2)),
            "rife_fp16": rife_params.get("fp16", True),
            "rife_uhd": rife_params.get("uhd", False),
            "rife_scale": rife_params.get("scale", 1.0),
            "rife_skip_static": rife_params.get("skip_static", False),
            "rife_original_fps": rife_params.get("original_fps"),
            "rife_target_fps": rife_params.get("target_fps"),
            "rife_final_fps": rife_params.get("final_fps"),
            "rife_fps_limit_enabled": rife_params.get("fps_limit_enabled", False),
            "rife_fps_limit": rife_params.get("fps_limit"),
            "rife_apply_to_chunks": rife_params.get("apply_to_chunks", False),
            "rife_apply_to_scenes": rife_params.get("apply_to_scenes", False),
            "rife_keep_original": rife_params.get("keep_original", True),
            "rife_overwrite_original": rife_params.get("overwrite_original", False),
            "rife_processing_time": processing_time,
            "rife_output_path": output_video_path,
            "ffmpeg_preset": rife_params.get("ffmpeg_preset", "N/A"),
            "ffmpeg_quality_value": rife_params.get("ffmpeg_quality_value", "N/A"),
            "ffmpeg_use_gpu": rife_params.get("ffmpeg_use_gpu", False),
        }
        
        # Create status_info with processing time
        status_info = {
            "processing_time_total": processing_time
        }
        
        # Use the main save_metadata function
        return save_metadata(
            save_flag=True,
            output_dir=output_dir,
            base_filename_no_ext=base_filename,
            params_dict=params_dict,
            status_info=status_info,
            logger=logger
        )
        
    except Exception as e:
        error_msg = f"Error in save_rife_metadata: {e}"
        if logger:
            logger.error(error_msg, exc_info=True)
        return False, error_msg

def save_metadata(
    save_flag: bool,
    output_dir: str,
    base_filename_no_ext: str,
    params_dict: dict,
    status_info: dict = None,
    logger: logging.Logger = None
) -> tuple[bool, str]:
    """
    Main function to save or update metadata.

    Args:
        save_flag: Boolean, if False, metadata saving is skipped.
        output_dir: Directory to save the metadata file.
        base_filename_no_ext: Base name for the metadata file (e.g., "0001" or "scene_0001_metadata").
        params_dict: Dictionary containing all static parameters from UI and derived values
                     (e.g., input_video_path, model_choice, orig_w, final_h).
        status_info: Optional dictionary for dynamic processing status.
                     Can include: "current_chunk", "total_chunks",
                                  "overall_process_start_time", "processing_time_total",
                                  "scene_specific_data" (itself a dictionary).
        logger: Logger instance.

    Returns:
        A tuple (success_boolean, message_string).
    """
    if not save_flag:
        return True, "Metadata saving disabled"

    try:
        metadata_filepath = os.path.join(output_dir, f"{base_filename_no_ext}.txt")
        
        # Prepare the comprehensive metadata dictionary
        # Pass copies to avoid modifying original dicts if they are re-used by the caller
        current_params_dict = params_dict.copy() if params_dict else {}
        current_status_info = status_info.copy() if status_info else {}
        
        full_metadata_dict = _prepare_metadata_dict(current_params_dict, current_status_info)

        success, message = _save_metadata_to_file_internal(full_metadata_dict, metadata_filepath, logger)

        if success and status_info and status_info.get("current_chunk") is not None and status_info.get("total_chunks") is not None:
            # This specific message is for chunk updates
            chunk_msg = f"Metadata updated after chunk {status_info.get('current_chunk')}/{status_info.get('total_chunks')}: {metadata_filepath}"
            if logger:
                logger.info(chunk_msg)
            return True, chunk_msg
        
        return success, message

    except Exception as e:
        error_msg = f"Error in save_metadata handler: {e}"
        if logger:
            # Log with stack trace for better debugging of the handler itself
            logger.error(error_msg, exc_info=True)
        return False, error_msg