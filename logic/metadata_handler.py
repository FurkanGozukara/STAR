import os
import time
import logging
from .common_utils import format_time

def _prepare_metadata_dict(params_dict: dict, status_info: dict = None) -> dict:
    """
    Creates the structured dictionary for metadata from a flat params_dict and status_info.
    Internal helper function.
    """
    metadata = {
        "input_video_path": os.path.abspath(params_dict.get("input_video_path", "")) if params_dict.get("input_video_path") else "N/A",
        "user_prompt": params_dict.get("user_prompt", "N/A"),
        "positive_prompt": params_dict.get("positive_prompt", "N/A"),
        "negative_prompt": params_dict.get("negative_prompt", "N/A"),
        "model_choice": params_dict.get("model_choice", "N/A"),
        "upscale_factor_slider_if_target_res_disabled": params_dict.get("upscale_factor_slider", "N/A"),
        "cfg_scale": params_dict.get("cfg_scale", "N/A"),
        "steps": params_dict.get("ui_total_diffusion_steps", "N/A"),
        "solver_mode": params_dict.get("solver_mode", "N/A"),
        "max_chunk_len": params_dict.get("max_chunk_len", "N/A"),
        "vae_chunk": params_dict.get("vae_chunk", "N/A"),
        "color_fix_method": params_dict.get("color_fix_method", "N/A"),
        "enable_tiling": params_dict.get("enable_tiling", False),
        "tile_size_if_tiling_enabled": params_dict.get("tile_size", "N/A") if params_dict.get("enable_tiling") else "N/A",
        "tile_overlap_if_tiling_enabled": params_dict.get("tile_overlap", "N/A") if params_dict.get("enable_tiling") else "N/A",
        "enable_sliding_window": params_dict.get("enable_sliding_window", False),
        "window_size_if_sliding_enabled": params_dict.get("window_size", "N/A") if params_dict.get("enable_sliding_window") else "N/A",
        "window_step_if_sliding_enabled": params_dict.get("window_step", "N/A") if params_dict.get("enable_sliding_window") else "N/A",
        "enable_target_res": params_dict.get("enable_target_res", False),
        "target_h_if_target_res_enabled": params_dict.get("target_h", "N/A") if params_dict.get("enable_target_res") else "N/A",
        "target_w_if_target_res_enabled": params_dict.get("target_w", "N/A") if params_dict.get("enable_target_res") else "N/A",
        "target_res_mode_if_target_res_enabled": params_dict.get("target_res_mode", "N/A") if params_dict.get("enable_target_res") else "N/A",
        "ffmpeg_preset": params_dict.get("ffmpeg_preset", "N/A"),
        "ffmpeg_quality_value": params_dict.get("ffmpeg_quality_value", "N/A"),
        "ffmpeg_use_gpu": params_dict.get("ffmpeg_use_gpu", False),
        "enable_scene_split": params_dict.get("enable_scene_split", False),
        "scene_split_mode_if_enabled": params_dict.get("scene_split_mode", "N/A") if params_dict.get("enable_scene_split") else "N/A",
        "scene_min_scene_len_if_enabled": params_dict.get("scene_min_scene_len", "N/A") if params_dict.get("enable_scene_split") else "N/A",
        "scene_threshold_if_enabled": params_dict.get("scene_threshold", "N/A") if params_dict.get("enable_scene_split") else "N/A",
        "scene_manual_split_type_if_enabled": params_dict.get("scene_manual_split_type", "N/A") if params_dict.get("enable_scene_split") else "N/A",
        "scene_manual_split_value_if_enabled": params_dict.get("scene_manual_split_value", "N/A") if params_dict.get("enable_scene_split") else "N/A",
        "is_batch_mode": params_dict.get("is_batch_mode", False),
        "batch_output_dir_if_batch": params_dict.get("batch_output_dir", "N/A") if params_dict.get("is_batch_mode") else "N/A",
        "final_output_video_path": os.path.abspath(params_dict.get("final_output_path", "")) if params_dict.get("final_output_path") else "N/A",
        "original_video_resolution_wh": (params_dict.get("orig_w"), params_dict.get("orig_h")) if params_dict.get("orig_w") is not None and params_dict.get("orig_h") is not None else "N/A",
        "effective_input_fps": f"{params_dict.get('input_fps'):.2f}" if params_dict.get("input_fps") is not None else "N/A",
        "calculated_upscale_factor": f"{params_dict.get('upscale_factor'):.2f}" if params_dict.get("upscale_factor") is not None else "N/A",
        "final_output_resolution_wh": (params_dict.get("final_w"), params_dict.get("final_h")) if params_dict.get("final_w") is not None and params_dict.get("final_h") is not None else "N/A",
        "scene_prompt_used_for_chunk": params_dict.get("scene_prompt_used_for_chunk", "N/A"),
    }

    if status_info:
        processing_time_total = status_info.get("processing_time_total")
        overall_process_start_time = status_info.get("overall_process_start_time")
        current_chunk = status_info.get("current_chunk")
        total_chunks = status_info.get("total_chunks")
        scene_specific_data = status_info.get("scene_specific_data")

        current_processing_time = None
        if overall_process_start_time is not None and processing_time_total is None:
            current_processing_time = time.time() - overall_process_start_time
        
        effective_processing_time = processing_time_total if processing_time_total is not None else current_processing_time

        if effective_processing_time is not None:
            metadata["processing_time_seconds"] = f"{effective_processing_time:.2f}"
            metadata["processing_time_formatted"] = format_time(effective_processing_time)
        else:
            metadata["processing_time_seconds"] = "N/A"
            metadata["processing_time_formatted"] = "N/A"

        if current_chunk is not None and total_chunks is not None:
            metadata["current_chunk_progress"] = f"{current_chunk}/{total_chunks}"
            metadata["processing_status"] = f"In progress - chunk {current_chunk} of {total_chunks} completed"
        elif processing_time_total is not None:
            metadata["processing_status"] = "Completed"
        else:
            metadata["processing_status"] = "In progress" # Default if no other status applies

        if scene_specific_data:
            metadata.update({
                "scene_index": scene_specific_data.get("scene_index", "N/A"),
                "scene_name": scene_specific_data.get("scene_name", "N/A"),
                "scene_prompt_used": scene_specific_data.get("scene_prompt", "N/A"),
                "scene_frame_count": scene_specific_data.get("scene_frame_count", "N/A"),
                "scene_fps": f"{scene_specific_data.get('scene_fps', 0):.2f}" if scene_specific_data.get("scene_fps") else "N/A",
                "scene_processing_time_seconds": f"{scene_specific_data.get('scene_processing_time', 0):.2f}" if scene_specific_data.get("scene_processing_time") else "N/A",
                "scene_processing_time_formatted": format_time(scene_specific_data.get("scene_processing_time", 0)) if scene_specific_data.get("scene_processing_time") else "N/A",
                "scene_video_path": os.path.abspath(scene_specific_data.get("scene_video_path", "")) if scene_specific_data.get("scene_video_path") else "N/A"
            })
    else: # No status_info, assume basic fields might be missing timing/progress
        metadata["processing_time_seconds"] = "N/A"
        metadata["processing_time_formatted"] = "N/A"
        if "processing_status" not in metadata: # Only set if not already set by other logic
             metadata["processing_status"] = "Unknown"
        
    return metadata

def _save_metadata_to_file_internal(metadata_params: dict, filepath: str, logger: logging.Logger = None) -> tuple[bool, str]:
    """Writes the metadata dictionary to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in metadata_params.items():
                f.write(f"{key}: {value}\n")
        if logger:
            logger.info(f"Metadata saved to: {filepath}")
        return True, f"Metadata saved to: {filepath}"
    except Exception as e:
        error_msg = f"Error saving metadata to {filepath}: {e}"
        if logger:
            logger.error(error_msg)
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