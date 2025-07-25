import gradio as gr
import os
import time
import tempfile
import shutil
import math
import numpy as np
import cv2
import gc
import logging

import torch
# from easydict import EasyDict # This will be passed as EasyDict_class

# Imports from the 'logic' package
from .cancellation_manager import cancellation_manager, CancelledError
from .cogvlm_utils import auto_caption as util_auto_caption
from .common_utils import format_time
from .ffmpeg_utils import (
    run_ffmpeg_command as util_run_ffmpeg_command,
    extract_frames as util_extract_frames,
    create_video_from_frames as util_create_video_from_frames,
    decrease_fps as util_decrease_fps,
    get_common_fps_multipliers as util_get_common_fps_multipliers,
    validate_frame_extraction_consistency
)
from .file_utils import (
    get_batch_filename as util_get_batch_filename,
    get_next_filename as util_get_next_filename,
    cleanup_temp_dir as util_cleanup_temp_dir,
    get_video_resolution as util_get_video_resolution
)
from .scene_utils import (
    split_video_into_scenes as util_split_video_into_scenes,
    merge_scene_videos as util_merge_scene_videos
)
from .upscaling_utils import calculate_upscale_params as util_calculate_upscale_params
from .gpu_utils import get_gpu_device as util_get_gpu_device
from .nvenc_utils import should_fallback_to_cpu_encoding
from .scene_processing_core import process_single_scene
from .comparison_video import create_comparison_video, get_comparison_output_path
from .rife_interpolation import apply_rife_to_chunks, apply_rife_to_scenes
from .face_restoration_utils import (
    setup_codeformer_environment,
    restore_frames_batch_true,
    restore_video_frames,
    apply_face_restoration_to_frames,
    _check_video_has_audio
)


def run_upscale (
    input_video_path ,user_prompt ,positive_prompt ,negative_prompt ,model_choice ,
    upscale_factor_slider ,cfg_scale ,steps ,solver_mode ,
    max_chunk_len ,enable_chunk_optimization ,vae_chunk ,enable_vram_optimization ,color_fix_method ,
    enable_tiling ,tile_size ,tile_overlap ,
    enable_context_window ,context_overlap ,
    enable_target_res ,target_h ,target_w ,target_res_mode ,
    ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,
    save_frames ,save_metadata ,save_chunks ,save_chunk_frames ,

    enable_scene_split ,scene_split_mode ,scene_min_scene_len ,scene_drop_short ,scene_merge_last ,
    scene_frame_skip ,scene_threshold ,scene_min_content_val ,scene_frame_window ,
    scene_copy_streams ,scene_use_mkvmerge ,scene_rate_factor ,scene_preset ,scene_quiet_ffmpeg ,
    scene_manual_split_type ,scene_manual_split_value ,

    create_comparison_video_enabled ,

    # FPS decrease parameters
    enable_fps_decrease =False ,fps_decrease_mode ="multiplier" ,fps_multiplier_preset ="1/2x (Half FPS)" ,fps_multiplier_custom =0.5 ,target_fps =24.0 ,fps_interpolation_method ="drop",

    # RIFE interpolation parameters
    enable_rife_interpolation =False ,rife_multiplier =2 ,rife_fp16 =True ,rife_uhd =False ,rife_scale =1.0 ,
    rife_skip_static =False ,rife_enable_fps_limit =False ,rife_max_fps_limit =60 ,
    rife_apply_to_chunks =True ,rife_apply_to_scenes =True ,rife_keep_original =True ,rife_overwrite_original =False ,

    is_batch_mode =False ,batch_output_dir =None ,original_filename =None ,

    enable_auto_caption_per_scene =False ,
    cogvlm_quant =0 , # This will be the integer value
    cogvlm_unload ='full',

    # Image upscaler parameters
    enable_image_upscaler =False ,image_upscaler_model =None ,image_upscaler_batch_size =4 ,

    # Face restoration parameters
    enable_face_restoration =False ,face_restoration_fidelity =0.7 ,enable_face_colorization =False ,
    face_restoration_timing ="after_upscale" ,face_restoration_when ="after" ,codeformer_model =None ,
    face_restoration_batch_size =4 ,

    # SeedVR2 parameters
    enable_seedvr2 =False ,seedvr2_config =None ,

    # Injected dependencies
    logger: logging.Logger = None,
    app_config_module=None, # The app_config module from secourses_app
    metadata_handler_module=None, # The metadata_handler module
    VideoToVideo_sr_class=None,
    setup_seed_func=None,
    EasyDict_class=None,
    preprocess_func=None,
    collate_fn_func=None,
    tensor2vid_func=None,
    ImageSpliterTh_class=None,
    adain_color_fix_func=None,
    wavelet_color_fix_func=None,

    progress =gr .Progress (track_tqdm =True ),
    current_seed =99 # Added current_seed parameter with a default
):
    # Check for cancellation at the very start of the upscaling process
    cancellation_manager.check_cancel()
    
    if not input_video_path or not os .path .exists (input_video_path ):
        raise gr .Error ("Please select a valid input video file.")

    last_chunk_video_path =None
    last_chunk_status ="No chunks processed yet"

    logger.info(f"Using seed for this run: {current_seed}")
    setup_seed_func (current_seed) # Use the passed seed
    overall_process_start_time =time .time ()
    logger .info ("Overall upscaling process started.")

    # Check for cancellation after initial setup
    cancellation_manager.check_cancel()

    current_overall_progress =0.0

    params_for_metadata ={
    "input_video_path":input_video_path ,"user_prompt":user_prompt ,"positive_prompt":positive_prompt ,
    "negative_prompt":negative_prompt ,"model_choice":model_choice ,
    "upscale_factor_slider":upscale_factor_slider ,"cfg_scale":cfg_scale ,
    "ui_total_diffusion_steps":steps ,"solver_mode":solver_mode ,
    "max_chunk_len":max_chunk_len ,"enable_chunk_optimization":enable_chunk_optimization ,"vae_chunk":vae_chunk ,"color_fix_method":color_fix_method ,
    "enable_tiling":enable_tiling ,"tile_size":tile_size ,"tile_overlap":tile_overlap ,
    "enable_context_window":enable_context_window ,"context_overlap":context_overlap ,
    "enable_target_res":enable_target_res ,"target_h":target_h ,"target_w":target_w ,
    "target_res_mode":target_res_mode ,"ffmpeg_preset":ffmpeg_preset ,
    "ffmpeg_quality_value":ffmpeg_quality_value ,"ffmpeg_use_gpu":ffmpeg_use_gpu ,
    "enable_scene_split":enable_scene_split ,"scene_split_mode":scene_split_mode ,
    "scene_min_scene_len":scene_min_scene_len ,"scene_threshold":scene_threshold ,
    "scene_manual_split_type":scene_manual_split_type ,"scene_manual_split_value":scene_manual_split_value ,
    "is_batch_mode":is_batch_mode ,"batch_output_dir":batch_output_dir ,
    "current_seed": current_seed, # Added current_seed
    "upscaler_type": "seedvr2" if enable_seedvr2 else "STAR",
    
    # FPS decrease metadata
    "fps_decrease_enabled":enable_fps_decrease ,"fps_decrease_mode":fps_decrease_mode ,
    "fps_decrease_multiplier_preset":fps_multiplier_preset ,"fps_decrease_multiplier_custom":fps_multiplier_custom ,
    "fps_decrease_target":target_fps ,"fps_decrease_method":fps_interpolation_method ,
    
    # RIFE interpolation metadata
    "rife_enabled":enable_rife_interpolation ,"rife_multiplier":rife_multiplier ,"rife_fp16":rife_fp16 ,
    "rife_uhd":rife_uhd ,"rife_scale":rife_scale ,"rife_skip_static":rife_skip_static ,
    "rife_fps_limit_enabled":rife_enable_fps_limit ,"rife_fps_limit":rife_max_fps_limit ,
    "rife_apply_to_chunks":rife_apply_to_chunks ,"rife_apply_to_scenes":rife_apply_to_scenes ,
    "rife_keep_original":rife_keep_original ,"rife_overwrite_original":rife_overwrite_original ,

    # Image upscaler metadata
    "image_upscaler_enabled":enable_image_upscaler ,"image_upscaler_model":image_upscaler_model ,
    "image_upscaler_batch_size":image_upscaler_batch_size ,

    # Face restoration metadata
    "face_restoration_enabled":enable_face_restoration ,"face_restoration_fidelity":face_restoration_fidelity ,
    "face_colorization_enabled":enable_face_colorization ,"face_restoration_timing":face_restoration_timing ,
    "face_restoration_when":face_restoration_when ,"codeformer_model":codeformer_model ,
    "face_restoration_batch_size":face_restoration_batch_size ,

    "final_output_path":None ,"orig_w":None ,"orig_h":None ,
    "input_fps":None ,"upscale_factor":None ,"final_w":None ,"final_h":None ,
    }

    actual_cogvlm_quant_val = cogvlm_quant

    stage_weights ={
    "init_paths_res":0.03 ,
    "fps_decrease":0.05 if enable_fps_decrease else 0.0, # Added FPS decrease stage
    "scene_split":0.05 if enable_scene_split else 0.0 ,
    "downscale":0.07 , # This will be set to 0 if no downscale needed later
    "model_load":0.05 if not enable_image_upscaler else 0.0, # Only load STAR model if not using image upscaler
    "extract_frames":0.10 ,
    "copy_input_frames":0.05 if save_frames and not enable_scene_split else 0.0, # Adjusted
    "face_restoration_before":0.08 if enable_face_restoration and face_restoration_when == "before" else 0.0,
    "upscaling_loop":0.50 if enable_scene_split else 0.60 ,
    "scene_merge":0.05 if enable_scene_split else 0.0 ,
    "face_restoration_after":0.08 if enable_face_restoration and face_restoration_when == "after" else 0.0,
    "reassembly_copy_processed":0.05 if save_frames and not enable_scene_split else 0.0, # Adjusted
    "reassembly_audio_merge":0.03 ,
    "comparison_video": 0.03 if create_comparison_video_enabled else 0.0,
    "metadata":0.02
    }
    
    # Initial check for downscale weight based on enable_target_res
    if not enable_target_res : # If target res is disabled, no downscaling will happen
        stage_weights ["downscale"]=0.0
    # Further refinement of downscale weight happens after util_calculate_upscale_params

    total_weight =sum (w for w in stage_weights .values() if w > 0) # Sum only active stages
    if total_weight >0 :
        for key in stage_weights :
            stage_weights [key ]/=total_weight
    else :
        active_stages_count = len([w for w in stage_weights.values() if w > 0])
        if active_stages_count > 0:
            default_weight_per_stage = 1.0 / active_stages_count
            for key in stage_weights:
                stage_weights[key] = default_weight_per_stage if stage_weights[key] > 0 else 0.0
        elif len(stage_weights) > 0 : # if all weights are 0 for some reason (e.g. only one stage enabled and its weight was 0)
             for key in stage_weights : stage_weights [key ]=1 /len (stage_weights )
        else: # no stages
             pass

    # Check for cancellation after metadata setup
    cancellation_manager.check_cancel()

    if is_batch_mode and batch_output_dir and original_filename :
        base_output_filename_no_ext ,output_video_path ,batch_main_dir =util_get_batch_filename (batch_output_dir ,original_filename )
        main_output_dir =batch_main_dir
    else :
        base_output_filename_no_ext ,output_video_path =util_get_next_filename (app_config_module .DEFAULT_OUTPUT_DIR, logger=logger )
        main_output_dir =app_config_module .DEFAULT_OUTPUT_DIR

    params_for_metadata ["final_output_path"]=output_video_path

    run_id =f"star_run_{int(time.time())}_{np.random.randint(1000, 9999)}"
    temp_dir_base =tempfile .gettempdir ()
    temp_dir =os .path .join (temp_dir_base ,run_id )

    input_frames_dir =os .path .join (temp_dir ,"input_frames")
    output_frames_dir =os .path .join (temp_dir ,"output_frames")

    frames_output_subfolder =None
    input_frames_permanent_save_path =None
    processed_frames_permanent_save_path =None
    chunks_permanent_save_path =None

    if save_frames :
        if is_batch_mode :
            frames_output_subfolder =main_output_dir
        else :
            frames_output_subfolder =os .path .join (main_output_dir ,base_output_filename_no_ext )

        os .makedirs (frames_output_subfolder ,exist_ok =True )
        input_frames_permanent_save_path =os .path .join (frames_output_subfolder ,"input_frames")
        processed_frames_permanent_save_path =os .path .join (frames_output_subfolder ,"processed_frames")
        os .makedirs (input_frames_permanent_save_path ,exist_ok =True )
        os .makedirs (processed_frames_permanent_save_path ,exist_ok =True )

    # Directory setup will be handled by specific upscaler branches
    chunks_permanent_save_path = None
    chunk_frames_permanent_save_path = None

    os .makedirs (temp_dir ,exist_ok =True )
    os .makedirs (input_frames_dir ,exist_ok =True )
    os .makedirs (output_frames_dir ,exist_ok =True )

    # star_model =None # Model is no longer loaded here globally
    current_input_video_for_frames =input_video_path
    downscaled_temp_video =None
    status_log =["Process started..."]

    # Check for cancellation after directory setup
    cancellation_manager.check_cancel()

    def save_chunk_input_frames(chunk_idx, chunk_start_frame, chunk_end_frame, frame_files, input_frames_dir, chunk_frames_save_path, logger, chunk_type="", total_chunks=0):
        """Save input frames for a specific chunk for debugging purposes"""
        if not chunk_frames_save_path:
            return
        
        try:
            chunk_display_num = chunk_idx + 1
            chunk_frames_for_this_chunk = frame_files[chunk_start_frame:chunk_end_frame]
            
            if not chunk_frames_for_this_chunk:
                logger.warning(f"No frames to save for {chunk_type}chunk {chunk_display_num}")
                return
            
            # Create organized subfolder structure: chunk_frames/chunk1/ (no scene for direct processing)
            chunk_dir = os.path.join(chunk_frames_save_path, f"chunk{chunk_display_num}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            saved_count = 0
            for local_idx, frame_file in enumerate(chunk_frames_for_this_chunk):
                global_frame_idx = chunk_start_frame + local_idx
                src_path = os.path.join(input_frames_dir, frame_file)
                
                if os.path.exists(src_path):
                    # Use original frame number (1-based) in filename: frame1.png, frame2.png, etc.
                    dst_filename = f"frame{global_frame_idx + 1}.png"
                    dst_path = os.path.join(chunk_dir, dst_filename)
                    shutil.copy2(src_path, dst_path)
                    saved_count += 1
                else:
                    logger.warning(f"Input frame not found for {chunk_type}chunk {chunk_display_num}: {src_path}")
            
            if saved_count > 0:
                chunk_total_info = f"/{total_chunks}" if total_chunks > 0 else ""
                logger.info(f"Saved {saved_count} input frames for {chunk_type}chunk {chunk_display_num}{chunk_total_info} to chunk_frames/chunk{chunk_display_num}/")
                
        except Exception as e:
            logger.error(f"Error saving chunk input frames for {chunk_type}chunk {chunk_idx + 1}: {e}")

    ui_total_diffusion_steps =steps
    direct_upscale_msg =""

    try :
        progress (current_overall_progress ,desc ="Initializing...")
        status_log .append ("Initializing upscaling process...")
        logger .info ("Initializing upscaling process...")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status,None

        final_prompt =(user_prompt .strip ()+". "+positive_prompt .strip ()).strip ()

        model_file_path =app_config_module .LIGHT_DEG_MODEL_PATH if model_choice ==app_config_module .DEFAULT_MODEL_CHOICE else app_config_module .HEAVY_DEG_MODEL_PATH
        if not os .path .exists (model_file_path ):
            raise gr .Error (f"STAR model weight not found: {model_file_path}")

        orig_h_val ,orig_w_val =util_get_video_resolution (input_video_path, logger=logger )
        params_for_metadata ["orig_h"]=orig_h_val
        params_for_metadata ["orig_w"]=orig_w_val
        status_log .append (f"Original resolution: {orig_w_val}x{orig_h_val}")
        logger .info (f"Original resolution: {orig_w_val}x{orig_h_val}")

        current_overall_progress +=stage_weights ["init_paths_res"]
        progress (current_overall_progress ,desc ="Calculating target resolution...")

        # FPS Decrease Processing (if enabled) - MOVED TO BE FIRST
        fps_decreased_video_path = None
        current_input_video_for_frames = input_video_path  # Initialize with original input
        if enable_fps_decrease:
            fps_decrease_start_time = time.time()
            progress(current_overall_progress, desc="Applying FPS decrease...")
            
            # Convert UI parameters to backend format
            actual_target_fps = target_fps  # Default for fixed mode
            actual_fps_mode = fps_decrease_mode
            actual_multiplier = fps_multiplier_custom
            
            if fps_decrease_mode == "multiplier":
                # Convert preset to multiplier value if not using custom
                if fps_multiplier_preset != "Custom":
                    multiplier_map = {v: k for k, v in util_get_common_fps_multipliers().items()}
                    actual_multiplier = multiplier_map.get(fps_multiplier_preset, 0.5)
                else:
                    actual_multiplier = fps_multiplier_custom
                
                fps_decrease_status_msg = f"Decreasing FPS using {fps_multiplier_preset} (×{actual_multiplier:.3f}) with {fps_interpolation_method} method..."
            else:
                fps_decrease_status_msg = f"Decreasing FPS to {target_fps} using {fps_interpolation_method} method..."
            
            status_log.append(fps_decrease_status_msg)
            logger.info(fps_decrease_status_msg)
            yield None, "\n".join(status_log), last_chunk_video_path, fps_decrease_status_msg, None
            
            try:
                # Generate FPS decreased video path
                fps_decreased_video_path = os.path.join(temp_dir, "fps_decreased_input.mp4")
                
                # Apply FPS decrease to ORIGINAL input video (before any other processing)
                fps_success, fps_output_fps, fps_message = util_decrease_fps(
                    input_video_path=input_video_path,  # Use original input, not processed
                    output_video_path=fps_decreased_video_path,
                    target_fps=actual_target_fps,
                    interpolation_method=fps_interpolation_method,
                    ffmpeg_preset=ffmpeg_preset,
                    ffmpeg_quality_value=ffmpeg_quality_value,
                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                    logger=logger,
                    fps_mode=actual_fps_mode,
                    fps_multiplier=actual_multiplier
                )
                
                if fps_success:
                    # Update the input video path for all subsequent processing
                    current_input_video_for_frames = fps_decreased_video_path
                    # Update metadata
                    params_for_metadata["fps_decrease_applied"] = True
                    params_for_metadata["original_fps"] = params_for_metadata.get("input_fps", "unknown")
                    params_for_metadata["input_fps"] = fps_output_fps
                    params_for_metadata["fps_decrease_actual_mode"] = actual_fps_mode
                    params_for_metadata["fps_decrease_actual_multiplier"] = actual_multiplier
                    params_for_metadata["fps_decrease_actual_target"] = actual_target_fps
                    params_for_metadata["fps_decrease_calculated_fps"] = fps_output_fps
                    
                    # Save FPS decreased video to session-specific output folder for user reference
                    try:
                        if is_batch_mode:
                            session_output_dir = main_output_dir
                        else:
                            session_output_dir = os.path.join(main_output_dir, base_output_filename_no_ext)
                            os.makedirs(session_output_dir, exist_ok=True)
                        
                        fps_decreased_save_path = os.path.join(session_output_dir, "FPS_Reduced_Used_Video.mp4")
                        shutil.copy2(fps_decreased_video_path, fps_decreased_save_path)
                        logger.info(f"FPS decreased video saved to: {fps_decreased_save_path}")
                        params_for_metadata["fps_decrease_saved_path"] = fps_decreased_save_path
                    except Exception as e_save:
                        logger.warning(f"Failed to save FPS decreased video to session folder: {e_save}")
                        params_for_metadata["fps_decrease_saved_path"] = "Failed to save"
                    
                    fps_duration = time.time() - fps_decrease_start_time
                    fps_success_msg = f"FPS decrease completed in {format_time(fps_duration)}. {fps_message}"
                    status_log.append(fps_success_msg)
                    logger.info(fps_success_msg)
                    current_overall_progress += stage_weights["fps_decrease"]
                    progress(current_overall_progress, desc=f"FPS decreased to {fps_output_fps:.2f}")
                    yield None, "\n".join(status_log), last_chunk_video_path, f"FPS decreased to {fps_output_fps:.2f}", None
                else:
                    # FPS decrease failed, continue with original video
                    fps_duration = time.time() - fps_decrease_start_time
                    fps_error_msg = f"FPS decrease failed after {format_time(fps_duration)}: {fps_message}. Continuing with original video."
                    status_log.append(fps_error_msg)
                    logger.warning(fps_error_msg)
                    params_for_metadata["fps_decrease_applied"] = False
                    params_for_metadata["fps_decrease_error"] = fps_message
                    current_overall_progress += stage_weights["fps_decrease"]
                    yield None, "\n".join(status_log), last_chunk_video_path, "FPS decrease failed, using original", None
                    
            except Exception as e_fps:
                fps_duration = time.time() - fps_decrease_start_time
                fps_exception_msg = f"Exception during FPS decrease after {format_time(fps_duration)}: {str(e_fps)}. Continuing with original video."
                status_log.append(fps_exception_msg)
                logger.error(fps_exception_msg, exc_info=True)
                params_for_metadata["fps_decrease_applied"] = False
                params_for_metadata["fps_decrease_error"] = str(e_fps)
                current_overall_progress += stage_weights["fps_decrease"]
                yield None, "\n".join(status_log), last_chunk_video_path, "FPS decrease error, using original", None
        else:
            # FPS decrease disabled
            params_for_metadata["fps_decrease_applied"] = False
            current_overall_progress += stage_weights["fps_decrease"]  # Add zero weight if disabled

        upscale_factor_val =None
        final_h_val ,final_w_val =None ,None
        needs_downscale = False # Initialize

        if enable_target_res :
            # Pass SeedVR2 upscale factor if using SeedVR2
            custom_upscale_factor = None
            if enable_seedvr2 and seedvr2_config:
                from .star_dataclasses import DEFAULT_SEEDVR2_UPSCALE_FACTOR
                custom_upscale_factor = DEFAULT_SEEDVR2_UPSCALE_FACTOR
                
                # Add SeedVR2 configuration to metadata
                params_for_metadata.update({
                    # Basic settings
                    "seedvr2_model": seedvr2_config.model,
                    "seedvr2_batch_size": seedvr2_config.batch_size,
                    "seedvr2_quality_preset": seedvr2_config.quality_preset,
                    "seedvr2_use_gpu": seedvr2_config.use_gpu,
                    
                    # Advanced settings
                    "seedvr2_preserve_vram": seedvr2_config.preserve_vram,
                    "seedvr2_flash_attention": seedvr2_config.flash_attention,
                    "seedvr2_color_correction": seedvr2_config.color_correction,
                    "seedvr2_enable_multi_gpu": seedvr2_config.enable_multi_gpu,
                    "seedvr2_gpu_devices": seedvr2_config.gpu_devices,
                    
                    # Block swap configuration
                    "seedvr2_enable_block_swap": seedvr2_config.enable_block_swap,
                    "seedvr2_block_swap_counter": seedvr2_config.block_swap_counter,
                    "seedvr2_block_swap_offload_io": seedvr2_config.block_swap_offload_io,
                    "seedvr2_block_swap_model_caching": seedvr2_config.block_swap_model_caching,
                    
                    # Temporal consistency settings
                    "seedvr2_temporal_overlap": seedvr2_config.temporal_overlap,
                    "seedvr2_scene_awareness": seedvr2_config.scene_awareness,
                    "seedvr2_temporal_quality": seedvr2_config.temporal_quality,
                    "seedvr2_consistency_validation": seedvr2_config.consistency_validation,
                    "seedvr2_chunk_optimization": seedvr2_config.chunk_optimization,
                    "seedvr2_enable_temporal_consistency": seedvr2_config.enable_temporal_consistency,
                    
                    # Frame processing
                    "seedvr2_enable_frame_padding": seedvr2_config.enable_frame_padding,
                    "seedvr2_pad_last_chunk": seedvr2_config.pad_last_chunk,
                    "seedvr2_skip_first_frames": seedvr2_config.skip_first_frames,
                    
                    # Chunk preview configuration
                    "seedvr2_enable_chunk_preview": seedvr2_config.enable_chunk_preview,
                    "seedvr2_chunk_preview_frames": seedvr2_config.chunk_preview_frames,
                    "seedvr2_keep_last_chunks": seedvr2_config.keep_last_chunks,
                    
                    # Model configuration
                    "seedvr2_model_precision": seedvr2_config.model_precision,
                    "seedvr2_cfg_scale": seedvr2_config.cfg_scale,
                    "seedvr2_seed": seedvr2_config.seed,
                    "seedvr2_upscale_factor": custom_upscale_factor,
                })
            
            needs_downscale ,ds_h ,ds_w ,upscale_factor_calc ,final_h_calc ,final_w_calc =util_calculate_upscale_params (
            orig_h_val ,orig_w_val ,target_h ,target_w ,target_res_mode ,logger =logger ,image_upscaler_model =image_upscaler_model if enable_image_upscaler else None,
            custom_upscale_factor =custom_upscale_factor
            )
            upscale_factor_val =upscale_factor_calc
            final_h_val ,final_w_val =final_h_calc ,final_w_calc

            params_for_metadata ["upscale_factor"]=upscale_factor_val
            params_for_metadata ["final_h"]=final_h_val
            params_for_metadata ["final_w"]=final_w_val

            status_log .append (f"Target resolution mode: {target_res_mode}. Calculated upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}")
            logger .info (f"Target resolution mode: {target_res_mode}. Calculated upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Calculating resolution...",None

            if not needs_downscale and stage_weights["downscale"] > 0.0: # If no downscaling needed, but weight was assigned
                stage_weights["downscale"] = 0.0 # Zero it out
                # Re-normalize weights if downscale stage is skipped
                total_weight =sum (w for w in stage_weights .values() if w > 0)
                if total_weight >0 :
                    for key_renorm in stage_weights :
                        stage_weights [key_renorm ]/=total_weight
                # else: # if all weights are 0 for some reason, handle as before

            if needs_downscale :
                downscale_stage_start_time =time .time ()
                downscale_progress_start =current_overall_progress
                progress (current_overall_progress ,desc ="Downscaling input video...")
                downscale_status_msg =f"Downscaling input to {ds_w}x{ds_h} before upscaling."
                status_log .append (downscale_status_msg )
                logger .info (downscale_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,"Input Downscaling...",None

                downscaled_temp_video =os .path .join (temp_dir ,"downscaled_input.mp4")
                scale_filter =f"scale='trunc(iw*min({ds_w}/iw,{ds_h}/ih)/2)*2':'trunc(ih*min({ds_w}/iw,{ds_h}/ih)/2)*2'"

                # Get encoding configuration with automatic NVENC fallback
                from .nvenc_utils import get_nvenc_fallback_encoding_config, build_ffmpeg_video_encoding_args
                
                encoding_config = get_nvenc_fallback_encoding_config(
                    use_gpu=ffmpeg_use_gpu,
                    ffmpeg_preset=ffmpeg_preset,
                    ffmpeg_quality=ffmpeg_quality_value,
                    width=ds_w,
                    height=ds_h,
                    logger=logger
                )
                
                ffmpeg_opts_downscale = build_ffmpeg_video_encoding_args(encoding_config)
                
                if logger:
                    codec_info = f"Using {encoding_config['codec']} for downscaling with preset {encoding_config['preset']} and {encoding_config['quality_param'].upper()} {encoding_config['quality_value']}."
                    logger.info(codec_info)

                cmd =f'ffmpeg -y -i "{current_input_video_for_frames}" -vf "{scale_filter}" {ffmpeg_opts_downscale} -c:a copy "{downscaled_temp_video}"'
                util_run_ffmpeg_command (cmd ,"Input Downscaling with Audio Copy",logger =logger )
                current_input_video_for_frames =downscaled_temp_video

                orig_h_val ,orig_w_val =util_get_video_resolution (downscaled_temp_video, logger=logger )
                params_for_metadata ["orig_h"]=orig_h_val
                params_for_metadata ["orig_w"]=orig_w_val

                # Copy downscaled video to output folder for user access
                try:
                    # Create session output directory if it doesn't exist
                    if is_batch_mode:
                        session_output_dir = main_output_dir
                    else:
                        session_output_dir = os.path.join(main_output_dir, base_output_filename_no_ext)
                        os.makedirs(session_output_dir, exist_ok=True)
                    
                    downscaled_save_path = os.path.join(session_output_dir, "downscaled_input.mp4")
                    shutil.copy2(downscaled_temp_video, downscaled_save_path)
                    logger.info(f"Downscaled video saved to: {downscaled_save_path}")
                    params_for_metadata["downscaled_video_saved_path"] = downscaled_save_path
                except Exception as e_save:
                    logger.warning(f"Failed to save downscaled video to session folder: {e_save}")
                    params_for_metadata["downscaled_video_saved_path"] = "Failed to save"

                downscale_duration_msg =f"Input downscaling finished. Time: {format_time(time.time() - downscale_stage_start_time)}"
                status_log .append (downscale_duration_msg )
                logger .info (downscale_duration_msg )
                current_overall_progress =downscale_progress_start +stage_weights ["downscale"]
                progress (current_overall_progress ,desc ="Downscaling complete.")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,"Downscaling complete.",None
            else : # No downscaling needed, but target res was enabled
                 current_overall_progress +=stage_weights ["downscale"] # Add the (potentially zero) weight

        else : # Target res disabled
            if stage_weights ["downscale"]>0 : # This means it was enabled but now skipped
                current_overall_progress +=stage_weights ["downscale"] # Add its original weight allocation
            
            # Determine the actual upscale factor - use model scale for image upscalers
            if enable_image_upscaler and image_upscaler_model:
                # Load image upscaler model early to get its actual scale factor
                from .image_upscaler_utils import (
                    get_model_info, extract_model_filename_from_dropdown
                )
                
                actual_model_filename = extract_model_filename_from_dropdown(image_upscaler_model)
                if actual_model_filename:
                    model_path = os.path.join(app_config_module.UPSCALE_MODELS_DIR, actual_model_filename)
                    model_info = get_model_info(model_path, logger)
                    if "error" not in model_info:
                        upscale_factor_val = model_info.get("scale", upscale_factor_slider)
                        if logger:
                            logger.info(f"Using image upscaler scale factor: {upscale_factor_val}x from model {actual_model_filename}")
                    else:
                        upscale_factor_val = upscale_factor_slider
                        if logger:
                            logger.warning(f"Could not get scale from image upscaler model, using slider value: {upscale_factor_val}x")
                else:
                    upscale_factor_val = upscale_factor_slider
            else:
                upscale_factor_val = upscale_factor_slider
                
            final_h_val =int (round (orig_h_val *upscale_factor_val /2 )*2 )
            final_w_val =int (round (orig_w_val *upscale_factor_val /2 )*2 )

            params_for_metadata ["upscale_factor"]=upscale_factor_val
            params_for_metadata ["final_h"]=final_h_val
            params_for_metadata ["final_w"]=final_w_val

            direct_upscale_msg =f"Direct upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}"
            status_log .append (direct_upscale_msg )
            logger .info (direct_upscale_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status,None



        scene_video_paths =[]
        if enable_scene_split :
            scene_split_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Splitting video into scenes...")
            status_log .append ("Splitting video into scenes...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Scene Splitting...",None

            scene_split_params ={
            'split_mode':scene_split_mode ,
            'min_scene_len':scene_min_scene_len ,'drop_short_scenes':scene_drop_short ,'merge_last_scene':scene_merge_last ,
            'frame_skip':scene_frame_skip ,'threshold':scene_threshold ,'min_content_val':scene_min_content_val ,'frame_window':scene_frame_window ,
            'weights':[1.0 ,1.0 ,1.0 ,0.0 ],
            'copy_streams':scene_copy_streams ,'use_mkvmerge':scene_use_mkvmerge ,
            'rate_factor':scene_rate_factor ,'preset':scene_preset ,'quiet_ffmpeg':scene_quiet_ffmpeg ,
            'show_progress':True ,
            'manual_split_type':scene_manual_split_type ,'manual_split_value':scene_manual_split_value ,
            'use_gpu':ffmpeg_use_gpu
            }

            def scene_progress_callback (progress_val ,desc ):
                current_scene_progress =scene_split_progress_start +(progress_val *stage_weights ["scene_split"])
                progress (current_scene_progress ,desc =desc )

            try :
                scene_video_paths =util_split_video_into_scenes (
                current_input_video_for_frames ,
                temp_dir , # Scenes will be in temp_dir/scenes
                scene_split_params ,
                scene_progress_callback ,
                logger =logger
                )

                scene_split_msg =f"Video split into {len(scene_video_paths)} scenes"
                status_log .append (scene_split_msg )
                logger .info (scene_split_msg )

                current_overall_progress =scene_split_progress_start +stage_weights ["scene_split"]
                progress (current_overall_progress ,desc =f"Scene splitting complete: {len(scene_video_paths)} scenes")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_split_msg,None

            except Exception as e :
                logger .error (f"Scene splitting failed: {e}", exc_info=True)
                status_log .append (f"Scene splitting failed: {e}")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene splitting failed: {e}",None
                raise gr .Error (f"Scene splitting failed: {e}")
        else : # Scene split disabled, add its weight if it was > 0
            if stage_weights["scene_split"] > 0.0:
                current_overall_progress +=stage_weights ["scene_split"]

        # STAR Model loading is removed from here and handled per-scene or per-run.
        # Ensure model_load stage progress is still accounted for if it was active.
        if stage_weights["model_load"] > 0.0:
            model_load_progress_start = current_overall_progress
            status_log.append("STAR model loading will occur as needed (per scene or for direct upscale).")
            logger.info("STAR model loading deferred to per-scene or direct upscale processing.")
            current_overall_progress = model_load_progress_start + stage_weights["model_load"]
            progress(current_overall_progress, desc="STAR model loading deferred.")
            yield None, "\n".join(status_log), last_chunk_video_path, "STAR model loading deferred.", None

        input_fps_val =30.0 # Default, will be updated
        if not enable_scene_split :
            frame_extract_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Extracting frames...")
            frame_extraction_start_time =time .time ()
            frame_count ,input_fps_val ,frame_files =util_extract_frames (current_input_video_for_frames ,input_frames_dir ,logger =logger )
            params_for_metadata ["input_fps"]=input_fps_val

            # Add frame validation to detect potential duration mismatches early
            validation_result = validate_frame_extraction_consistency(current_input_video_for_frames, frame_count, logger)
            if not validation_result['is_consistent']:
                warning_msg = f"⚠️ Frame count inconsistency detected: expected {validation_result['expected_frames']}, extracted {frame_count}"
                status_log.append(warning_msg)
                logger.warning(warning_msg)
                for warning in validation_result['warnings']:
                    logger.warning(f"Frame validation: {warning}")

            frame_extract_msg =f"Extracted {frame_count} frames at {input_fps_val:.2f} FPS. Time: {format_time(time.time() - frame_extraction_start_time)}"
            status_log .append (frame_extract_msg )
            logger .info (frame_extract_msg )
            current_overall_progress =frame_extract_progress_start +stage_weights ["extract_frames"]
            progress (current_overall_progress ,desc =f"Extracted {frame_count} frames.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Extracted {frame_count} frames.",None
        else : # Scene split enabled, skip main frame extraction stage here
             if stage_weights["extract_frames"] > 0.0:
                current_overall_progress +=stage_weights ["extract_frames"]

        if save_frames and not enable_scene_split and input_frames_dir and input_frames_permanent_save_path :
            copy_input_frames_progress_start =current_overall_progress
            copy_input_frames_start_time =time .time ()
            num_frames_to_copy_input =frame_count if 'frame_count'in locals ()and frame_count is not None else len (os .listdir (input_frames_dir ))

            copy_input_msg =f"Copying {num_frames_to_copy_input} input frames to permanent storage: {input_frames_permanent_save_path}"
            status_log .append (copy_input_msg )
            logger .info (copy_input_msg )
            progress (current_overall_progress ,desc ="Copying input frames...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Copying input frames.",None

            frames_copied_count =0
            for frame_file_idx ,frame_file_name in enumerate (os .listdir (input_frames_dir )):
                shutil .copy2 (os .path .join (input_frames_dir ,frame_file_name ),os .path .join (input_frames_permanent_save_path ,frame_file_name ))
                frames_copied_count +=1
                if frames_copied_count %50 ==0 or frames_copied_count ==num_frames_to_copy_input :
                    loop_progress_frac =frames_copied_count /num_frames_to_copy_input if num_frames_to_copy_input >0 else 1.0
                    current_overall_progress_temp =copy_input_frames_progress_start +(loop_progress_frac *stage_weights ["copy_input_frames"])
                    progress (current_overall_progress_temp ,desc =f"Copying input frames: {frames_copied_count}/{num_frames_to_copy_input}")

            copied_input_msg =f"Input frames copied. Time: {format_time(time.time() - copy_input_frames_start_time)}"
            status_log .append (copied_input_msg )
            logger .info (copied_input_msg )

            current_overall_progress =copy_input_frames_progress_start +stage_weights ["copy_input_frames"]
            progress (current_overall_progress ,desc ="Input frames copied.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Input frames copied.",None
        else : # Not saving frames or scene split enabled
             if stage_weights["copy_input_frames"] > 0.0: # Ensure progress is added if stage was active
                current_overall_progress +=stage_weights ["copy_input_frames"]

        # FACE RESTORATION - Before Upscaling
        if enable_face_restoration and face_restoration_when == "before" and not enable_scene_split:
            face_restoration_before_progress_start = current_overall_progress
            progress(current_overall_progress, desc="Starting face restoration before upscaling...")
            
            face_restoration_before_start_time = time.time()
            logger.info("Applying face restoration before upscaling...")
            status_log.append("Applying face restoration before upscaling...")
            yield None, "\n".join(status_log), last_chunk_video_path, "Face restoration before upscaling...", None
            
            # Create face restoration output directory  
            face_restored_frames_dir = os.path.join(temp_dir, "face_restored_frames_before")
            
            # Progress callback for face restoration
            def face_restoration_before_progress_callback(progress_val, desc):
                abs_progress = face_restoration_before_progress_start + progress_val * stage_weights["face_restoration_before"]
                progress(abs_progress, desc=desc)
            
            # Apply face restoration to input frames
            face_restoration_result = apply_face_restoration_to_frames(
                input_frames_dir=input_frames_dir,
                output_frames_dir=face_restored_frames_dir,
                fidelity_weight=face_restoration_fidelity,
                enable_colorization=enable_face_colorization,
                model_path=codeformer_model,
                batch_size=face_restoration_batch_size,
                progress_callback=face_restoration_before_progress_callback,
                logger=logger
            )
            
            if face_restoration_result['success']:
                # Update input frames directory to use face-restored frames for upscaling
                input_frames_dir = face_restored_frames_dir
                face_restore_before_msg = f"Face restoration before upscaling completed: {face_restoration_result['processed_count']} frames processed. Time: {format_time(time.time() - face_restoration_before_start_time)}"
                status_log.append(face_restore_before_msg)
                logger.info(face_restore_before_msg)
            else:
                error_msg = f"Face restoration before upscaling failed: {face_restoration_result['error']}"
                status_log.append(error_msg)
                logger.warning(error_msg)
                # Continue with original frames if face restoration fails
                
            current_overall_progress = face_restoration_before_progress_start + stage_weights["face_restoration_before"]
            progress(current_overall_progress, desc="Face restoration before upscaling completed")
            yield None, "\n".join(status_log), last_chunk_video_path, "Face restoration before upscaling completed", None
        else:
            # Skip face restoration before stage
            if stage_weights["face_restoration_before"] > 0.0:
                current_overall_progress += stage_weights["face_restoration_before"]

        upscaling_loop_progress_start =current_overall_progress
        progress (current_overall_progress ,desc ="Preparing for upscaling...")
        total_noise_levels =900
        upscaling_loop_start_time =time .time ()
        gpu_device =util_get_gpu_device (logger =logger )
        scene_metadata_base_params =params_for_metadata .copy ()if enable_scene_split else None
        silent_upscaled_video_path = None # Initialize
        
        # Initialize progress variable for non-scene-split processing (used by frame copying section later)
        if not enable_scene_split:
            upscaling_loop_progress_start_no_scene_split = current_overall_progress

        # MAIN BRANCHING LOGIC: Image Upscaler vs STAR
        if enable_image_upscaler:
            # Route to image upscaler processing
            logger.info("Image upscaler mode enabled - using deterministic image upscaling")
            status_log.append("Using image-based upscaling instead of STAR model")
            yield None, "\n".join(status_log), last_chunk_video_path, "Initializing image upscaler...", None

            # --- NEW: Progress mapping wrapper ---
            # The image upscaler reports progress in the 0.0-1.0 range relative to its own work.
            # Map that relative value to the absolute progress space of the overall pipeline so
            # that the Gradio progress bar reflects the ongoing frame processing.
            def _image_upscaler_progress_wrapper(rel_val: float, desc: str = ""):
                """Translate image-upscaler relative progress (0-1) to absolute pipeline progress."""
                try:
                    # Clamp relative value to the valid range just to be safe
                    rel_val_clamped = max(0.0, min(1.0, rel_val))
                    abs_progress = upscaling_loop_progress_start + rel_val_clamped * stage_weights["upscaling_loop"]
                    progress(abs_progress, desc=desc)
                except Exception as prog_exc:
                    # Fail gracefully if the progress object is unavailable (e.g., during unit tests)
                    if logger:
                        logger.debug(f"Image upscaler progress wrapper error (ignored): {prog_exc}")

            # Import image upscaler core here to avoid circular imports
            from .image_upscaler_core import process_video_with_image_upscaler

            try:
                # Process with image upscaler
                for result in process_video_with_image_upscaler(
                    input_video_path=current_input_video_for_frames,
                    selected_model_filename=image_upscaler_model,
                    batch_size=image_upscaler_batch_size,
                    upscale_models_dir=app_config_module.UPSCALE_MODELS_DIR,
                    
                    # Video processing parameters
                    enable_target_res=enable_target_res,
                    target_h=target_h,
                    target_w=target_w,
                    target_res_mode=target_res_mode,
                    
                    # Frame saving and output parameters
                    save_frames=save_frames,
                    save_metadata=save_metadata,
                    save_chunks=save_chunks,
                    save_chunk_frames=save_chunk_frames,
                    
                    # Scene processing parameters
                    enable_scene_split=enable_scene_split,
                    scene_video_paths=scene_video_paths,
                    
                    # Directory and file management
                    temp_dir=temp_dir,
                    output_dir=main_output_dir,
                    base_output_filename_no_ext=base_output_filename_no_ext,
                    
                    # FPS parameters
                    input_fps=input_fps_val,
                    
                    # FFmpeg parameters
                    ffmpeg_preset=ffmpeg_preset,
                    ffmpeg_quality_value=ffmpeg_quality_value,
                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                    
                    # Dependencies
                    logger=logger,
                    progress=_image_upscaler_progress_wrapper,
                    
                    # Utility functions (injected dependencies)
                    util_extract_frames=util_extract_frames,
                    util_create_video_from_frames=util_create_video_from_frames,
                    util_get_gpu_device=util_get_gpu_device,
                    format_time=format_time,
                    
                    # Metadata parameters
                    params_for_metadata=params_for_metadata,
                    metadata_handler_module=metadata_handler_module,
                    
                    # Status tracking
                    status_log=status_log,
                    current_seed=current_seed
                ):
                    # The image upscaler yields: (output_video_path, status_message, chunk_video_path, chunk_status, comparison_video_path)
                    output_video_path, status_message, chunk_video_path, chunk_status, comparison_video_path = result
                    
                    if output_video_path is not None:
                        # For scene splitting with image upscaler, we need to ensure proper naming
                        # to avoid conflicts during audio merging
                        if enable_scene_split:
                            # Use STAR naming pattern for intermediate file
                            silent_upscaled_video_path = os.path.join(temp_dir, "silent_upscaled_video.mp4")
                            # Copy/move the image upscaler output to the expected path
                            if output_video_path != silent_upscaled_video_path:
                                shutil.copy2(output_video_path, silent_upscaled_video_path)
                                logger.info(f"Image upscaler: Copied output to intermediate path for audio merge: {silent_upscaled_video_path}")
                            
                            # Important: For partial results (cancellation), preserve the partial path
                            # Only restore permanent target path if this is a complete result
                            current_final_path = params_for_metadata.get("final_output_path")
                            if current_final_path and not any(indicator in os.path.basename(output_video_path) for indicator in ["partial", "cancelled"]):
                                # Complete result - use the permanent target path
                                image_upscaler_temp_output = output_video_path  # Save the temp path returned by image upscaler
                                output_video_path = current_final_path  # Restore the permanent target path
                                logger.info(f"Image upscaler: Restoring permanent output path: {output_video_path} (was temp: {image_upscaler_temp_output})")
                            else:
                                # Partial result or no permanent path set - preserve the path returned by image upscaler
                                logger.info(f"Image upscaler: Using partial/cancelled result path: {output_video_path}")
                        else:
                            # Direct upscaling without scene splitting
                            silent_upscaled_video_path = output_video_path
                        
                        params_for_metadata["input_fps"] = input_fps_val
                        # Final result - break out of loop
                        break
                    
                    if status_message:
                        status_log.append(status_message)
                    
                    if chunk_video_path is not None:
                        last_chunk_video_path = chunk_video_path
                    
                    if chunk_status:
                        last_chunk_status = chunk_status
                    
                    # Yield progress update
                    yield None, "\n".join(status_log), last_chunk_video_path, last_chunk_status, comparison_video_path
                
                # Update progress after image upscaling
                current_overall_progress = upscaling_loop_progress_start + stage_weights["upscaling_loop"]
                progress(current_overall_progress, desc="Image upscaling complete")
                status_log.append("Image upscaling completed successfully")
                logger.info("Image upscaling completed successfully")
                yield None, "\n".join(status_log), last_chunk_video_path, "Image upscaling complete", None
                
            except Exception as e:
                logger.error(f"Error in image upscaler processing: {e}", exc_info=True)
                status_log.append(f"Image upscaler error: {e}")
                yield None, "\n".join(status_log), last_chunk_video_path, f"Error: {e}", None
                raise gr.Error(f"Image upscaling failed: {e}")
                
        elif enable_seedvr2:
            # Route to SeedVR2 processing
            logger.info("SeedVR2 mode enabled - using advanced AI video upscaling with temporal consistency")
            status_log.append("Using SeedVR2 video upscaling for superior quality and temporal consistency")
            yield None, "\n".join(status_log), last_chunk_video_path, "Initializing SeedVR2...", None

            # --- SeedVR2 Progress mapping wrapper ---
            def _seedvr2_progress_wrapper(rel_val: float, desc: str = ""):
                """Translate SeedVR2 relative progress (0-1) to absolute pipeline progress."""
                try:
                    rel_val_clamped = max(0.0, min(1.0, rel_val))
                    abs_progress = upscaling_loop_progress_start + rel_val_clamped * stage_weights["upscaling_loop"]
                    progress(abs_progress, desc=desc)
                except Exception as prog_exc:
                    if logger:
                        logger.debug(f"SeedVR2 progress wrapper error (ignored): {prog_exc}")

            # Import SeedVR2 CLI core here to avoid circular imports and ComfyUI dependencies
            from .seedvr2_cli_core import process_video_with_seedvr2_cli

            try:
                # Call SeedVR2 processing with generator pattern
                seedvr2_generator = process_video_with_seedvr2_cli(
                    input_video_path=current_input_video_for_frames,
                    seedvr2_config=seedvr2_config,
                    
                    # Video processing parameters
                    enable_target_res=enable_target_res,
                    target_h=target_h,
                    target_w=target_w,
                    target_res_mode=target_res_mode,
                    
                    # Frame saving and output parameters
                    save_frames=save_frames,
                    save_metadata=save_metadata,
                    save_chunks=save_chunks,
                    save_chunk_frames=save_chunk_frames,
                    
                    # Scene processing parameters
                    enable_scene_split=enable_scene_split,
                    scene_min_scene_len=scene_min_scene_len,
                    scene_threshold=scene_threshold,
                    
                    # Output parameters
                    output_folder=main_output_dir,
                    temp_folder=temp_dir,
                    create_comparison_video=create_comparison_video_enabled,
                    
                    # Session directory management (NEW - use existing session)
                    session_output_dir=frames_output_subfolder if save_frames else os.path.join(main_output_dir, base_output_filename_no_ext),
                    base_output_filename_no_ext=base_output_filename_no_ext,
                    
                    # ✅ FIX: Pass user's chunk frame count setting from SeedVR2 config
                    max_chunk_len=seedvr2_config.chunk_preview_frames,
                    
                    # Global settings
                    ffmpeg_preset=ffmpeg_preset,
                    ffmpeg_quality=ffmpeg_quality_value,
                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                    seed=current_seed,
                    
                    # Callbacks
                    progress_callback=_seedvr2_progress_wrapper,
                    status_callback=lambda msg: status_log.append(msg),
                    
                    # Dependencies
                    logger=logger
                )
                
                # Process SeedVR2 generator and yield updates
                last_chunk_video_path = None
                output_video_path = None
                for result_output_video_path, status_msg, chunk_video_path, chunk_status, comparison_video_path in seedvr2_generator:
                    logger.info(f"🔄 SeedVR2 yield received - chunk_video_path: {chunk_video_path}, chunk_status: {chunk_status}")
                    
                    # ✅ FIX: Use the detailed status_msg for the main UI display and log it correctly.
                    effective_chunk_status = chunk_status or "SeedVR2 processing..."
                    if status_msg and isinstance(status_msg, str) and "SeedVR2 processing" not in status_msg:
                        # This is a rich progress message (e.g., "Batch 2/25... ETA: ...")
                        # Use it as the primary status and append it to the log if it's new.
                        effective_chunk_status = status_msg
                        if not status_log or status_log[-1] != status_msg:
                            status_log.append(status_msg)
                    elif status_msg and not isinstance(status_msg, str):
                        # Handle non-string status messages
                        logger.warning(f"Received non-string status_msg: {type(status_msg)}")
                        effective_chunk_status = chunk_status or "SeedVR2 processing..."

                    # Update chunk preview path
                    if chunk_video_path:
                        last_chunk_video_path = chunk_video_path
                        logger.info(f"📹 Updating chunk preview path to: {chunk_video_path}")
                    
                    # Update output video path
                    if result_output_video_path:
                        output_video_path = result_output_video_path
                    
                    # Yield progress update with the rich, effective status
                    logger.info(f"🔼 Yielding to UI - chunk_video: {last_chunk_video_path}, status: {effective_chunk_status}")
                    # For SeedVR2 batch progress updates, only yield the current status message
                    if any(marker in effective_chunk_status for marker in ["🎬 Batch", "⏳ Processing..."]):
                        yield None, effective_chunk_status, last_chunk_video_path, effective_chunk_status, comparison_video_path
                    else:
                        yield None, "\n".join(status_log), last_chunk_video_path, effective_chunk_status, comparison_video_path
                
                # SeedVR2 processing completed successfully
                if output_video_path and os.path.exists(output_video_path):
                    status_log.append(f"✅ SeedVR2 processing completed: {os.path.basename(output_video_path)}")
                    final_output_video_path = output_video_path
                    
                    # For scene splitting with SeedVR2, ensure proper naming for audio merging
                    if enable_scene_split:
                        # Use STAR naming pattern for intermediate file
                        silent_upscaled_video_path = os.path.join(temp_dir, "silent_upscaled_video.mp4")
                        # Copy/move the SeedVR2 output to the expected path
                        if output_video_path != silent_upscaled_video_path:
                            shutil.copy2(output_video_path, silent_upscaled_video_path)
                            logger.info(f"SeedVR2: Copied output to intermediate path for audio merge: {silent_upscaled_video_path}")
                        
                        # Handle partial results for cancellation
                        current_final_path = params_for_metadata.get("final_output_path")
                        if current_final_path and not any(indicator in os.path.basename(output_video_path) for indicator in ["partial", "cancelled"]):
                            # Complete result - use the permanent target path
                            seedvr2_temp_output = output_video_path  # Save the temp path returned by SeedVR2
                            output_video_path = current_final_path  # Restore the permanent target path
                            logger.info(f"SeedVR2: Restoring permanent output path: {output_video_path} (was temp: {seedvr2_temp_output})")
                        else:
                            # Partial result or no permanent path set - preserve the path returned by SeedVR2
                            logger.info(f"SeedVR2: Using partial/cancelled result path: {output_video_path}")
                    else:
                        # Direct upscaling without scene splitting
                        silent_upscaled_video_path = output_video_path
                    
                    params_for_metadata["input_fps"] = input_fps_val
                    
                    # Check for chunk preview files and yield updates
                    if seedvr2_config and seedvr2_config.enable_chunk_preview:
                        # Look for chunk preview files in the session directory
                        session_chunks_dir = os.path.join(main_output_dir, base_output_filename_no_ext, "chunks")
                        if os.path.exists(session_chunks_dir):
                            chunk_files = sorted([f for f in os.listdir(session_chunks_dir) 
                                               if f.startswith('chunk_') and f.endswith('.mp4')])
                            if chunk_files:
                                last_chunk_video_path = os.path.join(session_chunks_dir, chunk_files[-1])
                                logger.info(f"Found SeedVR2 chunk preview: {last_chunk_video_path}")
                                # Yield chunk preview update
                                yield None, "\n".join(status_log), last_chunk_video_path, "SeedVR2 chunk preview available", None
                else:
                    logger.error("SeedVR2 processing failed: no output video generated")
                    status_log.append("❌ SeedVR2 processing failed")
                    raise gr.Error("SeedVR2 processing failed to generate output video")
                
                # Update progress after SeedVR2 processing
                current_overall_progress = upscaling_loop_progress_start + stage_weights["upscaling_loop"]
                progress(current_overall_progress, desc="SeedVR2 upscaling complete")
                status_log.append("SeedVR2 upscaling completed successfully")
                logger.info("SeedVR2 upscaling completed successfully")
                yield None, "\n".join(status_log), last_chunk_video_path, "SeedVR2 upscaling complete", None
                
            except Exception as e:
                logger.error(f"Error in SeedVR2 processing: {e}", exc_info=True)
                status_log.append(f"SeedVR2 error: {e}")
                yield None, "\n".join(status_log), last_chunk_video_path, f"Error: {e}", None
                raise gr.Error(f"SeedVR2 upscaling failed: {e}")
                
        elif enable_scene_split and scene_video_paths :
            processed_scene_videos =[]
            total_scenes =len (scene_video_paths )
            first_scene_caption =None

            if enable_auto_caption_per_scene and total_scenes >0 :
                logger .info ("Auto-captioning first scene to update main prompt before processing...")
                progress (current_overall_progress ,desc ="Generating caption for first scene...")
                try :
                    # Check for cancellation before auto-captioning first scene
                    cancellation_manager.check_cancel("before first scene auto-captioning")
                    
                    first_scene_caption_result ,_ =util_auto_caption (
                    scene_video_paths [0 ],actual_cogvlm_quant_val ,cogvlm_unload ,
                    app_config_module .COG_VLM_MODEL_PATH ,logger =logger ,progress =progress
                    )
                    if not first_scene_caption_result .startswith ("Error:") and not first_scene_caption_result.startswith("Caption generation cancelled"):
                        first_scene_caption =first_scene_caption_result
                        logger .info (f"First scene caption generated for main prompt: '{first_scene_caption[:100]}...'")
                        caption_update_msg =f"First scene caption generated [FIRST_SCENE_CAPTION:{first_scene_caption}]"
                        status_log .append (caption_update_msg )
                        logger .info (f"Yielding first scene caption for immediate prompt update")
                        # Update metadata to reflect the auto-generated caption
                        params_for_metadata["user_prompt"] = first_scene_caption
                        params_for_metadata["auto_caption_used"] = True
                        params_for_metadata["original_user_prompt"] = user_prompt
                        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status,None
                    else :
                        if first_scene_caption_result.startswith("Caption generation cancelled"):
                            logger .info ("First scene auto-captioning cancelled by user")
                        else:
                            logger .warning ("First scene auto-captioning failed, using original prompt")
                except CancelledError:
                    logger .info ("First scene auto-captioning cancelled by user - stopping scene processing")
                    raise  # Re-raise to stop the whole process
                except Exception as e :
                    logger .error (f"Error auto-captioning first scene: {e}", exc_info=True)

            # Flag to track if processing was cancelled
            processing_cancelled = False
            
            for scene_idx ,scene_video_path_item in enumerate (scene_video_paths ):
                # Check for cancellation before processing each scene
                cancellation_manager.check_cancel(f"before processing scene {scene_idx + 1}")
                
                scene_progress_start_abs =upscaling_loop_progress_start +(scene_idx /total_scenes )*stage_weights ["upscaling_loop"]
                def scene_upscale_progress_callback (progress_val_rel ,desc_scene ):
                    current_scene_overall_progress =scene_progress_start_abs +(progress_val_rel /total_scenes )*stage_weights ["upscaling_loop"]
                    progress (current_scene_overall_progress ,desc =desc_scene )

                try :
                    scene_enable_auto_caption_current =enable_auto_caption_per_scene and scene_idx >0 # Only for subsequent scenes if first was pre-captioned
                    scene_prompt_override =first_scene_caption if scene_idx ==0 and first_scene_caption else final_prompt

                    scene_processor_generator = process_single_scene (
                        scene_video_path=scene_video_path_item, scene_index=scene_idx, total_scenes=total_scenes, temp_dir=temp_dir, # star_model (instance) removed
                        final_prompt=scene_prompt_override, upscale_factor=upscale_factor_val, final_h=final_h_val, final_w=final_w_val, ui_total_diffusion_steps=ui_total_diffusion_steps,
                        solver_mode=solver_mode, cfg_scale=cfg_scale, max_chunk_len=max_chunk_len, enable_chunk_optimization=enable_chunk_optimization, vae_chunk=vae_chunk, enable_vram_optimization=enable_vram_optimization, color_fix_method=color_fix_method,
                        enable_tiling=enable_tiling, tile_size=tile_size, tile_overlap=tile_overlap, enable_context_window=enable_context_window, context_overlap=context_overlap,
                        save_frames=save_frames, scene_output_dir=frames_output_subfolder, progress_callback=scene_upscale_progress_callback,
                        enable_auto_caption_per_scene=scene_enable_auto_caption_current,
                        cogvlm_quant=actual_cogvlm_quant_val,
                        cogvlm_unload=cogvlm_unload,
                        progress=progress,
                        save_chunks=save_chunks,
                        chunks_permanent_save_path=frames_output_subfolder, # This was scene_output_dir, should be specific path for chunks
                        ffmpeg_preset=ffmpeg_preset, ffmpeg_quality_value=ffmpeg_quality_value, ffmpeg_use_gpu=ffmpeg_use_gpu,
                        save_metadata=save_metadata, metadata_params_base=scene_metadata_base_params,
                        save_chunk_frames=save_chunk_frames,  # NEW: Pass save_chunk_frames parameter
                        
                        # FPS decrease parameters for scenes
                        enable_fps_decrease=enable_fps_decrease, fps_decrease_mode=fps_decrease_mode, 
                        fps_multiplier_preset=fps_multiplier_preset, fps_multiplier_custom=fps_multiplier_custom,
                        target_fps=target_fps, fps_interpolation_method=fps_interpolation_method,
                        
                        # RIFE interpolation parameters for scenes and chunks
                        enable_rife_interpolation=enable_rife_interpolation, rife_multiplier=rife_multiplier, rife_fp16=rife_fp16, 
                        rife_uhd=rife_uhd, rife_scale=rife_scale, rife_skip_static=rife_skip_static, 
                        rife_enable_fps_limit=rife_enable_fps_limit, rife_max_fps_limit=rife_max_fps_limit,
                        rife_apply_to_scenes=rife_apply_to_scenes, rife_apply_to_chunks=rife_apply_to_chunks, 
                        rife_keep_original=rife_keep_original, current_seed=current_seed,
                        
                        # Image upscaler parameters for scenes
                        enable_image_upscaler=enable_image_upscaler, image_upscaler_model=image_upscaler_model, 
                        image_upscaler_batch_size=image_upscaler_batch_size,
                        
                        # Face restoration parameters for scenes
                        enable_face_restoration=enable_face_restoration, face_restoration_fidelity=face_restoration_fidelity,
                        enable_face_colorization=enable_face_colorization, face_restoration_timing=face_restoration_timing,
                        face_restoration_when=face_restoration_when, codeformer_model=codeformer_model,
                        face_restoration_batch_size=face_restoration_batch_size,
                        
                        util_extract_frames=util_extract_frames, util_auto_caption=util_auto_caption, 
                        util_create_video_from_frames=util_create_video_from_frames, 
                        logger=logger,
                        metadata_handler=metadata_handler_module, format_time=format_time, preprocess=preprocess_func,
                        ImageSpliterTh=ImageSpliterTh_class, collate_fn=collate_fn_func, tensor2vid=tensor2vid_func,
                        adain_color_fix=adain_color_fix_func, wavelet_color_fix=wavelet_color_fix_func,
                        VideoToVideo_sr_class=VideoToVideo_sr_class, # Pass the class
                        EasyDict_class=EasyDict_class, # Pass the class
                        app_config_module_param=app_config_module, # Pass the module for paths etc.
                        util_get_gpu_device_param=util_get_gpu_device # Pass the function for device selection
                    )
                    processed_scene_video_path_final =None
                    for yield_type ,*data in scene_processor_generator :
                        if yield_type =="chunk_update":
                            chunk_vid_path ,chunk_stat_str =data
                            last_chunk_video_path =chunk_vid_path
                            last_chunk_status =chunk_stat_str
                            temp_status_log =status_log +[f"Processed: {chunk_stat_str}"]
                            yield None ,"\n".join (temp_status_log ),last_chunk_video_path ,last_chunk_status,None
                        elif yield_type =="scene_complete":
                            processed_scene_video_path_final ,scene_frame_count_ret ,scene_fps_ret ,scene_caption =data
                            if scene_idx ==0 :
                                input_fps_val =scene_fps_ret 
                                if first_scene_caption and not scene_caption: 
                                     scene_caption = first_scene_caption
                            scene_complete_msg =f"Scene {scene_idx + 1}/{total_scenes} processing complete"
                            if scene_idx ==0 and scene_caption and enable_auto_caption_per_scene :
                                scene_complete_msg +=f" [FIRST_SCENE_CAPTION:{scene_caption}]" 
                                logger .info (f"Scene 1 complete with caption for main prompt update")
                            status_log .append (scene_complete_msg )
                            logger .info (scene_complete_msg )
                            yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status,None
                        elif yield_type =="error":
                            error_message =data [0 ]
                            logger .error (f"Error from scene_processor_generator: {error_message}")
                            status_log .append (f"Error processing scene {scene_idx + 1}: {error_message}")
                            yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene {scene_idx + 1} processing failed: {error_message}",None
                            
                            # Check if the error is a cancellation - handle differently for graceful partial output
                            if "cancelled" in error_message.lower() or "cancellation" in error_message.lower():
                                logger.info(f"Scene {scene_idx + 1} was cancelled - checking for partial scene video")
                                processing_cancelled = True
                                # Don't break immediately - continue to process any subsequent yields (like scene_complete with partial video)
                                # The scene processor may yield a partial scene video after the cancellation error
                                continue  # Continue to process next yield instead of breaking
                            elif not is_batch_mode: # Only raise error if not in batch, to allow batch to continue
                                raise gr.Error(f"Scene {scene_idx + 1} processing failed: {error_message}")
                            else:
                                logger.error(f"BATCH MODE: Scene {scene_idx + 1} processing failed: {error_message}. Skipping this scene for merging.")
                                # Continue to next scene in batch context
                        elif yield_type == "scene_complete":
                            # Handle both normal and partial scene completion
                            processed_scene_video_path_final = data[0]
                            if len(data) > 1:
                                scene_frame_count = data[1]
                            if len(data) > 2:
                                scene_fps = data[2]
                            # Break out of the yield processing loop to add the scene to processed list
                            break
                    if processed_scene_video_path_final :
                         processed_scene_videos .append (processed_scene_video_path_final )
                         logger.info(f"Scene {scene_idx + 1}: Added processed scene video to merge list: {processed_scene_video_path_final}")
                    elif not processing_cancelled:
                        # Only raise error if it wasn't cancelled - cancellation is handled gracefully
                        logger .error (f"Scene {scene_idx+1} finished processing but no final video path was yielded by process_single_scene.")
                        raise gr .Error (f"Scene {scene_idx+1} did not complete correctly.")
                    
                    # If processing was cancelled, break from the scene loop after processing all yields
                    if processing_cancelled:
                        logger.info(f"Scene {scene_idx + 1} processing cancelled - stopping scene loop for partial output generation")
                        status_log.append(f"Scene {scene_idx + 1} processing cancelled - generating partial output")
                        yield None, "\n".join(status_log), last_chunk_video_path, f"Scene {scene_idx + 1} cancelled - creating partial output", None
                        break
                except CancelledError as e_cancel:
                    logger.info(f"Scene {scene_idx + 1} processing cancelled by user - proceeding with partial scene merge")
                    status_log.append(f"Scene {scene_idx + 1} processing cancelled - generating partial output")
                    yield None, "\n".join(status_log), last_chunk_video_path, f"Scene {scene_idx + 1} cancelled - creating partial output", None
                    processing_cancelled = True
                    # Break the loop to proceed with merging any completed scenes for partial output
                    break
                except Exception as e :
                    # Check if this is a cancellation-related error wrapped in another exception
                    error_str = str(e).lower()
                    is_cancellation_error = "cancelled" in error_str or "cancellation" in error_str
                    
                    logger .error (f"Error processing scene {scene_idx + 1} in run_upscale: {e}",exc_info =True )
                    status_log .append (f"Error processing scene {scene_idx + 1}: {e}")
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene {scene_idx + 1} processing failed: {e}",None
                    
                    if is_cancellation_error:
                        logger.info(f"Scene {scene_idx + 1} error was cancellation-related - proceeding with partial scene merge")
                        processing_cancelled = True
                        break  # Exit the scene processing loop to generate partial output
                    elif not is_batch_mode: # Only raise error if not in batch, to allow batch to continue
                        raise gr.Error(f"Scene {scene_idx + 1} processing failed: {e}")
                    else:
                        logger.error(f"BATCH MODE: Scene {scene_idx + 1} processing failed: {e}. Skipping this scene for merging.")
                        # Continue to next scene in batch context

            current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
            scene_merge_progress_start =current_overall_progress
            
            # Handle partial scene processing (from cancellation or errors)
            if processed_scene_videos:
                progress (current_overall_progress ,desc ="Merging processed scenes...")
                if len(processed_scene_videos) < total_scenes:
                    partial_msg = f"Merging {len(processed_scene_videos)}/{total_scenes} processed scenes (partial output)..."
                    status_log.append(partial_msg)
                    logger.info(partial_msg)
                else:
                    status_log.append("Merging processed scenes...")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,"Merging Scenes...",None

                silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
                # Allow partial merge when processing was cancelled but we have some completed scenes
                allow_partial = processing_cancelled if 'processing_cancelled' in locals() else False
                util_merge_scene_videos (processed_scene_videos ,silent_upscaled_video_path ,temp_dir ,
                ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger, allow_partial_merge=allow_partial )

                current_overall_progress =scene_merge_progress_start +stage_weights ["scene_merge"]
                progress (current_overall_progress ,desc ="Scene merging complete")
                if len(processed_scene_videos) < total_scenes:
                    scene_merge_msg = f"Successfully merged {len(processed_scene_videos)}/{total_scenes} processed scenes (partial output)"
                else:
                    scene_merge_msg = f"Successfully merged {len(processed_scene_videos)} processed scenes"
                status_log .append (scene_merge_msg )
                logger .info (scene_merge_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_merge_msg,None
            else:
                # No scenes were processed due to cancellation or errors
                if processing_cancelled:
                    logger.info("No scenes were processed due to user cancellation - creating empty output for audio transfer")
                    status_log.append("🚨 Processing cancelled - preparing audio transfer from original video")
                    yield None, "\n".join(status_log), last_chunk_video_path, "Cancelled - preparing audio transfer", None
                    # Create a placeholder/empty video for audio transfer to work
                    silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
                    # Copy the original video as a fallback for audio transfer
                    shutil.copy2(input_video_path, silent_upscaled_video_path)
                    logger.info("Using original video as fallback for audio transfer due to cancellation")
                else:
                    logger.warning("No scenes were processed successfully")
                    status_log.append("⚠️ No scenes were processed successfully - cannot create output video")
                    yield None, "\n".join(status_log), last_chunk_video_path, "No scenes processed", None
                    raise gr.Error("No scenes were processed successfully")
        else : # No scene splitting (direct upscale)
            # This is the start of the block that was mis-indented.
            # It should be an 'else' to the 'if enable_scene_split:'
            # However, the logic flow is: if scene_split, do the loop. 
            # If NOT scene_split, do the direct upscaling. So it's 'if enable_scene_split:' then 'else:'.
            # The 'and scene_video_paths' is for AFTER splitting.

            # The mis-indented 'else' started around here in the previous diff. 
            # The following is the NON-SCENE-SPLIT path.
            
            # Setup chunk directories for STAR processing only
            if save_chunks:
                chunks_permanent_save_path = os.path.join(frames_output_subfolder if frames_output_subfolder else main_output_dir, "chunks")
                os.makedirs(chunks_permanent_save_path, exist_ok=True)

            # Initialize chunk_frames directory for debugging if save_chunk_frames is enabled
            if save_chunk_frames:
                chunk_frames_permanent_save_path = os.path.join(frames_output_subfolder if frames_output_subfolder else main_output_dir, "chunk_frames")
                os.makedirs(chunk_frames_permanent_save_path, exist_ok=True)
                logger.info(f"Chunk input frames will be saved to: {chunk_frames_permanent_save_path}")
            
            upscaling_loop_progress_start_no_scene_split = current_overall_progress
            status_log.append(f"Starting direct upscaling of {frame_count} frames...")
            logger.info(f"Starting direct upscaling of {frame_count} frames, without scene splitting.")
            yield None, "\n".join(status_log), last_chunk_video_path, "Starting direct upscaling...", None

            # MODEL LOADING FOR NON-SCENE-SPLIT MODE
            logger.info("Non-scene-split: Initializing STAR model.")
            model_load_start_time_ns = time.time()
            model_cfg_ns = EasyDict_class()
            model_cfg_ns.model_path = model_file_path # model_file_path determined earlier
            model_device_ns = torch.device(util_get_gpu_device(logger=logger)) if torch.cuda.is_available() else torch.device('cpu')
            star_model_ns = VideoToVideo_sr_class(model_cfg_ns, device=model_device_ns, enable_vram_optimization=enable_vram_optimization)
            logger.info(f"Non-scene-split: STAR model loaded on {model_device_ns}. Load time: {format_time(time.time() - model_load_start_time_ns)}")

            all_lr_frames_bgr = []
            # Ensure frame_files is defined if not enable_scene_split path was taken before
            if 'frame_files' not in locals() or not frame_files:
                 # This implies frame extraction for non-scene-split happened before this block if logic is correct
                 # For safety, one might re-list or ensure it's passed correctly.
                 # Assuming frame_files IS available from the earlier frame extraction for non-scene-split mode.
                 pass # It should have been populated by lines ~390-400

            for frame_filename_ns in frame_files: # Use frame_files from extraction stage
                frame_lr_bgr_ns = cv2.imread(os.path.join(input_frames_dir, frame_filename_ns))
                if frame_lr_bgr_ns is None:
                    logger.error(f"Could not read frame {frame_filename_ns} for direct upscaling")
                    continue
                all_lr_frames_bgr.append(frame_lr_bgr_ns)

            if not all_lr_frames_bgr:
                raise gr.Error("No valid frames found for direct upscaling.")
            
            if enable_context_window:
                # CONTEXT WINDOW PROCESSING WITH CHUNK SAVING  
                loop_name = "Context Window Process"
                context_status_msg = f"Context Window: Max Chunk={max_chunk_len}, Context Overlap={context_overlap}. Processing {frame_count} frames."
                status_log.append(context_status_msg)
                logger.info(context_status_msg)
                yield None, "\n".join(status_log), last_chunk_video_path, context_status_msg, None

                # Import the new context processor
                from .context_processor import calculate_context_chunks, get_chunk_frame_indices, validate_chunk_plan, format_chunk_plan_summary

                # Calculate context-based chunk plan
                context_chunks = calculate_context_chunks(
                    total_frames=frame_count,
                    max_chunk_len=max_chunk_len,
                    context_overlap=context_overlap
                )

                # Validate chunk plan
                is_valid, validation_errors = validate_chunk_plan(context_chunks, frame_count)
                if not is_valid:
                    error_msg = f"Context chunk plan validation failed: {', '.join(validation_errors)}"
                    logger.error(error_msg)
                    raise gr.Error(error_msg)

                # Log chunk plan summary
                chunk_summary = format_chunk_plan_summary(context_chunks, frame_count, max_chunk_len, context_overlap)
                logger.info(f"Context processing plan:\n{chunk_summary}")
                status_log.append(f"📊 Context Plan: {len(context_chunks)} chunks")
                yield None, "\n".join(status_log), last_chunk_video_path, f"Context Plan: {len(context_chunks)} chunks", None

                # Initialize context processing parameters
                processed_frame_filenames = [None] * frame_count
                total_chunks_to_process = len(context_chunks)
                processed_chunks_tracker = [False] * total_chunks_to_process

                def save_context_window_chunk(chunk_info, chunk_idx, save_chunks, chunks_permanent_save_path, 
                                            temp_dir, output_frames_dir, frame_files, input_fps_val, ffmpeg_preset, 
                                            ffmpeg_quality_value, ffmpeg_use_gpu, logger, enable_rife_interpolation, 
                                            rife_apply_to_chunks, rife_multiplier, rife_fp16, rife_uhd, rife_scale, 
                                            rife_skip_static, rife_enable_fps_limit, rife_max_fps_limit, rife_keep_original, 
                                            current_seed):
                    """Save a chunk video from context window processed frames"""
                    if not save_chunks or not chunks_permanent_save_path:
                        return None, None
                    
                    if not chunk_info:
                        return None, None
                    
                    # Get output frame range (what was actually generated)
                    output_start_0 = chunk_info['output_start']  # Already 0-based from context processor
                    output_end_0 = chunk_info['output_end']      # Already 0-based from context processor
                    current_chunk_display_num = chunk_idx + 1
                    
                    chunk_video_filename = f"chunk_{current_chunk_display_num:04d}.mp4"
                    chunk_video_path = os.path.join(chunks_permanent_save_path, chunk_video_filename)
                    chunk_temp_dir = os.path.join(temp_dir, f"temp_sliding_chunk_{current_chunk_display_num}")
                    os.makedirs(chunk_temp_dir, exist_ok=True)

                    # Get the output frame range for chunk video creation
                    start_frame = output_start_0
                    end_frame = output_end_0 + 1  # Make end_frame exclusive for slicing
                    
                    frames_for_chunk = []
                    for k, frame_name in enumerate(frame_files[start_frame:end_frame]):
                        src_frame = os.path.join(output_frames_dir, frame_name)
                        dst_frame = os.path.join(chunk_temp_dir, f"frame_{k+1:06d}.png")
                        if os.path.exists(src_frame):
                            shutil.copy2(src_frame, dst_frame)
                            frames_for_chunk.append(dst_frame)
                        else:
                            logger.warning(f"Src frame {src_frame} not found for context window chunk video.")
                    
                    if frames_for_chunk:
                        # Use duration-preserved video creation for context window chunks
                        from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
                        create_video_from_frames_with_duration_preservation(
                            chunk_temp_dir, chunk_video_path, current_input_video_for_frames,
                            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)
                        
                        # Apply RIFE interpolation to context window chunk if enabled
                        final_chunk_video_path = chunk_video_path  # Default to original chunk
                        if enable_rife_interpolation and rife_apply_to_chunks and os.path.exists(chunk_video_path):
                            try:
                                # Import RIFE function locally to avoid circular imports
                                from .rife_interpolation import increase_fps_single
                                
                                # Generate RIFE chunk output path
                                chunk_video_dir = os.path.dirname(chunk_video_path)
                                chunk_video_name = os.path.splitext(os.path.basename(chunk_video_path))[0]
                                rife_chunk_video_path = os.path.join(chunk_video_dir, f"{chunk_video_name}_{rife_multiplier}x_FPS.mp4")
                                
                                # Apply RIFE interpolation to chunk
                                rife_result, rife_message = increase_fps_single(
                                    video_path=chunk_video_path,
                                    output_path=rife_chunk_video_path,
                                    multiplier=rife_multiplier,
                                    fp16=rife_fp16,
                                    uhd=rife_uhd,
                                    scale=rife_scale,
                                    skip_static=rife_skip_static,
                                    enable_fps_limit=rife_enable_fps_limit,
                                    max_fps_limit=rife_max_fps_limit,
                                    ffmpeg_preset=ffmpeg_preset,
                                    ffmpeg_quality_value=ffmpeg_quality_value,
                                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                                    overwrite_original=False,  # Don't overwrite chunks, keep both
                                    keep_original=rife_keep_original,
                                    output_dir=chunk_video_dir,
                                    seed=current_seed,
                                    logger=logger,
                                    progress=None  # Don't pass progress to avoid conflicts
                                )
                                
                                if rife_result:
                                    final_chunk_video_path = rife_result  # Use RIFE version as final chunk
                                    logger.info(f"Context Window Chunk {current_chunk_display_num}: RIFE interpolation completed")
                                else:
                                    logger.warning(f"Context Window Chunk {current_chunk_display_num}: RIFE interpolation failed: {rife_message}")
                                    
                            except Exception as e_chunk_rife:
                                logger.error(f"Context Window Chunk {current_chunk_display_num}: Error during RIFE interpolation: {e_chunk_rife}")
                        
                        shutil.rmtree(chunk_temp_dir)
                        
                        effective_frame_count = chunk_info['new_frames']  # Use new_frames from context processor
                        chunk_save_msg = f"Saved context window chunk {current_chunk_display_num} ({effective_frame_count} frames: {start_frame+1}-{end_frame}) to: {final_chunk_video_path}"
                        logger.info(chunk_save_msg)
                        
                        return final_chunk_video_path, f"Context Window Chunk {current_chunk_display_num} ({effective_frame_count} frames: {start_frame+1}-{end_frame})"
                    else:
                        logger.warning(f"No frames for context window chunk {current_chunk_display_num}, video not created.")
                        shutil.rmtree(chunk_temp_dir)
                        return None, None

                # Add missing utility functions for sliding window processing
                def get_chunk_frame_range(chunk_idx, max_chunk_len, total_frames):
                    """Get the frame range for a specific chunk"""
                    start_frame = chunk_idx * max_chunk_len
                    end_frame = min((chunk_idx + 1) * max_chunk_len, total_frames)
                    return start_frame, end_frame

                def map_window_to_chunks(start_idx, end_idx, max_chunk_len):
                    """Map a window frame range to affected chunk indices"""
                    first_chunk = start_idx // max_chunk_len
                    last_chunk = (end_idx - 1) // max_chunk_len
                    return list(range(first_chunk, last_chunk + 1))

                def is_chunk_complete(chunk_idx, processed_frames_tracker, max_chunk_len, total_frames):
                    """Check if all frames in a chunk have been processed"""
                    start_frame, end_frame = get_chunk_frame_range(chunk_idx, max_chunk_len, total_frames)
                    for frame_idx in range(start_frame, end_frame):
                        if not processed_frames_tracker[frame_idx]:
                            return False
                    return True

                def get_effective_chunk_mappings(frame_count, effective_window_step):
                    """Calculate effective chunk mappings for sliding window processing"""
                    chunks = []
                    chunk_idx = 0
                    start_frame = 0
                    
                    while start_frame < frame_count:
                        end_frame = min(start_frame + effective_window_step, frame_count)
                        frame_count_for_chunk = end_frame - start_frame
                        
                        chunk_info = {
                            'chunk_idx': chunk_idx,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'frame_count': frame_count_for_chunk
                        }
                        chunks.append(chunk_info)
                        
                        chunk_idx += 1
                        start_frame = end_frame
                    
                    return chunks

                # Process context window chunks
                processed_frame_filenames = [None] * frame_count
                total_chunks_to_process = len(context_chunks)

                for chunk_idx, chunk_info in enumerate(context_chunks):
                    # Check for cancellation at the start of each chunk
                    cancellation_manager.check_cancel()
                    
                    # Get chunk processing frames
                    processing_frames, output_frames_indices = get_chunk_frame_indices(
                        chunk_info
                    )
                    
                    # Get output frames (frames that will be saved from this chunk)
                    output_frames = output_frames_indices
                    
                    # Extract frames for processing
                    chunk_lr_frames_bgr = [all_lr_frames_bgr[i] for i in processing_frames]
                    
                    if not chunk_lr_frames_bgr:
                        continue

                    chunk_lr_video_data = preprocess_func(chunk_lr_frames_bgr)
                    chunk_pre_data = {'video_data': chunk_lr_video_data, 'y': final_prompt,
                                     'target_res': (final_h_val, final_w_val)}
                    chunk_data_cuda = collate_fn_func(chunk_pre_data, model_device_ns)

                    current_chunk_display_num = chunk_idx + 1

                    chunk_diffusion_timer = {'last_time': time.time()}
                    def diffusion_callback_for_chunk(step=None, total_steps=None, step_cb=None, total_steps_cb=None):
                        nonlocal chunk_diffusion_timer
                        
                        # Check for cancellation at each diffusion step callback
                        cancellation_manager.check_cancel()
                        
                        # Handle both keyword and positional argument styles
                        current_step = step if step is not None else step_cb
                        current_total_steps = total_steps if total_steps is not None else total_steps_cb
                        
                        current_time = time.time()
                        step_duration = current_time - chunk_diffusion_timer['last_time']
                        chunk_diffusion_timer['last_time'] = current_time

                        _current_chunk_loop_time_cb = time.time() - upscaling_loop_start_time
                        _avg_time_per_chunk_cb = _current_chunk_loop_time_cb / current_chunk_display_num if current_chunk_display_num > 0 else 0
                        _eta_seconds_chunk_cb = (total_chunks_to_process - current_chunk_display_num) * _avg_time_per_chunk_cb if current_chunk_display_num < total_chunks_to_process and _avg_time_per_chunk_cb > 0 else 0
                        _speed_chunk_cb = 1 / _avg_time_per_chunk_cb if _avg_time_per_chunk_cb > 0 else 0

                        base_desc_chunk = f"{loop_name}: {current_chunk_display_num}/{total_chunks_to_process} chunks (frames {chunk_info['output_start']}-{chunk_info['output_end']}) | ETA: {format_time(_eta_seconds_chunk_cb)} | Speed: {_speed_chunk_cb:.2f} c/s"
                        log_step_info = f"{step_duration:.2f}s/it" if step_duration > 0.001 else f"step {current_step}/{current_total_steps}"
                        logger.info(f"    ↳ {loop_name} - Chunk {current_chunk_display_num}/{total_chunks_to_process} (frames {chunk_info['output_start']}-{chunk_info['output_end']}) - {log_step_info}")

                        progress_val_rel = (current_chunk_display_num + (current_step / current_total_steps if current_total_steps > 0 else 1)) / total_chunks_to_process
                        current_overall_progress_temp = upscaling_loop_progress_start_no_scene_split + (progress_val_rel * stage_weights["upscaling_loop"])
                        progress(current_overall_progress_temp, desc=f"{base_desc_chunk} - Diffusion: {current_step}/{current_total_steps}")

                    with torch.no_grad():
                        # Reseed before each chunk to maintain deterministic noise across chunks for temporal consistency
                        if setup_seed_func is not None:
                            try:
                                setup_seed_func(current_seed)
                            except Exception as e_seed:
                                logger.warning(f"Reseed attempt inside context window failed: {e_seed}")

                        chunk_sr_tensor_bcthw = star_model_ns.test(
                            chunk_data_cuda, total_noise_levels, steps=ui_total_diffusion_steps, solver_mode=solver_mode,
                            guide_scale=cfg_scale, max_chunk_len=len(chunk_lr_frames_bgr), vae_decoder_chunk_size=min(vae_chunk, len(chunk_lr_frames_bgr)),
                            progress_callback=diffusion_callback_for_chunk, seed=current_seed
                        )
                    
                    chunk_sr_frames_uint8 = tensor2vid_func(chunk_sr_tensor_bcthw)

                    if color_fix_method != 'None':
                        if color_fix_method == 'AdaIN':
                            chunk_sr_frames_uint8 = adain_color_fix_func(chunk_sr_frames_uint8, chunk_lr_video_data)
                        elif color_fix_method == 'Wavelet':
                            chunk_sr_frames_uint8 = wavelet_color_fix_func(chunk_sr_frames_uint8, chunk_lr_video_data)

                    # Save output frames (only the new frames, not the context frames)
                    context_offset = len(processing_frames) - len(output_frames)
                    frames_written_in_chunk = 0
                    for local_idx, global_frame_idx in enumerate(output_frames):
                        processed_frame_idx = context_offset + local_idx
                        if 0 <= global_frame_idx < frame_count and processed_frame_filenames[global_frame_idx] is None:
                            frame_np_hwc_uint8 = chunk_sr_frames_uint8[processed_frame_idx].cpu().numpy()
                            frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                            out_f_path = os.path.join(output_frames_dir, frame_files[global_frame_idx])
                            cv2.imwrite(out_f_path, frame_bgr)
                            processed_frame_filenames[global_frame_idx] = frame_files[global_frame_idx]
                            frames_written_in_chunk += 1
                            
                            # Log progress every 25 frames or at the end
                            if frames_written_in_chunk % 25 == 0 or local_idx == len(output_frames) - 1:
                                save_progress_msg = f"Context chunk {current_chunk_display_num}/{total_chunks_to_process}: Saved {frames_written_in_chunk}/{len(output_frames)} frames to disk"
                                logger.info(save_progress_msg)

                    # Save processed frames immediately if enabled
                    if save_frames and processed_frames_permanent_save_path:
                        chunk_frames_saved_count = 0
                        for global_frame_idx in output_frames:
                            if 0 <= global_frame_idx < frame_count and processed_frame_filenames[global_frame_idx] is not None:
                                frame_name = frame_files[global_frame_idx]
                                src_frame_path = os.path.join(output_frames_dir, frame_name)
                                dst_frame_path = os.path.join(processed_frames_permanent_save_path, frame_name)
                                if os.path.exists(src_frame_path) and not os.path.exists(dst_frame_path):
                                    shutil.copy2(src_frame_path, dst_frame_path)
                                    chunk_frames_saved_count += 1
                        
                        if chunk_frames_saved_count > 0:
                            immediate_save_msg = f"Immediately saved {chunk_frames_saved_count} processed frames from context chunk {current_chunk_display_num}/{total_chunks_to_process}"
                            logger.info(immediate_save_msg)
                            status_log.append(immediate_save_msg)

                    # Save chunk video if enabled
                    if save_chunks and chunks_permanent_save_path:
                        chunk_video_path, chunk_status = save_context_window_chunk(
                            chunk_info, chunk_idx, save_chunks, chunks_permanent_save_path,
                            temp_dir, output_frames_dir, frame_files, input_fps_val, ffmpeg_preset,
                            ffmpeg_quality_value, ffmpeg_use_gpu, logger, enable_rife_interpolation,
                            rife_apply_to_chunks, rife_multiplier, rife_fp16, rife_uhd, rife_scale,
                            rife_skip_static, rife_enable_fps_limit, rife_max_fps_limit, rife_keep_original,
                            current_seed
                        )
                        if chunk_video_path:
                            last_chunk_video_path = chunk_video_path
                            last_chunk_status = chunk_status
                            status_log.append(f"Saved context chunk {current_chunk_display_num}")
                            yield None, "\n".join(status_log), last_chunk_video_path, last_chunk_status, None

                    del chunk_data_cuda, chunk_sr_tensor_bcthw, chunk_sr_frames_uint8
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Progress update
                    loop_progress_frac = current_chunk_display_num / total_chunks_to_process if total_chunks_to_process > 0 else 1.0
                    current_overall_progress_temp = upscaling_loop_progress_start_no_scene_split + (loop_progress_frac * stage_weights["upscaling_loop"])
                    progress(current_overall_progress_temp, desc=f"Context chunk {current_chunk_display_num}/{total_chunks_to_process} processed")
                    yield None, "\n".join(status_log), last_chunk_video_path, f"Context chunk {current_chunk_display_num}/{total_chunks_to_process} processed", None

                # Handle any missed frames with fallback
                num_missed_fallback = 0
                for idx_fb, fname_fb in enumerate(frame_files):
                    if processed_frame_filenames[idx_fb] is None:
                        num_missed_fallback += 1
                        logger.warning(f"Frame {fname_fb} (index {idx_fb}) was not processed by context window, copying LR frame.")
                        lr_frame_path = os.path.join(input_frames_dir, fname_fb)
                        if os.path.exists(lr_frame_path):
                            shutil.copy2(lr_frame_path, os.path.join(output_frames_dir, fname_fb))
                        else:
                            logger.error(f"LR frame {lr_frame_path} not found for fallback copy.")
                
                if num_missed_fallback > 0:
                    missed_msg = f"{loop_name} - Copied {num_missed_fallback} LR frames as fallback for unprocessed frames."
                    status_log.append(missed_msg)
                    logger.info(missed_msg)
                    yield None, "\n".join(status_log), last_chunk_video_path, missed_msg, None

            else:
                # DIRECT CHUNK PROCESSING with optimization
                if enable_chunk_optimization:
                    # Import chunk optimization module
                    from .chunk_optimization import (
                        calculate_optimized_chunk_boundaries,
                        get_chunk_frames_for_processing,
                        extract_output_frames_from_processed,
                        get_output_frame_names,
                        log_chunk_optimization_summary
                    )
                    
                    # Calculate optimized chunk boundaries
                    chunk_boundaries = calculate_optimized_chunk_boundaries(
                        total_frames=frame_count,
                        max_chunk_len=max_chunk_len,
                        logger=logger
                    )
                    
                    # Log optimization summary
                    log_chunk_optimization_summary(chunk_boundaries, frame_count, max_chunk_len, logger)
                    
                    num_chunks_direct = len(chunk_boundaries)
                    
                else:
                    # Standard chunking (fallback)
                    num_chunks_direct = math.ceil(frame_count / max_chunk_len) if max_chunk_len > 0 else 1
                    chunk_boundaries = []
                    for chunk_idx in range(num_chunks_direct):
                        start_idx = chunk_idx * max_chunk_len
                        end_idx = min((chunk_idx + 1) * max_chunk_len, frame_count)
                        chunk_boundaries.append({
                            'chunk_idx': chunk_idx,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'process_start_idx': start_idx,
                            'process_end_idx': end_idx,
                            'output_start_offset': 0,
                            'output_end_offset': end_idx - start_idx,
                            'actual_output_frames': end_idx - start_idx
                        })
                
                for chunk_info in chunk_boundaries:
                    # Check for cancellation at the start of each chunk
                    cancellation_manager.check_cancel()
                    
                    chunk_idx_direct = chunk_info['chunk_idx']
                    start_idx_direct = chunk_info['start_idx']
                    end_idx_direct = chunk_info['end_idx']
                    process_start_idx = chunk_info['process_start_idx']
                    process_end_idx = chunk_info['process_end_idx']
                    current_chunk_len_direct = process_end_idx - process_start_idx
                    
                    if current_chunk_len_direct == 0:
                        continue

                    chunk_diffusion_timer_direct = {'last_time': time.time()}
                    
                    # Get the frames to process (may be more than output frames for optimized chunks)
                    chunk_lr_frames_bgr_direct = all_lr_frames_bgr[process_start_idx:process_end_idx]
                    
                    def diffusion_callback_for_chunk_direct(step=None, total_steps=None, total_steps_chunk=None):
                        nonlocal chunk_diffusion_timer_direct
                        
                        # Check for cancellation at each diffusion step callback
                        cancellation_manager.check_cancel()
                        
                        # Handle both keyword and positional argument styles
                        current_step = step if step is not None else 0
                        current_total_steps = total_steps if total_steps is not None else (total_steps_chunk if total_steps_chunk is not None else 1)
                        
                        current_time_chunk = time.time()
                        step_duration_chunk = current_time_chunk - chunk_diffusion_timer_direct['last_time']
                        chunk_diffusion_timer_direct['last_time'] = current_time_chunk
                        if enable_chunk_optimization and process_start_idx != start_idx_direct:
                            desc_for_log_direct = f"Direct Upscale Chunk {chunk_idx_direct + 1}/{num_chunks_direct} (processing {process_start_idx}-{process_end_idx - 1}, output {start_idx_direct}-{end_idx_direct - 1})"
                        else:
                            desc_for_log_direct = f"Direct Upscale Chunk {chunk_idx_direct + 1}/{num_chunks_direct} (frames {start_idx_direct}-{end_idx_direct - 1})"
                        eta_seconds_chunk = step_duration_chunk * (current_total_steps - current_step) if step_duration_chunk > 0 and current_total_steps > 0 else 0
                        eta_formatted_chunk = format_time(eta_seconds_chunk)
                        logger.info(f"{desc_for_log_direct} - Diffusion: Step {current_step}/{current_total_steps}, Duration: {step_duration_chunk:.2f}s, ETA: {eta_formatted_chunk}")
                        
                        progress_val_rel_direct = (chunk_idx_direct + (current_step / current_total_steps if current_total_steps > 0 else 1)) / num_chunks_direct
                        current_overall_progress_temp = upscaling_loop_progress_start_no_scene_split + (progress_val_rel_direct * stage_weights["upscaling_loop"])
                        progress(current_overall_progress_temp, desc=f"{desc_for_log_direct} (Diff: {current_step}/{current_total_steps})")

                    chunk_lr_video_data_direct = preprocess_func(chunk_lr_frames_bgr_direct)
                    chunk_pre_data_direct = {
                        'video_data': chunk_lr_video_data_direct, 
                        'y': final_prompt, 
                        'target_res': (final_h_val, final_w_val)
                    }
                    chunk_data_cuda_direct = collate_fn_func(chunk_pre_data_direct, model_device_ns)

                    with torch.no_grad():
                        chunk_sr_tensor_bcthw_direct = star_model_ns.test(
                            chunk_data_cuda_direct, total_noise_levels, steps=ui_total_diffusion_steps,
                            solver_mode=solver_mode, guide_scale=cfg_scale,
                            max_chunk_len=current_chunk_len_direct, 
                            vae_decoder_chunk_size=min(vae_chunk, current_chunk_len_direct),
                            progress_callback=diffusion_callback_for_chunk_direct, seed=current_seed
                        )
                    chunk_sr_frames_uint8_direct = tensor2vid_func(chunk_sr_tensor_bcthw_direct)

                    if color_fix_method != 'None':
                        if color_fix_method == 'AdaIN':
                            chunk_sr_frames_uint8_direct = adain_color_fix_func(chunk_sr_frames_uint8_direct, chunk_lr_video_data_direct)
                        elif color_fix_method == 'Wavelet':
                            chunk_sr_frames_uint8_direct = wavelet_color_fix_func(chunk_sr_frames_uint8_direct, chunk_lr_video_data_direct)

                    # Extract only the output frames (for optimized chunks, this trims the result)
                    if enable_chunk_optimization:
                        # Convert to list for easier slicing
                        chunk_sr_frames_list = [frame for frame in chunk_sr_frames_uint8_direct]
                        # Extract the output frames using chunk optimization logic
                        output_frames = extract_output_frames_from_processed(chunk_sr_frames_list, chunk_info)
                        # Get the corresponding frame names for output
                        output_frame_names = get_output_frame_names(frame_files, chunk_info)
                    else:
                        # Standard processing - use all frames
                        output_frames = [frame for frame in chunk_sr_frames_uint8_direct]
                        output_frame_names = frame_files[start_idx_direct:end_idx_direct]

                    # Save the output frames with correct names
                    for k_direct, (frame_tensor, frame_name_direct) in enumerate(zip(output_frames, output_frame_names)):
                        frame_np_hwc_uint8_direct = frame_tensor.cpu().numpy()
                        frame_bgr_direct = cv2.cvtColor(frame_np_hwc_uint8_direct, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(output_frames_dir, frame_name_direct), frame_bgr_direct)
                        
                        # Log progress every 25 frames or at the end
                        if (k_direct + 1) % 25 == 0 or (k_direct + 1) == len(output_frames):
                            save_progress_msg = f"Direct chunk {chunk_idx_direct + 1}/{num_chunks_direct}: Saved {k_direct + 1}/{len(output_frames)} frames to disk"
                            logger.info(save_progress_msg)

                    # IMMEDIATE FRAME SAVING: Save processed frames immediately after chunk completion
                    if save_frames and processed_frames_permanent_save_path:
                        chunk_frames_saved_count = 0
                        for frame_name_direct in output_frame_names:
                            src_frame_path = os.path.join(output_frames_dir, frame_name_direct)
                            dst_frame_path = os.path.join(processed_frames_permanent_save_path, frame_name_direct)
                            if os.path.exists(src_frame_path):
                                shutil.copy2(src_frame_path, dst_frame_path)
                                chunk_frames_saved_count += 1
                            else:
                                logger.warning(f"Frame {src_frame_path} not found for immediate saving in chunk {chunk_idx_direct + 1}")
                        
                        if chunk_frames_saved_count > 0:
                            immediate_save_msg = f"Immediately saved {chunk_frames_saved_count} processed frames from chunk {chunk_idx_direct + 1}/{num_chunks_direct} to permanent storage"
                            logger.info(immediate_save_msg)
                            status_log.append(immediate_save_msg)

                    if save_chunks and chunks_permanent_save_path:
                        current_direct_chunks_save_path = chunks_permanent_save_path
                        os.makedirs(current_direct_chunks_save_path, exist_ok=True)
                        chunk_video_filename_direct = f"chunk_{chunk_idx_direct + 1:04d}.mp4"
                        chunk_video_path_direct = os.path.join(current_direct_chunks_save_path, chunk_video_filename_direct)
                        chunk_temp_assembly_dir_direct = os.path.join(temp_dir, f"temp_direct_chunk_{chunk_idx_direct+1}")
                        os.makedirs(chunk_temp_assembly_dir_direct, exist_ok=True)
                        frames_for_this_video_chunk_direct = []
                        for k_chunk_frame_direct, frame_name_in_chunk_direct in enumerate(output_frame_names):
                            src_direct = os.path.join(output_frames_dir, frame_name_in_chunk_direct)
                            dst_direct = os.path.join(chunk_temp_assembly_dir_direct, f"frame_{k_chunk_frame_direct+1:06d}.png")
                            if os.path.exists(src_direct):
                                shutil.copy2(src_direct, dst_direct)
                                frames_for_this_video_chunk_direct.append(dst_direct)
                            else:
                                logger.warning(f"Src frame {src_direct} not found for direct upscale chunk video.")
                    if frames_for_this_video_chunk_direct:
                        # Use duration-preserved video creation for direct chunks
                        from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
                        create_video_from_frames_with_duration_preservation(
                            chunk_temp_assembly_dir_direct, chunk_video_path_direct, current_input_video_for_frames,
                            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)
                        
                        # Apply RIFE interpolation to direct upscale chunk if enabled
                        final_chunk_video_path_direct = chunk_video_path_direct  # Default to original chunk
                        if enable_rife_interpolation and rife_apply_to_chunks and os.path.exists(chunk_video_path_direct):
                            try:
                                # Import RIFE function locally to avoid circular imports
                                from .rife_interpolation import increase_fps_single
                                
                                # Generate RIFE chunk output path
                                chunk_video_dir_direct = os.path.dirname(chunk_video_path_direct)
                                chunk_video_name_direct = os.path.splitext(os.path.basename(chunk_video_path_direct))[0]
                                rife_chunk_video_path_direct = os.path.join(chunk_video_dir_direct, f"{chunk_video_name_direct}_{rife_multiplier}x_FPS.mp4")
                                
                                # Apply RIFE interpolation to chunk
                                rife_result_direct, rife_message_direct = increase_fps_single(
                                    video_path=chunk_video_path_direct,
                                    output_path=rife_chunk_video_path_direct,
                                    multiplier=rife_multiplier,
                                    fp16=rife_fp16,
                                    uhd=rife_uhd,
                                    scale=rife_scale,
                                    skip_static=rife_skip_static,
                                    enable_fps_limit=rife_enable_fps_limit,
                                    max_fps_limit=rife_max_fps_limit,
                                    ffmpeg_preset=ffmpeg_preset,
                                    ffmpeg_quality_value=ffmpeg_quality_value,
                                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                                    overwrite_original=False,  # Don't overwrite chunks, keep both
                                    keep_original=rife_keep_original,
                                    output_dir=chunk_video_dir_direct,
                                    seed=current_seed,
                                    logger=logger,
                                    progress=None  # Don't pass progress to avoid conflicts
                                )
                                
                                if rife_result_direct:
                                    final_chunk_video_path_direct = rife_result_direct  # Use RIFE version as final chunk
                                    logger.info(f"Direct Upscale Chunk {chunk_idx_direct+1}: RIFE interpolation completed")
                                else:
                                    logger.warning(f"Direct Upscale Chunk {chunk_idx_direct+1}: RIFE interpolation failed: {rife_message_direct}")
                                    
                            except Exception as e_chunk_rife_direct:
                                logger.error(f"Direct Upscale Chunk {chunk_idx_direct+1}: Error during RIFE interpolation: {e_chunk_rife_direct}")
                        
                        last_chunk_video_path = final_chunk_video_path_direct
                        last_chunk_status = f"Direct Upscale Chunk {chunk_idx_direct + 1}/{num_chunks_direct} (frames {start_idx_direct+1}-{end_idx_direct})"
                        logger.info(f"Saved direct upscale chunk {chunk_idx_direct+1}/{num_chunks_direct} to: {final_chunk_video_path_direct}")
                        temp_status_log_direct = status_log + [f"Processed: {last_chunk_status}"]
                        yield None, "\n".join(temp_status_log_direct), last_chunk_video_path, last_chunk_status, None
                    else:
                        logger.warning(f"No frames for direct upscale chunk {chunk_idx_direct+1}/{num_chunks_direct}, video not created.")
                    shutil.rmtree(chunk_temp_assembly_dir_direct)
                
                del chunk_data_cuda_direct, chunk_sr_tensor_bcthw_direct, chunk_sr_frames_uint8_direct, chunk_lr_video_data_direct
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if star_model_ns is not None:
                logger.info("Non-scene-split: Deleting STAR model instance from memory after all direct chunks.")
                del star_model_ns
                star_model_ns = None # Explicitly set to None
            if torch.cuda.is_available():
                logger.info("Non-scene-split: Clearing CUDA cache after all direct chunks.")
                torch.cuda.empty_cache()
            gc.collect() # Added explicit gc.collect after non-scene-split model processing

            current_overall_progress = upscaling_loop_progress_start_no_scene_split + stage_weights["upscaling_loop"]
            progress(current_overall_progress, desc="Direct upscaling complete.")
            logger.info(f"Direct upscaling of all chunks finished. Time: {format_time(time.time() - upscaling_loop_start_time)}")

        if save_frames and not enable_scene_split and output_frames_dir and processed_frames_permanent_save_path :
            copy_processed_frames_start_time =time .time ()
            
            # Count frames that still need to be copied (skip already saved ones)
            temp_frames_list = os .listdir (output_frames_dir )
            frames_to_copy = []
            frames_already_saved = 0
            
            for frame_file_name in temp_frames_list:
                src_path = os.path.join(output_frames_dir, frame_file_name)
                dst_path = os.path.join(processed_frames_permanent_save_path, frame_file_name)
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    frames_to_copy.append(frame_file_name)
                elif os.path.exists(dst_path):
                    frames_already_saved += 1
            
            num_processed_frames_to_copy = len(frames_to_copy)
            total_frames_found = len(temp_frames_list)
            
            if frames_already_saved > 0:
                copy_proc_msg = f"Copying {num_processed_frames_to_copy} remaining processed frames to permanent storage (skipping {frames_already_saved} already saved): {processed_frames_permanent_save_path}"
            else:
                copy_proc_msg = f"Copying {num_processed_frames_to_copy} processed frames to permanent storage: {processed_frames_permanent_save_path}"
            
            status_log .append (copy_proc_msg )
            logger .info (copy_proc_msg )
            progress (current_overall_progress ,desc ="Copying processed frames...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Copying processed frames.",None
            
            frames_copied_count =0
            for frame_file_name in frames_to_copy:
                shutil .copy2 (os .path .join (output_frames_dir ,frame_file_name ),os .path .join (processed_frames_permanent_save_path ,frame_file_name ))
                frames_copied_count +=1
                if frames_copied_count %100 ==0 or frames_copied_count ==num_processed_frames_to_copy :
                    loop_prog_frac =frames_copied_count /num_processed_frames_to_copy if num_processed_frames_to_copy >0 else 1.0
                    temp_overall_progress =upscaling_loop_progress_start_no_scene_split +(loop_prog_frac *stage_weights ["reassembly_copy_processed"])
                    progress (temp_overall_progress ,desc =f"Copying processed frames: {frames_copied_count}/{num_processed_frames_to_copy}")
            
            if num_processed_frames_to_copy > 0:
                copied_proc_msg =f"Processed frames copied ({frames_copied_count} new, {frames_already_saved} already saved). Time: {format_time(time.time() - copy_processed_frames_start_time)}"
            else:
                copied_proc_msg =f"All {frames_already_saved} processed frames were already saved immediately during chunk processing. Time: {format_time(time.time() - copy_processed_frames_start_time)}"
            
            status_log .append (copied_proc_msg )
            logger .info (copied_proc_msg )
            current_overall_progress =upscaling_loop_progress_start_no_scene_split +stage_weights ["reassembly_copy_processed"]
            progress (current_overall_progress ,desc ="Processed frames copied.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Processed frames copied.",None
        else : # Not saving processed frames or scene split enabled
            if stage_weights["reassembly_copy_processed"] > 0.0:
                 current_overall_progress +=stage_weights ["reassembly_copy_processed"]

        # FACE RESTORATION - After Upscaling
        if enable_face_restoration and face_restoration_when == "after" and not enable_scene_split:
            face_restoration_after_progress_start = current_overall_progress
            progress(current_overall_progress, desc="Starting face restoration after upscaling...")
            
            face_restoration_after_start_time = time.time()
            logger.info("Applying face restoration after upscaling...")
            status_log.append("Applying face restoration after upscaling...")
            yield None, "\n".join(status_log), last_chunk_video_path, "Face restoration after upscaling...", None
            
            # Create face restoration output directory
            face_restored_frames_dir = os.path.join(temp_dir, "face_restored_frames_after")
            
            # Progress callback for face restoration
            def face_restoration_after_progress_callback(progress_val, desc):
                abs_progress = face_restoration_after_progress_start + progress_val * stage_weights["face_restoration_after"]
                progress(abs_progress, desc=desc)
            
            # Apply face restoration to upscaled frames
            face_restoration_result = apply_face_restoration_to_frames(
                input_frames_dir=output_frames_dir,
                output_frames_dir=face_restored_frames_dir,
                fidelity_weight=face_restoration_fidelity,
                enable_colorization=enable_face_colorization,
                model_path=codeformer_model,
                batch_size=face_restoration_batch_size,
                progress_callback=face_restoration_after_progress_callback,
                logger=logger
            )
            
            if face_restoration_result['success']:
                # Update output frames directory to use face-restored frames for final video
                output_frames_dir = face_restored_frames_dir
                face_restore_after_msg = f"Face restoration after upscaling completed: {face_restoration_result['processed_count']} frames processed. Time: {format_time(time.time() - face_restoration_after_start_time)}"
                status_log.append(face_restore_after_msg)
                logger.info(face_restore_after_msg)
                
                # Update processed frames permanent save path if saving frames
                if save_frames and processed_frames_permanent_save_path:
                    # Copy face-restored frames to permanent location
                    face_restored_files = os.listdir(face_restored_frames_dir)
                    for face_restored_file in face_restored_files:
                        src_path = os.path.join(face_restored_frames_dir, face_restored_file)
                        dst_path = os.path.join(processed_frames_permanent_save_path, face_restored_file)
                        shutil.copy2(src_path, dst_path)
                    logger.info(f"Face-restored frames copied to permanent storage: {processed_frames_permanent_save_path}")
            else:
                error_msg = f"Face restoration after upscaling failed: {face_restoration_result['error']}"
                status_log.append(error_msg)
                logger.warning(error_msg)
                # Continue with original upscaled frames if face restoration fails
                
            current_overall_progress = face_restoration_after_progress_start + stage_weights["face_restoration_after"]
            progress(current_overall_progress, desc="Face restoration after upscaling completed")
            yield None, "\n".join(status_log), last_chunk_video_path, "Face restoration after upscaling completed", None
        else:
            # Skip face restoration after stage
            if stage_weights["face_restoration_after"] > 0.0:
                current_overall_progress += stage_weights["face_restoration_after"]

        if not silent_upscaled_video_path: # If not already set (e.g., by scene merge)
            progress (current_overall_progress ,desc ="Creating silent video...")
            silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
            
            # Create video with proper error handling for NVENC limitations
            try:
                # Use duration-preserved video creation to ensure exact timing match with input
                from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
                video_creation_success = create_video_from_frames_with_duration_preservation(
                    output_frames_dir, silent_upscaled_video_path, current_input_video_for_frames,
                    ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger
                )
                
                if not video_creation_success or not os.path.exists(silent_upscaled_video_path):
                    error_msg = "Silent video creation failed, possibly due to resolution constraints. Frames were processed successfully."
                    status_log.append(f"⚠️ {error_msg}")
                    logger.warning(error_msg)
                    
                    # Try to create a basic video without GPU acceleration as fallback
                    logger.info("Attempting fallback video creation without GPU acceleration...")
                    fallback_success = create_video_from_frames_with_duration_preservation(
                        output_frames_dir, silent_upscaled_video_path, current_input_video_for_frames,
                        ffmpeg_preset, ffmpeg_quality_value, False, logger=logger
                    )
                    
                    if fallback_success and os.path.exists(silent_upscaled_video_path):
                        logger.info("Fallback video creation successful")
                        silent_video_msg ="Silent upscaled video created with CPU fallback. Merging audio..."
                        status_log .append (silent_video_msg )
                        logger .info (silent_video_msg )
                        yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg,None
                    else:
                        error_msg = "Both GPU and CPU video creation failed. Frames were processed successfully but video encoding failed."
                        status_log.append(f"❌ {error_msg}")
                        logger.error(error_msg)
                        yield None, "\n".join(status_log), last_chunk_video_path, "Video creation failed - frames processed", None
                        return  # Return gracefully instead of raising error
                else:
                    silent_video_msg ="Silent upscaled video created. Merging audio..."
                    status_log .append (silent_video_msg )
                    logger .info (silent_video_msg )
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg,None
                    
            except Exception as video_e:
                error_msg = f"Silent video creation failed with error: {str(video_e)}. Frames were processed successfully."
                status_log.append(f"❌ {error_msg}")
                logger.error(error_msg, exc_info=True)
                yield None, "\n".join(status_log), last_chunk_video_path, "Video creation failed - frames processed", None
                return  # Return gracefully instead of raising error
                
        else : # Video already merged from scenes (is silent_upscaled_video_path)
            silent_video_msg ="Scene-merged video ready. Merging audio..."
            status_log .append (silent_video_msg )
            logger .info (silent_video_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg,None


        initial_progress_audio_merge =current_overall_progress
        audio_source_video =current_input_video_for_frames
        final_output_path =output_video_path 
        params_for_metadata ["final_output_path"]=final_output_path

        if not os .path .exists (audio_source_video ):
            logger .warning (f"Audio source video '{audio_source_video}' not found. Output will be video-only.")
            if os .path .exists (silent_upscaled_video_path ):
                if final_output_path is not None:
                    shutil .copy2 (silent_upscaled_video_path ,final_output_path )
                else:
                    error_msg = "Final output path not set - cannot complete video processing. Processing was successful but final file could not be moved."
                    logger .error (error_msg)
                    status_log.append(f"❌ {error_msg}")
                    yield None, "\n".join(status_log), last_chunk_video_path, "Processing complete but output path issue", None
                    return  # Return gracefully instead of raising error
            else :
                error_msg = f"Silent upscaled video path {silent_upscaled_video_path} not found for copy. Processing may have failed."
                logger .error (error_msg)
                status_log.append(f"❌ {error_msg}")
                yield None, "\n".join(status_log), last_chunk_video_path, "Silent video not found", None
                return  # Return gracefully instead of raising error
        else :
            if os .path .exists (silent_upscaled_video_path ):
                # Use a temporary file for the merge to avoid FFmpeg "same input/output file" error
                temp_merged_video = os .path .join (temp_dir ,"temp_merged_with_audio.mp4")
                # Remove -shortest flag to preserve full video length, use -avoid_negative_ts for robust merging
                util_run_ffmpeg_command (f'ffmpeg -y -i "{silent_upscaled_video_path}" -i "{audio_source_video}" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? -avoid_negative_ts make_zero "{temp_merged_video}"',"Final Video and Audio Merge",logger =logger )
                
                # Move the temporary merged file to the final output path
                if os .path .exists (temp_merged_video ):
                    if final_output_path is not None:
                        shutil .move (temp_merged_video ,final_output_path )
                        logger .info (f"Moved merged video from temp location to final output: {final_output_path}")
                    else:
                        error_msg = "Final output path not set - cannot move merged video. Processing was successful but final file could not be moved."
                        logger .error (error_msg)
                        status_log.append(f"❌ {error_msg}")
                        yield None, "\n".join(status_log), last_chunk_video_path, "Processing complete but output path issue", None
                        return  # Return gracefully instead of raising error
                else :
                    error_msg = f"Temporary merged video not found: {temp_merged_video}. Audio merge may have failed."
                    logger .error (error_msg)
                    status_log.append(f"❌ {error_msg}")
                    yield None, "\n".join(status_log), last_chunk_video_path, "Audio merge failed", None
                    return  # Return gracefully instead of raising error
            else :
                error_msg = f"Silent upscaled video path {silent_upscaled_video_path} not found for audio merge. Video processing may have failed."
                logger .error (error_msg)
                status_log.append(f"❌ {error_msg}")
                yield None, "\n".join(status_log), last_chunk_video_path, "Silent video not found for audio merge", None
                return  # Return gracefully instead of raising error

        current_overall_progress =initial_progress_audio_merge +stage_weights ["reassembly_audio_merge"]
        progress (current_overall_progress ,desc ="Audio merged.")
        reassembly_done_msg =f"Video reassembly and audio merge finished."
        status_log .append (reassembly_done_msg )
        logger .info (reassembly_done_msg )

        final_save_msg =f"Upscaled video saved to: {final_output_path}"
        status_log .append (final_save_msg )
        logger .info (final_save_msg )
        yield final_output_path ,"\n".join (status_log ),last_chunk_video_path ,"Finalizing...",None

        # RIFE interpolation processing (if enabled)
        rife_output_path = None
        final_return_path = final_output_path  # Default to original output path
        if enable_rife_interpolation and final_output_path and os.path.exists(final_output_path):
            rife_start_time = time.time()
            progress(current_overall_progress, desc="Applying RIFE interpolation...")
            rife_status_msg = f"Applying RIFE {rife_multiplier}x interpolation to final video..."
            status_log.append(rife_status_msg)
            logger.info(rife_status_msg)
            yield final_output_path, "\n".join(status_log), last_chunk_video_path, rife_status_msg, None
            
            try:
                # Import RIFE function and file utilities locally to avoid circular imports
                from .rife_interpolation import increase_fps_single
                from .file_utils import cleanup_rife_temp_files, cleanup_backup_files
                
                # Generate RIFE output path
                final_output_dir = os.path.dirname(final_output_path)
                final_output_name = os.path.splitext(os.path.basename(final_output_path))[0]
                rife_output_path = os.path.join(final_output_dir, f"{final_output_name}_{rife_multiplier}x_FPS.mp4")
                
                # Pre-processing cleanup and validation
                cleanup_success, cleanup_msg = cleanup_rife_temp_files(final_output_dir, logger)
                if logger:
                    logger.info(f"Pre-RIFE cleanup: {cleanup_msg}")
                
                # Apply RIFE interpolation
                rife_result, rife_message = increase_fps_single(
                    video_path=final_output_path,
                    output_path=rife_output_path,
                    multiplier=rife_multiplier,
                    fp16=rife_fp16,
                    uhd=rife_uhd,
                    scale=rife_scale,
                    skip_static=rife_skip_static,
                    enable_fps_limit=rife_enable_fps_limit,
                    max_fps_limit=rife_max_fps_limit,
                    ffmpeg_preset=ffmpeg_preset,
                    ffmpeg_quality_value=ffmpeg_quality_value,
                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                    overwrite_original=rife_overwrite_original,
                    keep_original=rife_keep_original,
                    output_dir=final_output_dir,
                    seed=current_seed,
                    logger=logger,
                    progress=None  # Don't pass progress to avoid conflicts
                )
                
                if rife_result:
                    rife_processing_time = time.time() - rife_start_time
                    rife_success_msg = f"RIFE interpolation completed: {rife_message}. Time: {format_time(rife_processing_time)}"
                    status_log.append(rife_success_msg)
                    logger.info(rife_success_msg)
                    
                    # Update metadata with RIFE information
                    params_for_metadata["rife_processing_time"] = rife_processing_time
                    params_for_metadata["rife_output_path"] = rife_result
                    
                    # Following user rules: When RIFE enabled, return RIFE versions to gradio interface
                    rife_output_path = rife_result
                    final_return_path = rife_result  # Update final return path to RIFE version
                    
                    # If overwrite original is enabled, also update the file system
                    if rife_overwrite_original:
                        final_output_path = rife_result
                        params_for_metadata["final_output_path"] = final_output_path
                        yield final_return_path, "\n".join(status_log), last_chunk_video_path, "RIFE interpolation complete (original overwritten).", None
                    else:
                        # Keep original file but return RIFE version to interface (following user rules)
                        params_for_metadata["rife_output_path"] = rife_result
                        yield final_return_path, "\n".join(status_log), last_chunk_video_path, "RIFE interpolation complete.", None
                else:
                    rife_error_msg = f"RIFE interpolation failed: {rife_message}. Time: {format_time(time.time() - rife_start_time)}"
                    status_log.append(rife_error_msg)
                    logger.warning(rife_error_msg)
                    yield final_return_path, "\n".join(status_log), last_chunk_video_path, "RIFE interpolation failed.", None
                    
            except Exception as e_rife:
                rife_error_msg = f"Error during RIFE interpolation: {e_rife}. Time: {format_time(time.time() - rife_start_time)}"
                status_log.append(rife_error_msg)
                logger.error(rife_error_msg, exc_info=True)
                
                # Cleanup any partial RIFE files on error
                try:
                    cleanup_success, cleanup_msg = cleanup_rife_temp_files(final_output_dir, logger)
                    if logger:
                        logger.info(f"Post-error RIFE cleanup: {cleanup_msg}")
                except Exception as cleanup_error:
                    if logger:
                        logger.warning(f"Could not cleanup RIFE temp files after error: {cleanup_error}")
                
                yield final_return_path, "\n".join(status_log), last_chunk_video_path, "RIFE interpolation error.", None

        if create_comparison_video_enabled and final_output_path and os.path.exists(final_output_path):
            comparison_video_progress_start = current_overall_progress
            comparison_video_start_time = time.time()
            original_video_for_comparison = params_for_metadata.get("input_video_path", input_video_path)
            comparison_output_path = get_comparison_output_path(final_output_path)
            
            progress(current_overall_progress, desc="Creating comparison video...")
            comparison_status_msg = "Creating comparison video..."
            status_log.append(comparison_status_msg)
            yield final_return_path, "\n".join(status_log), last_chunk_video_path, comparison_status_msg,None
            
            try:
                # Determine which version to use as the upscaled video in comparison
                # Use RIFE-interpolated version as primary reference when available (following user rules)
                upscaled_video_for_comparison = rife_output_path if rife_output_path and os.path.exists(rife_output_path) else final_output_path
                
                comparison_success = create_comparison_video(
                    original_video_path=original_video_for_comparison,  # Original input video
                    upscaled_video_path=upscaled_video_for_comparison,  # Final upscaled (potentially RIFE) video
                    output_path=comparison_output_path,
                    ffmpeg_preset=ffmpeg_preset,
                    ffmpeg_quality=ffmpeg_quality_value,
                    ffmpeg_use_gpu=ffmpeg_use_gpu,
                    logger=logger
                )
                if comparison_success:
                    comparison_done_msg = f"Comparison video created: {comparison_output_path}. Time: {format_time(time.time() - comparison_video_start_time)}"
                    status_log.append(comparison_done_msg)
                    logger.info(comparison_done_msg)
                    yield final_return_path, "\n".join(status_log), last_chunk_video_path, "Comparison video complete.", comparison_output_path
                else:
                    comparison_error_msg = f"Comparison video creation failed. Time: {format_time(time.time() - comparison_video_start_time)}"
                    status_log.append(comparison_error_msg)
                    logger.warning(comparison_error_msg)
                    yield final_return_path, "\n".join(status_log), last_chunk_video_path, "Comparison video failed.", None
            except Exception as e_comparison:
                comparison_error_msg = f"Error creating comparison video: {e_comparison}. Time: {format_time(time.time() - comparison_video_start_time)}"
                status_log.append(comparison_error_msg)
                logger.error(comparison_error_msg, exc_info=True)
                yield final_return_path, "\n".join(status_log), last_chunk_video_path, "Comparison video error.", None
            
            current_overall_progress = comparison_video_progress_start + stage_weights.get("comparison_video", 0.0) # Use actual weight
            progress(current_overall_progress, desc="Comparison video processing complete.")
        else: # Comparison video disabled or final_output_path not ready
            if stage_weights["comparison_video"] > 0.0:
                 current_overall_progress +=stage_weights ["comparison_video"]


        if save_metadata :
            initial_progress_metadata =current_overall_progress
            progress (current_overall_progress ,desc ="Saving metadata...")
            metadata_save_start_time =time .time ()
            processing_time_total =time .time ()-overall_process_start_time
            final_status_info ={"processing_time_total":processing_time_total}
            success ,message =metadata_handler_module .save_metadata (
                save_flag =True , output_dir =main_output_dir , base_filename_no_ext =base_output_filename_no_ext ,
                params_dict =params_for_metadata , status_info =final_status_info , logger =logger )
            if success :
                meta_saved_msg =f"Final metadata saved: {message.split(': ')[-1]}. Time to save: {format_time(time.time() - metadata_save_start_time)}"
                status_log .append (meta_saved_msg )
                logger .info (meta_saved_msg )
            else :
                status_log .append (f"Error saving final metadata: {message}")
                logger .error (message )
            current_overall_progress =initial_progress_metadata +stage_weights ["metadata"]
            progress (current_overall_progress ,desc ="Metadata saved.")
            yield final_return_path ,"\n".join (status_log ),last_chunk_video_path ,meta_saved_msg if success else message,None
        else: # Metadata saving disabled
            if stage_weights["metadata"] > 0.0:
                current_overall_progress += stage_weights["metadata"]


        # Ensure progress reaches 1.0 if all stages completed without exact weight summation
        current_overall_progress = max(current_overall_progress, 0.99) # Leave a tiny bit for final desc
        
        is_error =any (err_msg in (status_log [-1] if status_log else "") for err_msg in ["Error:","Critical Error:"])
        final_desc ="Finished!"
        if is_error :
            final_desc =status_log [-1 ]if status_log else "Error occurred"
            progress (current_overall_progress ,desc =final_desc )
        else :
            progress (1.0 ,desc =final_desc )

        yield final_return_path ,"\n".join (status_log ),last_chunk_video_path ,"Processing complete!",None

    except CancelledError as e_cancel:
        logger.info("Upscaling process cancelled by user")
        
        # Set flag to prevent further yields from overwriting partial video
        processing_was_cancelled = True
        
        # Try to compose partial video from processed content
        partial_video_path = None
        partial_success_msg = "Process cancelled by user"
        
        try:
            # Check if we're in scene-split mode and have processed scene videos or a merged scene video available
            if 'enable_scene_split' in locals() and enable_scene_split:
                # First check if we have a merged scene video (scene merging completed)
                if 'silent_upscaled_video_path' in locals() and silent_upscaled_video_path and os.path.exists(silent_upscaled_video_path):
                    logger.info("🔄 Using merged scenes video for partial output after cancellation...")
                    status_log.append("🔄 Creating partial video from processed scenes...")
                    progress(current_overall_progress, desc="Creating partial video...")
                    
                    # Create partial video filename
                    partial_video_name = f"{base_output_filename_no_ext}_partial_cancelled.mp4"
                    partial_video_path = os.path.join(main_output_dir, partial_video_name)
                    
                    # Use the already-merged scene video as the base
                    silent_partial_path = silent_upscaled_video_path
                    video_creation_success = True  # Video already exists from scene merging
                    
                    # Get video properties for audio trimming
                    try:
                        import subprocess
                        # Get duration of the partial video
                        duration_cmd = f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{silent_partial_path}"'
                        result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True, check=True)
                        duration_str = result.stdout.strip()
                        if duration_str and duration_str != 'N/A':
                            partial_duration_seconds = float(duration_str)
                            logger.info(f"Partial video duration from merged scenes: {partial_duration_seconds:.2f} seconds")
                        else:
                            logger.warning(f"Could not parse duration from ffprobe output: '{duration_str}'")
                            partial_duration_seconds = None
                    except Exception as e_duration:
                        logger.warning(f"Could not determine partial video duration: {e_duration}")
                        partial_duration_seconds = None
                        
                # If no merged video, check if we have individual processed scene videos to merge
                elif 'processed_scene_videos' in locals() and processed_scene_videos:
                    logger.info(f"🔄 Merging {len(processed_scene_videos)} processed scene videos for partial output after cancellation...")
                    status_log.append(f"🔄 Merging {len(processed_scene_videos)} processed scene videos for partial output...")
                    progress(current_overall_progress, desc="Merging partial scenes...")
                    
                    # Create partial video filename
                    partial_video_name = f"{base_output_filename_no_ext}_partial_cancelled.mp4"
                    partial_video_path = os.path.join(main_output_dir, partial_video_name)
                    
                    # Create silent partial video by merging processed scenes
                    silent_partial_path = os.path.join(temp_dir, "silent_partial_from_scenes.mp4")
                    
                    try:
                        # Use already imported scene merging utility
                        # (util_merge_scene_videos is imported at top of file)
                        
                        # Merge the processed scene videos
                        util_merge_scene_videos(
                            processed_scene_videos, silent_partial_path, temp_dir,
                            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu,
                            logger=logger, allow_partial_merge=True
                        )
                        
                        if os.path.exists(silent_partial_path):
                            video_creation_success = True
                            logger.info(f"Successfully merged {len(processed_scene_videos)} scene videos for partial output")
                            
                            # Get video properties for audio trimming
                            try:
                                import subprocess
                                duration_cmd = f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{silent_partial_path}"'
                                result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True, check=True)
                                duration_str = result.stdout.strip()
                                if duration_str and duration_str != 'N/A':
                                    partial_duration_seconds = float(duration_str)
                                    logger.info(f"Partial video duration from scene merging: {partial_duration_seconds:.2f} seconds")
                                else:
                                    logger.warning(f"Could not parse duration from ffprobe output: '{duration_str}'")
                                    partial_duration_seconds = None
                            except Exception as e_duration:
                                logger.warning(f"Could not determine partial video duration: {e_duration}")
                                partial_duration_seconds = None
                        else:
                            logger.warning("Failed to merge scene videos for partial output")
                            video_creation_success = False
                            
                    except Exception as e_merge:
                        logger.error(f"Error merging scene videos for partial output: {e_merge}")
                        video_creation_success = False
                else:
                    logger.info("Scene split mode enabled but no processed scene videos found for partial output")
                    video_creation_success = False
                    
            # Fallback: Check if we have any processed frames to work with (non-scene-split mode)
            elif 'output_frames_dir' in locals() and output_frames_dir and os.path.exists(output_frames_dir):
                frame_files = sorted([f for f in os.listdir(output_frames_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                min_frames_for_video = 5  # Minimum frames required to create a meaningful video
                
                if len(frame_files) >= min_frames_for_video:
                    num_processed_frames = len(frame_files)
                    logger.info(f"🔄 Composing partial video from {num_processed_frames} processed frames after cancellation...")
                    status_log.append(f"🔄 Composing partial video from {num_processed_frames} processed frames...")
                    progress(current_overall_progress, desc="Composing partial video...")
                elif frame_files:
                    # Some frames processed but below threshold
                    num_processed_frames = len(frame_files)
                    logger.info(f"Only {num_processed_frames} frames processed (minimum {min_frames_for_video} required for video creation)")
                    partial_success_msg = f"⚠️ Process cancelled. Only {num_processed_frames} frames processed (minimum {min_frames_for_video} required for video creation)."
                    status_log.append(partial_success_msg)
                    yield None, "\n".join(status_log), last_chunk_video_path, partial_success_msg, None
                    return
                    
                    # Create partial video filename
                    partial_video_name = f"{base_output_filename_no_ext}_partial_cancelled.mp4"
                    partial_video_path = os.path.join(main_output_dir, partial_video_name)
                    
                    # Create video from processed frames (without audio first)
                    silent_partial_path = os.path.join(temp_dir, "silent_partial_video.mp4")
                    
                    logger.info(f"Creating silent partial video from {num_processed_frames} frames...")
                    # Use duration-preserved video creation for partial video
                    from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
                    video_creation_success = create_video_from_frames_with_duration_preservation(
                        output_frames_dir, silent_partial_path, current_input_video_for_frames,
                        ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger
                    )
                    
                    if video_creation_success:
                        # Calculate duration of partial video from frames
                        fps_to_use = input_fps_val if 'input_fps_val' in locals() else 30.0
                        partial_duration_seconds = num_processed_frames / fps_to_use
                        logger.info(f"Partial video duration: {partial_duration_seconds:.2f} seconds ({num_processed_frames} frames at {fps_to_use} FPS)")
                else:
                    video_creation_success = False
            else:
                logger.info("No processed content found for partial video creation")
                video_creation_success = False
            
            # If we have a valid silent video, add audio
            if video_creation_success and 'silent_partial_path' in locals() and os.path.exists(silent_partial_path):
                # Extract and trim audio from original video to match partial video length
                audio_added = False
                if 'input_video_path' in locals() and os.path.exists(input_video_path):
                    try:
                        logger.info("Extracting and trimming audio to match partial video length...")
                        
                        # Import the better audio detection function
                        from .face_restoration_utils import _check_video_has_audio
                        
                        # Check if original video has audio using proper method
                        has_audio = _check_video_has_audio(input_video_path, logger)
                        
                        if has_audio and 'partial_duration_seconds' in locals() and partial_duration_seconds and partial_duration_seconds > 0:
                            # Create final video with trimmed audio
                            trim_audio_cmd = (
                                f'ffmpeg -y -i "{silent_partial_path}" -i "{input_video_path}" '
                                f'-t {partial_duration_seconds:.3f} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 '
                                f'-avoid_negative_ts make_zero "{partial_video_path}"'
                            )
                            
                            audio_success = util_run_ffmpeg_command(
                                trim_audio_cmd, "Partial Video Audio Addition", logger, raise_on_error=False
                            )
                            
                            if audio_success and os.path.exists(partial_video_path):
                                audio_added = True
                                logger.info(f"Successfully added trimmed audio ({partial_duration_seconds:.3f}s) to partial video")
                            else:
                                logger.warning("Failed to add audio, using silent video")
                        else:
                            if not has_audio:
                                logger.info("Original video has no audio stream")
                            else:
                                logger.info("Partial video duration not available or invalid")
                            
                    except Exception as e_audio:
                        logger.warning(f"Error processing audio for partial video: {e_audio}")
                
                # If audio addition failed or wasn't needed, copy the silent video
                if not audio_added:
                    try:
                        shutil.copy2(silent_partial_path, partial_video_path)
                        logger.info("Using silent partial video (no audio)")
                    except Exception as e_copy:
                        logger.error(f"Failed to copy silent partial video: {e_copy}")
                        partial_video_path = None
                
                # Verify final partial video exists
                if partial_video_path and os.path.exists(partial_video_path):
                    # Update metadata for partial video
                    if 'save_metadata' in locals() and save_metadata and 'params_for_metadata' in locals() and 'metadata_handler_module' in locals() and metadata_handler_module:
                        try:
                            partial_metadata = params_for_metadata.copy()
                            partial_metadata["final_output_path"] = partial_video_path
                            partial_metadata["partial_processing"] = True
                            partial_metadata["cancellation_reason"] = "User requested cancellation"
                            if 'partial_duration_seconds' in locals() and partial_duration_seconds:
                                partial_metadata["partial_duration_seconds"] = partial_duration_seconds
                            
                            # Save metadata for partial video
                            partial_base_name = os.path.splitext(partial_video_name)[0]
                            success, message = metadata_handler_module.save_metadata(
                                save_flag=True, output_dir=main_output_dir, 
                                base_filename_no_ext=partial_base_name,
                                params_dict=partial_metadata, status_info={}, logger=logger
                            )
                            if success:
                                logger.info(f"Partial video metadata saved: {message}")
                            else:
                                logger.warning(f"Failed to save partial video metadata: {message}")
                        except Exception as e_meta:
                            logger.warning(f"Error saving partial video metadata: {e_meta}")
                    
                    duration_str = f" ({partial_duration_seconds:.1f}s)" if 'partial_duration_seconds' in locals() and partial_duration_seconds else ""
                    partial_success_msg = f"⚠️ Partial video created from processed scenes{duration_str}: {partial_video_path}"
                    logger.info(f"Cancellation handled - partial video saved: {partial_video_path}")
                else:
                    logger.warning("Failed to create partial video file")
                    partial_success_msg = "⚠️ Process cancelled by user - failed to create partial video"
            else:
                logger.info("No processed content found for partial video creation")
                partial_success_msg = "⚠️ Process cancelled by user - no processed content available"
        
        except Exception as e_partial:
            logger.error(f"Error during partial video creation: {e_partial}", exc_info=True)
            partial_success_msg = "⚠️ Process cancelled by user - error creating partial video"
        
        # Add cancellation message to status log
        status_log.append(partial_success_msg)
        current_overall_progress = min(current_overall_progress + 0.01, 1.0)
        progress(current_overall_progress, desc="Process cancelled")
        
        # Yield the partial video if available, otherwise None
        yield partial_video_path, "\n".join(status_log), last_chunk_video_path, partial_success_msg, None
        
        # Return early to prevent further yields that could overwrite the partial video
        return
    except gr .Error as e :
        logger .error (f"A Gradio UI Error occurred: {e}",exc_info =True )
        status_log .append (f"Error: {e}")
        current_overall_progress =min (current_overall_progress +0.01 ,1.0 ) # Small increment for error
        progress (current_overall_progress ,desc =f"Error: {str(e)[:50]}")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Error: {e}",None
    except Exception as e :
        logger .error (f"An unexpected error occurred during upscaling: {e}",exc_info =True )
        status_log .append (f"Critical Error: {e}")
        current_overall_progress =min (current_overall_progress +0.01 ,1.0 ) # Small increment for error
        progress (current_overall_progress ,desc =f"Critical Error: {str(e)[:50]}")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Critical Error: {e}",None
        # Re-raise to stop Gradio execution gracefully, or let the finally block handle cleanup
        # For Gradio, it's often better to let it complete the yield cycle.
    finally :
        # Global cleanup for temp_dir
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            try:
                util_cleanup_temp_dir(temp_dir, logger=logger)
            except Exception as e_clean:
                logger.error(f"Error cleaning up temp_dir {temp_dir}: {e_clean}", exc_info=True)
        
        # Cleanup old backup files and temp RIFE files
        try:
            from .file_utils import cleanup_backup_files, cleanup_rife_temp_files
            
            # Clean up old backup files (older than 24 hours)
            if main_output_dir and os.path.exists(main_output_dir):
                backup_cleanup_success, backup_cleanup_msg = cleanup_backup_files(
                    main_output_dir, backup_pattern=".backup", max_age_hours=24, logger=logger
                )
                if logger:
                    logger.info(f"Backup cleanup: {backup_cleanup_msg}")
                
                # Clean up any remaining RIFE temp files
                rife_cleanup_success, rife_cleanup_msg = cleanup_rife_temp_files(main_output_dir, logger)
                if logger:
                    logger.info(f"Final RIFE cleanup: {rife_cleanup_msg}")
                    
        except Exception as e_cleanup:
            if logger:
                logger.warning(f"Error during final cleanup: {e_cleanup}")

    total_processing_time = time.time() - overall_process_start_time
    params_for_metadata["total_processing_time"] = total_processing_time
    logger.info(f"STAR upscaling process finished and cleaned up. Total processing time: {format_time(total_processing_time)}")

    # Check if processing was cancelled - if so, skip final yields to preserve partial video
    processing_was_cancelled = 'processing_was_cancelled' in locals() and processing_was_cancelled
    if processing_was_cancelled:
        logger.info("Skipping final yields because processing was cancelled - preserving partial video result")
        return

    if save_metadata and main_output_dir and base_output_filename_no_ext:
        status_info_for_final_meta = {"processing_time_total": total_processing_time}
        success, message = metadata_handler_module.save_metadata(
            save_flag=True, output_dir=main_output_dir, base_filename_no_ext=base_output_filename_no_ext,
            params_dict=params_for_metadata, status_info=status_info_for_final_meta, logger=logger
        )
        if success:
            # Initialize metadata_save_start_time if not already set
            if 'metadata_save_start_time' not in locals():
                metadata_save_start_time = time.time()
            final_meta_msg = f"Final metadata saved: {message.split(': ')[-1]}. Time to save: {format_time(time.time() - metadata_save_start_time)}"
            status_log.append(final_meta_msg)
            logger.info(final_meta_msg)
        else:
            status_log.append(f"Error saving final metadata: {message}")
            logger.error(message)
        # Handle metadata progress increment safely
        if 'initial_progress_metadata' in locals():
            current_overall_progress = initial_progress_metadata + stage_weights["metadata"]
        else:
            current_overall_progress += stage_weights["metadata"]
        progress(current_overall_progress, desc="Metadata saved.")
        # Use final_output_path if available, otherwise use output_video_path as fallback
        final_path_for_yield = final_output_path if 'final_output_path' in locals() else output_video_path
        yield final_path_for_yield, "\n".join(status_log), last_chunk_video_path, final_meta_msg if success else message, None
    else: # Metadata saving disabled
        if stage_weights["metadata"] > 0.0:
            current_overall_progress += stage_weights["metadata"]


    # Ensure progress reaches 1.0 if all stages completed without exact weight summation
    current_overall_progress = max(current_overall_progress, 0.99) # Leave a tiny bit for final desc
    
    is_error =any (err_msg in (status_log [-1] if status_log else "") for err_msg in ["Error:","Critical Error:"])
    final_desc ="Finished!"
    if is_error :
        final_desc =status_log [-1 ]if status_log else "Error occurred"
        progress (current_overall_progress ,desc =final_desc )
    else :
        progress (1.0 ,desc =final_desc )

    # Use final_output_path if available, otherwise use output_video_path as fallback
    final_path_for_final_yield = final_output_path if 'final_output_path' in locals() else output_video_path
    yield final_path_for_final_yield ,"\n".join (status_log ),last_chunk_video_path ,"Processing complete!",None
