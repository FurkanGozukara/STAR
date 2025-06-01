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
from .cogvlm_utils import auto_caption as util_auto_caption
from .common_utils import format_time
from .ffmpeg_utils import (
    run_ffmpeg_command as util_run_ffmpeg_command,
    extract_frames as util_extract_frames,
    create_video_from_frames as util_create_video_from_frames,
    decrease_fps as util_decrease_fps,
    get_common_fps_multipliers as util_get_common_fps_multipliers
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
from .nvenc_utils import is_resolution_too_small_for_nvenc
from .scene_processing_core import process_single_scene
from .comparison_video import create_comparison_video, get_comparison_output_path
from .rife_interpolation import apply_rife_to_chunks, apply_rife_to_scenes


def run_upscale (
    input_video_path ,user_prompt ,positive_prompt ,negative_prompt ,model_choice ,
    upscale_factor_slider ,cfg_scale ,steps ,solver_mode ,
    max_chunk_len ,vae_chunk ,color_fix_method ,
    enable_tiling ,tile_size ,tile_overlap ,
    enable_sliding_window ,window_size ,window_step ,
    enable_target_res ,target_h ,target_w ,target_res_mode ,
    ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,
    save_frames ,save_metadata ,save_chunks ,

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
    if not input_video_path or not os .path .exists (input_video_path ):
        raise gr .Error ("Please select a valid input video file.")

    last_chunk_video_path =None
    last_chunk_status ="No chunks processed yet"

    logger.info(f"Using seed for this run: {current_seed}")
    setup_seed_func (current_seed) # Use the passed seed
    overall_process_start_time =time .time ()
    logger .info ("Overall upscaling process started.")

    current_overall_progress =0.0

    params_for_metadata ={
    "input_video_path":input_video_path ,"user_prompt":user_prompt ,"positive_prompt":positive_prompt ,
    "negative_prompt":negative_prompt ,"model_choice":model_choice ,
    "upscale_factor_slider":upscale_factor_slider ,"cfg_scale":cfg_scale ,
    "ui_total_diffusion_steps":steps ,"solver_mode":solver_mode ,
    "max_chunk_len":max_chunk_len ,"vae_chunk":vae_chunk ,"color_fix_method":color_fix_method ,
    "enable_tiling":enable_tiling ,"tile_size":tile_size ,"tile_overlap":tile_overlap ,
    "enable_sliding_window":enable_sliding_window ,"window_size":window_size ,"window_step":window_step ,
    "enable_target_res":enable_target_res ,"target_h":target_h ,"target_w":target_w ,
    "target_res_mode":target_res_mode ,"ffmpeg_preset":ffmpeg_preset ,
    "ffmpeg_quality_value":ffmpeg_quality_value ,"ffmpeg_use_gpu":ffmpeg_use_gpu ,
    "enable_scene_split":enable_scene_split ,"scene_split_mode":scene_split_mode ,
    "scene_min_scene_len":scene_min_scene_len ,"scene_threshold":scene_threshold ,
    "scene_manual_split_type":scene_manual_split_type ,"scene_manual_split_value":scene_manual_split_value ,
    "is_batch_mode":is_batch_mode ,"batch_output_dir":batch_output_dir ,
    "current_seed": current_seed, # Added current_seed
    
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

    "final_output_path":None ,"orig_w":None ,"orig_h":None ,
    "input_fps":None ,"upscale_factor":None ,"final_w":None ,"final_h":None ,
    }

    actual_cogvlm_quant_val = cogvlm_quant

    stage_weights ={
    "init_paths_res":0.03 ,
    "scene_split":0.05 if enable_scene_split else 0.0 ,
    "downscale":0.07 , # This will be set to 0 if no downscale needed later
    "model_load":0.05 ,
    "extract_frames":0.10 ,
    "copy_input_frames":0.05 if save_frames and not enable_scene_split else 0.0, # Adjusted
    "upscaling_loop":0.50 if enable_scene_split else 0.60 ,
    "scene_merge":0.05 if enable_scene_split else 0.0 ,
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

        input_frames_permanent_save_path =os .path .join (frames_output_subfolder ,"input_frames")
        processed_frames_permanent_save_path =os .path .join (frames_output_subfolder ,"processed_frames")
        os .makedirs (input_frames_permanent_save_path ,exist_ok =True )
        os .makedirs (processed_frames_permanent_save_path ,exist_ok =True )
        logger .info (f"Saving frames to: {frames_output_subfolder}")

    if save_chunks :
        if is_batch_mode :
            chunks_permanent_save_path =os .path .join (main_output_dir ,"chunks")
        else :
            chunks_permanent_save_path =os .path .join (main_output_dir ,base_output_filename_no_ext ,"chunks")
        os .makedirs (chunks_permanent_save_path ,exist_ok =True )
        logger .info (f"Saving chunks to: {chunks_permanent_save_path}")

    os .makedirs (temp_dir ,exist_ok =True )
    os .makedirs (input_frames_dir ,exist_ok =True )
    os .makedirs (output_frames_dir ,exist_ok =True )

    # star_model =None # Model is no longer loaded here globally
    current_input_video_for_frames =input_video_path
    downscaled_temp_video =None
    status_log =["Process started..."]

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

        upscale_factor_val =None
        final_h_val ,final_w_val =None ,None
        needs_downscale = False # Initialize

        if enable_target_res :
            needs_downscale ,ds_h ,ds_w ,upscale_factor_calc ,final_h_calc ,final_w_calc =util_calculate_upscale_params (
            orig_h_val ,orig_w_val ,target_h ,target_w ,target_res_mode ,logger =logger
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

                ffmpeg_opts_downscale =""
                use_cpu_fallback =ffmpeg_use_gpu and is_resolution_too_small_for_nvenc (ds_w ,ds_h ,logger )

                if ffmpeg_use_gpu and not use_cpu_fallback :
                    nvenc_preset_down =ffmpeg_preset
                    if ffmpeg_preset in ["ultrafast","superfast","veryfast","faster","fast"]:nvenc_preset_down ="fast"
                    elif ffmpeg_preset in ["slower","veryslow"]:nvenc_preset_down ="slow"
                    ffmpeg_opts_downscale =f'-c:v h264_nvenc -preset:v {nvenc_preset_down} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
                else :
                    if use_cpu_fallback :
                        logger .info (f"Falling back to CPU encoding for downscaling due to small target resolution: {ds_w}x{ds_h}")
                    ffmpeg_opts_downscale =f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'

                cmd =f'ffmpeg -y -i "{input_video_path}" -vf "{scale_filter}" {ffmpeg_opts_downscale} -c:a copy "{downscaled_temp_video}"'
                util_run_ffmpeg_command (cmd ,"Input Downscaling with Audio Copy",logger =logger )
                current_input_video_for_frames =downscaled_temp_video

                orig_h_val ,orig_w_val =util_get_video_resolution (downscaled_temp_video, logger=logger )
                params_for_metadata ["orig_h"]=orig_h_val
                params_for_metadata ["orig_w"]=orig_w_val

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
            
            upscale_factor_val =upscale_factor_slider
            final_h_val =int (round (orig_h_val *upscale_factor_val /2 )*2 )
            final_w_val =int (round (orig_w_val *upscale_factor_val /2 )*2 )

            params_for_metadata ["upscale_factor"]=upscale_factor_val
            params_for_metadata ["final_h"]=final_h_val
            params_for_metadata ["final_w"]=final_w_val

            direct_upscale_msg =f"Direct upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}"
            status_log .append (direct_upscale_msg )
            logger .info (direct_upscale_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status,None

        # FPS Decrease Processing (if enabled)
        fps_decreased_video_path = None
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
                
                # Apply FPS decrease with new parameters
                fps_success, fps_output_fps, fps_message = util_decrease_fps(
                    input_video_path=current_input_video_for_frames,
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
                    
                    fps_duration = time.time() - fps_decrease_start_time
                    fps_success_msg = f"FPS decrease completed in {format_time(fps_duration)}. {fps_message}"
                    status_log.append(fps_success_msg)
                    logger.info(fps_success_msg)
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
                    yield None, "\n".join(status_log), last_chunk_video_path, "FPS decrease failed, using original", None
                    
            except Exception as e_fps:
                fps_duration = time.time() - fps_decrease_start_time
                fps_exception_msg = f"Exception during FPS decrease after {format_time(fps_duration)}: {str(e_fps)}. Continuing with original video."
                status_log.append(fps_exception_msg)
                logger.error(fps_exception_msg, exc_info=True)
                params_for_metadata["fps_decrease_applied"] = False
                params_for_metadata["fps_decrease_error"] = str(e_fps)
                yield None, "\n".join(status_log), last_chunk_video_path, "FPS decrease error, using original", None
        else:
            # FPS decrease disabled
            params_for_metadata["fps_decrease_applied"] = False

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

        upscaling_loop_progress_start =current_overall_progress
        progress (current_overall_progress ,desc ="Preparing for upscaling...")
        total_noise_levels =900
        upscaling_loop_start_time =time .time ()
        gpu_device =util_get_gpu_device (logger =logger )
        scene_metadata_base_params =params_for_metadata .copy ()if enable_scene_split else None
        silent_upscaled_video_path = None # Initialize

        if enable_scene_split and scene_video_paths :
            processed_scene_videos =[]
            total_scenes =len (scene_video_paths )
            first_scene_caption =None

            if enable_auto_caption_per_scene and total_scenes >0 :
                logger .info ("Auto-captioning first scene to update main prompt before processing...")
                progress (current_overall_progress ,desc ="Generating caption for first scene...")
                try :
                    first_scene_caption_result ,_ =util_auto_caption (
                    scene_video_paths [0 ],actual_cogvlm_quant_val ,cogvlm_unload ,
                    app_config_module .COG_VLM_MODEL_PATH ,logger =logger ,progress =progress
                    )
                    if not first_scene_caption_result .startswith ("Error:"):
                        first_scene_caption =first_scene_caption_result
                        logger .info (f"First scene caption generated for main prompt: '{first_scene_caption[:100]}...'")
                        caption_update_msg =f"First scene caption generated [FIRST_SCENE_CAPTION:{first_scene_caption}]"
                        status_log .append (caption_update_msg )
                        logger .info (f"Yielding first scene caption for immediate prompt update")
                        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status,None
                    else :
                        logger .warning ("First scene auto-captioning failed, using original prompt")
                except Exception as e :
                    logger .error (f"Error auto-captioning first scene: {e}", exc_info=True)

            for scene_idx ,scene_video_path_item in enumerate (scene_video_paths ):
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
                        solver_mode=solver_mode, cfg_scale=cfg_scale, max_chunk_len=max_chunk_len, vae_chunk=vae_chunk, color_fix_method=color_fix_method,
                        enable_tiling=enable_tiling, tile_size=tile_size, tile_overlap=tile_overlap, enable_sliding_window=enable_sliding_window, window_size=window_size, window_step=window_step,
                        save_frames=save_frames, scene_output_dir=frames_output_subfolder, progress_callback=scene_upscale_progress_callback,
                        enable_auto_caption_per_scene=scene_enable_auto_caption_current,
                        cogvlm_quant=actual_cogvlm_quant_val,
                        cogvlm_unload=cogvlm_unload,
                        progress=progress,
                        save_chunks=save_chunks,
                        chunks_permanent_save_path=frames_output_subfolder, # This was scene_output_dir, should be specific path for chunks
                        ffmpeg_preset=ffmpeg_preset, ffmpeg_quality_value=ffmpeg_quality_value, ffmpeg_use_gpu=ffmpeg_use_gpu,
                        save_metadata=save_metadata, metadata_params_base=scene_metadata_base_params,
                        
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
                            if not is_batch_mode: # Only raise error if not in batch, to allow batch to continue
                                raise gr.Error(f"Scene {scene_idx + 1} processing failed: {error_message}")
                            else:
                                logger.error(f"BATCH MODE: Scene {scene_idx + 1} processing failed: {error_message}. Skipping this scene for merging.")
                                # Continue to next scene in batch context
                    if processed_scene_video_path_final :
                         processed_scene_videos .append (processed_scene_video_path_final )
                    else :
                        logger .error (f"Scene {scene_idx+1} finished processing but no final video path was yielded by process_single_scene.")
                        raise gr .Error (f"Scene {scene_idx+1} did not complete correctly.")
                except Exception as e :
                    logger .error (f"Error processing scene {scene_idx + 1} in run_upscale: {e}",exc_info =True )
                    status_log .append (f"Error processing scene {scene_idx + 1}: {e}")
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene {scene_idx + 1} processing failed: {e}",None
                    if not is_batch_mode: # Only raise error if not in batch, to allow batch to continue
                        raise gr.Error(f"Scene {scene_idx + 1} processing failed: {e}")
                    else:
                        logger.error(f"BATCH MODE: Scene {scene_idx + 1} processing failed: {e}. Skipping this scene for merging.")
                        # Continue to next scene in batch context

            current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
            scene_merge_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Merging processed scenes...")
            status_log .append ("Merging processed scenes...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Merging Scenes...",None

            silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
            util_merge_scene_videos (processed_scene_videos ,silent_upscaled_video_path ,temp_dir ,
            ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )

            current_overall_progress =scene_merge_progress_start +stage_weights ["scene_merge"]
            progress (current_overall_progress ,desc ="Scene merging complete")
            scene_merge_msg =f"Successfully merged {len(processed_scene_videos)} processed scenes"
            status_log .append (scene_merge_msg )
            logger .info (scene_merge_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_merge_msg,None
        else : # No scene splitting (direct upscale)
            # This is the start of the block that was mis-indented.
            # It should be an 'else' to the 'if enable_scene_split:'
            # However, the logic flow is: if scene_split, do the loop. 
            # If NOT scene_split, do the direct upscaling. So it's 'if enable_scene_split:' then 'else:'.
            # The 'and scene_video_paths' is for AFTER splitting.

            # The mis-indented 'else' started around here in the previous diff. 
            # The following is the NON-SCENE-SPLIT path.
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
            star_model_ns = VideoToVideo_sr_class(model_cfg_ns, device=model_device_ns)
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
            
            num_chunks_direct = math.ceil(frame_count / max_chunk_len) if max_chunk_len > 0 else 1
            for chunk_idx_direct in range(num_chunks_direct):
                start_idx_direct = chunk_idx_direct * max_chunk_len
                end_idx_direct = min((chunk_idx_direct + 1) * max_chunk_len, frame_count)
                current_chunk_len_direct = end_idx_direct - start_idx_direct
                if current_chunk_len_direct == 0:
                    continue

                chunk_diffusion_timer_direct = {'last_time': time.time()}
                def diffusion_callback_for_chunk_direct(step, total_steps_chunk):
                    nonlocal chunk_diffusion_timer_direct
                    current_time_chunk = time.time()
                    step_duration_chunk = current_time_chunk - chunk_diffusion_timer_direct['last_time']
                    chunk_diffusion_timer_direct['last_time'] = current_time_chunk
                    desc_for_log_direct = f"Direct Upscale Chunk {chunk_idx_direct + 1}/{num_chunks_direct} (frames {start_idx_direct}-{end_idx_direct - 1})"
                    eta_seconds_chunk = step_duration_chunk * (total_steps_chunk - step) if step_duration_chunk > 0 and total_steps_chunk > 0 else 0
                    eta_formatted_chunk = format_time(eta_seconds_chunk)
                    logger.info(f"{desc_for_log_direct} - Diffusion: Step {step}/{total_steps_chunk}, Duration: {step_duration_chunk:.2f}s, ETA: {eta_formatted_chunk}")
                    
                    progress_val_rel_direct = (chunk_idx_direct + (step / total_steps_chunk if total_steps_chunk > 0 else 1)) / num_chunks_direct
                    current_overall_progress_temp = upscaling_loop_progress_start_no_scene_split + (progress_val_rel_direct * stage_weights["upscaling_loop"])
                    progress(current_overall_progress_temp, desc=f"{desc_for_log_direct} (Diff: {step}/{total_steps_chunk})")

                chunk_lr_frames_bgr_direct = all_lr_frames_bgr[start_idx_direct:end_idx_direct]
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
                        progress_callback=diffusion_callback_for_chunk_direct
                    )
                chunk_sr_frames_uint8_direct = tensor2vid_func(chunk_sr_tensor_bcthw_direct)

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        chunk_sr_frames_uint8_direct = adain_color_fix_func(chunk_sr_frames_uint8_direct, chunk_lr_video_data_direct)
                    elif color_fix_method == 'Wavelet':
                        chunk_sr_frames_uint8_direct = wavelet_color_fix_func(chunk_sr_frames_uint8_direct, chunk_lr_video_data_direct)

                # Ensure frame_files is available for naming output files
                for k_direct, frame_name_direct in enumerate(frame_files[start_idx_direct:end_idx_direct]):
                    frame_np_hwc_uint8_direct = chunk_sr_frames_uint8_direct[k_direct].cpu().numpy()
                    frame_bgr_direct = cv2.cvtColor(frame_np_hwc_uint8_direct, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_frames_dir, frame_name_direct), frame_bgr_direct)

                if save_chunks and chunks_permanent_save_path:
                    current_direct_chunks_save_path = chunks_permanent_save_path
                    os.makedirs(current_direct_chunks_save_path, exist_ok=True)
                    chunk_video_filename_direct = f"chunk_{chunk_idx_direct + 1:04d}.mp4"
                    chunk_video_path_direct = os.path.join(current_direct_chunks_save_path, chunk_video_filename_direct)
                    chunk_temp_assembly_dir_direct = os.path.join(temp_dir, f"temp_direct_chunk_{chunk_idx_direct+1}")
                    os.makedirs(chunk_temp_assembly_dir_direct, exist_ok=True)
                    frames_for_this_video_chunk_direct = []
                    for k_chunk_frame_direct, frame_name_in_chunk_direct in enumerate(frame_files[start_idx_direct:end_idx_direct]):
                        src_direct = os.path.join(output_frames_dir, frame_name_in_chunk_direct)
                        dst_direct = os.path.join(chunk_temp_assembly_dir_direct, f"frame_{k_chunk_frame_direct+1:06d}.png")
                        if os.path.exists(src_direct):
                            shutil.copy2(src_direct, dst_direct)
                            frames_for_this_video_chunk_direct.append(dst_direct)
                        else:
                            logger.warning(f"Src frame {src_direct} not found for direct upscale chunk video.")
                    if frames_for_this_video_chunk_direct:
                        util_create_video_from_frames(
                            chunk_temp_assembly_dir_direct, chunk_video_path_direct, input_fps_val,
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
            num_processed_frames_to_copy =len (os .listdir (output_frames_dir ))
            copy_proc_msg =f"Copying {num_processed_frames_to_copy} processed frames to permanent storage: {processed_frames_permanent_save_path}"
            status_log .append (copy_proc_msg )
            logger .info (copy_proc_msg )
            progress (current_overall_progress ,desc ="Copying processed frames...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Copying processed frames.",None
            frames_copied_count =0
            for frame_file_idx ,frame_file_name in enumerate (os .listdir (output_frames_dir )):
                shutil .copy2 (os .path .join (output_frames_dir ,frame_file_name ),os .path .join (processed_frames_permanent_save_path ,frame_file_name ))
                frames_copied_count +=1
                if frames_copied_count %100 ==0 or frames_copied_count ==num_processed_frames_to_copy :
                    loop_prog_frac =frames_copied_count /num_processed_frames_to_copy if num_processed_frames_to_copy >0 else 1.0
                    temp_overall_progress =upscaling_loop_progress_start_no_scene_split +(loop_prog_frac *stage_weights ["reassembly_copy_processed"])
                    progress (temp_overall_progress ,desc =f"Copying processed frames: {frames_copied_count}/{num_processed_frames_to_copy}")
            copied_proc_msg =f"Processed frames copied. Time: {format_time(time.time() - copy_processed_frames_start_time)}"
            status_log .append (copied_proc_msg )
            logger .info (copied_proc_msg )
            current_overall_progress =upscaling_loop_progress_start_no_scene_split +stage_weights ["reassembly_copy_processed"]
            progress (current_overall_progress ,desc ="Processed frames copied.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Processed frames copied.",None
        else : # Not saving processed frames or scene split enabled
            if stage_weights["reassembly_copy_processed"] > 0.0:
                 current_overall_progress +=stage_weights ["reassembly_copy_processed"]

        if not silent_upscaled_video_path: # If not already set (e.g., by scene merge)
            progress (current_overall_progress ,desc ="Creating silent video...")
            silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
            util_create_video_from_frames (output_frames_dir ,silent_upscaled_video_path ,input_fps_val ,
            ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )
            silent_video_msg ="Silent upscaled video created. Merging audio..."
            status_log .append (silent_video_msg )
            logger .info (silent_video_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg,None
        else: # Video already merged from scenes (is silent_upscaled_video_path)
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
                shutil .copy2 (silent_upscaled_video_path ,final_output_path )
            else :
                logger .error (f"Silent upscaled video path {silent_upscaled_video_path} not found for copy.")
                raise gr .Error ("Silent video not found for final output.")
        else :
            if os .path .exists (silent_upscaled_video_path ):
                 util_run_ffmpeg_command (f'ffmpeg -y -i "{silent_upscaled_video_path}" -i "{audio_source_video}" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? -shortest "{final_output_path}"',"Final Video and Audio Merge",logger =logger )
            else :
                logger .error (f"Silent upscaled video path {silent_upscaled_video_path} not found for audio merge.")
                raise gr .Error ("Silent video not found for audio merge.")

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

    if save_metadata and main_output_dir and base_output_filename_no_ext:
        status_info_for_final_meta = {"processing_time_total": total_processing_time}
        success, message = metadata_handler_module.save_metadata(
            save_flag=True, output_dir=main_output_dir, base_filename_no_ext=base_output_filename_no_ext,
            params_dict=params_for_metadata, status_info=status_info_for_final_meta, logger=logger
        )
        if success:
            final_meta_msg = f"Final metadata saved: {message.split(': ')[-1]}. Time to save: {format_time(time.time() - metadata_save_start_time)}"
            status_log.append(final_meta_msg)
            logger.info(final_meta_msg)
        else:
            status_log.append(f"Error saving final metadata: {message}")
            logger.error(message)
        current_overall_progress = initial_progress_metadata + stage_weights["metadata"]
        progress(current_overall_progress, desc="Metadata saved.")
        yield final_output_path, "\n".join(status_log), last_chunk_video_path, final_meta_msg if success else message, None
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

    yield final_output_path ,"\n".join (status_log ),last_chunk_video_path ,"Processing complete!",None
