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
    create_video_from_frames as util_create_video_from_frames
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

    star_model =None
    current_input_video_for_frames =input_video_path
    downscaled_temp_video =None
    status_log =["Process started..."]

    ui_total_diffusion_steps =steps
    direct_upscale_msg =""

    try :
        progress (current_overall_progress ,desc ="Initializing...")
        status_log .append ("Initializing upscaling process...")
        logger .info ("Initializing upscaling process...")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status

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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Calculating resolution..." # Changed status

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
                yield None ,"\n".join (status_log ),last_chunk_video_path ,"Input Downscaling..."

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
                yield None ,"\n".join (status_log ),last_chunk_video_path ,"Downscaling complete."
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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status

        scene_video_paths =[]
        # scenes_temp_dir =None # Not strictly needed here if only used within scene_split_params
        if enable_scene_split :
            scene_split_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Splitting video into scenes...")
            status_log .append ("Splitting video into scenes...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Scene Splitting..."

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
                # scenes_temp_dir =os .path .join (temp_dir ,"scenes") # Path to where scenes are stored

                scene_split_msg =f"Video split into {len(scene_video_paths)} scenes"
                status_log .append (scene_split_msg )
                logger .info (scene_split_msg )

                current_overall_progress =scene_split_progress_start +stage_weights ["scene_split"]
                progress (current_overall_progress ,desc =f"Scene splitting complete: {len(scene_video_paths)} scenes")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_split_msg

            except Exception as e :
                logger .error (f"Scene splitting failed: {e}", exc_info=True)
                status_log .append (f"Scene splitting failed: {e}")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene splitting failed: {e}"
                raise gr .Error (f"Scene splitting failed: {e}")
        else : # Scene split disabled, add its weight if it was > 0
            if stage_weights["scene_split"] > 0.0:
                current_overall_progress +=stage_weights ["scene_split"]

        model_load_progress_start =current_overall_progress
        progress (current_overall_progress ,desc ="Loading STAR model...")
        star_model_load_start_time =time .time ()
        model_cfg =EasyDict_class () 
        model_cfg .model_path =model_file_path

        model_device =torch .device (util_get_gpu_device (logger =logger ))if torch .cuda .is_available ()else torch .device ('cpu')
        star_model =VideoToVideo_sr_class (model_cfg ,device =model_device ) 
        model_load_msg =f"STAR model loaded on device {model_device}. Time: {format_time(time.time() - star_model_load_start_time)}"
        status_log .append (model_load_msg )
        logger .info (model_load_msg )
        current_overall_progress =model_load_progress_start +stage_weights ["model_load"]
        progress (current_overall_progress ,desc ="STAR model loaded.")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,"STAR model loaded."

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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Extracted {frame_count} frames."
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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Copying input frames..."

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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Input frames copied."
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
                        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status
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
                        scene_video_path=scene_video_path_item, scene_index=scene_idx, total_scenes=total_scenes, temp_dir=temp_dir, star_model=star_model,
                        final_prompt=scene_prompt_override, upscale_factor=upscale_factor_val, final_h=final_h_val, final_w=final_w_val, ui_total_diffusion_steps=ui_total_diffusion_steps,
                        solver_mode=solver_mode, cfg_scale=cfg_scale, max_chunk_len=max_chunk_len, vae_chunk=vae_chunk, color_fix_method=color_fix_method,
                        enable_tiling=enable_tiling, tile_size=tile_size, tile_overlap=tile_overlap, enable_sliding_window=enable_sliding_window, window_size=window_size, window_step=window_step,
                        save_frames=save_frames, scene_output_dir=frames_output_subfolder, progress_callback=scene_upscale_progress_callback,
                        enable_auto_caption_per_scene=scene_enable_auto_caption_current,
                        cogvlm_quant=actual_cogvlm_quant_val,
                        cogvlm_unload=cogvlm_unload,
                        progress=progress,
                        save_chunks=save_chunks,
                        chunks_permanent_save_path=frames_output_subfolder,
                        ffmpeg_preset=ffmpeg_preset, ffmpeg_quality_value=ffmpeg_quality_value, ffmpeg_use_gpu=ffmpeg_use_gpu,
                        save_metadata=save_metadata, metadata_params_base=scene_metadata_base_params,
                        util_extract_frames=util_extract_frames, util_auto_caption=util_auto_caption, util_get_gpu_device=util_get_gpu_device,
                        util_create_video_from_frames=util_create_video_from_frames, app_config=app_config_module, logger=logger,
                        metadata_handler=metadata_handler_module, format_time=format_time, preprocess=preprocess_func,
                        ImageSpliterTh=ImageSpliterTh_class, collate_fn=collate_fn_func, tensor2vid=tensor2vid_func,
                        adain_color_fix=adain_color_fix_func, wavelet_color_fix=wavelet_color_fix_func
                    )
                    processed_scene_video_path_final =None
                    for yield_type ,*data in scene_processor_generator :
                        if yield_type =="chunk_update":
                            chunk_vid_path ,chunk_stat_str =data
                            last_chunk_video_path =chunk_vid_path
                            last_chunk_status =chunk_stat_str
                            temp_status_log =status_log +[f"Processed: {chunk_stat_str}"]
                            yield None ,"\n".join (temp_status_log ),last_chunk_video_path ,last_chunk_status
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
                            yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status
                        elif yield_type =="error":
                            error_message =data [0 ]
                            logger .error (f"Error from scene_processor_generator: {error_message}")
                            status_log .append (f"Error processing scene {scene_idx + 1}: {error_message}")
                            yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene {scene_idx + 1} processing failed: {error_message}"
                            raise gr .Error (f"Scene {scene_idx + 1} processing failed: {error_message}")
                    if processed_scene_video_path_final :
                         processed_scene_videos .append (processed_scene_video_path_final )
                    else :
                        logger .error (f"Scene {scene_idx+1} finished processing but no final video path was yielded by process_single_scene.")
                        raise gr .Error (f"Scene {scene_idx+1} did not complete correctly.")
                except Exception as e :
                    logger .error (f"Error processing scene {scene_idx + 1} in run_upscale: {e}",exc_info =True )
                    status_log .append (f"Error processing scene {scene_idx + 1}: {e}")
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene {scene_idx + 1} processing failed: {e}"
                    raise gr .Error (f"Scene {scene_idx + 1} processing failed: {e}")

            current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
            scene_merge_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Merging processed scenes...")
            status_log .append ("Merging processed scenes...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Merging Scenes..."

            silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
            util_merge_scene_videos (processed_scene_videos ,silent_upscaled_video_path ,temp_dir ,
            ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )

            current_overall_progress =scene_merge_progress_start +stage_weights ["scene_merge"]
            progress (current_overall_progress ,desc ="Scene merging complete")
            scene_merge_msg =f"Successfully merged {len(processed_scene_videos)} processed scenes"
            status_log .append (scene_merge_msg )
            logger .info (scene_merge_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_merge_msg
        else : # Not enable_scene_split
            if 'frame_count'not in locals ()or 'input_fps_val'not in locals ()or 'frame_files'not in locals ():
                logger .warning ("Re-extracting frames as they were not found before non-scene-split upscaling loop.")
                frame_count ,input_fps_val ,frame_files =util_extract_frames (current_input_video_for_frames ,input_frames_dir ,logger =logger )
                params_for_metadata ["input_fps"]=input_fps_val

            all_lr_frames_bgr_for_preprocess =[]
            for frame_filename in frame_files :
                frame_lr_bgr =cv2 .imread (os .path .join (input_frames_dir ,frame_filename ))
                if frame_lr_bgr is None :
                    logger .error (f"Could not read frame {frame_filename} from {input_frames_dir}. Skipping.")
                    placeholder_h =params_for_metadata ["orig_h"]if params_for_metadata ["orig_h"]else 256
                    placeholder_w =params_for_metadata ["orig_w"]if params_for_metadata ["orig_w"]else 256
                    all_lr_frames_bgr_for_preprocess .append (np .zeros ((placeholder_h ,placeholder_w ,3 ),dtype =np .uint8 ))
                    continue
                all_lr_frames_bgr_for_preprocess .append (frame_lr_bgr )
            if len (all_lr_frames_bgr_for_preprocess )!=frame_count :
                 logger .warning (f"Mismatch in frame count and loaded LR frames for colorfix: {len(all_lr_frames_bgr_for_preprocess)} vs {frame_count}")

            if enable_tiling : # This implies not enable_scene_split
                loop_name ="Tiling Process"
                tiling_status_msg =f"Tiling enabled: Tile Size={tile_size}, Overlap={tile_overlap}. Processing {len(frame_files)} frames."
                status_log .append (tiling_status_msg )
                logger .info (tiling_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,tiling_status_msg
                total_frames_to_tile =len (frame_files )
                for i ,frame_filename in enumerate (progress .tqdm (frame_files ,desc =f"{loop_name} - Initializing...",total =total_frames_to_tile )):
                    frame_lr_bgr =cv2 .imread (os .path .join (input_frames_dir ,frame_filename ))
                    if frame_lr_bgr is None :
                        logger .warning (f"Skipping frame {frame_filename} due to read error during tiling.")
                        placeholder_path =os .path .join (input_frames_dir ,frame_filename )
                        if os .path .exists (placeholder_path ): shutil .copy2 (placeholder_path ,os .path .join (output_frames_dir ,frame_filename ))
                        continue
                    single_lr_frame_tensor_norm =preprocess_func ([frame_lr_bgr ])
                    spliter =ImageSpliterTh_class (single_lr_frame_tensor_norm ,int (tile_size ),int (tile_overlap ),sf =upscale_factor_val )
                    num_patches_this_frame =getattr (spliter ,'num_patches',sum (1 for _ in ImageSpliterTh_class (single_lr_frame_tensor_norm ,int (tile_size ),int (tile_overlap ),sf =upscale_factor_val )))
                    spliter =ImageSpliterTh_class (single_lr_frame_tensor_norm ,int (tile_size ),int (tile_overlap ),sf =upscale_factor_val )
                    for patch_idx ,(patch_lr_tensor_norm ,patch_coords )in enumerate (spliter ):
                        patch_lr_video_data =patch_lr_tensor_norm
                        patch_pre_data ={'video_data':patch_lr_video_data ,'y':final_prompt ,
                        'target_res':(int (round (patch_lr_tensor_norm .shape [-2 ]*upscale_factor_val )), int (round (patch_lr_tensor_norm .shape [-1 ]*upscale_factor_val )))}
                        patch_data_tensor_cuda =collate_fn_func (patch_pre_data ,gpu_device )
                        callback_step_timer_patch ={'last_time':time .time ()}
                        def diffusion_callback_for_patch (step ,total_steps ):
                            nonlocal callback_step_timer_patch
                            current_time =time .time ()
                            step_duration =current_time -callback_step_timer_patch ['last_time']
                            callback_step_timer_patch ['last_time']=current_time
                            current_patch_desc =f"Frame {i+1}/{total_frames_to_tile}, Patch {patch_idx+1}/{num_patches_this_frame}"
                            tqdm_step_info =f"{step_duration:.2f}s/it ({step}/{total_steps})"if step_duration >0.001 else f"({step}/{total_steps})"
                            eta_seconds =step_duration *(total_steps -step )if step_duration >0 and total_steps >0 else 0
                            eta_formatted =format_time (eta_seconds )
                            logger .info (f"{current_patch_desc} - Diffusion: {tqdm_step_info}, ETA: {eta_formatted}")
                            diffusion_progress_in_patch =step /total_steps if total_steps >0 else 1.0
                            patches_processed_in_frame_fraction =(patch_idx +diffusion_progress_in_patch )/num_patches_this_frame if num_patches_this_frame >0 else 1.0
                            frames_processed_fraction =(i +patches_processed_in_frame_fraction )/total_frames_to_tile if total_frames_to_tile >0 else 1.0
                            current_loop_stage_progress =upscaling_loop_progress_start +(frames_processed_fraction *stage_weights ["upscaling_loop"])
                            progress (current_loop_stage_progress ,desc =f"{current_patch_desc} - Diffusion: {tqdm_step_info}")
                        with torch .no_grad ():
                            patch_sr_tensor_bcthw =star_model .test (
                            patch_data_tensor_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,solver_mode =solver_mode ,
                            guide_scale =cfg_scale ,max_chunk_len =1 ,vae_decoder_chunk_size =1 , progress_callback =diffusion_callback_for_patch )
                        patch_sr_frames_uint8 =tensor2vid_func (patch_sr_tensor_bcthw )
                        if color_fix_method !='None':
                            if color_fix_method =='AdaIN': patch_sr_frames_uint8 =adain_color_fix_func (patch_sr_frames_uint8 ,patch_lr_video_data )
                            elif color_fix_method =='Wavelet': patch_sr_frames_uint8 =wavelet_color_fix_func (patch_sr_frames_uint8 ,patch_lr_video_data )
                        single_patch_frame_hwc =patch_sr_frames_uint8 [0 ]
                        result_patch_chw_01 =single_patch_frame_hwc .permute (2 ,0 ,1 ).float ()/255.0
                        spliter .update_gaussian (result_patch_chw_01 .unsqueeze (0 ),patch_coords )
                        del patch_data_tensor_cuda ,patch_sr_tensor_bcthw ,patch_sr_frames_uint8
                        if torch .cuda .is_available ():torch .cuda .empty_cache ()
                    final_frame_tensor_chw =spliter .gather ()
                    final_frame_np_hwc_uint8 =(final_frame_tensor_chw .squeeze (0 ).permute (1 ,2 ,0 ).clamp (0 ,1 ).cpu ().numpy ()*255 ).astype (np .uint8 )
                    final_frame_bgr =cv2 .cvtColor (final_frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                    cv2 .imwrite (os .path .join (output_frames_dir ,frame_filename ),final_frame_bgr )
                    loop_progress_frac =(i +1 )/total_frames_to_tile if total_frames_to_tile >0 else 1.0
                    current_overall_progress_temp =upscaling_loop_progress_start +(loop_progress_frac *stage_weights ["upscaling_loop"])
                    progress (current_overall_progress_temp )
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Tiling frame {i+1}/{total_frames_to_tile} processed"
            elif enable_sliding_window : # This implies not enable_scene_split
                loop_name ="Sliding Window Process"
                sliding_status_msg =f"Sliding Window: Size={window_size}, Step={window_step}. Processing {frame_count} frames."
                status_log .append (sliding_status_msg )
                logger .info (sliding_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,sliding_status_msg
                processed_frame_filenames =[None ]*frame_count
                effective_window_size =int (window_size )
                effective_window_step =int (window_step )
                window_indices_to_process =list (range (0 ,frame_count ,effective_window_step ))
                total_windows_to_process =len (window_indices_to_process )
                for window_iter_idx ,i_start_idx in enumerate (progress .tqdm (window_indices_to_process ,desc =f"{loop_name} - Initializing...",total =total_windows_to_process )):
                    start_idx =i_start_idx
                    end_idx =min (i_start_idx +effective_window_size ,frame_count )
                    current_window_len =end_idx -start_idx
                    if current_window_len ==0 :continue
                    is_last_window_iteration =(window_iter_idx ==total_windows_to_process -1 )
                    if is_last_window_iteration and current_window_len <effective_window_size and frame_count >=effective_window_size :
                        start_idx =max (0 ,frame_count -effective_window_size )
                        end_idx =frame_count
                        current_window_len =end_idx -start_idx
                    window_lr_frames_bgr =all_lr_frames_bgr_for_preprocess [start_idx :end_idx ]
                    if not window_lr_frames_bgr :continue
                    window_lr_video_data =preprocess_func (window_lr_frames_bgr )
                    window_pre_data ={'video_data':window_lr_video_data ,'y':final_prompt ,'target_res':(final_h_val ,final_w_val )}
                    window_data_cuda =collate_fn_func (window_pre_data ,gpu_device )
                    current_window_display_num =window_iter_idx +1
                    callback_step_timer_window ={'last_time':time .time ()}
                    def diffusion_callback_for_window (step ,total_steps ):
                        nonlocal callback_step_timer_window
                        current_time =time .time ()
                        step_duration =current_time -callback_step_timer_window ['last_time']
                        callback_step_timer_window ['last_time']=current_time
                        base_desc_win =f"{loop_name}: {current_window_display_num}/{total_windows_to_process} windows (frames {start_idx}-{end_idx-1})"
                        tqdm_step_info =f"{step_duration:.2f}s/it ({step}/{total_steps})"if step_duration >0.001 else f"{step}/{total_steps}"
                        eta_seconds =step_duration *(total_steps -step )if step_duration >0 and total_steps >0 else 0
                        eta_formatted =format_time (eta_seconds )
                        logger .info (f"{base_desc_win} - Diffusion: {tqdm_step_info}, ETA: {eta_formatted}")
                        diffusion_progress_in_window =step /total_steps if total_steps >0 else 1.0
                        windows_processed_fraction =(window_iter_idx +diffusion_progress_in_window )/total_windows_to_process if total_windows_to_process >0 else 1.0
                        current_loop_stage_progress =upscaling_loop_progress_start +(windows_processed_fraction *stage_weights ["upscaling_loop"])
                        progress (current_loop_stage_progress ,desc =f"{base_desc_win} - Diffusion: {tqdm_step_info}")
                    with torch .no_grad ():
                        window_sr_tensor_bcthw =star_model .test (
                        window_data_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,solver_mode =solver_mode ,
                        guide_scale =cfg_scale ,max_chunk_len =current_window_len ,vae_decoder_chunk_size =min (vae_chunk ,current_window_len ),
                        progress_callback =diffusion_callback_for_window )
                    window_sr_frames_uint8 =tensor2vid_func (window_sr_tensor_bcthw )
                    if color_fix_method !='None':
                        if color_fix_method =='AdaIN': window_sr_frames_uint8 =adain_color_fix_func (window_sr_frames_uint8 ,window_lr_video_data )
                        elif color_fix_method =='Wavelet': window_sr_frames_uint8 =wavelet_color_fix_func (window_sr_frames_uint8 ,window_lr_video_data )
                    save_from_start_offset_local =0
                    save_to_end_offset_local =current_window_len
                    if total_windows_to_process >1 :
                        overlap_amount =effective_window_size -effective_window_step
                        if overlap_amount >0 :
                            if window_iter_idx ==0 : save_to_end_offset_local =effective_window_size -(overlap_amount //2 )
                            elif is_last_window_iteration : save_from_start_offset_local =(overlap_amount //2 )
                            else :
                                save_from_start_offset_local =(overlap_amount //2 )
                                save_to_end_offset_local =effective_window_size -(overlap_amount -save_from_start_offset_local )
                        save_from_start_offset_local =max (0 ,min (save_from_start_offset_local ,current_window_len -1 if current_window_len >0 else 0 ))
                        save_to_end_offset_local =max (save_from_start_offset_local ,min (save_to_end_offset_local ,current_window_len ))
                    for k_local in range (save_from_start_offset_local ,save_to_end_offset_local ):
                        k_global =start_idx +k_local
                        if 0 <=k_global <frame_count and processed_frame_filenames [k_global ]is None :
                            frame_np_hwc_uint8 =window_sr_frames_uint8 [k_local ].cpu ().numpy ()
                            frame_bgr =cv2 .cvtColor (frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                            out_f_path =os .path .join (output_frames_dir ,frame_files [k_global ])
                            cv2 .imwrite (out_f_path ,frame_bgr )
                            processed_frame_filenames [k_global ]=frame_files [k_global ]
                    del window_data_cuda ,window_sr_tensor_bcthw ,window_sr_frames_uint8
                    if torch .cuda .is_available ():torch .cuda .empty_cache ()
                    loop_progress_frac =current_window_display_num /total_windows_to_process if total_windows_to_process >0 else 1.0
                    current_overall_progress_temp =upscaling_loop_progress_start +(loop_progress_frac *stage_weights ["upscaling_loop"])
                    progress (current_overall_progress_temp )
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Sliding window {current_window_display_num}/{total_windows_to_process} processed"
                num_missed_fallback =0
                for idx_fb ,fname_fb in enumerate (frame_files ):
                    if processed_frame_filenames [idx_fb ]is None :
                        num_missed_fallback +=1
                        logger .warning (f"Frame {fname_fb} (index {idx_fb}) was not processed by sliding window, copying LR frame.")
                        lr_frame_path =os .path .join (input_frames_dir ,fname_fb )
                        if os .path .exists (lr_frame_path ): shutil .copy2 (lr_frame_path ,os .path .join (output_frames_dir ,fname_fb ))
                        else : logger .error (f"LR frame {lr_frame_path} not found for fallback copy.")
                if num_missed_fallback >0 :
                    missed_msg =f"{loop_name} - Copied {num_missed_fallback} LR frames as fallback for unprocessed frames."
                    status_log .append (missed_msg )
                    logger .info (missed_msg )
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,missed_msg
            else: # Standard chunked processing (not scene_split, not tiling, not sliding_window)
                loop_name ="Chunked Processing"
                chunk_status_msg ="Normal chunked processing."
                status_log .append (chunk_status_msg )
                logger .info (chunk_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,chunk_status_msg
                num_chunks =math .ceil (frame_count /max_chunk_len )if max_chunk_len >0 else (1 if frame_count >0 else 0 )
                if num_chunks ==0 and frame_count >0 :num_chunks =1
                for i_chunk_idx_tuple in enumerate (progress .tqdm (range (num_chunks ),desc =f"{loop_name} - Initializing...",total =num_chunks )):
                    i_chunk_idx = i_chunk_idx_tuple[0]
                    start_idx =i_chunk_idx *max_chunk_len
                    end_idx =min ((i_chunk_idx +1 )*max_chunk_len ,frame_count )
                    current_chunk_len =end_idx -start_idx
                    if current_chunk_len ==0 :continue
                    if end_idx >len (all_lr_frames_bgr_for_preprocess )or start_idx <0 :
                         logger .error (f"Chunk range {start_idx}-{end_idx} is invalid for LR frames list of len {len(all_lr_frames_bgr_for_preprocess)}")
                         continue
                    chunk_lr_frames_bgr =all_lr_frames_bgr_for_preprocess [start_idx :end_idx ]
                    if not chunk_lr_frames_bgr :continue
                    chunk_lr_video_data =preprocess_func (chunk_lr_frames_bgr )
                    chunk_pre_data ={'video_data':chunk_lr_video_data ,'y':final_prompt ,'target_res':(final_h_val ,final_w_val )}
                    chunk_data_cuda =collate_fn_func (chunk_pre_data ,gpu_device )
                    current_chunk_display_num =i_chunk_idx +1
                    callback_step_timer_chunk ={'last_time':time .time ()}
                    def diffusion_callback_for_chunk (step ,total_steps ):
                        nonlocal callback_step_timer_chunk
                        current_time =time .time ()
                        step_duration =current_time -callback_step_timer_chunk ['last_time']
                        callback_step_timer_chunk ['last_time']=current_time
                        tqdm_step_info =f"{step_duration:.2f}s/it ({step}/{total_steps})"if step_duration >0.001 else f"({step}/{total_steps})"
                        eta_seconds =step_duration *(total_steps -step )if step_duration >0 and total_steps >0 else 0
                        eta_formatted =format_time (eta_seconds )
                        logger .info (f"{loop_name}: Chunk {current_chunk_display_num}/{num_chunks} (Frames {start_idx}-{end_idx-1}) - Diffusion: {tqdm_step_info}, ETA: {eta_formatted}")
                        diffusion_progress_in_chunk =step /total_steps if total_steps >0 else 1.0
                        chunks_processed_fraction =(i_chunk_idx +diffusion_progress_in_chunk )/num_chunks if num_chunks >0 else 1.0
                        current_loop_stage_progress =upscaling_loop_progress_start +(chunks_processed_fraction *stage_weights ["upscaling_loop"])
                        desc_lines =[ f"{loop_name}", f"Current Batch: {current_chunk_display_num}/{num_chunks} (Frames: {start_idx} to {end_idx-1})", f"Diffusion Progress: {tqdm_step_info}" ]
                        progress (current_loop_stage_progress ,desc ="\n".join (desc_lines )) 
                    with torch .no_grad ():
                        chunk_sr_tensor_bcthw =star_model .test (
                        chunk_data_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,solver_mode =solver_mode ,
                        guide_scale =cfg_scale ,max_chunk_len =current_chunk_len ,vae_decoder_chunk_size =min (vae_chunk ,current_chunk_len ),
                        progress_callback =diffusion_callback_for_chunk )
                    chunk_sr_frames_uint8 =tensor2vid_func (chunk_sr_tensor_bcthw )
                    if color_fix_method !='None':
                        if color_fix_method =='AdaIN': chunk_sr_frames_uint8 =adain_color_fix_func (chunk_sr_frames_uint8 ,chunk_lr_video_data )
                        elif color_fix_method =='Wavelet': chunk_sr_frames_uint8 =wavelet_color_fix_func (chunk_sr_frames_uint8 ,chunk_lr_video_data )
                    for k ,frame_name in enumerate (frame_files [start_idx :end_idx ]):
                        frame_np_hwc_uint8 =chunk_sr_frames_uint8 [k ].cpu ().numpy ()
                        frame_bgr =cv2 .cvtColor (frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                        cv2 .imwrite (os .path .join (output_frames_dir ,frame_name ),frame_bgr )
                    if save_chunks and chunks_permanent_save_path :
                        chunk_video_filename =f"chunk_{current_chunk_display_num:04d}.mp4"
                        chunk_video_path =os .path .join (chunks_permanent_save_path ,chunk_video_filename )
                        chunk_temp_dir =os .path .join (temp_dir ,f"temp_chunk_{current_chunk_display_num}")
                        os .makedirs (chunk_temp_dir ,exist_ok =True )
                        for k_frame_in_chunk ,frame_name_in_chunk in enumerate (frame_files [start_idx :end_idx ]):
                            src_frame =os .path .join (output_frames_dir ,frame_name_in_chunk )
                            dst_frame =os .path .join (chunk_temp_dir ,f"frame_{k_frame_in_chunk + 1:06d}.png")
                            shutil .copy2 (src_frame ,dst_frame )
                        util_create_video_from_frames (chunk_temp_dir ,chunk_video_path ,input_fps_val ,
                        ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )
                        shutil .rmtree (chunk_temp_dir )
                        chunk_save_msg =f"Saved chunk {current_chunk_display_num}/{num_chunks} to: {chunk_video_path}"
                        status_log .append (chunk_save_msg )
                        logger .info (chunk_save_msg )
                        last_chunk_video_path =chunk_video_path
                        last_chunk_status =f"Chunk {current_chunk_display_num}/{num_chunks} (frames {start_idx+1}-{end_idx})"
                        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status
                    if save_metadata :
                        status_info_for_chunk_meta ={"current_chunk":current_chunk_display_num ,"total_chunks":num_chunks ,"overall_process_start_time":overall_process_start_time}
                        try :
                            metadata_handler_module .save_metadata (save_flag =True , output_dir =main_output_dir , base_filename_no_ext =base_output_filename_no_ext ,
                                params_dict =params_for_metadata , status_info =status_info_for_chunk_meta , logger =logger )
                        except Exception as e_meta : logger .warning (f"Failed to save/update metadata after chunk {current_chunk_display_num}: {e_meta}")
                    del chunk_data_cuda ,chunk_sr_tensor_bcthw ,chunk_sr_frames_uint8
                    if torch .cuda .is_available ():torch .cuda .empty_cache ()
                    loop_progress_frac =current_chunk_display_num /num_chunks if num_chunks >0 else 1.0
                    current_overall_progress_temp =upscaling_loop_progress_start +(loop_progress_frac *stage_weights ["upscaling_loop"])
                    progress (current_overall_progress_temp )
                    if not (save_chunks and chunks_permanent_save_path ): 
                         current_chunk_status_text =f"Chunk {current_chunk_display_num}/{num_chunks} processed"
                         yield None ,"\n".join (status_log ),last_chunk_video_path ,current_chunk_status_text
            
            # Common for all non-scene-split paths after loop
            current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
            upscaling_total_duration_msg =f"All frame upscaling operations finished. Total upscaling time: {format_time(time.time() - upscaling_loop_start_time)}"
            status_log .append (upscaling_total_duration_msg )
            logger .info (upscaling_total_duration_msg )
            progress (current_overall_progress ,desc ="Upscaling complete.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,upscaling_total_duration_msg
            
            # Scene merge stage for non-scene-split case (effectively a skip)
            if stage_weights["scene_merge"] > 0.0:
                current_overall_progress += stage_weights["scene_merge"]
                progress(current_overall_progress, desc="Skipping scene merge...") # Or just pass


        initial_progress_reassembly =current_overall_progress

        if save_frames and not enable_scene_split and output_frames_dir and processed_frames_permanent_save_path :
            copy_processed_frames_start_time =time .time ()
            num_processed_frames_to_copy =len (os .listdir (output_frames_dir ))
            copy_proc_msg =f"Copying {num_processed_frames_to_copy} processed frames to permanent storage: {processed_frames_permanent_save_path}"
            status_log .append (copy_proc_msg )
            logger .info (copy_proc_msg )
            progress (current_overall_progress ,desc ="Copying processed frames...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Copying processed frames..."
            frames_copied_count =0
            for frame_file_idx ,frame_file_name in enumerate (os .listdir (output_frames_dir )):
                shutil .copy2 (os .path .join (output_frames_dir ,frame_file_name ),os .path .join (processed_frames_permanent_save_path ,frame_file_name ))
                frames_copied_count +=1
                if frames_copied_count %100 ==0 or frames_copied_count ==num_processed_frames_to_copy :
                    loop_prog_frac =frames_copied_count /num_processed_frames_to_copy if num_processed_frames_to_copy >0 else 1.0
                    temp_overall_progress =initial_progress_reassembly +(loop_prog_frac *stage_weights ["reassembly_copy_processed"])
                    progress (temp_overall_progress ,desc =f"Copying processed frames: {frames_copied_count}/{num_processed_frames_to_copy}")
            copied_proc_msg =f"Processed frames copied. Time: {format_time(time.time() - copy_processed_frames_start_time)}"
            status_log .append (copied_proc_msg )
            logger .info (copied_proc_msg )
            current_overall_progress =initial_progress_reassembly +stage_weights ["reassembly_copy_processed"]
            progress (current_overall_progress ,desc ="Processed frames copied.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,"Processed frames copied."
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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg
        else: # Video already merged from scenes (is silent_upscaled_video_path)
            silent_video_msg ="Scene-merged video ready. Merging audio..."
            status_log .append (silent_video_msg )
            logger .info (silent_video_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg


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
        yield final_output_path ,"\n".join (status_log ),last_chunk_video_path ,"Finalizing..."

        if create_comparison_video_enabled and final_output_path and os.path.exists(final_output_path):
            comparison_video_progress_start = current_overall_progress
            comparison_video_start_time = time.time()
            original_video_for_comparison = params_for_metadata.get("input_video_path", input_video_path)
            comparison_output_path = get_comparison_output_path(final_output_path)
            
            progress(current_overall_progress, desc="Creating comparison video...")
            comparison_status_msg = "Creating comparison video..."
            status_log.append(comparison_status_msg)
            yield final_output_path, "\n".join(status_log), last_chunk_video_path, comparison_status_msg
            
            try:
                comparison_success = create_comparison_video(
                    original_video_path=original_video_for_comparison,
                    upscaled_video_path=final_output_path,
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
                else:
                    comparison_error_msg = f"Comparison video creation failed. Time: {format_time(time.time() - comparison_video_start_time)}"
                    status_log.append(comparison_error_msg)
                    logger.warning(comparison_error_msg)
            except Exception as e_comparison:
                comparison_error_msg = f"Error creating comparison video: {e_comparison}. Time: {format_time(time.time() - comparison_video_start_time)}"
                status_log.append(comparison_error_msg)
                logger.error(comparison_error_msg, exc_info=True)
            
            current_overall_progress = comparison_video_progress_start + stage_weights.get("comparison_video", 0.0) # Use actual weight
            progress(current_overall_progress, desc="Comparison video processing complete.")
            yield final_output_path, "\n".join(status_log), last_chunk_video_path, "Comparison video complete."
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
            yield final_output_path ,"\n".join (status_log ),last_chunk_video_path ,meta_saved_msg if success else message
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

        yield final_output_path ,"\n".join (status_log ),last_chunk_video_path ,"Processing complete!"

    except gr .Error as e :
        logger .error (f"A Gradio UI Error occurred: {e}",exc_info =True )
        status_log .append (f"Error: {e}")
        current_overall_progress =min (current_overall_progress +0.01 ,1.0 ) # Small increment for error
        progress (current_overall_progress ,desc =f"Error: {str(e)[:50]}")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Error: {e}"
    except Exception as e :
        logger .error (f"An unexpected error occurred during upscaling: {e}",exc_info =True )
        status_log .append (f"Critical Error: {e}")
        current_overall_progress =min (current_overall_progress +0.01 ,1.0 ) # Small increment for error
        progress (current_overall_progress ,desc =f"Critical Error: {str(e)[:50]}")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Critical Error: {e}"
        # Re-raise to stop Gradio execution gracefully, or let the finally block handle cleanup
        # For Gradio, it's often better to let it complete the yield cycle.
    finally :
        if star_model is not None :
            try :
                if hasattr (star_model ,'to'):star_model .to ('cpu')
                del star_model
            except Exception as e_del_model:
                 logger.warning(f"Exception during star_model cleanup: {e_del_model}")

        gc .collect ()
        if torch .cuda .is_available ():torch .cuda .empty_cache ()
        
        util_cleanup_temp_dir (temp_dir ,logger =logger )

        total_process_duration =time .time ()-overall_process_start_time
        final_cleanup_msg =f"STAR upscaling process finished and cleaned up. Total processing time: {format_time(total_process_duration)}"
        logger .info (final_cleanup_msg )

        final_output_video_path_check = locals().get('final_output_path', None)
        output_video_exists = final_output_video_path_check and os.path.exists(final_output_video_path_check)

        if not output_video_exists :
             if status_log and status_log [-1 ]and not status_log [-1 ].startswith ("Error:")and not status_log [-1 ].startswith ("Critical Error:"):
                no_output_msg ="Processing finished, but output video was not found or not created."
                logger .warning (no_output_msg )
                # Do not append to status_log here as it might be returned by the generator caller

        # Clean up .tmp lock file
        local_base_output_filename_no_ext = locals().get('base_output_filename_no_ext', None)
        local_main_output_dir = locals().get('main_output_dir', app_config_module.DEFAULT_OUTPUT_DIR if app_config_module else None)

        if local_base_output_filename_no_ext and local_main_output_dir:
            tmp_lock_file_to_delete =os .path .join (local_main_output_dir, f"{local_base_output_filename_no_ext}.tmp")
            if os .path .exists (tmp_lock_file_to_delete ):
                try :
                    os .remove (tmp_lock_file_to_delete )
                    logger .info (f"Successfully deleted lock file: {tmp_lock_file_to_delete}")
                except Exception as e_lock_del :
                    logger .error (f"Failed to delete lock file {tmp_lock_file_to_delete}: {e_lock_del}")