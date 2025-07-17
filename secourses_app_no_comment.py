import gradio as gr
import os
import platform
import sys
import torch
import torchvision
import subprocess
import cv2
import numpy as np
import math
import time
import shutil
import tempfile
import threading
import gc
from easydict import EasyDict
from argparse import ArgumentParser ,Namespace
import logging
import re
from pathlib import Path

from logic import config as app_config
from logic import metadata_handler # Import the new metadata handler

from logic .cogvlm_utils import (
load_cogvlm_model as util_load_cogvlm_model ,
unload_cogvlm_model as util_unload_cogvlm_model ,
auto_caption as util_auto_caption ,
COG_VLM_AVAILABLE as UTIL_COG_VLM_AVAILABLE ,
BITSANDBYTES_AVAILABLE as UTIL_BITSANDBYTES_AVAILABLE
)

from logic .common_utils import format_time

from logic .ffmpeg_utils import (
run_ffmpeg_command as util_run_ffmpeg_command ,
extract_frames as util_extract_frames ,
create_video_from_frames as util_create_video_from_frames
)

from logic .file_utils import (
sanitize_filename as util_sanitize_filename ,
get_batch_filename as util_get_batch_filename ,
get_next_filename as util_get_next_filename ,
cleanup_temp_dir as util_cleanup_temp_dir ,
get_video_resolution as util_get_video_resolution ,
get_available_drives as util_get_available_drives ,
open_folder as util_open_folder
)

from logic .scene_utils import (
split_video_into_scenes as util_split_video_into_scenes ,
merge_scene_videos as util_merge_scene_videos ,
split_video_only as util_split_video_only
)

from logic .upscaling_utils import (
calculate_upscale_params as util_calculate_upscale_params
)

from logic .gpu_utils import (
get_available_gpus as util_get_available_gpus ,
set_gpu_device as util_set_gpu_device ,
get_gpu_device as util_get_gpu_device ,
validate_gpu_availability as util_validate_gpu_availability
)

SELECTED_GPU_ID =0

parser =ArgumentParser (description ="Ultimate SECourses STAR Video Upscaler")
parser .add_argument ('--share',action ='store_true',help ="Enable Gradio live share")
parser .add_argument ('--outputs_folder',type =str ,default ="outputs",help ="Main folder for output videos and related files")
args =parser .parse_args ()

def process_batch_videos (
batch_input_folder_val ,batch_output_folder_val ,
user_prompt_val ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,

enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,

cogvlm_quant_radio_val =None ,cogvlm_unload_radio_val =None ,do_auto_caption_first_val =False ,
progress =gr .Progress (track_tqdm =True )
):

    if not batch_input_folder_val or not os .path .exists (batch_input_folder_val ):
        raise gr .Error ("Please provide a valid input folder.")

    if not batch_output_folder_val :
        raise gr .Error ("Please provide an output folder.")

    try :

        video_extensions =['.mp4','.avi','.mov','.mkv','.wmv','.flv','.webm','.m4v']
        video_files =[]

        for file in os .listdir (batch_input_folder_val ):
            if any (file .lower ().endswith (ext )for ext in video_extensions ):
                video_files .append (os .path .join (batch_input_folder_val ,file ))

        if not video_files :
            raise gr .Error (f"No video files found in: {batch_input_folder_val}")

        os .makedirs (batch_output_folder_val ,exist_ok =True )

        processed_files =[]
        failed_files =[]

        for i ,video_file in enumerate (video_files ):
            try :
                progress ((i /len (video_files ))*0.9 ,desc =f"Processing {i+1}/{len(video_files)}: {Path(video_file).name}")

                upscale_generator =run_upscale (
                video_file ,user_prompt_val ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
                upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
                max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
                enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
                enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
                enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
                ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
                save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,

                enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
                scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
                scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
                scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,

                is_batch_mode =True ,batch_output_dir =batch_output_folder_val ,original_filename =video_file ,

                enable_auto_caption_per_scene =do_auto_caption_first_val and enable_scene_split_check_val ,
                cogvlm_quant =cogvlm_quant_radio_val if cogvlm_quant_radio_val is not None else 0 , # Will be converted by get_quant_value_from_display
                cogvlm_unload =cogvlm_unload_radio_val if cogvlm_unload_radio_val is not None else 'full',
                progress =progress
                )

                final_output =None
                for output_val ,status_val , _ , _ in upscale_generator : # Adjusted to expect 4 values from run_upscale yield
                    final_output =output_val

                if final_output and os .path .exists (final_output ):
                    processed_files .append ({"input":video_file ,"output":final_output ,"status":"success"})
                else :
                    failed_files .append ((video_file ,"Output file not created"))

            except Exception as e :
                logger .error (f"Failed to process {video_file}: {e}")
                failed_files .append ((video_file ,str (e )))

        progress (1.0 ,desc =f"Batch processing complete: {len(processed_files)} successful, {len(failed_files)} failed")

        status_msg =f"Batch processing complete!\n"
        status_msg +=f"Successfully processed: {len(processed_files)} videos\n"
        status_msg +=f"Failed: {len(failed_files)} videos\n"
        status_msg +=f"Output folder: {batch_output_folder_val}"

        if failed_files :
            status_msg +=f"\n\nFailed files:\n"
            for file_path ,error in failed_files [:5 ]:
                status_msg +=f"- {Path(file_path).name}: {error}\n"
            if len (failed_files )>5 :
                status_msg +=f"... and {len(failed_files) - 5} more"

        return None ,status_msg

    except Exception as e :
        logger .error (f"Error in batch processing: {e}")
        raise gr .Error (f"Batch processing failed: {e}")

def process_single_scene (
scene_video_path ,scene_index ,total_scenes ,temp_dir ,star_model ,
final_prompt ,upscale_factor ,final_h ,final_w ,ui_total_diffusion_steps ,
solver_mode ,cfg_scale ,max_chunk_len ,vae_chunk ,color_fix_method ,
enable_tiling ,tile_size ,tile_overlap ,enable_sliding_window ,window_size ,window_step ,
save_frames ,scene_output_dir ,progress_callback =None ,

enable_auto_caption_per_scene =False ,cogvlm_quant =0 ,cogvlm_unload ='full',
progress =None ,save_chunks =False ,chunks_permanent_save_path =None ,ffmpeg_preset ="medium",ffmpeg_quality_value =23 ,ffmpeg_use_gpu =False ,
save_metadata =False ,metadata_params_base: dict =None # This is already structured correctly
):

    try :
        scene_start_time =time .time ()
        scene_name =f"scene_{scene_index+1:04d}"

        if progress_callback :
            progress_callback (0.0 ,f"Processing scene {scene_index+1}/{total_scenes}: {scene_name}")

        scene_temp_dir =os .path .join (temp_dir ,scene_name )
        scene_input_frames_dir =os .path .join (scene_temp_dir ,"input_frames")
        scene_output_frames_dir =os .path .join (scene_temp_dir ,"output_frames")

        os .makedirs (scene_temp_dir ,exist_ok =True )
        os .makedirs (scene_input_frames_dir ,exist_ok =True )
        os .makedirs (scene_output_frames_dir ,exist_ok =True )

        if progress_callback :
            progress_callback (0.1 ,f"Scene {scene_index+1}: Extracting frames...")

        scene_frame_count ,scene_fps ,scene_frame_files =util_extract_frames (scene_video_path ,scene_input_frames_dir ,logger =logger )

        scene_input_frames_permanent =None
        scene_output_frames_permanent =None
        if save_frames and scene_output_dir :
            scene_frames_dir =os .path .join (scene_output_dir ,"scenes",scene_name )
            scene_input_frames_permanent =os .path .join (scene_frames_dir ,"input_frames")
            scene_output_frames_permanent =os .path .join (scene_frames_dir ,"processed_frames")
            os .makedirs (scene_input_frames_permanent ,exist_ok =True )
            os .makedirs (scene_output_frames_permanent ,exist_ok =True )

            for frame_file in scene_frame_files :
                shutil .copy2 (
                os .path .join (scene_input_frames_dir ,frame_file ),
                os .path .join (scene_input_frames_permanent ,frame_file )
                )

        scene_prompt =final_prompt
        if enable_auto_caption_per_scene :
            if progress_callback :
                progress_callback (0.15 ,f"Scene {scene_index+1}: Generating caption...")

            try :

                scene_caption ,_ =util_auto_caption (
                scene_video_path ,
                cogvlm_quant ,
                cogvlm_unload ,
                app_config .COG_VLM_MODEL_PATH ,
                logger =logger ,
                progress =progress
                )
                if not scene_caption .startswith ("Error:"):
                    scene_prompt =scene_caption
                    logger .info (f"Scene {scene_index+1} auto-caption: {scene_caption[:100]}...")
                else :
                    logger .warning (f"Scene {scene_index+1} auto-caption failed, using original prompt")
            except Exception as e :
                logger .error (f"Error auto-captioning scene {scene_index+1}: {e}")

        if progress_callback :
            progress_callback (0.2 ,f"Scene {scene_index+1}: Loading frames...")

        all_lr_frames_bgr =[]
        for frame_filename in scene_frame_files :
            frame_lr_bgr =cv2 .imread (os .path .join (scene_input_frames_dir ,frame_filename ))
            if frame_lr_bgr is None :
                logger .error (f"Could not read frame {frame_filename} from scene {scene_name}")
                continue
            all_lr_frames_bgr .append (frame_lr_bgr )

        if not all_lr_frames_bgr :
            raise Exception (f"No valid frames found in scene {scene_name}")

        total_noise_levels =900

        if progress_callback :
            progress_callback (0.3 ,f"Scene {scene_index+1}: Starting upscaling...")

        gpu_device =util_get_gpu_device (logger =logger )

        # Upscaling logic (tiling, sliding window, normal chunking) - This part remains the same
        # ... (omitted for brevity, assuming it's unchanged from original)
        if enable_tiling :

            for i ,frame_filename in enumerate (scene_frame_files ):
                frame_lr_bgr =cv2 .imread (os .path .join (scene_input_frames_dir ,frame_filename ))
                if frame_lr_bgr is None :
                    logger .warning (f"Skipping frame {frame_filename} due to read error during tiling.")
                    continue

                single_lr_frame_tensor_norm =preprocess ([frame_lr_bgr ])
                spliter =ImageSpliterTh (single_lr_frame_tensor_norm ,int (tile_size ),int (tile_overlap ),sf =upscale_factor )

                for patch_idx ,(patch_lr_tensor_norm ,patch_coords )in enumerate (spliter ):
                    patch_lr_video_data =patch_lr_tensor_norm

                    patch_pre_data ={
                    'video_data':patch_lr_video_data ,
                    'y':scene_prompt ,
                    'target_res':(int (round (patch_lr_tensor_norm .shape [-2 ]*upscale_factor )),
                    int (round (patch_lr_tensor_norm .shape [-1 ]*upscale_factor )))
                    }
                    patch_data_tensor_cuda =collate_fn (patch_pre_data ,gpu_device )

                    with torch .no_grad ():
                        patch_sr_tensor_bcthw =star_model .test (
                        patch_data_tensor_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,
                        solver_mode =solver_mode ,guide_scale =cfg_scale ,
                        max_chunk_len =1 ,vae_decoder_chunk_size =1
                        )

                    patch_sr_frames_uint8 =tensor2vid (patch_sr_tensor_bcthw )

                    if color_fix_method !='None':
                        if color_fix_method =='AdaIN':
                            patch_sr_frames_uint8 =adain_color_fix (patch_sr_frames_uint8 ,patch_lr_video_data )
                        elif color_fix_method =='Wavelet':
                            patch_sr_frames_uint8 =wavelet_color_fix (patch_sr_frames_uint8 ,patch_lr_video_data )

                    single_patch_frame_hwc =patch_sr_frames_uint8 [0 ]
                    result_patch_chw_01 =single_patch_frame_hwc .permute (2 ,0 ,1 ).float ()/255.0

                    spliter .update_gaussian (result_patch_chw_01 .unsqueeze (0 ),patch_coords )

                    del patch_data_tensor_cuda ,patch_sr_tensor_bcthw ,patch_sr_frames_uint8
                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()

                final_frame_tensor_chw =spliter .gather ()
                final_frame_np_hwc_uint8 =(final_frame_tensor_chw .squeeze (0 ).permute (1 ,2 ,0 ).clamp (0 ,1 ).cpu ().numpy ()*255 ).astype (np .uint8 )
                final_frame_bgr =cv2 .cvtColor (final_frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                cv2 .imwrite (os .path .join (scene_output_frames_dir ,frame_filename ),final_frame_bgr )

                if progress_callback :
                    frame_progress =0.3 +(i /scene_frame_count )*0.5
                    progress_callback (frame_progress ,f"Scene {scene_index+1}: Processing frame {i+1}/{scene_frame_count}")

        elif enable_sliding_window :

            processed_frame_filenames =[None ]*scene_frame_count
            effective_window_size =int (window_size )
            effective_window_step =int (window_step )

            window_indices_to_process =list (range (0 ,scene_frame_count ,effective_window_step ))
            total_windows_to_process =len (window_indices_to_process )

            # Initialize chunk tracking for sliding window in scenes
            saved_chunks = set()  # Track which chunks have been saved
            processed_frames_tracker = [False] * scene_frame_count  # Track which frames have been processed

            # Helper functions for chunk mapping in scenes
            def map_window_to_chunks_scene(start_idx, end_idx, max_chunk_len):
                """Map a window frame range to affected chunk indices"""
                first_chunk = start_idx // max_chunk_len
                last_chunk = (end_idx - 1) // max_chunk_len
                return list(range(first_chunk, last_chunk + 1))

            def get_chunk_frame_range_scene(chunk_idx, max_chunk_len, total_frames):
                """Get the frame range for a specific chunk"""
                start_frame = chunk_idx * max_chunk_len
                end_frame = min((chunk_idx + 1) * max_chunk_len, total_frames)
                return start_frame, end_frame

            def is_chunk_complete_scene(chunk_idx, processed_frames_tracker, max_chunk_len, total_frames):
                """Check if all frames in a chunk have been processed"""
                start_frame, end_frame = get_chunk_frame_range_scene(chunk_idx, max_chunk_len, total_frames)
                for frame_idx in range(start_frame, end_frame):
                    if not processed_frames_tracker[frame_idx]:
                        return False
                return True

            def save_sliding_window_chunk_scene(chunk_idx, max_chunk_len, total_frames, save_chunks, scene_output_dir, 
                                            scene_name, temp_dir, scene_output_frames_dir, scene_frame_files, scene_fps, 
                                            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger):
                """Save a chunk video from sliding window processed frames in scene"""
                if not save_chunks or not scene_output_dir or not scene_name:
                    return None, None
                
                start_frame, end_frame = get_chunk_frame_range_scene(chunk_idx, max_chunk_len, total_frames)
                current_chunk_display_num = chunk_idx + 1
                
                current_scene_chunks_save_path = os.path.join(scene_output_dir, "scenes", scene_name, "chunks")
                os.makedirs(current_scene_chunks_save_path, exist_ok=True)
                
                chunk_video_filename = f"chunk_{current_chunk_display_num:04d}.mp4"
                chunk_video_path = os.path.join(current_scene_chunks_save_path, chunk_video_filename)
                chunk_temp_dir = os.path.join(temp_dir, scene_name, f"temp_sliding_chunk_{current_chunk_display_num}")
                os.makedirs(chunk_temp_dir, exist_ok=True)

                frames_for_chunk = []
                for k, frame_name in enumerate(scene_frame_files[start_frame:end_frame]):
                    src_frame = os.path.join(scene_output_frames_dir, frame_name)
                    dst_frame = os.path.join(chunk_temp_dir, f"frame_{k+1:06d}.png")
                    if os.path.exists(src_frame):
                        shutil.copy2(src_frame, dst_frame)
                        frames_for_chunk.append(dst_frame)
                    else:
                        logger.warning(f"Src frame {src_frame} not found for sliding window chunk video in scene {scene_name}.")
                
                if frames_for_chunk:
                    util_create_video_from_frames(chunk_temp_dir, chunk_video_path, scene_fps,
                                                ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)
                    
                    shutil.rmtree(chunk_temp_dir)
                    
                    chunk_save_msg = f"Saved sliding window chunk {current_chunk_display_num} (frames {start_frame+1}-{end_frame}) for scene {scene_name} to: {chunk_video_path}"
                    logger.info(chunk_save_msg)
                    
                    return chunk_video_path, f"Scene {scene_name} Sliding Window Chunk {current_chunk_display_num} (frames {start_frame+1}-{end_frame})"
                else:
                    logger.warning(f"No frames for sliding window chunk {current_chunk_display_num} in scene {scene_name}, video not created.")
                    shutil.rmtree(chunk_temp_dir)
                    return None, None

            for window_iter_idx ,i_start_idx in enumerate (window_indices_to_process ):
                start_idx =i_start_idx
                end_idx =min (i_start_idx +effective_window_size ,scene_frame_count )
                current_window_len =end_idx -start_idx

                if current_window_len ==0 :
                    continue

                is_last_window_iteration =(window_iter_idx ==total_windows_to_process -1 )
                if is_last_window_iteration and current_window_len <effective_window_size and scene_frame_count >=effective_window_size :
                    start_idx =max (0 ,scene_frame_count -effective_window_size )
                    end_idx =scene_frame_count
                    current_window_len =end_idx -start_idx

                window_lr_frames_bgr =all_lr_frames_bgr [start_idx :end_idx ]
                if not window_lr_frames_bgr :
                    continue

                window_lr_video_data =preprocess (window_lr_frames_bgr )

                window_pre_data ={
                'video_data':window_lr_video_data ,
                'y':scene_prompt ,
                'target_res':(final_h ,final_w )
                }
                window_data_cuda =collate_fn (window_pre_data ,gpu_device )

                with torch .no_grad ():
                    window_sr_tensor_bcthw =star_model .test (
                    window_data_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,
                    solver_mode =solver_mode ,guide_scale =cfg_scale ,
                    max_chunk_len =current_window_len ,vae_decoder_chunk_size =min (vae_chunk ,current_window_len )
                    )

                window_sr_frames_uint8 =tensor2vid (window_sr_tensor_bcthw )

                if color_fix_method !='None':
                    if color_fix_method =='AdaIN':
                        window_sr_frames_uint8 =adain_color_fix (window_sr_frames_uint8 ,window_lr_video_data )
                    elif color_fix_method =='Wavelet':
                        window_sr_frames_uint8 =wavelet_color_fix (window_sr_frames_uint8 ,window_lr_video_data )

                save_from_start_offset_local =0
                save_to_end_offset_local =current_window_len
                
                # Use simple overlap logic that works correctly for all cases
                if total_windows_to_process >1:
                    overlap_amount =effective_window_size -effective_window_step
                    if overlap_amount >0 :
                        if window_iter_idx ==0 :
                            # First window: save from start to end minus half overlap
                            save_to_end_offset_local =effective_window_size -(overlap_amount //2 )
                        elif is_last_window_iteration :
                            # Last window: save from half overlap to end
                            save_from_start_offset_local =(overlap_amount //2 )
                        else :
                            # Middle windows: save only the center portion, avoiding overlap zones
                            save_from_start_offset_local =(overlap_amount //2 )
                            save_to_end_offset_local =effective_window_size -(overlap_amount //2 )
                    
                    # Ensure bounds are valid
                    save_from_start_offset_local =max (0 ,min (save_from_start_offset_local ,current_window_len -1 if current_window_len >0 else 0 ))
                    save_to_end_offset_local =max (save_from_start_offset_local ,min (save_to_end_offset_local ,current_window_len ))

                for k_local in range (save_from_start_offset_local ,save_to_end_offset_local ):
                    k_global =start_idx +k_local
                    if 0 <=k_global <scene_frame_count and processed_frame_filenames [k_global ]is None :
                        frame_np_hwc_uint8 =window_sr_frames_uint8 [k_local ].cpu ().numpy ()
                        frame_bgr =cv2 .cvtColor (frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                        cv2 .imwrite (os .path .join (scene_output_frames_dir ,scene_frame_files [k_global ]),frame_bgr )
                        processed_frame_filenames [k_global ]=scene_frame_files [k_global ]
                        processed_frames_tracker[k_global] = True

                # Check and save complete chunks for sliding window in scenes
                affected_chunks = map_window_to_chunks_scene(start_idx, end_idx, max_chunk_len)
                for chunk_idx in affected_chunks:
                    if chunk_idx not in saved_chunks and is_chunk_complete_scene(chunk_idx, processed_frames_tracker, max_chunk_len, scene_frame_count):
                        chunk_video_path, chunk_status = save_sliding_window_chunk_scene(
                            chunk_idx, max_chunk_len, scene_frame_count, save_chunks, scene_output_dir,
                            scene_name, temp_dir, scene_output_frames_dir, scene_frame_files, scene_fps,
                            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger
                        )
                        if chunk_video_path:
                            saved_chunks.add(chunk_idx)
                            logger.info(f"Scene {scene_name}: Saved chunk {chunk_idx + 1} from sliding window processing")

                del window_data_cuda ,window_sr_tensor_bcthw ,window_sr_frames_uint8
                if torch .cuda .is_available ():
                    torch .cuda .empty_cache ()

                if progress_callback :
                    window_progress =0.3 +(window_iter_idx /total_windows_to_process )*0.5
                    progress_callback (window_progress ,f"Scene {scene_index+1}: Processing window {window_iter_idx+1}/{total_windows_to_process}")

            num_missed_fallback =0
            for idx_fb ,fname_fb in enumerate (scene_frame_files ):
                if processed_frame_filenames [idx_fb ]is None :
                    num_missed_fallback +=1
                    logger .warning (f"Frame {fname_fb} (index {idx_fb}) was not processed by sliding window, copying LR frame.")
                    lr_frame_path =os .path .join (scene_input_frames_dir ,fname_fb )
                    if os .path .exists (lr_frame_path ):
                        shutil .copy2 (lr_frame_path ,os .path .join (scene_output_frames_dir ,fname_fb ))
                        processed_frames_tracker[idx_fb] = True
                    else :
                        logger .error (f"LR frame {lr_frame_path} not found for fallback copy.")

            if num_missed_fallback >0 :
                missed_msg =f"Sliding window - Copied {num_missed_fallback} LR frames as fallback for unprocessed frames."
                logger .info (missed_msg )

            # Save any remaining incomplete chunks after all windows are processed for scenes
            num_total_chunks = math.ceil(scene_frame_count / max_chunk_len) if max_chunk_len > 0 else 1
            for chunk_idx in range(num_total_chunks):
                if chunk_idx not in saved_chunks:
                    # Force save incomplete chunks
                    chunk_video_path, chunk_status = save_sliding_window_chunk_scene(
                        chunk_idx, max_chunk_len, scene_frame_count, save_chunks, scene_output_dir,
                        scene_name, temp_dir, scene_output_frames_dir, scene_frame_files, scene_fps,
                        ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger
                    )
                    if chunk_video_path:
                        saved_chunks.add(chunk_idx)
                        logger.info(f"Scene {scene_name}: Saved remaining chunk {chunk_idx + 1} from sliding window processing")

        else : # Normal chunking for the scene
            num_chunks =math .ceil (scene_frame_count /max_chunk_len )if max_chunk_len >0 else 1

            for chunk_idx in range (num_chunks ):
                start_idx =chunk_idx *max_chunk_len
                end_idx =min ((chunk_idx +1 )*max_chunk_len ,scene_frame_count )
                current_chunk_len =end_idx -start_idx

                if current_chunk_len ==0 :
                    continue

                chunk_lr_frames_bgr =all_lr_frames_bgr [start_idx :end_idx ]
                chunk_lr_video_data =preprocess (chunk_lr_frames_bgr )

                chunk_pre_data ={
                'video_data':chunk_lr_video_data ,
                'y':scene_prompt ,
                'target_res':(final_h ,final_w )
                }
                chunk_data_cuda =collate_fn (chunk_pre_data ,gpu_device )

                with torch .no_grad ():
                    chunk_sr_tensor_bcthw =star_model .test (
                    chunk_data_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,
                    solver_mode =solver_mode ,guide_scale =cfg_scale ,
                    max_chunk_len =current_chunk_len ,vae_decoder_chunk_size =min (vae_chunk ,current_chunk_len )
                    )

                chunk_sr_frames_uint8 =tensor2vid (chunk_sr_tensor_bcthw )

                if color_fix_method !='None':
                    if color_fix_method =='AdaIN':
                        chunk_sr_frames_uint8 =adain_color_fix (chunk_sr_frames_uint8 ,chunk_lr_video_data )
                    elif color_fix_method =='Wavelet':
                        chunk_sr_frames_uint8 =wavelet_color_fix (chunk_sr_frames_uint8 ,chunk_lr_video_data )

                for k ,frame_name in enumerate (scene_frame_files [start_idx :end_idx ]):
                    frame_np_hwc_uint8 =chunk_sr_frames_uint8 [k ].cpu ().numpy ()
                    frame_bgr =cv2 .cvtColor (frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                    cv2 .imwrite (os .path .join (scene_output_frames_dir ,frame_name ),frame_bgr )

                if save_chunks and scene_output_dir and scene_name :
                    current_scene_chunks_save_path =os .path .join (scene_output_dir ,"scenes",scene_name ,"chunks")
                    os .makedirs (current_scene_chunks_save_path ,exist_ok =True )

                    chunk_video_filename =f"chunk_{chunk_idx+1:04d}.mp4"
                    chunk_video_path =os .path .join (current_scene_chunks_save_path ,chunk_video_filename )

                    chunk_temp_assembly_dir =os .path .join (temp_dir ,scene_name ,f"temp_chunk_{chunk_idx+1}")
                    os .makedirs (chunk_temp_assembly_dir ,exist_ok =True )

                    frames_for_this_video_chunk =[]
                    for k_chunk_frame ,frame_name_in_chunk in enumerate (scene_frame_files [start_idx :end_idx ]):
                        src_frame_for_chunk =os .path .join (scene_output_frames_dir ,frame_name_in_chunk )
                        dst_frame_for_chunk =os .path .join (chunk_temp_assembly_dir ,f"frame_{k_chunk_frame+1:06d}.png")
                        if os .path .exists (src_frame_for_chunk ):
                            shutil .copy2 (src_frame_for_chunk ,dst_frame_for_chunk )
                            frames_for_this_video_chunk .append (dst_frame_for_chunk )
                        else :
                            logger .warning (f"Source frame {src_frame_for_chunk} not found for scene chunk video (Scene: {scene_name}, Chunk: {chunk_idx+1}).")

                    if frames_for_this_video_chunk :
                        util_create_video_from_frames (
                        chunk_temp_assembly_dir ,
                        chunk_video_path ,
                        scene_fps ,
                        ffmpeg_preset ,
                        ffmpeg_quality_value ,
                        ffmpeg_use_gpu ,
                        logger =logger
                        )
                        logger .info (f"Saved scene chunk {chunk_idx+1}/{num_chunks} for scene {scene_name} to: {chunk_video_path}")
                    else :
                        logger .warning (f"No frames copied for scene chunk {chunk_idx+1}/{num_chunks} (Scene: {scene_name}), video not created.")

                    shutil .rmtree (chunk_temp_assembly_dir )

                    chunk_save_msg =f"Saved chunk {chunk_idx+1}/{num_chunks} to: {chunk_video_path}"
                    logger .info (chunk_save_msg )

                    if save_metadata and metadata_params_base :
                        try :
                            # Determine output directory for this specific metadata file
                            # metadata_params_base already contains 'is_batch_mode' and 'batch_output_dir'
                            # scene_output_dir is the main output for this scene if not batch
                            meta_output_dir = metadata_params_base.get('batch_output_dir') if metadata_params_base.get('is_batch_mode') else scene_output_dir
                            if metadata_params_base.get('is_batch_mode') and scene_output_dir and not meta_output_dir.startswith(scene_output_dir):
                                # If batch mode, and scene_output_dir is specific (e.g., .../batch_out/video_name/scenes/scene_0001)
                                # we might want to save scene-chunk progress there.
                                # For now, let's assume chunk progress for scenes in batch mode goes to the batch video's main folder.
                                # OR, it could go into the scene's specific folder under the batch video folder.
                                # The current logic: meta_output_dir = scene_output_dir (which is .../scenes/scene_name for non-batch)
                                # or batch_output_dir (which is .../batch_out_folder for batch)
                                # Let's clarify: scene-specific chunk progress should go into the scene's folder.
                                meta_chunk_prog_output_dir = os.path.join(scene_output_dir, "scenes", scene_name) if scene_output_dir else \
                                                             (os.path.join(metadata_params_base.get('batch_output_dir'), "scenes", scene_name) if metadata_params_base.get('is_batch_mode') else temp_dir) # fallback
                                os.makedirs(meta_chunk_prog_output_dir, exist_ok=True)


                            scene_chunk_status_info = {
                                "current_chunk": chunk_idx + 1,
                                "total_chunks": num_chunks,
                                "overall_process_start_time": scene_start_time, # For scene's own processing time calculation
                                "chunk_frame_range": (start_idx + 1, end_idx)  # 1-indexed frame range for this chunk
                                # scene_specific_data is not added here, as this is chunk progress within a scene
                                # but we need to pass scene_fps to params_dict if it's not already there correctly.
                                # metadata_params_base should have a general input_fps, _prepare_metadata_dict will prioritize scene_fps from scene_specific_data if present.
                                # For chunk progress, we might want to use scene_fps.
                            }
                            # Temporarily update input_fps in a copy of metadata_params_base for this chunk's metadata if needed
                            temp_params_for_chunk_meta = metadata_params_base.copy()
                            temp_params_for_chunk_meta['input_fps'] = scene_fps # Ensure this chunk uses scene_fps

                            metadata_handler.save_metadata(
                                save_flag=True, # Already checked save_metadata
                                output_dir=meta_chunk_prog_output_dir, # Save in scene's folder
                                base_filename_no_ext=f"{scene_name}_chunk_{chunk_idx+1:04d}_progress", # Unique name
                                params_dict=temp_params_for_chunk_meta,
                                status_info=scene_chunk_status_info,
                                logger=logger
                            )
                        except Exception as e_meta :
                            logger .warning (f"Failed to save metadata after scene chunk {chunk_idx+1}: {e_meta}")

                del chunk_data_cuda ,chunk_sr_tensor_bcthw ,chunk_sr_frames_uint8
                if torch .cuda .is_available ():
                    torch .cuda .empty_cache ()

                if progress_callback :
                    chunk_progress =0.3 +(chunk_idx /num_chunks )*0.5
                    progress_callback (chunk_progress ,f"Scene {scene_index+1}: Processing chunk {chunk_idx+1}/{num_chunks}")

        if save_frames and scene_output_frames_permanent :
            if progress_callback :
                progress_callback (0.85 ,f"Scene {scene_index+1}: Saving processed frames...")

            for frame_file in os .listdir (scene_output_frames_dir ):
                shutil .copy2 (
                os .path .join (scene_output_frames_dir ,frame_file ),
                os .path .join (scene_output_frames_permanent ,frame_file )
                )

        if progress_callback :
            progress_callback (0.9 ,f"Scene {scene_index+1}: Creating video...")

        scene_output_video =os .path .join (scene_temp_dir ,f"{scene_name}.mp4")
        util_create_video_from_frames (
        scene_output_frames_dir ,scene_output_video ,scene_fps ,
        "medium",23 ,False ,logger =logger # Uses fixed medium settings for intermediate scene videos
        )

        scene_duration =time .time ()-scene_start_time

        if progress_callback :
            progress_callback (1.0 ,f"Scene {scene_index+1}: Complete ({format_time(scene_duration)})")

        logger .info (f"Scene {scene_name} processed successfully in {format_time(scene_duration)}")

        if save_metadata and metadata_params_base and scene_output_dir :
            if progress_callback :
                progress_callback (0.95 ,f"Scene {scene_index+1}: Saving metadata...")

            scene_metadata_dir =os .path .join (scene_output_dir ,"scenes",scene_name )
            os .makedirs (scene_metadata_dir ,exist_ok =True )

            current_scene_specific_data ={ # Renamed to avoid conflict
                "scene_index":scene_index +1 ,
                "scene_name":scene_name ,
                "scene_prompt":scene_prompt , # Use the actual prompt used for this scene
                "scene_frame_count":scene_frame_count ,
                "scene_fps":scene_fps ,
                "scene_processing_time":scene_duration ,
                "scene_video_path":scene_output_video, # Temporary path, final path is part of main metadata
                "scene_frame_range": (1, scene_frame_count)  # 1-indexed frame range for this scene
            }
            
            # For the final scene metadata, we augment the base parameters with scene-specific info
            final_scene_status_info = {
                "scene_specific_data": current_scene_specific_data,
                "processing_time_total": scene_duration # This metadata file is specifically for this scene's summary
            }

            # metadata_params_base already contains all the main settings
            # The _prepare_metadata_dict function will merge scene_specific_data correctly
            success, message = metadata_handler.save_metadata(
                save_flag=True, # Already checked save_metadata
                output_dir=scene_metadata_dir,
                base_filename_no_ext=f"{scene_name}_metadata",
                params_dict=metadata_params_base, # Pass the base UI/config params
                status_info=final_scene_status_info, # Pass scene specific results and timing
                logger=logger
            )

            if success :
                logger .info (f"Scene {scene_index+1} metadata saved to: {message.split(': ')[-1]}")
            else :
                logger .warning (f"Failed to save scene {scene_index+1} metadata: {message}")

        return scene_output_video ,scene_frame_count ,scene_fps

    except Exception as e :
        logger .error (f"Error processing scene {scene_index+1}: {e}")
        raise e

try :

    script_dir =os .path .dirname (os .path .abspath (__file__ ))
    base_path =script_dir

    if not os .path .isdir (os .path .join (base_path ,'video_to_video')):
        print (f"Warning: 'video_to_video' directory not found in inferred base_path: {base_path}. Attempting to use parent directory.")
        base_path =os .path .dirname (base_path )
        if not os .path .isdir (os .path .join (base_path ,'video_to_video')):
            print (f"Error: Could not auto-determine STAR repository root. Please set 'base_path' manually.")
            print (f"Current inferred base_path: {base_path}")

    print (f"Using STAR repository base_path: {base_path}")
    if base_path not in sys .path :
        sys .path .insert (0 ,base_path )

except Exception as e_path :
    print (f"Error setting up base_path: {e_path}")
    print ("Please ensure app.py is correctly placed or base_path is manually set.")
    sys .exit (1 )

try :
    from video_to_video .video_to_video_model import VideoToVideo_sr
    from video_to_video .utils .seed import setup_seed
    from video_to_video .utils .logger import get_logger
    from video_super_resolution .color_fix import adain_color_fix ,wavelet_color_fix
    from inference_utils import tensor2vid ,preprocess ,collate_fn
    from video_super_resolution .scripts .util_image import ImageSpliterTh
    from video_to_video .utils .config import cfg as star_cfg
except ImportError as e :
    print (f"Error importing STAR components: {e}")
    print (f"Searched in sys.path: {sys.path}")
    print ("Please ensure the STAR repository is correctly in the Python path (set by base_path) and all dependencies from 'requirements.txt' are installed.")
    sys .exit (1 )

logger =get_logger ()

logger .setLevel (logging .INFO )
found_stream_handler =False
for handler in logger .handlers :
    if isinstance (handler ,logging .StreamHandler ):
        handler .setLevel (logging .INFO )
        found_stream_handler =True
        logger .info ("Diagnostic: Explicitly set StreamHandler level to INFO.")
if not found_stream_handler :

    ch =logging .StreamHandler ()
    ch .setLevel (logging .INFO )

    logger .addHandler (ch )
    logger .info ("Diagnostic: No StreamHandler found, added a new one with INFO level.")
logger .info (f"Logger '{logger.name}' configured with level: {logging.getLevelName(logger.level)}. Handlers: {logger.handlers}")

app_config .initialize_paths_and_prompts (base_path ,args .outputs_folder ,star_cfg )

os .makedirs (app_config .DEFAULT_OUTPUT_DIR ,exist_ok =True )

if not os .path .exists (app_config .LIGHT_DEG_MODEL_PATH ):
     logger .error (f"FATAL: Light degradation model not found at {app_config.LIGHT_DEG_MODEL_PATH}.")
if not os .path .exists (app_config .HEAVY_DEG_MODEL_PATH ):
     logger .error (f"FATAL: Heavy degradation model not found at {app_config.HEAVY_DEG_MODEL_PATH}.")

def run_upscale (
input_video_path ,user_prompt ,positive_prompt ,negative_prompt ,model_choice ,
upscale_factor_slider ,cfg_scale ,steps ,solver_mode , # `steps` is the arg name for ui_total_diffusion_steps
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

is_batch_mode =False ,batch_output_dir =None ,original_filename =None ,

enable_auto_caption_per_scene =False ,cogvlm_quant =0 ,cogvlm_unload ='full', # cogvlm_quant is display value here
progress =gr .Progress (track_tqdm =True )
):
    if not input_video_path or not os .path .exists (input_video_path ):
        raise gr .Error ("Please select a valid input video file.")

    last_chunk_video_path =None
    last_chunk_status ="No chunks processed yet"

    setup_seed (666 )
    overall_process_start_time =time .time ()
    logger .info ("Overall upscaling process started.")

    current_overall_progress =0.0

    # Prepare the base dictionary for metadata. This will be updated as more info becomes available.
    params_for_metadata = {
        "input_video_path": input_video_path, "user_prompt": user_prompt, "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt, "model_choice": model_choice,
        "upscale_factor_slider": upscale_factor_slider, "cfg_scale": cfg_scale,
        "ui_total_diffusion_steps": steps, "solver_mode": solver_mode,
        "max_chunk_len": max_chunk_len, "vae_chunk": vae_chunk, "color_fix_method": color_fix_method,
        "enable_tiling": enable_tiling, "tile_size": tile_size, "tile_overlap": tile_overlap,
        "enable_sliding_window": enable_sliding_window, "window_size": window_size, "window_step": window_step,
        "enable_target_res": enable_target_res, "target_h": target_h, "target_w": target_w,
        "target_res_mode": target_res_mode, "ffmpeg_preset": ffmpeg_preset,
        "ffmpeg_quality_value": ffmpeg_quality_value, "ffmpeg_use_gpu": ffmpeg_use_gpu,
        "enable_scene_split": enable_scene_split, "scene_split_mode": scene_split_mode,
        "scene_min_scene_len": scene_min_scene_len, "scene_threshold": scene_threshold,
        "scene_manual_split_type": scene_manual_split_type, "scene_manual_split_value": scene_manual_split_value,
        "is_batch_mode": is_batch_mode, "batch_output_dir": batch_output_dir,
        # Placeholders, will be updated
        "final_output_path": None, "orig_w": None, "orig_h": None,
        "input_fps": None, "upscale_factor": None, "final_w": None, "final_h": None,
    }
    # Convert cogvlm_quant from display value to actual int value if applicable
    actual_cogvlm_quant_val = cogvlm_quant # If already int
    if isinstance(cogvlm_quant, str) and app_config.UTIL_COG_VLM_AVAILABLE: # Check if it's a display string
        _temp_map_rev = {v: k for k, v in app_config.get_cogvlm_quant_choices_map(torch.cuda.is_available(), app_config.UTIL_BITSANDBYTES_AVAILABLE).items()}
        actual_cogvlm_quant_val = _temp_map_rev.get(cogvlm_quant, 0)


    stage_weights ={
    "init_paths_res":0.03 ,
    "scene_split":0.05 if enable_scene_split else 0.0 ,
    "downscale":0.07 ,
    "model_load":0.05 ,
    "extract_frames":0.10 ,
    "copy_input_frames":0.05 ,
    "upscaling_loop":0.50 if enable_scene_split else 0.60 ,
    "scene_merge":0.05 if enable_scene_split else 0.0 ,
    "reassembly_copy_processed":0.05 ,
    "reassembly_audio_merge":0.03 ,
    "metadata":0.02
    }

    if not enable_target_res or not util_calculate_upscale_params (1 ,1 ,1 ,1 ,target_res_mode ,logger =logger )[0 ]:
        stage_weights ["downscale"]=0.0
    if not save_frames :
        stage_weights ["copy_input_frames"]=0.0

    total_weight =sum (stage_weights .values ())
    if total_weight >0 :
        for key in stage_weights :
            stage_weights [key ]/=total_weight

    if is_batch_mode and batch_output_dir and original_filename :
        base_output_filename_no_ext ,output_video_path ,batch_main_dir =util_get_batch_filename (batch_output_dir ,original_filename )
        main_output_dir =batch_main_dir
    else :
        base_output_filename_no_ext ,output_video_path =util_get_next_filename (app_config .DEFAULT_OUTPUT_DIR )
        main_output_dir =app_config .DEFAULT_OUTPUT_DIR
    
    params_for_metadata["final_output_path"] = output_video_path # Initial assignment

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

    ui_total_diffusion_steps =steps # Already using the correct name from args
    direct_upscale_msg =""

    try :
        progress (current_overall_progress ,desc ="Initializing...")
        status_log .append ("Initializing upscaling process...")
        logger .info ("Initializing upscaling process...")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status

        final_prompt =(user_prompt .strip ()+". "+positive_prompt .strip ()).strip ()

        model_file_path =app_config .LIGHT_DEG_MODEL_PATH if model_choice ==app_config .DEFAULT_MODEL_CHOICE else app_config .HEAVY_DEG_MODEL_PATH
        if not os .path .exists (model_file_path ):
            raise gr .Error (f"STAR model weight not found: {model_file_path}")

        orig_h_val ,orig_w_val =util_get_video_resolution (input_video_path ) # Renamed to avoid conflict
        params_for_metadata["orig_h"] = orig_h_val
        params_for_metadata["orig_w"] = orig_w_val
        status_log .append (f"Original resolution: {orig_w_val}x{orig_h_val}")
        logger .info (f"Original resolution: {orig_w_val}x{orig_h_val}")

        current_overall_progress +=stage_weights ["init_paths_res"]
        progress (current_overall_progress ,desc ="Calculating target resolution...")

        upscale_factor_val = None # To store calculated upscale factor
        final_h_val, final_w_val = None, None # To store final calculated H, W

        if enable_target_res :
            needs_downscale ,ds_h ,ds_w ,upscale_factor_calc ,final_h_calc ,final_w_calc =util_calculate_upscale_params (
            orig_h_val ,orig_w_val ,target_h ,target_w ,target_res_mode ,logger =logger
            )
            upscale_factor_val = upscale_factor_calc
            final_h_val, final_w_val = final_h_calc, final_w_calc
            
            params_for_metadata["upscale_factor"] = upscale_factor_val
            params_for_metadata["final_h"] = final_h_val
            params_for_metadata["final_w"] = final_w_val

            status_log .append (f"Target resolution mode: {target_res_mode }. Calculated upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}")
            logger .info (f"Target resolution mode: {target_res_mode }. Calculated upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}")
            if needs_downscale :
                downscale_stage_start_time =time .time ()

                downscale_progress_start =current_overall_progress
                progress (current_overall_progress ,desc ="Downscaling input video...")
                downscale_status_msg =f"Downscaling input to {ds_w}x{ds_h} before upscaling."
                status_log .append (downscale_status_msg )
                logger .info (downscale_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,downscale_status_msg
                downscaled_temp_video =os .path .join (temp_dir ,"downscaled_input.mp4")
                scale_filter =f"scale='trunc(iw*min({ds_w}/iw,{ds_h}/ih)/2)*2':'trunc(ih*min({ds_w}/iw,{ds_h}/ih)/2)*2'"

                ffmpeg_opts_downscale =""
                if ffmpeg_use_gpu :
                    nvenc_preset_down =ffmpeg_preset
                    if ffmpeg_preset in ["ultrafast","superfast","veryfast","faster","fast"]:nvenc_preset_down ="fast"
                    elif ffmpeg_preset in ["slower","veryslow"]:nvenc_preset_down ="slow"
                    ffmpeg_opts_downscale =f'-c:v h264_nvenc -preset:v {nvenc_preset_down} -cq:v {ffmpeg_quality_value} -pix_fmt yuv420p'
                else :
                    ffmpeg_opts_downscale =f'-c:v libx264 -preset {ffmpeg_preset} -crf {ffmpeg_quality_value} -pix_fmt yuv420p'

                cmd =f'ffmpeg -y -i "{input_video_path}" -vf "{scale_filter}" {ffmpeg_opts_downscale} -c:a copy "{downscaled_temp_video}"'
                util_run_ffmpeg_command (cmd ,"Input Downscaling with Audio Copy",logger =logger )
                current_input_video_for_frames =downscaled_temp_video
                # Update orig_h, orig_w for metadata if downscaled
                orig_h_val ,orig_w_val =util_get_video_resolution (downscaled_temp_video )
                params_for_metadata["orig_h"] = orig_h_val # Update with downscaled res for consistency
                params_for_metadata["orig_w"] = orig_w_val

                downscale_duration_msg =f"Input downscaling finished. Time: {format_time(time.time() - downscale_stage_start_time)}"
                status_log .append (downscale_duration_msg )
                logger .info (downscale_duration_msg )
                current_overall_progress =downscale_progress_start +stage_weights ["downscale"]
                progress (current_overall_progress ,desc ="Downscaling complete.")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,downscale_status_msg
            else :
                 current_overall_progress +=stage_weights ["downscale"]
        else :
            if stage_weights ["downscale"]>0 :
                current_overall_progress +=stage_weights ["downscale"]
            upscale_factor_val =upscale_factor_slider
            final_h_val =int (round (orig_h_val *upscale_factor_val /2 )*2 )
            final_w_val =int (round (orig_w_val *upscale_factor_val /2 )*2 )
            
            params_for_metadata["upscale_factor"] = upscale_factor_val
            params_for_metadata["final_h"] = final_h_val
            params_for_metadata["final_w"] = final_w_val

            direct_upscale_msg =f"Direct upscale: {upscale_factor_val:.2f}x. Target output: {final_w_val}x{final_h_val}"
            status_log .append (direct_upscale_msg )
            logger .info (direct_upscale_msg )

        yield None ,"\n".join (status_log ),last_chunk_video_path ,direct_upscale_msg

        scene_video_paths =[]
        scenes_temp_dir =None
        if enable_scene_split :
            scene_split_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Splitting video into scenes...")

            scene_split_params ={
            'split_mode':scene_split_mode ,
            'min_scene_len':scene_min_scene_len ,
            'drop_short_scenes':scene_drop_short ,
            'merge_last_scene':scene_merge_last ,
            'frame_skip':scene_frame_skip ,
            'threshold':scene_threshold ,
            'min_content_val':scene_min_content_val ,
            'frame_window':scene_frame_window ,
            'weights':[1.0 ,1.0 ,1.0 ,0.0 ],
            'copy_streams':scene_copy_streams ,
            'use_mkvmerge':scene_use_mkvmerge ,
            'rate_factor':scene_rate_factor ,
            'preset':scene_preset ,
            'quiet_ffmpeg':scene_quiet_ffmpeg ,
            'show_progress':True ,
            'manual_split_type':scene_manual_split_type ,
            'manual_split_value':scene_manual_split_value
            }

            def scene_progress_callback (progress_val ,desc ):
                current_scene_progress =scene_split_progress_start +(progress_val *stage_weights ["scene_split"])
                progress (current_scene_progress ,desc =desc )

            try :
                scene_video_paths =util_split_video_into_scenes (
                current_input_video_for_frames ,
                temp_dir ,
                scene_split_params ,
                scene_progress_callback ,
                logger =logger
                )
                scenes_temp_dir =os .path .join (temp_dir ,"scenes")

                scene_split_msg =f"Video split into {len(scene_video_paths)} scenes"
                status_log .append (scene_split_msg )
                logger .info (scene_split_msg )

                current_overall_progress =scene_split_progress_start +stage_weights ["scene_split"]
                progress (current_overall_progress ,desc =f"Scene splitting complete: {len(scene_video_paths)} scenes")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_split_msg

            except Exception as e :
                logger .error (f"Scene splitting failed: {e}")
                status_log .append (f"Scene splitting failed: {e}")
                yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene splitting failed: {e}"
                raise gr .Error (f"Scene splitting failed: {e}")
        else :
            current_overall_progress +=stage_weights ["scene_split"]

        yield None ,"\n".join (status_log ),last_chunk_video_path ,"Scene splitting complete"

        model_load_progress_start =current_overall_progress
        progress (current_overall_progress ,desc ="Loading STAR model...")
        star_model_load_start_time =time .time ()
        model_cfg =EasyDict ()
        model_cfg .model_path =model_file_path

        model_device =torch .device (util_get_gpu_device (logger =logger ))if torch .cuda .is_available ()else torch .device ('cpu')
        star_model =VideoToVideo_sr (model_cfg ,device =model_device )
        model_load_msg =f"STAR model loaded on device {model_device}. Time: {format_time(time.time() - star_model_load_start_time)}"
        status_log .append (model_load_msg )
        logger .info (model_load_msg )
        current_overall_progress =model_load_progress_start +stage_weights ["model_load"]
        progress (current_overall_progress ,desc ="STAR model loaded.")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,model_load_msg
        
        input_fps_val = 30.0 # Default, will be updated
        if not enable_scene_split :
            frame_extract_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Extracting frames...")
            frame_extraction_start_time =time .time ()
            frame_count ,input_fps_val ,frame_files =util_extract_frames (current_input_video_for_frames ,input_frames_dir ,logger =logger )
            params_for_metadata["input_fps"] = input_fps_val # Update for non-scene split

            frame_extract_msg =f"Extracted {frame_count} frames at {input_fps_val:.2f} FPS. Time: {format_time(time.time() - frame_extraction_start_time)}"
            status_log .append (frame_extract_msg )
            logger .info (frame_extract_msg )
            current_overall_progress =frame_extract_progress_start +stage_weights ["extract_frames"]
            progress (current_overall_progress ,desc =f"Extracted {frame_count} frames.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,frame_extract_msg
        else :
            current_overall_progress +=stage_weights ["extract_frames"] # Allocate progress even if skipping direct extraction here

        if save_frames and not enable_scene_split and input_frames_dir and input_frames_permanent_save_path :
            copy_input_frames_progress_start =current_overall_progress
            copy_input_frames_start_time =time .time ()
            # frame_count should be defined if not enable_scene_split
            copy_input_msg =f"Copying {frame_count if 'frame_count' in locals() else 'N/A'} input frames to permanent storage: {input_frames_permanent_save_path}"
            status_log .append (copy_input_msg )
            logger .info (copy_input_msg )
            progress (current_overall_progress ,desc ="Copying input frames...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,copy_input_msg

            frames_copied_count =0
            # Ensure frame_count is available
            num_frames_to_copy_input = frame_count if 'frame_count' in locals() and frame_count is not None else len(os.listdir(input_frames_dir))

            for frame_file in os .listdir (input_frames_dir ):
                shutil .copy2 (os .path .join (input_frames_dir ,frame_file ),os .path .join (input_frames_permanent_save_path ,frame_file ))
                frames_copied_count +=1
                if frames_copied_count %50 ==0 or frames_copied_count ==num_frames_to_copy_input :
                    loop_progress_frac =frames_copied_count /num_frames_to_copy_input if num_frames_to_copy_input >0 else 1.0
                    current_overall_progress =copy_input_frames_progress_start +(loop_progress_frac *stage_weights ["copy_input_frames"])
                    progress (current_overall_progress ,desc =f"Copying input frames: {frames_copied_count}/{num_frames_to_copy_input}")

            copied_input_msg =f"Input frames copied. Time: {format_time(time.time() - copy_input_frames_start_time)}"
            status_log .append (copied_input_msg )
            logger .info (copied_input_msg )

            current_overall_progress =copy_input_frames_progress_start +stage_weights ["copy_input_frames"]
            progress (current_overall_progress ,desc ="Input frames copied.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,copied_input_msg
        else :
             current_overall_progress +=stage_weights ["copy_input_frames"]

        upscaling_loop_progress_start =current_overall_progress
        progress (current_overall_progress ,desc ="Preparing for upscaling...")
        total_noise_levels =900

        upscaling_loop_start_time =time .time ()

        gpu_device =util_get_gpu_device (logger =logger )

        # Metadata_params_base for scenes: We use a copy of the main params_for_metadata
        # It already contains most necessary fields. Scene-specific FPS will be handled by process_single_scene.
        scene_metadata_base_params = params_for_metadata.copy() if enable_scene_split else None

        if enable_scene_split and scene_video_paths :
            processed_scene_videos =[]
            total_scenes =len (scene_video_paths )

            for scene_idx ,scene_video_path in enumerate (scene_video_paths ):
                scene_progress_start =upscaling_loop_progress_start +(scene_idx /total_scenes )*stage_weights ["upscaling_loop"]

                def scene_upscale_progress_callback (progress_val ,desc ):
                    current_scene_progress =scene_progress_start +(progress_val /total_scenes )*stage_weights ["upscaling_loop"]
                    progress (current_scene_progress ,desc =desc )

                try :
                    processed_scene_video ,scene_frame_count_ret ,scene_fps_ret =process_single_scene (
                    scene_video_path ,scene_idx ,total_scenes ,temp_dir ,star_model ,
                    final_prompt ,upscale_factor_val ,final_h_val ,final_w_val ,ui_total_diffusion_steps ,
                    solver_mode ,cfg_scale ,max_chunk_len ,vae_chunk ,color_fix_method ,
                    enable_tiling ,tile_size ,tile_overlap ,enable_sliding_window ,window_size ,window_step ,
                    save_frames ,frames_output_subfolder ,scene_upscale_progress_callback ,

                    enable_auto_caption_per_scene =enable_auto_caption_per_scene ,
                    cogvlm_quant =actual_cogvlm_quant_val ,
                    cogvlm_unload =cogvlm_unload ,
                    progress =progress ,save_chunks =save_chunks ,
                    chunks_permanent_save_path =chunks_permanent_save_path , # This might need to be scene specific if saved inside scene folder
                    ffmpeg_preset =ffmpeg_preset ,
                    ffmpeg_quality_value =ffmpeg_quality_value ,
                    ffmpeg_use_gpu =ffmpeg_use_gpu ,
                    save_metadata =save_metadata ,
                    metadata_params_base =scene_metadata_base_params # Pass the prepared base dict
                    )
                    processed_scene_videos .append (processed_scene_video )

                    scene_complete_msg =f"Scene {scene_idx+1}/{total_scenes} processing complete"
                    status_log .append (scene_complete_msg )
                    logger .info (scene_complete_msg )
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_complete_msg

                except Exception as e :
                    logger .error (f"Error processing scene {scene_idx+1}: {e}")
                    status_log .append (f"Error processing scene {scene_idx+1}: {e}")
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Scene {scene_idx+1} processing failed: {e}"
                    raise gr .Error (f"Scene {scene_idx+1} processing failed: {e}")

            current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
            scene_merge_progress_start =current_overall_progress
            progress (current_overall_progress ,desc ="Merging processed scenes...")

            silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
            # For merging, use the main FFmpeg settings from UI, as this is a final step for scene-split video
            util_merge_scene_videos (processed_scene_videos ,silent_upscaled_video_path ,temp_dir ,
            ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )

            current_overall_progress =scene_merge_progress_start +stage_weights ["scene_merge"]
            progress (current_overall_progress ,desc ="Scene merging complete")

            scene_merge_msg =f"Successfully merged {len(processed_scene_videos)} processed scenes"
            status_log .append (scene_merge_msg )
            logger .info (scene_merge_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,scene_merge_msg
            # Note: input_fps for the final merged video is tricky. FFmpeg usually handles it.
            # If all scenes had same FPS, it's that. If varied, ffprobe on merged video might be needed for exact final FPS.
            # For metadata, the scene-specific FPS are more relevant if scene splitting was used.

        else : # Not enable_scene_split: upscaling loop for the whole video
            # Ensure frame_count, input_fps_val, frame_files are defined from earlier frame extraction
            if 'frame_count' not in locals() or 'input_fps_val' not in locals() or 'frame_files' not in locals():
                # This case should ideally not happen if enable_scene_split is false, as frames are extracted above.
                # Fallback extraction if somehow missed (e.g. if logic changes)
                logger.warning("Re-extracting frames as they were not found before non-scene-split upscaling loop.")
                frame_count ,input_fps_val ,frame_files =util_extract_frames (current_input_video_for_frames ,input_frames_dir ,logger =logger )
                params_for_metadata["input_fps"] = input_fps_val


            all_lr_frames_bgr_for_preprocess =[]
            for frame_filename in frame_files :
                frame_lr_bgr =cv2 .imread (os .path .join (input_frames_dir ,frame_filename ))
                if frame_lr_bgr is None :
                    logger .error (f"Could not read frame {frame_filename} from {input_frames_dir}. Skipping.")
                    # Create a placeholder black frame if needed, or ensure preprocess handles None
                    placeholder_h = params_for_metadata["orig_h"] if params_for_metadata["orig_h"] else 256
                    placeholder_w = params_for_metadata["orig_w"] if params_for_metadata["orig_w"] else 256
                    all_lr_frames_bgr_for_preprocess .append (np .zeros ((placeholder_h ,placeholder_w ,3 ),dtype =np .uint8 ))
                    continue
                all_lr_frames_bgr_for_preprocess .append (frame_lr_bgr )

            if len (all_lr_frames_bgr_for_preprocess )!=frame_count :
                 logger .warning (f"Mismatch in frame count and loaded LR frames for colorfix: {len(all_lr_frames_bgr_for_preprocess)} vs {frame_count}")


            # Tiling / Sliding Window / Normal Chunking for non-scene_split mode
            # (This extensive logic is assumed to be the same as original, omitted for brevity)
            # ...

            if not enable_scene_split and enable_tiling :
                loop_name ="Tiling Process"
                tiling_status_msg =f"Tiling enabled: Tile Size={tile_size}, Overlap={tile_overlap}. Processing {len(frame_files)} frames."
                status_log .append (tiling_status_msg )
                logger .info (tiling_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,tiling_status_msg

                total_frames_to_tile =len (frame_files )
                frame_tqdm_iterator =progress .tqdm (enumerate (frame_files ),total =total_frames_to_tile ,desc =f"{loop_name} - Initializing...")

                for i ,frame_filename in frame_tqdm_iterator :
                    frame_proc_start_time =time .time ()
                    frame_lr_bgr =cv2 .imread (os .path .join (input_frames_dir ,frame_filename ))
                    if frame_lr_bgr is None :
                        logger .warning (f"Skipping frame {frame_filename} due to read error during tiling.")
                        placeholder_path =os .path .join (input_frames_dir ,frame_filename )
                        if os .path .exists (placeholder_path ):
                            shutil .copy2 (placeholder_path ,os .path .join (output_frames_dir ,frame_filename ))
                        continue

                    single_lr_frame_tensor_norm =preprocess ([frame_lr_bgr ])
                    spliter =ImageSpliterTh (single_lr_frame_tensor_norm ,int (tile_size ),int (tile_overlap ),sf =upscale_factor_val ) # Use upscale_factor_val

                    try :
                        num_patches_this_frame =len (list (spliter ))
                        # Re-initialize spliter as iterating consumes it
                        spliter =ImageSpliterTh (single_lr_frame_tensor_norm ,int (tile_size ),int (tile_overlap ),sf =upscale_factor_val)
                    except :
                        num_patches_this_frame =getattr (spliter ,'num_patches','N/A')

                    patch_tqdm_iterator =progress .tqdm (enumerate (spliter ),total =num_patches_this_frame if isinstance (num_patches_this_frame ,int )else None ,desc =f"Frame {i+1}/{total_frames_to_tile} Patches - Initializing...")

                    for patch_idx ,(patch_lr_tensor_norm ,patch_coords )in patch_tqdm_iterator :
                        patch_proc_start_time_local =time .time () # Renamed from patch_proc_start_time
                        patch_lr_video_data =patch_lr_tensor_norm

                        patch_pre_data ={'video_data':patch_lr_video_data ,'y':final_prompt ,
                        'target_res':(int (round (patch_lr_tensor_norm .shape [-2 ]*upscale_factor_val )), # Use upscale_factor_val
                        int (round (patch_lr_tensor_norm .shape [-1 ]*upscale_factor_val )))} # Use upscale_factor_val
                        patch_data_tensor_cuda =collate_fn (patch_pre_data ,gpu_device )

                        def diffusion_callback_for_patch (step_cb ,total_steps_cb ): # Renamed args
                            current_time =time .time ()
                            step_duration =current_time -callback_step_timer ['last_time']
                            callback_step_timer ['last_time']=current_time

                            current_patch_desc =f"Frame {i+1}/{total_frames_to_tile}, Patch {patch_idx+1}/{num_patches_this_frame}"

                            tqdm_step_info =f"{step_duration:.2f}s/it ({step_cb}/{total_steps_cb})"if step_duration >0.001 else f"{step_cb}/{total_steps_cb}"
                            patch_tqdm_iterator .desc =f"{current_patch_desc} - Diffusion: {tqdm_step_info}"

                            log_step_info =f"{step_duration:.2f} second / it"if step_duration >0.001 else f"step {step_cb}/{total_steps_cb}"
                            logger .info (f"    ↳ {loop_name} - {current_patch_desc} - {log_step_info} - so see step speed")

                        star_model_call_patch_start_time =time .time ()
                        callback_step_timer ={'last_time':time .time ()}
                        with torch .no_grad ():
                            patch_sr_tensor_bcthw =star_model .test (
                            patch_data_tensor_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,solver_mode =solver_mode ,
                            guide_scale =cfg_scale ,max_chunk_len =1 ,vae_decoder_chunk_size =1 ,
                            progress_callback =diffusion_callback_for_patch
                            )

                        star_model_call_patch_duration =time .time ()-star_model_call_patch_start_time
                        logger .info (f"{loop_name} - Frame {i+1}/{total_frames_to_tile}, Patch {patch_idx+1}/{num_patches_this_frame}: Finished STAR model processing. Duration: {format_time(star_model_call_patch_duration)}")
                        patch_sr_frames_uint8 =tensor2vid (patch_sr_tensor_bcthw )

                        if color_fix_method !='None':
                            if color_fix_method =='AdaIN':
                                patch_sr_frames_uint8 =adain_color_fix (patch_sr_frames_uint8 ,patch_lr_video_data )
                            elif color_fix_method =='Wavelet':
                                patch_sr_frames_uint8 =wavelet_color_fix (patch_sr_frames_uint8 ,patch_lr_video_data )

                        single_patch_frame_hwc =patch_sr_frames_uint8 [0 ]
                        result_patch_chw_01 =single_patch_frame_hwc .permute (2 ,0 ,1 ).float ()/255.0

                        spliter .update_gaussian (result_patch_chw_01 .unsqueeze (0 ),patch_coords )

                        del patch_data_tensor_cuda ,patch_sr_tensor_bcthw ,patch_sr_frames_uint8
                        if torch .cuda .is_available ():
                            torch .cuda .empty_cache ()

                        patch_duration =time .time ()-patch_proc_start_time_local # Use renamed var
                        patch_tqdm_iterator .desc =f"Frame {i+1} Patch {patch_idx+1}/{num_patches_this_frame} (took {patch_duration:.2f}s)"


                final_frame_tensor_chw =spliter .gather ()
                final_frame_np_hwc_uint8 =(final_frame_tensor_chw .squeeze (0 ).permute (1 ,2 ,0 ).clamp (0 ,1 ).cpu ().numpy ()*255 ).astype (np .uint8 )
                final_frame_bgr =cv2 .cvtColor (final_frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                cv2 .imwrite (os .path .join (output_frames_dir ,frame_filename ),final_frame_bgr )

                frames_processed_tile =i +1
                current_tile_loop_time =time .time ()-upscaling_loop_start_time
                avg_time_per_frame_tile =current_tile_loop_time /frames_processed_tile if frames_processed_tile > 0 else 0
                eta_seconds_tile =(total_frames_to_tile -frames_processed_tile )*avg_time_per_frame_tile if frames_processed_tile <total_frames_to_tile and avg_time_per_frame_tile > 0 else 0
                speed_tile =1 /avg_time_per_frame_tile if avg_time_per_frame_tile >0 else 0

                frame_tqdm_desc =f"{loop_name}: {frames_processed_tile}/{total_frames_to_tile} frames | ETA: {format_time(eta_seconds_tile)} | Speed: {speed_tile:.2f} f/s"
                frame_tqdm_iterator .desc =frame_tqdm_desc

                detailed_frame_msg =f"{frame_tqdm_desc} | Current frame processed in {time.time() - frame_proc_start_time:.2f}s. Total elapsed: {format_time(current_tile_loop_time)}"
                status_log .append (detailed_frame_msg )
                logger .info (detailed_frame_msg )

                loop_progress_frac =frames_processed_tile /total_frames_to_tile if total_frames_to_tile >0 else 1.0
                current_overall_progress =upscaling_loop_progress_start +(loop_progress_frac *stage_weights ["upscaling_loop"])
                progress (current_overall_progress ,desc =frame_tqdm_iterator .desc )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Tiling frame {frames_processed_tile}/{total_frames_to_tile} processed"

            elif not enable_scene_split and enable_sliding_window :
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

                sliding_tqdm_iterator =progress .tqdm (enumerate (window_indices_to_process ),total =total_windows_to_process ,desc =f"{loop_name} - Initializing...")

                # Initialize chunk tracking for sliding window
                saved_chunks = set()  # Track which chunks have been saved
                processed_frames_tracker = [False] * frame_count  # Track which frames have been processed

                # Helper functions for chunk mapping
                def map_window_to_chunks(start_idx, end_idx, max_chunk_len):
                    """Map a window frame range to affected chunk indices"""
                    first_chunk = start_idx // max_chunk_len
                    last_chunk = (end_idx - 1) // max_chunk_len
                    return list(range(first_chunk, last_chunk + 1))

                def get_chunk_frame_range(chunk_idx, max_chunk_len, total_frames):
                    """Get the frame range for a specific chunk"""
                    start_frame = chunk_idx * max_chunk_len
                    end_frame = min((chunk_idx + 1) * max_chunk_len, total_frames)
                    return start_frame, end_frame

                def is_chunk_complete(chunk_idx, processed_frames_tracker, max_chunk_len, total_frames):
                    """Check if all frames in a chunk have been processed"""
                    start_frame, end_frame = get_chunk_frame_range(chunk_idx, max_chunk_len, total_frames)
                    for frame_idx in range(start_frame, end_frame):
                        if not processed_frames_tracker[frame_idx]:
                            return False
                    return True

                for window_iter_idx ,i_start_idx in enumerate (sliding_tqdm_iterator ):
                    # window_proc_start_time =time .time () # Original var, seems unused for ETA calculation in this loop
                    start_idx =i_start_idx
                    end_idx =min (i_start_idx +effective_window_size ,frame_count )
                    current_window_len =end_idx -start_idx

                    if current_window_len ==0 :
                        continue

                    is_last_window_iteration =(window_iter_idx ==total_windows_to_process -1 )
                    if is_last_window_iteration and current_window_len <effective_window_size and frame_count >=effective_window_size :
                        start_idx =max (0 ,frame_count -effective_window_size )
                        end_idx =frame_count
                        current_window_len =end_idx -start_idx

                    window_lr_frames_bgr =all_lr_frames_bgr_for_preprocess [start_idx :end_idx ]
                    if not window_lr_frames_bgr :
                        continue

                    window_lr_video_data =preprocess (window_lr_frames_bgr )

                    window_pre_data ={'video_data':window_lr_video_data ,'y':final_prompt ,
                    'target_res':(final_h_val ,final_w_val )} # Use final_h_val, final_w_val
                    window_data_cuda =collate_fn (window_pre_data ,gpu_device )

                    current_window_display_num =window_iter_idx +1

                    def diffusion_callback_for_window (step_cb ,total_steps_cb ): # Renamed args
                        current_time =time .time ()
                        step_duration =current_time -callback_step_timer ['last_time']
                        callback_step_timer ['last_time']=current_time

                        _current_slide_loop_time_cb =time .time ()-upscaling_loop_start_time
                        _avg_time_per_window_cb =_current_slide_loop_time_cb /current_window_display_num if current_window_display_num >0 else 0
                        _eta_seconds_slide_cb =(total_windows_to_process -current_window_display_num )*_avg_time_per_window_cb if current_window_display_num <total_windows_to_process and _avg_time_per_window_cb >0 else 0
                        _speed_slide_cb =1 /_avg_time_per_window_cb if _avg_time_per_window_cb >0 else 0

                        base_desc_win =f"{loop_name}: {current_window_display_num}/{total_windows_to_process} windows (frames {start_idx}-{end_idx-1}) | ETA: {format_time(_eta_seconds_slide_cb)} | Speed: {_speed_slide_cb:.2f} w/s"

                        tqdm_step_info =f"{step_duration:.2f}s/it ({step_cb}/{total_steps_cb})"if step_duration >0.001 else f"{step_cb}/{total_steps_cb}"
                        sliding_tqdm_iterator .desc =f"{base_desc_win} - Diffusion: {tqdm_step_info}"

                        log_step_info =f"{step_duration:.2f} second / it"if step_duration >0.001 else f"step {step_cb}/{total_steps_cb}"
                        logger .info (f"    ↳ {loop_name} - Window {current_window_display_num}/{total_windows_to_process} (frames {start_idx}-{end_idx-1}) - {log_step_info} - so see step speed")


                    star_model_call_window_start_time =time .time ()
                    callback_step_timer ={'last_time':time .time ()}
                    with torch .no_grad ():
                        window_sr_tensor_bcthw =star_model .test (
                        window_data_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,solver_mode =solver_mode ,
                        guide_scale =cfg_scale ,max_chunk_len =current_window_len ,vae_decoder_chunk_size =min (vae_chunk ,current_window_len ),
                        progress_callback =diffusion_callback_for_window
                        )
                    star_model_call_window_duration =time .time ()-star_model_call_window_start_time
                    logger .info (f"{loop_name} - Window {current_window_display_num}/{total_windows_to_process}: Finished STAR model processing. Duration: {format_time(star_model_call_window_duration)}")
                    window_sr_frames_uint8 =tensor2vid (window_sr_tensor_bcthw )

                    if color_fix_method !='None':
                        if color_fix_method =='AdaIN':
                            window_sr_frames_uint8 =adain_color_fix (window_sr_frames_uint8 ,window_lr_video_data )
                        elif color_fix_method =='Wavelet':
                            window_sr_frames_uint8 =wavelet_color_fix (window_sr_frames_uint8 ,window_lr_video_data )

                    save_from_start_offset_local =0
                    save_to_end_offset_local =current_window_len
                    
                    # Use simple overlap logic that works correctly for all cases
                    if total_windows_to_process >1:
                        overlap_amount =effective_window_size -effective_window_step
                        if overlap_amount >0 :
                            if window_iter_idx ==0 :
                                # First window: save from start to end minus half overlap
                                save_to_end_offset_local =effective_window_size -(overlap_amount //2 )
                            elif is_last_window_iteration :
                                # Last window: save from half overlap to end
                                save_from_start_offset_local =(overlap_amount //2 )
                            else :
                                # Middle windows: save only the center portion, avoiding overlap zones
                                save_from_start_offset_local =(overlap_amount //2 )
                                save_to_end_offset_local =effective_window_size -(overlap_amount //2 )
                        
                        # Ensure bounds are valid
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
                            processed_frames_tracker[k_global] = True

                    # Check and save complete chunks
                    affected_chunks = map_window_to_chunks(start_idx, end_idx, max_chunk_len)
                    for chunk_idx in affected_chunks:
                        if chunk_idx not in saved_chunks and is_chunk_complete(chunk_idx, processed_frames_tracker, max_chunk_len, frame_count):
                            if save_chunks and chunks_permanent_save_path:
                                start_frame, end_frame = get_chunk_frame_range(chunk_idx, max_chunk_len, frame_count)
                                current_chunk_display_num = chunk_idx + 1
                                
                                chunk_video_filename = f"chunk_{current_chunk_display_num:04d}.mp4"
                                chunk_video_path = os.path.join(chunks_permanent_save_path, chunk_video_filename)
                                chunk_temp_dir = os.path.join(temp_dir, f"temp_sliding_chunk_{current_chunk_display_num}")
                                os.makedirs(chunk_temp_dir, exist_ok=True)

                                for k, frame_name in enumerate(frame_files[start_frame:end_frame]):
                                    src_frame = os.path.join(output_frames_dir, frame_name)
                                    dst_frame = os.path.join(chunk_temp_dir, f"frame_{k+1:06d}.png")
                                    shutil.copy2(src_frame, dst_frame)

                                util_create_video_from_frames(chunk_temp_dir, chunk_video_path, input_fps_val,
                                    ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)

                                shutil.rmtree(chunk_temp_dir)

                                chunk_save_msg = f"Saved chunk {current_chunk_display_num}/{math.ceil(frame_count/max_chunk_len)} to: {chunk_video_path}"
                                status_log.append(chunk_save_msg)
                                logger.info(chunk_save_msg)

                                last_chunk_video_path = chunk_video_path
                                last_chunk_status = f"Chunk {current_chunk_display_num}/{math.ceil(frame_count/max_chunk_len)} (frames {start_frame+1}-{end_frame})"

                                saved_chunks.add(chunk_idx)
                                yield None, "\n".join(status_log), last_chunk_video_path, last_chunk_status

                    del window_data_cuda ,window_sr_tensor_bcthw ,window_sr_frames_uint8
                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()

                    # This progress_callback was for scenes, might need adjustment for main video sliding window.
                    # For now, relying on tqdm for main progress.
                    # if progress_callback : # This was from scene context, probably not applicable here directly
                    #     window_progress =0.3 +(window_iter_idx /total_windows_to_process )*0.5
                    #     progress_callback (window_progress ,f"Scene {scene_index+1}: Processing window {window_iter_idx+1}/{total_windows_to_process}")
                    loop_progress_frac = current_window_display_num / total_windows_to_process if total_windows_to_process > 0 else 1.0
                    current_overall_progress = upscaling_loop_progress_start + (loop_progress_frac * stage_weights["upscaling_loop"])
                    progress(current_overall_progress, desc=sliding_tqdm_iterator.desc)
                    yield None, "\n".join(status_log), last_chunk_video_path, f"Sliding window {current_window_display_num}/{total_windows_to_process} processed"


                num_missed_fallback =0
                for idx_fb ,fname_fb in enumerate (frame_files ):
                    if processed_frame_filenames [idx_fb ]is None :
                        num_missed_fallback +=1
                        logger .warning (f"Frame {fname_fb} (index {idx_fb}) was not processed by sliding window, copying LR frame.")
                        lr_frame_path =os .path .join (input_frames_dir ,fname_fb )
                        if os .path .exists (lr_frame_path ):
                            shutil .copy2 (lr_frame_path ,os .path .join (output_frames_dir ,fname_fb ))
                            processed_frames_tracker[idx_fb] = True
                        else :
                            logger .error (f"LR frame {lr_frame_path} not found for fallback copy.")
                if num_missed_fallback >0 :
                    missed_msg =f"{loop_name} - Copied {num_missed_fallback} LR frames as fallback for unprocessed frames."
                    status_log .append (missed_msg )
                    logger .info (missed_msg )
                    yield None ,"\n".join (status_log ),last_chunk_video_path ,missed_msg

                # Save any remaining incomplete chunks after all windows are processed
                num_total_chunks = math.ceil(frame_count / max_chunk_len) if max_chunk_len > 0 else 1
                for chunk_idx in range(num_total_chunks):
                    if chunk_idx not in saved_chunks and save_chunks and chunks_permanent_save_path:
                        start_frame, end_frame = get_chunk_frame_range(chunk_idx, max_chunk_len, frame_count)
                        current_chunk_display_num = chunk_idx + 1
                        
                        chunk_video_filename = f"chunk_{current_chunk_display_num:04d}.mp4"
                        chunk_video_path = os.path.join(chunks_permanent_save_path, chunk_video_filename)
                        chunk_temp_dir = os.path.join(temp_dir, f"temp_sliding_remaining_chunk_{current_chunk_display_num}")
                        os.makedirs(chunk_temp_dir, exist_ok=True)

                        frames_for_chunk = []
                        for k, frame_name in enumerate(frame_files[start_frame:end_frame]):
                            src_frame = os.path.join(output_frames_dir, frame_name)
                            dst_frame = os.path.join(chunk_temp_dir, f"frame_{k+1:06d}.png")
                            if os.path.exists(src_frame):
                                shutil.copy2(src_frame, dst_frame)
                                frames_for_chunk.append(dst_frame)
                            else:
                                logger.warning(f"Frame {src_frame} not found for remaining chunk {current_chunk_display_num}")

                        if frames_for_chunk:
                            util_create_video_from_frames(chunk_temp_dir, chunk_video_path, input_fps_val,
                                ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)

                            chunk_save_msg = f"Saved remaining chunk {current_chunk_display_num}/{num_total_chunks} to: {chunk_video_path}"
                            status_log.append(chunk_save_msg)
                            logger.info(chunk_save_msg)

                            last_chunk_video_path = chunk_video_path
                            last_chunk_status = f"Remaining Chunk {current_chunk_display_num}/{num_total_chunks} (frames {start_frame+1}-{end_frame})"

                            saved_chunks.add(chunk_idx)
                            yield None, "\n".join(status_log), last_chunk_video_path, last_chunk_status
                        else:
                            logger.warning(f"No frames for remaining chunk {current_chunk_display_num}, video not created.")

                        shutil.rmtree(chunk_temp_dir)

            elif not enable_scene_split : # Normal Chunked Processing
                loop_name ="Chunked Processing"
                chunk_status_msg ="Normal chunked processing."
                status_log .append (chunk_status_msg )
                logger .info (chunk_status_msg )
                yield None ,"\n".join (status_log ),last_chunk_video_path ,chunk_status_msg

                num_chunks =math .ceil (frame_count /max_chunk_len )if max_chunk_len >0 else (1 if frame_count >0 else 0 )
                if num_chunks ==0 and frame_count >0 :num_chunks =1

                chunk_tqdm_iterator =progress .tqdm (range (num_chunks ),total =num_chunks ,desc =f"{loop_name} - Initializing...")

                for i_chunk_idx in chunk_tqdm_iterator :
                    # chunk_proc_start_time =time .time () # Original var, seems unused for ETA
                    start_idx =i_chunk_idx *max_chunk_len
                    end_idx =min ((i_chunk_idx +1 )*max_chunk_len ,frame_count )
                    current_chunk_len =end_idx -start_idx
                    if current_chunk_len ==0 :
                        continue

                    if end_idx >len (all_lr_frames_bgr_for_preprocess )or start_idx <0 :
                         logger .error (f"Chunk range {start_idx}-{end_idx} is invalid for LR frames {len(all_lr_frames_bgr_for_preprocess)}")
                         continue

                    chunk_lr_frames_bgr =all_lr_frames_bgr_for_preprocess [start_idx :end_idx ]
                    if not chunk_lr_frames_bgr :
                        continue

                    chunk_lr_video_data =preprocess (chunk_lr_frames_bgr )

                    chunk_pre_data ={'video_data':chunk_lr_video_data ,'y':final_prompt ,
                    'target_res':(final_h_val ,final_w_val )} # Use final_h_val, final_w_val
                    chunk_data_cuda =collate_fn (chunk_pre_data ,gpu_device )

                    current_chunk_display_num =i_chunk_idx +1

                    def diffusion_callback_for_chunk (step_cb ,total_steps_cb ): # Renamed args
                        current_time =time .time ()
                        step_duration =current_time -callback_step_timer ['last_time']
                        callback_step_timer ['last_time']=current_time

                        _current_chunk_loop_time_cb =time .time ()-upscaling_loop_start_time
                        _avg_time_per_chunk_cb =_current_chunk_loop_time_cb /current_chunk_display_num if current_chunk_display_num >0 else 0
                        _eta_seconds_chunk_cb =(num_chunks -current_chunk_display_num )*_avg_time_per_chunk_cb if current_chunk_display_num <num_chunks and _avg_time_per_chunk_cb >0 else 0
                        _speed_chunk_cb =1 /_avg_time_per_chunk_cb if _avg_time_per_chunk_cb >0 else 0

                        base_desc_chunk =f"{loop_name}: {current_chunk_display_num}/{num_chunks} chunks | ETA: {format_time(_eta_seconds_chunk_cb)} | Speed: {_speed_chunk_cb:.2f} ch/s"

                        tqdm_step_info =f"{step_duration:.2f}s/it ({step_cb}/{total_steps_cb})"if step_duration >0.001 else f"{step_cb}/{total_steps_cb}"
                        chunk_tqdm_iterator .desc =f"{base_desc_chunk} - Diffusion: {tqdm_step_info}"

                        log_step_info =f"{step_duration:.2f} second / it"if step_duration >0.001 else f"step {step_cb}/{total_steps_cb}"
                        logger .info (f"    ↳ {loop_name} - Chunk {current_chunk_display_num}/{num_chunks} (frames {start_idx}-{end_idx-1}) - {log_step_info} - so see step speed")

                    star_model_call_chunk_start_time =time .time ()
                    callback_step_timer ={'last_time':time .time ()}
                    with torch .no_grad ():
                        chunk_sr_tensor_bcthw =star_model .test (
                        chunk_data_cuda ,total_noise_levels ,steps =ui_total_diffusion_steps ,solver_mode =solver_mode ,
                        guide_scale =cfg_scale ,max_chunk_len =current_chunk_len ,vae_decoder_chunk_size =min (vae_chunk ,current_chunk_len ),
                        progress_callback =diffusion_callback_for_chunk
                        )
                    star_model_call_chunk_duration =time .time ()-star_model_call_chunk_start_time
                    logger .info (f"{loop_name} - Chunk {current_chunk_display_num}/{num_chunks}: Finished STAR model processing. Duration: {format_time(star_model_call_chunk_duration)}")
                    chunk_sr_frames_uint8 =tensor2vid (chunk_sr_tensor_bcthw )

                    if color_fix_method !='None':
                        if color_fix_method =='AdaIN':
                            chunk_sr_frames_uint8 =adain_color_fix (chunk_sr_frames_uint8 ,chunk_lr_video_data )
                        elif color_fix_method =='Wavelet':
                            chunk_sr_frames_uint8 =wavelet_color_fix (chunk_sr_frames_uint8 ,chunk_lr_video_data )

                    for k ,frame_name in enumerate (frame_files [start_idx :end_idx ]):
                        frame_np_hwc_uint8 =chunk_sr_frames_uint8 [k ].cpu ().numpy ()
                        frame_bgr =cv2 .cvtColor (frame_np_hwc_uint8 ,cv2 .COLOR_RGB2BGR )
                        cv2 .imwrite (os .path .join (output_frames_dir ,frame_name ),frame_bgr )

                    if save_chunks and chunks_permanent_save_path :
                        chunk_video_filename =f"chunk_{current_chunk_display_num:04d}.mp4"
                        chunk_video_path =os .path .join (chunks_permanent_save_path ,chunk_video_filename )
                        chunk_temp_dir =os .path .join (temp_dir ,f"temp_chunk_{current_chunk_display_num}")
                        os .makedirs (chunk_temp_dir ,exist_ok =True )

                        for k ,frame_name in enumerate (frame_files [start_idx :end_idx ]):
                            src_frame =os .path .join (output_frames_dir ,frame_name )
                            dst_frame =os .path .join (chunk_temp_dir ,f"frame_{k+1:06d}.png")
                            shutil .copy2 (src_frame ,dst_frame )

                        util_create_video_from_frames (chunk_temp_dir ,chunk_video_path ,input_fps_val , # Use input_fps_val
                        ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )

                        shutil .rmtree (chunk_temp_dir )

                        chunk_save_msg =f"Saved chunk {current_chunk_display_num}/{num_chunks} to: {chunk_video_path}"
                        status_log .append (chunk_save_msg )
                        logger .info (chunk_save_msg )

                        last_chunk_video_path =chunk_video_path
                        last_chunk_status =f"Chunk {current_chunk_display_num}/{num_chunks} (frames {start_idx+1}-{end_idx})"

                        yield None ,"\n".join (status_log ),last_chunk_video_path ,last_chunk_status

                    if save_metadata: # Save metadata after each chunk for non-scene_split
                        status_info_for_chunk_meta = {
                            "current_chunk": current_chunk_display_num,
                            "total_chunks": num_chunks,
                            "overall_process_start_time": overall_process_start_time,
                            "chunk_frame_range": (start_idx + 1, end_idx)  # 1-indexed frame range for this chunk
                        }
                        try:
                            metadata_handler.save_metadata(
                                save_flag=True, # Already checked save_metadata
                                output_dir=main_output_dir,
                                base_filename_no_ext=base_output_filename_no_ext,
                                params_dict=params_for_metadata, # Pass the main, updated params dict
                                status_info=status_info_for_chunk_meta,
                                logger=logger
                            )
                        except Exception as e_meta:
                            logger.warning(f"Failed to save/update metadata after chunk {current_chunk_display_num}: {e_meta}")


                    del chunk_data_cuda ,chunk_sr_tensor_bcthw ,chunk_sr_frames_uint8
                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()

                    # This progress_callback was for scenes, might need adjustment for main video chunking.
                    # For now, relying on tqdm for main progress.
                    # if progress_callback : # This was from scene context
                    #     chunk_progress =0.3 +(i_chunk_idx /num_chunks )*0.5
                    #     progress_callback (chunk_progress ,f"Scene {scene_index+1}: Processing chunk {current_chunk_display_num}/{num_chunks}")
                    loop_progress_frac = current_chunk_display_num / num_chunks if num_chunks > 0 else 1.0
                    current_overall_progress = upscaling_loop_progress_start + (loop_progress_frac * stage_weights["upscaling_loop"])
                    progress(current_overall_progress, desc=chunk_tqdm_iterator.desc)
                    # Yield after processing a chunk, not inside the diffusion callback for main loop
                    # yield None, "\n".join(status_log), last_chunk_video_path, f"Chunk {current_chunk_display_num}/{num_chunks} processed"


            current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
            upscaling_total_duration_msg =f"All frame upscaling operations finished. Total upscaling time: {format_time(time.time() - upscaling_loop_start_time)}"
            status_log .append (upscaling_total_duration_msg )
            logger .info (upscaling_total_duration_msg )
            progress (current_overall_progress ,desc ="Upscaling complete.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,upscaling_total_duration_msg

        # This yield was inside the non-scene_split block, should be outside if it's a general marker
        # current_overall_progress =upscaling_loop_progress_start +stage_weights ["upscaling_loop"]
        # upscaling_total_duration_msg =f"All frame upscaling operations finished. Total upscaling time: {format_time(time.time() - upscaling_loop_start_time)}"
        # status_log .append (upscaling_total_duration_msg )
        # logger .info (upscaling_total_duration_msg )
        # progress (current_overall_progress ,desc ="Upscaling complete.")
        # yield None ,"\n".join (status_log ),last_chunk_video_path ,upscaling_total_duration_msg

        initial_progress_reassembly =current_overall_progress

        if save_frames and not enable_scene_split and output_frames_dir and processed_frames_permanent_save_path :
            copy_processed_frames_start_time =time .time ()
            num_processed_frames_to_copy =len (os .listdir (output_frames_dir ))
            copy_proc_msg =f"Copying {num_processed_frames_to_copy} processed frames to permanent storage: {processed_frames_permanent_save_path}"
            status_log .append (copy_proc_msg )
            logger .info (copy_proc_msg )
            progress (current_overall_progress ,desc ="Copying processed frames...")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,copy_proc_msg

            frames_copied_count =0

            for frame_file in os .listdir (output_frames_dir ):
                shutil .copy2 (os .path .join (output_frames_dir ,frame_file ),os .path .join (processed_frames_permanent_save_path ,frame_file ))
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
            yield None ,"\n".join (status_log ),last_chunk_video_path ,copied_proc_msg
        else :
            current_overall_progress +=stage_weights ["reassembly_copy_processed"]

        if not enable_scene_split : # This silent video is created only if not scene splitting (scenes already made their videos)
            initial_progress_silent_video =current_overall_progress
            progress (current_overall_progress ,desc ="Creating silent video...")
            silent_upscaled_video_path =os .path .join (temp_dir ,"silent_upscaled_video.mp4")
            # input_fps_val should be defined from frame extraction for non-scene split
            util_create_video_from_frames (output_frames_dir ,silent_upscaled_video_path ,input_fps_val ,ffmpeg_preset ,ffmpeg_quality_value ,ffmpeg_use_gpu ,logger =logger )
            current_overall_progress =initial_progress_silent_video # silent video creation is part of audio merge stage in spirit
            # progress (current_overall_progress ,desc ="Silent video created.") # This progress update might be too quick

            silent_video_msg ="Silent upscaled video created. Merging audio..."
            status_log .append (silent_video_msg )
            logger .info (silent_video_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg
        else : # Scene splitting was enabled, silent_upscaled_video_path is the merged scene video
            silent_video_msg ="Scene-merged video ready. Merging audio..."
            status_log .append (silent_video_msg )
            logger .info (silent_video_msg )
            yield None ,"\n".join (status_log ),last_chunk_video_path ,silent_video_msg

        initial_progress_audio_merge =current_overall_progress
        audio_source_video =current_input_video_for_frames # This is either original or downscaled input
        final_output_path =output_video_path # The one determined at the start (batch or single)
        params_for_metadata["final_output_path"] = final_output_path # Ensure final path is in metadata

        if not os .path .exists (audio_source_video ):
            logger .warning (f"Audio source video '{audio_source_video}' not found. Output will be video-only.")
            if os.path.exists(silent_upscaled_video_path):
                shutil .copy2 (silent_upscaled_video_path ,final_output_path )
            else:
                logger.error(f"Silent upscaled video path {silent_upscaled_video_path} not found for copy.")
                raise gr.Error("Silent video not found for final output.")

        else :
            if os.path.exists(silent_upscaled_video_path):
                 # Remove -shortest flag to preserve full video length, use -avoid_negative_ts for robust merging
                 util_run_ffmpeg_command (f'ffmpeg -y -i "{silent_upscaled_video_path}" -i "{audio_source_video}" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? -avoid_negative_ts make_zero "{final_output_path}"',"Final Video and Audio Merge",logger =logger )
            else:
                logger.error(f"Silent upscaled video path {silent_upscaled_video_path} not found for audio merge.")
                raise gr.Error("Silent video not found for audio merge.")


        current_overall_progress =initial_progress_audio_merge +stage_weights ["reassembly_audio_merge"]
        progress (current_overall_progress ,desc ="Audio merged.")
        reassembly_done_msg =f"Video reassembly and audio merge finished."
        status_log .append (reassembly_done_msg )
        logger .info (reassembly_done_msg )

        final_save_msg =f"Upscaled video saved to: {final_output_path}"
        status_log .append (final_save_msg )
        logger .info (final_save_msg )
        # progress (1.0 ,"Finished!") # Moved after metadata

        if save_metadata :
            initial_progress_metadata =current_overall_progress
            progress (current_overall_progress ,desc ="Saving metadata...")
            metadata_save_start_time =time .time ()
            processing_time_total =time .time ()-overall_process_start_time

            final_status_info = {
                "processing_time_total": processing_time_total,
                # overall_process_start_time is not needed here as we provide total
            }
            
            # Add frame range information for the complete video
            if frame_count > 0:
                final_status_info["video_frame_range"] = (1, frame_count)  # 1-indexed total video frame range
            # Ensure all derived params in params_for_metadata are up-to-date
            # input_fps for the overall video (if not scene split) should be in params_for_metadata["input_fps"]
            # If scene split, that field might be less relevant for the *final* metadata if scenes had different FPS.
            # The scene-specific metadata files will have their own FPS.

            success ,message =metadata_handler.save_metadata(
                save_flag=True, # Already checked save_metadata
                output_dir=main_output_dir,
                base_filename_no_ext=base_output_filename_no_ext,
                params_dict=params_for_metadata, # Contains all UI and derived values
                status_info=final_status_info,   # Contains final timing
                logger=logger
            )

            if success :
                meta_saved_msg =f"Final metadata saved: {message.split(': ')[-1]}. Time to save: {format_time(time.time() - metadata_save_start_time)}"
                status_log .append (meta_saved_msg )
                logger .info (meta_saved_msg )
            else :
                status_log .append (f"Error saving final metadata: {message}")
                logger .error (message )

            current_overall_progress =initial_progress_metadata +stage_weights ["metadata"]
            progress (current_overall_progress ,desc ="Metadata saved.")
            yield None ,"\n".join (status_log ),last_chunk_video_path ,meta_saved_msg if success else message

        current_overall_progress +=stage_weights .get ("final_cleanup_buffer",0.0 ) # Small buffer for cleanup
        current_overall_progress =min (current_overall_progress ,1.0 )

        is_error =any (err_msg in status_log [-1 ]for err_msg in ["Error:","Critical Error:"])if status_log else False
        final_desc ="Finished!"
        if is_error :
            final_desc =status_log [-1 ]if status_log else "Error occurred"
            progress (current_overall_progress ,desc =final_desc ) # Show error in progress
        else :
            progress (1.0 ,desc =final_desc ) # Ensure 100% on success

        yield final_output_path ,"\n".join (status_log ),last_chunk_video_path ,"Processing complete!"

    except gr .Error as e :
        logger .error (f"A Gradio UI Error occurred: {e}",exc_info =True )
        status_log .append (f"Error: {e}")
        current_overall_progress =min (current_overall_progress +stage_weights .get ("final_cleanup_buffer",0.01 ),1.0 ) # Small progress for error
        progress (current_overall_progress ,desc =f"Error: {str(e)[:50]}")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Error: {e}"
        # raise e # Re-raising gr.Error will be caught by Gradio and displayed
    except Exception as e :
        logger .error (f"An unexpected error occurred during upscaling: {e}",exc_info =True )
        status_log .append (f"Critical Error: {e}")
        current_overall_progress =min (current_overall_progress +stage_weights .get ("final_cleanup_buffer",0.01 ),1.0 )
        progress (current_overall_progress ,desc =f"Critical Error: {str(e)[:50]}")
        yield None ,"\n".join (status_log ),last_chunk_video_path ,f"Critical Error: {e}"
        raise gr .Error (f"Upscaling failed critically: {e}") # Wrap in gr.Error for Gradio
    finally :
        if star_model is not None :
            try :
                if hasattr (star_model ,'to'):star_model .to ('cpu')
                del star_model
            except :pass

        gc .collect ()
        if torch .cuda .is_available ():torch .cuda .empty_cache ()
        logger .info ("STAR upscaling process finished and cleaned up.")

        util_cleanup_temp_dir (temp_dir ,logger =logger )

        total_process_duration =time .time ()-overall_process_start_time
        final_cleanup_msg =f"STAR upscaling process finished and cleaned up. Total processing time: {format_time(total_process_duration)}"
        logger .info (final_cleanup_msg )

        # Check if output_video_path was actually created
        output_video_exists = 'final_output_path' in locals() and final_output_path and os.path.exists(final_output_path)

        if not output_video_exists:
             if status_log and status_log [-1 ]and not status_log [-1 ].startswith ("Error:")and not status_log [-1 ].startswith ("Critical Error:"):
                no_output_msg ="Processing finished, but output video was not found or not created."
                status_log .append (no_output_msg )
                logger .warning (no_output_msg )
        
        # Clean up lock file
        if 'base_output_filename_no_ext'in locals () and base_output_filename_no_ext :
            tmp_lock_file_to_delete =os .path .join (main_output_dir if 'main_output_dir' in locals() else app_config.DEFAULT_OUTPUT_DIR ,f"{base_output_filename_no_ext}.tmp")
            if os .path .exists (tmp_lock_file_to_delete ):
                try :
                    os .remove (tmp_lock_file_to_delete )
                    logger .info (f"Successfully deleted lock file: {tmp_lock_file_to_delete}")
                except Exception as e_lock_del :
                    logger .error (f"Failed to delete lock file {tmp_lock_file_to_delete}: {e_lock_del}")
            # else : # This warning can be noisy if the process failed before lock creation
            #     logger .warning (f"Lock file {tmp_lock_file_to_delete} not found for deletion (might be normal if process failed early).")


css ="""
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdfdf; }
"""

def wrapper_split_video_only_for_gradio (
input_video_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,
progress =gr .Progress (track_tqdm =True )
):

    return util_split_video_only (
    input_video_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
    scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
    scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
    scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,
    app_config .DEFAULT_OUTPUT_DIR ,
    logger ,
    progress =progress
    )

with gr .Blocks (css =css ,theme =gr .themes .Soft ())as demo :
    gr .Markdown ("# Ultimate SECourses STAR Video Upscaler V10")

    with gr .Row ():
        with gr .Column (scale =1 ):
            with gr .Group ():
                input_video =gr .Video (
                label ="Input Video",
                sources =["upload"],
                interactive =True ,height =512
                )
                with gr .Row ():
                    user_prompt =gr .Textbox (
                    label ="Describe the Video Content (Prompt)",
                    lines =3 ,
                    placeholder ="e.g., A panda playing guitar by a lake at sunset.",
                    info ="""Describe the main subject and action in the video. This guides the upscaling process.
Combined with the Positive Prompt below, the effective text length influencing the model is limited to 77 tokens.
If CogVLM2 is available, you can use the button below to generate a caption automatically."""
                    )
                with gr .Row ():
                    auto_caption_then_upscale_check =gr .Checkbox (label ="Auto-caption then Upscale",value =app_config .DEFAULT_AUTO_CAPTION_THEN_UPSCALE ,info ="If checked, clicking 'Upscale Video' will first generate a caption and use it as the prompt.")

                    available_gpus =util_get_available_gpus ()
                    gpu_choices =["Auto"]+available_gpus if available_gpus else ["Auto","No CUDA GPUs detected"]
                    default_gpu =available_gpus [0 ]if available_gpus else "Auto"

                    gpu_selector =gr .Dropdown (
                    label ="GPU Selection",
                    choices =gpu_choices ,
                    value =default_gpu ,
                    info ="Select which GPU to use for processing. 'Auto' uses all available GPUs.",
                    scale =1
                    )

                if app_config .UTIL_COG_VLM_AVAILABLE :
                    with gr .Row ():
                        auto_caption_btn =gr .Button ("Generate Caption with CogVLM2",variant ="primary",icon ="icons/caption.png")
                        upscale_button =gr .Button ("Upscale Video",variant ="primary",icon ="icons/upscale.png")
                    caption_status =gr .Textbox (label ="Captioning Status",interactive =False ,visible =False )
                else :
                    upscale_button =gr .Button ("Upscale Video",variant ="primary",icon ="icons/upscale.png")

            with gr .Accordion ("Prompt Settings",open =True ):
                 pos_prompt =gr .Textbox (
                 label ="Default Positive Prompt (Appended)",
                 value =app_config .DEFAULT_POS_PROMPT ,
                 lines =2 ,
                 info ="""Appended to your 'Describe Video Content' prompt. Focuses on desired quality aspects (e.g., realism, detail).
The total combined prompt length is limited to 77 tokens."""
                 )
                 neg_prompt =gr .Textbox (
                 label ="Default Negative Prompt (Appended)",
                 value =app_config .DEFAULT_NEG_PROMPT ,
                 lines =2 ,
                 info ="Guides the model *away* from undesired aspects (e.g., bad quality, artifacts, specific styles). This does NOT count towards the 77 token limit for positive guidance."
                 )

            with gr .Accordion ("Advanced: Target Resolution",open =True ):
                 enable_target_res_check =gr .Checkbox (
                 label ="Enable Max Target Resolution",
                 value =app_config .DEFAULT_ENABLE_TARGET_RES ,
                 info ="Check this to manually control the maximum output resolution instead of using the simple Upscale Factor."
                 )
                 target_res_mode_radio =gr .Radio (
                 label ="Target Resolution Mode",
                 choices =['Ratio Upscale','Downscale then 4x'],value =app_config .DEFAULT_TARGET_RES_MODE ,
                 info ="""How to apply the target H/W limits.
'Ratio Upscale': Upscales by the largest factor possible without exceeding Target H/W, preserving aspect ratio.
'Downscale then 4x': If input is large, downscales it towards Target H/W divided by 4, THEN applies a 4x upscale. Can clean noisy high-res input before upscaling."""
                 )
                 with gr .Row ():
                     target_h_num =gr .Slider (
                     label ="Max Target Height (px)",
                     value =app_config .DEFAULT_TARGET_H ,minimum =128 ,maximum =4096 ,step =16 ,
                     info ="""Maximum allowed height for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                     )
                     target_w_num =gr .Slider (
                     label ="Max Target Width (px)",
                     value =app_config .DEFAULT_TARGET_W ,minimum =128 ,maximum =4096 ,step =16 ,
                     info ="""Maximum allowed width for the output video. Overrides Upscale Factor if enabled.
- VRAM Impact: Very High (Lower value = Less VRAM).
- Quality Impact: Direct (Lower value = Less detail).
- Speed Impact: Faster (Lower value = Faster)."""
                     )

            with gr .Accordion ("Performance & VRAM Optimization",open =True ):
                max_chunk_len_slider =gr .Slider (
                label ="Max Frames per Batch (VRAM)",
                minimum =4 ,maximum =96 ,value =app_config .DEFAULT_MAX_CHUNK_LEN ,step =4 ,
                info ="""IMPORTANT for VRAM. This is the standard way the application manages VRAM. It divides the entire sequence of video frames into sequential, non-overlapping chunks.
- Mechanism: The STAR model processes one complete chunk (of this many frames) at a time.
- VRAM Impact: High Reduction (Lower value = Less VRAM).
- Quality Impact: Can reduce Temporal Consistency (flicker/motion issues) between chunks if too low, as the model doesn't have context across chunk boundaries. Keep as high as VRAM allows.
- Speed Impact: Slower (Lower value = Slower, as more chunks are processed)."""
                )
                vae_chunk_slider =gr .Slider (
                label ="VAE Decode Chunk (VRAM)",
                minimum =1 ,maximum =16 ,value =app_config .DEFAULT_VAE_CHUNK ,step =1 ,
                info ="""Controls max latent frames decoded back to pixels by VAE simultaneously.
- VRAM Impact: High Reduction (Lower value = Less VRAM during decode stage).
- Quality Impact: Minimal / Negligible. Safe to lower.
- Speed Impact: Slower (Lower value = Slower decoding)."""
                )

            if app_config .UTIL_COG_VLM_AVAILABLE :
                with gr .Accordion ("Auto-Captioning Settings (CogVLM2)",open =True ):
                    cogvlm_quant_choices_map =app_config .get_cogvlm_quant_choices_map (torch .cuda .is_available (),app_config .UTIL_BITSANDBYTES_AVAILABLE )
                    cogvlm_quant_radio_choices_display =list (cogvlm_quant_choices_map .values ())
                    default_quant_display_val =app_config .get_default_cogvlm_quant_display (cogvlm_quant_choices_map )

                    with gr .Row ():
                        cogvlm_quant_radio =gr .Radio (
                        label ="CogVLM2 Quantization",
                        choices =cogvlm_quant_radio_choices_display ,
                        value =default_quant_display_val ,
                        info ="Quantization for the CogVLM2 captioning model (uses less VRAM). INT4/8 require CUDA & bitsandbytes.",
                        interactive =True if len (cogvlm_quant_radio_choices_display )>1 else False
                        )
                        cogvlm_unload_radio =gr .Radio (
                        label ="CogVLM2 After-Use",
                        choices =['full','cpu'],value =app_config .DEFAULT_COGVLM_UNLOAD_AFTER_USE ,
                        info ="""Memory management after captioning.
'full': Unload model completely from VRAM/RAM (frees most memory).
'cpu': Move model to RAM (faster next time, uses RAM, not for quantized models)."""
                        )
            else :
                gr .Markdown ("_(Auto-captioning disabled as CogVLM2 components are not fully available.)_")

            with gr .Accordion ("FFmpeg Encoding Settings",open =True ):
                ffmpeg_use_gpu_check =gr .Checkbox (
                label ="Use NVIDIA GPU for FFmpeg (h264_nvenc)",
                value =app_config .DEFAULT_FFMPEG_USE_GPU ,
                info ="If checked, uses NVIDIA's NVENC for FFmpeg video encoding (downscaling and final video creation). Requires NVIDIA GPU and correctly configured FFmpeg with NVENC support."
                )
                with gr .Row ():
                    ffmpeg_preset_dropdown =gr .Dropdown (
                    label ="FFmpeg Preset",
                    choices =['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],
                    value =app_config .DEFAULT_FFMPEG_PRESET ,
                    info ="Controls encoding speed vs. compression efficiency. 'ultrafast' is fastest with lowest quality/compression, 'veryslow' is slowest with highest quality/compression. Note: NVENC presets behave differently (e.g. p1-p7 or specific names like 'slow', 'medium', 'fast')."
                    )

                    ffmpeg_quality_slider =gr .Slider (
                    label ="FFmpeg Quality (CRF for libx264 / CQ for NVENC)",
                    minimum =0 ,maximum =51 ,value =app_config .DEFAULT_FFMPEG_QUALITY_CPU ,step =1 ,
                    info ="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default). For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28."
                    )

            with gr .Accordion ("Advanced: Sliding Window (Long Videos)",open =True ):
                 enable_sliding_window_check =gr .Checkbox (
                 label ="Enable Sliding Window",
                 value =app_config .DEFAULT_ENABLE_SLIDING_WINDOW ,
                 info ="""Processes the video in overlapping temporal chunks (windows). Use for very long videos where 'Max Frames per Batch' isn't enough or causes too many artifacts.
- Mechanism: Takes a 'Window Size' of frames, processes it, saves results from the central part, then slides the window forward by 'Window Step', processing overlapping frames.
- VRAM Impact: High Reduction (limits frames processed temporally, similar to Max Frames per Batch but with overlap).
- Quality Impact: Moderate risk of discontinuities at window boundaries if overlap (Window Size - Window Step) is small. Aims for better consistency than small non-overlapping chunks.
- Speed Impact: Slower (due to processing overlapping frames multiple times). When enabled, 'Window Size' dictates batch size instead of 'Max Frames per Batch'."""
                 )
                 with gr .Row ():
                     window_size_num =gr .Slider (
                     label ="Window Size (frames)",
                     value =app_config .DEFAULT_WINDOW_SIZE ,minimum =2 ,step =4 ,
                     info ="Number of frames in each temporal window. Acts like 'Max Frames per Batch' but applied as a sliding window. Lower value = less VRAM, less temporal context."
                     )
                     window_step_num =gr .Slider (
                     label ="Window Step (frames)",
                     value =app_config .DEFAULT_WINDOW_STEP ,minimum =1 ,step =1 ,
                     info ="How many frames to advance for the next window. (Window Size - Window Step) = Overlap. Smaller step = more overlap = better consistency but slower. Recommended: Step = Size / 2."
                     )

            with gr .Accordion ("Advanced: Tiling (Very High Res / Low VRAM)",open =True ):
                 enable_tiling_check =gr .Checkbox (
                 label ="Enable Tiled Upscaling",
                 value =app_config .DEFAULT_ENABLE_TILING ,
                 info ="""Processes each frame in small spatial patches (tiles). Use ONLY if necessary for extreme resolutions or very low VRAM.
- VRAM Impact: Very High Reduction.
- Quality Impact: High risk of tile seams/artifacts. Can harm global coherence and severely reduce temporal consistency.
- Speed Impact: Extremely Slow."""
                 )
                 with gr .Row ():
                     tile_size_num =gr .Number (
                     label ="Tile Size (px, input res)",
                     value =app_config .DEFAULT_TILE_SIZE ,minimum =64 ,step =32 ,
                     info ="Size of the square patches (in input resolution pixels) to process. Smaller = less VRAM per tile but more tiles = slower."
                     )
                     tile_overlap_num =gr .Number (
                     label ="Tile Overlap (px, input res)",
                     value =app_config .DEFAULT_TILE_OVERLAP ,minimum =0 ,step =16 ,
                     info ="How much the tiles overlap (in input resolution pixels). Higher overlap helps reduce seams but increases processing time. Recommend 1/4 to 1/2 of Tile Size."
                     )

        with gr .Column (scale =1 ):
            output_video =gr .Video (label ="Upscaled Video",interactive =False )
            status_textbox =gr .Textbox (label ="Log",interactive =False ,lines =8 ,max_lines =20 )

            with gr .Accordion ("Last Processed Chunk",open =True ):
                last_chunk_video =gr .Video (
                label ="Last Processed Chunk Preview",
                interactive =False ,
                height =300 ,
                visible =True
                )
                chunk_status_text =gr .Textbox (
                label ="Chunk Status",
                interactive =False ,
                lines =1 ,
                value ="No chunks processed yet"
                )

            with gr .Group ():
                gr .Markdown ("### Core Upscaling Settings")
                model_selector =gr .Dropdown (
                label ="STAR Model",
                choices =["Light Degradation","Heavy Degradation"],
                value =app_config .DEFAULT_MODEL_CHOICE ,
                info ="""Choose the model based on input video quality.
'Light Degradation': Better for relatively clean inputs (e.g., downloaded web videos).
'Heavy Degradation': Better for inputs with significant compression artifacts, noise, or blur."""
                )
                upscale_factor_slider =gr .Slider (
                label ="Upscale Factor (if Target Res disabled)",
                minimum =1.0 ,maximum =8.0 ,value =app_config .DEFAULT_UPSCALE_FACTOR ,step =0.1 ,
                info ="Simple multiplication factor for output resolution if 'Enable Max Target Resolution' is OFF. E.g., 4.0 means 4x height and 4x width."
                )
                cfg_slider =gr .Slider (
                label ="Guidance Scale (CFG)",
                minimum =1.0 ,maximum =15.0 ,value =app_config .DEFAULT_CFG_SCALE ,step =0.5 ,
                info ="Controls how strongly the model follows your combined text prompt. Higher values mean stricter adherence, lower values allow more creativity. Typical values: 5.0-10.0."
                )
                with gr .Row ():
                    solver_mode_radio =gr .Radio (
                    label ="Solver Mode",
                    choices =['fast','normal'],value =app_config .DEFAULT_SOLVER_MODE ,
                    info ="""Diffusion solver type.
'fast': Fewer steps (default ~15), much faster, good quality usually.
'normal': More steps (default ~50), slower, potentially slightly better detail/coherence."""
                    )

                    steps_slider =gr .Slider (
                    label ="Diffusion Steps",
                    minimum =5 ,maximum =100 ,value =app_config .DEFAULT_DIFFUSION_STEPS_FAST ,step =1 ,
                    info ="Number of denoising steps. 'Fast' mode uses a fixed ~15 steps. 'Normal' mode uses the value set here.",
                    interactive =False
                    )
                color_fix_dropdown =gr .Dropdown (
                label ="Color Correction",
                choices =['AdaIN','Wavelet','None'],value =app_config .DEFAULT_COLOR_FIX_METHOD ,
                info ="""Attempts to match the color tone of the output to the input video. Helps prevent color shifts.
'AdaIN' / 'Wavelet': Different algorithms for color matching. AdaIN is often a good default.
'None': Disables color correction."""
                )

            with gr .Accordion ("Scene Splitting",open =True ):
                enable_scene_split_check =gr .Checkbox (
                label ="Enable Scene Splitting",
                value =app_config .DEFAULT_ENABLE_SCENE_SPLIT ,
                info ="""Split video into scenes and process each scene individually. This can improve quality and speed by processing similar content together.
- Quality Impact: Better temporal consistency within scenes, improved auto-captioning per scene.
- Speed Impact: Can be faster for long videos with distinct scenes.
- Memory Impact: Reduces peak memory usage by processing smaller segments."""
                )

                with gr .Row ():
                    scene_split_mode_radio =gr .Radio (
                    label ="Split Mode",
                    choices =['automatic','manual'],value =app_config .DEFAULT_SCENE_SPLIT_MODE ,
                    info ="""'automatic': Uses scene detection algorithms to find natural scene boundaries.
'manual': Splits video at fixed intervals (duration or frame count)."""
                    )

                with gr .Group ():
                    gr .Markdown ("**Automatic Scene Detection Settings**")
                    with gr .Row ():
                        scene_min_scene_len_num =gr .Number (
                        label ="Min Scene Length (seconds)",
                        value =app_config .DEFAULT_SCENE_MIN_SCENE_LEN ,minimum =0.1 ,step =0.1 ,
                        info ="Minimum duration for a scene. Shorter scenes will be merged or dropped."
                        )
                        scene_threshold_num =gr .Number (
                        label ="Detection Threshold",
                        value =app_config .DEFAULT_SCENE_THRESHOLD ,minimum =0.1 ,maximum =10.0 ,step =0.1 ,
                        info ="Sensitivity of scene detection. Lower values detect more scenes."
                        )

                    with gr .Row ():
                        scene_drop_short_check =gr .Checkbox (
                        label ="Drop Short Scenes",
                        value =app_config .DEFAULT_SCENE_DROP_SHORT ,
                        info ="If enabled, scenes shorter than minimum length are dropped instead of merged."
                        )
                        scene_merge_last_check =gr .Checkbox (
                        label ="Merge Last Scene",
                        value =app_config .DEFAULT_SCENE_MERGE_LAST ,
                        info ="If the last scene is too short, merge it with the previous scene."
                        )

                    with gr .Row ():
                        scene_frame_skip_num =gr .Number (
                        label ="Frame Skip",
                        value =app_config .DEFAULT_SCENE_FRAME_SKIP ,minimum =0 ,step =1 ,
                        info ="Skip frames during detection to speed up processing. 0 = analyze every frame."
                        )
                        scene_min_content_val_num =gr .Number (
                        label ="Min Content Value",
                        value =app_config .DEFAULT_SCENE_MIN_CONTENT_VAL ,minimum =0.0 ,step =1.0 ,
                        info ="Minimum content change required to detect a scene boundary."
                        )
                        scene_frame_window_num =gr .Number (
                        label ="Frame Window",
                        value =app_config .DEFAULT_SCENE_FRAME_WINDOW ,minimum =1 ,step =1 ,
                        info ="Number of frames to analyze for scene detection."
                        )

                with gr .Group ():
                    gr .Markdown ("**Manual Split Settings**")
                    with gr .Row ():
                        scene_manual_split_type_radio =gr .Radio (
                        label ="Manual Split Type",
                        choices =['duration','frame_count'],value =app_config .DEFAULT_SCENE_MANUAL_SPLIT_TYPE ,
                        info ="""'duration': Split every N seconds.
'frame_count': Split every N frames."""
                        )
                        scene_manual_split_value_num =gr .Number (
                        label ="Split Value",
                        value =app_config .DEFAULT_SCENE_MANUAL_SPLIT_VALUE ,minimum =1.0 ,step =1.0 ,
                        info ="Duration in seconds or number of frames for manual splitting."
                        )

                with gr .Group ():
                    gr .Markdown ("**Encoding Settings**")
                    with gr .Row ():
                        scene_copy_streams_check =gr .Checkbox (
                        label ="Copy Streams",
                        value =app_config .DEFAULT_SCENE_COPY_STREAMS ,
                        info ="Copy video/audio streams without re-encoding during scene splitting (faster)."
                        )
                        scene_use_mkvmerge_check =gr .Checkbox (
                        label ="Use MKVMerge",
                        value =app_config .DEFAULT_SCENE_USE_MKVMERGE ,
                        info ="Use mkvmerge instead of ffmpeg for splitting (if available)."
                        )

                    with gr .Row ():
                        scene_rate_factor_num =gr .Number (
                        label ="Rate Factor (CRF)",
                        value =app_config .DEFAULT_SCENE_RATE_FACTOR ,minimum =0 ,maximum =51 ,step =1 ,
                        info ="Quality setting for re-encoding (lower = better quality). Only used if Copy Streams is disabled."
                        )
                        scene_preset_dropdown =gr .Dropdown (
                        label ="Encoding Preset",
                        choices =['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],
                        value =app_config .DEFAULT_SCENE_ENCODING_PRESET ,
                        info ="Encoding speed vs quality trade-off. Only used if Copy Streams is disabled."
                        )

                    scene_quiet_ffmpeg_check =gr .Checkbox (
                    label ="Quiet FFmpeg",
                    value =app_config .DEFAULT_SCENE_QUIET_FFMPEG ,
                    info ="Suppress ffmpeg output during scene splitting."
                    )

            with gr .Accordion ("Batch Processing",open =True ):
                with gr .Row ():
                    batch_input_folder =gr .Textbox (
                    label ="Input Folder",
                    placeholder ="Path to folder containing videos to process...",
                    info ="Folder containing video files to process in batch mode."
                    )
                    batch_output_folder =gr .Textbox (
                    label ="Output Folder",
                    placeholder ="Path to output folder for processed videos...",
                    info ="Folder where processed videos will be saved, preserving original filenames."
                    )

            with gr .Accordion ("Output Options",open =True ):
                save_frames_checkbox =gr .Checkbox (
                label ="Save Input and Processed Frames",
                value =app_config .DEFAULT_SAVE_FRAMES ,
                info ="If checked, saves the extracted input frames and the upscaled output frames into a subfolder named after the output video (e.g., '0001/input_frames' and '0001/processed_frames')."
                )
                save_metadata_checkbox =gr .Checkbox (
                label ="Save Processing Metadata",
                value =app_config .DEFAULT_SAVE_METADATA ,
                info ="If checked, saves a .txt file (e.g., '0001.txt') in the main output folder, containing all processing parameters and total processing time."
                )
                save_chunks_checkbox =gr .Checkbox (
                label ="Save Processed Chunks",
                value =app_config .DEFAULT_SAVE_CHUNKS ,
                info ="If checked, saves each processed chunk as a video file in a 'chunks' subfolder (e.g., '0001/chunks/chunk_0001.mp4'). Uses the same FFmpeg settings as the final video."
                )
                open_output_folder_button =gr .Button ("Open Output Folder")

            with gr .Row ():
                split_only_button =gr .Button ("Split Video Only (No Upscaling)",variant ="secondary")
                batch_process_button =gr .Button ("Process Batch Folder",variant ="primary",visible =True )

    def update_steps_display (mode ):
        if mode =='fast':
            return gr .update (value =app_config .DEFAULT_DIFFUSION_STEPS_FAST ,interactive =False )
        else :
            return gr .update (value =app_config .DEFAULT_DIFFUSION_STEPS_NORMAL ,interactive =True )
    solver_mode_radio .change (update_steps_display ,solver_mode_radio ,steps_slider )

    enable_target_res_check .change (lambda x :[gr .update (interactive =x )]*3 ,inputs =enable_target_res_check ,outputs =[target_h_num ,target_w_num ,target_res_mode_radio ])
    enable_tiling_check .change (lambda x :[gr .update (interactive =x )]*2 ,inputs =enable_tiling_check ,outputs =[tile_size_num ,tile_overlap_num ])
    enable_sliding_window_check .change (lambda x :[gr .update (interactive =x )]*2 ,inputs =enable_sliding_window_check ,outputs =[window_size_num ,window_step_num ])

    enable_scene_split_check .change (
    lambda x :[gr .update (interactive =x )]*15 ,
    inputs =enable_scene_split_check ,
    outputs =[scene_split_mode_radio ,scene_min_scene_len_num ,scene_threshold_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_manual_split_type_radio ,scene_manual_split_value_num ,scene_copy_streams_check ,
    scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check ]
    )

    def update_ffmpeg_quality_settings (use_gpu ):
        if use_gpu :
            return gr .Slider (label ="FFmpeg Quality (CQ for NVENC)",value =app_config .DEFAULT_FFMPEG_QUALITY_GPU ,info ="For h264_nvenc (GPU): Constrained Quality (CQ). Lower values generally mean better quality. Typical range for NVENC CQ is 18-28.")
        else :
            return gr .Slider (label ="FFmpeg Quality (CRF for libx264)",value =app_config .DEFAULT_FFMPEG_QUALITY_CPU ,info ="For libx264 (CPU): Constant Rate Factor (CRF). Lower values mean higher quality (0 is lossless, 23 is default).")

    ffmpeg_use_gpu_check .change (
    fn =update_ffmpeg_quality_settings ,
    inputs =ffmpeg_use_gpu_check ,
    outputs =ffmpeg_quality_slider
    )

    open_output_folder_button .click (
    fn =lambda :util_open_folder (app_config .DEFAULT_OUTPUT_DIR , logger=logger), # Added logger
    inputs =[],
    outputs =[]
    )

    cogvlm_display_to_quant_val_map_global ={}
    if app_config .UTIL_COG_VLM_AVAILABLE :
        _temp_map =app_config .get_cogvlm_quant_choices_map (torch .cuda .is_available (),app_config .UTIL_BITSANDBYTES_AVAILABLE )
        cogvlm_display_to_quant_val_map_global ={v :k for k ,v in _temp_map .items ()}

    def get_quant_value_from_display (display_val ):
        # This function is now primarily used by upscale_director_logic
        # run_upscale will receive the display_val and convert it internally if it's a string
        if isinstance(display_val, int): return display_val # Already an int
        return cogvlm_display_to_quant_val_map_global .get (display_val ,0 )


    def upscale_director_logic (
    input_video_val ,user_prompt_val ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
    upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
    max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
    enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
    enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
    enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
    ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
    save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,

    enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
    scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
    scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
    scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,
    cogvlm_quant_radio_val =None ,cogvlm_unload_radio_val =None , # These are display values from UI
    do_auto_caption_first_val =False ,
    progress =gr .Progress (track_tqdm =True )
    ):
        current_user_prompt =user_prompt_val
        status_updates =[]
        output_updates_for_prompt_box =gr .update ()

        logger .info (f"In upscale_director_logic. do_auto_caption_first_val: {do_auto_caption_first_val}, User prompt: '{user_prompt_val[:50]}...'")

        # Convert cogvlm_quant_radio_val (display string) to actual int value for util_auto_caption
        actual_cogvlm_quant_for_captioning = get_quant_value_from_display(cogvlm_quant_radio_val) if cogvlm_quant_radio_val is not None else 0


        if do_auto_caption_first_val and app_config.UTIL_COG_VLM_AVAILABLE : # Added check for UTIL_COG_VLM_AVAILABLE
            logger .info ("Attempting auto-captioning before upscale.")
            progress (0 ,desc ="Starting auto-captioning before upscale...")
            yield None ,"Starting auto-captioning...",output_updates_for_prompt_box ,gr .update (visible =True ),None ,"Starting auto-captioning..."
            try :
                logger .info ("Preparing to call util_auto_caption function.")
                # quant_val already converted: actual_cogvlm_quant_for_captioning

                caption_text ,caption_stat_msg =util_auto_caption (
                input_video_val ,
                actual_cogvlm_quant_for_captioning , # Pass the converted int value
                cogvlm_unload_radio_val ,
                app_config .COG_VLM_MODEL_PATH ,
                logger =logger ,
                progress =progress
                )
                logger .info (f"util_auto_caption function returned. Caption text (first 50 chars): '{caption_text[:50]}...'")
                status_updates .append (f"Auto-caption status: {caption_stat_msg}")
                if not caption_text .startswith ("Error:"):
                    current_user_prompt =caption_text
                    logger .info (f"Using generated caption as prompt for upscaling: '{current_user_prompt}'")
                    status_updates .append (f"Using generated caption as prompt: '{caption_text[:50]}...'")
                    output_updates_for_prompt_box =gr .update (value =current_user_prompt )
                else :
                    status_updates .append ("Caption generation failed. Using original prompt.")

                yield None ,"\n".join (status_updates ),output_updates_for_prompt_box ,caption_stat_msg ,None ,"Auto-captioning finished."
            except Exception as e_ac :
                logger .error (f"Exception during auto-caption call or its setup: {e_ac}",exc_info =True )
                status_updates .append (f"Error during auto-caption pre-step: {e_ac}")
                yield None ,"\n".join (status_updates ),gr .update (),str (e_ac ),None ,f"Error during auto-caption: {e_ac}"
        else :
            logger .info ("Skipping auto-captioning before upscale. Either CogVLM not available or option not selected.")

        # run_upscale expects cogvlm_quant as the display string or int, it handles conversion internally if needed
        upscale_generator =run_upscale (
        input_video_val ,current_user_prompt ,pos_prompt_val ,neg_prompt_val ,model_selector_val ,
        upscale_factor_slider_val ,cfg_slider_val ,steps_slider_val ,solver_mode_radio_val ,
        max_chunk_len_slider_val ,vae_chunk_slider_val ,color_fix_dropdown_val ,
        enable_tiling_check_val ,tile_size_num_val ,tile_overlap_num_val ,
        enable_sliding_window_check_val ,window_size_num_val ,window_step_num_val ,
        enable_target_res_check_val ,target_h_num_val ,target_w_num_val ,target_res_mode_radio_val ,
        ffmpeg_preset_dropdown_val ,ffmpeg_quality_slider_val ,ffmpeg_use_gpu_check_val ,
        save_frames_checkbox_val ,save_metadata_checkbox_val ,save_chunks_checkbox_val ,

        enable_scene_split_check_val ,scene_split_mode_radio_val ,scene_min_scene_len_num_val ,scene_drop_short_check_val ,scene_merge_last_check_val ,
        scene_frame_skip_num_val ,scene_threshold_num_val ,scene_min_content_val_num_val ,scene_frame_window_num_val ,
        scene_copy_streams_check_val ,scene_use_mkvmerge_check_val ,scene_rate_factor_num_val ,scene_preset_dropdown_val ,scene_quiet_ffmpeg_check_val ,
        scene_manual_split_type_radio_val ,scene_manual_split_value_num_val ,

        False ,None ,None , # is_batch_mode, batch_output_dir, original_filename

        do_auto_caption_first_val and enable_scene_split_check_val and app_config.UTIL_COG_VLM_AVAILABLE, # enable_auto_caption_per_scene
        cogvlm_quant_radio_val, # Pass display value, run_upscale will convert
        cogvlm_unload_radio_val if cogvlm_unload_radio_val else 'full',
        progress =progress
        )

        final_video_output =None
        final_status_output =""
        final_chunk_video =None
        final_chunk_status ="No chunks processed yet"

        for result in upscale_generator :
            # run_upscale now yields: output_val, status_val, chunk_video_val, chunk_status_val
            output_val ,status_val ,chunk_video_val ,chunk_status_val =result
            
            final_video_output =output_val
            final_status_output =(("\n".join (status_updates )+"\n")if status_updates else "")+(status_val if status_val else "")
            if chunk_video_val is not None : # Ensure it's not None before updating
                final_chunk_video =chunk_video_val
            if chunk_status_val : # Ensure it's not empty/None before updating
                final_chunk_status =chunk_status_val

            caption_status_update = caption_status.value if app_config.UTIL_COG_VLM_AVAILABLE and hasattr(caption_status, 'value') and do_auto_caption_first_val else ""


            yield final_video_output ,final_status_output .strip (),output_updates_for_prompt_box ,caption_status_update ,final_chunk_video ,final_chunk_status
            output_updates_for_prompt_box =gr .update () # Reset prompt update after first yield

        final_caption_status_val = caption_status.value if app_config.UTIL_COG_VLM_AVAILABLE and hasattr(caption_status, 'value') else ""
        yield final_video_output ,final_status_output .strip (),output_updates_for_prompt_box ,final_caption_status_val ,final_chunk_video ,final_chunk_status


    click_inputs =[
    input_video ,user_prompt ,pos_prompt ,neg_prompt ,model_selector ,
    upscale_factor_slider ,cfg_slider ,steps_slider ,solver_mode_radio ,
    max_chunk_len_slider ,vae_chunk_slider ,color_fix_dropdown ,
    enable_tiling_check ,tile_size_num ,tile_overlap_num ,
    enable_sliding_window_check ,window_size_num ,window_step_num ,
    enable_target_res_check ,target_h_num ,target_w_num ,target_res_mode_radio ,
    ffmpeg_preset_dropdown ,ffmpeg_quality_slider ,ffmpeg_use_gpu_check ,
    save_frames_checkbox ,save_metadata_checkbox ,save_chunks_checkbox ,

    enable_scene_split_check ,scene_split_mode_radio ,scene_min_scene_len_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_threshold_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_copy_streams_check ,scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check ,
    scene_manual_split_type_radio ,scene_manual_split_value_num
    ]

    click_outputs_list =[output_video ,status_textbox ,user_prompt ]
    if app_config .UTIL_COG_VLM_AVAILABLE :
        click_outputs_list .append (caption_status )
    else: # Need a placeholder if caption_status is not in outputs
        click_outputs_list.append(gr.State(None))


    click_outputs_list .extend ([last_chunk_video ,chunk_status_text ])

    if app_config .UTIL_COG_VLM_AVAILABLE :
        click_inputs .extend ([cogvlm_quant_radio ,cogvlm_unload_radio ,auto_caption_then_upscale_check ])
    else :
        click_inputs .extend ([gr .State (None ),gr .State (None ),gr .State (False )])

    upscale_button .click (
    fn =upscale_director_logic ,
    inputs =click_inputs ,
    outputs =click_outputs_list
    )

    if app_config .UTIL_COG_VLM_AVAILABLE :
        auto_caption_btn .click (
        fn =lambda vid ,quant_display ,unload_strat ,progress =gr .Progress (track_tqdm =True ):util_auto_caption (vid ,get_quant_value_from_display (quant_display ),unload_strat ,app_config .COG_VLM_MODEL_PATH ,logger =logger ,progress =progress ),
        inputs =[input_video ,cogvlm_quant_radio ,cogvlm_unload_radio ],
        outputs =[user_prompt ,caption_status ]
        ).then (lambda :gr .update (visible =True ), None, caption_status ) # Added None for inputs to .then

    split_only_button .click (
    fn =wrapper_split_video_only_for_gradio ,
    inputs =[
    input_video ,scene_split_mode_radio ,scene_min_scene_len_num ,scene_drop_short_check ,scene_merge_last_check ,
    scene_frame_skip_num ,scene_threshold_num ,scene_min_content_val_num ,scene_frame_window_num ,
    scene_copy_streams_check ,scene_use_mkvmerge_check ,scene_rate_factor_num ,scene_preset_dropdown ,scene_quiet_ffmpeg_check ,
    scene_manual_split_type_radio ,scene_manual_split_value_num
    ],
    outputs =[output_video ,status_textbox ]
    )

    batch_process_inputs = [
        batch_input_folder, batch_output_folder,
        user_prompt, pos_prompt, neg_prompt, model_selector,
        upscale_factor_slider, cfg_slider, steps_slider, solver_mode_radio,
        max_chunk_len_slider, vae_chunk_slider, color_fix_dropdown,
        enable_tiling_check, tile_size_num, tile_overlap_num,
        enable_sliding_window_check, window_size_num, window_step_num,
        enable_target_res_check, target_h_num, target_w_num, target_res_mode_radio,
        ffmpeg_preset_dropdown, ffmpeg_quality_slider, ffmpeg_use_gpu_check,
        save_frames_checkbox, save_metadata_checkbox, save_chunks_checkbox,
        enable_scene_split_check, scene_split_mode_radio, scene_min_scene_len_num, scene_drop_short_check, scene_merge_last_check,
        scene_frame_skip_num, scene_threshold_num, scene_min_content_val_num, scene_frame_window_num,
        scene_copy_streams_check, scene_use_mkvmerge_check, scene_rate_factor_num, scene_preset_dropdown, scene_quiet_ffmpeg_check,
        scene_manual_split_type_radio, scene_manual_split_value_num
    ]
    if app_config.UTIL_COG_VLM_AVAILABLE:
        batch_process_inputs.extend([cogvlm_quant_radio, cogvlm_unload_radio, auto_caption_then_upscale_check])
    else:
        batch_process_inputs.extend([gr.State(None), gr.State(None), gr.State(False)])


    batch_process_button .click (
    fn =process_batch_videos ,
    inputs = batch_process_inputs,
    outputs =[output_video ,status_textbox ] # Output video might not be relevant for batch, maybe just status
    )

    gpu_selector .change (
    fn =lambda gpu_id :util_set_gpu_device (gpu_id ,logger =logger ),
    inputs =gpu_selector ,
    outputs =status_textbox # Changed from None to status_textbox to show GPU change status
    )

if __name__ =="__main__":

    os .makedirs (app_config .DEFAULT_OUTPUT_DIR ,exist_ok =True )
    logger .info (f"Gradio App Starting. Default output to: {os.path.abspath(app_config.DEFAULT_OUTPUT_DIR)}")
    logger .info (f"STAR Models expected at: {app_config.LIGHT_DEG_MODEL_PATH}, {app_config.HEAVY_DEG_MODEL_PATH}")
    if app_config .UTIL_COG_VLM_AVAILABLE :
        logger .info (f"CogVLM2 Model expected at: {app_config.COG_VLM_MODEL_PATH}")

    available_gpus =util_get_available_gpus ()
    if available_gpus :
        default_gpu =available_gpus [0 ]
        util_set_gpu_device (default_gpu ,logger =logger )
        logger .info (f"Initialized with default GPU: {default_gpu}")
    else :
        logger .info ("No CUDA GPUs detected, using CPU mode")
        util_set_gpu_device ("Auto",logger =logger ) # This will set to CPU if no CUDA

    effective_allowed_paths =util_get_available_drives (app_config .DEFAULT_OUTPUT_DIR ,base_path, logger=logger ) # Added logger

    demo .queue ().launch (
    debug =True ,
    max_threads =100 , # Default from Gradio is 40, consider if 100 is too high or optimal
    inbrowser =True ,
    share =args .share ,
    allowed_paths =effective_allowed_paths ,
    prevent_thread_lock =True
    )