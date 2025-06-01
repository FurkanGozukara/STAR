"""
Scene processing core functionality.
Handles the processing of individual video scenes with various upscaling methods.
"""

import os
import time
import shutil
import cv2
import torch
import numpy as np
import gc # Added for garbage collection


def process_single_scene(
    scene_video_path, scene_index, total_scenes, temp_dir,
    final_prompt, upscale_factor, final_h, final_w, ui_total_diffusion_steps,
    solver_mode, cfg_scale, max_chunk_len, vae_chunk, color_fix_method,
    enable_tiling, tile_size, tile_overlap, enable_sliding_window, window_size, window_step,
    save_frames, scene_output_dir, progress_callback=None,

    enable_auto_caption_per_scene=False, cogvlm_quant=0, cogvlm_unload='full',
    progress=None, save_chunks=False, chunks_permanent_save_path=None, ffmpeg_preset="medium", ffmpeg_quality_value=23, ffmpeg_use_gpu=False,
    save_metadata=False, metadata_params_base: dict = None,
    
    # FPS decrease parameters for scenes
    enable_fps_decrease=False, fps_decrease_mode="multiplier", fps_multiplier_preset="1/2x (Half FPS)", fps_multiplier_custom=0.5, target_fps=24.0, fps_interpolation_method="drop",
    
    # RIFE interpolation parameters for scenes and chunks
    enable_rife_interpolation=False, rife_multiplier=2, rife_fp16=True, rife_uhd=False, rife_scale=1.0,
    rife_skip_static=False, rife_enable_fps_limit=False, rife_max_fps_limit=60,
    rife_apply_to_scenes=True, rife_apply_to_chunks=True, rife_keep_original=True, current_seed=99,
    
    # Dependencies injected as parameters
    util_extract_frames=None,
    util_auto_caption=None, 
    util_create_video_from_frames=None,
    logger=None,
    metadata_handler=None,
    format_time=None,
    preprocess=None,
    ImageSpliterTh=None,
    collate_fn=None,
    tensor2vid=None,
    adain_color_fix=None,
    wavelet_color_fix=None,
    VideoToVideo_sr_class=None, # Added
    EasyDict_class=None, # Added
    app_config_module_param=None, # Renamed to avoid conflict if app_config was already a local var name
    util_get_gpu_device_param=None # Renamed
):
    """
    Process a single video scene with upscaling and optional RIFE interpolation.
    
    This function handles the complete processing of one scene including:
    - Frame extraction
    - Auto-captioning (if enabled)
    - Upscaling with various methods (tiling, sliding window, or chunked)
    - Video reconstruction
    - RIFE interpolation (if enabled)
    - Metadata saving
    
    Args:
        scene_video_path: Path to the scene video file
        scene_index: Index of the current scene (0-based)
        total_scenes: Total number of scenes being processed
        temp_dir: Temporary directory for processing
        final_prompt: Text prompt for upscaling guidance
        upscale_factor: Factor by which to upscale the video
        final_h, final_w: Target height and width
        ui_total_diffusion_steps: Number of diffusion steps
        solver_mode: Diffusion solver mode
        cfg_scale: Guidance scale for diffusion
        max_chunk_len: Maximum chunk length for processing
        vae_chunk: VAE chunk size
        color_fix_method: Color correction method
        enable_tiling: Whether to use tiling mode
        tile_size, tile_overlap: Tiling parameters
        enable_sliding_window: Whether to use sliding window mode
        window_size, window_step: Sliding window parameters
        save_frames: Whether to save extracted frames
        scene_output_dir: Directory for scene outputs
        progress_callback: Callback for progress updates
        
        # RIFE interpolation parameters:
        enable_rife_interpolation: Whether to apply RIFE interpolation to scene and chunks
        rife_multiplier: FPS multiplier (2 or 4)
        rife_fp16: Use FP16 precision for RIFE
        rife_uhd: UHD mode for RIFE
        rife_scale: Scale factor for RIFE
        rife_skip_static: Skip static frames in RIFE
        rife_enable_fps_limit: Enable FPS limiting
        rife_max_fps_limit: Maximum FPS when limiting enabled
        rife_apply_to_scenes: Whether to apply RIFE to scene videos
        rife_apply_to_chunks: Whether to apply RIFE to chunk videos
        rife_keep_original: Keep original scene and chunk video files
        current_seed: Random seed for consistent results
        
        (... other parameters for various features ...)
        
        # Injected dependencies:
        util_extract_frames: Function to extract frames from video
        util_auto_caption: Function for auto-captioning
        util_create_video_from_frames: Function to create video from frames
        logger: Logger instance
        metadata_handler: Metadata handling utility
        format_time: Time formatting utility
        preprocess: Preprocessing function
        ImageSpliterTh: Image splitting utility
        collate_fn: Data collation function
        tensor2vid: Tensor to video conversion
        adain_color_fix, wavelet_color_fix: Color correction functions
        VideoToVideo_sr_class: Video to video super-resolution class
        EasyDict_class: Easy dictionary class for model configuration
        app_config_module_param: Application configuration module parameter
        util_get_gpu_device_param: Function to get GPU device parameter
        
    Yields:
        Various progress updates and completion status
        Final yield: ("scene_complete", final_scene_video, scene_frame_count, scene_fps, generated_scene_caption)
        Where final_scene_video is the RIFE-interpolated version if RIFE is enabled and successful,
        otherwise the original upscaled scene video.
    """
    star_model_instance = None # Initialize to ensure cleanup can run
    try:
        # Model loading moved inside
        if logger: logger.info(f"Scene {scene_index + 1}: Initializing STAR model for this scene.")
        model_load_start_time = time.time()
        # Use the passed app_config_module_param and util_get_gpu_device_param
        # Ensure app_config_module_param has the necessary attributes like LIGHT_DEG_MODEL_PATH etc.
        # Determine model_file_path based on a passed parameter or a heuristic if not directly available
        # For now, assuming a way to get model_choice or it's implicitly handled by app_config_module_param paths

        # This part needs to be adapted based on how model_choice is determined or passed.
        # For simplicity, let's assume app_config_module_param.CURRENT_MODEL_FILE_PATH exists or similar
        # Or, we might need to pass model_choice into this function as well.
        # For now, this is a placeholder, needs correct model path derivation.
        # It's likely process_single_scene will need model_choice passed to it.
        # Let's assume for now 'model_file_path_for_scene' is derived correctly. This is a CRITICAL part.
        # We'll assume it needs to be passed in or determined from app_config.

        # Placeholder: model_choice would ideally be passed to process_single_scene
        # For this refactor, let's assume it's available via metadata_params_base or similar for now
        # Or that app_config_module_param has a way to get the default/current one if not specified.
        
        model_choice_from_meta = metadata_params_base.get("model_choice", app_config_module_param.DEFAULT_MODEL_CHOICE if app_config_module_param else "Light Degradation")

        model_file_path_for_scene = app_config_module_param.LIGHT_DEG_MODEL_PATH if model_choice_from_meta == app_config_module_param.DEFAULT_MODEL_CHOICE else app_config_module_param.HEAVY_DEG_MODEL_PATH

        if not os.path.exists(model_file_path_for_scene):
            if logger: logger.error(f"STAR model not found for scene: {model_file_path_for_scene}")
            raise FileNotFoundError(f"STAR model not found for scene: {model_file_path_for_scene}")

        model_cfg = EasyDict_class()
        model_cfg.model_path = model_file_path_for_scene
        
        # Use the passed util_get_gpu_device_param
        model_device = torch.device(util_get_gpu_device_param(logger=logger)) if torch.cuda.is_available() else torch.device('cpu')
        star_model_instance = VideoToVideo_sr_class(model_cfg, device=model_device)
        if logger: logger.info(f"Scene {scene_index + 1}: STAR model loaded on {model_device}. Load time: {format_time(time.time() - model_load_start_time)}")

        scene_start_time = time.time()
        scene_name = f"scene_{scene_index + 1:04d}"

        if progress_callback:
            progress_callback(0.0, f"Processing scene {scene_index + 1}/{total_scenes}: {scene_name}")

        scene_temp_dir = os.path.join(temp_dir, scene_name)
        scene_input_frames_dir = os.path.join(scene_temp_dir, "input_frames")
        scene_output_frames_dir = os.path.join(scene_temp_dir, "output_frames")

        os.makedirs(scene_temp_dir, exist_ok=True)
        os.makedirs(scene_input_frames_dir, exist_ok=True)
        os.makedirs(scene_output_frames_dir, exist_ok=True)

        if progress_callback:
            progress_callback(0.1, f"Scene {scene_index + 1}: Extracting frames...")

        scene_frame_count, scene_fps, scene_frame_files = util_extract_frames(scene_video_path, scene_input_frames_dir, logger=logger)

        scene_input_frames_permanent = None
        scene_output_frames_permanent = None
        if save_frames and scene_output_dir:
            scene_frames_dir = os.path.join(scene_output_dir, "scenes", scene_name)
            scene_input_frames_permanent = os.path.join(scene_frames_dir, "input_frames")
            scene_output_frames_permanent = os.path.join(scene_frames_dir, "processed_frames")
            os.makedirs(scene_input_frames_permanent, exist_ok=True)
            os.makedirs(scene_output_frames_permanent, exist_ok=True)

            for frame_file in scene_frame_files:
                shutil.copy2(
                    os.path.join(scene_input_frames_dir, frame_file),
                    os.path.join(scene_input_frames_permanent, frame_file)
                )

        scene_prompt = final_prompt
        generated_scene_caption = None
        if enable_auto_caption_per_scene:
            if progress_callback:
                progress_callback(0.15, f"Scene {scene_index + 1}: Generating caption...")
            try:
                scene_caption, _ = util_auto_caption(
                    scene_video_path, cogvlm_quant, cogvlm_unload,
                    app_config_module_param.COG_VLM_MODEL_PATH, logger=logger, progress=progress
                )
                if not scene_caption.startswith("Error:"):
                    scene_prompt = scene_caption
                    generated_scene_caption = scene_caption
                    logger.info(f"Scene {scene_index + 1} auto-caption: {scene_caption[:100]}...")
                    if scene_index == 0:
                        logger.info(f"FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE: {scene_caption}")
                else:
                    logger.warning(f"Scene {scene_index + 1} auto-caption failed, using original prompt")
            except Exception as e:
                logger.error(f"Error auto-captioning scene {scene_index + 1}: {e}")

        if progress_callback:
            progress_callback(0.2, f"Scene {scene_index + 1}: Loading frames...")

        all_lr_frames_bgr = []
        for frame_filename in scene_frame_files:
            frame_lr_bgr = cv2.imread(os.path.join(scene_input_frames_dir, frame_filename))
            if frame_lr_bgr is None:
                logger.error(f"Could not read frame {frame_filename} from scene {scene_name}")
                continue
            all_lr_frames_bgr.append(frame_lr_bgr)

        if not all_lr_frames_bgr:
            raise Exception(f"No valid frames found in scene {scene_name}")

        total_noise_levels = 900
        if progress_callback:
            progress_callback(0.3, f"Scene {scene_index + 1}: Starting upscaling...")

        if enable_tiling:
            # Tiling mode processing
            for i, frame_filename in enumerate(scene_frame_files):
                frame_lr_bgr = cv2.imread(os.path.join(scene_input_frames_dir, frame_filename))
                if frame_lr_bgr is None:
                    continue
                single_lr_frame_tensor_norm = preprocess([frame_lr_bgr])
                _temp_spliter_for_count = ImageSpliterTh(single_lr_frame_tensor_norm, int(tile_size), int(tile_overlap), sf=upscale_factor)
                num_patches_this_frame_for_cb = sum(1 for _ in _temp_spliter_for_count)
                del _temp_spliter_for_count
                spliter = ImageSpliterTh(single_lr_frame_tensor_norm, int(tile_size), int(tile_overlap), sf=upscale_factor)

                for patch_idx, (patch_lr_tensor_norm, patch_coords) in enumerate(spliter):
                    scene_patch_diffusion_timer = {'last_time': time.time()}
                    
                    def current_scene_patch_diffusion_cb(step, total_steps):
                        nonlocal scene_patch_diffusion_timer
                        current_time = time.time()
                        step_duration = current_time - scene_patch_diffusion_timer['last_time']
                        scene_patch_diffusion_timer['last_time'] = current_time
                        _desc_for_log = f"Scene {scene_index+1} Frame {i+1}/{scene_frame_count}, Patch {patch_idx+1}/{num_patches_this_frame_for_cb}"

                        eta_seconds = step_duration * (total_steps - step) if step_duration > 0 and total_steps > 0 else 0
                        eta_formatted = format_time(eta_seconds)
                        logger.info(f"{_desc_for_log} - Diffusion: Step {step}/{total_steps}, Duration: {step_duration:.2f}s, ETA: {eta_formatted}")

                        if progress_callback:
                            progress_callback((0.3 + ((i + ((patch_idx + 1) / num_patches_this_frame_for_cb if num_patches_this_frame_for_cb > 0 else 1)) / scene_frame_count) * 0.5), f"{_desc_for_log} (Diff: {step}/{total_steps})")

                    patch_lr_video_data = patch_lr_tensor_norm
                    patch_pre_data = {'video_data': patch_lr_video_data, 'y': scene_prompt,
                                    'target_res': (int(round(patch_lr_tensor_norm.shape[-2] * upscale_factor)),
                                                    int(round(patch_lr_tensor_norm.shape[-1] * upscale_factor)))}
                    patch_data_tensor_cuda = collate_fn(patch_pre_data, model_device)
                    with torch.no_grad():
                        patch_sr_tensor_bcthw = star_model_instance.test(
                            patch_data_tensor_cuda, total_noise_levels, steps=ui_total_diffusion_steps,
                            solver_mode=solver_mode, guide_scale=cfg_scale,
                            max_chunk_len=1, vae_decoder_chunk_size=1,
                            progress_callback=current_scene_patch_diffusion_cb)
                    patch_sr_frames_uint8 = tensor2vid(patch_sr_tensor_bcthw)
                    if color_fix_method != 'None':
                        if color_fix_method == 'AdaIN':
                            patch_sr_frames_uint8 = adain_color_fix(patch_sr_frames_uint8, patch_lr_video_data)
                        elif color_fix_method == 'Wavelet':
                            patch_sr_frames_uint8 = wavelet_color_fix(patch_sr_frames_uint8, patch_lr_video_data)
                    single_patch_frame_hwc = patch_sr_frames_uint8[0]
                    result_patch_chw_01 = single_patch_frame_hwc.permute(2, 0, 1).float() / 255.0
                    spliter.update_gaussian(result_patch_chw_01.unsqueeze(0), patch_coords)
                    del patch_data_tensor_cuda, patch_sr_tensor_bcthw, patch_sr_frames_uint8
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                final_frame_tensor_chw = spliter.gather()
                final_frame_np_hwc_uint8 = (final_frame_tensor_chw.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                final_frame_bgr = cv2.cvtColor(final_frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(scene_output_frames_dir, frame_filename), final_frame_bgr)

        elif enable_sliding_window:
            # Sliding window mode processing
            processed_frame_filenames = [None] * scene_frame_count
            effective_window_size = int(window_size)
            effective_window_step = int(window_step)
            window_indices_to_process = list(range(0, scene_frame_count, effective_window_step))
            total_windows_to_process = len(window_indices_to_process)
            
            for window_iter_idx, i_start_idx in enumerate(window_indices_to_process):
                start_idx = i_start_idx
                end_idx = min(i_start_idx + effective_window_size, scene_frame_count)
                current_window_len = end_idx - start_idx
                if current_window_len == 0:
                    continue
                is_last_window_iteration = (window_iter_idx == total_windows_to_process - 1)
                if is_last_window_iteration and current_window_len < effective_window_size and scene_frame_count >= effective_window_size:
                    start_idx = max(0, scene_frame_count - effective_window_size)
                    end_idx = scene_frame_count
                    current_window_len = end_idx - start_idx
                window_lr_frames_bgr = all_lr_frames_bgr[start_idx:end_idx]
                if not window_lr_frames_bgr:
                    continue
                scene_window_diffusion_timer = {'last_time': time.time()}
                
                def current_scene_window_diffusion_cb(step, total_steps):
                    nonlocal scene_window_diffusion_timer
                    current_time = time.time()
                    step_duration = current_time - scene_window_diffusion_timer['last_time']
                    scene_window_diffusion_timer['last_time'] = current_time
                    _desc_for_log = f"Scene {scene_index+1} Window {window_iter_idx+1}/{total_windows_to_process} (frames {start_idx}-{end_idx-1})"

                    eta_seconds = step_duration * (total_steps - step) if step_duration > 0 and total_steps > 0 else 0
                    eta_formatted = format_time(eta_seconds)
                    logger.info(f"{_desc_for_log} - Diffusion: Step {step}/{total_steps}, Duration: {step_duration:.2f}s, ETA: {eta_formatted}")

                    if progress_callback:
                        progress_callback((0.3 + ((window_iter_idx + (step / total_steps if total_steps > 0 else 1)) / total_windows_to_process) * 0.5), f"{_desc_for_log} (Diff: {step}/{total_steps})")
                
                window_lr_video_data = preprocess(window_lr_frames_bgr)
                window_pre_data = {'video_data': window_lr_video_data, 'y': scene_prompt, 'target_res': (final_h, final_w)}
                window_data_cuda = collate_fn(window_pre_data, model_device)
                with torch.no_grad():
                    window_sr_tensor_bcthw = star_model_instance.test(
                        window_data_cuda, total_noise_levels, steps=ui_total_diffusion_steps,
                        solver_mode=solver_mode, guide_scale=cfg_scale,
                        max_chunk_len=current_window_len, vae_decoder_chunk_size=min(vae_chunk, current_window_len),
                        progress_callback=current_scene_window_diffusion_cb)
                window_sr_frames_uint8 = tensor2vid(window_sr_tensor_bcthw)
                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        window_sr_frames_uint8 = adain_color_fix(window_sr_frames_uint8, window_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        window_sr_frames_uint8 = wavelet_color_fix(window_sr_frames_uint8, window_lr_video_data)
                
                save_from_start_offset_local = 0
                save_to_end_offset_local = current_window_len
                if total_windows_to_process > 1:
                    overlap_amount = effective_window_size - effective_window_step
                    if overlap_amount > 0:
                        if window_iter_idx == 0:
                            save_to_end_offset_local = effective_window_size - (overlap_amount // 2)
                        elif is_last_window_iteration:
                            save_from_start_offset_local = (overlap_amount // 2)
                        else:
                            save_from_start_offset_local = (overlap_amount // 2)
                            save_to_end_offset_local = effective_window_size - (overlap_amount - save_from_start_offset_local)
                    save_from_start_offset_local = max(0, min(save_from_start_offset_local, current_window_len - 1 if current_window_len > 0 else 0))
                    save_to_end_offset_local = max(save_from_start_offset_local, min(save_to_end_offset_local, current_window_len))
                
                for k_local in range(save_from_start_offset_local, save_to_end_offset_local):
                    k_global = start_idx + k_local
                    if 0 <= k_global < scene_frame_count and processed_frame_filenames[k_global] is None:
                        frame_np_hwc_uint8 = window_sr_frames_uint8[k_local].cpu().numpy()
                        frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(scene_output_frames_dir, scene_frame_files[k_global]), frame_bgr)
                        processed_frame_filenames[k_global] = scene_frame_files[k_global]
                del window_data_cuda, window_sr_tensor_bcthw, window_sr_frames_uint8
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            num_missed_fallback = 0
            for idx_fb, fname_fb in enumerate(scene_frame_files):
                if processed_frame_filenames[idx_fb] is None:
                    num_missed_fallback += 1
                    logger.warning(f"Frame {fname_fb} (index {idx_fb}) was not processed by sliding window, copying LR frame.")
                    lr_frame_path = os.path.join(scene_input_frames_dir, fname_fb)
                    if os.path.exists(lr_frame_path):
                        shutil.copy2(lr_frame_path, os.path.join(scene_output_frames_dir, fname_fb))
                    else:
                        logger.error(f"LR frame {lr_frame_path} not found for fallback copy.")
            if num_missed_fallback > 0:
                logger.info(f"Sliding window - Copied {num_missed_fallback} LR frames as fallback for unprocessed frames.")

        else:
            # Chunked processing mode
            import math
            num_chunks = math.ceil(scene_frame_count / max_chunk_len) if max_chunk_len > 0 else 1
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_chunk_len
                end_idx = min((chunk_idx + 1) * max_chunk_len, scene_frame_count)
                current_chunk_len = end_idx - start_idx
                if current_chunk_len == 0:
                    continue

                scene_chunk_diffusion_timer = {'last_time': time.time()}
                
                def current_scene_chunk_diffusion_cb(step, total_steps):
                    nonlocal scene_chunk_diffusion_timer
                    current_time = time.time()
                    step_duration = current_time - scene_chunk_diffusion_timer['last_time']
                    scene_chunk_diffusion_timer['last_time'] = current_time
                    _desc_for_log = f"Scene {scene_index+1} Chunk {chunk_idx+1}/{num_chunks} (frames {start_idx}-{end_idx-1})"

                    eta_seconds = step_duration * (total_steps - step) if step_duration > 0 and total_steps > 0 else 0
                    eta_formatted = format_time(eta_seconds)

                    logger.info(f"{_desc_for_log} - Diffusion: Step {step}/{total_steps}, Duration: {step_duration:.2f}s, ETA: {eta_formatted}")

                    if progress_callback:
                        progress_callback((0.3 + ((chunk_idx + (step / total_steps if total_steps > 0 else 1)) / num_chunks) * 0.5), f"{_desc_for_log} (Diff: {step}/{total_steps})")

                chunk_lr_frames_bgr = all_lr_frames_bgr[start_idx:end_idx]
                chunk_lr_video_data = preprocess(chunk_lr_frames_bgr)
                chunk_pre_data = {'video_data': chunk_lr_video_data, 'y': scene_prompt, 'target_res': (final_h, final_w)}
                chunk_data_cuda = collate_fn(chunk_pre_data, model_device)

                with torch.no_grad():
                    chunk_sr_tensor_bcthw = star_model_instance.test(
                        chunk_data_cuda, total_noise_levels, steps=ui_total_diffusion_steps,
                        solver_mode=solver_mode, guide_scale=cfg_scale,
                        max_chunk_len=current_chunk_len, vae_decoder_chunk_size=min(vae_chunk, current_chunk_len),
                        progress_callback=current_scene_chunk_diffusion_cb)
                chunk_sr_frames_uint8 = tensor2vid(chunk_sr_tensor_bcthw)

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        chunk_sr_frames_uint8 = adain_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        chunk_sr_frames_uint8 = wavelet_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)

                for k, frame_name in enumerate(scene_frame_files[start_idx:end_idx]):
                    frame_np_hwc_uint8 = chunk_sr_frames_uint8[k].cpu().numpy()
                    frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(scene_output_frames_dir, frame_name), frame_bgr)

                if save_chunks and scene_output_dir and scene_name:
                    current_scene_chunks_save_path = os.path.join(scene_output_dir, "scenes", scene_name, "chunks")
                    os.makedirs(current_scene_chunks_save_path, exist_ok=True)
                    chunk_video_filename = f"chunk_{chunk_idx + 1:04d}.mp4"
                    chunk_video_path = os.path.join(current_scene_chunks_save_path, chunk_video_filename)
                    chunk_temp_assembly_dir = os.path.join(temp_dir, scene_name, f"temp_chunk_{chunk_idx+1}")
                    os.makedirs(chunk_temp_assembly_dir, exist_ok=True)
                    frames_for_this_video_chunk = []
                    for k_chunk_frame, frame_name_in_chunk in enumerate(scene_frame_files[start_idx:end_idx]):
                        src = os.path.join(scene_output_frames_dir, frame_name_in_chunk)
                        dst = os.path.join(chunk_temp_assembly_dir, f"frame_{k_chunk_frame+1:06d}.png")
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                            frames_for_this_video_chunk.append(dst)
                        else:
                            logger.warning(f"Src frame {src} not found for scene chunk video.")
                    if frames_for_this_video_chunk:
                        util_create_video_from_frames(
                            chunk_temp_assembly_dir, chunk_video_path, scene_fps,
                            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)
                        
                        # Apply RIFE interpolation to chunk if enabled
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
                                    logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: RIFE interpolation completed")
                                else:
                                    logger.warning(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: RIFE interpolation failed: {rife_message}")
                                    
                            except Exception as e_chunk_rife:
                                logger.error(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: Error during RIFE interpolation: {e_chunk_rife}")
                        
                        chunk_status_str = f"Scene {scene_index + 1} Chunk {chunk_idx + 1}/{num_chunks} (frames {start_idx+1}-{end_idx})"
                        logger.info(f"Saved scene chunk {chunk_idx+1}/{num_chunks} to: {final_chunk_video_path}")
                        yield "chunk_update", final_chunk_video_path, chunk_status_str
                    else:
                        logger.warning(f"No frames for scene chunk {chunk_idx+1}/{num_chunks}, video not created.")
                    shutil.rmtree(chunk_temp_assembly_dir)

                    if save_metadata and metadata_params_base:
                        meta_chunk_dir = os.path.join(scene_output_dir, "scenes", scene_name, "chunk_progress_metadata") if scene_output_dir and scene_name else os.path.join(temp_dir, scene_name or f"s_unknown_temp_{chunk_idx+1}", "chunk_progress_metadata")
                        os.makedirs(meta_chunk_dir, exist_ok=True)
                        
                        # Include RIFE chunk information in status
                        chunk_status_info = {
                            "current_chunk": chunk_idx + 1, 
                            "total_chunks": num_chunks, 
                            "overall_process_start_time": scene_start_time,
                            "chunk_video_path": final_chunk_video_path,
                            "original_chunk_video_path": chunk_video_path,
                            "rife_applied_to_chunk": enable_rife_interpolation and rife_apply_to_chunks,
                            "rife_multiplier_used_for_chunk": rife_multiplier if enable_rife_interpolation and rife_apply_to_chunks else None
                        }
                        
                        # Add comprehensive RIFE metadata to chunk-specific metadata
                        chunk_params_meta = metadata_params_base.copy()
                        chunk_params_meta['input_fps'] = scene_fps
                        chunk_params_meta['scene_prompt_used_for_chunk'] = scene_prompt if 'scene_prompt' in locals() and scene_prompt and scene_prompt != final_prompt else final_prompt
                        
                        if enable_rife_interpolation and rife_apply_to_chunks:
                            chunk_params_meta.update({
                                'rife_chunk_applied': True,
                                'rife_chunk_multiplier': rife_multiplier,
                                'rife_chunk_fp16': rife_fp16,
                                'rife_chunk_uhd': rife_uhd,
                                'rife_chunk_scale': rife_scale,
                                'rife_chunk_skip_static': rife_skip_static,
                                'rife_chunk_enable_fps_limit': rife_enable_fps_limit,
                                'rife_chunk_max_fps_limit': rife_max_fps_limit,
                                'rife_chunk_keep_original': rife_keep_original,
                                'rife_chunk_seed': current_seed
                            })
                        
                        metadata_handler.save_metadata(True, meta_chunk_dir, f"{scene_name}_chunk_{chunk_idx+1:04d}_progress", chunk_params_meta, chunk_status_info, logger)

                del chunk_data_cuda, chunk_sr_tensor_bcthw, chunk_sr_frames_uint8
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if save_frames and scene_output_frames_permanent:
            if progress_callback:
                progress_callback(0.85, f"Scene {scene_index + 1}: Saving processed frames...")
            for frame_file in os.listdir(scene_output_frames_dir):
                shutil.copy2(os.path.join(scene_output_frames_dir, frame_file), os.path.join(scene_output_frames_permanent, frame_file))

        if progress_callback:
            progress_callback(0.9, f"Scene {scene_index + 1}: Creating video...")
        scene_output_video = os.path.join(scene_temp_dir, f"{scene_name}.mp4")
        util_create_video_from_frames(
            scene_output_frames_dir, scene_output_video, scene_fps,
            ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger=logger)
        
        # Apply RIFE interpolation to scene if enabled
        final_scene_video = scene_output_video  # Default to original scene video
        if enable_rife_interpolation and rife_apply_to_scenes and os.path.exists(scene_output_video):
            rife_scene_start_time = time.time()
            if progress_callback:
                progress_callback(0.92, f"Scene {scene_index + 1}: Applying RIFE {rife_multiplier}x interpolation...")
            
            try:
                # Import RIFE function locally to avoid circular imports
                from .rife_interpolation import increase_fps_single
                
                # Generate RIFE scene output path
                scene_output_dir_path = os.path.dirname(scene_output_video)
                scene_name_base = os.path.splitext(os.path.basename(scene_output_video))[0]
                rife_scene_output = os.path.join(scene_output_dir_path, f"{scene_name_base}_{rife_multiplier}x_FPS.mp4")
                
                # Apply RIFE interpolation to scene
                rife_result, rife_message = increase_fps_single(
                    video_path=scene_output_video,
                    output_path=rife_scene_output,
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
                    overwrite_original=False,  # Don't overwrite for scenes, keep both
                    keep_original=rife_keep_original,
                    output_dir=scene_output_dir_path,
                    seed=current_seed,
                    logger=logger,
                    progress=None  # Don't pass progress to avoid conflicts
                )
                
                if rife_result:
                    rife_processing_time = time.time() - rife_scene_start_time
                    final_scene_video = rife_result  # Use RIFE version as final scene video
                    logger.info(f"Scene {scene_index + 1}: RIFE interpolation completed in {format_time(rife_processing_time)}")
                    
                    # Update scene FPS for RIFE version
                    from .rife_interpolation import get_video_fps
                    rife_scene_fps = get_video_fps(rife_result, logger)
                    scene_fps = rife_scene_fps  # Update scene FPS to RIFE FPS for return
                    
                    if progress_callback:
                        progress_callback(0.95, f"Scene {scene_index + 1}: RIFE interpolation complete")
                else:
                    logger.warning(f"Scene {scene_index + 1}: RIFE interpolation failed: {rife_message}")
                    if progress_callback:
                        progress_callback(0.95, f"Scene {scene_index + 1}: RIFE interpolation failed, using original")
                    
            except Exception as e_rife:
                rife_processing_time = time.time() - rife_scene_start_time
                logger.error(f"Scene {scene_index + 1}: Error during RIFE interpolation: {e_rife}. Time: {format_time(rife_processing_time)}")
                if progress_callback:
                    progress_callback(0.95, f"Scene {scene_index + 1}: RIFE error, using original")
        
        scene_duration = time.time() - scene_start_time
        if progress_callback:
            progress_callback(1.0, f"Scene {scene_index + 1}: Complete ({format_time(scene_duration)})")
        logger.info(f"Scene {scene_name} processed successfully in {format_time(scene_duration)}")

        if save_metadata and metadata_params_base and scene_output_dir:
            if progress_callback:
                progress_callback(0.95, f"Scene {scene_index+1}: Saving metadata...")

            scene_meta_dir = os.path.join(scene_output_dir, "scenes", scene_name)
            os.makedirs(scene_meta_dir, exist_ok=True)
            
            # Include RIFE information in scene metadata
            scene_data_meta = {
                "scene_index": scene_index + 1, 
                "scene_name": scene_name, 
                "scene_prompt": scene_prompt,
                "scene_frame_count": scene_frame_count, 
                "scene_fps": scene_fps,
                "scene_processing_time": scene_duration, 
                "scene_video_path": final_scene_video,  # Use final_scene_video (could be RIFE)
                "original_scene_video_path": scene_output_video,  # Always include original path
                "rife_applied": enable_rife_interpolation and rife_apply_to_scenes,
                "rife_multiplier_used": rife_multiplier if enable_rife_interpolation and rife_apply_to_scenes else None
            }
            
            # Add comprehensive RIFE metadata to scene-specific metadata
            scene_metadata_with_rife = metadata_params_base.copy()
            if enable_rife_interpolation and rife_apply_to_scenes:
                scene_metadata_with_rife.update({
                    'rife_scene_applied': True,
                    'rife_scene_multiplier': rife_multiplier,
                    'rife_scene_fp16': rife_fp16,
                    'rife_scene_uhd': rife_uhd,
                    'rife_scene_scale': rife_scale,
                    'rife_scene_skip_static': rife_skip_static,
                    'rife_scene_enable_fps_limit': rife_enable_fps_limit,
                    'rife_scene_max_fps_limit': rife_max_fps_limit,
                    'rife_scene_keep_original': rife_keep_original,
                    'rife_scene_seed': current_seed
                })
            
            final_status_info_meta = {"scene_specific_data": scene_data_meta, "processing_time_total": scene_duration}
            metadata_handler.save_metadata(True, scene_meta_dir, f"{scene_name}_metadata", scene_metadata_with_rife, final_status_info_meta, logger)

        yield "scene_complete", final_scene_video, scene_frame_count, scene_fps, generated_scene_caption

    except Exception as e:
        logger.error(f"Error processing scene {scene_index + 1}: {e}", exc_info=True)
        yield "error", str(e), None, None, None
    finally:
        if star_model_instance is not None:
            if logger: logger.info(f"Scene {scene_index + 1}: Deleting STAR model instance from memory.")
            del star_model_instance
            star_model_instance = None # Ensure it's marked as gone
        if torch.cuda.is_available():
            if logger: logger.info(f"Scene {scene_index + 1}: Clearing CUDA cache.")
            torch.cuda.empty_cache()
        # gc.collect() # Removed from here, should be handled by Python's regular GC
        if logger: logger.info(f"Scene {scene_index + 1}: Cleanup finished.") 