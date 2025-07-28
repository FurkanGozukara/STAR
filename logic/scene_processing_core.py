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
from .cancellation_manager import cancellation_manager, CancelledError
from .face_restoration_utils import (
    setup_codeformer_environment,
    restore_frames_batch_true,
    restore_video_frames,
    apply_face_restoration_to_scene_frames
)


def process_single_scene(
    scene_video_path, scene_index, total_scenes, temp_dir,
    final_prompt, upscale_factor, final_h, final_w, ui_total_diffusion_steps,
    solver_mode, cfg_scale, max_chunk_len, enable_chunk_optimization, vae_chunk, enable_vram_optimization, color_fix_method,
    enable_tiling, tile_size, tile_overlap, enable_context_window, context_overlap,
    save_frames, scene_output_dir, progress_callback=None,

    enable_auto_caption_per_scene=False, cogvlm_quant=0, cogvlm_unload='full',
    progress=None, save_chunks=False, chunks_permanent_save_path=None, ffmpeg_preset="medium", ffmpeg_quality_value=23, ffmpeg_use_gpu=False,
    save_metadata=False, metadata_params_base: dict = None,
    save_chunk_frames=False,
    
    # FPS decrease parameters for scenes
    enable_fps_decrease=False, fps_decrease_mode="multiplier", fps_multiplier_preset="1/2x (Half FPS)", fps_multiplier_custom=0.5, target_fps=24.0, fps_interpolation_method="drop",
    
    # RIFE interpolation parameters for scenes and chunks
    enable_rife_interpolation=False, rife_multiplier=2, rife_fp16=True, rife_uhd=False, rife_scale=1.0,
    rife_skip_static=False, rife_enable_fps_limit=False, rife_max_fps_limit=60,
    rife_apply_to_scenes=True, rife_apply_to_chunks=True, rife_keep_original=True, current_seed=99,
    
    # Image upscaler parameters for scenes
    enable_image_upscaler=False, image_upscaler_model=None, image_upscaler_batch_size=4,
    
    # Face restoration parameters for scenes
    enable_face_restoration=False, face_restoration_fidelity=0.7, enable_face_colorization=False,
    face_restoration_timing="after_upscale", face_restoration_when="after", codeformer_model=None,
    face_restoration_batch_size=4,
    
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
        enable_context_window: Whether to use context window mode
        context_overlap: Number of frames to use as context for each chunk
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
    model_device = None # Initialize for later use
    
    try:
        # Check for cancellation at the start of scene processing
        cancellation_manager.check_cancel()
        
        # Only load STAR model if not using image upscaler
        if not enable_image_upscaler:
            # Model loading moved inside
            if logger: logger.info(f"Scene {scene_index + 1}: Initializing STAR model for this scene.")
            model_load_start_time = time.time()
            
            # Check for cancellation before model loading
            cancellation_manager.check_cancel()
            
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
            
            # Check for cancellation before loading the STAR model
            cancellation_manager.check_cancel()
            
            star_model_instance = VideoToVideo_sr_class(model_cfg, device=model_device, enable_vram_optimization=enable_vram_optimization)
            
            # Check for cancellation after model loading
            cancellation_manager.check_cancel()
            
            if logger: logger.info(f"Scene {scene_index + 1}: STAR model loaded on {model_device}. Load time: {format_time(time.time() - model_load_start_time)}")
        else:
            if logger: logger.info(f"Scene {scene_index + 1}: Skipping STAR model loading (using image upscaler)")
            # Still need device for potential other operations
            model_device = torch.device(util_get_gpu_device_param(logger=logger)) if torch.cuda.is_available() else torch.device('cpu')

        def save_chunk_input_frames_scene(chunk_idx, chunk_start_frame, chunk_end_frame, frame_files, input_frames_dir, chunk_frames_save_path, logger, chunk_type="", total_chunks=0):
            """Save input frames for a specific chunk for debugging purposes in scene processing"""
            if not chunk_frames_save_path:
                return
            
            try:
                chunk_display_num = chunk_idx + 1
                chunk_frames_for_this_chunk = frame_files[chunk_start_frame:chunk_end_frame]
                
                if not chunk_frames_for_this_chunk:
                    logger.warning(f"No frames to save for {chunk_type}chunk {chunk_display_num}")
                    return
                
                # Create organized subfolder structure: chunk_frames/chunk1/ (no extra scene folder since we're already in scene directory)
                chunk_folder_name = f"chunk{chunk_display_num}"
                chunk_specific_path = os.path.join(chunk_frames_save_path, chunk_folder_name)
                os.makedirs(chunk_specific_path, exist_ok=True)
                
                saved_count = 0
                for local_idx, frame_file in enumerate(chunk_frames_for_this_chunk):
                    src_frame_path = os.path.join(input_frames_dir, frame_file)
                    if os.path.exists(src_frame_path):
                        # Use original frame numbering (chunk_start_frame + local_idx + 1 for 1-based indexing)
                        original_frame_num = chunk_start_frame + local_idx + 1
                        dst_frame_name = f"frame{original_frame_num:06d}.png"
                        dst_frame_path = os.path.join(chunk_specific_path, dst_frame_name)
                        
                        try:
                            shutil.copy2(src_frame_path, dst_frame_path)
                            saved_count += 1
                        except Exception as copy_e:
                            logger.warning(f"Failed to copy frame {frame_file} to chunk frames folder: {copy_e}")
                
                if logger:
                    logger.info(f"Saved {saved_count}/{len(chunk_frames_for_this_chunk)} input frames for {chunk_type}chunk {chunk_display_num} to: {chunk_specific_path}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Error saving chunk input frames for {chunk_type}chunk {chunk_idx + 1}: {e}")

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
        scene_chunk_frames_permanent = None  # NEW: Initialize chunk_frames path
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

        # NEW: Initialize chunk_frames directory for debugging if save_chunk_frames is enabled
        if save_chunk_frames and scene_output_dir:
            scene_frames_dir = os.path.join(scene_output_dir, "scenes", scene_name)
            scene_chunk_frames_permanent = os.path.join(scene_frames_dir, "chunk_frames")
            os.makedirs(scene_chunk_frames_permanent, exist_ok=True)
            if logger: logger.info(f"Scene {scene_index + 1}: Chunk input frames will be saved to: {scene_chunk_frames_permanent}")

        scene_prompt = final_prompt
        generated_scene_caption = None
        if enable_auto_caption_per_scene:
            if progress_callback:
                progress_callback(0.15, f"Scene {scene_index + 1}: Generating caption...")
            try:
                # Check for cancellation before auto-captioning scene
                cancellation_manager.check_cancel(f"before scene {scene_index + 1} auto-captioning")
                
                scene_caption, _ = util_auto_caption(
                    scene_video_path, cogvlm_quant, cogvlm_unload,
                    app_config_module_param.COG_VLM_MODEL_PATH, logger=logger, progress=progress
                )
                if not scene_caption.startswith("Error:") and not scene_caption.startswith("Caption generation cancelled"):
                    scene_prompt = scene_caption
                    generated_scene_caption = scene_caption
                    logger.info(f"Scene {scene_index + 1} auto-caption: {scene_caption[:100]}...")
                    if scene_index == 0:
                        logger.info(f"FIRST_SCENE_CAPTION_IMMEDIATE_UPDATE: {scene_caption}")
                else:
                    if scene_caption.startswith("Caption generation cancelled"):
                        logger.info(f"Scene {scene_index + 1} auto-captioning cancelled by user")
                        raise CancelledError("Scene auto-captioning was cancelled by the user.")
                    else:
                        logger.warning(f"Scene {scene_index + 1} auto-caption failed, using original prompt")
            except CancelledError:
                logger.info(f"Scene {scene_index + 1} processing cancelled by user")
                raise  # Re-raise to stop scene processing
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

        # FACE RESTORATION - Apply to Scene Input Frames Before Upscaling
        if enable_face_restoration and face_restoration_when == "before":
            if progress_callback:
                progress_callback(0.25, f"Scene {scene_index + 1}: Applying face restoration before upscaling...")
            
            # Create face restoration output directory for input frames
            scene_face_restored_input_dir = os.path.join(scene_temp_dir, "face_restored_input_frames")
            
            # Progress callback for face restoration
            def scene_face_restoration_input_progress_callback(progress_val, desc):
                if progress_callback:
                    # Map face restoration progress to the 0.25-0.3 range
                    mapped_progress = 0.25 + (progress_val * 0.05)
                    progress_callback(mapped_progress, desc)
            
            # Apply face restoration to input scene frames before upscaling
            face_restoration_result = apply_face_restoration_to_scene_frames(
                scene_frames_dir=scene_input_frames_dir,
                output_frames_dir=scene_face_restored_input_dir,
                fidelity_weight=face_restoration_fidelity,
                enable_colorization=enable_face_colorization,
                model_path=codeformer_model,
                batch_size=face_restoration_batch_size,
                progress_callback=scene_face_restoration_input_progress_callback,
                logger=logger
            )
            
            if face_restoration_result['success']:
                # Update input frames directory to use face-restored frames for upscaling
                scene_input_frames_dir = scene_face_restored_input_dir
                # Reload frames from face-restored directory
                all_lr_frames_bgr = []
                for frame_filename in scene_frame_files:
                    frame_lr_bgr = cv2.imread(os.path.join(scene_input_frames_dir, frame_filename))
                    if frame_lr_bgr is None:
                        logger.error(f"Could not read face-restored frame {frame_filename} from scene {scene_name}")
                        continue
                    all_lr_frames_bgr.append(frame_lr_bgr)
                logger.info(f"Scene {scene_index + 1}: Face restoration before upscaling completed successfully")
            else:
                logger.warning(f"Scene {scene_index + 1}: Face restoration before upscaling failed: {face_restoration_result['error']}")
                # Continue with original input frames if face restoration fails

        total_noise_levels = 900
        if progress_callback:
            progress_callback(0.3, f"Scene {scene_index + 1}: Starting upscaling...")

        # MAIN BRANCHING LOGIC: Image Upscaler vs STAR for scenes
        if enable_image_upscaler:
            # Route to image upscaler processing for this scene
            if logger:
                logger.info(f"Scene {scene_index + 1}: Using image upscaler instead of STAR")
            
            # Import image upscaler utilities
            from .image_upscaler_utils import (
                load_model, get_model_info, process_frames_batch,
                extract_model_filename_from_dropdown
            )
            
            try:
                # Extract actual model filename
                actual_model_filename = extract_model_filename_from_dropdown(image_upscaler_model)
                if not actual_model_filename:
                    raise Exception(f"Invalid image upscaler model: {image_upscaler_model}")
                
                # Get model path
                upscale_models_dir = app_config_module_param.UPSCALE_MODELS_DIR
                model_path = os.path.join(upscale_models_dir, actual_model_filename)
                
                if progress_callback:
                    progress_callback(0.35, f"Scene {scene_index + 1}: Loading image upscaler model...")
                
                # Load the image upscaler model
                device = "cuda" if util_get_gpu_device_param and util_get_gpu_device_param(logger=logger) != "cpu" else "cpu"
                model = load_model(model_path, device=device, logger=logger)
                
                if model is None:
                    raise Exception(f"Failed to load image upscaler model: {actual_model_filename}")
                
                model_info = get_model_info(model_path, logger)
                model_scale = model_info.get("scale", "Unknown")
                model_arch = model_info.get("architecture_name", "Unknown")
                
                if logger:
                    logger.info(f"Scene {scene_index + 1}: Loaded image upscaler {actual_model_filename} (Scale: {model_scale}x, Architecture: {model_arch})")
                
                if progress_callback:
                    progress_callback(0.4, f"Scene {scene_index + 1}: Processing frames with image upscaler...")
                
                # Process frames with image upscaler
                def scene_progress_callback(progress_val, desc):
                    if progress_callback:
                        # Map progress from 0.4 to 0.8 for frame processing
                        mapped_progress = 0.4 + (progress_val * 0.4)
                        progress_callback(mapped_progress, f"Scene {scene_index + 1}: {desc}")
                
                processed_count, failed_count = process_frames_batch(
                    frame_files=scene_frame_files,
                    input_dir=scene_input_frames_dir,
                    output_dir=scene_output_frames_dir,
                    model=model,
                    batch_size=image_upscaler_batch_size,
                    device=device,
                    progress_callback=scene_progress_callback,
                    logger=logger
                )
                
                if processed_count == 0:
                    raise Exception(f"No frames processed for scene {scene_index + 1}")
                
                if logger:
                    logger.info(f"Scene {scene_index + 1}: Image upscaler processed {processed_count} frames, {failed_count} failed")
                
                # Clean up model to free memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                error_msg = f"Scene {scene_index + 1}: Image upscaler error: {str(e)}"
                if logger:
                    logger.error(error_msg, exc_info=True)
                yield "error", error_msg, None, None, None
                return
                
        elif enable_tiling:
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
                            progress_callback=current_scene_patch_diffusion_cb, seed=current_seed)
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

        elif enable_context_window:
            # Context window processing for scenes
            logger.info(f"Scene {scene_index + 1}: Processing with context window (max_chunk_len={max_chunk_len}, context_overlap={context_overlap})")
            
            # Import the context processor module
            from .context_processor import calculate_context_chunks, get_chunk_frame_indices, validate_chunk_plan, format_chunk_plan_summary

            # Calculate context-based chunk plan for this scene
            context_chunks = calculate_context_chunks(
                total_frames=scene_frame_count,
                max_chunk_len=max_chunk_len,
                context_overlap=context_overlap
            )

            # Validate chunk plan
            is_valid, validation_errors = validate_chunk_plan(context_chunks, scene_frame_count)
            if not is_valid:
                error_msg = f"Scene {scene_index + 1}: Context chunk plan validation failed: {', '.join(validation_errors)}"
                logger.error(error_msg)
                raise Exception(error_msg)

            # Log chunk plan summary
            chunk_summary = format_chunk_plan_summary(context_chunks, scene_frame_count, max_chunk_len, context_overlap)
            logger.info(f"Scene {scene_index + 1} context processing plan:\n{chunk_summary}")

            # Process each context chunk
            processed_frame_filenames = [None] * scene_frame_count
            total_context_chunks = len(context_chunks)

            for chunk_idx, context_chunk_info in enumerate(context_chunks):
                process_start_0 = context_chunk_info['process_start']  # Already 0-based
                process_end_0 = context_chunk_info['process_end']      # Already 0-based
                output_start_0 = context_chunk_info['output_start']    # Already 0-based
                output_end_0 = context_chunk_info['output_end']        # Already 0-based
                
                current_chunk_len = process_end_0 - process_start_0 + 1
                
                if current_chunk_len == 0:
                    continue

                scene_context_diffusion_timer = {'last_time': time.time()}
                
                def current_scene_context_diffusion_cb(step, total_steps):
                    nonlocal scene_context_diffusion_timer
                    
                    # Check for cancellation at each diffusion step callback  
                    try:
                        from .cancellation_manager import cancellation_manager
                        cancellation_manager.check_cancel()
                    except ImportError:
                        # If cancellation manager is not available, continue without cancellation
                        pass
                    except Exception as e:
                        # If there's a CancelledError or other cancellation exception, re-raise it
                        if "cancel" in str(e).lower() or "cancelled" in str(e).lower():
                            raise e
                        # For other exceptions, continue without cancellation
                        pass
                    
                    current_time = time.time()
                    step_duration = current_time - scene_context_diffusion_timer['last_time']
                    scene_context_diffusion_timer['last_time'] = current_time
                    
                    _desc_for_log = f"Scene {scene_index+1} Context Chunk {chunk_idx+1}/{total_context_chunks} (processing {process_start_0+1}-{process_end_0+1}, output {output_start_0+1}-{output_end_0+1})"

                    eta_seconds = step_duration * (total_steps - step) if step_duration > 0 and total_steps > 0 else 0
                    eta_formatted = format_time(eta_seconds)

                    logger.info(f"{_desc_for_log} - Diffusion: Step {step}/{total_steps}, Duration: {step_duration:.2f}s, ETA: {eta_formatted}")

                    if progress_callback:
                        progress_callback((0.3 + ((chunk_idx + (step / total_steps if total_steps > 0 else 1)) / total_context_chunks) * 0.5), f"{_desc_for_log} (Diff: {step}/{total_steps})")

                # Get the frames to process (includes context frames)
                chunk_lr_frames_bgr = all_lr_frames_bgr[process_start_0:process_end_0+1]
                
                # NEW: Save chunk input frames for debugging if enabled
                if save_chunk_frames and scene_chunk_frames_permanent:
                    save_chunk_input_frames_scene(
                        chunk_idx=chunk_idx,
                        chunk_start_frame=process_start_0,
                        chunk_end_frame=process_end_0+1,  # End is exclusive for slicing
                        frame_files=scene_frame_files,
                        input_frames_dir=scene_input_frames_dir,
                        chunk_frames_save_path=scene_chunk_frames_permanent,
                        logger=logger,
                        chunk_type="context ",
                        total_chunks=total_context_chunks
                    )
                
                chunk_lr_video_data = preprocess(chunk_lr_frames_bgr)
                chunk_pre_data = {'video_data': chunk_lr_video_data, 'y': scene_prompt, 'target_res': (final_h, final_w)}
                chunk_data_cuda = collate_fn(chunk_pre_data, model_device)

                with torch.no_grad():
                    chunk_sr_tensor_bcthw = star_model_instance.test(
                        chunk_data_cuda, total_noise_levels, steps=ui_total_diffusion_steps,
                        solver_mode=solver_mode, guide_scale=cfg_scale,
                        max_chunk_len=current_chunk_len, vae_decoder_chunk_size=min(vae_chunk, current_chunk_len),
                        progress_callback=current_scene_context_diffusion_cb, seed=current_seed)
                chunk_sr_frames_uint8 = tensor2vid(chunk_sr_tensor_bcthw)

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        chunk_sr_frames_uint8 = adain_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        chunk_sr_frames_uint8 = wavelet_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)

                # Extract only the output frames (excluding context frames)
                output_frame_count = output_end_0 - output_start_0 + 1
                context_frame_count = context_chunk_info['context_frames']
                
                # Convert to list for easier slicing
                chunk_sr_frames_list = [frame for frame in chunk_sr_frames_uint8]
                
                # Extract the output frames: skip context frames, take only new frames
                if context_frame_count > 0:
                    output_frames = chunk_sr_frames_list[context_frame_count:context_frame_count + output_frame_count]
                else:
                    output_frames = chunk_sr_frames_list[:output_frame_count]
                
                # Get the corresponding frame names for output
                output_frame_names = scene_frame_files[output_start_0:output_end_0+1]

                # Save the output frames with correct names
                for k, (frame_tensor, frame_name) in enumerate(zip(output_frames, output_frame_names)):
                    frame_np_hwc_uint8 = frame_tensor.cpu().numpy()
                    frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(scene_output_frames_dir, frame_name), frame_bgr)
                    
                    # Mark frame as processed
                    processed_frame_filenames[output_start_0 + k] = frame_name
                    
                    # Log progress every 25 frames or at the end
                    if (k + 1) % 25 == 0 or (k + 1) == len(output_frames):
                        save_progress_msg = f"Scene {scene_index + 1} context chunk {chunk_idx + 1}/{total_context_chunks}: Saved {k + 1}/{len(output_frames)} frames to disk"
                        logger.info(save_progress_msg)

                # IMMEDIATE FRAME SAVING: Save processed frames immediately after scene chunk completion
                if save_frames and scene_output_frames_permanent:
                    scene_chunk_frames_saved_count = 0
                    for frame_name in output_frame_names:
                        src_frame_path = os.path.join(scene_output_frames_dir, frame_name)
                        dst_frame_path = os.path.join(scene_output_frames_permanent, frame_name)
                        if os.path.exists(src_frame_path) and not os.path.exists(dst_frame_path):
                            shutil.copy2(src_frame_path, dst_frame_path)
                            scene_chunk_frames_saved_count += 1
                    
                    if scene_chunk_frames_saved_count > 0:
                        immediate_save_msg = f"Immediately saved {scene_chunk_frames_saved_count} processed frames from scene {scene_index + 1} context chunk {chunk_idx + 1}/{total_context_chunks}"
                        logger.info(immediate_save_msg)

                if save_chunks and scene_output_dir and scene_name:
                    current_scene_chunks_save_path = os.path.join(scene_output_dir, "scenes", scene_name, "chunks")
                    os.makedirs(current_scene_chunks_save_path, exist_ok=True)
                    chunk_video_filename = f"chunk_{chunk_idx + 1:04d}.mp4"
                    chunk_video_path = os.path.join(current_scene_chunks_save_path, chunk_video_filename)
                    final_chunk_video_path = None  # Initialize with default value
                    
                    chunk_temp_assembly_dir = os.path.join(temp_dir, scene_name, f"temp_context_chunk_{chunk_idx+1}")
                    os.makedirs(chunk_temp_assembly_dir, exist_ok=True)
                    frames_for_this_video_chunk = []
                    for k_chunk_frame, frame_name_in_chunk in enumerate(output_frame_names):
                        src = os.path.join(scene_output_frames_dir, frame_name_in_chunk)
                        dst = os.path.join(chunk_temp_assembly_dir, f"frame_{k_chunk_frame+1:06d}.png")
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                            frames_for_this_video_chunk.append(dst)
                        else:
                            logger.warning(f"Src frame {src} not found for scene context chunk video.")
                    
                    if frames_for_this_video_chunk:
                        # Create chunk video using proper FPS
                        # For chunks, use scene FPS directly instead of duration preservation
                        # which incorrectly uses total scene duration for chunk FPS calculation
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
                                    logger.info(f"Scene {scene_index + 1} Context Chunk {chunk_idx+1}: RIFE interpolation completed")
                                else:
                                    logger.warning(f"Scene {scene_index + 1} Context Chunk {chunk_idx+1}: RIFE interpolation failed: {rife_message}")
                                    
                            except Exception as e_chunk_rife:
                                logger.error(f"Scene {scene_index + 1} Context Chunk {chunk_idx+1}: Error during RIFE interpolation: {e_chunk_rife}")
                        
                        chunk_status_str = f"Scene {scene_index + 1} Context Chunk {chunk_idx+1}/{total_context_chunks} (frames {output_start_0+1}-{output_end_0+1})"
                        logger.info(f"Saved scene context chunk {chunk_idx+1}/{total_context_chunks} to: {final_chunk_video_path}")
                        yield "chunk_update", final_chunk_video_path, chunk_status_str
                    else:
                        logger.warning(f"No frames for scene context chunk {chunk_idx+1}/{total_context_chunks}, video not created.")
                    shutil.rmtree(chunk_temp_assembly_dir)

                    if save_metadata and metadata_params_base:
                        meta_chunk_dir = os.path.join(scene_output_dir, "scenes", scene_name, "chunk_progress_metadata") if scene_output_dir and scene_name else os.path.join(temp_dir, scene_name or f"s_unknown_temp_{chunk_idx+1}", "chunk_progress_metadata")
                        os.makedirs(meta_chunk_dir, exist_ok=True)
                        
                        # Include RIFE chunk information in status and frame range
                        chunk_status_info = {
                            "current_chunk": chunk_idx + 1, 
                            "total_chunks": total_context_chunks, 
                            "overall_process_start_time": scene_start_time,
                            "chunk_video_path": final_chunk_video_path,
                            "original_chunk_video_path": chunk_video_path,
                            "rife_applied_to_chunk": enable_rife_interpolation and rife_apply_to_chunks,
                            "rife_multiplier_used_for_chunk": rife_multiplier if enable_rife_interpolation and rife_apply_to_chunks else None,
                            "chunk_frame_range": (output_start_0 + 1, output_end_0 + 1)  # 1-indexed frame range for this chunk
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
                        
                        metadata_handler.save_metadata(True, meta_chunk_dir, f"{scene_name}_context_chunk_{chunk_idx+1:04d}_progress", chunk_params_meta, chunk_status_info, logger)
                else:
                    # Initialize final_chunk_video_path for metadata even when save_chunks is disabled
                    final_chunk_video_path = None

        else:  # Regular chunked processing (non-context mode)
            # Chunked processing mode with optimization
            import math
            
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
                    total_frames=scene_frame_count,
                    max_chunk_len=max_chunk_len,
                    logger=logger
                )
                
                # Log optimization summary
                log_chunk_optimization_summary(chunk_boundaries, scene_frame_count, max_chunk_len, logger)
                
                num_chunks = len(chunk_boundaries)
                
            else:
                # Standard chunking (fallback)
                num_chunks = math.ceil(scene_frame_count / max_chunk_len) if max_chunk_len > 0 else 1
                chunk_boundaries = []
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * max_chunk_len
                    end_idx = min((chunk_idx + 1) * max_chunk_len, scene_frame_count)
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
                chunk_idx = chunk_info['chunk_idx']
                start_idx = chunk_info['start_idx']
                end_idx = chunk_info['end_idx']
                process_start_idx = chunk_info['process_start_idx']
                process_end_idx = chunk_info['process_end_idx']
                current_chunk_len = process_end_idx - process_start_idx
                
                if current_chunk_len == 0:
                    continue

                scene_chunk_diffusion_timer = {'last_time': time.time()}
                
                def current_scene_chunk_diffusion_cb(step, total_steps):
                    nonlocal scene_chunk_diffusion_timer
                    
                    # Check for cancellation at each diffusion step callback  
                    try:
                        from .cancellation_manager import cancellation_manager
                        cancellation_manager.check_cancel()
                    except ImportError:
                        # If cancellation manager is not available, continue without cancellation
                        pass
                    except Exception as e:
                        # If there's a CancelledError or other cancellation exception, re-raise it
                        if "cancel" in str(e).lower() or "cancelled" in str(e).lower():
                            raise e
                        # For other exceptions, continue without cancellation
                        pass
                    
                    current_time = time.time()
                    step_duration = current_time - scene_chunk_diffusion_timer['last_time']
                    scene_chunk_diffusion_timer['last_time'] = current_time
                    
                    if enable_chunk_optimization and process_start_idx != start_idx:
                        _desc_for_log = f"Scene {scene_index+1} Chunk {chunk_idx+1}/{num_chunks} (processing {process_start_idx}-{process_end_idx-1}, output {start_idx}-{end_idx-1})"
                    else:
                        _desc_for_log = f"Scene {scene_index+1} Chunk {chunk_idx+1}/{num_chunks} (frames {start_idx}-{end_idx-1})"

                    eta_seconds = step_duration * (total_steps - step) if step_duration > 0 and total_steps > 0 else 0
                    eta_formatted = format_time(eta_seconds)

                    logger.info(f"{_desc_for_log} - Diffusion: Step {step}/{total_steps}, Duration: {step_duration:.2f}s, ETA: {eta_formatted}")

                    if progress_callback:
                        progress_callback((0.3 + ((chunk_idx + (step / total_steps if total_steps > 0 else 1)) / num_chunks) * 0.5), f"{_desc_for_log} (Diff: {step}/{total_steps})")

                # Get the frames to process (may be more than output frames for optimized chunks)
                chunk_lr_frames_bgr = all_lr_frames_bgr[process_start_idx:process_end_idx]
                
                # NEW: Save chunk input frames for debugging if enabled
                if save_chunk_frames and scene_chunk_frames_permanent:
                    save_chunk_input_frames_scene(
                        chunk_idx=chunk_idx,
                        chunk_start_frame=process_start_idx,
                        chunk_end_frame=process_end_idx,  # End is exclusive for slicing
                        frame_files=scene_frame_files,
                        input_frames_dir=scene_input_frames_dir,
                        chunk_frames_save_path=scene_chunk_frames_permanent,
                        logger=logger,
                        chunk_type="",
                        total_chunks=num_chunks
                    )
                
                chunk_lr_video_data = preprocess(chunk_lr_frames_bgr)
                chunk_pre_data = {'video_data': chunk_lr_video_data, 'y': scene_prompt, 'target_res': (final_h, final_w)}
                chunk_data_cuda = collate_fn(chunk_pre_data, model_device)

                with torch.no_grad():
                    chunk_sr_tensor_bcthw = star_model_instance.test(
                        chunk_data_cuda, total_noise_levels, steps=ui_total_diffusion_steps,
                        solver_mode=solver_mode, guide_scale=cfg_scale,
                        max_chunk_len=current_chunk_len, vae_decoder_chunk_size=min(vae_chunk, current_chunk_len),
                        progress_callback=current_scene_chunk_diffusion_cb, seed=current_seed)
                chunk_sr_frames_uint8 = tensor2vid(chunk_sr_tensor_bcthw)

                if color_fix_method != 'None':
                    if color_fix_method == 'AdaIN':
                        chunk_sr_frames_uint8 = adain_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)
                    elif color_fix_method == 'Wavelet':
                        chunk_sr_frames_uint8 = wavelet_color_fix(chunk_sr_frames_uint8, chunk_lr_video_data)

                # Extract only the output frames (for optimized chunks, this trims the result)
                if enable_chunk_optimization:
                    # Convert to list for easier slicing
                    chunk_sr_frames_list = [frame for frame in chunk_sr_frames_uint8]
                    # Extract the output frames using chunk optimization logic
                    output_frames = extract_output_frames_from_processed(chunk_sr_frames_list, chunk_info)
                    # Get the corresponding frame names for output
                    output_frame_names = get_output_frame_names(scene_frame_files, chunk_info)
                else:
                    # Standard processing - use all frames
                    output_frames = [frame for frame in chunk_sr_frames_uint8]
                    output_frame_names = scene_frame_files[start_idx:end_idx]

                # Save the output frames with correct names
                for k, (frame_tensor, frame_name) in enumerate(zip(output_frames, output_frame_names)):
                    frame_np_hwc_uint8 = frame_tensor.cpu().numpy()
                    # Ensure the frame is uint8 format to avoid CV2 depth issues
                    if frame_np_hwc_uint8.dtype != np.uint8:
                        frame_np_hwc_uint8 = frame_np_hwc_uint8.astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_np_hwc_uint8, cv2.COLOR_RGB2BGR)
                    frame_output_path = os.path.join(scene_output_frames_dir, frame_name)
                    cv2.imwrite(frame_output_path, frame_bgr)
                    # Verify frame was actually written
                    if not os.path.exists(frame_output_path):
                        logger.error(f"Failed to write frame: {frame_output_path}")
                    else:
                        logger.debug(f"Successfully wrote frame: {frame_output_path}")
                    
                    # Log progress every 25 frames or at the end
                    if (k + 1) % 25 == 0 or (k + 1) == len(output_frames):
                        save_progress_msg = f"Scene {scene_index + 1} direct chunk {chunk_idx + 1}/{num_chunks}: Saved {k + 1}/{len(output_frames)} frames to disk"
                        logger.info(save_progress_msg)
                
                # Debug: List what frames are actually in the output directory
                if os.path.exists(scene_output_frames_dir):
                    actual_frames = os.listdir(scene_output_frames_dir)
                    logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: {len(actual_frames)} frames written to {scene_output_frames_dir}")
                    logger.debug(f"Frame files: {sorted(actual_frames)[:5]}...")  # Show first 5 files
                else:
                    logger.error(f"Output frames directory does not exist: {scene_output_frames_dir}")

                # IMMEDIATE FRAME SAVING: Save processed frames immediately after scene chunk completion
                if save_frames and scene_output_frames_permanent:
                    scene_chunk_frames_saved_count = 0
                    for frame_name in output_frame_names:
                        src_frame_path = os.path.join(scene_output_frames_dir, frame_name)
                        dst_frame_path = os.path.join(scene_output_frames_permanent, frame_name)
                        if os.path.exists(src_frame_path) and not os.path.exists(dst_frame_path):
                            shutil.copy2(src_frame_path, dst_frame_path)
                            scene_chunk_frames_saved_count += 1
                    
                    if scene_chunk_frames_saved_count > 0:
                        immediate_save_msg = f"Immediately saved {scene_chunk_frames_saved_count} processed frames from scene {scene_index + 1} chunk {chunk_idx + 1}/{num_chunks}"
                        logger.info(immediate_save_msg)

                if save_chunks and scene_output_dir and scene_name:
                    current_scene_chunks_save_path = os.path.join(scene_output_dir, "scenes", scene_name, "chunks")
                    os.makedirs(current_scene_chunks_save_path, exist_ok=True)
                    chunk_video_filename = f"chunk_{chunk_idx + 1:04d}.mp4"
                    chunk_video_path = os.path.join(current_scene_chunks_save_path, chunk_video_filename)
                    final_chunk_video_path = None  # Initialize with default value
                    
                    chunk_temp_assembly_dir = os.path.join(temp_dir, scene_name, f"temp_chunk_{chunk_idx+1}")
                    os.makedirs(chunk_temp_assembly_dir, exist_ok=True)
                    frames_for_this_video_chunk = []
                    logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: Looking for {len(output_frame_names)} frames in {scene_output_frames_dir}")
                    logger.debug(f"Expected frame names: {output_frame_names[:3]}...")  # Show first 3 expected names
                    
                    for k_chunk_frame, frame_name_in_chunk in enumerate(output_frame_names):
                        src = os.path.join(scene_output_frames_dir, frame_name_in_chunk)
                        dst = os.path.join(chunk_temp_assembly_dir, f"frame_{k_chunk_frame+1:06d}.png")
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                            frames_for_this_video_chunk.append(dst)
                        else:
                            logger.warning(f"Src frame {src} not found for scene chunk video.")
                    
                    logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: Found {len(frames_for_this_video_chunk)} out of {len(output_frame_names)} expected frames")
                    
                    if frames_for_this_video_chunk:
                        # Create chunk video using proper FPS
                        # For chunks, use scene FPS directly instead of duration preservation
                        # which incorrectly uses total scene duration for chunk FPS calculation
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
                        
                        # Include RIFE chunk information in status and frame range
                        chunk_status_info = {
                            "current_chunk": chunk_idx + 1, 
                            "total_chunks": num_chunks, 
                            "overall_process_start_time": scene_start_time,
                            "chunk_video_path": final_chunk_video_path,
                            "original_chunk_video_path": chunk_video_path,
                            "rife_applied_to_chunk": enable_rife_interpolation and rife_apply_to_chunks,
                            "rife_multiplier_used_for_chunk": rife_multiplier if enable_rife_interpolation and rife_apply_to_chunks else None,
                            "chunk_frame_range": (start_idx + 1, end_idx)  # 1-indexed frame range for this chunk
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
                else:
                    # Initialize final_chunk_video_path for metadata even when save_chunks is disabled
                    final_chunk_video_path = None

                del chunk_data_cuda, chunk_sr_tensor_bcthw, chunk_sr_frames_uint8
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # FACE RESTORATION - Apply to Scene Frames After Upscaling
        if enable_face_restoration and face_restoration_when == "after":
            if progress_callback:
                progress_callback(0.8, f"Scene {scene_index + 1}: Applying face restoration...")
            
            # Create face restoration output directory
            scene_face_restored_frames_dir = os.path.join(scene_temp_dir, "face_restored_frames")
            
            # Progress callback for face restoration
            def scene_face_restoration_progress_callback(progress_val, desc):
                if progress_callback:
                    # Map face restoration progress to the 0.8-0.85 range
                    mapped_progress = 0.8 + (progress_val * 0.05)
                    progress_callback(mapped_progress, desc)
            
            # Apply face restoration to upscaled scene frames
            face_restoration_result = apply_face_restoration_to_scene_frames(
                scene_frames_dir=scene_output_frames_dir,
                output_frames_dir=scene_face_restored_frames_dir,
                fidelity_weight=face_restoration_fidelity,
                enable_colorization=enable_face_colorization,
                model_path=codeformer_model,
                batch_size=face_restoration_batch_size,
                progress_callback=scene_face_restoration_progress_callback,
                logger=logger
            )
            
            if face_restoration_result['success']:
                # Update output frames directory to use face-restored frames
                scene_output_frames_dir = scene_face_restored_frames_dir
                logger.info(f"Scene {scene_index + 1}: Face restoration completed successfully")
            else:
                logger.warning(f"Scene {scene_index + 1}: Face restoration failed: {face_restoration_result['error']}")
                # Continue with original upscaled frames if face restoration fails

        if save_frames and scene_output_frames_permanent:
            if progress_callback:
                progress_callback(0.85, f"Scene {scene_index + 1}: Saving processed frames...")
            for frame_file in os.listdir(scene_output_frames_dir):
                shutil.copy2(os.path.join(scene_output_frames_dir, frame_file), os.path.join(scene_output_frames_permanent, frame_file))

        if progress_callback:
            progress_callback(0.9, f"Scene {scene_index + 1}: Creating video...")
        scene_output_video = os.path.join(scene_temp_dir, f"{scene_name}.mp4")
        # Use duration-preserved video creation for main scene video
        from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
        create_video_from_frames_with_duration_preservation(
            scene_output_frames_dir, scene_output_video, scene_video_path,
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
                "rife_multiplier_used": rife_multiplier if enable_rife_interpolation and rife_apply_to_scenes else None,
                "scene_frame_range": (1, scene_frame_count)  # 1-indexed frame range for this scene
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

    except CancelledError as e_cancel:
        logger.info(f"Scene {scene_index + 1} processing cancelled by user")
        
        # Check if we have any completed chunks that can be merged into a partial scene video
        partial_scene_video = None
        try:
            if save_chunks and scene_output_dir:
                # Compute the actual chunks directory where chunks are saved
                scene_name = f"scene_{scene_index+1:04d}"
                actual_chunks_directory = os.path.join(scene_output_dir, "scenes", scene_name, "chunks")
                
                # Look for completed chunk videos in the actual chunks directory
                chunk_videos = []
                if os.path.exists(actual_chunks_directory):
                    for chunk_file in os.listdir(actual_chunks_directory):
                        if chunk_file.startswith("chunk_") and chunk_file.endswith(".mp4"):
                            chunk_path = os.path.join(actual_chunks_directory, chunk_file)
                            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                                chunk_videos.append(chunk_path)
                else:
                    logger.info(f"Scene {scene_index + 1}: Chunks directory does not exist: {actual_chunks_directory}")
                
                if chunk_videos:
                    # Sort chunk videos to ensure proper order
                    chunk_videos.sort()
                    logger.info(f"Scene {scene_index + 1}: Found {len(chunk_videos)} completed chunks for partial scene creation")
                    
                    # Create partial scene video path
                    partial_scene_video = os.path.join(temp_dir, f"{scene_name}_partial.mp4")
                    
                    # Use scene merging utility to combine chunks
                    from .scene_utils import merge_scene_videos
                    merge_success = merge_scene_videos(
                        chunk_videos, partial_scene_video, temp_dir,
                        ffmpeg_preset, ffmpeg_quality_value, ffmpeg_use_gpu, logger, allow_partial_merge=True
                    )
                    
                    if merge_success and os.path.exists(partial_scene_video):
                        logger.info(f"Scene {scene_index + 1}: Successfully created partial scene video from {len(chunk_videos)} chunks: {partial_scene_video}")
                        yield "scene_complete", partial_scene_video, None, None, None
                        return
                    else:
                        logger.warning(f"Scene {scene_index + 1}: Failed to merge chunks into partial scene video")
                        partial_scene_video = None
                else:
                    logger.info(f"Scene {scene_index + 1}: No completed chunks found for partial scene creation")
            else:
                logger.info(f"Scene {scene_index + 1}: Chunk saving not enabled or scene_output_dir not available, cannot create partial scene video")
        except Exception as e_partial:
            logger.error(f"Scene {scene_index + 1}: Error creating partial scene video: {e_partial}", exc_info=True)
            partial_scene_video = None
        
        # If we couldn't create a partial scene video, yield the cancellation error
        yield "error", "Scene processing cancelled by user", None, None, None
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