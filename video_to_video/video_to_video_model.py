import os
import os.path as osp
from math import ceil
import random
from typing import Any, Dict
import time

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.modules import *
from video_to_video.utils.config import cfg
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger
from video_to_video.modules.autoencoder_kl_temporal_decoder_feature_resetting import AutoencoderKLTemporalDecoderFeatureResetting

from diffusers import AutoencoderKLTemporalDecoder

# Attempt to import format_time from a common_utils location
# This assumes secourses_app.py and logic/ are in the Python path,
# which should be the case if running from the app's root.
try:
    from logic.common_utils import format_time
except ImportError:
    # Fallback if logic.common_utils is not directly importable
    # This might happen if video_to_video_model.py is run in a context
    # where 'logic' isn't in sys.path correctly.
    # For a robust solution, ensure sys.path is configured or use relative imports if structure allows.
    def format_time(seconds):
        # Basic fallback implementation
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h)}h {int(m)}m {s:.2f}s"
        elif m > 0:
            return f"{int(m)}m {s:.2f}s"
        else:
            return f"{s:.2f}s"
    print("Warning: Could not import format_time from logic.common_utils. Using fallback implementation.")

logger = get_logger()

class VideoToVideo_sr():
    def __init__(self, opt, device=torch.device(f'cuda:0'), enable_vram_optimization=True):
        self.opt = opt
        self.device = device # torch.device(f'cuda:0')
        self.enable_vram_optimization = enable_vram_optimization
        
        # Text encoder cache for VRAM optimization
        self.text_encoder = None
        self.text_encoder_cache = {}  # Cache for encoded prompts
        self.text_encoder_loaded = False
        
        # Initialize text encoder
        self._load_text_encoder()
        
        # Cache negative prompt encoding
        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt
        
        if self.enable_vram_optimization:
            self._log_vram_usage("before negative prompt encoding")
        
        negative_y = self._encode_text(self.negative_prompt)
        self.negative_y = negative_y
        
        # Unload text encoder if optimization is enabled
        if self.enable_vram_optimization:
            self._unload_text_encoder()
            logger.info("VRAM Optimization: Text encoder unloaded after negative prompt encoding")

        # U-Net with ControlNet
        generator = ControlledV2VUNet()
        generator = generator.to(self.device)
        generator.eval()

        cfg.model_path = opt.model_path
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        if 'state_dict' in load_dict:
            load_dict = load_dict['state_dict']
        ret = generator.load_state_dict(load_dict, strict=False)
        
        # Try to use fp8 for UNet, fallback to fp16 if not supported
        self.generator_dtype = None  # Track the actual dtype used
        fp8_success = False
        
        try:
            # Check if fp8 is available and supported by the hardware
            if hasattr(torch, 'float8_e4m3fn') and torch.cuda.is_available():
                # Test if fp8 operations work on this hardware
                test_tensor = torch.randn(2, 2, device=self.device)
                test_fp8 = test_tensor.to(torch.float8_e4m3fn)
                _ = test_fp8.to(torch.float16)  # Test conversion back
                
                # If test passes, use fp8 for UNet
                self.generator = generator.to(torch.float8_e4m3fn)
                self.generator_dtype = torch.float8_e4m3fn
                fp8_success = True
                logger.info('UNet loaded with fp8 precision (float8_e4m3fn) - hardware compatible')
                
            elif hasattr(torch, 'float8_e5m2') and torch.cuda.is_available():
                # Try alternative fp8 format
                test_tensor = torch.randn(2, 2, device=self.device)
                test_fp8 = test_tensor.to(torch.float8_e5m2)
                _ = test_fp8.to(torch.float16)  # Test conversion back
                
                self.generator = generator.to(torch.float8_e5m2)
                self.generator_dtype = torch.float8_e5m2
                fp8_success = True
                logger.info('UNet loaded with fp8 precision (float8_e5m2) - hardware compatible')
            else:
                logger.info('fp8 not available in this PyTorch version, using fp16')
                
        except Exception as e:
            logger.info(f'fp8 testing failed ({e}), falling back to fp16')
            fp8_success = False
        
        # Fallback to fp16 if fp8 is not available or failed
        if not fp8_success:
            self.generator = generator.half()
            self.generator_dtype = torch.float16
            logger.info('UNet loaded with fp16 precision')
        
        logger.info('Load model path {}, with local status {}'.format(cfg.model_path, ret))

        # Noise scheduler
        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0)
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info('Build diffusion with GaussianDiffusion')

        # Temporal VAE with VRAM optimization
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
        vae.eval()
        vae.requires_grad_(False)
        
        # Explicitly convert VAE to fp16 for compatibility with fp16 inputs
        vae = vae.half()
        logger.info('VAE converted to fp16 precision for compatibility with fp16 video_data tensors')
        
        # VAE placement based on optimization setting
        if self.enable_vram_optimization:
            vae.to('cpu')  # Start on CPU to save VRAM
            self.vae_on_gpu = False
            logger.info('Build Temporal VAE (initialized on CPU for VRAM optimization)')
            self._log_vram_usage("after VAE CPU initialization")
        else:
            vae.to(self.device)  # Traditional GPU placement
            self.vae_on_gpu = True
            logger.info('Build Temporal VAE (on GPU)')
        
        self.vae = vae
        torch.cuda.empty_cache()

    def _clear_text_encoder_cache(self):
        """Clear the text encoder cache to free CPU memory"""
        if self.enable_vram_optimization and self.text_encoder_cache:
            cache_size = len(self.text_encoder_cache)
            self.text_encoder_cache.clear()
            logger.info(f"VRAM Optimization: Cleared text encoder cache ({cache_size} entries)")

    def _move_vae_to_gpu(self):
        """Move VAE to GPU when needed for encode/decode operations"""
        if self.enable_vram_optimization and not self.vae_on_gpu:
            logger.info("VRAM Optimization: Moving VAE to GPU")
            self._log_vram_usage("before VAE GPU load")
            
            # Clear CUDA cache before moving
            torch.cuda.empty_cache()
            
            try:
                self.vae.to(self.device)
                # Ensure VAE remains in fp16 precision for compatibility with fp16 inputs
                self.vae = self.vae.half()
                self.vae_on_gpu = True
                
                self._log_vram_usage("after VAE GPU load")
                logger.info("VRAM Optimization: VAE moved to GPU successfully (maintaining fp16 precision)")
            except Exception as e:
                logger.error(f"VRAM Optimization: Failed to move VAE to GPU: {e}")
                raise

    def _move_vae_to_cpu(self):
        """Move VAE to CPU to free VRAM when not needed"""
        if self.enable_vram_optimization and self.vae_on_gpu:
            logger.info("VRAM Optimization: Moving VAE to CPU")
            self._log_vram_usage("before VAE CPU move")
            
            try:
                # Move VAE to CPU
                self.vae.cpu()
                self.vae_on_gpu = False
                
                # Force garbage collection and clear CUDA cache
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Additional CUDA cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                self._log_vram_usage("after VAE CPU move")
                logger.info("VRAM Optimization: VAE moved to CPU and VRAM cleared")
            except Exception as e:
                logger.warning(f"VRAM Optimization: Error moving VAE to CPU: {e}")

    def _log_vram_usage(self, context=""):
        """Log current VRAM usage for debugging"""
        if torch.cuda.is_available() and self.enable_vram_optimization:
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
                logger.info(f"VRAM {context}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            except Exception as e:
                logger.warning(f"Could not log VRAM usage: {e}")

    def _is_fp8_dtype(self, dtype):
        """Check if dtype is fp8"""
        return dtype in [torch.float8_e4m3fn, torch.float8_e5m2] if hasattr(torch, 'float8_e4m3fn') else False

    def _get_computation_context(self):
        """Get the appropriate autocast context for mixed precision computation"""
        if hasattr(self, 'generator_dtype') and self._is_fp8_dtype(self.generator_dtype):
            # For fp8 models, use mixed precision with fp16 compute but keep weights in fp8
            return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        else:
            # For fp16 models, use standard autocast
            return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)

    def _load_text_encoder(self):
        """Load text encoder to GPU"""
        if not self.text_encoder_loaded:
            # Clean up any existing encoder first
            if self.text_encoder is not None:
                self._unload_text_encoder()
            
            if self.enable_vram_optimization:
                logger.info("VRAM Optimization: Loading text encoder to GPU")
                self._log_vram_usage("before text encoder load")
            
            # Clear CUDA cache before loading
            torch.cuda.empty_cache()
            
            text_encoder = FrozenOpenCLIPEmbedder(device=self.device, pretrained="laion2b_s32b_b79k")
            text_encoder.model.to(self.device)
            self.text_encoder = text_encoder
            self.text_encoder_loaded = True
            
            if not self.enable_vram_optimization:
                logger.info(f'Build encoder with FrozenOpenCLIPEmbedder')
            else:
                self._log_vram_usage("after text encoder load")
                logger.info("VRAM Optimization: Text encoder loaded successfully")

    def _unload_text_encoder(self):
        """Completely unload text encoder to free VRAM"""
        if self.text_encoder_loaded and self.enable_vram_optimization:
            logger.info("VRAM Optimization: Unloading text encoder from GPU")
            self._log_vram_usage("before text encoder unload")
            
            # Move to CPU first to release GPU memory
            if self.text_encoder is not None:
                try:
                    # Move main model to CPU
                    self.text_encoder.cpu()
                    if hasattr(self.text_encoder, 'model'):
                        self.text_encoder.model.cpu()
                    
                    # Clear all parameters explicitly (more aggressive cleanup)
                    if hasattr(self.text_encoder, 'model') and hasattr(self.text_encoder.model, 'parameters'):
                        for param in self.text_encoder.model.parameters():
                            if param.is_cuda:
                                param.data = param.data.cpu()
                                if param.grad is not None:
                                    param.grad = param.grad.cpu()
                    
                except Exception as e:
                    logger.warning(f"VRAM Optimization: Error moving text encoder to CPU: {e}")
            
            # Delete the text encoder
            del self.text_encoder
            self.text_encoder = None
            self.text_encoder_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Additional CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            self._log_vram_usage("after text encoder unload")
            logger.info("VRAM Optimization: Text encoder completely unloaded and VRAM cleared")

    def _encode_text(self, text_prompt):
        """Encode text with caching support"""
        # Check cache first
        if text_prompt in self.text_encoder_cache:
            if self.enable_vram_optimization:
                logger.info(f"VRAM Optimization: Using cached embedding for prompt: '{text_prompt[:50]}...'")
            return self.text_encoder_cache[text_prompt].to(self.device)
        
        # Load text encoder if needed
        if not self.text_encoder_loaded:
            self._load_text_encoder()
        
        # Encode text
        encoded = self.text_encoder(text_prompt).detach()
        
        # Cache the result
        self.text_encoder_cache[text_prompt] = encoded.cpu()  # Store on CPU to save VRAM
        
        if self.enable_vram_optimization:
            logger.info(f"VRAM Optimization: Encoded and cached prompt: '{text_prompt[:50]}...'")
            # Unload text encoder after encoding
            self._unload_text_encoder()
        
        return encoded

    def test(self, input: Dict[str, Any], total_noise_levels=1000, \
                 steps=50, solver_mode='fast', guide_scale=7.5, max_chunk_len=32, vae_decoder_chunk_size=3,
                 progress_callback=None, seed=None):
        video_data = input['video_data']
        y = input['y']
        (target_h, target_w) = input['target_res']

        video_data = F.interpolate(video_data, [target_h,target_w], mode='bilinear')

        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        video_data_feature = self.vae_encode(video_data)
        torch.cuda.empty_cache()

        y = self._encode_text(y)

        # Use the appropriate precision context for computation
        with self._get_computation_context():

            t = torch.LongTensor([total_noise_levels-1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)

            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': video_data_feature})

            torch.cuda.empty_cache()
            chunk_inds = make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len) if frames_num > max_chunk_len else None

            solver = 'dpmpp_2m_sde' # 'heun' | 'dpmpp_2m_sde' 
            sampling_start_time = time.time()
            
            # Use the UNet directly in its native precision (fp8 or fp16)
            gen_vid = self.diffusion.sample_sr(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing',
                chunk_inds=chunk_inds,
                progress_callback=progress_callback,
                seed=seed)
            torch.cuda.empty_cache()
            
            sampling_duration = time.time() - sampling_start_time
            logger.info(f'sampling, finished in {format_time(sampling_duration)} total.')

            vae_decode_start_time = time.time()
            logger.info(f'Starting VAE decoding for {frames_num} frames at {target_h}x{target_w} resolution...')
            vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=vae_decoder_chunk_size, progress_callback=progress_callback)
            vae_decode_duration = time.time() - vae_decode_start_time
            logger.info(f'temporal vae decoding, finished in {format_time(vae_decode_duration)} total.')

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:,:,h1:h+h1,w1:w+w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        torch.cuda.empty_cache()
        
        return gen_video.type(torch.float32).cpu()

    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z/self.vae.config.scaling_factor, num_frames=num_f).sample

    def vae_decode_chunk(self, z, chunk_size=3, progress_callback=None):
        # Move VAE to GPU if needed for decoding
        self._move_vae_to_gpu()
        
        z = rearrange(z, "b c f h w -> (b f) c h w")
        total_frames = z.shape[0]
        total_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        logger.info(f'VAE decoding {total_frames} frames in {total_chunks} chunks (chunk_size={chunk_size})')
        
        video = []
        for chunk_idx, ind in enumerate(range(0, z.shape[0], chunk_size)):
            chunk_start_time = time.time()
            num_f = z[ind:ind+chunk_size].shape[0]
            
            decoded_chunk = self.temporal_vae_decode(z[ind:ind+chunk_size], num_f)
            video.append(decoded_chunk)
            
            chunk_duration = time.time() - chunk_start_time
            progress_pct = ((chunk_idx + 1) / total_chunks) * 100
            
            logger.info(f'VAE Decode Chunk {chunk_idx + 1}/{total_chunks} (frames {ind}-{ind+num_f-1}) - Duration: {format_time(chunk_duration)}, Progress: {progress_pct:.1f}%')
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(
                        current_step=chunk_idx + 1,
                        total_steps=total_chunks,
                        stage="vae_decode",
                        message=f"VAE Decode Chunk {chunk_idx + 1}/{total_chunks}"
                    )
                except:
                    # If progress callback fails, continue processing
                    pass
        
        video = torch.cat(video)
        
        # Move VAE back to CPU after decoding to free VRAM
        self._move_vae_to_cpu()
        
        return video

    def vae_encode(self, t, chunk_size=1):
        # Move VAE to GPU if needed for encoding
        self._move_vae_to_gpu()
        
        num_f = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0,t.shape[0],chunk_size):
            z_list.append(self.vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        result = z * self.vae.config.scaling_factor
        
        # Move VAE back to CPU after encoding to free VRAM for diffusion
        self._move_vae_to_cpu()
        
        return result


class Vid2VidFr(VideoToVideo_sr):
    """
    Video to video model with feature resetting.
    """
    def __init__(self, opt, device=torch.device(f'cuda:0'), enable_vram_optimization=True):
        super().__init__(opt, device, enable_vram_optimization)
        
        # Replace with feature resetting VAE and apply VRAM optimization
        vae = AutoencoderKLTemporalDecoderFeatureResetting.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
        vae.eval()
        vae.requires_grad_(False)
        
        # Explicitly convert VAE to fp16 for compatibility with fp16 inputs
        vae = vae.half()
        logger.info('Feature Resetting VAE converted to fp16 precision for compatibility with fp16 video_data tensors')
        
        # VAE placement based on optimization setting
        if self.enable_vram_optimization:
            vae.to('cpu')  # Start on CPU to save VRAM
            self.vae_on_gpu = False
            logger.info('Build Temporal VAE with Feature Resetting Decoder (initialized on CPU for VRAM optimization)')
            self._log_vram_usage("after VAE FR CPU initialization")
        else:
            vae.to(self.device)  # Traditional GPU placement
            self.vae_on_gpu = True
            logger.info('Build Temporal VAE with Feature Resetting Decoder (on GPU)')
        
        self.vae = vae
        torch.cuda.empty_cache()

    def vae_decode_fr(self, z, z_prev, feature_map_prev, is_first_batch, out_win_step, out_win_overlap, progress_callback=None):
        # Move VAE to GPU if needed for feature resetting decoding
        self._move_vae_to_gpu()
        
        z = rearrange(z, "b c f h w -> (b f) c h w")
        num_f = z.shape[0]
        num_steps = int(ceil(num_f/out_win_step))
        
        logger.info(f'VAE decoding with feature resetting: {num_f} frames in {num_steps} steps (step_size={out_win_step}, overlap={out_win_overlap})')
        
        video = []
        for i in range(num_steps):
            step_start_time = time.time()
            
            # Only if both input & output 1st batch, is_first_batch is true
            if i == 0 and is_first_batch:
                is_first_batch = True
                z_prev = z[0,:,:,:].repeat(out_win_overlap,1,1,1)
            else:
                is_first_batch = False
            z_chunk = z[i*out_win_step:(i+1)*out_win_step,:,:,:]
            z_chunk = torch.cat((z_prev, z_chunk), dim=0)
            v, feature_map_prev = self.vae.decode(z_chunk / self.vae.config.scaling_factor,
                                                 feature_map_prev=feature_map_prev,
                                                 num_frames=z_chunk.shape[0],
                                                 is_first_batch=is_first_batch,
                                                 frame_overlap_num=out_win_overlap)
            v = v.sample
            video.append(v[out_win_overlap:,:,:,:]) # handle corner case.
            z_prev = z_chunk[-out_win_overlap:,:,:,:] #always ensure out_win_overlap win size

            step_duration = time.time() - step_start_time
            progress_pct = ((i + 1) / num_steps) * 100
            
            logger.info(f'VAE FR Decode Step {i + 1}/{num_steps} - Duration: {format_time(step_duration)}, Progress: {progress_pct:.1f}%')
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(
                        current_step=i + 1,
                        total_steps=num_steps,
                        stage="vae_decode_fr",
                        message=f"VAE FR Decode Step {i + 1}/{num_steps}"
                    )
                except:
                    # If progress callback fails, continue processing
                    pass

        video = torch.cat(video)
        
        # Move VAE back to CPU after feature resetting decoding to free VRAM
        self._move_vae_to_cpu()

        return video, feature_map_prev, z_prev


    def infer(self,
              input: Dict[str, Any],
              feature_map_prev: Dict,
              z_prev,
              is_first_batch: bool = False,
              out_win_step:int=1,
              out_win_overlap: int = 1,
              total_noise_levels=1000,
              steps=50,
              solver_mode='fast',
              guide_scale=7.5,
              max_chunk_len=32,
              progress_callback=None,
              seed=None):
        video_data = input['video_data']
        y = input['y']
        (target_h, target_w) = input['target_res']

        video_data = F.interpolate(video_data, [target_h, target_w], mode='bilinear')

        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        video_data_feature = self.vae_encode(video_data)
        torch.cuda.empty_cache()

        y = self._encode_text(y)

        # Use the appropriate precision context for computation  
        with self._get_computation_context():
            t = torch.LongTensor([total_noise_levels - 1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)

            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': video_data_feature})

            torch.cuda.empty_cache()
            chunk_inds = make_chunks(frames_num, interp_f_num=0,
                                     max_chunk_len=max_chunk_len) if frames_num > max_chunk_len else None

            solver = 'dpmpp_2m_sde'  # 'heun' | 'dpmpp_2m_sde'
            sampling_start_time_fr = time.time()
            
            # Use the UNet directly in its native precision (fp8 or fp16)
            gen_vid = self.diffusion.sample_sr(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing',
                chunk_inds=chunk_inds,
                progress_callback=progress_callback,
                seed=seed)
            torch.cuda.empty_cache()
            
            sampling_duration_fr = time.time() - sampling_start_time_fr
            logger.info(f'sampling, finished in {format_time(sampling_duration_fr)} total.')

            vae_decode_fr_start_time = time.time()
            logger.info(f'Starting VAE decoding with feature resetting for {frames_num} frames at {target_h}x{target_w} resolution...')
            vid_tensor_gen, feature_map_prev, z_prev = self.vae_decode_fr(z=gen_vid,
                                                                          z_prev=z_prev,
                                                                          feature_map_prev=feature_map_prev,
                                                                          is_first_batch=is_first_batch,
                                                                          out_win_step=out_win_step,
                                                                          out_win_overlap=out_win_overlap,
                                                                          progress_callback=progress_callback)
            vae_decode_fr_duration = time.time() - vae_decode_fr_start_time
            logger.info(f'temporal vae decoding with feature resetting, finished in {format_time(vae_decode_fr_duration)} total.')

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:, :, h1:h + h1, w1:w + w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        torch.cuda.empty_cache()

        return gen_video.type(torch.float32).cpu(), feature_map_prev, z_prev


def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = max_chunk_len
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    o_len = int((MAX_O_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(length, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind<length:
        if ind+window_size*1.25>=length:
            coords.append((ind,length))
            break
        else:
            coords.append((ind,ind+window_size))
            ind += stride  
    return coords
