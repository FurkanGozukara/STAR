import os
import subprocess
import tempfile
import cv2
import torch
from PIL import Image
from typing import Mapping
from einops import rearrange
import numpy as np
import torchvision.transforms.functional as transforms_F
from video_to_video.utils.logger import get_logger

logger = get_logger()


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0
    images = rearrange(video, 'b c f h w -> b f h w c')[0]
    return images


def preprocess(input_frames, use_fp16=True, force_fp32=False):
    """
    Preprocess input frames for STAR model.
    
    Args:
        input_frames: List of input frames
        use_fp16: If True, return fp16 tensors to reduce VRAM usage (default: True)
        force_fp32: If True, force fp32 even if use_fp16=True (for VAE compatibility fallback)
    
    Returns:
        Preprocessed video tensor with dtype float16 (if use_fp16=True) or float32
    """
    out_frame_list = []
    for pointer in range(len(input_frames)):
        frame = input_frames[pointer]
        frame = frame[:, :, ::-1]
        frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
        frame = transforms_F.to_tensor(frame)
        out_frame_list.append(frame)
    out_frames = torch.stack(out_frame_list, dim=0)
    out_frames.clamp_(0, 1)
    mean = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    std = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    out_frames.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
    
    # Convert to fp16 to reduce VRAM usage by ~50%, unless forced to fp32
    if use_fp16 and not force_fp32:
        try:
            out_frames = out_frames.half()
            
            # Calculate memory usage info for detailed logging
            num_elements = out_frames.numel()
            fp32_bytes = num_elements * 4  # 4 bytes per float32
            fp16_bytes = num_elements * 2  # 2 bytes per float16
            memory_saved_mb = (fp32_bytes - fp16_bytes) / (1024 * 1024)
            
            logger.info(f"video_data shape: {out_frames.shape} | dtype: {out_frames.dtype} | "
                       f"VRAM: {fp16_bytes / (1024 * 1024):.1f} MB (saved {memory_saved_mb:.1f} MB vs fp32)")
        except Exception as e:
            logger.warning(f"Failed to convert to fp16, falling back to fp32: {e}")
            # Keep as fp32 if fp16 conversion fails
            fp32_bytes = out_frames.numel() * 4
            logger.info(f"video_data shape: {out_frames.shape} | dtype: {out_frames.dtype} | "
                       f"VRAM: {fp32_bytes / (1024 * 1024):.1f} MB (fp32 fallback)")
    else:
        reason = "force_fp32=True" if force_fp32 else "use_fp16=False"
        fp32_bytes = out_frames.numel() * 4
        logger.info(f"video_data shape: {out_frames.shape} | dtype: {out_frames.dtype} | "
                   f"VRAM: {fp32_bytes / (1024 * 1024):.1f} MB ({reason})")
    
    return out_frames


def adjust_resolution(h, w, up_scale):
    if h*up_scale < 720:
        up_s = 720/h
        target_h = int(up_s*h//2*2)
        target_w = int(up_s*w//2*2)
    elif h*w*up_scale*up_scale > 1280*2048:
        up_s = np.sqrt(1280*2048/(h*w))
        target_h = int(up_s*h//2*2)
        target_w = int(up_s*w//2*2)
    else:
        target_h = int(up_scale*h//2*2)
        target_w = int(up_scale*w//2*2)
    return (target_h, target_w)


def make_mask_cond(in_f_num, interp_f_num):
    mask_cond = []
    interp_cond = [-1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(i)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    return mask_cond


def load_video(vid_path):
    capture = cv2.VideoCapture(vid_path)
    _fps = capture.get(cv2.CAP_PROP_FPS)
    _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    pointer = 0
    frame_list = []
    stride = 1
    while len(frame_list) < _total_frame_num:
        ret, frame = capture.read()
        pointer += 1
        if (not ret) or (frame is None):
            break
        if pointer >= _total_frame_num + 1:
            break
        if pointer % stride == 0:
            frame_list.append(frame)
    capture.release()
    return frame_list, _fps


def save_video(video, save_dir, file_name, fps=16.0, ffmpeg_preset="medium", ffmpeg_quality=23, ffmpeg_use_gpu=False):
    output_path = os.path.join(save_dir, file_name)
    images = [(img.numpy()).astype('uint8') for img in video]
    temp_dir = tempfile.mkdtemp()
    
    for fid, frame in enumerate(images):
        tpth = os.path.join(temp_dir, '%06d.png' % (fid + 1))
        cv2.imwrite(tpth, frame[:, :, ::-1])
    
    tmp_path = os.path.join(save_dir, 'tmp.mp4')
    
    # Get encoding configuration with automatic NVENC fallback
    try:
        from logic.nvenc_utils import get_nvenc_fallback_encoding_config, build_ffmpeg_video_encoding_args
        
        # Estimate video dimensions from first frame
        if images:
            height, width = images[0].shape[:2]
        else:
            width, height = 1920, 1080  # Default fallback
        
        encoding_config = get_nvenc_fallback_encoding_config(
            use_gpu=ffmpeg_use_gpu,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality,
            width=width,
            height=height
        )
        
        video_codec_opts = build_ffmpeg_video_encoding_args(encoding_config)
        
        cmd = f'ffmpeg -y -f image2 -framerate {fps} -i {temp_dir}/%06d.png {video_codec_opts} "{tmp_path}"'
        
        # Use centralized ffmpeg command execution
        from logic.ffmpeg_utils import run_ffmpeg_command
        success = run_ffmpeg_command(cmd, "Video Save", raise_on_error=False)
        
        if not success:
            logger.error('Save Video Error: FFmpeg command failed')
            return
        
    except ImportError:
        # If centralized system is not available, log error and return
        logger.error('Save Video Error: Centralized ffmpeg system not available')
        return
    
    os.system(f'rm -rf {temp_dir}')
    os.rename(tmp_path, output_path)

def load_frames(frame_dir, frame_name_list):
    frame_list = []
    for frame_name in frame_name_list:
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        frame_list.append(frame)
    return frame_list

def save_frames(frame_dir, frame_name_list, frame_list):
    assert len(frame_name_list) == len(frame_list), "Number of frame name list and frame list must match"
    for i in range(len(frame_name_list)):
        frame = frame_list[i]
        frame = frame[:,:,::-1]
        frame_name = frame_name_list[i]
        frame_path = os.path.join(frame_dir, frame_name)
        cv2.imwrite(frame_path, frame)


def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')