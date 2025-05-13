import os
import torch
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict
from math import ceil

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_sr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix

from inference_utils import *

from util_image import ImageSpliterTh
from video_super_resolution.color_fix import adain_color_fix, wavelet_color_fix

logger = get_logger()


class STAR():
    def __init__(self, 
                 result_dir='./results/',
                 model_path='',
                 solver_mode='fast',
                 steps=15,
                 guide_scale=7.5,
                 upscale=4,
                 max_chunk_len=32,
                 vae_decoder_chunk_size=3,
                 color_fix_method='adain',
                 ):
        self.model_path=model_path
        logger.info('checkpoint_path: {}'.format(self.model_path))

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo_sr(model_cfg)

        steps = 15 if solver_mode == 'fast' else steps
        self.solver_mode=solver_mode
        self.steps=steps
        self.guide_scale=guide_scale
        self.upscale = upscale
        self.max_chunk_len=max_chunk_len
        self.vae_decoder_chunk_size=vae_decoder_chunk_size
        self.color_fix_method=color_fix_method

    def enhance_dir(self, input_frames_dir, input_fps, prompt, win_size, win_step, idx_i, idx_j):
        """
        Enhance the images inside a directory.
        Assume that the dir only contains images of the same format, and the name indicates the frame order.
        """
        img_name_list = os.listdir(input_frames_dir)
        img_name_list.sort()

        n_frames = len(img_name_list)
        assert n_frames >= win_size, "Input frame size should be larger than window size."
        n_steps = ceil((n_frames-win_size)/win_step) + 1

        for w_i in range(n_steps):
            w_start_idx = win_step*w_i
            w_end_idx = w_start_idx + win_size if w_i != n_steps-1 else n_frames
            w_img_name_list = img_name_list[w_start_idx:w_end_idx]
            frames = load_frames(input_frames_dir, w_img_name_list)
            video_sr = self.enhance_a_video(input_frames=frames, input_fps=input_fps, prompt=prompt)
            image_sr_list = [(img.numpy()).astype('uint8')[:,:,::-1] for img in video_sr]
            if w_i == 0:
                save_frame_list = image_sr_list[:idx_j]
                save_name_list = w_img_name_list[:idx_j]
            elif w_i==n_steps-1:
                save_frame_list = image_sr_list[idx_i:]
                save_name_list = w_img_name_list[idx_i:]
            else:
                ## Only save the frames between idx_i and idx_j
                save_frame_list = image_sr_list[idx_i:idx_j]
                save_name_list = w_img_name_list[idx_i:idx_j]
            for i in range(len(save_name_list)):
                out_path = os.path.join(self.result_dir, save_name_list[i])
                cv2.imwrite(out_path, save_frame_list[i])


    def enhance_a_video(self, input_frames, input_fps, prompt):
        text = prompt
        logger.info('text: {}'.format(text))
        caption = text + self.model.positive_prompt

        logger.info('input fps: {}'.format(input_fps))

        print('Using STableSR Tiling Approach...')


        video_data = preprocess(input_frames)

        im_spliter = ImageSpliterTh(video_data, 256, 128, sf=self.upscale)
        tile_num = len(im_spliter)
        print(f"{tile_num} tiles to process")
        for im_lq_pch, index_infos in im_spliter:
            _, _, h, w = im_lq_pch.shape
            logger.info('input resolution: {}'.format((h, w)))
            target_h, target_w = int(h * self.upscale), int(w * self.upscale)  # adjust_resolution(h, w, up_scale=4)
            logger.info('target resolution: {}'.format((target_h, target_w)))

            pre_data = {'video_data': im_lq_pch, 'y': caption}
            pre_data['target_res'] = (target_h, target_w)

            total_noise_levels = 900
            setup_seed(666)

            with torch.no_grad():
                data_tensor = collate_fn(pre_data, 'cuda:0')
                output = self.model.test(data_tensor, total_noise_levels, steps=self.steps, \
                                         solver_mode=self.solver_mode, guide_scale=self.guide_scale, \
                                         max_chunk_len=self.max_chunk_len,
                                         vae_decoder_chunk_size=self.vae_decoder_chunk_size
                                         )
                mean = torch.tensor([0.5, 0.5, 0.5], device=output.device).reshape(1, -1, 1, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5], device=output.device).reshape(1, -1, 1, 1, 1)
                output = output.mul_(std).add_(mean)
                output = output.clamp_(0, 1)
                output = output.sub_(mean).div_(std)
                output = torch.squeeze(output)
                output = torch.permute(output, (1,0,2,3))
            im_spliter.update_gaussian(output, index_infos)

        im_sr = im_spliter.gather()
        im_sr = torch.permute(im_sr, (1,0,2,3))
        im_sr = im_sr[None,:,:,:,:]
        output = tensor2vid(im_sr)
        # Using color fix
        if self.color_fix_method == 'adain':
            output = adain_color_fix(output, video_data)
        elif self.color_fix_method == 'wavelet':
            output = wavelet_color_fix(output, video_data)
        else:
            raise NotImplementedError
        return output


    

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    parser.add_argument("--file_name", type=str, help="file name")
    parser.add_argument("--model_path", type=str, default='./pretrained_weight/light_deg.pt', help="model path")
    parser.add_argument("--prompt", type=str, default='a good video', help="prompt")
    parser.add_argument("--upscale", type=float, default=4, help='up-scale')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max_chunk_len')

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | normal')
    parser.add_argument("--steps", type=int, default=15)

    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--sliding_window_size", type=int, default=12, help="sliding window size")
    parser.add_argument("--sliding_window_step", type=int, default=4, help="sliding window step")
    parser.add_argument("--save_idx_i", type=int, default=4, help="beginning index(inclusive)")
    parser.add_argument("--save_idx_j", type=int, default=8, help="end index(exclusive)")

    parser.add_argument("--vae_decoder_chunk_size",type=int,default=3,help="vae_decoder_chunk_size")

    parser.add_argument("--color_fix_method", type=str, default="adain", help="color_fix method")
    return parser.parse_args()


def main():
    
    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    model_path = args.model_path
    save_dir = args.save_dir
    upscale = args.upscale
    max_chunk_len = args.max_chunk_len

    sliding_window_size = args.sliding_window_size
    sliding_window_step = args.sliding_window_step
    save_idx_i = args.save_idx_i
    save_idx_j = args.save_idx_j

    vae_decoder_chunk_size = args.vae_decoder_chunk_size

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    assert solver_mode in ('fast', 'normal')

    star = STAR(
                result_dir=save_dir,
                model_path=model_path,
                solver_mode=solver_mode,
                steps=steps,
                guide_scale=guide_scale,
                upscale=upscale,
                max_chunk_len=max_chunk_len,
                vae_decoder_chunk_size=vae_decoder_chunk_size,
                )

    star.enhance_dir(input_frames_dir=input_path,
                     input_fps=args.input_fps,
                     prompt=prompt,
                     win_size=sliding_window_size,
                     win_step=sliding_window_step,
                     idx_i=save_idx_i,
                     idx_j=save_idx_j)


if __name__ == '__main__':
    main()
