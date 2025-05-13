import torch


class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        bs, chn, height, width= im.shape
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, int(height*sf), int(width*sf)], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, int(height*sf), int(width*sf)], dtype=im.dtype, device=im.device)
        self.weight = self._gaussian_weights(pch_size, pch_size, bs, im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for i in range(len(starts)):
                if starts[i] + self.pch_size > length:
                    starts[i] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_pchs < self.length:
            w_start_idx = self.num_pchs // len(self.height_starts_list)
            w_start = self.width_starts_list[w_start_idx]
            w_end = w_start + self.pch_size

            h_start_idx = self.num_pchs % len(self.height_starts_list)
            h_start = self.height_starts_list[h_start_idx]
            h_end = h_start + self.pch_size

            pch = self.im_ori[:, :, h_start:h_end, w_start:w_end,]

            h_start = int(self.sf*h_start)
            h_end = int(self.sf*h_end)
            w_start = int(self.sf*w_start)
            w_end = int(self.sf*w_end)

            self.w_start, self.w_end = w_start, w_end
            self.h_start, self.h_end = h_start, h_end

            self.num_pchs += 1
        else:
            raise StopIteration()

        return pch, (h_start, h_end, w_start, w_end)

    def _gaussian_weights(self, tile_width, tile_height, nbatches, device):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = int(tile_width*self.sf)
        latent_height = int(tile_height*self.sf)

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=device), (nbatches, 3, 1, 1))

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def update_gaussian(self, pch_res, index_info):
        # pch_res: (B,C,H,W)
        # index_info: (h_start, h_end, w_start, w_end)
        h_start, h_end, w_start, w_end = index_info

        # Resize self.weight to match the spatial dimensions of pch_res
        # self.weight is initially (H_in_patch, W_in_patch)
        # pch_res is (B, C, H_out_patch, W_out_patch)
        if self.weight.shape[-2:] != pch_res.shape[-2:]:
            resized_weight = torch.nn.functional.interpolate(
                self.weight.to(pch_res.device, dtype=pch_res.dtype), # Pass self.weight (4D) directly
                size=pch_res.shape[-2:], # Target (H_out_patch, W_out_patch)
                mode='bilinear',
                align_corners=False
            )
        else:
            resized_weight = self.weight.to(pch_res.device, dtype=pch_res.dtype)

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res * resized_weight
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += resized_weight

    def gather(self, mode='gaussian'):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

def img_test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--pch_size", type=int, default=256)
    parser.add_argument("--pch_overlap", type=int, default=128)

    args = parser.parse_args()
    from PIL import Image
    from torchvision import transforms
    import os

    img = Image.open(args.img_path)
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(img)
    tensor_img = tensor_img.unsqueeze(0)
    img_spliter = ImageSpliterTh(tensor_img, args.pch_size, args.pch_overlap)
    to_pil = transforms.ToPILImage()

    for i, (pch, _) in enumerate(img_spliter):
        img_pil = to_pil(pch.squeeze())
        img_pil.save(os.path.join(args.out_path, f"{i:03d}.png"))


def video_test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--tile_num", type=int, default=15)
    parser.add_argument("--pch_size", type=int, default=256)
    parser.add_argument("--pch_overlap", type=int, default=128)

    args = parser.parse_args()
    from PIL import Image
    from torchvision import transforms
    import os

    img_name_list = sorted(os.listdir(args.in_dir))
    for img_name in img_name_list:
        img_path = os.path.join(args.in_dir, img_name)

        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)
        img_spliter = ImageSpliterTh(tensor_img, args.pch_size, args.pch_overlap)
        to_pil = transforms.ToPILImage()

        for i, (pch, _) in enumerate(img_spliter):
            img_pil = to_pil(pch.squeeze())
            sub_out_dir = f"{args.out_dir}/{i:03d}"
            if not os.path.exists(sub_out_dir):
                os.makedirs(sub_out_dir)
            img_pil.save(os.path.join(sub_out_dir, img_name))



if __name__ == "__main__":
    video_test()


