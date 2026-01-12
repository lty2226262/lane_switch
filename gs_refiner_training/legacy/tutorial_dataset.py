import json
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch
import os
import json


class MyDataset(Dataset):
    def __init__(self, split='train', generator=None):
        self.data = []
        self.json_file = './example_kitti.json'
        with open(self.json_file, 'r') as f:
            data = json.load(f)

        self.data = data['render']
        self.gt_data = data['gt']
        self.prompt_data = data['cond']
        all_val_length = 10

        if split == 'train':
            self.data = self.data[:-all_val_length]
            self.gt_data = self.gt_data[:-all_val_length]
            self.prompt_data = self.prompt_data[:-all_val_length]
        elif split == 'val':
            self.data = self.data[-1:]
            self.gt_data = self.gt_data[-1:]
            self.prompt_data = self.prompt_data[-1:]
        #1408, 376
        self.generator = generator

        self.aspect_ratios = {
            # '1.0': (768, 768),
            '1.5': (640, 480),
            # '3.0': (1296, 432),
            '3.66': (1917, 512),
            # '4.0': (1536, 384),
        }

        self.ratio_nums = {}

        for k, v in self.aspect_ratios.items():
            self.ratio_nums[k] = 0
        

        for i in range(len(self.data)):
            render_filename = os.path.join(self.data[i])
            with Image.open(render_filename) as render:
                original_width, original_height = render.size
            aspect_ratio = original_width / original_height
            nearest_aspect_ratio = min(self.aspect_ratios.keys(), key=lambda x: abs(float(x) - aspect_ratio))
            self.ratio_nums[nearest_aspect_ratio] += 1
        print(self.ratio_nums)
        
    def __len__(self):
        return len(self.data)
    
    def get_data_info(self, idx):
        render_filename = os.path.join(self.data[idx])
        with Image.open(render_filename) as render:
            original_width, original_height = render.size
        return {'height': original_height, 'width': original_width}

    def __getitem__(self, idx):
        render_filename = os.path.join(self.data[idx])
        gt_filename = os.path.join(self.gt_data[idx])
        prompt_filename = os.path.join(self.prompt_data[idx])

        render = Image.open(render_filename)
        gt = Image.open(gt_filename)
        prompt = Image.open(prompt_filename)

        original_width, original_height = render.size
        aspect_ratio = original_width / original_height
        # Find the nearest aspect ratio
        nearest_aspect_ratio = min(self.aspect_ratios.keys(), key=lambda x: abs(float(x) - aspect_ratio))
        nearest_aspect_ratio_shape = self.aspect_ratios[nearest_aspect_ratio]

        rand_x = torch.randint(0, nearest_aspect_ratio_shape[0] - nearest_aspect_ratio_shape[1], (1,)).item()


        render = np.array(render.resize(nearest_aspect_ratio_shape, Image.ANTIALIAS))[:, rand_x:rand_x + nearest_aspect_ratio_shape[1], :]
        gt = np.array(gt.resize(nearest_aspect_ratio_shape, Image.ANTIALIAS))[:, rand_x:rand_x + nearest_aspect_ratio_shape[1], :]
        prompt_pil = prompt.resize(nearest_aspect_ratio_shape, Image.ANTIALIAS).crop((rand_x, 0, rand_x + nearest_aspect_ratio_shape[1], nearest_aspect_ratio_shape[1]))
        prompt_img = np.array(prompt_pil)

        # Normalize source images to [-1, 1].
        render = torch.Tensor((render.astype(np.float32) / 127.5) - 1.0).movedim(-1, 0) # [0, 255] -> [-1, 1]
        prompt_img = torch.Tensor((prompt_img.astype(np.float32) / 127.5) - 1.0).movedim(-1, 0) # [0, 255] -> [-1, 1]
        gt = torch.Tensor((gt.astype(np.float32) / 127.5) - 1.0).movedim(-1, 0) # [0, 255] -> [-1, 1]

        return dict(gt=gt, prompt_img=prompt_img, render=render, prompt_pil=prompt_pil)