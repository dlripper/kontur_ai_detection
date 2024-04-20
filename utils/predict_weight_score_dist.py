import os
import torch
import numpy as np
import pandas as pd
import random
import cv2
import re

from torchvision import transforms
from utils.MANIQA.models.maniqa import MANIQA
from torch.utils.data import DataLoader
from utils.MANIQA.config import Config
from utils.MANIQA.utils.inference_process import ToTensor, Normalize
from tqdm import tqdm

from PIL import Image as PIL_Image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        print(self.img.shape)
        new_h = 224
        new_w = 224

        self.img_patches = []
        self.top = []
        self.left = []
        hor = 4
        vert = 5
        step_hor = (224 * hor - self.img.shape[2]) // (hor - 1) - 1
        step_vert = (224 * vert - self.img.shape[1]) // (vert - 1) - 1
        for i in range(num_crops):
            cur_hor = i % hor
            cur_vert = i // hor

            top =  min(cur_vert * 224 - (cur_vert) * step_vert, self.img.shape[1] - 224) 
            left = min(cur_hor * 224 - (cur_hor) * step_hor, self.img.shape[2] - 224)
            
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)
            self.top.append(top)
            self.left.append(left)
        self.img_patches = np.array(self.img_patches)
        self.top = np.array(self.top)
        self.left = np.array(self.left)
        

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        top = self.top[idx]
        left = self.left[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        sample['top'] = top
        sample['left'] = left
        return sample


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # image path
        "image_path": "generated-or-not/images/aJ0VSzX2A7.png",

        # valid times
        "num_crops": 20,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        # checkpoint path
        "ckpt_path": "MANIQA/ckpt_koniq10k.pt",
    })
    
    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()


    images_paths = ["data/imagenet_ai_0419_biggan/train/nature/n01491361_7778.jpeg", 
    "data/generated-or-not/images/Bz9RPbGAPC.jpeg", "data/generated-or-not/images/CJJ5WO9GJi.png", 
    "data/generated-or-not/images/zkQZTIwPu7.jpg"]
    for image_path in images_paths:
        Img = Image(image_path=image_path,
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            num_crops=config.num_crops)
        img_numpy = cv2.imread(image_path)
        weight = torch.zeros((img_numpy.shape[0], img_numpy.shape[1]))
        print(weight.shape)
        counter = torch.zeros_like(weight)
        scores = torch.zeros_like(weight)

        avg_score = 0
        for num in tqdm(range(config.num_crops)):
            with torch.no_grad():
                net.eval()
                patch_sample = Img.get_patch(num)
                patch = patch_sample['d_img_org'].cuda()
                top = patch_sample['top']
                left = patch_sample['left']
                patch = patch.unsqueeze(0)
                score, w_batched, f_batched = net(patch)
                score, w_batched, f_batched = score.detach().cpu(), w_batched.detach().cpu(), f_batched.detach().cpu()

                cropped_weights = torch.zeros((224, 224))
                cropped_scores = torch.zeros((224, 224))
                for i in range(28):
                    for j in range(28):
                        cropped_weights[(8*i):(8*i+8),(8*j):(8*j+8)] = torch.ones((8, 8)) *  w_batched[0][i][j]
                        cropped_scores[(8*i):(8*i+8),(8*j):(8*j+8)] = torch.ones((8, 8)) *  f_batched[0][i][j]
                weight[top:(top+224),left:(left+224)] += cropped_weights
                scores[top:(top+224),left:(left+224)] += cropped_scores
                counter[top:(top+224),left:(left+224)] += np.ones((224, 224))

                avg_score += score

        #adjust weights and scores
        mask = counter != 0
        weight[mask] /= counter[mask]
        scores[mask] /= counter[mask]
                
        w_normalized = weight.numpy()
        s_normalized = scores.numpy()

        np.save(f"visualization/{re.split(r'[/.]', image_path)[-2]}_w_heatmap.npy", w_normalized)
        np.save(f"visualization/{re.split(r'[/.]', image_path)[-2]}_s_heatmap.npy", s_normalized)
