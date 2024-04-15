import os
import torch
import numpy as np
import pandas as pd
import random
import cv2

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
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)
        
        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
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
        "image_path": "../generated-or-not/images/fPowDyaVsI.webp",

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
        "ckpt_path": "./ckpt_koniq10k.pt",
    })
    
    # data load
    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()

    df = pd.DataFrame(columns=["id", "maniqa_score"])
    files = os.listdir("data/generated-or-not/images")
    for pos, file_name in enumerate(files):
        if pos % 100 == 0:
            print(f"\n\n{pos}\n\n")
        img = PIL_Image.open(f"data/generated-or-not/images/{file_name}")
        if img.size[0] < 225 or img.size[1] < 225:
            continue
        Img = Image(image_path=f"data/generated-or-not/images/{file_name}",
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            num_crops=config.num_crops)
        avg_score = 0
        patches = []

        for i in range(config.num_crops):
            patch_sample = Img.get_patch(i)
            patch = patch_sample['d_img_org'].cuda()
            patches.append(patch.unsqueeze(0))

        # Concatenate patches along axis 0
        concatenated_patches = torch.cat(patches, dim=0)
                
        with torch.no_grad():
            net.eval()
            score = net(concatenated_patches)
            avg_score = torch.sum(score)
        df.loc[len(df)] = {"id": file_name, "maniqa_score": float(avg_score) / config.num_crops}
            
        print("Image {} score: {}".format(Img.img_name, avg_score / config.num_crops))
    df.to_csv("utils/maniqa.csv", index=False)

    