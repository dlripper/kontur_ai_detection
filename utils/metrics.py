import os
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import sqrtm
from data.recover import get_recovered
from data.get_dataloader import CustomDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pair_fid(dist_1, dist_2): 
    cov_1, cov_2 = np.cov(dist_1, rowvar=False), np.cov(dist_2, rowvar=False)
    diff = np.mean(cov_1, axis=0) - np.mean(cov_2, axis=0)
    fid = np.sum(diff**2) + np.trace(cov_1 + cov_2 - 2 * sqrtm(sqrtm(cov_1).real @ cov_2 @ sqrtm(cov_1).real).real)
    return fid, fid


def frechet_inception_distance():
    #step 1: getting compact representations
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor.to(device)

    model_types = ['_original', 'imagenet_ai_0419_biggan', 'imagenet_ai_0419_vqdm',
       'imagenet_ai_0424_sdv5', 'imagenet_ai_0424_wukong',
       'imagenet_ai_0508_adm', 'imagenet_glide', 'imagenet_midjourney']
    # for model_type in model_types:
    #     if model_type == '_original':
    #         df = get_recovered("train.csv")
    #         subset = CustomDataset(df=df.copy(), root_dir=f'data/generated-or-not/images')
    #     else:
    #         df = pd.DataFrame(columns=["id", "target"])
    #         df["id"] = os.listdir(f"data/{model_type}/train/ai")
    #         df["target"] = 1
    #         subset = CustomDataset(df=df.copy(), root_dir=f'data/{model_type}/train/ai') 
    #     dataloader = DataLoader(subset, batch_size=48, shuffle=False)
    #     real_activations = []
    #     for inputs, _, _ in tqdm(dataloader):
    #         with torch.no_grad():
    #             activations = feature_extractor(inputs.to(device)).squeeze().cpu().numpy()
    #         real_activations.append(activations)
    #     real_activations = np.concatenate(real_activations, axis=0)

    #     np.save(f"utils/{model_type.split('_')[-1]}_fid.npy", real_activations)

    #step 2: counting pairwise fid
    pairwise_distances = np.zeros((len(model_types), len(model_types)))
    for i, model_type in enumerate(model_types):
        for j in range(i + 1, len(model_types)):
            dist_1 = np.load(f"../{model_type.split('_')[-1]}_fid.npy")
            dist_2 = np.load(f"../{model_types[j].split('_')[-1]}_fid.npy")
            pairwise_distances[i, j], pairwise_distances[j, i] = pair_fid(dist_1, dist_2)


    #step 3: visualisation
    origin_of_data = [model_type.split('_')[-1] for model_type in model_types]
    sns.heatmap(pairwise_distances, cmap="viridis", annot=True, fmt=".2f", cbar=True,
            xticklabels=origin_of_data, yticklabels=origin_of_data)
    plt.title("Frechet Inception Distance Matrix")
    plt.show()


def maximum_mean_discrepancy():
    pass