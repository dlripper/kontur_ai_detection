import os
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
from data.recover import get_recovered
from data.get_dataloader import CustomDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pair_fid(dist_1, dist_2, eps=1e-6): 
    cov_1, cov_2 = np.cov(dist_1, rowvar=False), np.cov(dist_2, rowvar=False)
    diff = np.mean(cov_1, axis=0) - np.mean(cov_2, axis=0)

    offset = np.eye(cov_1.shape[0]) * eps
    matmul = torch.from_numpy(cov_1 + offset).to(device) @ torch.from_numpy(cov_2 + offset).to(device)
    sqrt_internal = sqrtm(matmul.cpu().numpy()).real

    fid = np.sum(diff**2) + np.trace(cov_1) + np.trace(cov_2) - 2 * np.trace(sqrt_internal) 
    return fid, fid


model_types = ['_original', 'imagenet_ai_0419_biggan', 'imagenet_ai_0419_vqdm',
       'imagenet_ai_0424_sdv5', 'imagenet_ai_0424_wukong',
       'imagenet_ai_0508_adm', 'imagenet_glide', 'imagenet_midjourney']


def frechet_inception_distance():
    #step 1: getting compact representations
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor.to(device)
    for model_type in model_types:
        if model_type == '_original':
            df = get_recovered("train.csv")
            subset = CustomDataset(df=df.copy(), root_dir=f'data/generated-or-not/images')
        else:
            df = pd.DataFrame(columns=["id", "target"])
            df["id"] = os.listdir(f"data/{model_type}/train/ai")
            df["target"] = 1
            subset = CustomDataset(df=df.copy(), root_dir=f'data/{model_type}/train/ai') 
        dataloader = DataLoader(subset, batch_size=48, shuffle=False)
        real_activations = []
        for inputs, _, _ in tqdm(dataloader):
            with torch.no_grad():
                activations = feature_extractor(inputs.to(device)).squeeze().cpu().numpy()
            real_activations.append(activations)
        real_activations = np.concatenate(real_activations, axis=0)

        np.save(f"utils/{model_type.split('_')[-1]}_fid.npy", real_activations)

    #step 2: counting pairwise fid
    pairwise_distances = np.zeros((len(model_types), len(model_types)))
    for i, model_type in enumerate(model_types):
        for j in range(i + 1, len(model_types)):
            dist_1 = np.load(f"utils/{model_type.split('_')[-1]}_fid.npy") #utils
            dist_2 = np.load(f"utils/{model_types[j].split('_')[-1]}_fid.npy") #utils
            pairwise_distances[i, j], pairwise_distances[j, i] = pair_fid(dist_1, dist_2)

    return pairwise_distances


def mean_gaussian_kernel(x, y, sigma=1.0):
    n2 = y.size(0)
    dim = x.size(1)

    kernels = []
    y = y.unsqueeze(0)
    for i, cur_x in enumerate(x):
        cur_x = cur_x.unsqueeze(0).unsqueeze(0).expand(1, n2, dim)
        
        
        distance = torch.pow(cur_x - y, 2).sum(2)
        kernels.append(float(torch.mean(torch.exp(-distance / (2 * (sigma ** 2))))))

    return np.mean(kernels)


def pair_mmd(x, y, sigma=1.0):
    kernel_xx = mean_gaussian_kernel(x, x, sigma)
    kernel_yy = mean_gaussian_kernel(y, y, sigma)
    kernel_xy = mean_gaussian_kernel(x, y, sigma)
    
    mmd = np.sqrt(kernel_xx + kernel_yy - 2 * kernel_xy)
    return mmd, mmd 


def maximum_mean_discrepancy(sigma=1.0):
    pairwise_distances = np.zeros((len(model_types), len(model_types)))
    for i, model_type in enumerate(model_types):
        for j in range(i + 1, len(model_types)):
            dist_1 = np.load(f"utils/{model_type.split('_')[-1]}_fid.npy") #utils
            dist_2 = np.load(f"utils/{model_types[j].split('_')[-1]}_fid.npy") #utils
            pairwise_distances[i, j], pairwise_distances[j, i] = pair_mmd(torch.from_numpy(dist_1).to(device), torch.from_numpy(dist_2).to(device), sigma=sigma)


    origin_of_data = [model_type.split('_')[-1] for model_type in model_types]
    sns.heatmap(pairwise_distances, cmap="viridis", annot=True, fmt=".2f", cbar=True,
            xticklabels=origin_of_data, yticklabels=origin_of_data)
    plt.title("Maximum Mean Discrepancy Matrix")
    plt.show()

def tsne_compression():
    concatenated_labels = []
    for i, model_type in enumerate(model_types):
        dist_1 = np.load(f"utils/{model_type.split('_')[-1]}_fid.npy") #utils
        concatenated_labels += [model_type.split("_")[-1]] *  dist_1.shape[0]
        if i == 0:
                concatenated_data = dist_1
        else:
                concatenated_data = np.concatenate([concatenated_data, dist_1], axis=0)
    unique_labels = np.unique(concatenated_labels)

    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(concatenated_data)
    scaler = StandardScaler()
    tsne_data_normalized = scaler.fit_transform(tsne_data)

    return tsne_data_normalized, concatenated_labels, unique_labels

def tsne_comparison():
    pass

def get_density_interpretation():
    fid_pairwise_distances = frechet_inception_distance()
    tsne_data_normalized, concatenated_labels, unique_labels = tsne_compression()

    #step 3: visualisation
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    #axs_0
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        if label == "original":
            axs[0].scatter(tsne_data_normalized[np.array(concatenated_labels) == label, 0], 
                        tsne_data_normalized[np.array(concatenated_labels) == label, 1], 
                        color=colors[i], label=label, alpha=0.9)
        else:
            axs[0].scatter(tsne_data_normalized[np.array(concatenated_labels) == label, 0], 
                        tsne_data_normalized[np.array(concatenated_labels) == label, 1], 
                        color=colors[i], label=label, alpha=0.15)
    axs[0].set_xlabel('t-SNE Component 1')
    axs[0].set_ylabel('t-SNE Component 2')
    axs[0].set_title('t-SNE Visualization')
    axs[0].legend()

    #axs_1
    origin_of_data = [model_type.split('_')[-1] for model_type in model_types]
    sns.heatmap(fid_pairwise_distances, cmap="viridis", annot=True, fmt=".2f", cbar=True,
            xticklabels=origin_of_data, yticklabels=origin_of_data)
    axs[1].set_title("Frechet Inception Distance Matrix")

    plt.show()