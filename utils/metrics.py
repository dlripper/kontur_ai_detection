import os
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
from torchvision.models.resnet import ResNet50_Weights
from data.recover import get_recovered
from data.get_dataloader import CustomDataset
from torch.utils.data import DataLoader
from typing import Tuple, List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pair_fid(dist_1: np.ndarray, dist_2: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """
    Calculate the Frechet Inception Distance (FID) between two distributions.

    The FID is calculated based on the covariance matrices and mean differences
    between the two distributions.

    Args:
        dist_1 (np.ndarray): First distribution (samples x features).
        dist_2 (np.ndarray): Second distribution (samples x features).
        eps (float, optional): Small value to ensure positive definite matrices. Default is 1e-6.
    
    Returns:
        Tuple[float, float]: The FID and an additional value (used for verification or other purposes).
    """
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


def frechet_inception_distance() -> np.ndarray:
    """
    Compute the pairwise Frechet Inception Distance between different types of models.

    This function first extracts compact representations (features) from each dataset
    and then calculates the FID for each pair of distributions.

    Returns:
        np.ndarray: A matrix containing the pairwise FID values for all model types.
    """
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


def mean_gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    """
    Calculate the mean Gaussian kernel value between two sets of feature vectors.

    This function computes the mean of Gaussian kernels between each pair of 
    vectors from the two sets, using the specified sigma for the kernel function.

    Args:
        x (torch.Tensor): First set of feature vectors (shape: [n1, d]).
        y (torch.Tensor): Second set of feature vectors (shape: [n2, d]).
        sigma (float, optional): The standard deviation used in the Gaussian kernel. Default is 1.0.
    
    Returns:
        float: The mean Gaussian kernel value between the two sets.
    """
    n2 = y.size(0)
    dim = x.size(1)

    kernels = []
    y = y.unsqueeze(0)
    for i, cur_x in enumerate(x):
        cur_x = cur_x.unsqueeze(0).unsqueeze(0).expand(1, n2, dim)
        
        
        distance = torch.pow(cur_x - y, 2).sum(2)
        kernels.append(float(torch.mean(torch.exp(-distance / (2 * (sigma ** 2))))))

    return np.mean(kernels)


def pair_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> Tuple[float, float]:
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two sets of feature vectors.

    MMD is a measure of the difference between two distributions, calculated using a Gaussian kernel.

    Args:
        x (torch.Tensor): First set of feature vectors.
        y (torch.Tensor): Second set of feature vectors.
        sigma (float, optional): The standard deviation used in the Gaussian kernel. Default is 1.0.
    
    Returns:
        Tuple[float, float]: The calculated MMD and a verification value (typically the same as MMD).
    """
    kernel_xx = mean_gaussian_kernel(x, x, sigma)
    kernel_yy = mean_gaussian_kernel(y, y, sigma)
    kernel_xy = mean_gaussian_kernel(x, y, sigma)
    
    mmd = np.sqrt(kernel_xx + kernel_yy - 2 * kernel_xy)
    return mmd, mmd 


def maximum_mean_discrepancy(sigma: float = 1.0) -> np.ndarray:
    """
    Calculate the pairwise Maximum Mean Discrepancy (MMD) for a set of model types.

    This function computes the MMD for each pair of models in `model_types`, loading their feature representations from files.

    Args:
        sigma (float, optional): The standard deviation used in the Gaussian kernel. Default is 1.0.
    
    Returns:
        np.ndarray: A matrix containing the pairwise MMD values for all model types.
    """
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


def tsne_compression() -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compress feature representations using t-SNE for visualization.

    This function loads feature representations from the defined `model_types`,
    concatenates them, and applies t-SNE for dimensionality reduction. The resulting
    2D coordinates are then normalized with a standard scaler.

    Returns:
        Tuple[np.ndarray, List[str], np.ndarray]: The t-SNE coordinates, concatenated labels, 
        and unique labels from the compressed data.
    """
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


def get_density_interpretation() -> None:
    """
    Visualize data density and the Frechet Inception Distance (FID) matrix.

    This function plots the t-SNE compression of feature representations along with the 
    Frechet Inception Distance (FID) matrix as a heatmap.

    Returns:
        None: This function is designed for visualization; it doesn't return any values.
    """
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
