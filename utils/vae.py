import os
import subprocess
import visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
import cv2
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from PIL import Image
from data.recover import get_recovered


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def create_vae_variation() -> None:
    """
    Create VAE (Variational Autoencoder) variations of existing images and save them to a specified directory.

    This function does the following steps:
    1. Loads a pretrained VAE model from a specified directory.
    2. Creates directories to store the generated VAE images.
    3. Copies a CSV file to the new directory for tracking purposes.
    4. Iterates over the images in the original dataset and applies the VAE to generate new images.
    5. Saves the generated images in the specified directory.

    The VAE variations can be used for data augmentation, testing, or other analysis.

    Parameters:
    - None

    Returns:
    - None: The function performs operations but does not return a value.

    Notes:
    - Ensure the pretrained VAE model exists in the specified directory before running the function.
    - The original images are expected to be in a known location (specified in the function).
    - The generated VAE images are saved with the same file names as the originals in the specified output directory.
    """
    vae = AutoencoderKL.from_pretrained("utils/my_model_dir", local_files_only=True)
    vae.to(device)

    os.mkdir("data/vae_generated")
    os.mkdir("data/vae_generated/images")
    subprocess.run(["cp", "data/generated-or-not/test.csv", "data/vae_generated/"])
    files = os.listdir("data/generated-or-not/images")
    for _, file_name in enumerate(files):
        pic = Image.open(f"data/generated-or-not/images/{file_name}").convert('RGB')
        pic_np = np.array(pic)
        torch_pic = torch.from_numpy(np.array(pic)).to(torch.float32).permute(2, 0, 1)

        with torch.no_grad():
            pred_pic = vae.forward(torch_pic.to(device).unsqueeze(0) / 255, sample_posterior=False, return_dict=False)

        numpy_pred_pic = (pred_pic[0][0].clamp_(0, 1) * 255).permute(1, 2, 0).round().cpu().numpy().astype(np.uint8)
        Image.fromarray(numpy_pred_pic).resize(pic.size).save(f"data/vae_generated/images/{file_name}")


def get_vae_variations() -> None:
    """
    Create VAE variations and evaluate them with a pretrained model.

    This function does the following steps:
    1. Calls `create_vae_variation` to generate VAE variations of existing images.
    2. Loads a ResNet-50 model with a modified output layer to evaluate the VAE-generated images.
    3. Conducts inference on the generated images to get predictions and log loss.
    4. Displays example comparisons between the original and VAE-generated images.

    Parameters:
    - None

    Returns:
    - None: This function is designed to conduct operations but does not return a value.

    Notes:
    - Ensure the pretrained ResNet-50 model is available for loading.
    - The function depends on the `create_vae_variation` function to generate VAE images.
    - The resulting predictions are stored in a CSV file.
    """
    create_vae_variation()
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("utils/png_1.pth")["model_state_dict"])
    get_inference([model], "utils/vae_predict.csv", path="data/vae_generated")
    df_inf = pd.read_csv("utils/vae_predict.csv")

    #visualization
    print("Log Loss on VAE generated data is", sklearn.metrics.log_loss(np.ones_like(df_inf.target), df_inf.target, labels=[0,1]))
    print("You can see the following example below:")
    visualization.show_difference("data/generated-or-not/images/HXKXONqX6P.png", "data/vae_generated/HXKXONqX6P.png")
