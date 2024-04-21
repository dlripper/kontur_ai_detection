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
from imwatermark import WatermarkEncoder, WatermarkDecoder


def add_watermark() -> None:
    """
    Add watermarks to existing images and save them to a specified directory.

    This function does the following steps:
    1. Loads the WatermarkEncoder model to embed watermarks into images.
    2. Creates directories to store watermark-embedded images.
    3. Copies a CSV file to the new directory for reference.
    4. Iterates over a set of JPEG images and embeds a predefined watermark.
    5. Saves the watermark-embedded images in the specified directory with a specified quality.

    The watermarked images can be used for testing or validation purposes.

    Parameters:
    - None

    Returns:
    - None: The function performs operations but does not return a value.

    Notes:
    - Ensure the WatermarkEncoder model is correctly imported and initialized.
    - The function creates new directories for storing watermarked images.
    - The function expects a list of images to process and embed watermarks.
    """
    WatermarkEncoder.loadModel()
    os.mkdir("data/watermark_generated")
    os.mkdir("data/watermark_generated/images")
    subprocess.run(["cp", "data/generated-or-not/test.csv", "data/watermark_generated/"])
    
    files = np.array(get_recovered("train.csv", formats=["jpeg", "jpg"]).id)
    WatermarkEncoder.loadModel()
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', 'aign'.encode('utf-8'))
    for pos, file_name in enumerate(files):        
        bgr = np.array(Image.open(f"data/generated-or-not/images/{file_name}").convert('RGB'))
        bgr_encoded = encoder.encode(bgr, 'rivaGan')
        Image.fromarray(bgr_encoded).save(f"data/watermark_generated/images/{file_name}", quality=80)


def get_added_watermarks() -> None:
    """
    Add watermarks to existing images and evaluate them with a pretrained model.

    This function does the following steps:
    1. Calls `add_watermark` to embed watermarks into existing images.
    2. Loads a ResNet-50 model with a modified output layer for inference.
    3. Conducts inference on the watermarked images and calculates log loss.
    4. Displays example comparisons between original and watermarked images.

    Parameters:
    - None

    Returns:
    - None: This function performs operations but does not return a value.

    Notes:
    - Ensure the ResNet-50 model with a modified output layer is available for inference.
    - The function relies on `add_watermark` to create watermarked images.
    - The results of inference are stored in a CSV file for further analysis.
    """
    add_watermark()
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("utils/jpeg_1.pth")["model_state_dict"])
    get_inference([model], "utils/watermark_predict.csv", path="data/watermark_generated")
    df_inf = pd.read_csv("utils/watermark_predict.csv")
    ground_truth = pd.read_csv("final_predict.csv")

    #visualization
    print("Log Loss on Watermark-added data is", sklearn.metrics.log_loss((ground_truth.target >= 0.5).astype(np.int32), df_inf.target, labels=[0,1]))
    print("You can see the following example below:")
    visualization.show_difference("data/generated-or-not/images/HXKXONqX6P.png", "data/watermark_generated/HXKXONqX6P.png")
