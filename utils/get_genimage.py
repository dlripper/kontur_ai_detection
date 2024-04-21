import os
import subprocess
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import List


gen_im_classes = ['imagenet_ai_0419_biggan', 'imagenet_ai_0419_vqdm',
       'imagenet_ai_0424_sdv5', 'imagenet_ai_0424_wukong',
       'imagenet_ai_0508_adm', 'imagenet_glide', 'imagenet_midjourney']


def convert_to_jpeg(gen_im_classes: List[str]) -> None:
    """
    Convert image files from various formats to JPEG with random compression quality.

    This function iterates over different directories for each model name in `gen_im_classes`,
    converts the images to RGB, and saves them as JPEG with a randomly selected compression quality.

    Args:
        gen_im_classes (list): A list of model names that indicate which datasets to convert.

    Returns:
        None

    Notes:
    - JPEG compression quality is randomly chosen from a pre-defined set of values [69, 79].
    - The converted JPEG files are stored in new directories named after the original model name
      with "_jpeg" appended to it.
    """
    for model_name in gen_im_classes:
        os.makedirs("data/" + model_name + "_jpeg", exist_ok=True)
        for folder_name in ["/train/ai/", "/val/ai/", "/train/nature/", "/val/nature/"]:
            folder = os.listdir("data/" + model_name + folder_name)
            os.makedirs("data/" + model_name + "_jpeg" + folder_name, exist_ok=True)
            for file_name in folder:
                image = Image.open("data/" + model_name + folder_name + file_name).convert('RGB')
                cv2.imwrite("data/" + model_name + "_jpeg" + folder_name + file_name.split(".")[0] + ".jpeg", np.array(image), [int(cv2.IMWRITE_JPEG_QUALITY), random.choice([69, 79])])


def get_genimage(positive_rate: float = 0.5, format: str = "png") -> pd.DataFrame:
    """
    Generate a dataset with a specified positive-to-negative ratio and optionally convert to JPEG format.

    The function creates a dataset with a specific ratio of positive to negative samples. If the
    desired format is "jpeg" and the required directories do not exist, it triggers the conversion
    of images to JPEG. The resulting DataFrame contains the paths to the images and their corresponding
    target labels.

    Args:
        positive_rate (float): The desired ratio of positive samples in the dataset. Defaults to 0.5.
        format (str): The desired image format. Defaults to "png". If "jpeg", conversion is initiated.

    Returns:
        pd.DataFrame: A DataFrame with two columns: "id" (image path) and "target" (label, 0 or 1).

    Notes:
    - The `positive_rate` parameter defines the ratio of positive samples to negative samples in the dataset.
    - The default positive rate is 0.5, creating an even distribution of positive and negative samples.
    - If the format is "jpeg" and the required directories do not exist, `convert_to_jpeg()` is called to convert the dataset.
    """
    if not os.path.exists("data/imagenet_ai_0419_biggan"):
        subprocess.run(["bash", "data/download_genimage_dataset.sh"])
    if format == "jpeg":
        if not os.path.exists("data/imagenet_ai_0419_biggan_jpeg"):
            convert_to_jpeg()
        gen_im_classes = [class_name + "_jpeg" for class_name in gen_im_classes]
    
    df_train = pd.DataFrame(columns = ["id", "target"])
    if positive_rate > 0.5:
        nature_rate, generated_rate = (1 - positive_rate) / positive_rate, 1
    else:
        nature_rate, generated_rate = 1, positive_rate / (1 - positive_rate)

    for model_name in gen_im_classes:
        model_name = "data/" + model_name
        ai = random.sample(os.listdir(model_name + "/train/ai"), int(2000 * generated_rate))
        ai_val = random.sample(os.listdir(model_name + "/val/ai"), int(500 * generated_rate))
        for ai_pic in ai:
            df_train.loc[len(df_train)] = {"id": model_name + "/train/ai/" + ai_pic, "target": 1}
        for ai_pic in ai_val:
            df_train.loc[len(df_train)] = {"id": model_name + "/val/ai/" + ai_pic, "target": 1}
        
        nature = random.sample(os.listdir(model_name + "/train/nature"), int(2000 * nature_rate)) 
        nature_val = random.sample(os.listdir(model_name + "/val/nature"), int(500 * nature_rate))
        for nature_pic in nature:
            df_train.loc[len(df_train)] = {"id": model_name  + "/train/nature/" + nature_pic, "target": 0}
        for nature_pic in nature_val:
            df_train.loc[len(df_train)] = {"id": model_name  + "/val/nature/" + nature_pic, "target": 0}
        
        return df_train
