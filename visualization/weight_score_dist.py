import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from PIL import Image

from data.recover import get_recovered

def get_maniqa_weights_scores_visualisation():
    subprocess.run(["cp", "utils/predict_weight_score_dist.py", "utils/MANIQA/"]) 
    subprocess.run(["python", "utils/MANIQA/predict_weight_score_dist.py"])

    images_paths = images_paths = ["data/imagenet_ai_0419_biggan/train/nature/n01491361_7778.jpeg", 
    "data/generated-or-not/images/Bz9RPbGAPC.jpeg", "data/generated-or-not/images/CJJ5WO9GJi.png", 
    "data/generated-or-not/images/zkQZTIwPu7.jpg"]
    df = get_recovered("train.csv")
    maniqa = pd.read_csv("data/maniqa.csv")
    merged_df = pd.merge(df,maniqa, on='id', how='left')
    maniqa_genimage = pd.read_csv("data/maniqa_imagenet.csv")

    for image_path in images_paths:
        img = np.array(Image.open(image_path).convert("RGB"))

        w_normalized = np.load(f"visualization/{re.split(r'[/.]', image_path)[-2]}_w_heatmap.npy")
        s_normalized = np.load(f"visualization/{re.split(r'[/.]', image_path)[-2]}_s_heatmap.npy")

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img)
        axs[0].set_title(f'Orig Image, gt {int(np.array(df.target)[df.id == image_path.split("/")[-1]]) if image_path.split("/")[-1] in list(df.id) else 0}')
        axs[0].axis('off')

        axs[1].imshow(w_normalized, cmap='jet', interpolation='nearest')
        axs[1].set_title('Weight Heatmap')
        axs[1].axis('off')

        axs[2].imshow(s_normalized, cmap='jet', interpolation='nearest')
        axs[2].set_title('Score Heatmap')
        axs[2].axis('off')

        plt.show()