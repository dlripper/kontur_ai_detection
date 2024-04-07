import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


def visualise(file_names, amount_per_row=4):
    num_images = len(file_names)
    num_rows = (num_images + amount_per_row - 1) // amount_per_row  

    fig, axes = plt.subplots(num_rows, amount_per_row, figsize=(12, 3*num_rows))  

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:  
            img_path = file_names[i]
            if os.path.exists(img_path): 
                img = plt.imread(img_path)  
                ax.imshow(img) 
                ax.axis('off') 
        else:
            ax.axis('off') 

    plt.tight_layout() 
    plt.show()

def visualise_height_width_distr(df):
    unique_types = np.unique(df.format)
    for cur_type in unique_types:
        height, width = [], []

        for file in df[df.format == cur_type].id:
            image = Image.open(f"data/generated-or-not/images/{file}").convert('RGB')
            
            height.append(image.size[0])
            width.append(image.size[1])

        plt.title(f"Распределение высоты и ширины для {cur_type}")
        plt.hist(height, color='blue', alpha=0.3, label='Высота'); plt.hist(width, color='red', alpha=0.3, label='Ширина');
        plt.legend()
        plt.show()

