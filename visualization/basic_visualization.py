import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
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


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def show_difference(path_init, path_modified, addition=None):
    pic_np = Image.open(path_init).convert('RGB')
    pic_np = (np.array(img_transform(pic_np).permute(1, 2, 0).clamp_(0, 1)) * 255).round().astype(np.int32)
    numpy_pred_pic = np.array(Image.open(path_modified).convert('RGB'))

    grad_img_norm = np.linalg.norm(pic_np - numpy_pred_pic.astype(np.int32), axis=2)
    grad_img_norm /= (np.max(grad_img_norm) if np.max(grad_img_norm) != 0 else 1)
    cmap = plt.cm.jet  
    grad_img_gray = cmap(grad_img_norm)[:, :, :3] 

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(pic_np)
    axs[0].set_title('Initial Image')
    axs[0].axis('off')

    axs[1].imshow(numpy_pred_pic)
    axs[1].set_title('Modified Image')
    axs[1].axis('off')

    axs[2].imshow(grad_img_gray)
    axs[2].set_title('Gradient Image')
    axs[2].axis('off')

    plt.suptitle(f'Comparison of Images for addition {addition}', fontsize=16)
    plt.show()


def show_difference_predefined(path_init, path_modified):
    pic_np = np.array(Image.open(path_init).convert('RGB').resize((224, 224)))
    numpy_pred_pic = np.array(Image.open(path_modified).convert('RGB'))

    grad_img_norm = np.linalg.norm(pic_np.astype(np.int32) - numpy_pred_pic.astype(np.int32), axis=2)
    grad_img_norm /= np.max(grad_img_norm)
    cmap = plt.cm.jet  
    grad_img_gray = cmap(grad_img_norm)[:, :, :3] 

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(pic_np)
    axs[0].set_title('Initial Image')
    axs[0].axis('off')

    axs[1].imshow(numpy_pred_pic)
    axs[1].set_title('Modified Image')
    axs[1].axis('off')

    axs[2].imshow(grad_img_gray)
    axs[2].set_title('Gradient Image')
    axs[2].axis('off')

    plt.show()
