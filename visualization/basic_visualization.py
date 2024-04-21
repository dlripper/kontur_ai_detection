import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import List, Optional
from PIL import Image


def visualise(file_names: List[str], amount_per_row: int = 4) -> None:
    """
    Visualize a grid of images based on a list of file names.

    This function creates a grid layout to display images given a list of file names.
    It adapts the number of rows based on the number of images and the specified number
    of images per row.

    Parameters:
    - file_names: A list of strings representing the paths to the image files.
    - amount_per_row: The number of images to display per row (default is 4).

    Returns:
    - None: This function shows a plot and does not return any value.
    """
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


def visualise_height_width_distr(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of image height and width for different formats.

    This function creates histograms to represent the distribution of image
    heights and widths for each unique image format in the given DataFrame.
    Each format has its own histogram, and they are shown with different colors.

    Parameters:
    - df: A pandas DataFrame containing information about the images.
          It is expected to have at least two columns: 'format' and 'id'.
          The 'format' column indicates the format of the images (e.g., "png", "jpeg"),
          and the 'id' column represents the filename of the image.

    Returns:
    - None: The function displays the histograms but does not return a value.
    """
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


def show_difference(path_init: str, path_modified: str, addition: Optional[str] = None) -> None:
    """
    Show the visual difference between two images.

    This function displays the original and modified images, and a third image
    that visualizes the gradient difference between them. If an optional
    "addition" argument is provided, it adds a relevant title.

    Parameters:
    - path_init: The file path to the initial image.
    - path_modified: The file path to the modified image.
    - addition: An optional string to be included in the title.

    Returns:
    - None: The function displays a plot with three images but does not return a value.
    """
    if addition is not None:
        pic_np = Image.open(path_init).convert('RGB')
        pic_np = (np.array(img_transform(pic_np).permute(1, 2, 0).clamp_(0, 1)) * 255).round().astype(np.int32)
        numpy_pred_pic = np.array(Image.open(path_modified).convert('RGB'))
    else:
        pic_np = np.array(Image.open(path_init).convert('RGB'))
        numpy_pred_pic = np.array(Image.open(path_modified).convert('RGB'))


    grad_img_norm = np.linalg.norm(pic_np - numpy_pred_pic.astype(np.int32), axis=2)
    grad_img_norm /= (np.max(grad_img_norm) if np.max(grad_img_norm) != 0 else 1)
    cmap = plt.cm.jet  
    grad_img_gray = cmap(grad_img_norm)[:, :, :3] 
    # diff_img = np.clip(pic_np.astype(np.int32) - numpy_pred_pic.astype(np.int32) + 127, 0, 255)

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

    if addition is not None:
        plt.suptitle(f'Comparison of Images for addition {addition}', fontsize=16)
    plt.show()


def show_attack_results(buf: pd.DataFrame) -> None:
    """
    Display the attack results using bar plots.

    This function generates two bar plots. The first plot shows the Peak Signal-to-Noise Ratio (PSNR) of the input and
    modified input as a function of the allowed addition. The second plot shows the original and modified prediction 
    accuracy as a function of the allowed addition.

    Parameters:
    - buf: A Pandas DataFrame containing the data for plotting. It must contain the following columns:
      - 'addition': The amount of allowed addition.
      - 'input_psnr': The PSNR of the input image.
      - 'original_pred': The original prediction accuracy.
      - 'modified_pred': The modified prediction accuracy.

    Returns:
    - None: This function displays two bar plots and does not return any values.
    """
    buf.reset_index(inplace=True)
    plt.figure(figsize=(4, 4))
    sns.barplot(data=buf, x="addition", y="input_psnr", color='blue')
    plt.title('PSNR of input and modified input depending on allowed addition')
    plt.ylabel("psnr")
    plt.show()

    plt.figure(figsize=(4, 4))
    sns.barplot(data=buf, x="addition", y="original_pred", color='red', alpha=0.3, label="original_pred")
    sns.barplot(data=buf, x="addition", y="modified_pred", color='blue', alpha=0.3, label="modified_pred")
    plt.title('Average prediction accuracy depending on the size of the addition')
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
