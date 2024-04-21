import os
import cv2
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.inf import get_inference
from data.recover import get_recovered


def get_retina_train_dataset() -> None:
    """
    Prepare the training dataset for a retina-based model.

    This function performs three steps:
    1. Copy faces from the initial directory to the training dataset directory if their dimensions are valid.
    2. Load the original training data to extract relevant metadata.
    3. Create a new DataFrame to represent the face-based training dataset.
    """
    #step 1
    face_files = os.listdir("data/generated-or-not-faces/images_initial")
    matching = pd.read_csv("data/generated-or-not-faces/matching.csv")
    matching = dict(zip(matching.modified_image, matching.orig_image))
    for face_file in face_files: 
        image_path = f"data/generated-or-not-faces/images_initial/{face_file}"  # Replace "your_image_path.jpg" with the actual path to your image
        image = cv2.imread(image_path)
        if image.shape[0] > 30 and image.shape[1] > 30:
            if not os.path.exists("data/generated-or-not-faces/images"):
                subprocess.run(["mkdir", "data/generated-or-not-faces/images"])
            destination_path = f"data/generated-or-not-faces/images/{face_file}"
            shutil.copy(image_path, destination_path)

    #step 2
    df = get_recovered("train.csv")
   
    #step 3
    df_face = pd.DataFrame(columns=["id", "target", "orig_id"])
    face_files = os.listdir("data/generated-or-not-faces/images")
    for face_file in face_files:
        face_file_name = face_file.split("_")[0]
        for pos, file_name in enumerate(df.id):
            if face_file_name in file_name:
                df_face.loc[len(df_face)] = {"id": face_file, "target": df.target[pos], "orig_id": matching[face_file].split("/")[-1]}
                break

    df_face.to_csv("data/generated-or-not-faces/train.csv", index=False)


def get_retina_inf_dataset() -> None:
    """
    Prepare the inference dataset for a retina-based model.

    This function creates a DataFrame for the retina-based inference dataset by matching face files with original IDs.
    """
    matching = pd.read_csv("data/generated-or-not-faces/matching.csv")
    matching = dict(zip(matching.modified_image, matching.orig_image))
    df = get_recovered("test.csv")
    df_face = pd.DataFrame(columns=["id", "orig_id"])
    face_files = os.listdir("data/generated-or-not-faces/images")
    for face_file in face_files:
        face_file_name = face_file.split("_")[0]
        for pos, file_name in enumerate(df.id):
            if face_file_name in file_name:
                df_face.loc[len(df_face)] = {"id": face_file, "orig_id": matching[face_file].split("/")[-1]}
                break

    df_face.to_csv("data/generated-or-not-faces/test.csv", index=False)


def get_faces_predicts(
    model: torch.nn.Module, 
    face_model: torch.nn.Module, 
    csv_output: str = "utils/predict.csv"
) -> None:
    """
    Get predictions for the face-based dataset and combine them with the main model's predictions.

    This function performs the following steps:
    1. Obtain predictions for both the main dataset and the retina-based face dataset.
    2. Combine these predictions, ensuring valid predictions are used.
    3. Save the combined predictions to a CSV file and visualize them.
    """
    get_inference([model], "utils/main_model_predict.csv")
    get_inference([face_model], "utils/face_model_predict.csv", "data/generated-or-not-faces") ##<---> need to solve it somehow!

    main_model_predict = pd.read_csv("utils/main_model_predict.csv")
    face_model_predict = pd.read_csv("utils/face_model_predict.csv")

    face_model_predict.loc[face_model_predict.target == -1] = main_model_predict.loc[face_model_predict.target == -1]
    face_model_predict.to_csv(csv_output, index=False)
    visualise_faces_predicts()


def visualise_faces_predicts() -> None:
    """
    Visualize the predictions for the face-based dataset.

    This function plots the predicted images with their corresponding targets to show positive and negative predictions.
    """
    face_model_predict = pd.read_csv("utils/face_model_predict.csv")
    positive_predict = face_model_predict[face_model_predict.target > 0.5][:3]
    negative_predict = face_model_predict[(face_model_predict.target != -1) & (face_model_predict.target < 0.5)][:3]
    buffered_predict = pd.concat([positive_predict, negative_predict], ignore_index=True)

    _, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (id, target) in enumerate(zip(buffered_predict.id, buffered_predict.target)):
        image_path = f"data/generated-or-not-faces/detected_images/{id.split('.')[0] + '.png'}"
        image = plt.imread(image_path)
        axes[idx].imshow(image)
        axes[idx].set_title(f"ID: {id}\nTarget: {target:.4f}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
