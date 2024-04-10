import os
import cv2
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.inf import get_inference


def get_retina_dataset() -> None:
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
    subprocess.run(["rm", "-rf", "data/generated-or-not-faces/images_initial"])

    #step 2
    df = pd.read_csv('data/generated-or-not/train.csv')
    files = os.listdir('data/generated-or-not/images')
    for pos, i in enumerate(np.array(df.id)):
        if len(i.split(".")) == 1 or i.split(".")[1] not in ["png", "jpeg", "jpg"]:
            for filename in files:
                if i in filename:
                    df.iloc[pos, 0] = filename
   
    #step 3
    df_face = pd.DataFrame(columns=["id", "target", "orig_id"])
    face_files = os.listdir("data/generated-or-not-faces/images")
    for face_file in face_files:
        face_file_name = face_file.split("_")[0]
        for pos, file_name in enumerate(df.id):
            if face_file_name in file_name:
                df_face.loc[len(df_face)] = {"id": face_file, "target": df.target[pos], "orig_id": matching[face_file]}
                break

    df_face.to_csv("data/generated-or-not-faces/train.csv", ignore_index=True)


def get_faces_predicts(model, face_model, csv_output="utils/predict.csv"):
    get_inference(model, "utils/main_model_predict.csv")
    get_inference(face_model, "utils/face_model_predict.csv", "data/generated-or-not-faces") ##<---> need to solve it somehow!

    main_model_predict = pd.read_csv("utils/main_model_predict.csv")
    face_model_predict = pd.read_csv("utils/face_model_predict.csv")

    face_model_predict.target[face_model_predict.target == -1] = main_model_predict.target[face_model_predict.target == -1]
    face_model_predict.to_csv(csv_output, index=False)
    visualise_faces_predicts()

    subprocess.run(["rm", "utils/main_model_predict.csv"])
    subprocess.run(["rm", "utils/face_model_predict.csv"])


def visualise_faces_predicts():
    face_model_predict = pd.read_csv("utils/face_model_predict.csv")
    buffered_predict = face_model_predict[face_model_predict.target != -1][:6]

    _, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (orig_id, target) in enumerate(zip(buffered_predict.orig_id, buffered_predict.target)):
        image_path = f"data/generated-or-not-faces/detected_images/{orig_id}"
        image = plt.imread(image_path)
        axes[idx].imshow(image)
        axes[idx].set_title(f"ID: {orig_id}\nTarget: {target}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
