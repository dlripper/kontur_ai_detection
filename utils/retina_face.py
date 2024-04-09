import os
import subprocess
import shutil
import pandas as pd
import numpy as np


def get_retina_dataset():
    #step 1
    for face_file in face_files: 
        image_path = f"/home/aim_m_tre/BSRGAN/module_2/generated-or-not/face_images/{face_file}"  # Replace "your_image_path.jpg" with the actual path to your image
        image = cv2.imread(image_path)
        if image.shape[0] > 30 and image.shape[1] > 30:
            destination_path = f"/home/aim_m_tre/BSRGAN/module_2/generated-or-not/face_images_filtered/{face_file}"
            shutil.copy(image_path, destination_path)

    #step 2
    df = pd.read_csv('generated-or-not/train.csv')
    directory = './generated-or-not/images'  # Current directory
    files = os.listdir(directory)
    for pos, i in enumerate(np.array(df.id)):
        if len(i.split(".")) == 1 or i.split(".")[1] not in ["png", "jpeg", "jpg"]:
            for filename in files:
                if i in filename:
                    df.iloc[pos, 0] = filename
   
    #step 3
    df_face = df[:0].copy()

    face_files = os.listdir("/home/aim_m_tre/BSRGAN/module_2/generated-or-not/face_images_filtered")
    for face_file in face_files:
        face_file_name = face_file.split("_")[0]
        for pos, file_name in enumerate(df.id):
            if face_file_name in file_name:
                df_face.loc[len(df_face)] = {"id": face_file, "target": df.target[pos]}
                break

    return df_face