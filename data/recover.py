import os
import pandas as pd
import numpy as np

def get_untyped(csv_name):
    df = pd.read_csv(f'data/generated-or-not/{csv_name}')
    files = os.listdir('data/generated-or-not/images')

    untyped_images, missed_types = [], []
    for pos, i in enumerate(np.array(df.id)):
        if len(i.split(".")) == 1 or i.split(".")[1] not in ["png", "jpeg", "jpg"]:
            for filename in files:
                if i in filename:
                    untyped_images.append(f'data/generated-or-not/images/{filename}')
                    missed_types.append(filename.split(".")[1])

    return untyped_images, missed_types


def get_recovered(csv_name, formats=None):
    df = pd.read_csv(f'data/generated-or-not/{csv_name}')
    files = os.listdir('data/generated-or-not/images')

    for pos, i in enumerate(np.array(df.id)):
        if len(i.split(".")) == 1 or i.split(".")[1] not in ["png", "jpeg", "jpg"]:
            for filename in files:
                if i in filename:
                    df.iloc[pos, 0] = filename

    df["format"] = [el.split(".")[1] for el in df.id]
    if formats is not None:
        df = df[df.format.isin(formats)]

    return df