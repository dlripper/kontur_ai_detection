import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

def get_untyped(csv_name: str) -> Tuple[List[str], List[str]]:
    """
    Get untyped images and missed types from a CSV file.

    Args:
        csv_name (str): Name of the CSV file containing image data.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - 
            untyped_images: List of paths to untyped images.
            missed_types: List of missed types for untyped images.
    """
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


def get_recovered(csv_name: str, formats: Optional[list] = None, path: Optional[str] = "data/generated-or-not") -> pd.DataFrame:
    """
    Get recovered DataFrame from a CSV file.

    Args:
        csv_name (str): Name of the CSV file containing image data.
        formats (list, optional): List of image formats to filter (default is None).
        path (str, optional): Path to the directory containing CSV file and images (default is "data/generated-or-not").

    Returns:
        pd.DataFrame: Recovered DataFrame.
    """
    df = pd.read_csv(f'{path}/{csv_name}')
    files = os.listdir(f'{path}/images')

    for pos, i in enumerate(np.array(df.id)):
        if len(i.split(".")) == 1 or i.split(".")[1] not in ["png", "jpeg", "jpg"]:
            for filename in files:
                if i in filename:
                    df.iloc[pos, 0] = filename

    df["format"] = [el.split(".")[1] for el in df.id]
    if formats is not None:
        df = df[df.format.isin(formats)]

    return df