import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from typing import List, Tuple, Union, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from data.recover import get_recovered
from data.patchcraft_transform import generate_patches


transform_dict = {
    "crop": transforms.RandomResizedCrop(224),
    "flip": transforms.RandomHorizontalFlip(p=0.25), 
    "rotation": transforms.RandomRotation(degrees=10), 
    "colorjitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    "gaussianblur": transforms.GaussianBlur(kernel_size=3)
}


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
inf_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform: Optional[callable] = None) -> None:
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name = os.path.join(self.root_dir, self.annotations.id[idx])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(int(self.annotations.target[idx]))
        
        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(idx)


class InfDataset(CustomDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = os.path.join(self.root_dir, self.annotations.id[idx])
        image = Image.open(img_name).convert('RGB')        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(idx)


class PatchDataset(CustomDataset):
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        img_name = os.path.join(self.root_dir, self.annotations.id[idx])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(int(self.annotations.target[idx]))

        poor, rich = generate_patches(image.unsqueeze(0))

        return (poor.squeeze(0), rich.squeeze(0)), label, torch.tensor(idx)


def get_train_test_dataloader(
    additional_train_data: List[pd.DataFrame],
    included_formats: List[str] = ["png", "jpeg", "jpg"],
    train_transforms: List[str] = ["crop"],
    batch_size: int = 48,
    test_rate: float = 0.2,
    random_state: int = 42,
    path: str = "data/generated-or-not",
    patchcraft: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and testing datasets.

    This function generates DataLoaders for training and testing based on the given parameters.
    It allows for additional training data, multiple image formats, various training transformations,
    and optional patch-based datasets.

    Parameters:
    - additional_train_data: A list of additional DataFrames to be included in the training set.
    - included_formats: A list of image formats to include (default: ["png", "jpeg", "jpg"]).
    - train_transforms: A list of transformation names to apply during training (default: ["crop"]).
    - batch_size: The size of the batches for the DataLoaders (default: 48).
    - test_rate: The proportion of the dataset to use for testing (default: 0.2).
    - random_state: The random seed for data splitting (default: 42).
    - path: The root directory for the data (default: "data/generated-or-not").
    - patchcraft: Whether to use patch-based datasets (default: False).

    Returns:
    - A tuple containing the training DataLoader and the testing DataLoader.
    """
    df = get_recovered(csv_name="train.csv", formats=included_formats, path=path)
    if path == "data/generated-or-not-faces":
        df_orig = get_recovered(csv_name="train.csv", formats=included_formats)
        df_train_orig, df_test_orig = train_test_split(df_orig, test_size=test_rate, random_state=random_state)
        df_train = df[df["orig_id"].isin(df_train_orig.id)]
        df_test = df[df["orig_id"].isin(df_test_orig.id)]
    else:
        df_train, df_test = train_test_split(df, test_size=test_rate, random_state=random_state)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_train.id = path + "/images/"+ df_train.id
    
    for df_additional in additional_train_data:
        df_train = concatenated_df = pd.concat([df_train, df_additional], ignore_index=True)
    df_train.index = np.arange(len(df_train))
    transform_list = []
    for train_transform in train_transforms:
        transform_list.append(transform_dict[train_transform])
    transform_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    train_transform = transforms.Compose(transform_list)

    train_subset = CustomDataset(df=df_train, root_dir='.', transform=train_transform) if not patchcraft else PatchDataset(df=df_train, root_dir='.', transform=train_transform)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    test_subset = CustomDataset(df=df_test, root_dir=path + "/images", transform=test_transform) if not patchcraft else PatchDataset(df=df_test, root_dir=path + "/images", transform=test_transform)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
    

def get_inf_dataloader(
    included_formats: List[str] = ["png", "jpeg", "jpg", "webp"],
    batch_size: int = 48,
    path: str = "data/generated-or-not",
    single_image_path: Optional[str] = None
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Create a DataLoader for inference and return the associated DataFrame.

    This function creates a DataLoader for inference. It can operate on a dataset or a single image,
    depending on the provided parameters.

    Parameters:
    - included_formats: A list of image formats to include (default: ["png", "jpeg", "jpg", "webp"]).
    - batch_size: The size of the batch for the DataLoader (default: 48).
    - path: The base directory for the dataset (default: "data/generated-or-not").
    - single_image_path: Optional; if provided, creates a DataFrame with this single image path.

    Returns:
    - A tuple containing the DataLoader for inference and the corresponding DataFrame.
    """
    if not single_image_path:
        df = get_recovered(csv_name="test.csv", formats=included_formats, path=path)
        path = path + "/images"
    else:
        df = pd.DataFrame({"id": [single_image_path]})
        path = "."
    inf_subset = InfDataset(df=df, root_dir=path, transform=inf_transform)
    inf_dataloader = DataLoader(inf_subset, batch_size=batch_size, shuffle=False)

    return inf_dataloader, df
    