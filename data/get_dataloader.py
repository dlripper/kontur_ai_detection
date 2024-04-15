import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from data.recover import get_recovered


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
    def __init__(self, df, root_dir, transform=test_transform):           
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.id[idx])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(int(self.annotations.target[idx]))
        
        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(idx)


class InfDataset(CustomDataset):
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.id[idx])
        image = Image.open(img_name).convert('RGB')        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(idx)


def get_train_test_dataloader(additional_train_data, included_formats=["png", "jpeg", "jpg"], train_transforms=["crop"], batch_size=48, test_rate=0.2, random_state=42, path="data/generated-or-not"):
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

    train_subset = CustomDataset(df=df_train, root_dir='.', transform=train_transform)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    test_subset = CustomDataset(df=df_test, root_dir=path + "/images", transform=test_transform)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
    


def get_inf_dataloder(included_formats=["png", "jpeg", "jpg", "webp"], batch_size=48, path="data/generated-or-not", single_image_path=None):
    if not single_image_path:
        df = get_recovered(csv_name="test.csv", formats=included_formats, path=path)
        path = path + "/images"
    else:
        df = pd.DataFrame({"id": [single_image_path]})
        path = "."
    inf_subset = InfDataset(df=df, root_dir=path, transform=inf_transform)
    inf_dataloader = DataLoader(inf_subset, batch_size=batch_size, shuffle=False)

    return inf_dataloader, df