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

class CustomDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):           
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
        image = Image.open("img_name").convert('RGB')        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(idx)


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


def get_train_test_dataloader(additional_train_data, included_formats=["png", "jpeg", "jpg"], train_transforms=["crop"], batch_size=48, test_rate=0.2, random_state=42):
    df = get_recovered(csv_name="train.csv", formats=included_formats)
    df_train, df_test = train_test_split(df, test_size=test_rate, random_state=random_state)
    df_train.id = "data/generated-or-not/images/" + df_train.id
    
    for df_additional in additional_train_data:
        df_train = concatenated_df = pd.concat([df_train, df_additional], ignore_index=True)
    df_train.index = np.arange(len(df_train))
    transform_list = []
    for train_transform in train_transforms:
        transform_list.append(transform_dict[train_transform])
    transform_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    train_transform = transforms.Compose(transform_list)

    train_subset = CustomDataset(df=df_train, root_dir='.', transform=train_transform)
    train_dataloader = DataLoader(train_subset, batch_size=48, shuffle=True)

    test_subset = CustomDataset(df=df_test, root_dir='data/generated-or-not/images', transform=test_transform)
    test_dataloader = DataLoader(test_subset, batch_size=48, shuffle=False)

    return train_dataloader, test_dataloader
    


def get_inf_dataloder():
    pass