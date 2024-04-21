import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
from torch.utils.data import DataLoader

from data.get_dataloader import get_inf_dataloader, inf_transform


def predicted_calc(outputs: torch.Tensor) -> torch.Tensor:
    """
    Calculates predicted values from model outputs. Handles binary classification and single-output 
    cases, applying appropriate activation functions.

    Args:
        outputs (torch.Tensor): The output tensor from a model, shape can vary depending on 
                                the task (binary classification, etc.).

    Returns:
        torch.Tensor: The predicted values, transformed with appropriate activation functions.
    """
    if outputs.shape[-1] == 2:
        return F.softmax(outputs, dim=1)[:, 1]
    elif outputs.shape[-1] == 1:
        return outputs.reshape(-1).data.sigmoid()
    else:
        raise NotImplementedError("This function is not implemented yet")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def update_predicts(
    model: nn.Module, 
    df: pd.DataFrame, 
    predict: np.ndarray
) -> None:
    """
    Updates predictions for images with specified dimensions (1024x1024).

    Args:
        model (nn.Module): The PyTorch model used for inference.
        df (pd.DataFrame): The dataframe containing image data.
        predict (np.ndarray): Array where predictions will be stored.
    """
    height = []
    width = []
    for file_name in df.id:
        image = Image.open(f"data/generated-or-not/images/{file_name}").convert('RGB')
        height.append(image.size[0])
        width.append(image.size[1])
    ind = np.where((np.array(height) == 1024) & (np.array(width) == 1024))[0]
    final_predicts = []
    for image in tqdm(df.iloc[ind].id):
        img = Image.open(f"data/generated-or-not/images/{image}").convert("RGB")
        h, w = img.size[0], img.size[1]
        patches = []
        new_h = h // 2
        new_w = w // 2
        
        for i in range(20):
             top = np.random.randint(0, h - new_h)
             left = np.random.randint(0, w - new_w)
             patch = img.crop((left, top, left + new_w, top + new_h))
             patches.append(inf_transform(patch).unsqueeze(0))

        concatenated_patches = torch.cat(patches, dim=0)
        concatenated_patches = concatenated_patches.to(device)
        outputs = model(concatenated_patches).reshape(-1)
        predicts = outputs.data.sigmoid()
        final_predicts.append(float(torch.max(predicts)))
    
    predict[ind] = final_predicts


def get_format_predicts(
    model: nn.Module,
    inf_dataloader: DataLoader,
    df: pd.DataFrame,
    format: str = "jpeg"
) -> np.ndarray:
    """
    Get predictions for the given DataLoader and update if required based on the specified format.
    
    Args:
        model (nn.Module): The PyTorch model used for inference.
        inf_dataloader (DataLoader): The DataLoader containing the input data.
        df (pd.DataFrame): The DataFrame containing additional data information.
        format (str): The format of the data, defaults to "jpeg".
    
    Returns:
        np.ndarray: Array of predictions.
    """
    model.to(device)
    model.eval()
    counter = 0
    with torch.no_grad():
        for inputs, idxs in tqdm(inf_dataloader):
            inputs = inputs.to(device)

            predicts = predicted_calc(model(inputs))
            if counter != 0:
                predict = np.concatenate((predict, predicts.data.cpu()))
            else:
                predict = np.array(predicts.cpu())
            counter += 1

    if format == "jpeg":
        update_predicts(model, df, predict)

    return predict


def get_inference(
    models: List[nn.Module], 
    csv_output: str, 
    path: str = "data/generated-or-not"
) -> None:
    """
    Get inference results from one or more models, and save the results to a CSV file.
    
    Args:
        models (List[nn.Module]): A list of models to use for inference.
        csv_output (str): The path to the CSV file where results will be saved.
        path (str): The path to the dataset, defaults to "data/generated-or-not".
    """
    inf_dataloader, df = get_inf_dataloader(path=path)
    if len(models) == 1:
        predict = get_format_predicts(models[0], inf_dataloader, df)
    elif len(models) == 2:
        predict = np.zeros_like(df.id)
        model_jpeg = models[0]
        inf_dataloader_jpeg, df_jpeg = get_inf_dataloader(included_formats=["webp", "jpeg", "jpg"], path=path)
        predict[np.where(df.format != "png")[0]] = get_format_predicts(model_jpeg, inf_dataloader_jpeg, df_jpeg)

        model_png = models[1]
        inf_dataloader_png, df_png = get_inf_dataloader(included_formats=["png"], path=path)
        predict[np.where(df.format == "png")[0]] = get_format_predicts(model_png, inf_dataloader_png, df_png, format="png")
    else:
        raise NotImplementedError("In the current implementation only 2 models were trained good enough:)")
    

    df_inf = pd.read_csv('data/generated-or-not/sample_submission.csv')
    if path == "data/generated-or-not-faces":
        df["target"], df_inf["target"] = predict, -1
        agg = pd.DataFrame(df.groupby(["orig_id"])["target"].max())
        df_inf.update({"id": list(agg.index), "target": list(agg.target)})
    else:   
        df_inf["target"] = predict
    df_inf.to_csv(csv_output, index=False)


def get_single_image_inference(model: nn.Module, single_image_path: str) -> float:
    """
    Get inference prediction for a single image using a specified model.
    
    Args:
        model (nn.Module): The model to use for inference.
        single_image_path (str): The path to the single image to be inferred.
    
    Returns:
        float: The prediction result for the single image.
    """
    inf_dataloader, _ = get_inf_dataloader(single_image_path=single_image_path)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for inputs, idxs in tqdm(inf_dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs).reshape(-1)
            predicts = outputs.data.sigmoid()
    
    predicts = np.array([predicts.item()])
    format = single_image_path.split(".")[-1]
    if format == "jpeg":
        data = {'id': [single_image_path]}
        df = pd.DataFrame(data)
        update_predicts(model, df, predicts)

    return predicts[0]


def get_single_image_ensemble(single_image_path: str) -> float:
    """
    Perform an ensemble inference for a single image using multiple models.
    
    Args:
        single_image_path (str): The path to the single image to be inferred.
    
    Returns:
        float: The ensemble prediction result for the single image.
    """
    format = single_image_path.split(".")[-1]
    model_type = "png" if format == "png" else "jpeg"
    predicts = []
    for i in range(1, 5):
        model = models.resnet50()
        model.fc = nn.Linear(2048, 1)
        model.load_state_dict(torch.load(f"../{model_type}_{i}.pth")["model_state_dict"])
        predicts.append(get_single_image_inference(model, single_image_path))

    pr_1, pr_2, pr_3, pr_4 = predicts[0], predicts[1], predicts[2], predicts[3]
    sorted = np.sort([pr_1, pr_2, pr_3, pr_4])

    if sorted[2] - sorted[0] < sorted[3] -  sorted[1]:
        ensemble_predict = np.mean(sorted[:2])
    else:
        ensemble_predict = np.mean(sorted[2:])

    return ensemble_predict
