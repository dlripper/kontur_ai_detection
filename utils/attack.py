import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).to(device)


def denormalize_tensor(image_input_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor representing an image.

    Args:
        image_input_tensor (Tensor): Normalized image tensor.
        mean (Tensor): Mean value used for normalization.
        std (Tensor): Standard deviation value used for normalization.

    Returns:
        Tensor: Denormalized image tensor.
    """
    denormalized_tensor = image_input_tensor.clone()  
    denormalized_tensor *= std
    denormalized_tensor += mean    
    return denormalized_tensor


def psnr_loss(predicted_image: torch.Tensor, target_image: torch.Tensor, max_pixel_value: float = 1.0) -> torch.Tensor:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        predicted_image (Tensor): Predicted image.
        target_image (Tensor): Target image (ground truth).
        max_pixel_value (float, optional): Maximum pixel value in the image (default is 1.0).

    Returns:
        Tensor: PSNR value.
    """
    mse = F.mse_loss(predicted_image, target_image)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr


def single_attack(model: nn.Module, dataloader: DataLoader, attack_type: str) -> None:
    """
    Apply a single attack using Iterative Fast Gradient Sign Method (IFGSM) on the model and save the results.

    The function performs the following steps:
    1. Sets up a DataFrame for storing attack results.
    2. Creates necessary directories for saving the modified images.
    3. Applies IFGSM attacks with varying levels of perturbation (defined by `addition`).
    4. Calculates predictions on the original and modified images.
    5. Saves modified images and attack results in the specified directory.

    Args:
        model (nn.Module): The model to be attacked.
        dataloader (DataLoader): DataLoader containing the dataset to be used for the attack.
        attack_type (str): The type of attack being applied (in this case, "ifgsm").

    Returns:
        None

    Notes:
    - The attack applies perturbations to input images using gradients from the model's predictions.
    - Perturbations are applied iteratively based on the `addition` parameter.
    - Results are saved to CSV and modified images are stored for later analysis.
    """
    model.to(device)
    columns = ['image_modified', 'input_psnr', 'original_pred', 'modified_pred', 'addition']
    df = pd.DataFrame(columns=columns)
    
    #creating  dir for outputs
    dir_path = f"data/ifgsm"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for addition in [1, 2, 4, 8, 16, 32]:
        T = 10 #4 * 20 * addition
        alpha = addition / 255 / T
        print(alpha)
        mod = []
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            orig_image_tensor = inputs.clone().detach()
            inputs.requires_grad_(True)   

            for i in range(T + 1):
                cur_model = copy.deepcopy(model)
                cur_model.train()
                outputs = cur_model(inputs).reshape(-1)
                loss = torch.mean(outputs.sigmoid())
                loss.backward()

                gradients = inputs.grad
                updated_inputs = (inputs - (gradients).sign() * alpha * 4)
                # grad_cum = 0.5 * (grad_cum + gradients.data) <-->momentum analogue
                if i != T:
                    inputs = updated_inputs.clone().detach().requires_grad_(True)

        
            model.eval()
            with torch.no_grad():
                original_pred = model(orig_image_tensor).reshape(-1).data.sigmoid()
                modified_pred = model(inputs).reshape(-1).data.sigmoid()
            print("max diff is", torch.max(orig_image_tensor - inputs), torch.max(inputs - orig_image_tensor))
            print(torch.mean(original_pred), torch.mean(modified_pred))
            #saving modified input
            inputs = denormalize_tensor(inputs.detach())
            orig_image_tensor = denormalize_tensor(orig_image_tensor)
            print("max diff is", torch.max(orig_image_tensor - inputs), torch.max(inputs - orig_image_tensor))
            for pos, (cur_input, cur_orig, cur_or_pr, cur_mod_pr) in enumerate(zip(inputs, orig_image_tensor, original_pred, modified_pred)):
                if not os.path.exists(f"{dir_path}/ifgsm_modified_input_{addition}_1"):
                    os.makedirs(f"{dir_path}/ifgsm_modified_input_{addition}_1")

                
                ou = (cur_input.clamp_(0, 1) * 255).permute(1, 2, 0).round().cpu().numpy().astype(np.uint8)
                image_pil = Image.fromarray(ou)
                image_modified_path = f"{dir_path}/ifgsm_modified_input_{addition}_1/{pos}.png"
                image_pil.save(image_modified_path)

                input_psnr = psnr_loss(cur_input, cur_orig).item()
                print(image_modified_path, input_psnr, cur_or_pr, cur_mod_pr, addition)
        
                df.loc[len(df)] = {'image_modified_path': image_modified_path, 'input_psnr': input_psnr, 
                'original_pred': cur_or_pr.item(), 'modified_pred': cur_mod_pr.item(), 'addition': addition}
                

    df.to_csv(f"{dir_path}/report.csv")

  
def universal_attack(model: torch.nn.Module, dataloader: DataLoader, attack_type: str) -> None:
    """
    Apply a Universal Adversarial Perturbations (UAP) attack on the model and save the results.

    The function performs the following steps:
    1. Sets up a DataFrame for storing attack results.
    2. Creates necessary directories for saving the modified images.
    3. Applies UAP attacks with varying levels of perturbation (`addition`).
    4. Trains a universal perturbation tensor to apply to the dataset.
    5. Saves modified images and attack results in the specified directory.

    Args:
        model (torch.rnn.Module): The model to be attacked.
        dataloader (DataLoader): DataLoader containing the dataset to be used for the attack.
        attack_type (str): The type of attack being applied (in this case, "opt_uap").

    Returns:
        None

    Notes:
    - The UAP attack involves training a universal perturbation tensor.
    - Perturbations are applied uniformly across all images based on the `addition` parameter.
    - Results are saved to CSV and modified images are stored for later analysis.
    """
    model.to(device)
    columns = ['image_modified', 'input_psnr', 'original_pred', 'modified_pred', 'addition']
    df = pd.DataFrame(columns=columns)
    
    dir_path = f"data/opt_uap"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for addition in [1, 2, 4, 8, 16, 32]:
        alpha = 4 * addition / 255
        print(alpha)
        opt_uap = torch.zeros((3, 224, 224)).to(device)
        opt_uap.requires_grad = True

        for epoch in range(10):
            cur_model = copy.deepcopy(model)
            cur_model.train()
            for inputs, labels, _ in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
       
                outputs = cur_model(inputs + opt_uap).reshape(-1)

                # print(outputs.data.sigmoid().item())
                loss = torch.mean(outputs.sigmoid())
                with torch.no_grad():
                    for param in cur_model.parameters():
                        param.grad = None

                loss.backward()

                gradients = opt_uap.grad
                print(loss)
                updated_uap = (opt_uap - 0.2 * (gradients).sign() * alpha).clamp_(-alpha, alpha)
                # print(updated_uap.shape)
                opt_uap = updated_uap.clone().detach().requires_grad_(True)

        model.eval()
        for inputs, _, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            orig_image_tensor = inputs.clone()
            inputs = inputs + opt_uap.detach()

            with torch.no_grad():
                original_pred = model(orig_image_tensor).reshape(-1).data.sigmoid()
                modified_pred = model(inputs).reshape(-1).data.sigmoid()
            print(torch.mean(original_pred), torch.mean(modified_pred))
            #saving modified input
            inputs = denormalize_tensor(inputs.detach())
            orig_image_tensor = denormalize_tensor(orig_image_tensor)
            for pos, (cur_input, cur_orig, cur_or_pr, cur_mod_pr) in enumerate(zip(inputs, orig_image_tensor, original_pred, modified_pred)):
                if not os.path.exists(f"{dir_path}/opt_uap_modified_input_{addition}_1"):
                    os.makedirs(f"{dir_path}/opt_uap_modified_input_{addition}_1")

                
                ou = (cur_input.clamp_(0, 1) * 255).permute(1, 2, 0).round().cpu().numpy().astype(np.uint8)
                image_pil = Image.fromarray(ou)
                image_modified_path = f"{dir_path}/opt_uap_modified_input_{addition}_1/{pos}.png"
                image_pil.save(image_modified_path)

                input_psnr = psnr_loss(cur_input, cur_orig).item()
                print(image_modified_path, input_psnr, cur_or_pr, cur_mod_pr, addition)
        
                df.loc[len(df)] = {'image_modified_path': image_modified_path, 'input_psnr': input_psnr, 
                'original_pred': cur_or_pr.item(), 'modified_pred': cur_mod_pr.item(), 'addition': addition}

    df.to_csv(f"{dir_path}/report.csv")
            
         
def get_attacked(model: torch.nn.Module, dataloader: DataLoader, attack_type: str) -> None:
    """
    Apply adversarial attacks to a model using the specified attack type.

    Args:
        model (torch.nn.Module): The model to be attacked.
        dataloader (DataLoader): DataLoader containing the dataset to be used for the attack.
        attack_type (str): The type of adversarial attack to apply. Supported types:
            - "ifgsm": Iterative Fast Gradient Sign Method (FGSM) attack.
            - "opt_uap": Universal Adversarial Perturbations (UAP) attack.

    Returns:
        None

    Raises:
        NotImplementedError: If the specified attack_type is not implemented.
    """
    if attack_type == "ifgsm":
        return single_attack(model, dataloader, attack_type)
    elif attack_type == "opt_uap":
        return universal_attack(model, dataloader, attack_type)
    else:
        raise NotImplementedError("This function is not implemented yet")
