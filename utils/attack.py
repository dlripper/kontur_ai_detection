import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).to(device)


def denormalize_tensor(image_input_tensor):
    denormalized_tensor = image_input_tensor.clone()  
    denormalized_tensor *= std
    denormalized_tensor += mean
    return denormalized_tensor


def psnr_loss(predicted, target, max_pixel=1.0):
    mse = F.mse_loss(predicted, target)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def single_attack(model, dataloader, attack_type):
    model.to(device)
    model.train()
    columns = ['image_modified', 'input_psnr', 'original_pred', 'modified_pred', 'addition']
    df = pd.DataFrame(columns=columns)
    
    #creating  dir for outputs
    dir_path = f"data/ifgsm"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for addition in [1, 2, 4, 8, 16, 32]:
        T = 4 * 20 * addition
        alpha = addition / 255 / T
        print(alpha)
        mod = []
        for pos, (inputs, labels, _) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad_(True)
            orig_image_tensor = inputs.clone().detach()

            for i in range(T + 1):
                outputs = model(inputs).reshape(-1)
                if i == 0:
                    original_pred = outputs.data.sigmoid().item()
                if i == T:
                    modified_pred = outputs.data.sigmoid().item()
                loss = outputs.sigmoid()
                loss.backward()

                gradients = inputs.grad
                updated_inputs = (inputs - (gradients).sign() * alpha * 4)
                # grad_cum = 0.5 * (grad_cum + gradients.data) <-->momentum analogue
                if i != T:
                    inputs = updated_inputs.clone().detach().requires_grad_(True)
        
            #saving modified input
            if not os.path.exists(f"{dir_path}/ifgsm_modified_input_{addition}_1"):
                os.makedirs(f"{dir_path}/ifgsm_modified_input_{addition}_1")

            inputs = denormalize_tensor(inputs.detach())
            ou = (inputs[0].clamp_(0, 1) * 255).permute(1, 2, 0).round().cpu().numpy().astype(np.uint8)
            image_pil = Image.fromarray(ou)
            image_modified_path = f"{dir_path}/ifgsm_modified_input_{addition}_1/{pos}.png"
            image_pil.save(image_modified_path)

            input_psnr = psnr_loss(inputs, denormalize_tensor(orig_image_tensor)).item()
            print(input_psnr, original_pred, modified_pred)
     
            df.loc[len(df)] = {'image_modified_path': image_modified_path, 'input_psnr': input_psnr, 
            'original_pred': original_pred, 'modified_pred': modified_pred, 'addition': addition}

    df.to_csv(f"{dir_path}/report.csv")

  
def universal_attack(model, dataloader, attack_type):
    model.to(device)
    model.train()
    columns = ['image_modified', 'input_psnr', 'original_pred', 'modified_pred', 'addition']
    df = pd.DataFrame(columns=columns)
    
    dir_path = f"data/opt_uap"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for addition in [1, 2, 4, 8, 16, 32]:
        alpha = 4 * addition / 255
        opt_uap = torch.zeros((3, 224, 224)).to(device)
        opt_uap.requires_grad = True

        for epoch in range(10):
            for inputs, labels, _ in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
       
                outputs = model(inputs + opt_uap).reshape(-1)

                # print(outputs.data.sigmoid().item())
                loss = torch.sum(outputs.sigmoid())
                loss.backward()

                gradients = opt_uap.grad

                updated_uap = (opt_uap - 0.2 * (grad_cum).sign() * alpha * 4).clamp_(-alpha, alpha)
                # print(updated_uap.shape)
                opt_uap = updated_uap.clone().detach().requires_grad_(True)

        for inputs, _, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            orig_image_tensor = inputs.clone()
            inputs = inputs + opt_uap.detach()

            with torch.no_grad():
                original_pred = model(orig_image_tensor).reshape(-1).data.sigmoid().item()
                modified_pred = model(inputs).reshape(-1).data.sigmoid().item()

            #saving modified input
            if not os.path.exists(f"{dir_path}/opt_uap_modified_input_{addition}_1"):
                os.makedirs(f"{dir_path}/opt_uap_modified_input_{addition}_1")

            inputs = denormalize_tensor(inputs.detach())
            ou = (inputs[0].clamp_(0, 1) * 255).permute(1, 2, 0).round().cpu().numpy().astype(np.uint8)
            image_pil = Image.fromarray(ou)
            image_modified_path = f"{dir_path}/opt_uap_modified_input_{addition}_1/{pos}.png"
            image_pil.save(image_modified_path)

            input_psnr = psnr_loss(inputs, denormalize_tensor(orig_image_tensor)).item()
            print(input_psnr, original_pred, modified_pred)
     
            df.loc[len(df)] = {'image_modified_path': image_modified_path, 'input_psnr': input_psnr, 
            'original_pred': original_pred, 'modified_pred': modified_pred, 'addition': addition}

    df.to_csv(f"{dir_path}/report.csv")
            
         
def get_attacked(model, dataloader, attack_type):
    if attack_type == "ifgsm":
        return single_attack(model, dataloader, attack_type)
    elif attack_type == "opt_uap":
        return universal_attack(model, dataloader, attack_type)
    else:
        raise NotImplementedError("This function is not implemented yet")