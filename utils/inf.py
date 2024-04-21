import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
from data.get_dataloader import get_inf_dataloder, inf_transform


def predicted_calc(outputs):
    if outputs.shape[-1] == 2:
        return F.softmax(outputs, dim=1)[:, 1]
    elif outputs.shape[-1] == 1:
        return outputs.reshape(-1).data.sigmoid()
    else:
        raise NotImplementedError("This function is not implemented yet")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def update_predicts(model, df, predict):
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


def get_format_predicts(model, inf_dataloader, df, format="jpeg"):
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


def get_inference(models, csv_output, path="data/generated-or-not"):
    inf_dataloader, df = get_inf_dataloder(path=path)
    if len(models) == 1:
        predict = get_format_predicts(models[0], inf_dataloader, df)
    elif len(models) == 2:
        predict = np.zeros_like(df.id)
        model_jpeg = models[0]
        inf_dataloader_jpeg, df_jpeg = get_inf_dataloder(included_formats=["webp", "jpeg", "jpg"], path=path)
        predict[np.where(df.format != "png")[0]] = get_format_predicts(model_jpeg, inf_dataloader_jpeg, df_jpeg)

        model_png = models[1]
        inf_dataloader_png, df_png = get_inf_dataloder(included_formats=["png"], path=path)
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


def get_single_image_inference(model, single_image_path):
    inf_dataloader, _ = get_inf_dataloder(single_image_path=single_image_path)
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

def get_single_image_ensemble(single_image_path):
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