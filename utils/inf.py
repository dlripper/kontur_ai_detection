import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.get_dataloader import get_inf_dataloder


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#для начала давайте это сделаем просто для одного репозитория, а потом уже будем думать!
def get_inference(model, csv_output, path="data/generated-or-not"):
    inf_dataloader, df = get_inf_dataloder(path=path)
    model.eval()
    counter = 0

    with torch.no_grad():
        for inputs, idxs in tqdm(inf_dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs).reshape(-1)
            predicts = outputs.data.sigmoid()
            if counter != 0:
                predict = np.concatenate((predict, predicts.data.cpu()))
            else:
                predict = np.array(predicts.cpu())
            counter += 1

    df_inf = pd.read_csv('generated-or-not/sample_submission.csv')
    if path == "data/generated-or-not-faces":
        df["target"], df_inf["target"] = predict, -1
        agg = pd.DataFrame(df.groupby(["orig_id"])["target"].max())
        df.update(agg)
    else:   
        df_inf["target"] = predict
    df_inf.to_csv(csv_output, index=False)


def get_single_image_inference(model, path):
    inf_dataloader, _ = get_inf_dataloder(path=path)
    model.eval()
    
    with torch.no_grad():
        for inputs, idxs in tqdm(inf_dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs).reshape(-1)
            predicts = outputs.data.sigmoid()
           
    return predicts.item()