
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.recover import get_recovered

def visualise_maniqa_scores():
    subprocess.run(["git", "clone", "https://github.com/IIGROUP/MANIQA.git utils/"])
    #also here need to upload weights to dir utils/MANIQA/weights of this repo manually
    subprocess.run(["cp", "utils/predict.py", "utils/MANIQA/"])
    subprocess.run(["python", "utils/MANIQA/predict.py"])

    maniqa = pd.read_csv("maniqa.csv")
    df = get_recovered("train.csv")
    merged_df = pd.merge(df,maniqa, on='id', how='left')

    plt.hist(merged_df.maniqa_score[merged_df.format == "jpeg"], alpha=0.5, label='jpeg, gt 1')
    plt.hist(merged_df.maniqa_score[merged_df.format == "jpg"], alpha=0.5, label='jpg, gt 0')
    plt.hist(merged_df.maniqa_score[merged_df.format == "png"], alpha=0.5, label='png, gt 1')

    plt.xlabel('Maniqa Score')
    plt.ylabel('Frequency')
    plt.title('Maniqa score depending on format')
    plt.legend(loc='upper right')

    plt.show()