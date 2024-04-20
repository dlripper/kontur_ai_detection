import subprocess
import matplotlib.pyplot as plt
import pandas as pd

from data.recover import get_recovered

def visualise_maniqa_scores(df):
    subprocess.run(["git", "clone", "https://github.com/IIGROUP/MANIQA.git utils/"])
    #also here need to upload weights to dir utils/MANIQA/weights of this repo manually
    subprocess.run(["cp", "utils/predict.py", "utils/MANIQA/"]) 
    subprocess.run(["python", "utils/MANIQA/predict.py"])
    subprocess.run(["bash", "download_genimage_dataset.sh"])

    maniqa = pd.read_csv("data/maniqa.csv")
    maniqa_genimage = pd.read_csv("data/maniqa_imagenet.csv")
    # df = get_recovered("train.csv")
    merged_df = pd.merge(df,maniqa, on='id', how='left')

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    axs[0].hist(merged_df.maniqa_score[merged_df.format == "jpeg"], alpha=0.5, label='jpeg, gt 1', color="blue")
    axs[0].hist(merged_df.maniqa_score[merged_df.format == "jpg"], alpha=0.5, label='jpg, gt 0', color="orange")
    axs[0].set_xlabel('Maniqa Score')  
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Maniqa score depending on format')
    axs[0].legend(loc='upper right')

    axs[1].hist(merged_df.maniqa_score[merged_df.format == "png"], alpha=0.5, label='png, gt 1', color="green")
    axs[1].hist(maniqa_genimage.maniqa_score[:700], alpha=0.5, label='imagenet(png), gt 0', color="red")
    axs[1].set_xlabel('Maniqa Score')  
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Maniqa score depending on format')
    axs[1].legend(loc='upper right')

    plt.show()
