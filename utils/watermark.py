import cv2
import numpy as np
import pandas as pd
from imwatermark import WatermarkEncoder, WatermarkDecoder


def add_watermark():
    WatermarkEncoder.loadModel()
    os.mkdir("data/watermark_generated")
    os.mkdir("data/watermark_generated/images")
    subprocess.run(["cp", "data/generated-or-not/test.csv", "data/watermark_generated/"])
    
    files = np.array(get_recovered("train.csv", formats=["jpeg", "jpg"]).id)
    WatermarkEncoder.loadModel()
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', 'aign'.encode('utf-8'))
    for pos, file_name in enumerate(files):        
        bgr = np.array(Image.open(f"data/generated-or-not/images/{file_name}").convert('RGB'))
        bgr_encoded = encoder.encode(bgr, 'rivaGan')
        Image.fromarray(bgr_encoded).save(f"data/watermark_generated/images/{file_name}", quality=80)

def get_added_watermarks():
    add_watermark()
    model = models.resnet50()
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("utils/jpeg_1.pth")["model_state_dict"])
    get_inference([model], "utils/watermark_predict.csv", path="data/watermark_generated")
    df_inf = pd.read_csv("utils/watermark_predict.csv")
    ground_truth = pd.read_csv("final_predict.csv")

    #visualization
    print("Log Loss on Watermark-added data is", log_loss((ground_truth.target >= 0.5).astype(np.int32), df_inf.target, labels=[0,1]))
    print("You can see the following example below:")
    visualization.show_difference("data/generated-or-not/images/HXKXONqX6P.png", "data/watermark_generated/HXKXONqX6P.png")