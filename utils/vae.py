import os
import subprocess
import visualization
import matplotlib.pyplot as plt
import sklearn.metrics.log_loss as log_loss
import cv2
from diffusers import AutoencoderKL
from PIL import Image
from data.recover import get_recovered


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def create_vae_variation():
    vae = AutoencoderKL.from_pretrained("utils/my_model_dir", local_files_only=True)
    vae.to(device)

    os.mkdir("data/vae_generated")
    os.mkdir("data/vae_generated/images")
    subprocess.run(["cp", "data/generated-or-not/test.csv", "data/vae_generated/"])
    files = os.listdir("data/generated-or-not/images")
    for _, file_name in enumerate(files):
        pic = Image.open(f"data/generated-or-not/images/{file_name}").convert('RGB')
        pic_np = np.array(pic)
        torch_pic = torch.from_numpy(np.array(pic)).to(torch.float32).permute(2, 0, 1)

        with torch.no_grad():
            pred_pic = vae.forward(torch_pic.to(device).unsqueeze(0) / 255, sample_posterior=False, return_dict=False)

        numpy_pred_pic = (pred_pic[0][0].clamp_(0, 1) * 255).permute(1, 2, 0).round().cpu().numpy().astype(np.uint8)
        Image.fromarray(numpy_pred_pic).resize(pic.size).save(f"data/vae_generated/images/{file_name}")


def get_vae_variations():
    create_vae_variation()
    model = models.resnet50()
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("utils/png_1.pth")["model_state_dict"])
    get_inference([model], "utils/vae_predict.csv", path="data/vae_generated")
    df_inf = pd.read_csv("utils/vae_predict.csv")

    #visualization
    print("Log Loss on VAE generated data is", log_loss(np.ones_like(df_inf.target), df_inf.target, labels=[0,1]))
    print("You can see the following example below:")
    visualization.show_difference("data/generated-or-not/images/HXKXONqX6P.png", "data/vae_generated/HXKXONqX6P.png")
