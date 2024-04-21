import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models

from utils.inf import get_single_image_inference, get_single_image_ensemble


parser = argparse.ArgumentParser(description="Conduct inference on a single image with optional ensemble.")
parser.add_argument(
    "img_path", 
    type=str, 
    help="Path to the image to conduct inference on."  
)
parser.add_argument(
    "--use_ensemble", 
    action="store_true",
    help="Use ensemble method for inference. Default is False."
)
args = parser.parse_args()


def main():
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image file '{args.img_path}' does not exist.")
    
    if args.use_ensemble:
        predict = get_single_image_ensemble(args.img_path)
    else:
        format = args.img_path.split(".")[-1]
        model_type = "png" if format == "png" else "jpeg"
        model = models.resnet50()
        model.fc = nn.Linear(2048, 1)
        model.load_state_dict(torch.load(f"../{model_type}_1.pth")["model_state_dict"])
        predict = get_single_image_inference(model, args.img_path)

    print(f"Prob of being fake is {predict:.3f}")

    
if __name__ == "__main__":
    main()
