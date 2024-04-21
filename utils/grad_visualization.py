import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from PIL import Image
from typing import List, Tuple
from torchvision import models, transforms

from data.get_dataloader import test_transform

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_registered = False
        
    def register_hooks(self) -> None:
        """
        Registers forward and backward hooks to capture activations and gradients.

        This method sets up hooks on the target layer to store forward activations and backward gradients.
        """
        def forward_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.activations = output
            
        def backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
            self.gradients = grad_output[0]
        
        if not self.hook_registered:
            target_layer = self.target_layer
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
            self.hook_registered = True
    
    def backward(self, input_image: torch.Tensor, target_class: int) -> None:
        """
        Initiates backpropagation to obtain gradients for a specific target class.

        Args:
            input_image (torch.Tensor): The input image to the model.
            target_class (int): The target class for which gradients are calculated.
        """
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        grad_target_map[0][target_class] = 1
        model_output.backward(grad_target_map)

        return 
    
    def generate_heatmap(self) -> np.ndarray:
        """
        Generates a heatmap based on the gradients and activations.

        Returns:
            np.ndarray: A heatmap that highlights areas of importance for a given prediction.
        """
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(self.gradients.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()


def apply_gradcam(model: nn.Module, image_path: str, target_class: int = 0) -> np.ndarray:
    """
    Applies Grad-CAM to a given image and model, returning a heatmap overlaid on the image.

    Args:
        model (torch.nn.Module): The model on which Grad-CAM is applied.
        image_path (str): The path to the image.
        target_class (int): The target class for which Grad-CAM is computed (default is 0).

    Returns:
        np.ndarray: The heatmap overlaid on the original image.
    """
    target_layer = model.layer4[-1].conv3

    gradcam = GradCAM(model, target_layer)
    gradcam.register_hooks()
    
    image = Image.open(image_path).convert('RGB') 
    image_tensor = test_transform(image).unsqueeze(0).requires_grad_(True)
    
    gradcam.backward(image_tensor, target_class)
    
    heatmap = gradcam.generate_heatmap()
    
    image = np.array(image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_on_image = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    
    return heatmap_on_image


class Guided_backprop():
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.image_reconstruction = None 
        self.activation_maps = [] 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self) -> None:
        def first_layer_hook_fn(module: nn.Module, grad_in: Tuple[torch.Tensor], grad_out: Tuple[torch.Tensor]) -> None:
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            self.activation_maps.append(output)

        def backward_hook_fn(module: nn.Module, grad_in: Tuple[torch.Tensor], grad_out: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
            grad = self.activation_maps.pop() 
            grad[grad > 0] = 1 
            
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        first_relu = True
        for name, module in self.model.named_modules(): 
            if isinstance(module, nn.ReLU):
                if first_relu:
                    module.register_backward_hook(first_layer_hook_fn)
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
                first_relu = False

    def backward(self, input_image: torch.Tensor, target_class: int) -> np.ndarray:
        model_output = self.model(input_image)
        self.model.zero_grad()
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        grad_target_map[0][target_class] = 1
        model_output.backward(grad_target_map)   
        result = self.image_reconstruction.data[0].permute(1,2,0)
        return result.numpy()


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image to have mean 0.5 and standard deviation 0.1.

    Args:
        image (np.ndarray): The image to normalize.

    Returns:
        np.ndarray: The normalized image.
    """
    norm = (image - image.mean())/image.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm


def apply_guided_backprop(model: nn.Module, image_path: str, target_class: int = 0) -> np.ndarray:
    """
    Applies Guided Backpropagation to a given image and model, returning a visualization of the reconstructed activations.

    Args:
        model (torch.nn.Module): The model to apply Guided Backpropagation on.
        image_path (str): The path to the image.
        target_class (int): The target class for which Guided Backpropagation is applied (default is 0).

    Returns:
        np.ndarray: The reconstructed activations after Guided Backpropagation.
    """
    image = Image.open(image_path).convert('RGB') 
    image_tensor = test_transform(image).unsqueeze(0).requires_grad_(True)

    guided_bp = Guided_backprop(model)
    result = guided_bp.backward(image_tensor, 0)

    grad_required = image_tensor.grad[0].permute(1,2,0).cpu().numpy()
    grad_required[grad_required < 0] = 0
    result = normalize(grad_required)
    return result



def get_model_visualisation(model: nn.Module, image_path: str) -> None:
    """
    Displays model visualizations using Grad-CAM and Guided Backpropagation.

    Args:
        model (torch.nn.Module): The model for which visualizations are generated.
        image_path (str): The path to the image to visualize.

    Returns:
        None: This function plots the visualization but does not return anything.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    init_image = Image.open(image_path).convert("RGB")
    axs[0].imshow(init_image)
    axs[0].set_title('Initial Image')
    axs[0].axis('off')

    heatmap_on_image = apply_gradcam(model, image_path)
    axs[1].imshow(cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Grad-CAM')
    axs[1].axis('off')

    guided_backprop = apply_guided_backprop(model, image_path)
    axs[2].imshow(guided_backprop)
    axs[2].set_title('Guided Backprop')
    axs[2].axis('off')

    plt.show()
