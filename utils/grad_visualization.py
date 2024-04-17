import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from data.get_dataloader import test_transform

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_registered = False
        
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        if not self.hook_registered:
            target_layer = self.target_layer
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
            self.hook_registered = True
    
    def backward(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        grad_target_map[0][target_class] = 1
        model_output.backward(grad_target_map)

        return 
    
    def generate_heatmap(self):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(self.gradients.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()

def apply_gradcam(model, image_path, target_class=0):
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
    # Plot and display the heatmap on the image
    # plt.imshow(cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()


class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None 
        self.activation_maps = [] 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
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

    def backward(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        grad_target_map[0][target_class] = 1
        model_output.backward(grad_target_map)   
        result = self.image_reconstruction.data[0].permute(1,2,0)
        return result.numpy()


def normalize(image):
    norm = (image - image.mean())/image.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm


def apply_guided_backprop(model, image_path, target_class=0):
    image = Image.open(image_path).convert('RGB') 
    image_tensor = test_transform(image).unsqueeze(0).requires_grad_(True)

    guided_bp = Guided_backprop(model)
    result = guided_bp.backward(image_tensor, 0)

    grad_required = image_tensor.grad[0].permute(1,2,0).cpu().numpy()
    grad_required[grad_required < 0] = 0
    result = normalize(grad_required)
    return result
    # plt.imshow(result)
    # plt.show()



def get_model_visualisation(model, image_path):
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