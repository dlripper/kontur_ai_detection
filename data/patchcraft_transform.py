import torch
from typing import Tuple

def calculate_texture_diversity(patch: torch.Tensor) -> torch.Tensor:
    """
    Calculate the texture diversity of a given patch.

    This function computes the diversity in texture based on the sum of absolute differences
    in multiple directions (horizontal, vertical, diagonal, and counter-diagonal).

    Parameters:
    - patch: A 3D tensor representing the image patch (with dimensions [C, H, W], where
            C is the number of channels, H is the height, and W is the width).

    Returns:
    - The calculated texture diversity as a single tensor value.
    """
    M = patch.size(-1)
    ldiv = (
        torch.sum(torch.abs(patch[:, :, :-1] - patch[:, :, 1:])) +  # Horizontal differences
        torch.sum(torch.abs(patch[:, :-1, :] - patch[:, 1:, :])) +  # Vertical differences
        torch.sum(torch.abs(patch[:, :-1, :-1] - patch[:, 1:, 1:])) +  # Diagonal differences
        torch.sum(torch.abs(patch[:, 1:, :-1] - patch[:, :-1, 1:]))  # Counter-diagonal differences
    )
    return ldiv


def calculate_and_save_texture_diversity(batch: torch.Tensor) -> torch.Tensor:
    """
    Calculate texture diversity for each patch in a batch and return the results.

    This function iterates through a batch of patches, calculates the texture diversity for each,
    and returns a tensor containing the results.

    Parameters:
    - batch: A 4D tensor representing a batch of patches (with dimensions [B, C, H, W], where
            B is the batch size, C is the number of channels, H is the height, and W is the width).

    Returns:
    - A 1D tensor with the texture diversity values for each patch in the batch.
    """
    batch_size = batch.size(0)
    texture_diversities = torch.zeros(batch_size)
    for i in range(batch_size):
        patch = batch[i]  
        texture_diversities[i] = calculate_texture_diversity(patch)
    return texture_diversities


def generate_patches(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two sets of patches (poor and rich) from a given batch of images.

    This function creates a set of patches from the given input batch. It sorts the patches by texture diversity
    and separates them into two groups: "poor" and "rich".

    Parameters:
    - input: A 4D tensor representing a batch of images (with dimensions [B, C, H, W], where
            B is the batch size, C is the number of channels, H is the height, and W is the width).

    Returns:
    - A tuple containing two 4D tensors:
      - "poor": Patches with lower texture diversity.
      - "rich": Patches with higher texture diversity.
    """
    batch_size, num_channels, height, width = input.size()
    num_patches = 192
    patch_height, patch_width = 32, 32  

    poor = torch.zeros((batch_size, num_channels, 8 * patch_height, 8 * patch_width))
    rich = torch.zeros((batch_size, num_channels, 8 * patch_height, 8 * patch_width))
    

    for i in range(batch_size):  
        image = input[i]  
        buffer = torch.zeros((num_patches, num_channels, patch_height, patch_width))
        for j in range(num_patches):  
            top = torch.randint(0, height - patch_height + 1, (1,))
            left = torch.randint(0, width - patch_width + 1, (1,))
            patch = image[:, top:(top + patch_height), left:(left + patch_width)]
            buffer[j] = patch
        measures = calculate_and_save_texture_diversity(buffer)
        sorted_indices = torch.argsort(measures, descending=False)
        buffer = buffer[sorted_indices]
        for vert in range(8):
            for hor in range(8):
                poor[i, :, vert*32:(vert*32+32), hor*32:(hor*32+32)] = buffer[vert*8 + hor]
                rich[i, :, vert*32:(vert*32+32), hor*32:(hor*32+32)] = buffer[-vert*8 - hor - 1]

    return poor, rich
    