import torch

def calculate_texture_diversity(patch):
    M = patch.size(-1)
    ldiv = (
        torch.sum(torch.abs(patch[:, :, :-1] - patch[:, :, 1:])) +  # Horizontal differences
        torch.sum(torch.abs(patch[:, :-1, :] - patch[:, 1:, :])) +  # Vertical differences
        torch.sum(torch.abs(patch[:, :-1, :-1] - patch[:, 1:, 1:])) +  # Diagonal differences
        torch.sum(torch.abs(patch[:, 1:, :-1] - patch[:, :-1, 1:]))  # Counter-diagonal differences
    )
    return ldiv

def calculate_and_save_texture_diversity(batch):
    batch_size = batch.size(0)
    texture_diversities = torch.zeros(batch_size)
    for i in range(batch_size):
        patch = batch[i]  
        texture_diversities[i] = calculate_texture_diversity(patch)
    return texture_diversities

def generate_patches(input):
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