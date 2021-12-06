import torch

def rgba_to_rgb(rgba):
    return rgba[..., :-1, :, :] * rgba[..., -1:, :, :]

def rgb_to_rgba(rgb):
    alpha_channel = torch.ones_like(rgb[..., 0, :, :])
    return torch.concat([rgb, alpha_channel], dim=-3)

def normalize_functional(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return (tensor-mean)/std

def unnormalize_functional(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return ((tensor*std)+mean).clamp(0, 1)
