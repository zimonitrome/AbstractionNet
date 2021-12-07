import torch
from einops import rearrange


# Normalization / Standardization functions

def normalize_functional(tensor: torch.Tensor, mean: list, std: list):
    """
    Standardizes tensor in the channel dimension (dim -3) using mean and std.
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor-mean)/std

def unnormalize_functional(tensor: torch.Tensor, mean: list, std: list):
    """
    Un-standardizes tensor in the channel dimension (dim -3) using mean and std.
    Also clips the tensor to be in the range [0, 1].
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return ((tensor*std)+mean).clamp(0, 1)

def unnormalize_to(x, x_min, x_max):
    """
    Linear normalization of x to [x_min, x_max].
    In other words maps x.min() -> x_min and x.max() -> x_max.
    """
    return x * (x_max - x_min) + x_min


# Image convertion functions

def rgba_to_rgb(rgba: torch.Tensor):
    """
    Converts tensor or shape [... 4 H W] into [... 3 H W].
    Multiplies first 3 channels with the last channel.
    """
    return rgba[..., :-1, :, :] * rgba[..., -1:, :, :]

def rgb_to_rgba(rgb: torch.Tensor, fill: float = 1.0):
    """
    Converts tensor or shape [... 3 H W] into [... 4 H W].
    Alpha layer will be filled with 1 by default, but can also be specified.
    """
    alpha_channel = torch.full_like(rgb[..., 0, :, :], fill_value=fill)
    return torch.concat([rgb, alpha_channel], dim=-3)


# Alpha compositing/decompositing functions

def get_visible_mask(shapes):
    shape_iterator = rearrange(shapes, "... N C H W -> N ... C H W").flip(0)
    accumulated_alpha = torch.zeros_like(shape_iterator[0,..., 0, :, :]) # empty like first image, single channel
    shape_maks = torch.zeros_like(shape_iterator[..., 0, :, :]) # empty image for each shape layer
    for i, shape in enumerate(shape_iterator):
        # a over b alpha compositioning
        # alpha_0 = (1 - alpha_a) * alpha_b + alpha_a
        # get b
        # alpha_b = (alpha_0 - alpha_a) / (1 - alpha_a)
        shape_alpha = shape[..., -1, :, :]
        alpha_visible = shape_alpha - accumulated_alpha * shape_alpha
        shape_maks[i] = alpha_visible
        accumulated_alpha = (1 - shape_alpha) * accumulated_alpha + shape_alpha
    
    return rearrange(shape_maks.flip(0), "N ... H W -> ... N H W").unsqueeze(-3)