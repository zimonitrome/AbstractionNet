import torch
from einops import rearrange


# Normalization / Standardization functions

def normalize_functional(tensor: torch.Tensor, mean: list, std: list):
    """
    Standardizes tensor in the channel dimension (dim -3) using mean and std.
    [... C H W] -> [... C H W]
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor-mean)/std

def unnormalize_functional(tensor: torch.Tensor, mean: list, std: list):
    """
    Un-standardizes tensor in the channel dimension (dim -3) using mean and std.
    Also clips the tensor to be in the range [0, 1].
    [... C H W] -> [... C H W]
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
    Converts tensor from 3 channels into 4.
    Multiplies first 3 channels with the last channel.
    [... 4 H W] -> [... 3 H W]
    """
    return rgba[..., :-1, :, :] * rgba[..., -1:, :, :]

def rgb_to_rgba(rgb: torch.Tensor, fill: float = 1.0):
    """
    Converts tensor from 4 channels into 3.
    Alpha layer will be filled with 1 by default, but can also be specified.
    [... 3 H W] -> [... 4 H W]
    """
    alpha_channel = torch.full_like(rgb[..., 0, :, :], fill_value=fill)
    return torch.concat([rgb, alpha_channel], dim=-3)


# Alpha compositing/decompositing functions

def alpha_composite(base, added, eps=1e-8):
    """
    Composite two tensors, i.e., layers `added` on top of `base`,
    where the last channel is assumed to be an alpha channel.
    [... C H W], [... C H W] -> [... C H W]
    """
    # Separate color and alpha
    alpha_b =  base[..., -1:, :, :]
    alpha_a = added[..., -1:, :, :]

    color_b =  base[..., :-1, :, :]
    color_a = added[..., :-1, :, :]

    # https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    alpha_0 = (1 - alpha_a) * alpha_b + alpha_a
    color_0 = ((1-alpha_a) * alpha_b*color_b + alpha_a*color_a) / (alpha_0 + eps)

    # Re-combine new color and alpha
    return torch.concat([color_0, alpha_0], dim=-3)

def alpha_composite_multiple(images_tensor):
    """
    Composite tensor of N images into a single image.
    Assumes last channel is an alpha channel.
    [... N C H W] -> [... C H W]
    """

    image_iterator = rearrange(images_tensor, "... N C H W -> N ... C H W")

    # Get first image
    compositioned_image = image_iterator[0]

    # Add the rest of the images
    for image in image_iterator[1:]:
        # TODO: Possibly need to add .copy() to prevent assignment error in autograd.
        compositioned_image = alpha_composite(compositioned_image, image)

    return compositioned_image

def get_visible_mask(shapes):
    """
    Inputs a set of rendered images where C > 1 and the last channel is an alpha channel.
    Assuming that images were to be compositioned first to last (N=0, 1, 2...),
    returns a mask for each image that show what pixels of that image is visible in the final composition.
    [... N C H W] -> [... N H W]
    """
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