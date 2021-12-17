import torch
import torch.nn.functional as F
from einops import repeat

sobel_kernel = torch.tensor([
        [ 1,  2,  1], 
        [ 0,  0,  0], 
        [-1, -2, -1]
    ], dtype=torch.float)

def sobel_filter(tensor):
    # Tensor of shape BCWH
    filtered_x = F.conv2d(
        tensor,
        sobel_kernel.view(1, 1, 3, 3),
        stride=1,
        padding=1
    )
    filtered_y = F.conv2d(
        tensor,
        sobel_kernel.T.view(1, 1, 3, 3),
        stride=1,
        padding=1
    )

    return (filtered_x**2 + filtered_y**2)**.5

sobel_kernel = torch.tensor([
        [ 1,  2,  1], 
        [ 0,  0,  0], 
        [-1, -2, -1]
    ], dtype=torch.float)

def get_sobel_filter(channels=1, device="cuda"):
    filter_x = repeat(sobel_kernel, "H W -> B C H W", B=1, C=channels).to(device)
    filter_y = repeat(sobel_kernel, "H W -> B C W H", B=1, C=channels).to(device)

    def filter(tensor):
        # Tensor of shape BCWH
        filtered_x = F.conv2d(tensor, filter_x, stride=1, padding=1)
        filtered_y = F.conv2d(tensor, filter_y, stride=1, padding=1)

        return (filtered_x**2 + filtered_y**2 + 1e-8)**.5

    return filter




def make_gaussian_kernel(sigma, kernel_size):
    ts = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()
    return kernel

def fast_gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(sigma * 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = make_gaussian_kernel(sigma, kernel_size)

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="constant", value=0)

    # Separable 2d conv
    kernel = kernel.view(1, 1, kernel_size, 1)
    img = F.conv1d(img, kernel)
    kernel = kernel.view(1, 1, 1, kernel_size)
    img = F.conv1d(img, kernel)

    return img

def get_static_fast_gauss_blur(channels=1, sigma=2, device="cuda"):
    kernel_size = int(sigma * 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = make_gaussian_kernel(sigma, kernel_size).to(device)
    color_kernel = kernel.view(1, kernel_size).expand(channels, kernel_size)
    kernel_1st_conv = color_kernel.view(1, channels, kernel_size, 1)
    kernel_2nd_conv = color_kernel.view(1, channels, 1, kernel_size)

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    def static_fast_gaussian_blur(img: torch.Tensor) -> torch.Tensor:
        padded = F.pad(img, padding, mode="constant", value=0)

        # Separable 2d conv
        blur_x = F.conv1d(padded, kernel_1st_conv)
        blur_y = F.conv1d(blur_x, kernel_2nd_conv)

        return blur_y

    return static_fast_gaussian_blur