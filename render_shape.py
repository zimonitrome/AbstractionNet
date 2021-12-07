import torch
from torch import nn
import torchvision.models as models
from einops import rearrange, repeat
from differentiable_sorter_torchsort import soft_sort_by_column
from utils import unnormalize_to


class ShapeRenderer(nn.Module):
    def __init__(self, device="cpu", imsize=64, minimum_sharpness=1, return_mode="bitmap"):
        super().__init__()
        self.imsize = imsize

        ramp = torch.linspace(0, 1, self.imsize).to(device)
        self.x_ramp, self.y_ramp = torch.meshgrid(ramp, ramp, indexing=None)

        self.transparent = torch.ones([4, self.imsize, self.imsize]).to(device)
        self.min_sharpness = minimum_sharpness
        self.max_sharpness = 20

        self.return_mode = return_mode

    def draw_circle(self, args_tensor):
        """
        Tranforms a tensor of shape [..., 5] into image(s) of circles of shape [..., H, W].
        Expects last dimensions of args_tensor to be [sharpness, pos_x_, pos_y_, width, height].
        """
        eps = 1e-8

        # Move argument dimension furthest out and add 2 dimensions
        sharpness, pos_z, pos_x, pos_y, width, height, angle, squareness_ = repeat(args_tensor, "... (A H W) -> A ... H W", H=1, W=1)

        # Re-scale values

        pos_x_2 = pos_x
        pos_y_2 = pos_y
        sharpness_2 = unnormalize_to(sharpness, self.min_sharpness, self.max_sharpness)

        # Make either circle or square (no squircle) while in evaluation mode
        if not self.training:
            squareness = torch.round(squareness_)
        else:
            squareness = unnormalize_to(squareness_, 0.1, 0.9)

        # minimum radius = half of 1 pixel
        # => smallest possible shape = 1x1 pixel
        eps_1px = (1 / self.imsize) / 2
        radius_x = width + eps_1px
        radius_y = height + eps_1px
        angle_2 = torch.pi*angle

        # Get ramps ready
        # Get any trailing dimensions (i.e. [...])
        t_shape = list(args_tensor.shape)[:-1]
        # Reshape [H, W] -> [..., H, W]
        x_ramp = self.x_ramp.view(*len(t_shape)*[1], *self.x_ramp.shape).expand(*t_shape, *self.x_ramp.shape)
        y_ramp = self.y_ramp.view(*len(t_shape)*[1], *self.y_ramp.shape).expand(*t_shape, *self.y_ramp.shape)

        # Draw the shape
        translated_x = x_ramp - pos_x_2
        translated_y = y_ramp - pos_y_2

        rotated_x = translated_x*torch.cos(angle_2) - translated_y*torch.sin(angle_2)
        rotated_y = translated_x*torch.sin(angle_2) + translated_y*torch.cos(angle_2)

        scaled_x = rotated_x / radius_x
        scaled_y = rotated_y / radius_y

        x_s = scaled_x**2
        y_s = scaled_y**2
        p = -(x_s + y_s)
        q = x_s * y_s * squareness
        squircle = (-p / 2 + ( (p/2)**2 - q + eps)**.5)**.5

        image = sharpness_2*(1 - squircle)

        output = torch.sigmoid(image)

        if self.training:
            return output
        else:
            return (output > 0.5).float()


    def draw_colored_circle(self, args_tensor):
        """
        [..., N, 8] -> [..., N, 4, H, W]
        N shapes with 8 arguments each to N RGBA circle images
        """
        circle = self.draw_circle(args_tensor[..., :-3])
        rgb_ = args_tensor[..., -3:]
        rgb = unnormalize_to(rgb_, -1.5, 1.5) # 0 and 1 in the normalized realm

        return torch.concat([
            repeat(rgb, "... C -> ... C H W", H=self.imsize, W=self.imsize),
            rearrange(circle, "... (C H) W -> ... C H W", C=1)
        ], dim=-3)


    def composite_shapes(self, shape_images):
        """
        [..., N, 8] -> [..., 4, H, W] # DOES NOT HOLD ANYMORE
        N shapes with 8 arguments each -> RGBA image with shapes merged
        """
        shape_iterator_ = rearrange(shape_images, "... N C H W -> N ... C H W")
        # shape_iterator = unnormalize_to(shape_iterator_, 0.5, 1.0)
        shape_iterator = shape_iterator_


        # Get first image
        canvases = [self.transparent]

        # Add the rest of the images
        for image in shape_iterator:
            composite = self.alpha_composite(canvases[-1], image)
            canvases.append(composite)

        out = canvases[-1]

        return out


    def alpha_composite(self, base, added, eps=1e-8):
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


    def forward(self, args_tensor):
        sorted_args_tensor = soft_sort_by_column(args_tensor, column=1)

        if self.return_mode == "shapes":
            return sorted_args_tensor

        # [B N C W H]
        rencered_circles = self.draw_colored_circle(sorted_args_tensor)

        # [B C W H]
        rendered_images = self.composite_shapes(rencered_circles)

        return rendered_images
