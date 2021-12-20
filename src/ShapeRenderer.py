import torch
from torch import nn
from einops import rearrange, repeat
from src.SoftSortByColumn import soft_sort_by_column
from src.utils import alpha_composite_multiple, unnormalize_to


class ShapeRenderer(nn.Module):

    def __init__(self, device="cpu", imsize=64, minimum_sharpness=1):
        super().__init__()
        self.imsize = imsize

        ramp = torch.linspace(0, 1, self.imsize).to(device)
        self.x_ramp, self.y_ramp = torch.meshgrid(ramp, ramp, indexing=None)

        self.min_sharpness = minimum_sharpness
        self.max_sharpness = 20
        self.evaluation_sharpness = 200


    def process_shape_arguments(self, shape_arguments):
        """
        Processes sigmoided outputs from the model to desired values.
        Most values are good within the [0, 1] range but a few arguments need different scales.
        [B N A] -> [B N A]
        """
        evaluation_mode = not self.training
        shape_arguments = soft_sort_by_column(shape_arguments, regularization="l2", regularization_strength=1.0, column=1)
        new_shape_arguments = shape_arguments.clone()

        # Rescale some arguments
        # All arguments are assumed to be in the range [0, 1] due to sigmoid in model.

        # Rescale RGB from the range [0, 1] to [-1.5, 1.5]
        # These values roughly correspond to 0 and 1 in the normalized color range.
        # Thus, we can unnormalize the image later using the channel mean and std.
        # TODO: This mapping can be learned by the network instead. Better?
        new_shape_arguments[..., -3:] = unnormalize_to(shape_arguments[..., -3:], -1.5, 1.5)

        # Rescale angle from [0, 1] to [0, pi]
        new_shape_arguments[..., 6] = torch.pi * shape_arguments[..., 6]

        # Rescale x_radius and y_radius from [0, 1] to [eps, 1+eps]
        # Calculate what 1 pixel of the image is as a proportion of [0, 1]
        # Add the half of this value to each radius to assure a minimum size of 1 px per rendered shape.
        half_of_1px = (1 / self.imsize) / 2
        new_shape_arguments[..., 3:5] = shape_arguments[..., 3:5] + half_of_1px

        if evaluation_mode:
            # Make shapes have hard edges
            new_shape_arguments[..., 0] = self.evaluation_sharpness

            # Make shape either square or circle
            new_shape_arguments[..., 7] = torch.round(shape_arguments[..., 7])
            print("AA", new_shape_arguments.min(), new_shape_arguments.mean(), new_shape_arguments.max())
        else:
            # Rescale sharpness to min and max values
            new_shape_arguments[..., 0] = unnormalize_to(shape_arguments[..., 0], self.min_sharpness, self.max_sharpness)

            # Helps restrict inf/nan gradients... maybe
            # TODO: Is this transformation needed?
            new_shape_arguments[..., 7] = unnormalize_to(shape_arguments[..., 7], 0.1, 0.9)


        return new_shape_arguments



    def draw_squircle(self, args_tensor):
        """
        Tranforms an argument tensor into an image of shape.
        This is where the "actual rendering" happens.
        
        [..., A] -> [..., H, W]
        """
        eps = 1e-8

        # Move argument dimension furthest out and add 2 dimensions
        sharpness, pos_z, pos_x, pos_y, radius_x, radius_y, angle, squareness = repeat(args_tensor, "... (A H W) -> A ... H W", H=1, W=1)
        print("bb_0", args_tensor.min(), args_tensor.mean(), args_tensor.max())

        # Get ramps ready (efficient way)
        # Get any trailing dimensions (i.e. [...])
        t_shape = list(args_tensor.shape)[:-1]
        # Reshape [H, W] -> [..., H, W]
        x_ramp = self.x_ramp.view(*len(t_shape)*[1], *self.x_ramp.shape).expand(*t_shape, *self.x_ramp.shape)
        y_ramp = self.y_ramp.view(*len(t_shape)*[1], *self.y_ramp.shape).expand(*t_shape, *self.y_ramp.shape)

        # Transform the coordinate system
        # transform, rotate, scale
        # TODO: Try affine trasnforms instead
        translated_x = x_ramp - pos_x
        translated_y = y_ramp - pos_y

        rotated_x = translated_x*torch.cos(angle) - translated_y*torch.sin(angle)
        rotated_y = translated_x*torch.sin(angle) + translated_y*torch.cos(angle)

        scaled_x = rotated_x / radius_x
        scaled_y = rotated_y / radius_y

        # Draw Fernández–Guasti squircle
        # https://en.wikipedia.org/wiki/Squircle#Fern%C3%A1ndez%E2%80%93Guasti_squircle
        x_s = scaled_x**2
        y_s = scaled_y**2
        print("bb", scaled_x.min(), scaled_x.mean(), scaled_x.max())
        p = -(x_s + y_s)
        print("p", p.min(), p.mean(), p.max())
        q = x_s * y_s * squareness
        print("q", q.min(), q.mean(), q.max())
        squircle = (-p / 2 + ( (p/2)**2 - q + eps)**.5 + eps)**.5

        print("BB", squircle.min(), squircle.mean(), squircle.max())


        # Adjust sharpness
        image = sharpness*(1 - squircle)
        image = torch.sigmoid(image)
        print("CC", squircle.min(), squircle.mean(), squircle.max())

        return image


    def draw_colored_squircle(self, args_tensor):
        """
        Tranforms an argument tensor into an image of shape of color.

        [..., N, A] -> [..., N, 4, H, W]
        """
        squircle = self.draw_squircle(args_tensor[..., :-3])
        rgb = args_tensor[..., -3:]

        return torch.concat([
            repeat(rgb, "... C -> ... C H W", H=self.imsize, W=self.imsize),
            rearrange(squircle, "... (C H) W -> ... C H W", C=1)
        ], dim=-3)


    def forward(self, shape_arguments):
        shape_arguments = self.process_shape_arguments(shape_arguments)

        # [B N C W H]
        rencered_squircles = self.draw_colored_squircle(shape_arguments)

        # [B C W H]
        rendered_images = alpha_composite_multiple(rencered_squircles)

        return rendered_images
