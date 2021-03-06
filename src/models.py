import torch
from torch import nn
import torchvision.models as models

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class Model(nn.Module):
    def __init__(self, n_shapes=2):
        super().__init__()

        backbone = models.resnet18()

        parameter_fields = [
            "sharpness", 
            "pos_z", 
            "pos_x", 
            "pos_y",
            "width", 
            "height", 
            "rotation", 
            "squareness", 
            "red", 
            "green", 
            "blue"
        ]
        npf = len(parameter_fields)

        self.main = nn.Sequential(
            backbone,
            nn.Linear(1000, 32 * n_shapes*npf),
            nn.Linear(32 * n_shapes*npf, 8  * n_shapes*npf),
            nn.Linear(8*n_shapes*npf, n_shapes*npf),
            nn.Unflatten(1,   [n_shapes,npf]),
            nn.Sigmoid(),
        )

        initial_bias = self.grid_init(n_shapes, npf)
        self.main[-3].bias.data.copy_(
            initial_bias.flatten().float()
        )

    @staticmethod
    def grid_init(n_shapes, n_parameter_fields):
        # Initialize all shapes to be in 4x4 grid (Optional but looks cool)
        nrow = int(n_shapes**.5)
        radius = 1/(2*nrow)
        xy_positions = torch.linspace(0+radius, 1-radius, nrow)
        x_positions, y_positions = torch.meshgrid(xy_positions, xy_positions)
        unit_tensor = torch.full([n_shapes, n_parameter_fields], 0.5)
        unit_tensor[:, 2] = x_positions.flatten()
        unit_tensor[:, 3] = y_positions.flatten()
        unit_tensor[:, 4] = radius-0.02
        unit_tensor[:, 5] = radius-0.02
        return torch.logit(unit_tensor)

    def forward(self, input_tensor):
        return self.main(input_tensor)
