from filters import get_sobel_filter, get_static_fast_gauss_blur
from utils import *
from models import Model
from render_shape import ShapeRenderer
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.utils as vutils
from datetime import datetime
from pathlib import Path
from itertools import count
import tables


# Customizable variables
n_shapes = 16
workers = 3
batch_size = 64
image_size = 64
lr = 1e-5
log_interval = 500
save_model_interval = 10000
save_shape_args_interval = 10

# Automatic variables
date = datetime.today().strftime('%Y-%m-%d-%H.%M.%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean and std per channel for CelebA
mean = [0.5061, 0.4254, 0.3828]
std = [0.3043, 0.2838, 0.2833]

basic_transforms = T.Compose([
    T.Resize((image_size,image_size)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# Create image datasets and their dataloaders
dataset = ImageFolder(root=r"C:\projects\data\celeba", transform=basic_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=(workers > 0), pin_memory=True)
dl_length = len(dataloader)

dataset_val = ImageFolder(root=r"C:\projects\data\celeba_val", transform=basic_transforms)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
# Get a fixed batch that we evaluate against
fixed_target = next(iter(dataloader_val))[0].to(device)

# Create model, optimizer, and renderer
model = Model(n_shapes=n_shapes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
renderer = ShapeRenderer(device=device, imsize=image_size, minimum_sharpness=10).to(device)

# Create the loss function
# Normal L1/L2 will also do. Using additional features is experimental.
sobel_filter = get_sobel_filter(4, device)
gauss_filter = get_static_fast_gauss_blur(1, 2, device)

def criterion(pred_image, target):
    rgba_target = rgb_to_rgba(target)

    general_loss = nn.functional.mse_loss(pred_image, rgba_target)

    detail_loss = nn.functional.l1_loss(
        sobel_filter(pred_image),
        sobel_filter(rgba_target)
    )

    return 0.9*general_loss + 0.1*detail_loss

# Nice tools to speed up training
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

# Start training loop
if __name__ == "__main__":
    print("Starting Training Loop...")
    for epoch in count():
        for i, data in enumerate(dataloader, 0):
            global_step = (epoch*dl_length) + i

            # Training
            target = data[0].to(device)
            
            model.zero_grad()
            with torch.cuda.amp.autocast():
                shape_arguments = model(target)
                pred = renderer(shape_arguments)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Validation
            if (global_step % log_interval == 0):
                with torch.no_grad():
                    model.eval()
                    shape_arguments_val = model(fixed_target)
                    # Render output in smooth training mode
                    pred_val = renderer(shape_arguments_val)
                    loss_val = criterion(pred_val, fixed_target)

                    # Render output in crisp eval mode
                    renderer.eval()
                    pred_val_crisp = renderer(shape_arguments_val)

                    pred_val = unnormalize_functional(rgba_to_rgb(pred_val), mean, std)
                    pred_val_crisp = unnormalize_functional(rgba_to_rgb(pred_val_crisp), mean, std)

                    # Reset both modules to training
                    renderer.train()
                    model.train()

                print(f"Global step: {global_step}\tValidation loss:", loss_val.item())

                # Save images in /progress/<date>/<name>.png
                im_target = vutils.make_grid(fixed_target, padding=2, normalize=True)
                im_pred = vutils.make_grid(pred_val, padding=2, normalize=True)
                im_pred_crisp = vutils.make_grid(pred_val_crisp, padding=2, normalize=True)
                total_image = torch.concat(
                    [im_target, im_pred, im_pred_crisp], dim=-2).detach().cpu()
                i_path = Path(f"./progress/{date}")
                i_path.mkdir(parents=True, exist_ok=True)
                vutils.save_image(total_image, i_path / f"{global_step}__{loss_val.item()}.png")

            # Save model in /checkpoints/<date>/<name>.pt
            if (global_step % save_model_interval == 0):
                c_path = Path(f"./checkpoints/{date}")
                c_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                },
                c_path / f"{global_step}__{loss_val.item()}.pt")

            # Save shape arguments to interpolate between them later
            if (global_step % save_shape_args_interval == 0):
                with torch.no_grad():
                    model.eval()
                    shape_arguments_save = model(fixed_target)
                    model.train()

                i_path = Path(f"./progress/{date}")
                i_path.mkdir(parents=True, exist_ok=True)
                f_path = i_path / "batch_outputs.h5"
                if not f_path.exists():
                    f = tables.open_file(str(f_path), mode='w')
                    atom = tables.Float64Atom()
                    batches_ea = f.create_earray(f.root, 'batches', atom, shape=(0, *shape_arguments_save.shape))
                else:
                    f = tables.open_file(str(f_path), mode='a')
                    f.root.batches.append(shape_arguments_save.unsqueeze(0).cpu().numpy())
                f.close()
