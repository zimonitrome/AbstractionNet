# from models3_3 import CircleNetColorMulti
from filters import get_sobel_filter, get_static_fast_gauss_blur
from models import Model
import torch
from torch import nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datetime import datetime
from pathlib import Path
from itertools import count
import torch.multiprocessing as mp
import tables
import numpy as np

from render_shape import ShapeRenderer


n_shapes = 16
workers = 3
batch_size = 64
image_size = 64
lr = 1e-5

date = datetime.today().strftime('%Y-%m-%d-%H.%M.%S')

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_functional(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(device)
    std = torch.tensor(std).view(3, 1, 1).to(device)
    return (tensor-mean)/std

def unnormalize_functional(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(device)
    std = torch.tensor(std).view(3, 1, 1).to(device)
    return ((tensor*std)+mean).clamp(0, 1)

mean = [0.5061, 0.4254, 0.3828]
std = [0.3043, 0.2838, 0.2833]

my_transforms = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=r"C:\projects\data\celeba",
                           transform=my_transforms)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, persistent_workers=(workers > 0), pin_memory=True)
dl_length = len(dataloader)


dataset_val = dset.ImageFolder(root=r"C:\projects\data\celeba_val",
                               transform=my_transforms)
workers = 0
# Create the dataloader
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16,
                                             shuffle=False, num_workers=workers, persistent_workers=(workers > 0), pin_memory=True)

# Create the network
model = Model(n_shapes=n_shapes).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

renderer = ShapeRenderer(device=device, imsize=image_size, minimum_sharpness=10).to(device)


sobel_filter = get_sobel_filter(4, device)
gauss_filter = get_static_fast_gauss_blur(1, 2, device)

def to_rgb(rgba):
    return rgba[..., :-1, :, :] * rgba[..., -1:, :, :]

white = torch.ones([image_size, image_size], dtype=torch.float).to(device)
def to_rgba(rgb):
    alpha_channel = white.unsqueeze(0).unsqueeze(0).expand(rgb.shape[0], -1, -1, -1)
    return torch.concat([rgb, alpha_channel], dim=-3)

def criterion(pred_image, target):
    rgba_target = to_rgba(target)

    general_loss = nn.functional.mse_loss(pred_image, rgba_target)

    detail_loss = nn.functional.l1_loss(
        sobel_filter(pred_image),
        sobel_filter(rgba_target)
    )

    return 0.9*general_loss + 0.1*detail_loss

scaler = torch.cuda.amp.GradScaler()

fixed_target = next(iter(dataloader_val))[0].to(device)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in count():
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            global_step = (epoch*dl_length) + i

            target = data[0].to(device)
            b_size = target.size(0)

            
            model.zero_grad()
            with torch.cuda.amp.autocast():
                pred = renderer(model(target))
                err = criterion(pred, target)

            scaler.scale(err).backward()
            scaler.step(optimizer)

            scaler.update()

            # Check how the generator is doing by saving G's output on fixed_noise
            if (global_step % 500 == 0):
                with torch.no_grad():
                    # Render output in smooth training mode
                    model.eval()
                    model_pred_val = model(fixed_target)
                    pred_val = renderer(model_pred_val)
                    err_val = criterion(pred_val, fixed_target)

                    # Render output in crisp eval mode
                    renderer.eval()
                    pred_val_crisp = renderer(model_pred_val)

                    pred_val = unnormalize_functional(to_rgb(pred_val), mean, std)
                    pred_val_crisp = unnormalize_functional(to_rgb(pred_val_crisp), mean, std)

                    # Reset both modules to training
                    renderer.train()
                    model.train()


                print(err_val.item())
                im_target = vutils.make_grid(fixed_target, padding=2, normalize=True)
                im_pred = vutils.make_grid(pred_val, padding=2, normalize=True)
                im_pred_crisp = vutils.make_grid(pred_val_crisp, padding=2, normalize=True)
                total_image = torch.concat(
                    [im_target, im_pred, im_pred_crisp], dim=-2).detach().cpu()
                i_path = Path(f"./progress/{date}")
                i_path.mkdir(parents=True, exist_ok=True)
                vutils.save_image(total_image, i_path / f"{global_step}__{err_val.item()}.png")

            if (global_step % 10000 == 0):
                c_path = Path(f"./checkpoints/{date}")
                c_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "G": model.state_dict(),
                    "opt": optimizer.state_dict()
                },
                c_path / f"{global_step}__{err_val.item()}.pt")

            # Save tensors each output
            with torch.no_grad():
                model.return_mode = "shapes"
                model.eval()
                pred_val_crisp = model(fixed_target)
                model.return_mode = "bitmap"
                model.train()

            i_path = Path(f"./progress/{date}")
            i_path.mkdir(parents=True, exist_ok=True)
            f_path = i_path / "batch_outputs.h5"
            if not f_path.exists():
                f = tables.open_file(str(f_path), mode='w')
                atom = tables.Float64Atom()
                batches_ea = f.create_earray(f.root, 'batches', atom, shape=(0, *pred_val_crisp.shape))
            else:
                f = tables.open_file(str(f_path), mode='a')
                f.root.batches.append(pred_val_crisp.unsqueeze(0).cpu().numpy())
            f.close()
