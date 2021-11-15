# from models3_3 import CircleNetColorMulti
from models import CircleNetColorMulti
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


n_shapes = 8
# workers = 5
workers = 3
batch_size = 64
# batch_size = 2
image_size = 64
lr = 0.00001

date = datetime.today().strftime('%Y-%m-%d-%H.%M.%S')

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTrans():
    def __call__(self, tensor):
        return (tensor > 0.05).float()

def normalize_functional(tensor, mean, std):
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    return (tensor-mean)/std

def unnormalize_functional(tensor, mean, std):
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    return ((tensor*std)+mean).clamp(0, 1)

mean = [0.5772, 0.4661, 0.4431]
std = [0.4190, 0.3720, 0.3788]

my_transforms = transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

mean = torch.tensor(mean).to(device)
std =  torch.tensor(std).to(device)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=r"G:\Simon\Documents\programming\github\flag_gan\data",
                           transform=my_transforms)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, persistent_workers=(workers > 0), pin_memory=True)
dl_length = len(dataloader)


dataset_val = dset.ImageFolder(root=r"G:\Simon\Documents\programming\github\flag_gan\val",
                               transform=my_transforms)
workers = 0
# Create the dataloader
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                             shuffle=False, num_workers=workers, persistent_workers=(workers > 0), pin_memory=True)

# Create the networks
netG = CircleNetColorMulti(
    device=device,
    imsize=image_size,
    n_shapes=n_shapes
).to(device)

optimizer_netG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


def to_rgb(rgba):
    return rgba[..., :-1, :, :] * rgba[..., -1:, :, :]

white = torch.ones([image_size, image_size], dtype=torch.float).to(device)
def to_rgba(rgb):
    alpha_channel = white.unsqueeze(0).unsqueeze(0).expand(rgb.shape[0], -1, -1, -1)
    return torch.concat([rgb, alpha_channel], dim=-3)

def criterion(pred_image, target):
    rgba_target = to_rgba(target)

    return nn.functional.mse_loss(pred_image, rgba_target)

scaler = torch.cuda.amp.GradScaler()

fixed_target = next(iter(dataloader_val))[0].to(device)

torch.backends.cudnn.benchmark = True

# saved_state = torch.load(r"G:\Simon\Documents\programming\github\shapenet\checkpoints\2021-11-03-08.54.03\30000__0.07207467406988144.pt")
# netG.load_state_dict(saved_state["G"])
# optimizer_netG.load_state_dict(saved_state["opt"])

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

            
            netG.zero_grad()
            with torch.cuda.amp.autocast():
                pred = netG(target)
                err = criterion(pred, target)

            scaler.scale(err).backward()
            scaler.step(optimizer_netG)

            scaler.update()

            # Check how the generator is doing by saving G's output on fixed_noise
            if (global_step % 500 == 0):
                with torch.no_grad():
                    pred_val = netG(fixed_target)
                    err_val = criterion(pred_val, fixed_target)

                    netG.eval()
                    pred_val_crisp = netG(fixed_target)
                    netG.train()

                    pred_val = unnormalize_functional(to_rgb(pred_val), mean, std)
                    pred_val_crisp = unnormalize_functional(to_rgb(pred_val_crisp), mean, std)


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
                    "G": netG.state_dict(),
                    "opt": optimizer_netG.state_dict()
                },
                c_path / f"{global_step}__{err_val.item()}.pt")