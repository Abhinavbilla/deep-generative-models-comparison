import os
import torch
from torchvision.utils import save_image


def save_checkpoint(model, epoch, config):
    model_dir = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(model_dir, f"{config.model_name}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), path)


def save_reconstructions(model, x, epoch):
    model.eval()

    with torch.no_grad():
        recon = model.generate(x)

    os.makedirs("outputs", exist_ok=True)

    save_image(x[:8], f"outputs/original_epoch_{epoch}.png", nrow=4, normalize=True)
    save_image(recon[:8], f"outputs/recon_epoch_{epoch}.png", nrow=4, normalize=True)