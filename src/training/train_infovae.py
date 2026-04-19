# TRAINING INFOVAE CODE


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from torchvision import models
from torchvision.models import VGG16_Weights
from src.models.info_vae import InfoVAE
from src.training.utils import set_seed, get_dataloader



class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2
        return nn.functional.l1_loss(self.vgg(x), self.vgg(y))


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--reg_weight", type=float, default=50.0)

    parser.add_argument("--perceptual_w", type=float, default=0.1)

    args = parser.parse_args()

    
    set_seed(1265)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    loader = get_dataloader(args.data_root, args.batch_size)

    model = InfoVAE(
        latent_dim=args.latent_dim,
        alpha=args.alpha,
        beta=args.beta,
        reg_weight=args.reg_weight
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    
    perceptual_fn = VGGPerceptualLoss().to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"InfoVAE Epoch {epoch}")

        for x, _ in loop:
            x = x.to(device)

            optimizer.zero_grad()

            out = model(x)

            
            p_loss = perceptual_fn(out[0], x)

            loss_dict = model.loss_function(
                *out,
                M_N=args.batch_size / len(loader.dataset),
                perceptual_loss=p_loss,
                perceptual_w=args.perceptual_w
            )

            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=loss.item(),
                recon=loss_dict["Reconstruction_Loss"].item(),
                kld=loss_dict["KLD"].item(),
                mmd=loss_dict["MMD"].item()
            )

        print(f"Epoch {epoch}: {total_loss / len(loader):.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train()

