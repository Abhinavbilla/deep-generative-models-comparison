# TRAINING StandardVAE CODE

import torch
import torch.optim as optim
from tqdm import tqdm
import argparse

from src.models.standard_vae import VanillaVAE
from src.training.utils import set_seed, get_dataloader


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--kld_weight", type=float, default=0.00025)

    args = parser.parse_args()

    
    set_seed(1265)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    loader = get_dataloader(args.data_root, args.batch_size)

    model = VanillaVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"VAE Epoch {epoch}")

        for x, _ in loop:
            x = x.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss_dict = model.loss_function(*out, M_N=args.kld_weight)

            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=loss.item(),
                recon=loss_dict["Reconstruction_Loss"].item(),
                kld=loss_dict["KLD"].item()
            )

        print(f"Epoch {epoch}: {total_loss / len(loader):.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train()