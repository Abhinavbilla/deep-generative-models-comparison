# TRAINING VQVAE CODE 

import torch
from torch import optim
from tqdm import tqdm
import argparse

from src.models.vq_vae import VQVAE
from src.training.utils import set_seed, get_dataloader


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data/celeba")
    parser.add_argument("--dataset", type=str, default="celeba")  # NEW
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    set_seed(1265)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    loader = get_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        batch_size=args.batch_size,
        image_size=64
    )

    model = VQVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"VQ-VAE Epoch {epoch}")

        for x, _ in loop:
            x = x.to(device)

            optimizer.zero_grad()

            loss_dict = model.compute_loss(x)
            loss = loss_dict["loss"]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch}: {total_loss / len(loader):.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train()