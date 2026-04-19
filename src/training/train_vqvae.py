# TRAINING VQVAE CODE 



import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import random
import numpy as np

from src.models.vq_vae import VQVAE


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class CelebALocal(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(list(Path(root).rglob("*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), 0


def get_dataloader(path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = CelebALocal(path, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )



def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_dataloader(args.data_root, args.batch_size)

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