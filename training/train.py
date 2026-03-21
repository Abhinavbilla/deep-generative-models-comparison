import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm

from training.config import Config
from training.utils import save_checkpoint, save_reconstructions
from torchvision.datasets import ImageFolder

# =========================
# Setup
# =========================
config = Config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")


# =========================
# Dataset
# =========================
transform = transforms.Compose([
    transforms.CenterCrop(148),
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])



dataset = ImageFolder(
    root="./data/celeba",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True
)


# =========================
# Model Factory
# =========================
def get_model(config):
    if config.model_name == "vqvae":
        from models.vq_vae import VQVAE
        return VQVAE(
            in_channels=3,
            embed_dim=config.embed_dim,
            vocab_size=config.num_embeddings,
            beta=config.beta
        )

    elif config.model_name == "vae":
        from models.standard_vae import VAE
        return VAE()

    elif config.model_name == "infovae":
        from models.info_vae import InfoVAE
        return InfoVAE()

    else:
        raise ValueError(f"Unknown model: {config.model_name}")


# =========================
# Initialize Model
# =========================
model = get_model(config).to(device)


# =========================
# Optimizer
# =========================
optimizer = Adam(model.parameters(), lr=config.lr)


# =========================
# Training Loop
# =========================
for epoch in range(1, config.epochs + 1):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"{config.model_name.upper()} Epoch {epoch}/{config.epochs}")

    for i, (x, _) in enumerate(pbar):
        x = x.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(x)

        # Loss
        loss_dict = model.loss_function(*outputs)
        loss = loss_dict["loss"]

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logging
        if i % config.log_interval == 0:
            log_data = {"loss": loss.item()}

            if "Reconstruction_Loss" in loss_dict:
                log_data["recon"] = loss_dict["Reconstruction_Loss"].item()

            if "VQ_Loss" in loss_dict:
                log_data["vq"] = loss_dict["VQ_Loss"].item()

            if "Codebook_Utilization" in loss_dict:
                log_data["util"] = loss_dict["Codebook_Utilization"]

            pbar.set_postfix(log_data)

    # =========================
    # Epoch Summary
    # =========================
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.4f}")

    # =========================
    # Save Model
    # =========================
    save_checkpoint(model, epoch, config)

    # =========================
    # Save Reconstructions
    # =========================
    save_reconstructions(model, x, epoch)