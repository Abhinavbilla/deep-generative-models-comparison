import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from pathlib import Path
from PIL import Image


def set_seed(seed=1265):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class CelebALocal(Dataset):
    def __init__(self, root, image_size=64):
        self.files = sorted(list(Path(root).rglob("*.jpg")))

        if len(self.files) == 0:
            raise ValueError(
                f"No images found in {root}. "
                "Make sure dataset is placed in ./data/celeba"
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), 0



def get_cifar10_dataset(batch_size, image_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )


def get_dataloader(dataset_name, data_path, batch_size, image_size=64):
    if dataset_name.lower() == "celeba":
        dataset = CelebALocal(data_path, image_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

    elif dataset_name.lower() == "cifar10":
        return get_cifar10_dataset(batch_size, image_size)

    else:
        raise ValueError("Dataset must be either 'celeba' or 'cifar10'")