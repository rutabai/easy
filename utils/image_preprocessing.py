import torch
from torchvision import transforms
from PIL import Image as PILImage
import os

# Standartinis dydis visiems modeliams
from utils.constants import IMAGE_SIZE

# ImageNet normalizacijos reikšmės (tinka CNN ir ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Treniravimo transformacijos su augmentacija."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Validavimo ir testavimo transformacijos.
    Resize(256) → CenterCrop(224) išsaugo proporcijas.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Transformacijos vartotojo nuotraukai spėjimui.
    Identiškas val transformacijoms.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_image(filepath: str) -> PILImage.Image:
    """Įkelia nuotrauką ir konvertuoja į RGB."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nuotrauka nerasta: {filepath}")

    with PILImage.open(filepath) as img:
        img = img.convert("RGB")
        return img.copy()


def preprocess_for_inference(filepath: str) -> torch.Tensor:
    """
    Paruošia vieną nuotrauką modelio spėjimui.
    Grąžina tensor: [1, 3, 224, 224]
    """
    img = load_image(filepath)
    transform = get_inference_transforms()
    tensor = transform(img)
    return tensor.unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Atgalinė normalizacija — vizualizacijai ir debuginimui."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)