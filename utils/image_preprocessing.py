import torch
from torchvision import transforms
from PIL import Image as PILImage
import os

# Standartinis dydis visiems modeliams
from utils.constants import IMAGE_SIZE

import json
import cv2
import numpy as np
from skimage.feature import hog


# ImageNet normalizacijos reikšmės (tinka CNN ir ViT)                                   #stadartines CNN ir ViT modeliams visos nuotraukos normalizuojamos vienodai, kad modelis geriau mokytusi
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:                                       #paruosia nuotraukas treniravimui su augmentacija
    """Treniravimo transformacijos su augmentacija."""
    return transforms.Compose([
        transforms.Resize((256, 256)),                                                  #padidina/sumazina. standartizuoja dydi
        transforms.RandomCrop(IMAGE_SIZE),                                              #atsitiktinai iskerpa 224x224 is 256x5=256
        transforms.RandomHorizontalFlip(p=0.5),                                         #50 proc tikimybe, kad apvercia nuotrauka horizontaliai kaip veidrodis
        transforms.RandomRotation(degrees=15),                                          #atsitiktinai pasuka nuotrauka -/+15 laipsniu
        transforms.ColorJitter(                                                         #atsitiktinai keicia spalvu savybes. kiekviena epocha siek tiek kitaip
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),                                                          #pavercia pil nuotrauka i pytorch tensor. pil: (aukstis 224, plotis 224, kanalai3 ->torch: 3kanalai, 224 aukstis, 224 plotis) pytorch greiciau skaiciuoja kai kanalai yra pirmi
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)                      #normalizuojami duomenys
    ])


def get_val_transforms() -> transforms.Compose:                                         #paruosia nuotraukas validavimui be augmentacijos, visada vienodai, kad rezultatai butu palyginami tarp epochu
    """
    Validavimo ir testavimo transformacijos.
    Resize(256) → CenterCrop(224) išsaugo proporcijas.
    """
    return transforms.Compose([
        transforms.Resize(256),                                                         #pakeicia nuotraukos dydi taip,kad trumpesne krastine butu 256
        transforms.CenterCrop(IMAGE_SIZE),                                              #iskerpa is centro
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms() -> transforms.Compose:                                   #paruosia vartotojo ikelta nuotrauka spejimui
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


def load_image(filepath: str) -> PILImage.Image:                                            #tiesiog atidaoro nuotrauka is disko ir pavercia i RGB. Naudojama pries transformacijas
    """Įkelia nuotrauką ir konvertuoja į RGB."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nuotrauka nerasta: {filepath}")

    with PILImage.open(filepath) as img:
        img = img.convert("RGB")
        return img.copy()


def preprocess_for_inference(filepath: str) -> torch.Tensor:                                  #paruosa viena nuotrauka modelio spejimui. transformuoja ir padeda batch dimensija
    """
    Paruošia vieną nuotrauką modelio spėjimui.
    Grąžina tensor: [1, 3, 224, 224]
    """
    img = load_image(filepath)
    transform = get_inference_transforms()
    tensor = transform(img)
    return tensor.unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:                                          #atgaline normalizacija. tik debug ir vizualizacijai. Pavercia tensor atgal i normalia nuotrauka, kad galetum pamatyti ka modelis mato
    """Atgalinė normalizacija — vizualizacijai ir debuginimui."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def extract_hog_features(image_path: str) -> str | None:                                        #istraukia HOG pozymius - naudojama KNN modeliui
    """
    Ištraukia HOG požymius iš nuotraukos ir grąžina JSON tekstą.
    Jei nepavyksta – grąžina None.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True,
        )

        return json.dumps(features.tolist())

    except Exception as e:
        print(f"HOG išgavimas nepavyko {image_path}: {e}")
        return None