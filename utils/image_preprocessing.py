import torch
from torchvision import transforms
from PIL import Image as PILImage
import os

# Standartinis dydis visiems modeliams
IMAGE_SIZE = (224, 224)

#ToTensor() pikseliai tampa intervale [0,1] ir tada jie normalizuojami pagal sias reiksmes
# Normalizacijos reikšmės (ImageNet standartai - tinka CNN ir ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:  #grazina transformacijas treniruojant modeli
    """
    Treniravimo transformacijos su augmentacija.
    Augmentacija padeda modeliui geriau išmokti iš mažai duomenų.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),              # šiek tiek didesnis nei reikia
        transforms.RandomCrop(IMAGE_SIZE),          # random apkarpymas iki 224x224
        transforms.RandomHorizontalFlip(p=0.5),     # horizontalus apvertimas
        transforms.RandomRotation(degrees=15),      # pasukimas iki 15 laipsnių
        transforms.ColorJitter(                     # spalvų variacija
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),                      # PIL → PyTorch tensor
        transforms.Normalize(                       # normalizacija
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

#Si funkcija grazina transformacijas vaidacijai ir testui
def get_val_transforms() -> transforms.Compose:
    """
    Validavimo ir testavimo transformacijos — BE augmentacijos.
    Tik resize ir normalizacija.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),              # tiesiog resize į 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

#Si funkcija skirta vartotojo nuotraukai interfce metu. interface turi buti stabilus be flip rotate
def get_inference_transforms() -> transforms.Compose:
    """
    Transformacijos vartotojo nuotraukai (spėjimui).
    Tokios pačios kaip val — be augmentacijos.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


#si funkcija atidaro paveksleli ir uztikrina, kad jis butu RGB
def load_image(filepath: str) -> PILImage.Image:
    """
    Įkelia nuotrauką ir konvertuoja į RGB.
    Tvarko ir pilkos spalvos (grayscale) nuotraukas.
    """
    if not os.path.exists(filepath):  #iskart patikrina ar kitas failas egzistuoja
        raise FileNotFoundError(f"Nuotrauka nerasta: {filepath}")

    img = PILImage.open(filepath)   #atidaro paveiksleli

    
    # Konvertuok į RGB jei reikia
    with PILImage.open(filepath) as img:
        if img.mode != "RGB":   #jei paeikslelis yra grayscale, PNG su alpha, CMYK ar kitokio formato uztikrina, kad bus RGB
            img = img.convert("RGB")
        else:
            img = img.copy()
    return img

#si funkcija paruosia viena paveiksleli modeliui prognozei
def preprocess_for_inference(filepath: str) -> torch.Tensor:
    """
    Paruošia vieną nuotrauką modelio spėjimui.
    Grąžina tensor su batch dimensija: [1, 3, 224, 224]
    """
    img = load_image(filepath)
    transform = get_inference_transforms()
    tensor = transform(img)
    return tensor.unsqueeze(0)  # prideda batch dimensiją


#si funkcija atlieka atvirkstine normalizacija
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Atgalinė normalizacija — tensor → vaizduojama nuotrauka.
    Naudinga vizualizacijai ir debuginimui.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)