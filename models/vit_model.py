import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor      #ViTForImageClassification — HuggingFace pretrained ViT modelis. ViTImageProcessor — specialus įrankis kuris paruošia nuotraukas ViT formatui (resize, normalizacija pagal ImageNet statistiką).
from sqlalchemy.orm import Session

from utils.constants import CATEGORIES, NUM_CLASSES, IDX_TO_CATEGORY

# HuggingFace pretrained modelio pavadinimas
VIT_MODEL_NAME = "google/vit-base-patch16-224"

# Train augmentacijos prieš ViTImageProcessor
TRAIN_AUGMENTATIONS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),                                    # atsitiktinai apkarpoma nuotrauka ir pakeičiamas dydis į 224x224. scale=(0.8, 1.0) reiškia apkarpoma nuo 80% iki 100% originalaus dydžio.
    transforms.RandomHorizontalFlip(p=0.5),                                                 # 50% tikimybe nuotrauka apverčiama horizontaliai.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),                   # atsitiktinai keičiamas ryškumas, kontrastas ir sodrumas iki 20%.
])


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) modelis nuotraukų kategorizavimui.
    Naudoja pretrained google/vit-base-patch16-224 ir fine-tunina
    paskutinį klasifikavimo sluoksnį mūsų 5 kategorijoms.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()

        self.vit = ViTForImageClassification.from_pretrained(                                   # parsisiunčia pretrained Google ViT modelį iš HuggingFace ir išsaugo kaip klasės atributą.
            VIT_MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        if freeze_backbone:                                                                      # backbone svoriai neliečiami, treniruojamas tik classifier. Greičiau, mažiau duomenų reikia.
            for param in self.vit.vit.parameters():                                              # backbone — pagrindinė dalis kuri jau "supranta" nuotraukas — ištraukia savybes. Ji istreniruota ant milijonų nuotraukų.
                param.requires_grad = False
            print("ViT backbone užšaldytas — treniruojamas tik classifier")                     
        else:
            print("ViT fine-tuning — treniruojamas visas modelis")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=x)                                                      # paleidžia nuotrauką per ViT modelį. pixel_values yra būtinas parametro pavadinimas kurį ViT reikalauja.
        return outputs.logits                                                                   # grąžina 5 skaičius kiekvienai kategorijai. logits yra neapdoroti spėjimai prieš softmax — vėliau predict_single_vit funkcija pritaiko softmax ir gauna tikimybes.


class ViTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset ViT modeliui.
    Train splite naudoja augmentacijas, val/test — tik processor.
    """

    def __init__(self, image_records: list, processor: ViTImageProcessor, is_train: bool = False):
        self.records = image_records
        self.processor = processor
        self.is_train = is_train
        self.category_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]

        from utils.image_preprocessing import load_image
        img = load_image(record.filepath)

        # Train splitui — augmentacijos prieš processor
        if self.is_train:
            img = TRAIN_AUGMENTATIONS(img)

        # ViTImageProcessor tvarko resize ir normalizaciją
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 224, 224]

        label = self.category_to_idx[record.category]
        return pixel_values, label


def build_vit_dataloaders(db: Session, batch_size: int = 16) -> tuple[dict, ViTImageProcessor]:
    """
    Sukuria train / val / test DataLoader objektus ViT modeliui.
    Grąžina (loaders, processor).
    """
    from database.models import TrainingImage

    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

    # Patikrink ar splitai nėra tušti
    for split in ["train", "val"]:
        count = db.query(TrainingImage).filter_by(split=split).count()
        if count == 0:
            raise ValueError(
                f"'{split}' splitas tuščias! "
                f"Paleisk load_data_to_db() prieš treniravimą."
            )

    test_count = db.query(TrainingImage).filter_by(split="test").count()
    if test_count == 0:
        print("'test' splitas tuščias — testavimas nebus galimas.")

    loaders = {}
    for split in ["train", "val", "test"]:
        records = db.query(TrainingImage).filter_by(split=split).all()
        dataset = ViTDataset(
            records,
            processor=processor,
            is_train=(split == "train")  # augmentacijos tik train splitui
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"📦 {split}: {len(dataset)} nuotraukų | augmentacija: {split == 'train'}")

    return loaders, processor


def predict_single_vit(
    model: ViTModel,
    processor: ViTImageProcessor,
    filepath: str,
    device: str = "cpu"
) -> tuple[str, float]:
    """
    Spėja vieną nuotrauką naudojant ViT.
    Grąžina (kategorija, confidence).
    """
    from utils.image_preprocessing import load_image

    model.eval()
    model = model.to(device)

    img = load_image(filepath)
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values)
        proba = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(proba))
        confidence = float(proba[idx])
        category = IDX_TO_CATEGORY[idx]

    return category, confidence


def get_vit_model(freeze_backbone: bool = False) -> ViTModel:
    """Grąžina naują ViT modelį."""
    return ViTModel(num_classes=NUM_CLASSES, freeze_backbone=freeze_backbone)


def save_vit_model(model: ViTModel, path: str) -> None:
    """Išsaugo ViT modelį į failą."""
    torch.save(model.state_dict(), path)
    print(f"ViT modelis išsaugotas: {path}")


def load_vit_model(path: str, freeze_backbone: bool = False) -> ViTModel:
    """Įkelia ViT modelį iš failo."""
    model = ViTModel(num_classes=NUM_CLASSES, freeze_backbone=freeze_backbone)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"ViT modelis įkeltas: {path}")
    return model