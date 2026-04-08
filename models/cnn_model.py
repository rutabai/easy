import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Kategorijos — turi sutapti su data_loader.py
CATEGORIES = ["food", "portrait", "landscape", "product", "lifestyle"]
NUM_CLASSES = len(CATEGORIES)


class CNNModel(nn.Module):
    """
    CNN modelis nuotraukų kategorizavimui.
    Architektūra: 4x (Conv2D → BatchNorm → ReLU → MaxPool) → Flatten → Dropout → FC → FC
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # ── Konvoliucinis blokas ───────────────────────────────────
        self.features = nn.Sequential(
            # 1 blokas: 3 → 32 kanalai | 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2 blokas: 32 → 64 kanalai | 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3 blokas: 64 → 128 kanalai | 56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 4 blokas: 128 → 256 kanalai | 28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Klasifikavimo blokas ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset treniravimo nuotraukoms.
    Naudoja TrainingImage įrašus iš DB.
    """

    def __init__(self, image_records: list, transform=None):
        self.records = image_records
        self.transform = transform
        self.category_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]

        from utils.image_preprocessing import load_image
        img = load_image(record.filepath)

        if self.transform:
            img = self.transform(img)

        label = self.category_to_idx[record.category]
        return img, label


def build_dataloaders(db, batch_size: int = 32) -> dict:
    """Sukuria train / val / test DataLoader objektus iš DB duomenų."""
    from database.models import TrainingImage
    from utils.image_preprocessing import get_train_transforms, get_val_transforms

    transforms_map = {
        "train": get_train_transforms(),
        "val":   get_val_transforms(),
        "test":  get_val_transforms(),
    }

    loaders = {}
    for split, transform in transforms_map.items():
        records = db.query(TrainingImage).filter_by(split=split).all()
        dataset = CNNDataset(records, transform=transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"📦 {split}: {len(dataset)} nuotraukų")

    return loaders


def get_model(dropout: float = 0.5) -> CNNModel:
    """Grąžina naują CNN modelį."""
    return CNNModel(num_classes=NUM_CLASSES, dropout=dropout)


def save_model(model: CNNModel, path: str) -> None:
    """Išsaugo modelį į failą."""
    torch.save(model.state_dict(), path)
    print(f"✅ Modelis išsaugotas: {path}")


def load_model(path: str, dropout: float = 0.5) -> CNNModel:
    """Įkelia modelį iš failo."""
    model = CNNModel(num_classes=NUM_CLASSES, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"✅ Modelis įkeltas: {path}")
    return model