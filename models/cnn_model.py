import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Kategorijos — turi sutapti su data_loader.py
from utils.constants import CATEGORIES, NUM_CLASSES


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
            nn.Conv2d(3, 32, kernel_size=3, padding=1),                                     #konvoliucinis sluoksnis. 3 = RGB kanalai, 32 = kiek savybių ieško, padding=1 = prideda kraštus kad nuotrauka nesumažėtų.
            nn.BatchNorm2d(32),                                                             #nn.BatchNorm2d(32) — normalizuoja skaičius kad treniravimas būtų stabilesnis ir greitesnis.
            nn.ReLU(inplace=True),                                                          # aktyvacijos funkcija. Neigiamas reikšmes paverčia į 0, teigiamas palieka. Tai leidžia modeliui mokytis nelinijinių savybių.
            nn.MaxPool2d(kernel_size=2, stride=2),                                          #umažina nuotrauką per pusę pasiimant didžiausią reikšmę iš kiekvieno 2x2 langelio. Todėl 224 → 112.

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
            nn.Flatten(),                                                           #averčia 3D matricą (256 kanalai, 14x14 pikseliai) į vieną ilgą skaičių sąrašą — 256 * 14 * 14 = 50176 skaičių.
            nn.Dropout(p=dropout),                                                  #atsitiktinai išjungia 50% neuronų treniravimo metu — neleidžia persimokti.
            nn.Linear(256 * 14 * 14, 512),                                          #pilnai sujungtas sluoksnis, sumažina 50176 skaičių iki 512.
            nn.ReLU(inplace=True),                                                  ## aktyvacijos funkcija. Neigiamas reikšmes paverčia į 0, teigiamas palieka. Tai leidžia modeliui mokytis nelinijinių savybių.
            nn.Dropout(p=dropout / 2),                                              # dar kartą dropout bet silpnesnis — 25%.
            nn.Linear(512, num_classes),                                            #paskutinis sluoksnis, sumažina 512 iki 5 (kategorijų skaičius). Šie 5 skaičiai yra galutiniai spėjimai kiekvienai kategorijai.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:                             #Tai funkcija kuri aprašo kaip duomenys juda per modelį.
        x = self.features(x)                                                        #nuotrauka eina per 4 konvoliucinius blokus.
        x = self.classifier(x)                                                      #rezultatas eina per klasifikavimo bloką
        return x                                                                    #grąžina 5 skaičius — po vieną kiekvienai kategorijai. Didžiausias skaičius = modelio spėjama kategorija.


class CNNDataset(torch.utils.data.Dataset):                                         #yra PyTorch duomenų rinkinio klasė kuri susieja DB įrašus su treniravimo procesu. Ji paveldi iš torch.utils.data.Dataset — tai PyTorch standartinis interfeisas. DataLoader reikalauja kad duomenų rinkinys turėtų __len__ ir __getitem__ metodus — ši klasė juos įgyvendina. Trumpai: ji paima TrainingImage įrašus iš DB ir paverčia juos į formą kurią PyTorch supranta.

    """
    PyTorch Dataset treniravimo nuotraukoms.
    Naudoja TrainingImage įrašus iš DB.
    """

    def __init__(self, image_records: list, transform=None):                        # is esmes paruosia data set objekta | Inicializuoja dataset objektą — išsaugo tris dalykus kurie reikalingi vėliau kai DataLoader prašys nuotraukų: Nuotraukų sąrašą iš DB, Transformacijas kurios bus taikomos kiekvienai nuotraukai, Žemėlapį kategorija → skaičius
        self.records = image_records                                                #išsaugo nuotraukų įrašus iš DB.
        self.transform = transform                                                  #išsaugo transformacijas kurios bus taikomos nuotraukoms (pvz. resize, normalizacija).
        self.category_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}         # paverčia kategorijų pavadinimus į skaičius. Pvz. {"food": 0, "portrait": 1, "landscape": 2, "product": 3, "lifestyle": 4}. Modelis dirba su skaičiais, ne tekstais.

    def __len__(self) -> int:                                                       #grazina kiek nuotrauku yra data loade. naudoja sia fukncija kad zinotu kiek batch reikes sukurti
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:                    #kelia nuotrauką → pritaiko transformacijas (iš utils arba atsarginį) → paverčia kategoriją į skaičių → grąžina vieną porą (nuotrauka, kategorija).#grazina viena nuotrauka su jos kategorija pagal fukcija
        record = self.records[idx]                                                  #paima įrašą iš sąrašo pagal numerį.

        from utils.image_preprocessing import load_image
        from torchvision import transforms
        img = load_image(record.filepath)                                           # įkelia nuotrauką iš disko.

        if self.transform:
            img = self.transform(img)                                               #pritaiko transformacijas — resize, normalizacija ir t.t. Jei transform nenurodytas — tiesiog paverčia į tensor.
        else:
            # Jei transform nenurodytas — vis tiek konvertuojam į tensor
            img = transforms.ToTensor()(img)

        label = self.category_to_idx[record.category]                               #— paverčia kategorijos pavadinimą į skaičių, pvz. "food" → 0.
        return img, label                                                           # grąžina nuotrauką ir jos kategoriją. DataLoader šią funkciją kviečia kiekvienai nuotraukai batch'o formavimo metu.


def build_dataloaders(db, batch_size: int = 32) -> dict:
    """
    Sukuria train / val / test DataLoader objektus iš DB duomenų.
    Tikrina ar splitai nėra tušti prieš kuriant loaderius.
    """
    from database.models import TrainingImage
    from utils.image_preprocessing import get_train_transforms, get_val_transforms

    transforms_map = {
        "train": get_train_transforms(),
        "val":   get_val_transforms(),
        "test":  get_val_transforms(),
    }

    # Patikrink ar train ir val nėra tušti — kritinė klaida
    for split in ["train", "val"]:
        count = db.query(TrainingImage).filter_by(split=split).count()
        if count == 0:
            raise ValueError(
                f"❌ '{split}' splitas tuščias! "
                f"Paleisk load_data_to_db() prieš treniravimą."
            )

    # Įspėjimas jei test tuščias — nekritinė klaida
    test_count = db.query(TrainingImage).filter_by(split="test").count()
    if test_count == 0:
        print("⚠️  'test' splitas tuščias — testavimas nebus galimas.")

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


def get_model(dropout: float = 0.5) -> CNNModel:                                    #sukuria naują tuščią CNN modelį su nurodytu dropout.
    """Grąžina naują CNN modelį."""
    return CNNModel(num_classes=NUM_CLASSES, dropout=dropout)


def save_model(model: CNNModel, path: str) -> None:                                 #šsaugo modelio svorius į .pth failą. state_dict() grąžina visus modelio svorius kaip žodyną.
    """Išsaugo modelį į failą."""
    torch.save(model.state_dict(), path)
    print(f"Modelis išsaugotas: {path}")


def load_model(path: str, dropout: float = 0.5) -> CNNModel:                        #įkelia modelį iš failo. Sukuria tuščią CNN, užkrauna svorius iš failo,
    """Įkelia modelį iš failo."""
    model = CNNModel(num_classes=NUM_CLASSES, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()                                                                    #įjungia testavimo režimą — išjungia dropout
    print(f"Modelis įkeltas: {path}")
    return model