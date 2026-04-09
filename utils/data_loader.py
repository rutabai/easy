import os
import random
from pathlib import Path
from PIL import Image as PILImage
from sqlalchemy.orm import Session
from database.models import TrainingImage

# Kategorijos
from utils.constants import CATEGORIES, SPLIT_RATIOS

# Duomenų padalijimas

# Atkartojamas atsitiktinumas
RANDOM_SEED = 42


def get_image_info(filepath: str) -> dict:
    """
    Gauna nuotraukos metaduomenis DB įrašymui.
    Klaidos atveju atsispausdina pranešimą ir grąžina None reikšmes.
    """
    try:
        with PILImage.open(filepath) as img:
            width, height = img.size
        file_size = os.path.getsize(filepath)
        return {"width": width, "height": height, "file_size": file_size}
    except Exception as e:
        print(f"⚠️  Nepavyko nuskaityti {filepath}: {e}")
        return {"width": None, "height": None, "file_size": None}


def collect_image_files(category_path: Path) -> list[Path]:
    """
    Surenka visus nuotraukų failus iš aplanko.
    Sumaišo failus prieš grąžinant — kad split būtų atsitiktinis.
    """
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(category_path.glob(ext))

    # Sumaišom prieš splitą — vietinis generatorius kad nedarytų įtakos global state
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(image_files)
    return image_files


def assign_split(index: int, total: int) -> str:
    """Priskiria split pagal indeksą (70/15/15)"""
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    if index < train_end:
        return "train"
    elif index < val_end:
        return "val"
    else:
        return "test"


def load_data_to_db(data_dir: str, db: Session, source: str = "kaggle") -> dict:
    """
    Įkelia nuotraukų metaduomenis į DB.
    data_dir struktūra:
        data/
        ├── food/
        ├── portrait/
        ├── landscape/
        ├── product/
        └── lifestyle/
    """
    stats = {cat: {"train": 0, "val": 0, "test": 0} for cat in CATEGORIES}
    data_path = Path(data_dir)

    for category in CATEGORIES:
        category_path = data_path / category

        if not category_path.exists():
            print(f"⚠️  Aplankas nerastas: {category_path}")
            continue

        # Surink ir sumaišyk failus
        image_files = collect_image_files(category_path)
        total = len(image_files)
        print(f"📁 {category}: {total} nuotraukų")

        for i, filepath in enumerate(image_files):
            # Patikrink ar jau įkelta
            existing = db.query(TrainingImage).filter_by(
                filepath=str(filepath)
            ).first()
            if existing:
                continue

            split = assign_split(i, total)
            info = get_image_info(str(filepath))

            image = TrainingImage(
                filename=filepath.name,
                filepath=str(filepath),
                category=category,
                split=split,
                width=info["width"],
                height=info["height"],
                file_size=info["file_size"],
                source=source,
                is_augmented=False
            )
            db.add(image)
            stats[category][split] += 1

        db.commit()
        print(f"   ✅ train={stats[category]['train']} | val={stats[category]['val']} | test={stats[category]['test']}")

    return stats


def add_single_image(filepath: str, category: str, split: str,
                     db: Session, source: str = "manual") -> TrainingImage:
    """
    Prideda vieną nuotrauką į treniravimo duomenis per Flask UI.
    Tikrina ar toks failas jau egzistuoja DB.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Neteisinga kategorija: {category}. Galimos: {CATEGORIES}")

    if split not in ["train", "val", "test"]:
        raise ValueError(f"Neteisingas split: {split}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Failas nerastas: {filepath}")

    # Tikrink ar jau egzistuoja DB
    existing = db.query(TrainingImage).filter_by(filepath=filepath).first()
    if existing:
        raise ValueError(f"Šis failas jau yra DB: {filepath}")

    info = get_image_info(filepath)

    image = TrainingImage(
        filename=os.path.basename(filepath),
        filepath=filepath,
        category=category,
        split=split,
        width=info["width"],
        height=info["height"],
        file_size=info["file_size"],
        source=source,
        is_augmented=False
    )
    db.add(image)
    db.commit()
    return image


def get_stats(db: Session) -> dict:
    """Grąžina statistiką apie treniravimo duomenų bazę"""
    stats = {}
    for category in CATEGORIES:
        stats[category] = {}
        for split in ["train", "val", "test"]:
            count = db.query(TrainingImage).filter_by(
                category=category,
                split=split
            ).count()
            stats[category][split] = count
    return stats