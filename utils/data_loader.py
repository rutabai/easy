import os                                                                               
import random                                                                           #nuotrauku sumaisymui
from pathlib import Path                                                                #modernesne os.path verija
from PIL import Image as PILImage                                                       #Pillow biblioteka, kuri atidaro nuotraukas ir istraukia duoenis(widthxheight)
from sqlalchemy.orm import Session
from database.models import TrainingImage
from utils.image_preprocessing import extract_hog_features                              #HOG pozymiu istraukimo funkcija


#eina per data aplankus, kiekvienai nuotraukai istraukia meraduomenis ir HOG pozymius, priskiria train/val/test frupes ir iraso i trainimage lentele DB
# Kategorijos
from utils.constants import CATEGORIES, SPLIT_RATIOS

# Atkartojamas atsitiktinumas
RANDOM_SEED = 42                                                                         #uztikrina, kad atsitiktinis sumaisymas visada bus vienodas. Visada tos pacios nuotraukos tose paciose grupese


def get_image_info(filepath: str) -> dict:
    """
    Gauna nuotraukos metaduomenis DB įrašymui.
    Klaidos atveju atsispausdina pranešimą ir grąžina None reikšmes.
    """
    try:
        with PILImage.open(filepath) as img:
            width, height = img.size
        file_size = os.path.getsize(filepath)                                            #paima nuotraukos dydi baitais
        return {"width": width, "height": height, "file_size": file_size}
    except Exception as e:
        print(f"⚠️  Nepavyko nuskaityti {filepath}: {e}")
        return {"width": None, "height": None, "file_size": None}


def collect_image_files(category_path: Path) -> list[Path]:                               #eina per failus ir sudeda nuotraukas i sarasa
    """
    Surenka visus nuotraukų failus iš aplanko.
    Sumaišo failus prieš grąžinant — kad split būtų atsitiktinis.
    """
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(category_path.glob(ext))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(image_files)
    return image_files                                                                    #grazina sumaisytu nuotrauku sarasa


def assign_split(index: int, total: int) -> str:                                          #suskirsto nuotraukas i treain, val ir test pagal indeksa
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
    stats = {cat: {"train": 0, "val": 0, "test": 0} for cat in CATEGORIES}                  #sukuria statistikos zodyna ir skaiciuoja kiek ikelta
    data_path = Path(data_dir)                                                              #pavercia teksta i path objekta, kad butu galima naudoti

    for category in CATEGORIES:                                                             #eina per kiekviena kategorija
        category_path = data_path / category

        if not category_path.exists():
            print(f"⚠️  Aplankas nerastas: {category_path}")
            continue

        image_files = collect_image_files(category_path)                                    #surenka ir sumaiso failus
        total = len(image_files)
        print(f"📁 {category}: {total} nuotraukų")

        for i, filepath in enumerate(image_files):                                          #eina per kiekviena nuotrauka ir tikrina ar yra dublikatu jei taip, tada praleidzia
            existing = db.query(TrainingImage).filter_by(
                filepath=str(filepath)
            ).first()
            if existing:
                continue

            split = assign_split(i, total)                                                  #priskiria train val test
            info = get_image_info(str(filepath))                                            #istraukia matmenis
            hog_features = extract_hog_features(str(filepath))                              #hog istraukia

            image = TrainingImage(                                                          #sukuria trainingimage eilute ir prideda i db 
                filename=filepath.name,     
                filepath=str(filepath),
                category=category,
                split=split,
                width=info["width"],
                height=info["height"],
                file_size=info["file_size"],
                source=source,
                is_augmented=False,
                hog_features=hog_features,
            )
            db.add(image)
            stats[category][split] += 1

        db.commit()
        print(
            f"   ✅ train={stats[category]['train']} | "
            f"val={stats[category]['val']} | "
            f"test={stats[category]['test']}"
        )

    return stats


def add_single_image(
    filepath: str,
    category: str,
    split: str,
    db: Session,
    source: str = "manual"
) -> TrainingImage:
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

    existing = db.query(TrainingImage).filter_by(filepath=filepath).first()
    if existing:
        raise ValueError(f"Šis failas jau yra DB: {filepath}")

    info = get_image_info(filepath)
    hog_features = extract_hog_features(str(filepath))

    image = TrainingImage(
        filename=os.path.basename(filepath),
        filepath=filepath,
        category=category,
        split=split,
        width=info["width"],
        height=info["height"],
        file_size=info["file_size"],
        source=source,
        is_augmented=False,
        hog_features=hog_features,
    )
    db.add(image)
    db.commit()
    return image        


def get_stats(db: Session) -> dict:                                                              #apskaiciuoja kiek nuotrauku yra DB kiekvienoje kategorijoje ir grueje          
    """Grąžina statistiką apie treniravimo duomenų bazę"""
    stats = {}                                                                                   #tuscias zoynas kur kaupsis statistika
    for category in CATEGORIES:
        stats[category] = {}
        for split in ["train", "val", "test"]:
            count = db.query(TrainingImage).filter_by(
                category=category,
                split=split
            ).count()
            stats[category][split] = count
    return stats