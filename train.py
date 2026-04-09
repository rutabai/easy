import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from sqlalchemy.orm import Session

from database.db import init_db, SessionLocal
from database.models import TrainingSession
from utils.data_loader import load_data_to_db, get_stats


# ── Įrenginys ─────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Naudojamas įrenginys: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# CNN TRENIRAVIMAS
# ══════════════════════════════════════════════════════════════

def train_cnn(
    db: Session,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10,
    optimizer_name: str = "adam",
    dropout: float = 0.5,
    save_path: str = "saved_models/cnn_model.pth",
) -> dict:
    """
    Treniruoja CNN modelį ir išsaugo rezultatus į DB.
    Grąžina metrikų žodyną.
    """
    from models.cnn_model import get_model, build_dataloaders, save_model

    print("\n🚀 Pradedamas CNN treniravimas...")
    print(f"   lr={learning_rate} | batch={batch_size} | epochs={epochs} | optimizer={optimizer_name} | dropout={dropout}")

    # Sukurk dataloaderius
    loaders = build_dataloaders(db, batch_size=batch_size)

    # Sukurk modelį
    model = get_model(dropout=dropout).to(DEVICE)

    # Loss funkcija
    criterion = nn.CrossEntropyLoss()

    # Optimizeris
    optimizer = _get_optimizer(optimizer_name, model.parameters(), learning_rate)

    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in loaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc  = train_correct / train_total
        train_loss = train_loss / len(loaders["train"])

        # ── Validation ─────────────────────────────────────────
        val_acc, val_loss = _evaluate_loader(model, loaders["val"], criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"   Epoch {epoch+1}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Išsaugok geriausią modelį
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_model(model, save_path)

    # ── Test metrikos — naudok geriausią checkpointą ──────────
    from models.cnn_model import load_model as _load_cnn
    best_model = _load_cnn(save_path, dropout=dropout).to(DEVICE)
    test_metrics = _compute_test_metrics(best_model, loaders["test"])

    # ── Įrašyk į DB ───────────────────────────────────────────
    session = TrainingSession(
        model_type="cnn",
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer_name,
        dropout=dropout,
        train_accuracy=round(train_acc, 4),
        val_accuracy=round(best_val_acc, 4),
        test_accuracy=test_metrics["accuracy"],
        train_loss=round(train_losses[-1], 4),
        val_loss=round(val_losses[-1], 4),
        precision=test_metrics["precision"],
        recall=test_metrics["recall"],
        f1_score=test_metrics["f1"],
        model_path=save_path,
    )
    db.add(session)
    db.commit()

    print(f"\n✅ CNN treniravimas baigtas! Test accuracy: {test_metrics['accuracy']:.4f}")
    return test_metrics


# ══════════════════════════════════════════════════════════════
# VIT TRENIRAVIMAS
# ══════════════════════════════════════════════════════════════

def train_vit(
    db: Session,
    learning_rate: float = 0.0001,
    batch_size: int = 16,
    epochs: int = 5,
    optimizer_name: str = "adam",
    freeze_backbone: bool = False,
    save_path: str = "saved_models/vit_model.pth",
) -> dict:
    """
    Treniruoja ViT modelį ir išsaugo rezultatus į DB.
    Grąžina metrikų žodyną.
    """
    from models.vit_model import get_vit_model, build_vit_dataloaders, save_vit_model

    print("\n🚀 Pradedamas ViT treniravimas...")
    print(f"   lr={learning_rate} | batch={batch_size} | epochs={epochs} | freeze={freeze_backbone}")

    loaders, processor = build_vit_dataloaders(db, batch_size=batch_size)
    model = get_vit_model(freeze_backbone=freeze_backbone).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = _get_optimizer(optimizer_name, model.parameters(), learning_rate)

    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in loaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc  = train_correct / train_total
        train_loss = train_loss / len(loaders["train"])

        # ── Validation ─────────────────────────────────────────
        val_acc, val_loss = _evaluate_loader(model, loaders["val"], criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"   Epoch {epoch+1}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_vit_model(model, save_path)

    # ── Test metrikos — naudok geriausią checkpointą ──────────
    from models.vit_model import load_vit_model as _load_vit
    best_model = _load_vit(save_path, freeze_backbone=freeze_backbone).to(DEVICE)
    test_metrics = _compute_test_metrics(best_model, loaders["test"])

    # ── Įrašyk į DB ───────────────────────────────────────────
    session = TrainingSession(
        model_type="vit",
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer_name,
        dropout=None,
        train_accuracy=round(train_acc, 4),
        val_accuracy=round(best_val_acc, 4),
        test_accuracy=test_metrics["accuracy"],
        train_loss=round(train_losses[-1], 4),
        val_loss=round(val_losses[-1], 4),
        precision=test_metrics["precision"],
        recall=test_metrics["recall"],
        f1_score=test_metrics["f1"],
        model_path=save_path,
        notes=f"freeze_backbone={freeze_backbone}",
    )
    db.add(session)
    db.commit()

    print(f"\n✅ ViT treniravimas baigtas! Test accuracy: {test_metrics['accuracy']:.4f}")
    return test_metrics


# ══════════════════════════════════════════════════════════════
# KNN TRENIRAVIMAS
# ══════════════════════════════════════════════════════════════

def train_knn(
    db: Session,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    save_path: str = "saved_models/knn_model.pkl",
    cnn_model_path: str = "saved_models/cnn_model.pth",
) -> dict:
    """
    Treniruoja KNN modelį naudodamas CNN embeddings.
    CNN modelis turi būti jau ištreniruotas.
    Grąžina metrikų žodyną.
    """
    from models.cnn_model import load_model, build_dataloaders
    from models.knn_model import get_knn_model, extract_features_cnn

    print("\n🚀 Pradedamas KNN treniravimas...")
    print(f"   k={n_neighbors} | metric={metric}")

    if not os.path.exists(cnn_model_path):
        raise FileNotFoundError(
            f"❌ CNN modelis nerastas: {cnn_model_path}. "
            f"Pirmiau ištreniruok CNN!"
        )

    # Įkrauk CNN kaip feature extractor
    cnn = load_model(cnn_model_path)
    loaders = build_dataloaders(db, batch_size=32)

    # Ištrauk CNN embeddings
    print("   Ištraukiami CNN embeddings...")
    train_features, train_labels = extract_features_cnn(cnn, loaders["train"], DEVICE)
    test_features, test_labels   = extract_features_cnn(cnn, loaders["test"], DEVICE)

    # Treniruok KNN
    knn = get_knn_model(n_neighbors=n_neighbors, metric=metric)
    knn.fit(train_features, train_labels)

    # Įvertink
    metrics = knn.evaluate(test_features, test_labels)

    # Išsaugok
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    knn.save(save_path)

    # Įrašyk į DB
    session = TrainingSession(
        model_type="knn",
        learning_rate=None,
        batch_size=None,
        epochs=None,
        optimizer=None,
        dropout=None,
        test_accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics.get("f1_score", metrics.get("f1", 0.0)),
        model_path=save_path,
        notes=f"k={n_neighbors}, metric={metric}",
    )
    db.add(session)
    db.commit()

    print(f"\n✅ KNN treniravimas baigtas! Test accuracy: {metrics['accuracy']:.4f}")
    return metrics


# ══════════════════════════════════════════════════════════════
# PAGALBINĖS FUNKCIJOS
# ══════════════════════════════════════════════════════════════

def _get_optimizer(name: str, params, lr: float):
    """Grąžina optimizerį pagal pavadinimą."""
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr)
    else:
        raise ValueError(f"❌ Nežinomas optimizeris: {name}. Galimi: adam, sgd, rmsprop, adamw")


def _evaluate_loader(model, loader: DataLoader, criterion) -> tuple[float, float]:
    """Apskaičiuoja accuracy ir loss validation loaderiui."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

    return correct / total, total_loss / len(loader)


def _compute_test_metrics(model, loader: DataLoader) -> dict:
    """Apskaičiuoja visas metrikas test loaderiui."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    if len(all_labels) == 0:
        print("⚠️  Test splitas tuščias — metrikos negalimos.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
    }


# ══════════════════════════════════════════════════════════════
# PALEIDIMAS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Inicializuok DB
    init_db()
    db = SessionLocal()

    # Įkrauk duomenis į DB
    print("\n📂 Įkeliami duomenys į DB...")
    load_data_to_db("data", db)

    # Statistika
    stats = get_stats(db)
    print("\n📊 Duomenų statistika:")
    for cat, splits in stats.items():
        print(f"   {cat}: train={splits['train']} | val={splits['val']} | test={splits['test']}")

    # Treniruok CNN
    train_cnn(db, learning_rate=0.001, batch_size=32, epochs=10)

    # Treniruok KNN (naudoja CNN embeddings)
    train_knn(db)

    # Treniruok ViT (bonus)
    # train_vit(db, learning_rate=0.0001, epochs=5)

    db.close()