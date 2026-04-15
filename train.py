import torch
import torch.nn as nn                                                                   #nn - neuroniniu tinklu sluoksniai, loss funkcijos
import torch.optim as optim                                                             #optimizeriai (Adam, SGD ir tt)
from torch.utils.data import DataLoader                                                 #automatiskai dalina duomenis i batch treniravimui
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score     #sclearn metrikos modelio kokybei vertinti
import os
from sqlalchemy.orm import Session

from database.db import init_db, SessionLocal
from database.models import TrainingSession
from utils.data_loader import load_data_to_db, get_stats


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"                                          #CUDA yra NVIDIA technologija kuri leidzia naudoti GPU skaicivimams vietoj CPU. GPU turi tukstancius mazu branduoliu, kurie gali vienu metu atlikti daug skaiciavimu lygegriaciai. 
print(f"🖥️  Naudojamas įrenginys: {DEVICE}")


def train_cnn(                                                                             #jei vartotojas nieko nenurodo, tai yra numatytos reiksmes modelio treniravimui. Funkcijos paskirtis: istreniruoti CNN modeli ir issaugoti rezultatus. Ivestis: DB sesija, hiperparametrai ir kelias kur issaugoti modeli. Grazina: zodyna su test metrikomis accuracy, precisio recall f1, keicia: issaugo istreniruota modeli diske kaip .pth
    db: Session,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10,
    optimizer_name: str = "adam",
    dropout: float = 0.5,
    save_path: str = "saved_models/cnn_model.pth",
) -> dict:
    from models.cnn_model import get_model, build_dataloaders, save_model

    print("\nPradedamas CNN treniravimas...")
    print(f"   lr={learning_rate} | batch={batch_size} | epochs={epochs} | optimizer={optimizer_name} | dropout={dropout}")

    loaders   = build_dataloaders(db, batch_size=batch_size)                                #paruosia duomenu pakrovejus train/val/test splitams
    model     = get_model(dropout=dropout).to(DEVICE)                                       #sukuria CNN modeli ir perkelia i GPU arba CPU
    criterion = nn.CrossEntropyLoss()                                                       #loss funkcija klasifikavimui - matuoja kiek modeliu spejimas skiriasi nuo teisingu atsakymu
    optimizer = _get_optimizer(optimizer_name, model.parameters(), learning_rate)           #sukuria optimaizeri, kuris naudos modelio svorius

    best_val_acc = 0.0
    train_losses, val_losses = [], []                                                       #paruosia kintamuosius geriausio rezultto sekimui ir loss instorijos saugojimui

    for epoch in range(epochs):
        model.train()                                                                       #ijungia treniravimo rezima
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in loaders["train"]:                                              #vidinis ciklas eina per kiekviena batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)                            #perkelia nuotraukas i GPU
            optimizer.zero_grad()                                                            #isvalo praejusio batch gradientus
            outputs = model(images)                                                          #modelis daro spejimus
            loss = criterion(outputs, labels)                                                #apskaiciuoja kiek spejimai klysta
            loss.backward()                                                                  #apskaiciuoja gradientus, kuria kryptimi keisti svorius
            optimizer.step()                                                                 #atnaujina modelio svorius pagal gradientus

            train_loss    += loss.item()                                                     #prideda batcho loss reiksme prie bendros sumos.item() pavercia pytorch tensoriu i paprasta python skaiciu
            preds          = outputs.argmax(dim=1)                                           #is modelio isvesties paima indeksa su didziausia reiksme ir tai yra spejama kategorija
            train_correct += (preds == labels).sum().item()                                  #palygina spejimus su teisingais atsakymais ir suskaiciuoja kiek yra teisingu
            train_total   += labels.size(0)                                                  #suskaiciuoja kiek nuotrauku buvo apdorota ir grazina batch dydi

        train_acc  = train_correct / train_total                                             #apskaiciuoja train tiksluma: teisingi spejimai/visu nuotrauku kiekis
        train_loss = train_loss / len(loaders["train"])                                      #apskaiciuoja vidutini loss padalinant is batch skaiciaus

        val_acc, val_loss = _evaluate_loader(model, loaders["val"], criterion)               #iškviečia _evaluate_loader funkciją kuri paskaičiuoja tikslumą ir loss ant val duomenų — tai duomenys kurių modelis nematė treniravimo metu.
        train_losses.append(train_loss)                                                      #61-62 eilutės — išsaugo kiekvienos epochos loss reikšmes į sąrašus — vėliau naudojama grafiko braižymui.
        val_losses.append(val_loss)

        print(f"   Epoch {epoch+1}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_acc > best_val_acc:                                                            #Jei šios epochos val tikslumas geresnis nei geriausias iki šiol — išsaugo modelį.
            best_val_acc = val_acc                                                            #atnaujina geriausią rezultatą.
            os.makedirs(os.path.dirname(save_path), exist_ok=True)                            #sukuria aplanką jei jo dar nėra
            save_model(model, save_path)                                                      # išsaugo modelio svorius į .pth failą.

    from models.cnn_model import load_model as _load_cnn                                      #71-72 eilutės — užkrauna geriausią išsaugotą modelį iš disko ir perkelia į GPU/CPU.
    best_model   = _load_cnn(save_path, dropout=dropout).to(DEVICE)                             
    test_metrics = _compute_test_metrics(best_model, loaders["test"])                         # apskaičiuoja galutines metrikas ant test duomenų — tai duomenys kurių modelis nematė nei treniravimo nei validavimo metu.

    session = TrainingSession(                                                                #sukuria TrainingSession įrašą su visais hyperparametrais ir rezultatais.
        model_type     = "cnn",
        learning_rate  = learning_rate,
        batch_size     = batch_size,
        epochs         = epochs,
        optimizer      = optimizer_name,
        dropout        = dropout,
        train_accuracy = round(train_acc, 4),
        val_accuracy   = round(best_val_acc, 4),
        test_accuracy  = test_metrics["accuracy"],
        train_loss     = round(train_losses[-1], 4),
        val_loss       = round(val_losses[-1], 4),
        precision      = test_metrics["precision"],
        recall         = test_metrics["recall"],
        f1_score       = test_metrics["f1_score"],
        model_path     = save_path,
    )
    db.add(session)
    db.commit()

    print(f"\n✅ CNN treniravimas baigtas! Test accuracy: {test_metrics['accuracy']:.4f}")
    return test_metrics


def train_vit(
    db: Session,
    learning_rate: float = 0.0001,
    batch_size: int = 16,
    epochs: int = 5,
    optimizer_name: str = "adam",
    freeze_backbone: bool = False,
    save_path: str = "saved_models/vit_model.pth",
) -> dict:
    from models.vit_model import get_vit_model, build_vit_dataloaders, save_vit_model

    print("\nPradedamas ViT treniravimas...")
    print(f"   lr={learning_rate} | batch={batch_size} | epochs={epochs} | freeze={freeze_backbone}")

    loaders, processor = build_vit_dataloaders(db, batch_size=batch_size)                       # paruošia duomenų pakrovėjus ViT modeliui. Grąžina du dalykus: loaders (train/val/test) ir processor (HuggingFace įrankis kuris paruošia nuotraukas ViT formatui).
    model     = get_vit_model(freeze_backbone=freeze_backbone).to(DEVICE)                       #sukuria ViT modelį. freeze_backbone=False reiškia kad visi modelio svoriai bus treniruojami, ne tik paskutinis sluoksnis.
    criterion = nn.CrossEntropyLoss()                                                           #ta pati loss funkcija kaip CNN.
    optimizer = _get_optimizer(optimizer_name, model.parameters(), learning_rate)               #sukuria optimizerį su mažesniu learning rate nei CNN (0.0001 vietoj 0.001) — ViT jautresnis didiems learning rate.

    best_val_acc = 0.0                                                                          #pradinė geriausio tikslumo reikšmė. Kiekviena epocha lyginama su šia reikšme, ir jei geresnė — atnaujinama.
    train_losses, val_losses = [], []                                                           # tušti sąrašai kuriuose bus saugomi kiekvienos epochos loss reikšmės. Vėliau naudojami grafiko braižymui.

    for epoch in range(epochs):                                                                 #Tas pats kaip CNN treniravime — išorinis ciklas per epochas, model.train() įjungia treniravimo režimą, ir iš naujo nulinami loss bei teisingų spėjimų skaitikliai kiekvienai epochai.
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in loaders["train"]:                                                 
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_acc  = train_correct / train_total
        train_loss = train_loss / len(loaders["train"])

        val_acc, val_loss = _evaluate_loader(model, loaders["val"], criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"   Epoch {epoch+1}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_vit_model(model, save_path)

    from models.vit_model import load_vit_model as _load_vit
    best_model   = _load_vit(save_path, freeze_backbone=freeze_backbone).to(DEVICE)
    test_metrics = _compute_test_metrics(best_model, loaders["test"])

    session = TrainingSession(
        model_type     = "vit",
        learning_rate  = learning_rate,
        batch_size     = batch_size,
        epochs         = epochs,
        optimizer      = optimizer_name,
        dropout        = None,
        train_accuracy = round(train_acc, 4),
        val_accuracy   = round(best_val_acc, 4),
        test_accuracy  = test_metrics["accuracy"],
        train_loss     = round(train_losses[-1], 4),
        val_loss       = round(val_losses[-1], 4),
        precision      = test_metrics["precision"],
        recall         = test_metrics["recall"],
        f1_score       = test_metrics["f1_score"],
        model_path     = save_path,
        notes          = f"freeze_backbone={freeze_backbone}",
    )
    db.add(session)
    db.commit()

    print(f"\nViT treniravimas baigtas! Test accuracy: {test_metrics['accuracy']:.4f}")
    return test_metrics


def train_knn(
    db: Session,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    save_path: str = "saved_models/knn_model.pkl",
    cnn_model_path: str = "saved_models/cnn_model.pth",
) -> dict:
    from models.cnn_model import load_model, build_dataloaders
    from models.knn_model import get_knn_model, extract_features_cnn

    print("\nPradedamas KNN treniravimas...")
    print(f"   k={n_neighbors} | metric={metric}")

    if not os.path.exists(cnn_model_path):
        raise FileNotFoundError(
            f"CNN modelis nerastas: {cnn_model_path}. "
            f"Pirmiau ištreniruok CNN!"
        )

    cnn     = load_model(cnn_model_path)
    loaders = build_dataloaders(db, batch_size=32)

    print("   Ištraukiami CNN embeddings...")
    train_features, train_labels = extract_features_cnn(cnn, loaders["train"], DEVICE)
    test_features,  test_labels  = extract_features_cnn(cnn, loaders["test"],  DEVICE)

    knn = get_knn_model(n_neighbors=n_neighbors, metric=metric)
    knn.fit(train_features, train_labels)                                                       #KNN 'mokosi' tiesiog isimena visus train embeddings su ju kategorijomis
    metrics = knn.evaluate(test_features, test_labels)                                          #kai ateina nauja nuotrauka, KNN paima jos embeddings ir randa 5 artimiausiu train embeddings. Kuri kategorija dazniausiai pasitaiko tarp tu 5 kaimynu - tas ir spejimas. CNN istraukia savybes, o KNN sprendzia pagal panasuma i kaimynus

    os.makedirs(os.path.dirname(save_path), exist_ok=True)                                      #issaugo .pkl formatu treniruota faila. 
    knn.save(save_path)

    session = TrainingSession(
        model_type    = "knn",
        learning_rate = None,
        batch_size    = None,
        epochs        = None,
        optimizer     = None,
        dropout       = None,
        test_accuracy = metrics["accuracy"],
        precision     = metrics["precision"],
        recall        = metrics["recall"],
        f1_score      = metrics["f1_score"],
        model_path    = save_path,
        notes         = f"k={n_neighbors}, metric={metric}",
    )
    db.add(session)
    db.commit()

    print(f"\nKNN treniravimas baigtas! Test accuracy: {metrics['accuracy']:.4f}")
    return metrics


def _get_optimizer(name: str, params, lr: float):                                                   #si funkcija yra kaip tarpininkas - pavercia teksta i realu PyTorch objekta
    name = name.lower()
    if name == "adam":    return optim.Adam(params, lr=lr)
    elif name == "sgd":   return optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "rmsprop": return optim.RMSprop(params, lr=lr)
    elif name == "adamw": return optim.AdamW(params, lr=lr)
    else:
        raise ValueError(f"Nežinomas optimizeris: {name}. Galimi: adam, sgd, rmsprop, adamw")


def _evaluate_loader(model, loader: DataLoader, criterion) -> tuple[float, float]:                  #apskaiciuoja modelio tiksluma ir loss ant val arba test duomenu
    model.eval()                                                                                    #isjungia treniravimo rezuma 
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():                                                                           #išjungia gradientų skaičiavimą — nereikia nes tik testuojame, ne mokomės. Tai greičiau ir naudoja mažiau atminties.
        for images, labels in loader:                                                               #249-257 — eina per visus batch'us, daro spėjimus ir skaičiuoja kiek teisingų.
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    if total == 0:                                                                                  #259-260 — apsauga jei loader tuščias. 
        return 0.0, 0.0
    return correct / total, total_loss / len(loader)                                                #261 — grąžina tikslumą ir vidutinį loss.


def _compute_test_metrics(model, loader: DataLoader) -> dict:                                               #apskaiciuoja visas galutins metrikas ant test duomenu. 
    model.eval()
    all_preds, all_labels = [], []                                                                          #265-274 — eina per visus test batch'us, daro spėjimus ir kaupia juos į sąrašus. .cpu().numpy() paverčia PyTorch tensorių į numpy masyvą kurį sklearn supranta.

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    if len(all_labels) == 0:                                                                                #276-278 — apsauga jei test splitas tuščias.
        print("Test splitas tuščias — metrikos negalimos.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    return {                                                                                                 #apskaiciuoja metrikas
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
    }


if __name__ == "__main__":                                                                                  #šis blokas vykdomas tik kai paleidžiamas train.py
    init_db()
    db = SessionLocal()

    print("\nĮkeliami duomenys į DB...")
    load_data_to_db("data", db)

    stats = get_stats(db)
    print("\nDuomenų statistika:")
    for cat, splits in stats.items():
        print(f"   {cat}: train={splits['train']} | val={splits['val']} | test={splits['test']}")

    train_cnn(db, learning_rate=0.001, batch_size=32, epochs=10)
    train_knn(db)
    # train_vit(db, learning_rate=0.0001, epochs=5)

    db.close()