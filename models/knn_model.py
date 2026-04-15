import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils.constants import CATEGORIES, IDX_TO_CATEGORY


class KNNModel:
    """
    KNN modelis nuotraukų kategorizavimui.
    Naudoja CNN ištrauktus feature vektorius (embeddings) vietoj pikselių.
    Tai leidžia KNN veikti daug tiksliau nei su raw pikseliais.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):            #inicijuoja KNN modeli
        self.n_neighbors = n_neighbors                                              #issaugo parametrus
        self.metric = metric
        self.model = KNeighborsClassifier(                                          #sukuria sklearn KNN objektą. n_jobs=-1 reiškia naudoti visus CPU branduolius — greitesnis.
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()                                              # normalizavimo įrankis, paruošiamas bet dar nenaudojamas.
        self.is_fitted = False                                                      # normalizavimo įrankis, paruošiamas bet dar nenaudojamas.

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Treniruoja KNN modelį su CNN embeddings."""
        if len(features) == 0:                                                      #apsauga jei embedding sarasas tuscias
            raise ValueError(" Treniravimo duomenys tušti!")

        if len(features) != len(labels):                                            # apsauga jei embeddings ir kategorijų skaičius nesutampa — pvz. 1000 nuotraukų bet tik 999 kategorijų.
            raise ValueError(
                f" features ({len(features)}) ir labels ({len(labels)}) ilgiai nesutampa!"
            )

        features_scaled = self.scaler.fit_transform(features)                       #fit_transform normalizuoja embeddings — apskaičiuoja vidurkį ir standartinį nuokrypį, tada sulygins visus skaičius.
        self.model.fit(features_scaled, labels)                                     #KNN "mokosi" — tiesiog įsimena visus embeddings su kategorijomis.
        self.is_fitted = True                                                       #pažymi kad modelis istreniruotas.
        print(f"KNN ištreniruotas: {len(features)} nuotraukų, k={self.n_neighbors}")

    def predict(self, features: np.ndarray) -> np.ndarray:                          #Spėja kategorijas pagal embeddings. grąžina tik vieną kategoriją — pvz. "food".
        """Spėja kategorijas pagal features."""
        if not self.is_fitted:                                                      #tikrina ar modelis istreniruotas, jei ne — išmeta klaidą.
            raise RuntimeError(" Modelis dar neištreniruotas! Paleisk fit() pirmiau.")
        features_scaled = self.scaler.transform(features)                           #normalizuoja embeddings naudojant tą patį scaler kaip treniravimo metu — svarbu kad skaičiai būtų tame pačiame diapazone.
        return self.model.predict(features_scaled)                                  #KNN randa k artimiausius kaimynus ir grąžina dažniausiai pasitaikančią kategoriją.

    def predict_proba(self, features: np.ndarray) -> np.ndarray:                    #grąžina tikimybes visoms kategorijoms — pvz. [0.7, 0.1, 0.05, 0.1, 0.05]. Tai reiškia 70% food, 10% portrait ir t.t.
        """Grąžina tikimybes kiekvienai kategorijai."""
        if not self.is_fitted:                                                      #tikrina ar modelis istreniruotas, jei ne — išmeta klaidą.
            raise RuntimeError(" Modelis dar neištreniruotas!")
        features_scaled = self.scaler.transform(features)                           #ormalizuoja embeddings. Čia transform o ne fit_transform — nes scaler jau išmokytas treniravimo metu, dabar tik pritaikomas.
        return self.model.predict_proba(features_scaled)                            #grąžina tikimybių masyvą kiekvienai kategorijai.

    def predict_single(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """
        Spėja vieną nuotrauką.
        Grąžina (kategorija, confidence).
        """
        if feature_vector.ndim == 1:                                                # jei feature_vector yra 1D masyvas, paverčia į 2D — predict_proba reikalauja 2D įvesties.
            feature_vector = feature_vector.reshape(1, -1)

        proba = self.predict_proba(feature_vector)[0]                               #gauna tikimybes visoms 5 kategorijoms. [0] paima pirmą eilutę nes grąžinamas 2D masyvas.
        idx = int(np.argmax(proba))                                                 # argmax randa indeksą su didžiausia tikimybe.
        confidence = float(proba[idx])                                              #paima tą tikimybę kaip confidence.
        category = IDX_TO_CATEGORY[idx]                                             #paverčia indeksą į kategorijos pavadinimą, pvz. 0 → "food".
        return category, confidence                                                 #grąžina kategoriją ir confidence.

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict:           #np.ndarray — tai numpy masyvas, paprastai skaičių matrica. Čia naudojamas kaip tipo anotacija — nurodo kad funkcija tikisi numpy masyvo.
        """Apskaičiuoja metrikas testavimo duomenims."""
        if not self.is_fitted:                                                      #tikrina ar modelis istreniruotas.
            raise RuntimeError("Modelis dar neištreniruotas!")

        predictions = self.predict(features)                                        #daro spėjimus visoms test nuotraukoms.

        metrics = {
            "accuracy":  round(accuracy_score(labels, predictions), 4),
            "precision": round(precision_score(labels, predictions, average="weighted", zero_division=0), 4),
            "recall":    round(recall_score(labels, predictions, average="weighted", zero_division=0), 4),
            "f1_score":  round(f1_score(labels, predictions, average="weighted", zero_division=0), 4),
            # Aiškiai nurodyti labels ir target_names — kad veiktų net jei kuri klasė nepasirodė
            "report": classification_report(
                labels,
                predictions,
                labels=list(range(len(CATEGORIES))),
                target_names=CATEGORIES,
                zero_division=0
            ),
        }
        return metrics

    def save(self, path: str) -> None:                                              
        """Išsaugo KNN modelį ir scaler į failą."""
        if not self.is_fitted:
            raise RuntimeError("Modelis dar neištreniruotas!")
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)                 # biblioteka kuri išsaugo Python objektus į failą ir vėliau juos įkelia atgal. Panašu į pickle bet geriau veikia su numpy masyvais ir sklearn modeliais — greičiau ir efektyviau suspaudžia didelius duomenis.

        print(f"KNN išsaugotas: {path}")

    def load(self, path: str) -> None:
        """Įkelia KNN modelį ir scaler iš failo."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True
        print(f"KNN įkeltas: {path}")


def extract_features_cnn(model, dataloader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """
    Ištraukia CNN embeddings iš nuotraukų.
    Naudoja CNN.features bloką kaip feature extractor (be classifier dalies).

    SVARBU: ši funkcija skirta būtent CNNModel architektūrai,
    kuri turi .features atributą su išėjimu [batch, channels, h, w].
    """
    import torch

    model.eval()
    model = model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():                                                           #istraukia gradientu skaiciavima, nes cia nereikalingas
        for images, labels in dataloader:
            images = images.to(device)

            # Gauk tik features (be classifier dalies)
            features = model.features(images)

            # Global Average Pooling — [batch, 256, 14, 14] → [batch, 256]
            features = features.mean(dim=[2, 3])

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())  # .cpu() saugiau

    # Tikrink ar dataloader nebuvo tuščias
    if len(all_features) == 0:
        raise ValueError("❌ Dataloader tuščias — nėra iš ko ištraukti features!")

    return np.concatenate(all_features), np.concatenate(all_labels)


def get_knn_model(n_neighbors: int = 5, metric: str = "euclidean") -> KNNModel:
    """Grąžina naują KNN modelį."""
    return KNNModel(n_neighbors=n_neighbors, metric=metric)