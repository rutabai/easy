from flask import Blueprint, render_template, request, jsonify, g, current_app
from database.models import TrainingSession
from utils.constants import (
    CATEGORIES, MODEL_TYPES,
    VALID_OPTIMIZERS, VALID_KNN_METRICS,
)
from utils.data_loader import get_stats

admin_bp = Blueprint("admin", __name__)


# ── Admin pagrindinis ──────────────────────────────────────────
@admin_bp.route("/")
def index():
    sessions = (
        g.db.query(TrainingSession)
        .order_by(TrainingSession.created_at.desc())
        .limit(10)
        .all()
    )
    return render_template("admin/index.html", sessions=sessions)


# ── Treniravimo puslapis ───────────────────────────────────────
@admin_bp.route("/train")
def train():
    return render_template(
        "admin/train.html",
        model_types = MODEL_TYPES,
        optimizers  = VALID_OPTIMIZERS,
        knn_metrics = VALID_KNN_METRICS,
    )


# ── Paleisti treniravimą ───────────────────────────────────────
@admin_bp.route("/train/run", methods=["POST"])
def train_run():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Nėra duomenų"}), 400

    model_type = data.get("model_type", "cnn")

    if model_type not in MODEL_TYPES:
        return jsonify({
            "error": f"Nežinomas modelio tipas: '{model_type}'. Galimi: {MODEL_TYPES}"
        }), 400

    # Dispatch map — švaresnis nei if/elif
    handlers = {
        "cnn": _train_cnn,
        "vit": _train_vit,
        "knn": _train_knn,
    }

    try:
        metrics = handlers[model_type](data)
        return jsonify({"success": True, "metrics": metrics})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Statistikos puslapis ───────────────────────────────────────
@admin_bp.route("/stats")
def stats():
    data_stats = get_stats(g.db)

    sessions = (
        g.db.query(TrainingSession)
        .order_by(TrainingSession.created_at.desc())
        .all()
    )

    # Geriausių modelių skaičiavimas Python pusėje
    best_models = {}
    for s in sessions:
        if s.test_accuracy is not None:
            if (s.model_type not in best_models or
                    s.test_accuracy > best_models[s.model_type].test_accuracy):
                best_models[s.model_type] = s

    return render_template(
        "admin/stats.html",
        data_stats  = data_stats,
        sessions    = sessions,
        best_models = best_models,
    )


# ── Duomenų įkėlimas į DB ─────────────────────────────────────
@admin_bp.route("/load-data", methods=["POST"])
def load_data():
    from utils.data_loader import load_data_to_db
    data_folder = current_app.config.get("DATA_FOLDER", "data")
    try:
        stats = load_data_to_db(data_folder, g.db)
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# PAGALBINĖS FUNKCIJOS
# ══════════════════════════════════════════════════════════════

def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return default


def _parse_float(
    value,
    default: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """Išgauna float reikšmę. Jei value None ar tuščias — naudoja default."""
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Negalima konvertuoti į skaičių: '{value}'")
    if not (min_val <= result <= max_val):
        raise ValueError(
            f"Reikšmė {result} turi būti tarp {min_val} ir {max_val}"
        )
    return result


def _parse_int(
    value,
    default: int,
    min_val: int = 1,
) -> int:
    """Išgauna int reikšmę. Jei value None ar tuščias — naudoja default."""
    if value is None or value == "":
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Negalima konvertuoti į sveikąjį skaičių: '{value}'")
    if result < min_val:
        raise ValueError(
            f"Reikšmė {result} turi būti >= {min_val}"
        )
    return result


def _train_cnn(data: dict) -> dict:
    from train import train_cnn

    optimizer = data.get("optimizer", "adam")
    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Nežinomas optimizeris: '{optimizer}'. Galimi: {VALID_OPTIMIZERS}"
        )

    return train_cnn(
        db             = g.db,
        learning_rate  = _parse_float(data.get("learning_rate"), 0.001,  1e-6, 1.0),
        batch_size     = _parse_int(data.get("batch_size"),      32,     1),
        epochs         = _parse_int(data.get("epochs"),          10,     1),
        optimizer_name = optimizer,
        dropout        = _parse_float(data.get("dropout"),       0.5,    0.0, 0.9),
    )


def _train_vit(data: dict) -> dict:
    from train import train_vit

    optimizer = data.get("optimizer", "adam")
    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Nežinomas optimizeris: '{optimizer}'. Galimi: {VALID_OPTIMIZERS}"
        )

    return train_vit(
        db              = g.db,
        learning_rate   = _parse_float(data.get("learning_rate"), 0.0001, 1e-6, 1.0),
        batch_size      = _parse_int(data.get("batch_size"),      16,     1),
        epochs          = _parse_int(data.get("epochs"),          5,      1),
        optimizer_name  = optimizer,
        freeze_backbone = _parse_bool(data.get("freeze_backbone"), False),
    )


def _train_knn(data: dict) -> dict:
    from train import train_knn

    metric = data.get("metric", "euclidean")
    if metric not in VALID_KNN_METRICS:
        raise ValueError(
            f"Nežinoma metrika: '{metric}'. Galimos: {VALID_KNN_METRICS}"
        )

    return train_knn(
        db          = g.db,
        n_neighbors = _parse_int(data.get("n_neighbors"), 5, 1),
        metric      = metric,
    )