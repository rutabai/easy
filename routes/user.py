import os
import uuid
from flask import (
    Blueprint, render_template, request,
    redirect, url_for, jsonify, send_file, g, current_app
)
from database.models import Upload, Post, PostUpload
from utils.post_helpers import create_post_with_uploads, validate_upload_count
from utils.caption_generator import generate_caption, refine_caption
from utils.text_placement import process_image_with_text
from utils.constants import VALID_GOALS, VALID_CTA_TYPES, AVAILABLE_FILTERS, IDX_TO_CATEGORY

user_bp = Blueprint("user", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

MIME_TYPES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file, upload_folder: str) -> tuple[str, str, str]:
    """Išsaugo įkeltą failą. Grąžina (original_name, filename, filepath)."""
    original_name = file.filename
    ext           = original_name.rsplit(".", 1)[1].lower()
    filename      = f"{uuid.uuid4().hex}.{ext}"
    filepath      = os.path.join(upload_folder, filename)
    file.save(filepath)
    return original_name, filename, filepath


def _build_caption_from_parts(hook: str, story: str, cta: str) -> str:
    """Sudaro caption iš atskirų dalių."""
    parts = [p for p in [hook, story, f"→ {cta}" if cta else ""] if p]
    return "\n\n".join(parts)


# ── Pagrindinis puslapis ───────────────────────────────────────
@user_bp.route("/")
def index():
    return render_template(
        "index.html",
        goals     = VALID_GOALS,
        cta_types = VALID_CTA_TYPES,
        filters   = AVAILABLE_FILTERS,
    )


# ── Nuotraukų įkėlimas ────────────────────────────────────────
@user_bp.route("/upload", methods=["POST"])
def upload():
    files       = request.files.getlist("images")
    post_type   = request.form.get("post_type", "story")
    topic       = request.form.get("topic")
    goal        = request.form.get("goal")
    cta_type    = request.form.get("cta_type")
    notes       = request.form.get("additional_notes")
    filter_name = request.form.get("filter_name", "original")

    if filter_name not in AVAILABLE_FILTERS:
        return jsonify({"error": f"Nežinomas filtras: {filter_name}"}), 400

    valid_files = [f for f in files if f and _allowed_file(f.filename)]
    if not valid_files:
        return jsonify({"error": "Nepridėta tinkamų nuotraukų"}), 400

    count     = len(valid_files)
    ok, error = validate_upload_count(post_type, count)
    if not ok:
        return jsonify({"error": error}), 400

    upload_folder   = current_app.config["UPLOAD_FOLDER"]
    upload_ids      = []
    saved_filepaths = []

    try:
        for file in valid_files:
            original_name, filename, filepath = _save_upload(file, upload_folder)
            saved_filepaths.append(filepath)

            ext       = filepath.rsplit(".", 1)[1].lower()
            mime_type = MIME_TYPES.get(ext, "image/jpeg")

            from PIL import Image as PILImage
            try:
                with PILImage.open(filepath) as img:
                    width, height = img.size
                file_size = os.path.getsize(filepath)
            except Exception:
                width = height = file_size = None

            upload_obj = Upload(
                filename      = filename,
                original_name = original_name,
                filepath      = filepath,
                mime_type     = mime_type,
                file_size     = file_size,
                width         = width,
                height        = height,
            )
            g.db.add(upload_obj)
            g.db.flush()
            upload_ids.append(upload_obj.id)

        post = create_post_with_uploads(
            db               = g.db,
            upload_ids       = upload_ids,
            post_type        = post_type,
            topic            = topic,
            goal             = goal,
            cta_type         = cta_type,
            additional_notes = notes,
        )

        post.filter_name = filter_name
        g.db.commit()

    except Exception as e:
        g.db.rollback()
        for fp in saved_filepaths:
            if os.path.exists(fp):
                os.remove(fp)
        return jsonify({"error": str(e)}), 500

    return redirect(url_for("user.preview", post_id=post.id))


# ── Preview puslapis ───────────────────────────────────────────
@user_bp.route("/preview/<int:post_id>")
def preview(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post:
        return "Postas nerastas", 404

    post_uploads = (
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .all()
    )
    uploads = [pu.upload for pu in post_uploads]

    return render_template(
        "preview.html",
        post         = post,
        uploads      = uploads,
        post_uploads = post_uploads,
        filters      = AVAILABLE_FILTERS,
    )


# ── Nuotraukos klasifikavimas ir teksto generavimas ────────────
@user_bp.route("/generate/<int:post_id>", methods=["POST"])
def generate(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    post_uploads = (
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .all()
    )
    slide_count  = len(post_uploads)
    first_upload = post_uploads[0].upload if post_uploads else None

    # 1. Klasifikuok pirmą nuotrauką su ViT
    if first_upload:
        try:
            from models.vit_model import load_vit_model, predict_single_vit
            from transformers import ViTImageProcessor

            model_path = os.path.join(
                current_app.config["MODEL_FOLDER"], "vit_model.pth"
            )
            if os.path.exists(model_path):
                processor = ViTImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224"
                )
                model    = load_vit_model(model_path, freeze_backbone=False)
                category, confidence = predict_single_vit(
                    model     = model,
                    processor = processor,
                    filepath  = first_upload.filepath,
                    device    = "cpu",
                )
                post.predicted_category = category
                post.confidence         = confidence
                post.model_used         = "vit"
        except Exception as e:
            print(f"⚠️  Klasifikavimas nepavyko: {e}")

    category    = post.predicted_category or "lifestyle"
    post.status = "processing"
    g.db.commit()

    # 2. Generuok tekstą
    try:
        result = generate_caption(
            category         = category,
            post_type        = post.post_type,
            topic            = post.topic,
            goal             = post.goal,
            cta_type         = post.cta_type,
            additional_notes = post.additional_notes,
            slide_count      = slide_count,
        )

        post.hook    = result.get("hook", "")
        post.story   = result.get("story", "")
        post.cta     = result.get("cta", "")
        post.caption = result.get("caption", "")
        post.status  = "completed"

        # 3. Užpildyk slide_text
        for pu in post_uploads:
            if pu.slide_type in ("single", "hook"):
                pu.slide_text = post.hook
            elif pu.slide_type == "story":
                pu.slide_text = post.story
            elif pu.slide_type == "cta":
                pu.slide_text = post.cta

        g.db.commit()
        return jsonify(result)

    except Exception as e:
        post.status = "failed"
        g.db.commit()
        return jsonify({"error": str(e)}), 500


# ── Teksto koregavimas ─────────────────────────────────────────
@user_bp.route("/adjust/<int:post_id>", methods=["POST"])
def adjust(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Nėra duomenų"}), 400

    post.hook  = data.get("hook",  post.hook)
    post.story = data.get("story", post.story)
    post.cta   = data.get("cta",   post.cta)

    post.caption = _build_caption_from_parts(
        post.hook  or "",
        post.story or "",
        post.cta   or "",
    )

    g.db.commit()
    return jsonify({"success": True, "caption": post.caption})


# ── AI teksto tobulinimas ──────────────────────────────────────
@user_bp.route("/refine/<int:post_id>", methods=["POST"])
def refine(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Nėra duomenų"}), 400

    user_edit = data.get("user_edit", "")
    original  = _build_caption_from_parts(
        post.hook  or "",
        post.story or "",
        post.cta   or "",
    )

    try:
        result = refine_caption(
            original_text = original,
            user_edit     = user_edit,
            category      = post.predicted_category or "lifestyle",
            post_type     = post.post_type,
        )

        post.hook    = result.get("hook",    post.hook)
        post.story   = result.get("story",   post.story)
        post.cta     = result.get("cta",     post.cta)
        post.caption = result.get("caption") or _build_caption_from_parts(
            post.hook or "", post.story or "", post.cta or ""
        )
        g.db.commit()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Nuotraukos renderinimas su tekstu ─────────────────────────
@user_bp.route("/render/<int:post_id>", methods=["POST"])
def render_image(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    data        = request.get_json() or {}
    offset_x    = int(data.get("offset_x", 0))
    offset_y    = int(data.get("offset_y", 0))
    zone        = data.get("zone")
    filter_name = data.get("filter_name", post.filter_name or "original")

    if filter_name not in AVAILABLE_FILTERS:
        return jsonify({"error": f"Nežinomas filtras: {filter_name}"}), 400

    post.filter_name = filter_name

    first_pu = (
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .first()
    )
    if not first_pu:
        return jsonify({"error": "Nuotrauka nerasta"}), 404

    output_folder = current_app.config["OUTPUT_FOLDER"]
    output_path   = os.path.join(output_folder, f"post_{post_id}.jpg")

    try:
        process_image_with_text(
            image_path  = first_pu.upload.filepath,
            hook        = post.hook  or "",
            body        = post.story or "",
            cta         = post.cta   or "",
            output_path = output_path,
            category    = post.predicted_category,
            zone        = zone,
            offset_x    = offset_x,
            offset_y    = offset_y,
            filter_name = filter_name,
        )

        post.output_path = output_path
        g.db.commit()

        relative = output_path.replace("\\", "/")
        relative = relative.replace("static/", "", 1)
        return jsonify({
            "success":   True,
            "image_url": url_for("static", filename=relative),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Parsisiuntimas ─────────────────────────────────────────────
@user_bp.route("/download/<int:post_id>")
def download(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post or not post.output_path:
        return "Nuotrauka nerasta", 404

    if not os.path.exists(post.output_path):
        return "Failas nerastas diske", 404

    return send_file(
        post.output_path,
        as_attachment = True,
        download_name = f"instagram_post_{post_id}.jpg",
    )