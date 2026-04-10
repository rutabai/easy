import os
from flask import Flask, g
from dotenv import load_dotenv
from database.db import init_db, SessionLocal

load_dotenv()


def create_app() -> Flask:
    """Flask aplikacijos fabrikas."""
    app = Flask(__name__)

    # ── Konfigūracija ──────────────────────────────────────────
    app.config["SECRET_KEY"]         = os.getenv("SECRET_KEY", "dev-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
    app.config["UPLOAD_FOLDER"]      = os.path.join("static", "uploads")
    app.config["OUTPUT_FOLDER"]      = os.path.join("static", "outputs")
    app.config["MODEL_FOLDER"]       = "saved_models"
    app.config["DATA_FOLDER"]        = "data"

    # ── Sukurk reikalingus aplankus ────────────────────────────
    for folder in [
        app.config["UPLOAD_FOLDER"],
        app.config["OUTPUT_FOLDER"],
        app.config["MODEL_FOLDER"],
    ]:
        os.makedirs(folder, exist_ok=True)

    # ── Duomenų bazė ───────────────────────────────────────────
    init_db()

    @app.before_request
    def open_db():
        g.db = SessionLocal()

    @app.teardown_request
    def close_db(error=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ── Blueprint registravimas ────────────────────────────────
    from routes.user import user_bp
    from routes.admin import admin_bp

    app.register_blueprint(user_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")

    return app


if __name__ == "__main__":
    app = create_app()
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode)