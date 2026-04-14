import os #naudojama darbui su operacine sistema(paimti aplinkos kintamuosius, kurti folderius, jngti failu kelius)
from flask import Flask, g  #FLASK -biblioteka leidzianti Python kalbetis su narsykle. 
from dotenv import load_dotenv
from database.db import init_db, SessionLocal

#g yra laikinas konteineri vienai uzklauai. g reiskia global
#imituojam, jog Flask turi daug uzklausu ir kiekviena ju turi savo atskira g, kad nesusipainiotu duomenys

load_dotenv() #env failo uzkrovimas


def create_app() -> Flask:
    """Flask aplikacijos fabrikas."""
    app = Flask(__name__)

    # ── Konfigūracija ──────────────────────────────────────────
    app.config["SECRET_KEY"]         = os.getenv("SECRET_KEY", "dev-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB  vartotojs negales ikelti per dideliu failu
    app.config["UPLOAD_FOLDER"]      = os.path.join("static", "uploads")  #kur saugomos originalio vartotojo ikeltos nuotraukos
    app.config["OUTPUT_FOLDER"]      = os.path.join("static", "outputs")    #aplankas, kur saugomos sugeneruotos nuotraukos
    app.config["MODEL_FOLDER"]       = "saved_models"  #aplankas kur saugomi treniruoti modeliai
    app.config["DATA_FOLDER"]        = "data"  #kur yra treniravimo duomenys

#eina per sarasa folderiu ir kiekviena sukuria, jeigu jo dar nera
    # ── Sukurk reikalingus aplankus ────────────────────────────
    for folder in [
        app.config["UPLOAD_FOLDER"],
        app.config["OUTPUT_FOLDER"],
        app.config["MODEL_FOLDER"],
    ]:
        os.makedirs(folder, exist_ok=True)

    # ── Duomenų bazė ───────────────────────────────────────────
    init_db() #paleidzia funkcija is db.py, kuri sukuria lenteles pagal models.py (sukuria lenteles, jeigu ju nera)

    @app.before_request
    def open_db():
        g.db = SessionLocal() #sessionlocal()funkcija, kuri sukuria nauja db sesija
    #vartotoas atidaro puslapi, Flask gauna uzklausa, pirmiausia paleisk db. 

    @app.teardown_request    #funkcija paleidziama, kai sesija baigiasi
    def close_db(error=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()
    #paiam duomenis ir g konteinerio ir ideda i db, jei jie egzistuoja. o jeigu neegzistuoja, tuomet grazina None

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

