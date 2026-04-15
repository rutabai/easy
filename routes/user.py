import os 
import torch
import uuid                                                                 #uuid unikaliems failu vardams generuoti
import zipfile                                                              #carousel nuotrauku supakavimui i zip
import io                                                                   #zip kurimui atmintyje
from flask import (
    Blueprint, render_template , request,
    redirect, url_for, jsonify, send_file, g, current_app
)
#render template -html puslapiui rodyti
#request - gauti formos ir failu duomenis
#redirect perkelti vartotoja i kita puslapi
#urlfor - sugeneruoti url
#jsonify - grazinti JSON atsakyma
#send_file issiusti faila prsisiuntimui
#current_app - pasiekti app config (pvz upload faila 

from database.models import Upload, Post, PostUpload                                              #importuoja tris lenteles
from utils.post_helpers import create_post_with_uploads, validate_upload_count                    #sukuria posta ir susieja su uploadais, tikrina ar story/carousel turi tinkama nuotrauku kieki
from utils.text_placement import process_image_with_text                                          #uzdeda teksta ant nuotraukos
from utils.constants import VALID_GOALS, VALID_CTA_TYPES, AVAILABLE_FILTERS, IDX_TO_CATEGORY      #formos patikrinimas ir nenaudojamas - kategoriju mapingui

user_bp = Blueprint("user", __name__)                                                             #visi routai siae aie priklauso user daliai ir app.py prijungia ji prie aplikacijos

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}                                               #leidziami tik sie failu formatai

#zemelapis tarp failo pletinio ir mime tipo. Failo pletinys nurodo kokio tipo failas.
MIME_TYPES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
}

def _allowed_file(filename: str) -> bool:                                                       #tikrina failo tipa. Jei pletinys yra leidziamu srase, grazina true. 
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

#funkcija atsakinga tiesiog uz failo issaugojima diske, o db irasas sukuriamas atskirai
#si funkcija issprendzia vardu konfliktus ir sudetingus failu vardus
def _save_upload(file, upload_folder: str) -> tuple[str, str, str]:
    original_name = file.filename  #"manofoto.jpg"
    ext           = original_name.rsplit(".", 1)[1].lower()                                     #['mano.foto', 'jpg']
    filename      = f"{uuid.uuid4().hex}.{ext}"                                                 #sugeneruoja unikalu varda, uuid4 generuoja atsitiktini ID, .hex pavercia teksta be bruksneliu
    filepath      = os.path.join(upload_folder, filename)                                       #sudeda pilna kelia: uploads/sugeneruotaskodas.jpg
    file.save(filepath)                                                                         #fiziskai issaugo faila
    return original_name, filename, filepath                                                    #sugrazina tris reiksmes kaip tuple


def _build_caption_from_parts(hook: str, story: str, cta: str) -> str:                          #sudeda caption is daliu (3 tekstines dalis grazina i viena)
    parts = [p for p in [hook, story, f"→ {cta}" if cta else ""] if p]
    return "\n\n".join(parts)
#list comprahension
#sarasas - [hook, story, f"→ {cta}" if cta else ""]
#"\n\n" - palieka 2 eiluciu tarpa


def _get_slide_texts(pu: PostUpload, post: Post) -> tuple[str, str, str]:                       #parenka koki teksta deti ant konkrecios skaidres
    """
    Pagal slide_type nustato kokį hook/body/cta rodyti ant skaidrės.
    Kiekviena skaidrė gauna skirtingą teksto kompoziciją.
    """
    if pu.slide_type in ("single", "hook"):                                                     #jei skaidre yra single arba hook, dedamas tik hook
        return post.hook or "", "", ""                                                          #sako return hook arba "hook tekstas", "body", "cta" arba grazina tuscia eilute vietoj klaidos
    elif pu.slide_type == "story":                                                              #jei story, tai dedamas body tekstas
        return "", pu.slide_text or post.story or "", ""
    elif pu.slide_type == "cta":                                                                #jei CTA, tai dedamas tik CTA tekstas
        return "", "", post.cta or ""
    return post.hook or "", post.story or "", post.cta or ""                                    #jei niekas nesutapo, tai grazina viska

#atidaro parindini puslapi
# ── Pagrindinis puslapis ───────────────────────────────────────
@user_bp.route("/")
def index():                                                                                    #funkcija, kuri paleidziama atidarius puslapi
    return render_template(                                                                     #paima index.html
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
#tiesiog standartine apsauga nuo manipuliavimo, jei vartotojas siustu bet ka, tik ne matoma UI

    valid_files = [f for f in files if f and _allowed_file(f.filename)]
    if not valid_files:
        return jsonify({"error": "Nepridėta tinkamų nuotraukų"}), 400
#tiesiog apsauga del sistemos apejimo. narsykle dabar leidzia kelti tik tinkamo formato failus    

    count = len(valid_files)

    if count == 1:
        post_type = "story"
    else:
        post_type = "carousel"

    ok, error = validate_upload_count(post_type, count)
    if not ok:
         return jsonify({"error": error}), 400
#jsonify - pavercia python zodyna i json formata, kuri narsykle supranta. Tiksliai nesuprantu kaip veikia AJAX, bet puslapis neperkraunamas, atnaujinama tik specifine vieta 

    upload_folder   = current_app.config["UPLOAD_FOLDER"]                                   #current app, nes esame blueprint
#du sarasai duomenu kaupimui
    upload_ids      = []
    saved_filepaths = []

    try:
        for file in valid_files:                                                             #einam per kiekviena valid nuotrauka
            original_name, filename, filepath = _save_upload(file, upload_folder)      #issaugo diske 3 reiksmes
            saved_filepaths.append(filepath)                                           #is karto issaugo kelia i sarasa

            ext = filepath.rsplit(".", 1)[1].lower()
            mime_type = MIME_TYPES.get(ext, "image/jpeg")

            from PIL import Image as PILImage
            try:
                with PILImage.open(filepath) as img:                #atidaro nuotrauka. with uztikrina, kad failas bus uzdarytas, net jei atsirado klaida
                    width, height = img.size                        #ispakuoja nuotraukos ddi i du kintamuosius (pvz 080,1920 kaip tuple)
                file_size = os.path.getsize(filepath)               #paima failo dydi is operacines sitemos
            except Exception:
                width = height = file_size = None                   #jei nuotraukos negali atidaryti Pillow, tuomet programa nesustoja, tiesiog iraso None

            upload_obj = Upload(                                    #upload sukuria objekta
                filename      = filename,
                original_name = original_name,
                filepath      = filepath,
                mime_type     = mime_type,
                file_size     = file_size,
                width         = width,
                height        = height,
            )
            g.db.add(upload_obj)                                      #prideda i DB sesija
            g.db.flush()                                              #issiuncia i DB, bet necomitina
            upload_ids.append(upload_obj.id)                          #issaugo gauta ID i sarasa
                                                                      #kai visos nuotruakos issaugotos ir ju ID surinkti, iskvieciama funkcija, kuri sukuria POSt irPOstupload irasus
        post = create_post_with_uploads(
            db               = g.db,
            upload_ids       = upload_ids,
            post_type        = post_type,
            topic            = topic,
            goal             = goal,
            cta_type         = cta_type,
            additional_notes = notes,
        )

        post.filter_name = filter_name                                  #pridedamas filtras
        g.db.commit()                                                   #commit i DB

    except Exception as e:                                              #e -klaidos aprasymas
        g.db.rollback()                                                 # atsaukia visus DB pakeitimus, irasai isnyksta
        for fp in saved_filepaths:                                      # einama per visus failus diske ir irasai isnyksta
            if os.path.exists(fp):                                      #istrina visus failus is disko, jeigu kazkokie buvo issaugoti
                os.remove(fp)
        return jsonify({"error": str(e)}), 500                          #serverio klaidos kodas

    return redirect(url_for("user.preview", post_id=post.id))           #jei viskas pavyksta peradresuoja i preview puslapi


# ── Preview puslapis ───────────────────────────────────────────       #tai, ka vartotojas mato preview puslapyje
@user_bp.route("/preview/<int:post_id>")                                #<int:post_id> reiskia, kad URL gali buti /preview/1.../preview42 ir tt skaicius automatiskai paduodamas i funkija kai post_id
def preview(post_id: int):                                                 
    post = g.db.query(Post).filter_by(id=post_id).first()               #iesko posto DB pagal ID. fist() grazina pirma rasta reiksme
    if not post:
        return "Postas nerastas", 404

    post_uploads = (                                                     #gauna visas skaidres susijusias su siuo postu surikiuotas pagal pozicija, kad carousel skaidres eitu teisinga tvarka
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .all()
    )
    uploads = [pu.upload for pu in post_uploads]                        #buotrauku sarasas teisinga tvarka. THML'ui riekia upload objektu (nuotrauku duomenu), kad parodytu nuotraukas

    return render_template(                                             #render template atidaro preview.html ir pduoda 4 kintamuosius i sablona
        "preview.html",
        post         = post,                                            #kaire - vardas html sablone, o desine python kintamasis
        uploads      = uploads,
        post_uploads = post_uploads,
        filters      = AVAILABLE_FILTERS,
    )


# ── Nuotraukos klasifikavimas ir teksto generavimas ────────────       Ivestis: post_id, isvestis: JSON su hook, story, cta. Keicia: post irasa DB - prideda kategorija, teksta, status
@user_bp.route("/generate/<int:post_id>", methods=["POST"])
def generate(post_id: int):
    from utils.caption_generator import generate_caption                #importuojamas cia, nes caption generator gali buti sunkus ir letas ir importuojamas tik kai reikia

    post = g.db.query(Post).filter_by(id=post_id).first()               #iesko posto duomenu bazeje. jei neranda sustoja su 404
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    post_uploads = (                                                    #vel gauna visas surikiuotas nuotraukas
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .all()
    )
    slide_count  = len(post_uploads)                                     #gauna kiek nuotrauku is viso
    first_upload = post_uploads[0].upload if post_uploads else None      #pirmojis nuotrauka ji ir bus klasifikuojama

    # Klasifikuok pirmą nuotrauką su ViT
    if first_upload:
        try:
            from models.vit_model import load_vit_model, predict_single_vit      #jeigu yra nuotrauka, importuojami ViT modelio irankiai
            from transformers import ViTImageProcessor                           #paruosia nuotrauka pries paduodant i ViT. keicia nuotraukos dydi, normalizuoja pixels, pavercia i tennsor formata kuri modelis supranta

            model_path = os.path.join(                                           #sudeda pilna kelia iki istreniruoto modelio failo
                current_app.config["MODEL_FOLDER"], "vit_model.pth"               ## → "saved_models", → modelio failo vardas
            )
            if os.path.exists(model_path):                                       #tikrina, r istreniruotas modelis egzistuoja diske. jei ne - praleidziama
                processor = ViTImageProcessor.from_pretrained(                   #uzkraunamas ViT processorus (paruosia nuotrauka modeliui) ir pats modelis
                    "google/vit-base-patch16-224"                                #tai yra higging face modeli identifikatorius
                )
                model    = load_vit_model(model_path, freeze_backbone=False)     #dabar uzkrauna mano istreniruota ViT modeli. is saved mddels. freeze_backbone reiskia, kad visi modeliai aktyvus
                category, confidence = predict_single_vit(                       #funkcija paima nuotrauka is disko, paruosia ja, analizuoja modeli ir grazina kategorija pvzfood) ir confidence pvz 94 proc.
                    model     = model,
                    processor = processor,
                    filepath  = first_upload.filepath,
                    device    = "cuda" if torch.cuda.is_available() else "cpu",
                )
                post.predicted_category = category                                #issaugo post lenteleje kategorija, confidence gauta ir koks modelis
                post.confidence         = confidence
                post.model_used         = "vit"
        except Exception as e:
            print(f"⚠️  Klasifikavimas nepavyko: {e}")

    category    = post.predicted_category or "lifestyle"                        #jei klasifikavimas nepavyko, - naudojama lifestyle kaip numatytas
    post.status = "processing"
    g.db.commit()       

    try:
        result = generate_caption(                                              #iskvieciamas anthropic API funkcija. Grazina zodyna su sugeneruotu tekstu
            category         = category,
            post_type        = post.post_type,
            topic            = post.topic,
            goal             = post.goal,
            cta_type         = post.cta_type,
            additional_notes = post.additional_notes,
            slide_count      = slide_count,
        )

        post.hook    = result.get("hook", "")                                    #isssaugo sugeneruota teksta i post
        post.story   = result.get("story", "")
        post.cta     = result.get("cta", "")
        post.caption = result.get("caption", "")
        post.status  = "completed"                                               #jei API negrazino hook, naudojama tuscia eilute ir statusas pakeiciamas i completed

        for pu in post_uploads:                                                  #einama per kiekviena nuotrauka ir priskiramas tinkamas tekstas pagal tipa (hook, storry telling ad CTA)
            if pu.slide_type in ("single", "hook"):
                pu.slide_text = post.hook
            elif pu.slide_type == "story":
                pu.slide_text = post.story
            elif pu.slide_type == "cta":
                pu.slide_text = post.cta

        g.db.commit()                                                             #viskas issaugoma DB ir grazinamas JSON su sugeneruotu tekstu narsyklei
        return jsonify(result)

    except Exception as e:
        post.status = "failed"
        g.db.commit()
        return jsonify({"error": str(e)}), 500


# ── Teksto koregavimas ─────────────────────────────────────────
@user_bp.route("/adjust/<int:post_id>", methods=["POST"])
def adjust(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()                       #iesko posto DB
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404                       #jei neranda grazina 404

    data = request.get_json()                                                   #paima duomenis, kuriuos JavaScript siuncia is narsykles
    if not data:
        return jsonify({"error": "Nėra duomenų"}), 400

    post.hook  = data.get("hook",  post.hook)
    post.story = data.get("story", post.story)
    post.cta   = data.get("cta",   post.cta)
    post.caption = _build_caption_from_parts(                                   #automatiskai persiduoda caption is nauju daliu
        post.hook or "", post.story or "", post.cta or ""
    )

    g.db.commit()
    return jsonify({"success": True, "caption": post.caption})


# ── AI teksto tobulinimas ──────────────────────────────────────               Tercias teksto rezimas projekte. Tobulina vartotojo teksta
@user_bp.route("/refine/<int:post_id>", methods=["POST"])
def refine(post_id: int):
    from utils.caption_generator import refine_caption

    post = g.db.query(Post).filter_by(id=post_id).first()                       #perraso AI perdaryta vartotojo kurta caption
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Nėra duomenų"}), 400

    user_edit = data.get("user_edit", "")                                        #paima vartotojo generota teksta
    original  = _build_caption_from_parts(                                       #sudeda orifinalu teksta i viena, kad AI zinoti nuo ko pradeti tobultini
        post.hook or "", post.story or "", post.cta or ""
    )

    try:
        result = refine_caption(                                                #iskviecia anthropic API su dviem tekstais originaliu, vartotojo pakeistu. AI grazina tobula versija
            original_text = original,
            user_edit     = user_edit,
            category      = post.predicted_category or "lifestyle",
            post_type     = post.post_type,
        )

        post.hook    = result.get("hook",    post.hook)                          #atnaujina kiekviena dali
        post.story   = result.get("story",   post.story)
        post.cta     = result.get("cta",     post.cta)
        post.caption = result.get("caption") or _build_caption_from_parts(       #jei API negrazino kokios dalies, paliekama sena
            post.hook or "", post.story or "", post.cta or ""
        )
        g.db.commit()                                                            #issaugoma DB 
        return jsonify(result)                                                   #grazinama narsyklei

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Nuotraukos renderinimas ────────────────────────────────────
@user_bp.route("/render/<int:post_id>", methods=["POST"])
def render_image(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()
    if not post:
        return jsonify({"error": "Postas nerastas"}), 404

    data        = request.get_json() or {}
    offset_x    = int(data.get("offset_x", 0))                                               #teksto pozicijos poslinkis (kai vartotojas spaudžia rodyklių mygtukus ← → ↑ ↓)
    offset_y    = int(data.get("offset_y", 0))
    zone        = data.get("zone")                                                           #teksto zona (Viršus/Vidurys/Apačia)
    filter_name = data.get("filter_name", post.filter_name or "original")                    #pasirinktas filtras (COOL, WARM ir t.t.)

    if filter_name not in AVAILABLE_FILTERS:
        return jsonify({"error": f"Nežinomas filtras: {filter_name}"}), 400

    post.filter_name = filter_name
    output_folder    = current_app.config["OUTPUT_FOLDER"]

    post_uploads = (                                                                        #gauna visas nuotraukas is DB
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .all()
    )
    if not post_uploads:
        return jsonify({"error": "Nuotraukos nerastos"}), 404

    try:
        image_urls = []

        for pu in post_uploads:                                                             #pu yra Post_uploads:
            hook, body, cta = _get_slide_texts(pu, post)                                    #kiekvienai nuotruakai parenka tinkama teksta pagal tipa

            output_path = os.path.join(                                                     #sugeneruoja isvesties kelia
                output_folder,
                f"post_{post_id}_slide_{pu.position}.jpg"
            )

            process_image_with_text(                                                        #uždeda tekstą ant nuotraukos su filtru ir išsaugo naują failą diske.
                image_path  = pu.upload.filepath,
                hook        = hook,
                body        = body,
                cta         = cta,
                output_path = output_path,
                category    = post.predicted_category,
                zone        = zone,
                offset_x    = offset_x,
                offset_y    = offset_y,
                filter_name = filter_name,
                post_type   = post.post_type,
            )

            pu.output_image_path = output_path

            # Pirmosios skaidrės kelias — story atgaliniam suderinamumui
            if pu.position == 1:
                post.output_path = output_path

            relative  = output_path.replace("\\", "/").replace("static/", "", 1)            #pavercia disko kelia i URL kuri narsykle supranta
            image_urls.append(url_for("static", filename=relative))

        g.db.commit()

        return jsonify({                                                                     #grazina URL sarasa ir JAVA script atnaujina nuotrauka puslapyje be perkrovimo
            "success":    True,
            "image_url":  image_urls[0],   # pirmoji skaidrė preview
            "image_urls": image_urls,       # visos skaidrės carousel
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Parsisiuntimas ─────────────────────────────────────────────
@user_bp.route("/download/<int:post_id>")
def download(post_id: int):
    post = g.db.query(Post).filter_by(id=post_id).first()                                   #suranda duomenu bazehe ID posto
    if not post:
        return "Postas nerastas", 404

    post_uploads = (                                                                         #suranda konkrecias nuotraukas
        g.db.query(PostUpload)
        .filter_by(post_id=post_id)
        .order_by(PostUpload.position)
        .all()
    )

    # Surink visus sugeneruotus failus
    output_files = [                                                                        #surenka tik tuos postus, kurie turi kelia, fiziskai egzistuoja diske
        pu.output_image_path for pu in post_uploads
        if pu.output_image_path and os.path.exists(pu.output_image_path)
    ]

    if not output_files:
        return "Nuotraukos dar nesugenenuotos", 404

    # Story — vienas failas
    if post.post_type == "story" and len(output_files) == 1:                                  #jei failas yra story, tai Flask siuncia faila narsyklei, kuri parsiuncia, bet neatidaro
        return send_file(
            output_files[0],
            as_attachment = True,
            download_name = f"instagram_story_{post_id}.jpg",
        )

    # Carousel — ZIP archyvas
    zip_buffer = io.BytesIO()                                                                 #sukuria ZIP atmintyje, ne diske
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:                        #atidaro ZIP razymui. ZIP DEFLATED suspauimo algoritmas
        for i, filepath in enumerate(output_files, start=1):                                  #sunumeruoja nuo 1
            zf.write(filepath, arcname=f"slide_{i}.jpg")                                      #kaip faila vadinsis

    zip_buffer.seek(0)                                                                        #seek grazina zymekli i pradzia pries siuntima
    return send_file(
        zip_buffer,
        as_attachment = True,
        download_name = f"instagram_carousel_{post_id}.zip",
        mimetype      = "application/zip",                                                    #pasako narsyklei, kad tai ZIP failas
    )