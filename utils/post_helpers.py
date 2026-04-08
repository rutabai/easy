from database.models import Post, PostUpload, Upload
from sqlalchemy.orm import Session

# Limitas pagal posto tipą
from utils.constants import MAX_UPLOADS, MIN_UPLOADS, VALID_GOALS, VALID_CTA_TYPES, SLIDE_TYPES

# Galimos reikšmės

# Slide tipo aprašymai:
# single → story tipo postas su viena nuotrauka
# hook   → pirma carousel skaidrė
# story  → vidurinės carousel skaidrės
# cta    → paskutinė carousel skaidrė


def validate_upload_count(post_type: str, count: int) -> tuple[bool, str]:
    """Tikrina ar nuotraukų skaičius atitinka posto tipą."""
    if post_type not in MIN_UPLOADS:
        return False, f"Nežinomas posto tipas: {post_type}. Galimi: {list(MIN_UPLOADS.keys())}"
    if count < MIN_UPLOADS[post_type]:
        return False, f"{post_type.capitalize()} tipui reikia mažiausiai {MIN_UPLOADS[post_type]} nuotraukų"
    if count > MAX_UPLOADS[post_type]:
        return False, f"{post_type.capitalize()} tipui galima daugiausiai {MAX_UPLOADS[post_type]} nuotraukų"
    return True, ""


def validate_intent(goal: str | None, cta_type: str | None) -> tuple[bool, str]:
    """Tikrina ar goal ir cta_type yra iš leistinų reikšmių."""
    if goal is not None and goal not in VALID_GOALS:
        return False, f"Nežinomas tikslas: '{goal}'. Galimi: {VALID_GOALS}"
    if cta_type is not None and cta_type not in VALID_CTA_TYPES:
        return False, f"Nežinomas CTA tipas: '{cta_type}'. Galimi: {VALID_CTA_TYPES}"
    return True, ""


def assign_slide_types(count: int) -> list[str]:
    """
    Priskiria slide_type kiekvienai nuotraukai pagal kiekį.
    1 → ["single"]
    2 → ["hook", "cta"]
    3+ → ["hook", "story"..., "cta"]
    """
    if count < MIN_UPLOADS["story"]:
        raise ValueError(f"Nuotraukų skaičius negali būti mažesnis nei {MIN_UPLOADS['story']}")
    if count > MAX_UPLOADS["carousel"]:
        raise ValueError(f"Nuotraukų skaičius negali viršyti {MAX_UPLOADS['carousel']}")

    if count == 1:
        return ["single"]
    elif count == MIN_UPLOADS["carousel"]:
        return ["hook", "cta"]
    else:
        middle = ["story"] * (count - 2)
        return ["hook"] + middle + ["cta"]


def create_post_with_uploads(
    db: Session,
    upload_ids: list[int],
    post_type: str,
    topic: str | None = None,
    goal: str | None = None,
    cta_type: str | None = None,
    additional_notes: str | None = None,
) -> Post:
    """
    Sukuria Post ir susieja su Upload per PostUpload lentelę.

    Tikrina:
    - post_type teisingumą
    - upload_ids dublikatus
    - upload_ids egzistavimą DB (vienu užklausimu)
    - nuotraukų kiekio atitikimą tipo taisyklėms
    - goal ir cta_type leistinumą

    Grąžina sukurtą Post objektą.
    """
    count = len(upload_ids)

    # Tikrink dublikatus
    if len(upload_ids) != len(set(upload_ids)):
        raise ValueError("upload_ids sąraše yra pasikartojančių ID")

    # Validuok kiekį pagal tipo taisykles
    valid, error = validate_upload_count(post_type, count)
    if not valid:
        raise ValueError(error)

    # Validuok goal ir cta_type
    valid, error = validate_intent(goal, cta_type)
    if not valid:
        raise ValueError(error)

    # Tikrink ar visi upload_ids egzistuoja DB — vienu užklausimu
    found_ids = {
        row.id for row in db.query(Upload.id).filter(Upload.id.in_(upload_ids)).all()
    }
    missing = set(upload_ids) - found_ids
    if missing:
        raise ValueError(f"Šie Upload ID nerasti DB: {missing}")

    try:
        # Sukurk postą su vartotojo intencija
        post = Post(
            post_type=post_type,
            status="pending",
            topic=topic,
            goal=goal,
            cta_type=cta_type,
            additional_notes=additional_notes,
        )
        db.add(post)
        db.flush()  # gauti post.id prieš PostUpload kūrimą

        # Priskyrk slide tipus ir sukurk ryšius
        slide_types = assign_slide_types(count)

        for position, (upload_id, slide_type) in enumerate(
            zip(upload_ids, slide_types), start=1
        ):
            post_upload = PostUpload(
                post_id=post.id,
                upload_id=upload_id,
                position=position,
                slide_type=slide_type,
                # slide_text užpildomas vėliau — teksto generavimo etape
            )
            db.add(post_upload)

        db.commit()
        return post

    except Exception as e:
        db.rollback()
        raise e