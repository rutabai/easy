from database.models import Post, PostUpload, Upload
from sqlalchemy.orm import Session

# Limitas pagal posto tipą
MAX_UPLOADS = {
    "story": 1,
    "carousel": 10
}
MIN_UPLOADS = {
    "story": 1,
    "carousel": 2
}


def assign_slide_types(count: int) -> list[str]:
    """
    Priskiria slide_type kiekvienai nuotraukai pagal kiekį.

    1 nuotrauka  → ["single"]
    2 nuotraukos → ["hook", "cta"]
    3+ nuotraukos → ["hook", "story"..., "cta"]
    """
    if count == 1:
        return ["single"]
    elif count == 2:
        return ["hook", "cta"]
    else:
        middle = ["story"] * (count - 2)
        return ["hook"] + middle + ["cta"]


def validate_upload_count(post_type: str, count: int) -> tuple[bool, str]:
    """
    Tikrina ar nuotraukų skaičius atitinka posto tipą.
    Grąžina (True, "") jei gerai, arba (False, klaidos_pranešimas) jei blogai.
    """
    if post_type == "story":
        if count != 1:
            return False, "Story tipui reikalinga lygiai 1 nuotrauka"
    elif post_type == "carousel":
        if count < MIN_UPLOADS["carousel"]:
            return False, f"Carousel tipui reikia mažiausiai {MIN_UPLOADS['carousel']} nuotraukų"
        if count > MAX_UPLOADS["carousel"]:
            return False, f"Carousel tipui galima daugiausiai {MAX_UPLOADS['carousel']} nuotraukų"
    else:
        return False, f"Nežinomas posto tipas: {post_type}"

    return True, ""


def determine_post_type(count: int) -> str:
    """
    Automatiškai nustato posto tipą pagal nuotraukų skaičių.
    1 nuotrauka → story
    2-10 nuotraukų → carousel
    """
    if count == 1:
        return "story"
    return "carousel"


def create_post_with_uploads(
    db: Session,
    upload_ids: list[int],
    post_type: str
) -> Post:
    """
    Sukuria Post ir susieja su Upload per PostUpload.
    Automatiškai priskiria slide_type kiekvienai nuotraukai.
    """
    count = len(upload_ids)

    # Validacija
    valid, error = validate_upload_count(post_type, count)
    if not valid:
        raise ValueError(error)

    # Sukurk postą
    post = Post(
        post_type=post_type,
        status="pending"
    )
    db.add(post)
    db.flush()  # gauti post.id prieš PostUpload kūrimą

    # Priskyrk slide tipus
    slide_types = assign_slide_types(count)

    # Sukurk PostUpload ryšius
    for position, (upload_id, slide_type) in enumerate(
        zip(upload_ids, slide_types), start=1
    ):
        post_upload = PostUpload(
            post_id=post.id,
            upload_id=upload_id,
            position=position,
            slide_type=slide_type
        )
        db.add(post_upload)

    db.commit()
    return post