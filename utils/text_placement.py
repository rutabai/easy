import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps
import os

from utils.constants import AVAILABLE_FILTERS

# ── Default pozicijos pagal kategoriją ────────────────────────
DEFAULT_POSITIONS = {
    "food":      "bottom",
    "portrait":  "bottom",
    "landscape": "top",
    "product":   "bottom",
    "lifestyle": "bottom",
}

# ── Overlay nustatymai ─────────────────────────────────────────
OVERLAY_OPACITY = 0.58
OVERLAY_HEIGHT  = 0.30
OVERLAY_COLOR   = (0, 0, 0)

# ── Teksto nustatymai ──────────────────────────────────────────
TEXT_COLOR  = (255, 255, 255)
TEXT_PADDING = 32

FONT_SIZE_HOOK = 48
FONT_SIZE_BODY = 22
FONT_SIZE_CTA  = 22

# ── Teksto ilgio ribos (simboliais) ───────────────────────────
MAX_HOOK_CHARS = 50
MAX_BODY_CHARS = 120
MAX_CTA_CHARS  = 40

# ── Eilučių ribos ─────────────────────────────────────────────
MAX_BODY_LINES_STORY    = 2
MAX_BODY_LINES_CAROUSEL = 3


def _fix_exif_orientation(image: PILImage.Image) -> PILImage.Image:
    """
    Taiso nuotraukos orientaciją pagal EXIF duomenis.
    Naudoja ImageOps.exif_transpose() — patikimiau nei rankinis apdorojimas.
    SVARBU: kviesti PRIEŠ convert("RGB").
    """
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def _truncate(text: str, max_chars: int) -> str:
    """
    Sutrumpina tekstą iki max_chars simbolių.
    Jei tekstas ilgesnis — nukerpa prie paskutinio žodžio ir prideda '…'.
    """
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated.rstrip(".,!?—–") + "…"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Bando įkelti šriftą, grįžta prie default jei nepavyksta."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def find_empty_zone(image: PILImage.Image) -> str:
    """
    Ieško tuščiausios zonos nuotraukoje.
    Padalija į tris zonas: top / middle / bottom.
    Grąžina zoną su mažiausia pikselių dispersija —
    mažesnė dispersija = vienodesnė spalva = geriau skaitomas tekstas.
    """
    img_array = np.array(image.convert("L"))
    h, _ = img_array.shape

    zones = {
        "top":    img_array[:h // 3, :],
        "middle": img_array[h // 3:2 * h // 3, :],
        "bottom": img_array[2 * h // 3:, :],
    }

    variances = {zone: float(np.var(data)) for zone, data in zones.items()}
    return min(variances, key=variances.get)


def _get_overlay_y(zone: str, img_h: int) -> tuple[int, int]:
    """
    Apskaičiuoja overlay y koordinates pagal zoną.
    Grąžina (y_start, y_end).
    """
    overlay_h = int(img_h * OVERLAY_HEIGHT)

    if zone == "top":
        return 0, overlay_h
    elif zone == "bottom":
        return img_h - overlay_h, img_h
    else:  # middle
        mid  = img_h // 2
        half = overlay_h // 2
        return mid - half, mid + half


def _clamp_overlay(y_start: int, img_h: int) -> tuple[int, int]:
    """Užtikrina, kad overlay neišliptų už nuotraukos ribų."""
    overlay_h = int(img_h * OVERLAY_HEIGHT)
    y_start   = max(0, min(y_start, img_h - overlay_h))
    y_end     = y_start + overlay_h
    return y_start, y_end


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """
    Laužo tekstą į eilutes pagal maksimalų plotį pikseliais.
    Naudoja font.getbbox() tiksliam teksto pločio matavimui.
    """
    words        = text.split()
    lines        = []
    current_line = ""

    for word in words:
        test_line  = f"{current_line} {word}".strip()
        try:
            bbox       = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            text_width = len(test_line) * 10

        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def add_text_overlay(
    image: PILImage.Image,
    hook: str,
    body: str,
    cta: str,
    category: str | None = None,
    zone: str | None = None,
    offset_x: int = 0,
    offset_y: int = 0,
    post_type: str = "story",
) -> PILImage.Image:
    """
    Uždeda tekstą ant nuotraukos su pusiau permatomu overlay.

    Story ir carousel logika skiriasi:
    - Story:    hook + max 2 body eilutės + CTA
    - Carousel: hook + max 3 body eilutės (CTA tik paskutinėje skaidrėje)

    Overlay — per visą nuotraukos plotį.
    Tekstas — stumdomas su offset_x (leidžiamos ir neigiamos reikšmės).
    """
    img      = image.copy().convert("RGBA")
    img_w, img_h = img.size

    # 1. Zonos parinkimas
    if zone is None:
        if category in DEFAULT_POSITIONS:
            zone = DEFAULT_POSITIONS[category]
        else:
            zone = find_empty_zone(image)

    # 2. Y pozicija + offset + clamp
    y_start, _ = _get_overlay_y(zone, img_h)
    y_start     = y_start + offset_y
    y_start, y_end = _clamp_overlay(y_start, img_h)

    # 3. Overlay — per visą plotį
    overlay      = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    alpha        = int(255 * OVERLAY_OPACITY)
    draw_overlay.rectangle(
        [(0, y_start), (img_w, y_end)],
        fill=(*OVERLAY_COLOR, alpha)
    )
    img  = PILImage.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # 4. Šriftai — proporcingi nuotraukos dydžiui
    scale = min(img_w, img_h) / 800
    scale = max(0.6, min(scale, 1.8))

    font_hook = _load_font(int(FONT_SIZE_HOOK * scale))
    font_body = _load_font(int(FONT_SIZE_BODY * scale))
    font_cta  = _load_font(int(FONT_SIZE_CTA  * scale))

    line_h_hook = int(FONT_SIZE_HOOK * scale) + 10
    line_h_body = int(FONT_SIZE_BODY * scale) + 8
    line_h_cta  = int(FONT_SIZE_CTA  * scale) + 6

    # 5. Teksto x pozicija — offset_x į abi puses
    text_x     = TEXT_PADDING + offset_x
    text_x     = max(TEXT_PADDING, min(text_x, img_w - TEXT_PADDING - 100))
    max_text_w = img_w - text_x - TEXT_PADDING

    safe_bottom  = y_end - TEXT_PADDING
    cta_reserved = line_h_cta + TEXT_PADDING if cta else 0
    text_bottom  = safe_bottom - cta_reserved

    y = y_start + TEXT_PADDING

    # 6. Hook — sutrumpintas, didelis
    if hook:
        hook_short = _truncate(hook, MAX_HOOK_CHARS)
        for line in _wrap_text(hook_short, font_hook, max_text_w):
            if y + line_h_hook > text_bottom:
                break
            draw.text((text_x, y), line, font=font_hook, fill=TEXT_COLOR)
            y += line_h_hook
        y += 12  # didesnis tarpas po hook

    # 7. Body — sutrumpintas, max eilučių pagal tipą
    if body and y < text_bottom:
        body_short   = _truncate(body, MAX_BODY_CHARS)
        max_lines    = MAX_BODY_LINES_STORY if post_type == "story" else MAX_BODY_LINES_CAROUSEL
        wrapped_body = _wrap_text(body_short, font_body, max_text_w)[:max_lines]
        for line in wrapped_body:
            if y + line_h_body > text_bottom:
                break
            draw.text((text_x, y), line, font=font_body, fill=TEXT_COLOR)
            y += line_h_body

    # 8. CTA — visada apačioje, geltona, sutrumpinta
    if cta:
        cta_short = _truncate(cta, MAX_CTA_CHARS)
        cta_y     = safe_bottom - line_h_cta
        if cta_y > y_start + TEXT_PADDING:
            draw.text(
                (text_x, cta_y),
                f"→ {cta_short}",
                font=font_cta,
                fill=(255, 215, 50),
            )

    return img.convert("RGB")


def process_image_with_text(
    image_path: str,
    hook: str,
    body: str,
    cta: str,
    output_path: str,
    category: str | None = None,
    zone: str | None = None,
    offset_x: int = 0,
    offset_y: int = 0,
    filter_name: str = "original",
    post_type: str = "story",
) -> str:
    """
    Pilnas apdorojimo pipeline:
    1. Įkelia nuotrauką (be convert — EXIF turi išlikti)
    2. Taiso EXIF orientaciją
    3. Konvertuoja į RGB
    4. Pritaiko filtrą
    5. Uždeda tekstą
    6. Išsaugo rezultatą
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Nuotrauka nerasta: {image_path}")

    if filter_name not in AVAILABLE_FILTERS:
        raise ValueError(
            f"Nežinomas filtras: '{filter_name}'. Galimi: {AVAILABLE_FILTERS}"
        )

    from utils.image_filters import apply_filter

    # 1. Įkelk be convert
    image = PILImage.open(image_path)

    # 2. EXIF fix PRIEŠ convert
    image = _fix_exif_orientation(image)

    # 3. Konvertuok
    image = image.convert("RGB")

    # 4. Filtras prieš tekstą
    if filter_name != "original":
        image = apply_filter(image, filter_name)

    # 5. Tekstas
    result = add_text_overlay(
        image     = image,
        hook      = hook,
        body      = body,
        cta       = cta,
        category  = category,
        zone      = zone,
        offset_x  = offset_x,
        offset_y  = offset_y,
        post_type = post_type,
    )

    # 6. Išsaugok
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result.save(output_path, quality=95)
    print(f"✅ Nuotrauka išsaugota: {output_path}")
    return output_path