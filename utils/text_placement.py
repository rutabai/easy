import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import os

from utils.constants import AVAILABLE_FILTERS

# Teksto pozicijų numatytosios reikšmės pagal kategoriją
DEFAULT_POSITIONS = {
    "food":      "bottom",
    "portrait":  "bottom",
    "landscape": "top",
    "product":   "bottom",
    "lifestyle": "bottom",
}

# Overlay nustatymai
OVERLAY_OPACITY = 0.6
OVERLAY_HEIGHT  = 0.30
OVERLAY_COLOR   = (0, 0, 0)

# Teksto nustatymai
TEXT_COLOR     = (255, 255, 255)
TEXT_PADDING   = 20
FONT_SIZE_HOOK = 28
FONT_SIZE_BODY = 20
FONT_SIZE_CTA  = 18

# Tuščios vietos paieškos slenkstis
VARIANCE_THRESHOLD = 500


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Bando įkelti šriftą, grįžta prie default jei nepavyksta."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
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
    Padalina į tris zonas: top / middle / bottom.
    Grąžina zoną su mažiausia pikselių dispersija.
    """
    img_array = np.array(image.convert("L"))
    h, w = img_array.shape

    zones = {
        "top":    img_array[:h//3, :],
        "middle": img_array[h//3:2*h//3, :],
        "bottom": img_array[2*h//3:, :],
    }

    variances = {zone: float(np.var(data)) for zone, data in zones.items()}
    return min(variances, key=variances.get)


def _get_text_position(zone: str, img_h: int) -> tuple[int, int]:
    """
    Apskaičiuoja overlay y koordinates pagal zoną.
    Grąžina (y_start, y_end).
    """
    overlay_h = int(img_h * OVERLAY_HEIGHT)

    if zone == "top":
        return 0, overlay_h
    elif zone == "middle":
        mid = img_h // 2
        return mid - overlay_h // 2, mid + overlay_h // 2
    else:  # bottom
        return img_h - overlay_h, img_h


def _clamp_overlay(y_start: int, img_h: int) -> tuple[int, int]:
    """
    Užtikrina, kad overlay neišliptų už nuotraukos ribų.
    Grąžina patikslintą (y_start, y_end).
    """
    overlay_h = int(img_h * OVERLAY_HEIGHT)
    y_start   = max(0, min(y_start, img_h - overlay_h))
    y_end     = y_start + overlay_h
    return y_start, y_end


def add_text_overlay(
    image: PILImage.Image,
    hook: str,
    body: str,
    cta: str,
    category: str | None = None,
    zone: str | None = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> PILImage.Image:
    """
    Uždeda tekstą ant nuotraukos su pusiau permatomu overlay.

    Parametrai:
    - image:    PIL nuotrauka (jau su pritaikytu filtru)
    - hook:     pirmas sakinys
    - body:     pagrindinis tekstas
    - cta:      kvietimas veikti
    - category: nustato default poziciją
    - zone:     top / middle / bottom (jei None — ieško automatiškai)
    - offset_x, offset_y: vartotojo pozicijos koregavimas

    Grąžina naują PIL nuotrauką su tekstu.
    """
    img    = image.copy().convert("RGBA")
    img_w, img_h = img.size

    # 1. Nustatyk zoną
    if zone is None:
        zone = find_empty_zone(image)
        # Jei kategorija žinoma ir nuotrauka labai raiški — naudok default
        if category in DEFAULT_POSITIONS:
            variance = float(np.var(np.array(image.convert("L"))))
            if variance > VARIANCE_THRESHOLD * 3:
                zone = DEFAULT_POSITIONS[category]

    # 2. Bazinė y pozicija pagal zoną
    y_start, _ = _get_text_position(zone, img_h)

    # 3. Taikyk offset ir clamp — tekstas neišlips iš nuotraukos
    y_start       = y_start + offset_y
    y_start, y_end = _clamp_overlay(y_start, img_h)

    # 4. X offset su riba
    x_start = max(0, min(offset_x, img_w - 100))

    # 5. Overlay sluoksnis
    overlay      = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    alpha        = int(255 * OVERLAY_OPACITY)
    draw_overlay.rectangle(
        [(x_start, y_start), (img_w, y_end)],
        fill=(*OVERLAY_COLOR, alpha)
    )
    img  = PILImage.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # 6. Šriftai
    font_hook = _load_font(FONT_SIZE_HOOK)
    font_body = _load_font(FONT_SIZE_BODY)
    font_cta  = _load_font(FONT_SIZE_CTA)

    # 7. Teksto pozicijos
    x = x_start + TEXT_PADDING
    y = y_start + TEXT_PADDING

    max_text_width = img_w - x - TEXT_PADDING

    # Hook
    if hook:
        wrapped = _wrap_text(hook, font_hook, max_text_width)
        for line in wrapped:
            if y + FONT_SIZE_HOOK > y_end - TEXT_PADDING:
                break
            draw.text((x, y), line, font=font_hook, fill=TEXT_COLOR)
            y += FONT_SIZE_HOOK + 8

    # Body
    if body:
        wrapped = _wrap_text(body, font_body, max_text_width)
        for line in wrapped:
            if y + FONT_SIZE_BODY > y_end - TEXT_PADDING:
                break
            draw.text((x, y), line, font=font_body, fill=TEXT_COLOR)
            y += FONT_SIZE_BODY + 4

    # CTA
    if cta:
        cta_y = y_end - FONT_SIZE_CTA - TEXT_PADDING
        if cta_y > y:  # rodyti tik jei telpa
            draw.text((x, cta_y), f"→ {cta}", font=font_cta, fill=(255, 220, 50))

    return img.convert("RGB")


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Laužo ilgą tekstą į eilutes pagal maksimalų plotį."""
    words        = text.split()
    lines        = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        try:
            bbox       = font.getbbox(test_line)
            text_width = bbox[2]
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

    return lines


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
) -> str:
    """
    Pilnas apdorojimo pipeline:
    1. Įkelia nuotrauką
    2. Pritaiko filtrą
    3. Uždeda tekstą
    4. Išsaugo rezultatą

    Eiliškumas: nuotrauka → filtras → tekstas
    (filtras prieš tekstą — tekstas lieka aiškus)

    Grąžina output_path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Nuotrauka nerasta: {image_path}")

    # Validuok filtro pavadinimą
    if filter_name not in AVAILABLE_FILTERS:
        raise ValueError(
            f"Nežinomas filtras: '{filter_name}'. "
            f"Galimi: {AVAILABLE_FILTERS}"
        )

    from utils.image_filters import apply_filter

    # 1. Įkelk
    image = PILImage.open(image_path).convert("RGB")

    # 2. Filtras PRIEŠ tekstą
    if filter_name != "original":
        image = apply_filter(image, filter_name)

    # 3. Tekstas ant filtruotos nuotraukos
    result = add_text_overlay(
        image=image,
        hook=hook,
        body=body,
        cta=cta,
        category=category,
        zone=zone,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    # 4. Išsaugok
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path, quality=95)
    print(f"✅ Nuotrauka išsaugota: {output_path}")
    return output_path