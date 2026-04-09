import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import os


# Teksto pozicijų numatytosios reikšmės pagal kategoriją
DEFAULT_POSITIONS = {
    "food":      "bottom",
    "portrait":  "bottom",
    "landscape": "top",
    "product":   "bottom",
    "lifestyle": "bottom",
}

# Overlay nustatymai
OVERLAY_OPACITY = 0.6      # pusiau permatomas fonas
OVERLAY_HEIGHT  = 0.30     # overlay užima 30% nuotraukos aukščio
OVERLAY_COLOR   = (0, 0, 0)  # juodas fonas

# Teksto nustatymai
TEXT_COLOR       = (255, 255, 255)  # baltas tekstas
TEXT_PADDING     = 20               # tarpas nuo kraštų pikseliais
FONT_SIZE_HOOK   = 28
FONT_SIZE_BODY   = 20
FONT_SIZE_CTA    = 18

# Tuščios vietos paieškos slenkstis
VARIANCE_THRESHOLD = 500  # žema dispersija = tuščia vieta


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
    Padalina nuotrauką į tris zonas: top / middle / bottom.
    Grąžina zonos pavadinimą su mažiausia pikselių dispersija.
    """
    img_array = np.array(image.convert("L"))  # pilka spalva
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
    Apskaičiuoja overlay pradžios y koordinatę pagal zoną.
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
    - image: PIL nuotrauka
    - hook: pirmas sakinys
    - body: pagrindinis tekstas
    - cta: kvietimas veikti
    - category: nuotraukos kategorija (nustato default poziciją)
    - zone: top / middle / bottom (jei None — ieško automatiškai)
    - offset_x, offset_y: vartotojo pozicijos koregavimas rodyklėmis

    Grąžina naują PIL nuotrauką su tekstu.
    """
    img = image.copy().convert("RGBA")
    img_w, img_h = img.size

    # Nustatyk zoną
    if zone is None:
        zone = find_empty_zone(image)
        # Jei kategorija žinoma ir zona neatrodo gera — naudok default
        if category in DEFAULT_POSITIONS:
            variance = float(np.var(np.array(image.convert("L"))))
            if variance > VARIANCE_THRESHOLD * 3:
                zone = DEFAULT_POSITIONS[category]

    y_start, y_end = _get_text_position(zone, img_h)

    # Taikyk vartotojo offset
    y_start = max(0, min(y_start + offset_y, img_h - int(img_h * OVERLAY_HEIGHT)))
    y_end   = y_start + int(img_h * OVERLAY_HEIGHT)
    x_start = max(0, min(offset_x, img_w - 100))

    # Sukurk overlay sluoksnį
    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    alpha = int(255 * OVERLAY_OPACITY)
    draw_overlay.rectangle(
        [(x_start, y_start), (img_w, y_end)],
        fill=(*OVERLAY_COLOR, alpha)
    )

    # Sujunk overlay su nuotrauka
    img = PILImage.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Šriftai
    font_hook = _load_font(FONT_SIZE_HOOK)
    font_body = _load_font(FONT_SIZE_BODY)
    font_cta  = _load_font(FONT_SIZE_CTA)

    # Teksto pozicijos
    x = x_start + TEXT_PADDING
    y = y_start + TEXT_PADDING

    # Hook
    if hook:
        draw.text((x, y), hook, font=font_hook, fill=TEXT_COLOR)
        y += FONT_SIZE_HOOK + 8

    # Body
    if body:
        # Laužyk ilgą tekstą
        wrapped = _wrap_text(body, font_body, img_w - x - TEXT_PADDING)
        for line in wrapped:
            draw.text((x, y), line, font=font_body, fill=TEXT_COLOR)
            y += FONT_SIZE_BODY + 4

    # CTA
    if cta:
        y += 6
        draw.text((x, y), f"→ {cta}", font=font_cta, fill=(255, 220, 50))  # geltona CTA

    return img.convert("RGB")


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Laužo ilgą tekstą į eilutes pagal maksimalų plotį."""
    words = text.split()
    lines, current_line = [], ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        try:
            bbox = font.getbbox(test_line)
            text_width = bbox[2]
        except AttributeError:
            # Default font neturi getbbox — naudok getlength arba palik be laužymo
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
) -> str:
    """
    Įkelia nuotrauką, uždeda tekstą ir išsaugo.
    Grąžina output_path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Nuotrauka nerasta: {image_path}")

    image = PILImage.open(image_path).convert("RGB")
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path, quality=95)
    print(f"✅ Nuotrauka išsaugota: {output_path}")
    return output_path