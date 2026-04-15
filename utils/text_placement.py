import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps                       #PILImage — nuotraukos atidarymas, keitimas, išsaugojimas| ImageDraw — piešimas ant nuotraukos (overlay juosta, tekstas) |  ImageFont — šriftų įkėlimas | ImageOps — EXIF orientacijos taisymas
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
 
# ── Overlay nustatymai ─────────────────────────────────────────  overley yra pusiau permatomas sluoksnis uzdedamas ant nuotraukos, kad tekstas butu geriau matomas
OVERLAY_OPACITY = 0.62
OVERLAY_HEIGHT  = 0.32
OVERLAY_COLOR   = (0, 0, 0)
 
# ── Teksto nustatymai ────────────────────────────────────────── spalvos ir tarpas tarp teksto ir overlay krasto
TEXT_COLOR      = (255, 255, 255)   #balta
TEXT_COLOR_BODY = (220, 220, 220)   # šviesiai pilka body tekstui — sukuria hierarchiją
TEXT_PADDING    = 32
 
FONT_SIZE_HOOK = 48                     #sriftu dydziai
FONT_SIZE_BODY = 32
FONT_SIZE_CTA  = 32
 
# ── Šriftų keliai ─────────────────────────────────────────────
FONT_PATHS_SERIF = [                                    # hook — elegantiškesnis serif šriftas
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/georgia.ttf",
]
FONT_PATHS_REGULAR = [                                  # body — švarus regular šriftas
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]
FONT_PATHS_BOLD = [                                     # cta — ryškus bold šriftas
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
]
 
# ── Teksto ilgio ribos (simboliais) ───────────────────────────
MAX_HOOK_CHARS = 50
MAX_BODY_CHARS = 120
MAX_CTA_CHARS  = 40
 
# ── Eilučių ribos ─────────────────────────────────────────────
MAX_BODY_LINES_STORY    = 2
MAX_BODY_LINES_CAROUSEL = 3
 
 
def _fix_exif_orientation(image: PILImage.Image) -> PILImage.Image:                          #kai fotkini telefonu, nuotrauka kartais issaugoma pasukta, pvz laikomas telefonas vertikaliai, bet nuotrauka issaugoma horizontaliai. EXIF duomenyse buna irasyta si nuotrauka pasukta 90 laipsniu.
    """
    Taiso nuotraukos orientaciją pagal EXIF duomenis.
    Naudoja ImageOps.exif_transpose() — patikimiau nei rankinis apdorojimas.
    SVARBU: kviesti PRIEŠ convert("RGB").
    """
    try:
        return ImageOps.exif_transpose(image)                                                #exif_transponse perskaito informacija ir istaiso orientacija pris apdorojima, kad tekstas nebutu uzdetas ant pasuktos nuotraukos
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
    return truncated.rstrip(".,!?—–") + "…"                                                 #rstrip pasalina skyrybos zenklus
 
 
def _load_font_from_list(paths: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Bando įkelti šriftą iš kelių kelių, grįžta prie default jei nepavyksta."""
    for path in paths:
        if os.path.exists(path):                                                            #turi path parametrus, tai gali dirbti su bet kuriuo sarasu. konkrecios konstantos perduodamos per fukcijas _load_font_hook ir kitas
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()                                                         #kol naudoju windows, default niekada nepnaudos :)
 
 
def _load_font_hook(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Serif šriftas hook tekstui — elegantiškesnis."""
    return _load_font_from_list(FONT_PATHS_SERIF, size)
 
 
def _load_font_body(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Regular šriftas body tekstui — lengvai skaitomas."""
    return _load_font_from_list(FONT_PATHS_REGULAR, size)
 
 
def _load_font_cta(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Bold šriftas CTA tekstui — išsiskiriantis."""
    return _load_font_from_list(FONT_PATHS_BOLD, size)
 
 
def find_empty_zone(image: PILImage.Image) -> str:
    """
    Ieško tuščiausios zonos nuotraukoje.
    Padalija į tris zonas: top / middle / bottom.
    Grąžina zoną su mažiausia pikselių dispersija —
    mažesnė dispersija = vienodesnė spalva = geriau skaitomas tekstas.
    """
    img_array = np.array(image.convert("L"))
    h, _ = img_array.shape
 
    zones = {                                                                                   #zonu padalinimas
        "top":    img_array[:h // 3, :],
        "middle": img_array[h // 3:2 * h // 3, :],
        "bottom": img_array[2 * h // 3:, :],
    }
 
    variances = {zone: float(np.var(data)) for zone, data in zones.items()}                     #kiekvienai zonai apskaiciuojama dispersija np.var() apsaiciuoja dispersija
    return min(variances, key=variances.get)                                                    #grazina zona su maziausia dispersija
 
 
def _get_overlay_y(zone: str, img_h: int) -> tuple[int, int]:
    """
    Apskaičiuoja overlay y koordinates pagal zoną.
    Grąžina (y_start, y_end).
    """
    overlay_h = int(img_h * OVERLAY_HEIGHT)                                                     #apskaiciuoja overlay auksti pikseliais
 
    if zone == "top":
        return 0, overlay_h
    elif zone == "bottom":
        return img_h - overlay_h, img_h
    else:  # middle
        mid  = img_h // 2
        half = overlay_h // 2
        return mid - half, mid + half                                                           #grazina du skaicius, kur overlay prasideda ir baigiasi
 
 
def _clamp_overlay(y_start: int, img_h: int) -> tuple[int, int]:                                 #si funkcija uztikrina, kad overlay neisliptu uz nuotraukos ribu, kai vartotojas stumdo teksta
    """Užtikrina, kad overlay neišliptų už nuotraukos ribų."""
    overlay_h = int(img_h * OVERLAY_HEIGHT)
    y_start   = max(0, min(y_start, img_h - overlay_h))
    y_end     = y_start + overlay_h
    return y_start, y_end
 
 
def _draw_gradient_overlay(                                                                             #piesia gradientini overlay eilute pe eilutes
    img: PILImage.Image,
    y_start: int,
    y_end: int,
    zone: str,
) -> PILImage.Image:
    """
    Piešia gradientinį overlay vietoj plokščios juodos juostos.
    Gradientas eina iš permatomo į juodą — atrodo natūraliau nei stora juoda juosta.
    Kryptis priklauso nuo zonos: bottom → iš viršaus į apačią,
    top → iš apačios į viršų, middle → iš abiejų pusių į centrą.
    """
    overlay   = PILImage.new("RGBA", img.size, (0, 0, 0, 0))                                                    #RGB + Alpja (permatoma) sukuria nauja tuscia sluoksni tokiopaties dudzio kaip nuotrauka. reiskia juoda bet visiskai permatoma
    draw      = ImageDraw.Draw(overlay)                                                                         #sukuria piesimo iranki ant sluoksnio
    overlay_h = y_end - y_start                                                                                 #apskaiciuoja overley auksti pikseliais
    max_alpha = int(255 * OVERLAY_OPACITY)                                                                      #tai maksimalaus tamsumo lygis, kuri overlay gali pasiekti
 
    for i in range(overlay_h):
        if zone == "bottom":
            t = i / overlay_h           # permatoma viršuje → tamsu apačioje
        elif zone == "top":
            t = 1 - (i / overlay_h)     # tamsu viršuje → permatoma apačioje
        else:                           # middle
            t = 1 - abs(i / overlay_h - 0.5) * 2   # tamsu centre, permatoma kraštuose
 
        alpha = int(max_alpha * t)                                                                              #apskaiciuoja konkrecis eilutes alpha
        draw.line(                                                                                              #nupiesia viena horizontalia juoda linija per visanuotraukos ploti
            [(0, y_start + i), (img.size[0], y_start + i)],
            fill=(0, 0, 0, alpha),
        )
 
    return PILImage.alpha_composite(img, overlay)                                                               #sujungia originalia nuotrauka su gradientu i viena galutini vaizda
 
 
def _draw_text_with_shadow(
    draw: ImageDraw.Draw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple,
    shadow_offset: int = 2,
    shadow_alpha: int = 140,
) -> None:
    """
    Rašo tekstą su šešėliu — geriau išsiskiria ant bet kokio fono.
    Šešėlis — tamsus, šiek tiek pastumtas į apačią dešinę.
    """
    x, y = pos                                                                                                    #"ispakuoja" pozicija x ir y
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, shadow_alpha))              #nupiesia seseli
    draw.text((x, y), text, font=font, fill=fill)                                                                 #ant seselio nupiesia tikraji teksta
 
 
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
    Uždeda tekstą ant nuotraukos su gradientinio overlay.
 
    Story ir carousel logika skiriasi:
    - Story:    hook + max 2 body eilutės + CTA
    - Carousel: hook + max 3 body eilutės (CTA tik paskutinėje skaidrėje)
 
    Overlay — gradientinis, per visą nuotraukos plotį.
    Hook — serif šriftas su šešėliu.
    Body — regular šriftas, šviesiai pilkas.
    CTA — bold šriftas, geltona spalva su šešėliu.
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
 
    # 3. Gradientinis overlay vietoj plokščios juostos
    img  = _draw_gradient_overlay(img, y_start, y_end, zone)
    draw = ImageDraw.Draw(img)
 
    # 4. Šriftai — proporcingi nuotraukos dydžiui, skirtingi kiekvienam tekstui
    scale = min(img_w, img_h) / 800
    scale = max(0.6, min(scale, 1.8))
 
    font_hook = _load_font_hook(int(FONT_SIZE_HOOK * scale))   # serif
    font_body = _load_font_body(int(FONT_SIZE_BODY * scale))   # regular
    font_cta  = _load_font_cta(int(FONT_SIZE_CTA  * scale))    # bold
 
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
 
    # 6. Hook — serif šriftas, šešėlis, balta spalva
    if hook:
        hook_short = _truncate(hook, MAX_HOOK_CHARS)
        for line in _wrap_text(hook_short, font_hook, max_text_w):
            if y + line_h_hook > text_bottom:
                break
            _draw_text_with_shadow(draw, (text_x, y), line, font_hook, TEXT_COLOR)
            y += line_h_hook
        y += 12  # didesnis tarpas po hook
 
    # 7. Body — regular šriftas, šviesiai pilka spalva
    if body and y < text_bottom:
        body_short   = _truncate(body, MAX_BODY_CHARS)
        max_lines    = MAX_BODY_LINES_STORY if post_type == "story" else MAX_BODY_LINES_CAROUSEL
        wrapped_body = _wrap_text(body_short, font_body, max_text_w)[:max_lines]
        for line in wrapped_body:
            if y + line_h_body > text_bottom:
                break
            draw.text((text_x, y), line, font=font_body, fill=TEXT_COLOR_BODY)
            y += line_h_body
 
    # 8. CTA — bold šriftas, geltona, šešėlis, visada apačioje
    if cta:
        cta_short = _truncate(cta, MAX_CTA_CHARS)
        cta_y     = safe_bottom - line_h_cta
        if cta_y > y_start + TEXT_PADDING:
            _draw_text_with_shadow(
                draw,
                (text_x, cta_y),
                f"→ {cta_short}",
                font_cta,
                (255, 215, 50),
                shadow_offset=2,
                shadow_alpha=120,
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
    print(f"Nuotrauka išsaugota: {output_path}")
    return output_path