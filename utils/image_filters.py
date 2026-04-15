from PIL import Image as PILImage, ImageEnhance, ImageFilter
import numpy as np
from utils.constants import AVAILABLE_FILTERS


def apply_filter(image: PILImage.Image, filter_name: str) -> PILImage.Image:
    """
    Pritaiko pasirinktą filtrą ant nuotraukos.
    Grąžina naują PIL nuotrauką.
    """
    if filter_name not in AVAILABLE_FILTERS:
        raise ValueError(f"Nežinomas filtras: '{filter_name}'. Galimi: {AVAILABLE_FILTERS}")

    if filter_name == "original":
        return image.copy()
    elif filter_name == "warm":
        return _apply_warm(image)
    elif filter_name == "cool":
        return _apply_cool(image)
    elif filter_name == "bw":
        return _apply_bw(image)
    elif filter_name == "vivid":
        return _apply_vivid(image)
    elif filter_name == "soft":
        return _apply_soft(image)
    elif filter_name == "vintage":
        return _apply_vintage(image)
    else:
        raise ValueError(f"Filtras '{{filter_name}}' neimplementuotas!")


def _apply_warm(image: PILImage.Image) -> PILImage.Image:
    """Šiltas filtras — padidina raudonus ir geltonus tonus."""
    img = image.convert("RGB")
    r, g, b = img.split()                                                           #3 kanalai r raudona, g zalia, b melyna isskaido nuotrauka i tris atskirus kanalus

    # Padidink raudoną ir žalią, sumažink mėlyną
    r = r.point(lambda i: min(255, int(i * 1.15)))                                  #raudonos spalvos info. kiekvienam pixeliui rahdname kanale padidina reiksmes 15 proc
    g = g.point(lambda i: min(255, int(i * 1.05)))                                  #zalios spalvos info
    b = b.point(lambda i: int(i * 0.85))                                            #melynos spalvos info. melyba -15 proc

    img = PILImage.merge("RGB", (r, g, b))
    # Šiek tiek padidink sodrumą
    img = ImageEnhance.Color(img).enhance(1.2)                                      #enchas yra multiplikatorius, kuris padidina efekta
    return img


def _apply_cool(image: PILImage.Image) -> PILImage.Image:
    """Šaltas filtras — padidina mėlynus tonus."""
    img = image.convert("RGB")
    r, g, b = img.split()

    r = r.point(lambda i: int(i * 0.85))
    g = g.point(lambda i: int(i * 0.95))
    b = b.point(lambda i: min(255, int(i * 1.15)))                                  #apsaugo nuo virsijimo. pikselio maksimali reiksme yra 255

    img = PILImage.merge("RGB", (r, g, b))
    img = ImageEnhance.Color(img).enhance(1.1)
    return img


def _apply_bw(image: PILImage.Image) -> PILImage.Image:
    """Juodai balta — su kontrasto pagerinimu."""
    img = image.convert("L")               # pilka spalva
    img = ImageEnhance.Contrast(img).enhance(1.3)
    return img.convert("RGB")              # grąžink į RGB


def _apply_vivid(image: PILImage.Image) -> PILImage.Image:
    """Ryškus filtras — padidina sodrumą ir kontrastą."""
    img = image.convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.5)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


def _apply_soft(image: PILImage.Image) -> PILImage.Image:
    """Minkštas filtras — švelnus blur ir sumažintas kontrastas."""
    img = image.convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = ImageEnhance.Contrast(img).enhance(0.9)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img


def _apply_vintage(image: PILImage.Image) -> PILImage.Image:
    """Vintage filtras — sepia efektas su vignette."""
    img = image.convert("RGB")

    # Sepia efektas
    img_array = np.array(img, dtype=np.float64)
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    new_r = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
    new_g = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
    new_b = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)

    sepia = np.stack([new_r, new_g, new_b], axis=2).astype(np.uint8)
    img = PILImage.fromarray(sepia)

    # Vignette efektas
    img = _add_vignette(img, strength=0.5)

    # Sumažink sodrumą
    img = ImageEnhance.Color(img).enhance(0.8)
    return img


def _add_vignette(image: PILImage.Image, strength: float = 0.5) -> PILImage.Image:
    """
    Prideda vignette efektą — tamsesni kampai.
    strength: 0.0 (jokio efekto) → 1.0 (stiprus efektas)
    """
    img = image.convert("RGB")
    w, h = img.size

    # Sukurk vignette mask
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    dist = dist / dist.max()

    # Vignette: centre ryški, kampai tamsūs
    vignette = 1 - dist * strength
    vignette = np.clip(vignette, 0, 1)

    img_array = np.array(img, dtype=np.float64)
    for channel in range(3):
        img_array[:, :, channel] *= vignette

    return PILImage.fromarray(img_array.astype(np.uint8))