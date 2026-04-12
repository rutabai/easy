import anthropic
import json
import os
from dotenv import load_dotenv
from utils.constants import CATEGORIES, VALID_GOALS, VALID_CTA_TYPES

load_dotenv()

# Modelis
MODEL = "claude-opus-4-6"
MAX_TOKENS = 500

# CTA tipo vertimas į lietuvišką tekstą
CTA_TRANSLATIONS = {
    "visit_shop":     "Apsilankyti e-shop",
    "visit_profile":  "Apsilankyti profilyje",
    "send_message":   "Parašyti žinutę",
    "save_post":      "Išsaugoti įrašą",
    "comment":        "Komentuoti",
    "click_link":     "Spausti nuorodą",
}

# Tikslo vertimas
GOAL_TRANSLATIONS = {
    "sell":            "parduoti produktą ar paslaugą",
    "inform":          "informuoti ir šviesti auditoriją",
    "engage":          "padidinti įsitraukimą ir reakcijas",
    "brand_awareness": "didinti prekės ženklo žinomumą",
    "traffic":         "nukreipti srautą į svetainę ar profilį",
}


def _get_client() -> anthropic.Anthropic:
    """Grąžina Anthropic klientą. Sukuriamas tik kai reikia."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("❌ ANTHROPIC_API_KEY nerasta .env faile!")
    return anthropic.Anthropic(api_key=api_key)


def _build_prompt(
    category: str,
    post_type: str,
    topic: str | None,
    goal: str | None,
    cta_type: str | None,
    additional_notes: str | None,
    slide_count: int = 1,
) -> str:
    """Sukuria promptą Anthropic API."""

    goal_text = GOAL_TRANSLATIONS.get(goal, "sukurti įtraukiantį turinį") if goal else "sukurti įtraukiantį turinį"
    cta_text  = CTA_TRANSLATIONS.get(cta_type, "Sužinoti daugiau") if cta_type else "Sužinoti daugiau"
    topic_text = topic if topic else f"{category} turinys"
    notes_text = f"\nPapildoma informacija: {additional_notes}" if additional_notes else ""

    if post_type == "story":
        format_instruction = """Grąžink JSON tokiu formatu (tik JSON, be jokio papildomo teksto):
{
  "hook": "Trumpas įtraukiantis sakinys (max 10 žodžių)",
  "story": "Platesnis pasakojimas (2-3 sakiniai)",
  "cta": "Kvietimas veikti",
  "caption": "Visas tekstas kartu su emoji"
}"""
    else:
        format_instruction = f"""Tai {slide_count} skaidrių carousel. Grąžink JSON tokiu formatu (tik JSON, be jokio papildomo teksto):
{{
  "hook": "Pirmos skaidrės tekstas — kabina dėmesį (max 10 žodžių)",
  "story": "Vidurinių skaidrių turinys (2-4 sakiniai)",
  "cta": "Paskutinės skaidrės kvietimas veikti",
  "caption": "Visas posto aprašymas su hashtag ir emoji"
}}"""

    return f"""Sukurk profesionalų Instagram {post_type} tekstą lietuvių kalba.

Nuotraukos kategorija: {category}
Tema: {topic_text}
Tikslas: {goal_text}
Call to action: {cta_text}{notes_text}

Reikalavimai:
- Tonas: draugiškas, autentiškas, įtraukiantis
- Hook turi iškart patraukti dėmesį
- Story turi kurti vertę skaitytojui
- CTA turi būti aiškus ir konkretus
- Naudok emoji kur tinka

{format_instruction}"""


def generate_caption(
    category: str,
    post_type: str,
    topic: str | None = None,
    goal: str | None = None,
    cta_type: str | None = None,
    additional_notes: str | None = None,
    slide_count: int = 1,
) -> dict:
    """
    Generuoja Instagram tekstą per Anthropic API.

    Grąžina žodyną su:
    - hook
    - story
    - cta
    - caption
    """
    # Validacija
    if category not in CATEGORIES:
        raise ValueError(f"Neteisinga kategorija: {category}")
    if goal and goal not in VALID_GOALS:
        raise ValueError(f"Neteisinas tikslas: {goal}")
    if cta_type and cta_type not in VALID_CTA_TYPES:
        raise ValueError(f"Neteisingas CTA tipas: {cta_type}")

    prompt = _build_prompt(
        category=category,
        post_type=post_type,
        topic=topic,
        goal=goal,
        cta_type=cta_type,
        additional_notes=additional_notes,
        slide_count=slide_count,
    )

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_text = response.content[0].text.strip()

        # Išgauk JSON iš atsakymo
        import json
        # Pašalink galimus markdown backticks
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        result = json.loads(raw_text.strip())

        # Patikrink ar visi laukai yra
        for field in ["hook", "story", "cta", "caption"]:
            if field not in result:
                result[field] = ""

        return result

    except Exception as e:
        print(f"⚠️  Klaida generuojant tekstą: {e}")
        # Grąžink default tekstą klaidos atveju
        return {
            "hook": f"Atrask naujausią {category} turinį!",
            "story": topic or "Kažkas įdomaus laukia tavęs.",
            "cta": CTA_TRANSLATIONS.get(cta_type, "Sužinoti daugiau"),
            "caption": f"#{category} #instagram #content"
        }


def refine_caption(
    original_text: str,
    user_edit: str,
    category: str,
    post_type: str,
) -> dict:
    """
    Patobulina vartotojo redaguotą tekstą per Anthropic API.
    Išlaiko vartotojo mintį bet pagerina stilių.
    """
    prompt = f"""Vartotojas redagavo Instagram {post_type} tekstą. Patobulink jo versiją išlaikant pagrindinę mintį.

Originali AI versija:
{original_text}

Vartotojo versija:
{user_edit}

Kategorija: {category}

Reikalavimai:
- Išlaikyk vartotojo pagrindinę mintį ir toną
- Patobulink gramatiką ir stilių
- Pridėk emoji jei tinka
- Palik tekstą lietuvių kalba

Grąžink JSON (tik JSON, be papildomo teksto):
{{
  "hook": "patobulinta hook dalis",
  "story": "patobulinta story dalis",
  "cta": "patobulinta cta dalis",
  "caption": "visas patobulinas tekstas"
}}"""

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        raw_text = response.content[0].text.strip()
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        return json.loads(raw_text.strip())

    except Exception as e:
        print(f"⚠️  Klaida tobulinant tekstą: {e}")
        return {
            "hook": user_edit,
            "story": user_edit,
            "cta": "",
            "caption": user_edit
        }