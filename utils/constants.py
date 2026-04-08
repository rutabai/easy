# ──────────────────────────────────────────────────────────────
# Bendra konfigūracija — importuoti iš čia, ne dubliuoti kitur!
# ──────────────────────────────────────────────────────────────

# Nuotraukų kategorijos
CATEGORIES = ["food", "portrait", "landscape", "product", "lifestyle"]
NUM_CLASSES = len(CATEGORIES)

# Kategorijos → indeksas (modeliui)
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

# Indeksas → kategorija (spėjimui)
IDX_TO_CATEGORY = {i: cat for i, cat in enumerate(CATEGORIES)}

# Nuotraukų dydis
IMAGE_SIZE = (224, 224)

# Posto tipai
POST_TYPES = ["story", "carousel"]

# Posto tikslai
VALID_GOALS = ["sell", "inform", "engage", "brand_awareness", "traffic"]

# CTA tipai
VALID_CTA_TYPES = [
    "visit_shop",
    "visit_profile",
    "send_message",
    "save_post",
    "comment",
    "click_link",
]

# Slide tipai
SLIDE_TYPES = ["single", "hook", "story", "cta"]

# Nuotraukų limitai
MAX_UPLOADS = {"story": 1, "carousel": 10}
MIN_UPLOADS = {"story": 1, "carousel": 2}

# Duomenų padalijimas
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Modelių tipai
MODEL_TYPES = ["cnn", "vit", "knn"]