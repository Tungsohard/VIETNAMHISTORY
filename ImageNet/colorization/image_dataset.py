"""Metadata rules shared by historical still-image collection and evaluation."""
import re


MANIPULATION_TERMS = (
    "colorized", "colourized", "colorised", "colourised", "colorization",
    "colourization", "recolored", "recoloured", "ai generated", "ai-generated",
    "generative ai", "synthetic image",
)

CONTENT_RULES = (
    ("people", ("people", "portrait", "person", "soldier", "civilian", "woman", "women",
                "man", "men", "child", "children", "family", "crowd")),
    ("clothing_uniform", ("clothing", "uniform", "costume", "ao dai", "dress", "helmet")),
    ("transport", ("transport", "vehicle", "car", "truck", "bus", "train", "aircraft",
                   "helicopter", "boat", "ship", "motorcycle", "bicycle")),
    ("architecture_street", ("building", "architecture", "street", "city", "village",
                             "house", "temple", "pagoda", "church", "bridge")),
    ("nature_rural", ("nature", "landscape", "forest", "river", "mountain", "field",
                      "farm", "rural", "coast")),
    ("indoor_low_light", ("indoor", "interior", "night", "low light", "dark")),
    ("food_market", ("food", "market", "meal", "fruit", "vegetable")),
)


def normalized_metadata_text(row):
    return " ".join(str(row.get(key, "")) for key in (
        "title", "description", "credit", "source_category", "author"
    )).lower()


def manipulation_reason(row):
    """Return the suspicious metadata term, or None for an apparently authentic photo."""
    text = normalized_metadata_text(row)
    return next((term for term in MANIPULATION_TERMS if term in text), None)


def infer_content_category(row):
    text = normalized_metadata_text(row)
    scores = [(sum(term in text for term in terms), category)
              for category, terms in CONTENT_RULES]
    score, category = max(scores)
    return category if score else "other"


def infer_collection_id(row):
    """Derive a conservative source group so one collection cannot cross data splits."""
    author = re.sub(r"[^a-z0-9]+", "-", str(row.get("author", "")).lower()).strip("-")
    source = re.sub(r"[^a-z0-9]+", "-", str(row.get("source_category", "")).lower()).strip("-")
    if author and author not in {"unknown", "anonymous", "various", "n-a"}:
        return f"author-{author[:80]}"
    return f"category-{source[:100] or 'unknown'}"


def dhash_distance(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()


def has_near_duplicate(candidate, existing, max_distance=2):
    return any(dhash_distance(candidate, value) <= max_distance for value in existing)
