import re
import unicodedata

# Short on purpose; "not", "no" and "cannot" stay because they flip what a policy line means.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "in", "on", "for", "and",
    "or", "i", "my", "me", "we", "our", "you", "your", "it", "its", "do", "does", "can",
    "what", "how", "which", "who", "when", "where", "there", "this", "that", "with", "as",
    "at", "by", "any", "if", "am", "get", "much", "many",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9₹]+", text.lower()) if t not in STOPWORDS]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[*_`|]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()
