import nltk

def ensure_nltk_resources():
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()