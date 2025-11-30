import os
import pathlib
import pickle
import string
from typing import Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # current project
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"

# create cache folders and set env vars early
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["HF_METRICS_CACHE"] = str(HF_CACHE_DIR / "metrics")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

# lazy imports (avoid heavy work at import time)
_bm25 = None
_corpus = None
_ids = None

# Public aliases expected by other modules (static_embed used module-level names)
corpus = None
ids = None


def _lazy_imports():

    global load_dataset, BM25Okapi, nltk, word_tokenize
    try:
        from datasets import load_dataset
        from rank_bm25 import BM25Okapi
        import nltk
        from nltk.tokenize import word_tokenize
    except Exception as e:

        raise ImportError(f"Failed to import BM25 dependencies: {e}")

    globals()["load_dataset"] = load_dataset
    globals()["BM25Okapi"] = BM25Okapi
    globals()["nltk"] = nltk
    globals()["word_tokenize"] = word_tokenize



def preprocess(text: str):

    if "word_tokenize" not in globals():
        _lazy_imports()
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return globals()["word_tokenize"](text)


def _ensure_init(index_path: Optional[str] = None):

    global _bm25, _corpus, _ids
    if _bm25 is not None:
        return

    _lazy_imports()


    if index_path and os.path.exists(index_path):
        try:
            load_index(index_path)
            return
        except Exception:
            print(f"Failed to load BM25 index from {index_path}, will rebuild.")

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt')

    print("Loading HQ-small Collection...")
    dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small", split="collection")
    _corpus = [doc["text"] for doc in dataset]
    _ids = [doc["id"] for doc in dataset]

    global corpus, ids
    corpus = _corpus
    ids = _ids
    print(f"Loaded {len(_corpus)} collection documents.")

    print("Building BM25 index... (this may take some time)")
    corpus_tokenized = [preprocess(t) for t in _corpus]
    _bm25 = BM25Okapi(corpus_tokenized)
    print("BM25 ready.")



def save_index(path: str):
    global _bm25, _corpus, _ids
    if _bm25 is None:
        raise RuntimeError("BM25 is not initialized; cannot save.")
    data = {
        "corpus": _corpus,
        "ids": _ids,
        "bm25": _bm25,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_index(path: str):
    global _bm25, _corpus, _ids
    with open(path, "rb") as f:
        data = pickle.load(f)
    _corpus = data["corpus"]
    _ids = data["ids"]
    global corpus, ids
    corpus = _corpus
    ids = _ids
    _bm25 = data["bm25"]


def init(index_path: Optional[str] = None, force_rebuild: bool = False):
    if index_path and os.path.exists(index_path) and not force_rebuild:
        try:
            load_index(index_path)
            print(f"Loaded BM25 index from {index_path}")
            return
        except Exception:
            print(f"Failed loading BM25 index from {index_path}; rebuilding...")
    _ensure_init(index_path=index_path)
    if index_path:
        try:
            save_index(index_path)
            print(f"Saved BM25 index to {index_path}")
        except Exception as e:
            print(f"Warning: failed to save BM25 index to {index_path}: {e}")


def is_initialized() -> bool:
    return _bm25 is not None


def bm25_retrieve(query: str, topk: int = 10):
    _ensure_init()
    tokens = preprocess(query)
    scores = _bm25.get_scores(tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]

    results = []
    for idx, score in ranked:
        results.append({
            "id": _ids[idx],
            "score": float(score),
            "text": _corpus[idx]
        })
    return results

