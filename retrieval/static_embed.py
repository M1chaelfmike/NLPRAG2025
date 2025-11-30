import os
import json
import pickle
import pathlib
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from retrieval import bm25 as bm25_mod

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMB_DIR = PROJECT_ROOT / "hf_cache" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_DEFAULT = EMB_DIR / "glove.6B.300d.txt"
FAISS_INDEX_PATH = EMB_DIR / "faiss_index.idx"
VECTORS_NPY = EMB_DIR / "doc_vectors.npy"
DOC_MAP_JSON = EMB_DIR / "doc_id_map.json"
IDF_PKL = EMB_DIR / "idf_map.pkl"


def load_glove(glove_path: str) -> dict:
    glove = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove[word] = vec
    return glove


def ensure_glove(glove_path: str):
    """Ensure GloVe file exists at glove_path. If not, try to download via kagglehub.

    Returns the path to the glove file if available, otherwise raises FileNotFoundError with instructions.
    """
    glove_path = pathlib.Path(glove_path)
    if glove_path.exists():
        return str(glove_path)

    print(f"GloVe file not found at {glove_path}. Attempting to download via kagglehub...")
    try:
        import kagglehub
    except Exception as e:
        raise FileNotFoundError(
            f"GloVe file not found at {glove_path} and `kagglehub` is not available to download it automatically.\n"
            "Please download `glove.6B.300d.txt` manually and place it at: {glove_path} \n"
            "Or install kagglehub (pip install kagglehub) and ensure you have Kaggle credentials configured.")

    try:
        # dataset_download should return a local path where files were downloaded
        path = kagglehub.dataset_download("thanakomsn/glove6b300dtxt")
        # The dataset may contain multiple files; find glove.6B.300d.txt inside
        p = pathlib.Path(path)
        candidates = list(p.rglob("glove*.txt"))
        if not candidates:
            raise FileNotFoundError(f"Downloaded dataset but no glove txt found under {p}")
        # pick the first candidate that matches expected name, else first candidate
        chosen = None
        for c in candidates:
            if c.name == glove_path.name:
                chosen = c
                break
        if chosen is None:
            chosen = candidates[0]

        # copy to desired location
        os.replace(str(chosen), str(glove_path))
        print(f"Downloaded GloVe to {glove_path}")
        return str(glove_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download GloVe via kagglehub: {e}\nPlease download manually and place at {glove_path}.")


def ensure_index(glove_path: str = None):
    """Ensure FAISS index exists; if not, try to build it (which will download GloVe if needed).

    Returns None. Raises FileNotFoundError if building fails.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        return
    # Attempt to build index (this will call ensure_glove internally)
    try:
        build_index(glove_path=glove_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to build FAISS index: {e}")


def build_idf(corpus: List[str]) -> dict:
    from collections import Counter

    N = len(corpus)
    df = Counter()
    for text in corpus:
        toks = set(bm25_mod.preprocess(text))
        df.update(toks)
    idf = {w: float(np.log((N + 1) / (df[w] + 1))) for w in df}
    return idf


def doc_to_vec(text: str, glove: dict, idf_map: dict, dim: int) -> np.ndarray:
    toks = bm25_mod.preprocess(text)
    vec = np.zeros(dim, dtype=np.float32)
    wsum = 0.0
    for t in toks:
        if t in glove:
            w = idf_map.get(t, 1.0)
            vec += glove[t] * w
            wsum += w
    if wsum > 0:
        vec /= wsum
    return vec


def build_index(glove_path: str = None):
    """Build FAISS index over the collection using IDF-weighted GloVe vectors.
    Saves index, vectors and mappings to `hf_cache/embeddings/`.
    """
    import faiss

    glove_path = glove_path or str(GLOVE_DEFAULT)
    # ensure glove exists or attempt to download
    try:
        glove_path = ensure_glove(glove_path)
    except FileNotFoundError:
        # re-raise with same message
        raise

    print("Loading GloVe vectors (this may take a while)...")
    glove = load_glove(glove_path)
    dim = next(iter(glove.values())).shape[0]

    # Ensure BM25 module has loaded the collection (corpus/ids)
    if not getattr(bm25_mod, "corpus", None) or not getattr(bm25_mod, "ids", None):
        # try to init BM25 (will load dataset and optionally load/save index if configured)
        try:
            bm25_idx_path = None
            if hasattr(bm25_mod, "HF_CACHE_DIR"):
                bm25_idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl")
            bm25_mod.init(index_path=bm25_idx_path)
        except Exception as e:
            print(f"Warning: failed to initialize bm25 module before building static index: {e}")

    corpus = bm25_mod.corpus
    ids = bm25_mod.ids

    print("Computing IDF map...")
    idf_map = build_idf(corpus)
    with open(IDF_PKL, "wb") as f:
        pickle.dump(idf_map, f)

    vectors = np.zeros((len(corpus), dim), dtype=np.float32)
    print("Computing document vectors...")
    for i, text in enumerate(tqdm(corpus)):
        vectors[i] = doc_to_vec(text, glove, idf_map, dim)

    # normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    # build index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(str(VECTORS_NPY), vectors)

    # save id mapping (list index -> doc id)
    with open(DOC_MAP_JSON, "w", encoding="utf8") as f:
        json.dump(ids, f, ensure_ascii=False)

    print("FAISS index built and saved.")


_INDEX = None
_VECTORS = None
_DOC_IDS = None
_IDF = None
_GLOVE = None


def load_index(glove_path: str = None):
    """Load FAISS index and associated files into memory (lazy)."""
    global _INDEX, _VECTORS, _DOC_IDS, _IDF, _GLOVE
    if _INDEX is not None:
        return
    import faiss
    # ensure index exists (will attempt to build if missing)
    ensure_index(glove_path=glove_path)

    _INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    _VECTORS = np.load(str(VECTORS_NPY)) if os.path.exists(VECTORS_NPY) else None
    with open(DOC_MAP_JSON, "r", encoding="utf8") as f:
        _DOC_IDS = json.load(f)
    with open(IDF_PKL, "rb") as f:
        _IDF = pickle.load(f)

    glove_path = glove_path or str(GLOVE_DEFAULT)
    try:
        glove_path = ensure_glove(glove_path)
        _GLOVE = load_glove(glove_path)
    except FileNotFoundError:
        # If glove not available, continue with empty glove dict (query vectors will be zeros)
        print(f"Warning: GloVe not available at {glove_path}; static embedding queries will be empty.")
        _GLOVE = {}
    # Ensure BM25 corpus is available for mapping index positions -> texts
    if not getattr(bm25_mod, "corpus", None):
        try:
            bm25_idx_path = None
            if hasattr(bm25_mod, "HF_CACHE_DIR"):
                bm25_idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl")
            bm25_mod.init(index_path=bm25_idx_path)
        except Exception as e:
            print(f"Warning: failed to initialize bm25 module while loading static index: {e}")


def query_to_vec(query: str) -> np.ndarray:
    global _GLOVE, _IDF
    if _GLOVE is None or _IDF is None:
        load_index()
    dim = next(iter(_GLOVE.values())).shape[0] if _GLOVE else (_VECTORS.shape[1] if _VECTORS is not None else 300)
    vec = doc_to_vec(query, _GLOVE, _IDF, dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_retrieve(query: str, topk: int = 10) -> List[dict]:
    """Return topk documents using static-embedding FAISS index.

    Returns list of dicts with keys: id, score, text
    """
    try:
        load_index()
    except FileNotFoundError as e:
        # Fall back to BM25 if static index is not available / cannot be built
        print(f"Static index not available: {e}. Falling back to BM25.")
        return bm25_mod.bm25_retrieve(query, topk=topk)

    vec = query_to_vec(query).astype(np.float32)
    D, I = _INDEX.search(np.expand_dims(vec, 0), topk)
    ids = _DOC_IDS
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(ids):
            continue
        results.append({
            "id": ids[idx],
            "score": float(score),
            "text": bm25_mod.corpus[idx]
        })
    return results
