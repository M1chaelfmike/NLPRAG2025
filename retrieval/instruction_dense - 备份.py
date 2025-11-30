import os
import json
import pathlib
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# Lazy globals
_MODEL = None
_INDEX = None
_PASSAGES = None  # list of dicts: {pid, doc_id, text}

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMB_DIR = PROJECT_ROOT / "hf_cache" / "embeddings" / "dense_instruction"
EMB_DIR.mkdir(parents=True, exist_ok=True)

VECTORS_NPY = EMB_DIR / "passage_vectors.npy"
PASSAGE_MAP = EMB_DIR / "passage_map.json"
FAISS_INDEX = EMB_DIR / "dense_faiss.idx"


def _lazy_model(model_name: str = "intfloat/e5-base-v2"):
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers is required for instruction dense retrieval: pip install sentence-transformers")
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def _ensure_passages():
    """Reuse BM25 corpus and split into passages (same as dense retriever)."""
    global _PASSAGES
    if _PASSAGES is not None:
        return
    from retrieval import bm25 as bm25_mod
    if not getattr(bm25_mod, "corpus", None):
        try:
            idx_path = None
            if hasattr(bm25_mod, "HF_CACHE_DIR"):
                idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl")
            bm25_mod.init(index_path=idx_path)
        except Exception:
            pass

    corpus = getattr(bm25_mod, "corpus", None)
    ids = getattr(bm25_mod, "ids", None)
    if not corpus:
        raise RuntimeError("No corpus available to build dense passages. Ensure bm25 corpus is initialized.")

    passages = []
    for doc_idx, text in enumerate(corpus):
        doc_id = ids[doc_idx] if ids else f"doc-{doc_idx}"
        words = text.split()
        if not words:
            continue
        chunk_size = 150
        overlap = 50
        i = 0
        pidx = 0
        while i < len(words):
            chunk = words[i:i + chunk_size]
            ptext = " ".join(chunk)
            pid = f"{doc_id}__p{pidx}"
            passages.append({"pid": pid, "doc_id": doc_id, "text": ptext})
            pidx += 1
            i += chunk_size - overlap

    _PASSAGES = passages


def _encode_texts(texts: List[str], batch_size: int = 256) -> np.ndarray:
    model = _MODEL
    vectors = []
    total = len(texts)
    for i in tqdm(range(0, total, batch_size), desc="Encoding passages (instruction-dense)"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        vectors.append(emb)
    return np.vstack(vectors).astype(np.float32)


def build_index(model_name: str = "intfloat/e5-base-v2", batch_size: int = 256):
    """Build instruction-based dense passage vectors and FAISS index."""
    _lazy_model(model_name)
    _ensure_passages()
    global _PASSAGES

    # E5 类模型推荐加指令前缀
    texts = [f"passage: {p['text']}" for p in _PASSAGES]
    vectors = _encode_texts(texts, batch_size=batch_size)

    try:
        import faiss
    except Exception:
        raise ImportError("faiss is required to build index: pip install faiss-cpu (or faiss-gpu)")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    # 确保目录存在
    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用临时文件避免中文路径问题
    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile(delete=False, suffix='.idx') as tmp:
        tmp_path = tmp.name
    faiss.write_index(index, tmp_path)
    shutil.move(tmp_path, str(FAISS_INDEX))
    
    np.save(str(VECTORS_NPY), vectors)

    with open(PASSAGE_MAP, "w", encoding="utf8") as f:
        json.dump(_PASSAGES, f, ensure_ascii=False)

    print(f"Instruction-dense index built: {len(_PASSAGES)} passages, vectors saved to {VECTORS_NPY}")


def load_index(model_name: str = "intfloat/e5-base-v2"):
    """Load FAISS index, passage map and model lazily for instruction-dense retrieval."""
    global _INDEX, _PASSAGES, _MODEL
    if _INDEX is not None:
        return
    _lazy_model(model_name)
    try:
        import faiss
    except Exception:
        raise ImportError("faiss is required to load index: pip install faiss-cpu (or faiss-gpu)")

    if not os.path.exists(FAISS_INDEX):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX}; run build_index() first")

    _INDEX = faiss.read_index(str(FAISS_INDEX))
    if os.path.exists(PASSAGE_MAP):
        with open(PASSAGE_MAP, "r", encoding="utf8") as f:
            _PASSAGES = json.load(f)
    else:
        _PASSAGES = []


def ensure_index(model_name: str = "intfloat/e5-base-v2"):
    if os.path.exists(FAISS_INDEX) and os.path.exists(PASSAGE_MAP):
        return
    build_index(model_name=model_name)


def instruction_dense_retrieve(query: str, topk: int = 10, model_name: str = "intfloat/e5-base-v2") -> List[Dict]:
    """Instruction-based dense retrieval.

    Returns list of dicts: {id: doc_id, passage_id, score, text}
    """
    load_index(model_name=model_name)
    global _INDEX, _PASSAGES, _MODEL
    if _INDEX is None:
        raise RuntimeError("Index not loaded; call load_index() or ensure_index() first")

    # 对 query 加上指令前缀
    qtext = f"query: {query}"
    qvec = _MODEL.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)
    qvec = qvec.astype(np.float32)

    D, I = _INDEX.search(qvec, topk)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(_PASSAGES):
            continue
        p = _PASSAGES[idx]
        results.append({
            "id": p.get("doc_id"),
            "passage_id": p.get("pid"),
            "score": float(score),
            "text": p.get("text"),
        })
    return results
