import os
import json
import pathlib
import pickle
from typing import List, Optional, Dict

import numpy as np
import math
from tqdm import tqdm

# lazy imports for heavy libs
_MODEL = None
_INDEX = None
_PASSAGES = None  # list of dicts: {pid, doc_id, text}

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMB_DIR = PROJECT_ROOT / "hf_cache" / "embeddings" / "dense"
EMB_DIR.mkdir(parents=True, exist_ok=True)

VECTORS_NPY = EMB_DIR / "passage_vectors.npy"
PASSAGE_MAP = EMB_DIR / "passage_map.json"
FAISS_INDEX = EMB_DIR / "dense_faiss.idx"


def _lazy_model(model_name: str = "all-mpnet-base-v2"):
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers is required for dense retrieval: pip install sentence-transformers")
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def _ensure_passages():
    """Ensure bm25 corpus is available and split into passages."""
    global _PASSAGES
    if _PASSAGES is not None:
        return
    # lazy import bm25 module to get corpus
    from retrieval import bm25 as bm25_mod
    if not getattr(bm25_mod, "corpus", None):
        # initialize bm25 (will load dataset)
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
    # simple whitespace-based passage splitter
    for doc_idx, text in enumerate(corpus):
        doc_id = ids[doc_idx] if ids else f"doc-{doc_idx}"
        words = text.split()
        if not words:
            continue
        # chunk params
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


def build_index(model_name: str = "sentence-transformers/all-mpnet-base-v2", batch_size: int = 256):
    """Build dense passage vectors and FAISS index.

    Saves `passage_vectors.npy`, `passage_map.json` and FAISS index file.
    """
    _lazy_model(model_name)
    _ensure_passages()
    global _PASSAGES

    texts = [p["text"] for p in _PASSAGES]
    # encode in batches with a single overall tqdm and show remaining batches
    model = _MODEL
    vectors = []
    total_batches = math.ceil(len(texts) / batch_size) if batch_size > 0 else 0
    pbar = tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Encoding passages")
    for idx, i in enumerate(pbar):
        # update description to show remaining batches
        remaining = max(total_batches - idx - 1, 0)
        try:
            pbar.set_description(f"Encoding passages (remaining {remaining})")
        except Exception:
            pass
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.append(emb)
    pbar.close()
    vectors = np.vstack(vectors).astype(np.float32)

    # normalize to unit vectors for cosine via inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    # build faiss index
    try:
        import faiss
    except Exception:
        raise ImportError("faiss is required to build index: pip install faiss-cpu (or faiss-gpu)")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    # 确保目录存在
    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    
    # faiss 对中文路径支持不佳，直接保存到项目内英文子目录
    # 使用相对于C盘根目录的英文路径
    import shutil
    import uuid
    ascii_temp_dir = pathlib.Path("C:/faiss_temp_nlp")
    ascii_temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_idx = ascii_temp_dir / "dense_faiss.idx"
    tmp_npy = ascii_temp_dir / "passage_vectors.npy"
    tmp_map = ascii_temp_dir / "passage_map.json"
    
    faiss.write_index(index, str(tmp_idx))
    np.save(str(tmp_npy), vectors)
    
    # save map
    with open(tmp_map, "w", encoding="utf8") as f:
        json.dump(_PASSAGES, f, ensure_ascii=False)
    
    # 复制到目标位置
    shutil.copy(str(tmp_idx), str(FAISS_INDEX))
    shutil.copy(str(tmp_npy), str(VECTORS_NPY))
    shutil.copy(str(tmp_map), str(PASSAGE_MAP))
    
    # 清理临时文件
    try:
        tmp_idx.unlink()
        tmp_npy.unlink()
        tmp_map.unlink()
        ascii_temp_dir.rmdir()
    except:
        pass

    print(f"Dense index built: {len(_PASSAGES)} passages, vectors saved to {VECTORS_NPY}")


def load_index(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Load FAISS index, passage map and model lazily."""
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

    # faiss 对中文路径支持不佳，复制到纯英文临时目录读取
    import shutil
    ascii_temp_dir = pathlib.Path("C:/faiss_temp_nlp")
    ascii_temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_idx = ascii_temp_dir / "dense_faiss.idx"
    if not tmp_idx.exists():
        shutil.copy(str(FAISS_INDEX), str(tmp_idx))
    _INDEX = faiss.read_index(str(tmp_idx))
    
    if os.path.exists(PASSAGE_MAP):
        with open(PASSAGE_MAP, "r", encoding="utf8") as f:
            _PASSAGES = json.load(f)
    else:
        _PASSAGES = []


def ensure_index(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    if os.path.exists(FAISS_INDEX) and os.path.exists(PASSAGE_MAP):
        return
    build_index(model_name=model_name)


def dense_retrieve(query: str, topk: int = 10, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[Dict]:
    """Encode query and return topk passages in same format as other retrievers.

    Returns list of dicts: {id: doc_id, passage_id, score, text}
    """
    load_index(model_name=model_name)
    global _INDEX, _PASSAGES, _MODEL
    if _INDEX is None:
        raise RuntimeError("Index not loaded; call load_index() or ensure_index() first")

    qvec = _MODEL.encode([query], convert_to_numpy=True)
    qvec = qvec.astype(np.float32)
    # normalize
    qnorm = np.linalg.norm(qvec, axis=1, keepdims=True)
    qnorm[qnorm == 0] = 1.0
    qvec = qvec / qnorm

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
