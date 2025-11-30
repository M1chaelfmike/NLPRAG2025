

import pathlib
from typing import List, Dict

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

_TOKEN_MODEL = None


def _lazy_token_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    global _TOKEN_MODEL
    if _TOKEN_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for multi-vector retrieval: pip install sentence-transformers"
            )
        _TOKEN_MODEL = SentenceTransformer(model_name)
    return _TOKEN_MODEL


def _encode_tokens(texts: List[str]) -> List[np.ndarray]:
    model = _lazy_token_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    return [e.reshape(1, -1) for e in emb]


def _colbert_score(q_mat: np.ndarray, d_mat: np.ndarray) -> float:
    sim = np.matmul(q_mat, d_mat.T)

    max_per_q = sim.max(axis=1)
    return float(max_per_q.sum())


def multivector_retrieve(query: str, topk: int = 10, candidate_k: int = 50,
                         model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[Dict]:

    from retrieval import dense as dense_mod

    dense_candidates = dense_mod.dense_retrieve(query, topk=candidate_k, model_name=model_name)
    if not dense_candidates:
        return []

    q_mat_list = _encode_tokens([query])
    q_mat = q_mat_list[0]

    doc_texts = [d.get("text", "") for d in dense_candidates]
    d_mats = _encode_tokens(doc_texts)

    results = []
    for cand, d_mat in zip(dense_candidates, d_mats):
        score = _colbert_score(q_mat, d_mat)
        results.append({
            "id": cand.get("id"),
            "score": score,
            "text": cand.get("text"),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:topk]
    return results
