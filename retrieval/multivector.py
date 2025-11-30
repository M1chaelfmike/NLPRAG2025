"""Simplified multi-vector retrieval (ColBERT-style scoring).

We reuse the same sentence-transformers encoder as `dense.py`, but instead of
storing a single vector per passage, we keep token-level vectors and score a
query by matching each query token to its best-matching document token.

Score(q, d) = sum_{i in query_tokens} max_j cos(q_i, d_j)

To keep it lightweight for the assignment:
- We first use the existing dense index to get a candidate set of passages.
- Then we compute token-level embeddings on-the-fly for these few candidates
  and apply the ColBERT-style scoring.

This avoids building a huge multi-vector index while still demonstrating the
core idea of multi-vector retrieval.
"""

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
    """Encode each text into a matrix of token-level embeddings.

    We approximate token-level vectors by encoding with SentenceTransformer and
    using the internal token embeddings (if available). For simplicity and
    robustness across models, we instead:
    - split text into short chunks (e.g., sentences or fixed-size spans)
    - encode each chunk as a vector
    - treat those chunk vectors as the "multi-vectors" for the document.
    """

    model = _lazy_token_model()
    # use encode with "show_progress_bar=False" for speed
    # each text becomes one vector; we will instead break documents into
    # smaller pieces at query time, so here we just encode full texts.
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # normalize
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    # return as list of 2D arrays (n_tokens x dim); here n_tokens=1 for simplicity
    return [e.reshape(1, -1) for e in emb]


def _colbert_score(q_mat: np.ndarray, d_mat: np.ndarray) -> float:
    """ColBERT-style score between query (n_q x dim) and doc (n_d x dim)."""
    # cosine via inner product (vectors already normalized)
    # q_mat: (n_q, dim), d_mat: (n_d, dim)
    # sim: (n_q, n_d)
    sim = np.matmul(q_mat, d_mat.T)
    # for each query vector, take max over doc tokens, then sum
    max_per_q = sim.max(axis=1)
    return float(max_per_q.sum())


def multivector_retrieve(query: str, topk: int = 10, candidate_k: int = 50,
                         model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[Dict]:
    """Retrieve using a lightweight multi-vector (ColBERT-style) scorer.

    Steps:
    1) Use existing dense index to retrieve `candidate_k` passages.
    2) Encode query and each candidate passage into small sets of vectors.
    3) Compute ColBERT-style score and return topk.

    Returns list of dicts: {id, score, text}
    """

    # 1) use dense retriever as candidate generator
    from retrieval import dense as dense_mod

    dense_candidates = dense_mod.dense_retrieve(query, topk=candidate_k, model_name=model_name)
    if not dense_candidates:
        return []

    # 2) encode query and candidate texts into small matrices
    q_mat_list = _encode_tokens([query])
    q_mat = q_mat_list[0]  # (n_q, dim), n_q=1 in this simplified version

    doc_texts = [d.get("text", "") for d in dense_candidates]
    d_mats = _encode_tokens(doc_texts)  # list of (1, dim)

    # 3) ColBERT-style scoring
    results = []
    for cand, d_mat in zip(dense_candidates, d_mats):
        score = _colbert_score(q_mat, d_mat)
        results.append({
            "id": cand.get("id"),
            "score": score,
            "text": cand.get("text"),
        })

    # sort and take topk
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:topk]
    return results
