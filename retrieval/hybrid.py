"""Hybrid retriever combining BM25 and static (GloVe+FAISS) methods.

Provides two merge strategies:
- 'score': normalize each method's scores (by max) and combine via weighted sum alpha*bm25 + (1-alpha)*static
- 'rrf': Reciprocal Rank Fusion (rank-based) with k parameter

Returns list of dicts: {id, score, text, bm25_score, static_score}
"""
from typing import List, Dict, Optional
from collections import defaultdict

def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    if mx <= 0:
        return [0.0 for _ in scores]
    return [s / mx for s in scores]


def hybrid_retrieve(query: str, topk: int = 10, alpha: float = 0.5, mode: str = "score", rrf_k: int = 60) -> List[Dict]:
    """Combine BM25 and static retrieval.

    mode: 'score' (default) or 'rrf'
    alpha: weight for BM25 when using 'score' mode (0..1)
    rrf_k: constant for RRF when using 'rrf' mode
    """
    # lazy import so module import is cheap
    from retrieval import bm25 as bm25_mod
    try:
        from retrieval import static_embed as static_mod
    except Exception:
        static_mod = None

    # get candidates from both retrievers
    bm25_results = []
    try:
        bm25_results = bm25_mod.bm25_retrieve(query, topk=topk * 3)
    except Exception:
        bm25_results = []

    static_results = []
    if static_mod is not None:
        try:
            static_results = static_mod.embed_retrieve(query, topk=topk * 3)
        except Exception:
            static_results = []

    # map doc id -> info
    info = defaultdict(lambda: {"bm25_score": 0.0, "static_score": 0.0, "text": None})

    for i, d in enumerate(bm25_results):
        did = d.get("id")
        info[did]["bm25_score"] = max(info[did]["bm25_score"], d.get("score", 0.0))
        info[did]["text"] = info[did].get("text") or d.get("text")

    for i, d in enumerate(static_results):
        did = d.get("id")
        info[did]["static_score"] = max(info[did]["static_score"], d.get("score", 0.0))
        info[did]["text"] = info[did].get("text") or d.get("text")

    docs = list(info.items())  # list of (doc_id, data)

    if mode == "rrf":
        # build ranks
        bm25_order = [d.get("id") for d in bm25_results]
        static_order = [d.get("id") for d in static_results]
        scores = {}
        for did, _ in docs:
            r = 0.0
            if did in bm25_order:
                rank = bm25_order.index(did) + 1
                r += 1.0 / (rrf_k + rank)
            if did in static_order:
                rank = static_order.index(did) + 1
                r += 1.0 / (rrf_k + rank)
            scores[did] = r
        # normalize rrf scores
        vals = list(scores.values())
        norm = _normalize_scores(vals) if vals else []
        did_to_norm = {d: v for d, v in zip(scores.keys(), norm)}

        combined = []
        for did, dat in docs:
            combined.append({
                "id": did,
                "score": did_to_norm.get(did, 0.0),
                "text": dat.get("text"),
                "bm25_score": dat.get("bm25_score"),
                "static_score": dat.get("static_score"),
            })

    else:
        # score mode: normalize each method's scores by its max
        bm_scores = [v["bm25_score"] for _, v in docs]
        st_scores = [v["static_score"] for _, v in docs]
        bm_norm = _normalize_scores(bm_scores)
        st_norm = _normalize_scores(st_scores)

        combined = []
        for (did, dat), b_n, s_n in zip(docs, bm_norm, st_norm):
            combined_score = alpha * b_n + (1.0 - alpha) * s_n
            combined.append({
                "id": did,
                "score": combined_score,
                "text": dat.get("text"),
                "bm25_score": dat.get("bm25_score"),
                "static_score": dat.get("static_score"),
            })

    # sort and return topk
    combined = sorted(combined, key=lambda x: x["score"], reverse=True)[:topk]
    return combined

def hybrid_bm25_instruction_retrieve(
    query: str,
    topk: int = 10,
    alpha: float = 0.5,
    mode: str = "score",
    rrf_k: int = 60,
) -> List[Dict]:
    """Combine BM25 and instruction-dense retrieval at document level.

    This mirrors :func:`hybrid_retrieve` but uses the instruction-dense
    retriever (E5-based) instead of static GloVe embeddings.

    Parameters
    ----------
    query: str
        Input query text.
    topk: int
        Number of documents to return.
    alpha: float
        Weight for BM25 when ``mode='score'`` (0..1).
    mode: {"score", "rrf"}
        Fusion strategy: score-based or Reciprocal Rank Fusion.
    rrf_k: int
        Constant k for RRF when ``mode='rrf'``.
    """

    from retrieval import bm25 as bm25_mod
    from retrieval import instruction_dense as idense_mod

    # collect candidates from both retrievers (use larger cutoff for fusion)
    bm25_results: List[Dict] = []
    try:
        bm25_results = bm25_mod.bm25_retrieve(query, topk=topk * 3)
    except Exception:
        bm25_results = []

    idense_results: List[Dict] = []
    try:
        idense_results = idense_mod.instruction_dense_retrieve(query, topk=topk * 3)
    except Exception:
        idense_results = []

    # aggregate scores at document level (instruction-dense is passage-level)
    info = defaultdict(lambda: {"bm25_score": 0.0, "instr_score": 0.0, "text": None})

    for d in bm25_results:
        did = d.get("id")
        if did is None:
            continue
        score = float(d.get("score", 0.0))
        if score > info[did]["bm25_score"]:
            info[did]["bm25_score"] = score
            # prefer BM25 text when available
            info[did]["text"] = d.get("text") or info[did]["text"]

    for d in idense_results:
        did = d.get("id")
        if did is None:
            continue
        score = float(d.get("score", 0.0))
        if score > info[did]["instr_score"]:
            info[did]["instr_score"] = score
            # if no text yet, use instruction-dense passage text
            if info[did]["text"] is None:
                info[did]["text"] = d.get("text")

    docs = list(info.items())  # (doc_id, data)

    if not docs:
        return []

    if mode == "rrf":
        # build rank lists by document id
        bm25_order = [d.get("id") for d in bm25_results]
        idense_order = [d.get("id") for d in idense_results]

        scores = {}
        for did, _ in docs:
            r = 0.0
            if did in bm25_order:
                rank = bm25_order.index(did) + 1
                r += 1.0 / (rrf_k + rank)
            if did in idense_order:
                rank = idense_order.index(did) + 1
                r += 1.0 / (rrf_k + rank)
            scores[did] = r

        vals = list(scores.values())
        norm = _normalize_scores(vals) if vals else []
        did_to_norm = {d: v for d, v in zip(scores.keys(), norm)}

        combined: List[Dict] = []
        for did, dat in docs:
            combined.append(
                {
                    "id": did,
                    "score": did_to_norm.get(did, 0.0),
                    "text": dat.get("text"),
                    "bm25_score": dat.get("bm25_score"),
                    "instr_score": dat.get("instr_score"),
                }
            )
    else:
        # score mode: normalize each method's scores by its max
        bm_scores = [v["bm25_score"] for _, v in docs]
        in_scores = [v["instr_score"] for _, v in docs]
        bm_norm = _normalize_scores(bm_scores)
        in_norm = _normalize_scores(in_scores)

        combined = []
        for (did, dat), b_n, i_n in zip(docs, bm_norm, in_norm):
            combined_score = alpha * b_n + (1.0 - alpha) * i_n
            combined.append(
                {
                    "id": did,
                    "score": combined_score,
                    "text": dat.get("text"),
                    "bm25_score": dat.get("bm25_score"),
                    "instr_score": dat.get("instr_score"),
                }
            )

    combined = sorted(combined, key=lambda x: x["score"], reverse=True)[:topk]
    return combined
