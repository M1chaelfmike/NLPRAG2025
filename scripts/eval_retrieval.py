import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import bm25 as bm25_mod
from retrieval import static_embed
from retrieval import dense as dense_mod
from retrieval import instruction_dense as idense_mod
from retrieval import hybrid as hybrid_mod



def load_split(split: str):
    """Load HQ-small train/validation/test split via datasets."""
    ds = load_dataset("izhx/COMP5423-25Fall-HQ-small", split=split, cache_dir=str(PROJECT_ROOT / "hf_cache" / "datasets"))
    # Expect fields: 'id', 'question', 'answer', 'supporting_ids'
    return ds


def ensure_all_indexes():
    """Initialize all retrievers once."""
    # BM25 (corpus shared by others)
    idx_path = None
    if hasattr(bm25_mod, "HF_CACHE_DIR"):
        idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl")
    bm25_mod.init(index_path=idx_path)

    # static
    try:
        static_embed.ensure_index()
    except Exception as e:
        print(f"[WARN] static ensure_index failed: {e}")

    # dense
    try:
        dense_mod.ensure_index()
    except Exception as e:
        print(f"[WARN] dense ensure_index failed: {e}")

    # instruction-dense
    try:
        idense_mod.ensure_index()
    except Exception as e:
        print(f"[WARN] idense ensure_index failed: {e}")


def retrieve_ids(query: str, method: str, topk: int) -> List[str]:
    """Return list of document IDs for a query using specified retriever."""
    method = method.lower()
    if method == "bm25":
        docs = bm25_mod.bm25_retrieve(query, topk=topk)
        return [d[0] if isinstance(d, tuple) else d.get("id") for d in docs]
    elif method == "static":
        docs = static_embed.embed_retrieve(query, topk=topk)
        return [d["id"] for d in docs]
    elif method == "dense":
        docs = dense_mod.dense_retrieve(query, topk=topk)
        return [d["id"] for d in docs]
    elif method == "idense":
        docs = idense_mod.instruction_dense_retrieve(query, topk=topk)
        return [d["id"] for d in docs]
    elif method == "hybrid":
        docs = hybrid_mod.hybrid_retrieve(query, topk=topk)
        return [d["id"] for d in docs]
    elif method == "multivector":
        from retrieval import multivector as mv_mod
        docs = mv_mod.multivector_retrieve(query, topk=topk)
        return [d["id"] for d in docs]
    else:
        raise ValueError(f"Unknown method: {method}")


def eval_retriever(split: str, method: str, topk: int, max_samples: int = None) -> Dict[str, float]:
    ds = load_split(split)
    ensure_all_indexes()

    total = 0
    hit_at_k = 0  # at least one supporting doc retrieved
    avg_support_hits = 0.0

    for ex in ds:
        if max_samples is not None and total >= max_samples:
            break
        # In this dataset: question field is named 'text'
        q = ex.get("text")
        supporting_ids = ex.get("supporting_ids") or []
        if not q or not supporting_ids:
            continue

        retrieved_ids = retrieve_ids(q, method=method, topk=topk)
        retrieved_set = set(retrieved_ids)
        support_set = set(supporting_ids)
        inter = retrieved_set & support_set

        total += 1
        if inter:
            hit_at_k += 1
        avg_support_hits += len(inter)

        if total % 100 == 0:
            print(f"Processed {total} samples...")

    if total == 0:
        return {"n": 0, "hit_rate": 0.0, "avg_hits": 0.0}

    hit_rate = hit_at_k / total
    avg_hits = avg_support_hits / total
    return {"n": total, "hit_rate": hit_rate, "avg_hits": avg_hits}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever recall on HQ-small train/validation.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split: train/validation/test")
    parser.add_argument("--method", type=str, default="bm25", help="Retrieval method: bm25/static/dense/idense/hybrid/multivector")
    parser.add_argument("--topk", type=int, default=10, help="Top-k documents to retrieve")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max number of samples to evaluate (None = all)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    if args.max_samples <= 0:
        max_samples = None
    else:
        max_samples = args.max_samples

    print(f"Evaluating retriever='{args.method}' on split='{args.split}' topk={args.topk} max_samples={max_samples} ...")
    metrics = eval_retriever(split=args.split, method=args.method, topk=args.topk, max_samples=max_samples)
    # enrich with config info for easier reporting
    metrics.update({
        "method": args.method,
        "split": args.split,
        "topk": args.topk,
        "max_samples": max_samples,
    })

    print("==== Results ====")
    print(json.dumps(metrics, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
