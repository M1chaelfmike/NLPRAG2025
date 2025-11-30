import argparse
import json
from pathlib import Path

from datasets import load_dataset
from typing import Optional
# 保证可以 import 项目内模块
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import rag_pipeline  # 使用你现有的 RAG 流程


def load_split(split: str):
    ds = load_dataset(
        "izhx/COMP5423-25Fall-HQ-small",
        split=split,
        cache_dir=str(PROJECT_ROOT / "hf_cache" / "datasets"),
    )
    # fields: id, text, answer, supporting_ids
    return ds


def normalize_text(s: str) -> str:
    import re

    s = s.lower().strip()
    s = re.sub(r"[\s]+", " ", s)
    # 去掉简单标点
    s = re.sub(r"[\.,!?;:\"'()`\[\]{}]", "", s)
    return s


def compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold answer."""
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def eval_rag(split: str, method: str, max_samples: Optional[int] = None):
    ds = load_split(split)

    total = 0
    correct = 0
    not_found = 0
    total_f1 = 0.0

    for ex in ds:
        if max_samples is not None and total >= max_samples:
            break

        question = ex["text"]
        gold_answer = ex["answer"]

        # 调用你在 main.py 里的统一接口（其内部会使用默认 topk）
        result = rag_pipeline(question, method=method)
        final_answer = result.get("final_answer", "")

        total += 1
        if final_answer.strip() == "Not found in provided documents.":
            not_found += 1
            # F1 for not-found is 0
        else:
            gold_norm = normalize_text(str(gold_answer))
            pred_norm = normalize_text(str(final_answer))
            # 简单规则：gold 答案子串出现在预测中
            if gold_norm in pred_norm or pred_norm in gold_norm:
                correct += 1
            # Compute F1 score
            f1 = compute_f1(pred_norm, gold_norm)
            total_f1 += f1

        if total % 50 == 0:
            print(f"Processed {total} examples...")

    if total == 0:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0, "not_found_rate": 0.0}

    acc = correct / total
    avg_f1 = total_f1 / total
    nf_rate = not_found / total
    return {"n": total, "accuracy": acc, "f1": avg_f1, "not_found_rate": nf_rate}


def main():
    parser = argparse.ArgumentParser(description="Evaluate full RAG (retrieval + generation) on HQ-small.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split: train/validation/test")
    parser.add_argument("--method", type=str, default="idense", help="Retrieval method for rag_pipeline")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples to evaluate (<=0 for all)")
    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples
    print(
        f"Evaluating full RAG with retriever='{args.method}' split='{args.split}' max_samples={max_samples} ..."
    )
    metrics = eval_rag(split=args.split, method=args.method, max_samples=max_samples)
    print("==== RAG Results ====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
