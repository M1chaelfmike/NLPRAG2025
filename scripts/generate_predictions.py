"""
生成 test_prediction.jsonl 文件

格式要求 (来自 PDF):
{"id": <str>, "question": <str>, "answer": <str>, "retrieved_docs": [[id_1, score_1], [id_2, score_2], ..., [id_10, score_10]]}

使用方法:
    python scripts/generate_predictions.py --method bm25 --mode basic
    python scripts/generate_predictions.py --method bm25 --mode agentic
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from main import rag_pipeline

HF_CACHE_DIR = PROJECT_ROOT / "hf_cache" / "datasets"


def load_test_split():
    """加载 test split"""
    ds = load_dataset(
        "izhx/COMP5423-25Fall-HQ-small",
        split="test",
        cache_dir=str(HF_CACHE_DIR),
    )
    return ds


def generate_predictions(
    method: str = "bm25",
    mode: str = "basic",
    output_path: Path = None,
    max_samples: int = None
):
    """
    在 test set 上运行 RAG 并生成预测文件
    
    Args:
        method: 检索方法 (bm25, dense, idense, hybrid, etc.)
        mode: RAG模式 (basic, multi_turn, agentic)
        output_path: 输出文件路径
        max_samples: 最大样本数 (None = 全部)
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "test_prediction.jsonl"
    
    print(f"Loading test dataset...")
    ds = load_test_split()
    print(f"Test set size: {len(ds)}")
    
    # 初始化检索器
    print(f"Initializing retriever '{method}'...")
    from retrieval import bm25 as bm25_mod
    idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl") if hasattr(bm25_mod, "HF_CACHE_DIR") else None
    bm25_mod.init(index_path=idx_path)
    
    if method == "dense":
        try:
            from retrieval import dense as dense_mod
            dense_mod.ensure_index()
            dense_mod.load_index()
        except Exception as e:
            print(f"Warning: Dense not available: {e}, falling back to BM25")
            method = "bm25"
    elif method == "idense":
        try:
            from retrieval import instruction_dense as idense_mod
            idense_mod.ensure_index()
            idense_mod.load_index()
        except Exception as e:
            print(f"Warning: IDense not available: {e}, falling back to BM25")
            method = "bm25"
    
    print(f"Running predictions with method='{method}', mode='{mode}'...")
    
    predictions = []
    total = len(ds) if max_samples is None else min(len(ds), max_samples)
    
    for i, ex in enumerate(tqdm(ds, total=total, desc="Generating predictions")):
        if max_samples is not None and i >= max_samples:
            break
        
        question_id = ex["id"]
        question = ex["text"]
        
        # 运行 RAG pipeline
        try:
            result = rag_pipeline(question, method=method, mode=mode)
            
            # 提取答案
            answer = result.get("final_answer") or result.get("answer", "")
            
            # 提取 retrieved_docs: [[id, score], ...]
            docs = result.get("retrieved_docs", [])
            retrieved_docs = []
            for d in docs[:10]:  # 只取前10个
                doc_id = d.get("id", "")
                score = d.get("score", 0.0)
                retrieved_docs.append([doc_id, float(score)])
            
            # 补齐到10个 (如果不足)
            while len(retrieved_docs) < 10:
                retrieved_docs.append(["", 0.0])
            
        except Exception as e:
            print(f"\nError processing {question_id}: {e}")
            answer = ""
            retrieved_docs = [["", 0.0]] * 10
        
        prediction = {
            "id": question_id,
            "text": question,  # 使用 text 而不是 question，与老师格式一致
            "answer": answer,
            "retrieved_docs": retrieved_docs
        }
        predictions.append(prediction)
    
    # 写入文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Predictions saved to {output_path}")
    print(f"   Total: {len(predictions)} samples")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate test predictions for submission")
    parser.add_argument("--method", type=str, default="bm25",
                        choices=["bm25", "dense", "idense", "hybrid", "static", "multivector"],
                        help="Retrieval method")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "multi_turn", "agentic"],
                        help="RAG mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: test_prediction.jsonl)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (default: all)")
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    generate_predictions(
        method=args.method,
        mode=args.mode,
        output_path=output_path,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
