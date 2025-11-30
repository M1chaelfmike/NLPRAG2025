"""
改进的RAG评估脚本 - 包含更详细的指标
基于PDF要求实现标准评估
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import re

from datasets import load_dataset

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import rag_pipeline


def normalize_text(s: str) -> str:
    """文本标准化"""
    s = s.lower().strip()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[\.,!?;:\"'()`\[\]{}]", "", s)
    return s


def calculate_f1(prediction: str, ground_truth: str) -> float:
    """计算F1分数（基于token重叠）"""
    pred_tokens = set(normalize_text(prediction).split())
    gold_tokens = set(normalize_text(ground_truth).split())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    common = pred_tokens & gold_tokens
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def eval_rag_detailed(split: str = "validation", method: str = "bm25", max_samples: int = None):
    """详细的RAG评估"""
    ds = load_dataset(
        "izhx/COMP5423-25Fall-HQ-small",
        split=split,
        cache_dir=str(PROJECT_ROOT / "hf_cache" / "datasets"),
    )
    
    results = {
        "total": 0,
        "correct": 0,
        "not_found": 0,
        "wrong": 0,
        "f1_scores": [],
        "examples": []  # 保存一些示例
    }
    
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        
        question = ex["text"]
        gold_answer = ex["answer"]
        
        # 运行RAG
        result = rag_pipeline(question, method=method)
        predicted = result.get("answer", "")
        
        results["total"] += 1
        
        # 分类结果
        if predicted.strip() == "Not found in provided documents.":
            results["not_found"] += 1
            category = "not_found"
            f1 = 0.0
        else:
            f1 = calculate_f1(predicted, gold_answer)
            results["f1_scores"].append(f1)
            
            if f1 > 0.5:  # F1 > 0.5认为是正确
                results["correct"] += 1
                category = "correct"
            else:
                results["wrong"] += 1
                category = "wrong"
        
        # 保存前10个示例
        if len(results["examples"]) < 10:
            results["examples"].append({
                "question": question,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "f1": f1,
                "category": category,
                "top_doc_id": result.get("retrieved_docs", [{}])[0].get("id", "N/A") if result.get("retrieved_docs") else "N/A"
            })
        
        if (i + 1) % 20 == 0:
            print(f"已处理 {i + 1}/{min(max_samples or len(ds), len(ds))} 样本...")
    
    # 计算最终指标
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
        results["not_found_rate"] = results["not_found"] / results["total"]
        results["wrong_rate"] = results["wrong"] / results["total"]
        
        if results["f1_scores"]:
            results["avg_f1"] = sum(results["f1_scores"]) / len(results["f1_scores"])
            results["median_f1"] = sorted(results["f1_scores"])[len(results["f1_scores"]) // 2]
        else:
            results["avg_f1"] = 0.0
            results["median_f1"] = 0.0
    
    return results


def print_results(results: dict, method: str):
    """打印格式化结果"""
    print("\n" + "="*70)
    print(f"         RAG评估结果 - {method.upper()}")
    print("="*70)
    
    total = results["total"]
    if total == 0:
        print("没有评估样本")
        return
    
    print(f"\n总样本数: {total}")
    print(f"\n性能指标:")
    print(f"  • 准确率 (F1>0.5):  {results['accuracy']*100:>6.2f}%  ({results['correct']}/{total})")
    print(f"  • 错误率:           {results['wrong_rate']*100:>6.2f}%  ({results['wrong']}/{total})")
    print(f"  • 未找到率:         {results['not_found_rate']*100:>6.2f}%  ({results['not_found']}/{total})")
    print(f"\n质量指标:")
    print(f"  • 平均F1分数:       {results['avg_f1']:>6.4f}")
    print(f"  • 中位F1分数:       {results['median_f1']:>6.4f}")
    
    # 显示示例
    print(f"\n示例结果（前5个）:")
    print("-"*70)
    for i, ex in enumerate(results["examples"][:5], 1):
        print(f"\n示例 {i} [{ex['category'].upper()}] F1={ex['f1']:.3f}")
        print(f"问题: {ex['question'][:80]}...")
        print(f"标准答案: {ex['gold_answer'][:80]}...")
        print(f"系统输出: {ex['predicted'][:80]}...")
        print(f"顶部文档: {ex['top_doc_id']}")
    
    # 改进建议
    print("\n" + "="*70)
    print("改进建议:")
    print("="*70)
    
    if results['accuracy'] < 0.3:
        print("❌ 准确率很低（<30%），建议：")
        print("   1. 切换到 dense 或 idense 检索器")
        print("   2. 增加检索文档数量（topk=20）")
        print("   3. 检查prompt设计是否合理")
    elif results['accuracy'] < 0.5:
        print("⚠️  准确率较低（30-50%），建议：")
        print("   1. 优化prompt模板")
        print("   2. 调整验证机制")
        print("   3. 考虑使用更大的生成模型")
    else:
        print("✓ 准确率良好（>50%）")
    
    if results['not_found_rate'] > 0.5:
        print("⚠️  未找到率过高（>50%），建议：")
        print("   1. 降低答案验证标准")
        print("   2. 改进prompt，鼓励模型从文档生成答案")
    
    if results['wrong_rate'] > 0.3:
        print("⚠️  错误率较高（>30%），建议：")
        print("   1. 加强答案验证机制")
        print("   2. 在prompt中强调只使用提供的文档")


def main():
    parser = argparse.ArgumentParser(description="详细的RAG评估")
    parser.add_argument("--split", type=str, default="validation", help="数据集划分")
    parser.add_argument("--method", type=str, default="bm25", help="检索方法")
    parser.add_argument("--max-samples", type=int, default=100, help="最大评估样本数")
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    args = parser.parse_args()
    
    print(f"开始评估 method={args.method}, split={args.split}, max_samples={args.max_samples}")
    
    results = eval_rag_detailed(
        split=args.split,
        method=args.method,
        max_samples=args.max_samples
    )
    
    print_results(results, args.method)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 详细结果已保存到: {output_path}")
    else:
        # 默认保存位置
        output_path = PROJECT_ROOT / f"rag_eval_{args.method}_{args.split}.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
