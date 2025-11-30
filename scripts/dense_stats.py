#!/usr/bin/env python3
"""统计 dense passages 和估算批次数/向量内存的脚本。
用法示例：
  python scripts/dense_stats.py --batch_size 256 --model all-mpnet-base-v2
"""
import argparse
import math


def main(batch_size: int = 256, model_name: str = "all-mpnet-base-v2"):
    from retrieval import dense
    # 构建 passages（如果还没构建）
    dense._ensure_passages()
    n = len(dense._PASSAGES)
    print("总 passages:", n)
    print(f"外层批次数 (batch_size={batch_size}):", math.ceil(n / batch_size))

    # 尝试获取 embedding 维度（如未加载模型，会自动加载）
    try:
        model = dense._lazy_model(model_name)
    except Exception as e:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device="cuda")
        except Exception as e2:
            print("无法加载模型以获取 embedding 维度：", e2)
            return
    # get dimension
    try:
        dim = model.get_sentence_embedding_dimension()
    except Exception:
        # fallback if model does not expose that API
        import numpy as np
        vec = model.encode(["test"], convert_to_numpy=True)
        dim = vec.shape[1]
    print("embedding 维度:", dim)
    bytes_est = n * dim * 4  # float32 bytes
    print("向量估算内存（MB）:", bytes_est / 1024.0 / 1024.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2')
    args = parser.parse_args()
    main(batch_size=args.batch_size, model_name=args.model)
