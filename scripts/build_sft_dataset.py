import json
from pathlib import Path

from datasets import load_dataset
from typing import Optional
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache" / "datasets"


def main(split: str = "train", output_path: Optional[Path] = None, max_samples: Optional[int] = None) -> None:
    if output_path is None:
        output_path = PROJECT_ROOT / "hf_cache" / f"sft_{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "izhx/COMP5423-25Fall-HQ-small",
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )

    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            q = ex["text"]
            ans = ex["answer"]

            # 对于 SFT，我们只让模型学会在简单 context 下输出
            # 结构化的 Answer/Evidence/Sources，其中 Evidence 里
            # 至少包含 gold answer 子串。为了简单，这里只构造
            # 极简 context，并把 answer 直接当作 evidence 片段。
            # 这不是完美的 RAG，但足够教会模型输出模板。

            context = "[Document 1]: " + ans

            prompt = (
                "You are a Retrieval-Augmented Generation (RAG) assistant. "
                "Answer ONLY using the provided documents.\n\n"
                "Rules (must follow exactly):\n"
                "1) If an explicit sentence in the provided documents directly "
                "supports the factual answer, produce the following exact format:\n\n"
                "Answer: <one concise factual sentence>\n"
                "Evidence: \"<the exact sentence copied verbatim from one of the documents>\" (Document X)\n"
                "Sources: [Document X]\n\n"
                "2) If there is NO explicit supporting sentence in any provided document, reply exactly:\n"
                "Not found in provided documents.\n\n"
                "3) Do NOT use any outside knowledge. Do NOT hallucinate.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\n\n"
                "Respond now following rule 1 or 2 and the exact output format above."
            )

            target = (
                f"Answer: {ans}\n"
                f"Evidence: \"{ans}\" (Document 1)\n"
                "Sources: [Document 1]"
            )

            rec = {"prompt": prompt, "response": target}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {n} SFT examples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output is not None else None
    main(split=args.split, output_path=out, max_samples=args.max_samples)
