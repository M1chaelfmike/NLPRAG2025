import json
from pathlib import Path

from transformers import AutoTokenizer
from typing import Optional
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_TRAIN_PATH = PROJECT_ROOT / "hf_cache" / "sft_train.jsonl"
LOCAL_BASE_MODEL_DIR = (
    PROJECT_ROOT
    / "hf_cache"
    / "transformers"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)
MODEL_NAME = str(LOCAL_BASE_MODEL_DIR)


def main(path: Path = SFT_TRAIN_PATH, max_samples: Optional[int] = None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["prompt"] + "\n\n" + obj["response"]
            enc = tokenizer(text, truncation=False, add_special_tokens=True)
            lengths.append(len(enc["input_ids"]))

    if not lengths:
        print("No samples found.")
        return

    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        idx = int(p * n) - 1
        idx = max(0, min(idx, n - 1))
        return lengths[idx]

    print(f"Analyzed {n} samples from {path}")
    print(f"min length: {lengths[0]}")
    print(f"max length: {lengths[-1]}")
    print(f"median (p50): {pct(0.5)}")
    print(f"p90: {pct(0.9)}")
    print(f"p95: {pct(0.95)}")
    print(f"p99: {pct(0.99)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=str(SFT_TRAIN_PATH))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    main(path=Path(args.path), max_samples=args.max_samples)
