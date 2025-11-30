import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # ../pythonProject
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["HF_METRICS_CACHE"] = str(HF_CACHE_DIR / "metrics")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_TRAIN_PATH = PROJECT_ROOT / "hf_cache" / "sft_train.jsonl"  # 已有的 SFT 数据
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"                   # 原始模型
SFT_MODEL_DIR = PROJECT_ROOT / "hf_cache" / "qwen_0_5b_sft"      # 你训练好的模型目录

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_samples(path: Path, max_samples: int = 50):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
    return samples


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=4,
            early_stopping=True,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def main(max_samples: int = 50):
    print("Loading base model:", BASE_MODEL_NAME)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype="auto").to(DEVICE)
    base_model.eval()

    print("Loading SFT model from:", SFT_MODEL_DIR)
    sft_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR, torch_dtype="auto").to(DEVICE)
    sft_model.eval()

    samples = load_samples(SFT_TRAIN_PATH, max_samples=max_samples)
    print(f"Loaded {len(samples)} samples from {SFT_TRAIN_PATH}")

    # 简单统计：是否包含 gold answer 子串
    base_contains = 0
    sft_contains = 0

    for i, ex in enumerate(samples):
        prompt = ex["prompt"]
        gold_resp = ex["response"]
        # gold answer 就在 response 的 Answer 行
        gold_answer_line = gold_resp.splitlines()[0]
        gold_answer = gold_answer_line.replace("Answer:", "").strip()

        print(f"\n=== Sample {i+1} ===")
        print("Question part:")
        # 只打印最后一段 Question 行，方便看
        for line in prompt.splitlines()[-5:]:
            print(line)

        # base 模型输出
        base_out = generate(base_model, base_tokenizer, prompt)
        # sft 模型输出
        sft_out = generate(sft_model, sft_tokenizer, prompt)

        print("\n[BASE OUTPUT]")
        print(base_out.split("Context:")[-1])  # 只看输出部分

        print("\n[SFT OUTPUT]")
        print(sft_out.split("Context:")[-1])

        # 简单统计：输出中是否包含 gold_answer 子串
        if gold_answer and gold_answer in base_out:
            base_contains += 1
        if gold_answer and gold_answer in sft_out:
            sft_contains += 1

    print("\n=== Summary ===")
    print(f"Gold answer substring present in BASE outputs: {base_contains}/{len(samples)}")
    print(f"Gold answer substring present in SFT outputs:  {sft_contains}/{len(samples)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()
    main(max_samples=args.max_samples)