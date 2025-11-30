import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_TRAIN_PATH = PROJECT_ROOT / "hf_cache" / "sft_train.jsonl"

# 直接使用本地已缓存的 0.5B 权重，避免重复从 HuggingFace 下载。
LOCAL_BASE_MODEL_DIR = (
    PROJECT_ROOT
    / "hf_cache"
    / "transformers"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)

MODEL_NAME = str(LOCAL_BASE_MODEL_DIR)
OUTPUT_DIR = PROJECT_ROOT / "hf_cache" / "qwen_0_5b_sft"


class JsonlSftDataset(Dataset):
    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj = self.samples[idx]
        prompt = obj["prompt"]
        response = obj["response"]
        # 简单做法：拼接 prompt + response，让模型学习全序列
        text = prompt + "\n\n" + response
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # 这里我们直接让所有 token 参与 loss（简单 SFT）
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_sft(
    model_name: str = MODEL_NAME,
    train_path: Path = SFT_TRAIN_PATH,
    batch_size: int = 2,
    epochs: int = 1,
    lr: float = 5e-5,
    max_length: int = 512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    ds = JsonlSftDataset(train_path, tokenizer, max_length=max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(dl) * epochs
    print(f"Dataset size: {len(ds)} samples, batch_size={batch_size}, total_steps={total_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    step = 0
    start_time = time.time()
    for epoch in range(epochs):
        for batch in dl:
            step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 10 == 0 or step == 1 or step == total_steps:
                elapsed = time.time() - start_time
                avg_step_time = elapsed / step
                remaining_steps = total_steps - step
                eta_sec = remaining_steps * avg_step_time
                eta_min = eta_sec / 60.0
                print(
                    f"Epoch {epoch+1}/{epochs}, step {step}/{total_steps}, "
                    f"loss={loss.item():.4f}, elapsed={elapsed/60.0:.1f}min, "
                    f"eta={eta_min:.1f}min"
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved SFT model to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default=str(SFT_TRAIN_PATH))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train_sft(
        model_name=MODEL_NAME,
        train_path=Path(args.train_path),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
    )
