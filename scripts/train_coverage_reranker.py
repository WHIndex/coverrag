#!/usr/bin/env python3
"""Train a coverage-aware evidence reranker.

Expected JSONL input fields:
    text_a, text_b, label

The script uses a lightweight manual PyTorch loop so it does not require
Transformers Trainer or accelerate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class PairDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "text_a" not in row or "text_b" not in row:
                    continue
                self.rows.append(
                    {
                        "text_a": str(row["text_a"]),
                        "text_b": str(row["text_b"]),
                        "label": int(row.get("label", 0)),
                    }
                )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [row["text_a"] for row in batch],
            [row["text_b"] for row in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
        return encoded


def evaluate(model: Any, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            logits = output.logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            total += labels.numel()
            correct += int((preds == labels).sum().item())
            total_loss += float(output.loss.item()) * labels.numel()
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a coverage-aware evidence reranker.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file does not exist: {args.train_file}")
    if args.dev_file is not None and not args.dev_file.exists():
        raise FileNotFoundError(f"Dev file does not exist: {args.dev_file}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = PairDataset(args.train_file, tokenizer, args.max_length)
    if not train_dataset:
        raise ValueError(f"No training examples found in {args.train_file}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        collate_fn=train_dataset.collate,
    )
    dev_loader = None
    if args.dev_file is not None:
        dev_dataset = PairDataset(args.dev_file, tokenizer, args.max_length)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=max(1, args.batch_size),
            shuffle=False,
            collate_fn=dev_dataset.collate,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * max(1, args.epochs))
    warmup_steps = int(math.ceil(total_steps * max(0.0, args.warmup_ratio)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dev = -1.0
    for epoch in range(1, max(1, args.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            batch_size = int(batch["labels"].numel())
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={running_loss/max(1, seen):.4f}")

        train_loss = running_loss / max(1, seen)
        metrics = {"loss": train_loss, "accuracy": 0.0}
        if dev_loader is not None:
            metrics = evaluate(model, dev_loader, device)
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"dev_loss={metrics['loss']:.4f} dev_acc={metrics['accuracy']:.4f}"
            )
        else:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")

        should_save_best = dev_loader is None or metrics["accuracy"] >= best_dev
        if should_save_best:
            best_dev = metrics["accuracy"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "training_metrics.json").write_text(
                json.dumps({"epoch": epoch, "best_dev_accuracy": best_dev, **metrics}, indent=2),
                encoding="utf-8",
            )
        if args.save_every_epoch:
            epoch_dir = args.output_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

    print(f"Saved coverage-aware reranker to {args.output_dir}")


if __name__ == "__main__":
    main()
