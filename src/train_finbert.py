"""Fine-tune FinBERT using raw-ish text and configurable CPU/GPU settings."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features import ID_TO_LABEL, LABEL_TO_ID


class ArticleDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.labels = frame["label"].map(LABEL_TO_ID).astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            key: value.squeeze(0)
            for key, value in encoded.items()
        } | {"labels": torch.tensor(self.labels[index], dtype=torch.long)}


def _evaluate(model, loader, device):
    model.eval()
    predictions, expected = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            predictions.extend(model(**inputs).logits.argmax(dim=1).cpu().tolist())
            expected.extend(labels.cpu().tolist())
    return f1_score(expected, predictions, average="binary", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--output-dir", default="models/finbert")
    parser.add_argument("--base-model", default="ProsusAI/finbert")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.max_length <= 512 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("max-length must be 1..512; epochs and batch-size must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.val)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    train_loader = DataLoader(
        ArticleDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ArticleDataset(validation, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = -1.0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_f1 = _evaluate(model, val_loader, device)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={total_loss / max(1, len(train_loader)):.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "training_config.json").write_text(
                json.dumps(
                    {"base_model": args.base_model, "max_length": args.max_length},
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(f"Best validation F1: {best_f1:.4f}; saved model to {output}")


if __name__ == "__main__":
    main()
