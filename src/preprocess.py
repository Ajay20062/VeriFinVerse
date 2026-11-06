"""Preprocess labeled data and create deterministic stratified splits."""

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_primary_dataset


_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML = re.compile(r"<[^>]+>")
_NOISE = re.compile(r"[^a-zA-Z0-9$%.,!?'-]+")


def clean_for_tfidf(text: str) -> str:
    """Apply lightweight normalization for the classical bag-of-words models."""
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _NOISE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_splits(
    data: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    train_val, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        stratify=data["label"],
    )
    val_fraction = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val["label"],
    )
    return {"train": train, "val": validation, "test": test}


def parse_label_mapping(value: str) -> dict[str, str]:
    mapping = {}
    if value:
        for item in value.split(","):
            source, separator, target = item.partition("=")
            if not separator or not source.strip() or not target.strip():
                raise ValueError("Use comma-separated source=REAL/FAKE pairs")
            mapping[source.strip()] = target.strip().upper()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled primary CSV")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--label-map", default="", help="Example: 0=FAKE,1=REAL")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_primary_dataset(
        args.input,
        args.text_column,
        args.label_column,
        parse_label_mapping(args.label_map),
    )
    data["tfidf_text"] = data["text"].map(clean_for_tfidf)
    splits = make_splits(
        data,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} rows; labels={split['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
