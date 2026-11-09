"""Clean, deduplicate, and stratify Fin-Fact claims into train/val/test CSVs."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import clean_text, normalize_label


def preprocess(
    input_path: Path,
    text_column: str,
    label_column: str,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_path)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

    frame = frame[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].map(normalize_label)
    frame["tfidf_text"] = frame["text"].map(clean_text)
    frame = frame.dropna(subset=["label"])
    frame = frame[frame["tfidf_text"].ne("")]

    conflicting = frame.groupby("tfidf_text")["label"].nunique()
    conflicting_text = set(conflicting[conflicting > 1].index)
    frame = frame[~frame["tfidf_text"].isin(conflicting_text)]
    frame = frame.drop_duplicates(subset=["tfidf_text"], keep="first")
    frame = frame[["text", "label", "tfidf_text"]].reset_index(drop=True)

    if frame["label"].nunique() != 2 or frame["label"].value_counts().min() < 7:
        raise ValueError("At least seven claims per class are needed for stratified 70/15/15 splits.")

    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=seed,
        stratify=frame["label"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder["label"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", validation), ("test", test)):
        split.sort_index().to_csv(output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} claims; labels={split['label'].value_counts().to_dict()}")

    return train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/finfact_claims.csv"))
    parser.add_argument("--text-column", default="article")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess(args.input, args.text_column, args.label_column, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
