"""Train TF-IDF Logistic Regression and Complement Naive Bayes classifiers."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score

from src.algorithms import build_models

DEFAULT_MODEL_DIR = Path("models")


def train(
    train_path: Path = Path("data/processed/train.csv"),
    validation_path: Path = Path("data/processed/val.csv"),
    model_dir: Path = DEFAULT_MODEL_DIR,
    selected_model: str = "all",
) -> dict[str, Pipeline]:
    train_frame = pd.read_csv(train_path)
    val_frame = pd.read_csv(validation_path)
    if train_frame.empty or val_frame.empty:
        raise ValueError("Training and validation CSVs must both contain claims.")
    if train_frame["label"].nunique() < 2:
        raise ValueError("Training data must contain both REAL and FAKE labels.")

    models = build_models()
    requested = list(models) if selected_model == "all" else [selected_model]
    model_dir.mkdir(parents=True, exist_ok=True)
    trained = {}
    for name in requested:
        model = models[name]
        model.fit(train_frame["tfidf_text"], train_frame["label"].astype(int))
        predictions = model.predict(val_frame["tfidf_text"])
        score = f1_score(val_frame["label"].astype(int), predictions, average="macro", zero_division=0)
        output_path = model_dir / f"{name}.joblib"
        joblib.dump(model, output_path)
        print(f"{name}: validation macro-F1={score:.4f}; saved {output_path}")
        trained[name] = model
    return trained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", choices=["all", *build_models()], default="all")
    args = parser.parse_args()
    train(args.train, args.val, args.model_dir, args.model)


if __name__ == "__main__":
    main()
