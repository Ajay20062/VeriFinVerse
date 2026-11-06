"""Train TF-IDF Logistic Regression and XGBoost classifiers."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import LABEL_TO_ID, build_tfidf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("logistic", "xgboost", "all"), default="all")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    required = {"tfidf_text", "label"}
    if not required.issubset(train.columns):
        raise ValueError(f"Training CSV must contain columns {sorted(required)}")
    y = train["label"].map(LABEL_TO_ID)
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Training labels must include REAL and FAKE")

    requested = ("logistic", "xgboost") if args.model == "all" else (args.model,)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in requested:
        if name == "logistic":
            classifier = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=args.seed
            )
        else:
            from xgboost import XGBClassifier

            classifier = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=args.seed,
            )
        pipeline = Pipeline(
            [
                ("tfidf", build_tfidf(max_features=args.max_features)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(train["tfidf_text"].fillna(""), y.astype(int))
        target = output / f"{name}.joblib"
        joblib.dump(pipeline, target)
        print(f"Saved {name} to {target}")


if __name__ == "__main__":
    main()
