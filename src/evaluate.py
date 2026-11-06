"""Evaluate trained models, save measured metrics, and render comparison plots."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import LABEL_TO_ID
from src.inference import predict_proba


def _latency_ms(model_name: str, texts: list[str], warmups: int, repeats: int) -> float:
    sample = texts[: min(len(texts), 100)]
    for text in sample[:warmups]:
        predict_proba(model_name, [text])
    durations = []
    for text in sample:
        for _ in range(repeats):
            start = time.perf_counter()
            predict_proba(model_name, [text])
            durations.append((time.perf_counter() - start) * 1000)
    return float(np.mean(durations)) if durations else float("nan")


def evaluate_model(model_name: str, test: pd.DataFrame, warmups: int, repeats: int):
    text_column = "tfidf_text" if model_name in {"logistic", "xgboost"} else "text"
    texts = test[text_column].fillna("").astype(str).tolist()
    real, fake = predict_proba(model_name, texts)
    truth = test["label"].map(LABEL_TO_ID).astype(int).to_numpy()
    predicted = (fake >= real).astype(int)
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, fake)),
        "latency_ms_per_article": _latency_ms(model_name, texts, warmups, repeats),
    }
    return metrics, confusion_matrix(truth, predicted, labels=[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--models", nargs="+", choices=("logistic", "xgboost", "finbert"),
                        default=("logistic", "xgboost", "finbert"))
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.latency_repeats < 1:
        raise ValueError("warmups must be non-negative and latency-repeats positive")
    test = pd.read_csv(args.test)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, matrices = [], {}
    for name in args.models:
        metrics, matrix = evaluate_model(name, test, args.warmups, args.latency_repeats)
        results.append(metrics)
        matrices[name] = matrix
        print(json.dumps(metrics, indent=2))

        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
            xlabel="Predicted label", ylabel="True label",
            title=f"{name} confusion matrix",
        )
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(figure)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for row in results:
        axis.scatter(row["latency_ms_per_article"], row["f1"], label=row["model"])
        axis.annotate(row["model"], (row["latency_ms_per_article"], row["f1"]))
    axis.set(
        xlabel="Mean inference latency (ms/article)",
        ylabel="F1 score",
        title="F1 vs inference latency",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "f1_vs_latency.png", dpi=150)
    plt.close(figure)
    print(f"Saved measured metrics and plots in {output.resolve()}")


if __name__ == "__main__":
    main()
