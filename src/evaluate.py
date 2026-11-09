"""Evaluate saved models on the held-out test split and plot F1 vs latency."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
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

DEFAULT_TEST = Path("data/processed/test.csv")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_REPORT_DIR = Path("reports")


def measure_latency(model: Any, texts: list[str], warmups: int = 5, repeats: int = 5) -> float:
    sample = texts[:100]
    if not sample:
        raise ValueError("Cannot measure latency without test claims.")
    for text in sample[: min(warmups, len(sample))]:
        model.predict([text])
    durations = []
    for _ in range(repeats):
        for text in sample:
            started = time.perf_counter()
            model.predict([text])
            durations.append((time.perf_counter() - started) * 1000)
    return float(np.mean(durations))


def evaluate(test_path: Path, model_dir: Path, report_dir: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(test_path)
    if frame.empty:
        raise ValueError("Test CSV contains no claims.")
    y_true = frame["label"].astype(int).to_numpy()
    texts = frame["tfidf_text"].fillna("").astype(str).tolist()
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, float]] = {}
    for model_path in sorted(model_dir.glob("*.joblib")):
        model_name = model_path.stem
        model = joblib.load(model_path)
        predictions = model.predict(texts).astype(int)
        probabilities = model.predict_proba(texts)
        classes = list(model.named_steps["classifier"].classes_)
        fake_probabilities = probabilities[:, classes.index(1)]
        model_metrics = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "fake_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
            "fake_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, fake_probabilities)) if len(np.unique(y_true)) == 2 else float("nan"),
            "mean_latency_ms": measure_latency(model, texts),
        }
        metrics[model_name] = model_metrics
        confusion = confusion_matrix(y_true, predictions, labels=[0, 1])
        pd.DataFrame(confusion, index=["REAL", "FAKE"], columns=["predicted_REAL", "predicted_FAKE"]).to_csv(
            report_dir / f"confusion_{model_name}.csv", index_label="actual"
        )

    if not metrics:
        raise FileNotFoundError(f"No .joblib models found in {model_dir}")
    pd.DataFrame(
        [{"model": model_name, **values} for model_name, values in metrics.items()]
    ).to_csv(report_dir / "metrics.csv", index=False)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 4))
    for name, values in metrics.items():
        axis.scatter(values["mean_latency_ms"], values["macro_f1"], label=name)
        axis.annotate(name, (values["mean_latency_ms"], values["macro_f1"]), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Mean per-claim latency (ms)")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Test Macro-F1 vs Inference Latency")
    axis.grid(True, alpha=0.3)
    if len(metrics) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "f1_vs_latency.png", dpi=150)
    plt.close(figure)
    print(json.dumps(metrics, indent=2, allow_nan=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    evaluate(args.test, args.model_dir, args.report_dir)


if __name__ == "__main__":
    main()
