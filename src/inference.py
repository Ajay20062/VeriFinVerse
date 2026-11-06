"""Model loading and shared article prediction helpers."""

import json
import time
from functools import lru_cache
from pathlib import Path

import joblib

from src.preprocess import clean_for_tfidf

ROOT = Path(__file__).resolve().parents[1]


def available_models() -> list[str]:
    models = [
        name
        for name in ("logistic", "xgboost")
        if (ROOT / "models" / f"{name}.joblib").is_file()
    ]
    finbert_path = ROOT / "models" / "finbert"
    if (finbert_path / "training_config.json").is_file():
        models.append("finbert")
    return models


@lru_cache(maxsize=3)
def load_model(model_name: str):
    if model_name in {"logistic", "xgboost"}:
        path = ROOT / "models" / f"{model_name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Train the {model_name} model first: {path}")
        return joblib.load(path)
    if model_name == "finbert":
        path = ROOT / "models" / "finbert"
        if not path.is_dir():
            raise FileNotFoundError(f"Fine-tune FinBERT first: {path}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        config_path = path / "training_config.json"
        max_length = json.loads(config_path.read_text(encoding="utf-8")).get(
            "max_length", 128
        )
        return tokenizer, model, device, max_length
    raise ValueError(f"Unsupported model: {model_name}")


def predict_proba(model_name: str, texts: list[str]):
    loaded = load_model(model_name)
    if model_name in {"logistic", "xgboost"}:
        cleaned = [clean_for_tfidf(text) for text in texts]
        raw = loaded.predict_proba(cleaned)
        classes = list(loaded.named_steps["classifier"].classes_)
        return raw[:, classes.index(0)], raw[:, classes.index(1)]

    tokenizer, model, device, max_length = loaded
    import torch

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    label_map = model.config.label2id
    real_index = label_map.get("REAL", 0)
    fake_index = label_map.get("FAKE", 1)
    return probs[:, real_index], probs[:, fake_index]


def predict_one(model_name: str, text: str) -> dict:
    start = time.perf_counter()
    real_probability, fake_probability = predict_proba(model_name, [text])
    real_probability = float(real_probability[0])
    fake_probability = float(fake_probability[0])
    verdict = "FAKE" if fake_probability >= real_probability else "REAL"
    return {
        "verdict": verdict,
        "confidence": max(real_probability, fake_probability),
        "latency_ms": (time.perf_counter() - start) * 1000,
    }
