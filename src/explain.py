"""SHAP explanations with scores signed toward FAKE (positive) or REAL (negative)."""

import numpy as np

from src.inference import load_model, predict_proba
from src.preprocess import clean_for_tfidf


def explain_text(model_name: str, text: str, max_signals: int = 10) -> list[dict]:
    if max_signals < 1:
        raise ValueError("max_signals must be positive")
    if model_name in {"logistic", "xgboost"}:
        return _explain_classical(model_name, text, max_signals)
    if model_name == "finbert":
        return _explain_finbert(text, max_signals)
    raise ValueError(f"Unsupported model: {model_name}")


def _explain_classical(model_name: str, text: str, max_signals: int) -> list[dict]:
    import shap

    pipeline = load_model(model_name)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vector = vectorizer.transform([clean_for_tfidf(text)])
    names = vectorizer.get_feature_names_out()

    if model_name == "logistic":
        explanation = shap.LinearExplainer(classifier, vectorizer.transform([""]))
    else:
        explanation = shap.TreeExplainer(classifier)
    raw_values = explanation.shap_values(vector)
    if isinstance(raw_values, list):
        if len(raw_values) == 2:
            raw_values = np.asarray(raw_values[1]) - np.asarray(raw_values[0])
        else:
            raw_values = raw_values[-1]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[0]
        if values.shape[-1] == 2:
            values = values[:, 1] - values[:, 0]
    values = values.reshape(-1)
    if values.size != vector.shape[1]:
        raise ValueError(
            f"Unexpected {model_name} SHAP output size: {values.size}; "
            f"expected {vector.shape[1]} features"
        )
    present = vector.toarray()[0] != 0
    return _rank_signals(names[present], values[present], max_signals)


def _explain_finbert(text: str, max_signals: int) -> list[dict]:
    import shap

    tokenizer, _, _, _ = load_model("finbert")

    def score(batch):
        real, fake = predict_proba("finbert", list(batch))
        return np.column_stack((real, fake))

    explainer = shap.Explainer(score, shap.maskers.Text(tokenizer))
    result = explainer([text], max_evals=100)
    values = np.asarray(result.values)
    if values.ndim == 3:
        signed = values[0, :, 1] - values[0, :, 0]
    elif values.ndim == 2:
        signed = values[0]
    else:
        raise ValueError(f"Unexpected FinBERT SHAP shape: {values.shape}")
    tokens = np.asarray(result.data[0], dtype=object)
    return _rank_signals(tokens, signed, max_signals)


def _rank_signals(tokens, signed_values, max_signals):
    signals = []
    for token, score in zip(tokens, signed_values):
        word = str(token).strip()
        value = float(score)
        if not word or word in {" ", "\n"} or value == 0:
            continue
        signals.append(
            {
                "word": word,
                "score": value,
                "toward": "FAKE" if value > 0 else "REAL",
            }
        )
    signals.sort(key=lambda item: abs(item["score"]), reverse=True)
    return signals[:max_signals]
