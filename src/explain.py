"""Predict a claim and return additive TF-IDF feature contributions."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix

from src.common import LABEL_NAMES, clean_text


def explain_text(text: str, model: Any, top_n: int = 8) -> dict[str, Any]:
    cleaned = clean_text(text)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vector = vectorizer.transform([cleaned])
    tfidf_values = vector.toarray()[0]
    probabilities = model.predict_proba([cleaned])[0]
    classes = list(classifier.classes_)

    fake_index = classes.index(1)
    real_index = classes.index(0)
    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
    elif hasattr(classifier, "feature_log_prob_"):
        weights = classifier.feature_log_prob_[fake_index] - classifier.feature_log_prob_[real_index]
    else:
        weights = None

    nonzero = np.flatnonzero(tfidf_values)
    feature_names = vectorizer.get_feature_names_out()
    if weights is not None:
        contributions = tfidf_values * weights
        ranked = sorted(nonzero, key=lambda index: abs(contributions[index]), reverse=True)[:top_n]
    else:
        empty = csr_matrix((1, vector.shape[1]))
        baseline = classifier.predict_proba(empty)[0, fake_index]
        contributions = np.zeros(vector.shape[1])
        for index in nonzero:
            feature_row = csr_matrix(
                ([tfidf_values[index]], ([0], [index])),
                shape=(1, vector.shape[1]),
            )
            contributions[index] = classifier.predict_proba(feature_row)[0, fake_index] - baseline
        ranked = sorted(nonzero, key=lambda index: abs(contributions[index]), reverse=True)[:top_n]
    signals = [
        {"word": str(feature_names[index]), "contribution": float(contributions[index])}
        for index in ranked
    ]

    verdict_index = int(np.argmax(probabilities))
    probability_by_label = {int(label): float(probabilities[i]) for i, label in enumerate(classes)}
    return {
        "verdict": LABEL_NAMES[int(classes[verdict_index])],
        "confidence": float(probabilities[verdict_index]),
        "probabilities": {
            "REAL": probability_by_label[0],
            "FAKE": probability_by_label[1],
        },
        "signals": signals,
        "known_words": sum(
            1
            for feature in vectorizer.build_analyzer()(cleaned)
            if " " not in feature and feature in vectorizer.vocabulary_
        ),
    }


def load_model(path: Path) -> Any:
    return joblib.load(path)
