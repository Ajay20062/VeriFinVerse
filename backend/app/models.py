import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAMES = ("logistic_regression", "xgboost", "finbert")
MODEL_ENV_VARS = {
    "logistic_regression": "VERIFINVERSE_LOGISTIC_REGRESSION_MODEL",
    "xgboost": "VERIFINVERSE_XGBOOST_MODEL",
    "finbert": "VERIFINVERSE_FINBERT_MODEL",
}


def normalize_label(label: Any) -> str:
    value = str(label).strip().lower().replace("-", " ").replace("_", " ")
    if value in {"real", "true", "legitimate", "genuine", "reliable"}:
        return "REAL"
    if value in {
        "fake",
        "false",
        "not real",
        "misinformation",
        "unreliable",
        "fabricated",
    }:
        return "FAKE"
    return str(label).strip().upper()


class ModelInferenceError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}
        self._statuses = {
            name: {
                "name": name,
                "status": "not_configured",
                "details": f"Set {MODEL_ENV_VARS[name]} to a trained model artifact.",
            }
            for name in MODEL_NAMES
        }

    @classmethod
    def from_environment(cls) -> "ModelRegistry":
        registry = cls()
        for name in MODEL_NAMES:
            configured_path = os.getenv(MODEL_ENV_VARS[name])
            if not configured_path:
                continue
            path = Path(configured_path).expanduser()
            if not path.exists():
                registry._set_error(name, f"Model artifact does not exist: {path}")
                continue
            try:
                adapter = (
                    registry._load_finbert(path)
                    if name == "finbert"
                    else registry._load_joblib_model(name, path)
                )
                registry._adapters[name] = adapter
                registry._statuses[name] = {
                    "name": name,
                    "status": "ready",
                    "details": f"Loaded model artifact from {path}.",
                }
            except Exception as error:
                registry._set_error(name, str(error))
                logger.exception("Could not load configured %s model", name)
        return registry

    def _set_error(self, name: str, details: str) -> None:
        self._statuses[name] = {
            "name": name,
            "status": "error",
            "details": details,
        }

    @staticmethod
    def _load_joblib_model(name: str, path: Path) -> Any:
        import joblib

        model = joblib.load(path)
        if not hasattr(model, "predict"):
            raise TypeError(f"{name} artifact must provide a predict() method.")
        return model

    @staticmethod
    def _load_finbert(path: Path) -> Any:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model=str(path),
            tokenizer=str(path),
            local_files_only=True,
        )

    def is_available(self, name: str) -> bool:
        return name in self._adapters

    def statuses(self) -> list[dict[str, str]]:
        return [self._statuses[name].copy() for name in MODEL_NAMES]

    def predict(
        self, text: str, selected_models: list[str] | tuple[str, ...]
    ) -> list[dict[str, Any]]:
        predictions = []
        for name in selected_models:
            adapter = self._adapters.get(name)
            if adapter is None:
                continue
            try:
                predictions.append(
                    self._predict_finbert(name, adapter, text)
                    if name == "finbert"
                    else self._predict_sklearn(name, adapter, text)
                )
            except Exception as error:
                logger.exception("Inference failed for %s", name)
                raise ModelInferenceError(
                    f"Inference failed for {name}: {error}"
                ) from error
        return predictions

    @staticmethod
    def _predict_sklearn(name: str, model: Any, text: str) -> dict[str, Any]:
        predicted_label = normalize_label(model.predict([text])[0])
        probabilities: dict[str, float] = {}
        confidence = None
        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba([text])[0]
            classes = getattr(model, "classes_", None)
            if classes is None:
                raise TypeError(
                    f"{name} model provides predict_proba() but has no classes_ attribute."
                )
            for class_name, probability in zip(classes, raw_probabilities):
                normalized = normalize_label(class_name)
                probabilities[normalized] = (
                    probabilities.get(normalized, 0.0) + float(probability)
                )
            confidence = probabilities.get(predicted_label)
        return {
            "model": name,
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "explanation": None,
        }

    @staticmethod
    def _predict_finbert(name: str, model: Any, text: str) -> dict[str, Any]:
        output = model(text, truncation=True, max_length=512, top_k=None)
        if output and isinstance(output[0], list):
            output = output[0]
        if not output:
            raise ValueError("FinBERT returned no class scores.")
        probabilities = {
            normalize_label(item["label"]): float(item["score"]) for item in output
        }
        best_label = max(probabilities, key=probabilities.get)
        return {
            "model": name,
            "label": best_label,
            "confidence": probabilities[best_label],
            "probabilities": probabilities,
            "explanation": None,
        }
