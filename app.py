"""Flask interface for the VeriFinVerse claim classifier."""

import csv
import logging
import os
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, Flask, current_app, jsonify, render_template, request

from src.algorithms import ALGORITHMS, dataset_summary, load_dataset, run_algorithm
from src.explain import explain_text, load_model

ROOT = Path(__file__).resolve().parent
HISTORY_FIELDS = ["timestamp", "model", "text", "verdict", "confidence", "latency_ms"]


def _environment_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _default_config() -> dict[str, Any]:
    try:
        port = int(os.getenv("VERIFINVERSE_PORT", "5000"))
    except ValueError as error:
        raise ValueError("VERIFINVERSE_PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise ValueError("VERIFINVERSE_PORT must be between 1 and 65535.")

    return {
        "HOST": os.getenv("VERIFINVERSE_HOST", "127.0.0.1"),
        "PORT": port,
        "DEBUG": _environment_bool("VERIFINVERSE_DEBUG", False),
        "LOG_LEVEL": os.getenv("VERIFINVERSE_LOG_LEVEL", "INFO").upper(),
        "MODEL_DIR": _environment_path("VERIFINVERSE_MODEL_DIR", ROOT / "models"),
        "DATASET_PATH": _environment_path(
            "VERIFINVERSE_DATASET_PATH", ROOT / "data" / "raw" / "finfact_claims.csv"
        ),
        "HISTORY_PATH": _environment_path(
            "VERIFINVERSE_HISTORY_PATH", ROOT / "data" / "history.csv"
        ),
    }


def json_error(message: str, status_code: int, **extra: Any):
    return jsonify({"error": message, **extra}), status_code


def available_models() -> list[str]:
    model_dir: Path = current_app.config["MODEL_DIR"]
    if not model_dir.exists():
        return []
    return sorted(path.stem for path in model_dir.glob("*.joblib") if path.is_file())


def get_model(name: str) -> Any:
    if not name:
        raise ValueError("Model name is required.")

    model_path = current_app.config["MODEL_DIR"] / f"{name}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model '{name}' was not found.")

    model_cache: dict[str, tuple[int, Any]] = current_app.extensions[
        "verifinverse_model_cache"
    ]
    model_cache_lock: threading.Lock = current_app.extensions[
        "verifinverse_model_cache_lock"
    ]
    with model_cache_lock:
        current_mtime = model_path.stat().st_mtime_ns
        cached = model_cache.get(name)
        if cached is None or cached[0] != current_mtime:
            model_cache[name] = (current_mtime, load_model(model_path))
        return model_cache[name][1]


def append_history(row: dict[str, Any]) -> None:
    history_path: Path = current_app.config["HISTORY_PATH"]
    history_lock: threading.Lock = current_app.extensions["verifinverse_history_lock"]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_lock:
        needs_header = not history_path.exists() or history_path.stat().st_size == 0
        with history_path.open("a", encoding="utf-8", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


api = Blueprint("api", __name__)


@api.get("/api/v1/models")
@api.get("/models")
def models():
    return jsonify({"models": available_models()})


@api.get("/api/v1/algorithms")
@api.get("/algorithms")
def algorithms():
    return jsonify({"algorithms": ALGORITHMS})


@api.get("/api/v1/dataset/summary")
@api.get("/dataset/summary")
def dataset_statistics():
    try:
        dataset_path: Path = current_app.config["DATASET_PATH"]
        return jsonify(dataset_summary(load_dataset(dataset_path)))
    except (OSError, ValueError) as error:
        return json_error(str(error), 500)


@api.post("/api/v1/algorithms/run")
@api.post("/algorithms/run")
def run_algorithm_route():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Request body must be a JSON object.", 400)

    algorithm = payload.get("algorithm")
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
        return json_error("Unknown algorithm.", 400, algorithms=list(ALGORITHMS))

    try:
        dataset_path: Path = current_app.config["DATASET_PATH"]
        frame = load_dataset(dataset_path)
        return jsonify(
            {
                "algorithm": algorithm,
                "description": ALGORITHMS[algorithm],
                "result": run_algorithm(algorithm, frame),
            }
        )
    except (OSError, ValueError) as error:
        return json_error(str(error), 400)


@api.post("/api/v1/predict")
@api.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Request body must be a JSON object.", 400)

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return json_error("Text must be a non-empty string.", 400)
    if len(text) > 1000:
        return json_error("Text must be at most 1000 characters.", 400)

    model_name = payload.get("model", "logistic")
    available = available_models()
    if not isinstance(model_name, str) or model_name not in available:
        return json_error("Unknown model.", 404, available_models=available)

    try:
        model = get_model(model_name)
    except FileNotFoundError as error:
        return json_error(str(error), 404)

    started = time.perf_counter()
    result = explain_text(text, model)
    latency_ms = (time.perf_counter() - started) * 1000
    result["latency_ms"] = latency_ms
    append_history(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "text": text,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "latency_ms": latency_ms,
        }
    )
    return jsonify(result)


@api.post("/api/v1/predict/all")
@api.post("/predict/all")
def predict_all():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Request body must be a JSON object.", 400)

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return json_error("Text must be a non-empty string.", 400)
    if len(text) > 1000:
        return json_error("Text must be at most 1000 characters.", 400)

    model_names = available_models()
    if not model_names:
        return json_error("No trained models are available.", 404, results=[])

    results = []
    for model_name in model_names:
        try:
            model = get_model(model_name)
        except FileNotFoundError as error:
            return json_error(str(error), 404, results=results)

        started = time.perf_counter()
        result = explain_text(text, model)
        latency_ms = (time.perf_counter() - started) * 1000
        result["model"] = model_name
        result["latency_ms"] = latency_ms
        append_history(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "text": text,
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "latency_ms": latency_ms,
            }
        )
        results.append(result)

    return jsonify({"results": results})


@api.get("/api/v1/history")
@api.get("/history")
def history():
    try:
        limit = int(request.args.get("limit", "20"))
    except ValueError:
        return json_error("limit must be an integer.", 400)
    if limit < 0:
        return json_error("limit must be non-negative.", 400)
    limit = min(limit, 200)

    history_path: Path = current_app.config["HISTORY_PATH"]
    history_lock: threading.Lock = current_app.extensions["verifinverse_history_lock"]
    with history_lock:
        if not history_path.exists():
            rows = []
        else:
            with history_path.open("r", encoding="utf-8", newline="") as history_file:
                rows = list(csv.DictReader(history_file))

    history_rows = list(reversed(rows[-limit:])) if limit else []
    return jsonify({"history": history_rows})


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    """Create and configure a VeriFinVerse Flask application."""
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    app.config.from_mapping(_default_config())
    if config is not None:
        app.config.from_mapping(config)
    for setting in ("MODEL_DIR", "DATASET_PATH", "HISTORY_PATH"):
        path = Path(app.config[setting])
        app.config[setting] = path if path.is_absolute() else ROOT / path

    log_level = getattr(logging, str(app.config["LOG_LEVEL"]).upper(), None)
    if not isinstance(log_level, int):
        raise ValueError("LOG_LEVEL must be a valid Python logging level.")
    app.logger.setLevel(log_level)

    app.extensions["verifinverse_model_cache"] = {}
    app.extensions["verifinverse_model_cache_lock"] = threading.Lock()
    app.extensions["verifinverse_history_lock"] = threading.Lock()

    @app.before_request
    def start_request_timer():
        request.environ["verifinverse_request_started_at"] = time.perf_counter()

    @app.after_request
    def log_request(response):
        started = request.environ.get("verifinverse_request_started_at")
        if started is not None:
            duration_ms = (time.perf_counter() - started) * 1000
            app.logger.info(
                "%s %s %s %.2fms",
                request.method,
                request.path,
                response.status_code,
                duration_ms,
            )
        return response

    app.register_blueprint(api)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/workbench")
    def workbench():
        return render_template("workbench.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"],
    )
