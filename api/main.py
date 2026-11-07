"""VeriFinVerse prediction API and static web frontend."""

import logging
from pathlib import Path

import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db.database import get_history, initialize_database, save_prediction
from src.explain import explain_text
from src.inference import available_models, predict_one

app = FastAPI(title="VeriFinVerse API", version="1.0.0")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    model: str = "logistic"


@app.on_event("startup")
def startup():
    try:
        initialize_database()
    except (mysql.connector.Error, RuntimeError) as error:
        logger.warning("MySQL is unavailable; prediction history is disabled: %s", error)


@app.get("/models")
def models():
    return {"models": available_models()}


@app.post("/predict")
def predict(request: PredictionRequest):
    if request.model not in {"logistic", "xgboost", "finbert"}:
        raise HTTPException(
            status_code=400,
            detail="model must be one of logistic, xgboost, or finbert",
        )
    if request.model not in available_models():
        raise HTTPException(
            status_code=503,
            detail=f"The {request.model} model has not been trained",
        )
    try:
        result = predict_one(request.model, request.text)
        result["top_signals"] = explain_text(request.model, request.text)
        save_prediction(
            request.text,
            result["verdict"],
            result["confidence"],
            request.model,
            result["latency_ms"],
        )
        return result
    except mysql.connector.Error as error:
        raise HTTPException(status_code=503, detail="MySQL is unavailable") from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/history")
def history(limit: int = 20):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    try:
        return {"predictions": get_history(limit)}
    except mysql.connector.Error as error:
        raise HTTPException(status_code=503, detail="MySQL is unavailable") from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
