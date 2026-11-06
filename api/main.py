"""VeriFinVerse prediction API."""

from pathlib import Path

import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db.database import get_history, initialize_database, save_prediction
from src.explain import explain_text
from src.inference import available_models, predict_one

app = FastAPI(title="VeriFinVerse API", version="1.0.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    model: str = "logistic"


@app.on_event("startup")
def startup():
    initialize_database()


@app.post("/predict")
def predict(request: PredictionRequest):
    models = available_models()
    if request.model not in {"logistic", "xgboost", "finbert"}:
        raise HTTPException(
            status_code=400,
            detail="model must be one of logistic, xgboost, or finbert",
        )
    if request.model not in models:
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
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/models")
def models():
    return {"models": available_models()}


@app.get("/history")
def history(limit: int = 20):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    try:
        return {"predictions": get_history(limit)}
    except mysql.connector.Error as error:
        raise HTTPException(status_code=503, detail="MySQL is unavailable") from error


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parents[1] / "frontend", html=True),
    name="frontend",
)
