"""VeriFinVerse prediction API."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db.database import initialize_database, save_prediction
from src.explain import explain_text
from src.inference import predict_one

app = FastAPI(title="VeriFinVerse API", version="1.0.0")
AVAILABLE_MODELS = {"logistic", "xgboost", "finbert"}


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    model: str = "finbert"


@app.on_event("startup")
def startup():
    initialize_database()


@app.post("/predict")
def predict(request: PredictionRequest):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(AVAILABLE_MODELS)}",
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
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
