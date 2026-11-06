import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.database import create_history, initialize_database, list_history
from app.models import MODEL_NAMES, ModelInferenceError, ModelRegistry

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)
    models: list[Literal["logistic_regression", "xgboost", "finbert"]] | None = Field(
        default=None, min_length=1
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain non-whitespace characters.")
        return value.strip()

    @field_validator("models")
    @classmethod
    def model_names_must_be_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Each model may only be requested once.")
        return value


class ModelPrediction(BaseModel):
    model: str
    label: str
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    explanation: dict[str, object] | None = None


class AnalyzeResponse(BaseModel):
    id: int
    created_at: str
    prediction: str
    confidence: float | None
    models: list[ModelPrediction]


class HistoryItem(BaseModel):
    id: int
    created_at: str
    text: str
    prediction: str
    confidence: float | None
    models: list[ModelPrediction]


class ModelsResponse(BaseModel):
    models: list[dict[str, str]]


def _cors_origins() -> list[str]:
    configured = os.getenv(
        "VERIFINVERSE_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    app.state.model_registry = ModelRegistry.from_environment()
    yield


app = FastAPI(
    title="VeriFinVerse API",
    description="Financial news verification with pluggable trained-model adapters.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry is not initialized.",
        )
    return registry


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    return ModelsResponse(models=_registry(request).statuses())


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    registry = _registry(request)
    selected_models = payload.models or MODEL_NAMES
    unavailable = [
        name for name in selected_models if not registry.is_available(name)
    ]
    if unavailable and payload.models is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "One or more requested models are unavailable.",
                "unavailable_models": unavailable,
                "model_status": registry.statuses(),
            },
        )

    try:
        predictions = registry.predict(payload.text, selected_models)
    except ModelInferenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "No trained models are available. Configure model artifacts before analyzing news.",
                "model_status": registry.statuses(),
            },
        )

    votes: dict[str, int] = {}
    for prediction in predictions:
        votes[prediction["label"]] = votes.get(prediction["label"], 0) + 1
    highest_vote_count = max(votes.values())
    leaders = [label for label, count in votes.items() if count == highest_vote_count]
    overall_prediction = leaders[0] if len(leaders) == 1 else "UNCERTAIN"

    winning_confidences = [
        item["confidence"]
        for item in predictions
        if item["label"] == overall_prediction and item["confidence"] is not None
    ]
    overall_confidence = (
        sum(winning_confidences) / len(winning_confidences)
        if overall_prediction != "UNCERTAIN" and winning_confidences
        else None
    )
    created_at, record_id = create_history(
        text=payload.text,
        prediction=overall_prediction,
        confidence=overall_confidence,
        models=predictions,
    )
    return AnalyzeResponse(
        id=record_id,
        created_at=created_at,
        prediction=overall_prediction,
        confidence=overall_confidence,
        models=[ModelPrediction(**item) for item in predictions],
    )


@app.get("/api/history", response_model=list[HistoryItem])
def history(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[HistoryItem]:
    records = list_history(limit)
    result = []
    for record in records:
        try:
            predictions = json.loads(record["models_json"])
        except json.JSONDecodeError:
            logger.exception(
                "Stored model results are invalid for history id %s", record["id"]
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="A stored history record could not be read.",
            ) from None
        result.append(
            HistoryItem(
                id=record["id"],
                created_at=record["created_at"],
                text=record["text"],
                prediction=record["prediction"],
                confidence=record["confidence"],
                models=[ModelPrediction(**item) for item in predictions],
            )
        )
    return result
