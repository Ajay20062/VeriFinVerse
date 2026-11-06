# VeriFinVerse backend

FastAPI service for running trained financial-news classifiers and saving analysis
history in SQLite. The service does not generate placeholder predictions: when
models are not configured, `/api/analyze` returns HTTP 503 with their statuses.

## Run locally

From the `backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`; interactive documentation is
at `http://localhost:8000/docs`.

## Connect trained models

Set environment variables to local model artifacts before starting the server:

| Variable | Model | Artifact format |
| --- | --- | --- |
| `VERIFINVERSE_LOGISTIC_REGRESSION_MODEL` | Logistic Regression | Joblib-serialized scikit-learn estimator or pipeline |
| `VERIFINVERSE_XGBOOST_MODEL` | XGBoost | Joblib-serialized estimator or pipeline |
| `VERIFINVERSE_FINBERT_MODEL` | FinBERT | Local Hugging Face model/tokenizer directory |

The Logistic Regression and XGBoost pipelines must accept raw text in
`predict([text])`. `predict_proba([text])` and `classes_` are used for confidence
and class probabilities when available. Install compatible `scikit-learn` and,
for XGBoost artifacts, `xgboost` versions matching the training environment.
FinBERT additionally requires compatible `transformers` and `torch` packages.
FinBERT is loaded from the local directory only; the server will not download
model weights at startup.

Model labels such as `real`/`true` and `fake`/`false` are normalized to `REAL`
and `FAKE`. Other class names are returned in uppercase. The combined
prediction is the majority vote across available requested models; tied votes
return `UNCERTAIN`. Combined confidence is the mean confidence of models voting
for the winning label, and is omitted when those models do not provide
probabilities.

## API

- `GET /api/health` — process health.
- `GET /api/models` — load status for each configured model.
- `POST /api/analyze` — analyze text and persist successful results.
- `GET /api/history?limit=20` — list recent analyses (limit 1–100).

Example request:

```json
{
  "text": "Paste the financial news article or headline here."
}
```

Optionally pass a `models` array containing one or more of `logistic_regression`,
`xgboost`, or `finbert`. If an explicitly requested model is unavailable, the
request fails with HTTP 503 and includes current model statuses.

History is stored in `backend/data/history.sqlite3` by default. Set
`VERIFINVERSE_DATABASE_PATH` to use a different SQLite file. CORS allows
`localhost:3000` and `localhost:5173` by default; override this with a
comma-separated `VERIFINVERSE_CORS_ORIGINS` list.
