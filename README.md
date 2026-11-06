# VeriFinVerse

Financial-news verification demo with a FastAPI backend and responsive
dashboard. The API supports trained Logistic Regression, XGBoost, and local
FinBERT artifacts, and stores successful analyses in SQLite. The dashboard
shows live model availability and never substitutes demo predictions.

## Backend

See [backend/README.md](backend/README.md) for setup and model configuration.
From the `backend` directory, install `requirements.txt` and run:

```powershell
uvicorn app.main:app --reload
```

The API and interactive documentation are available at
`http://localhost:8000` and `http://localhost:8000/docs`.

## Frontend

See [frontend/README.md](frontend/README.md). With the backend running, start
the static frontend from the `frontend` directory:

```powershell
python -m http.server 5173
```

Open `http://localhost:5173`. Model predictions require trained artifacts
configured on the backend; none are included in this repository.
