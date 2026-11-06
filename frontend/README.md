# VeriFinVerse frontend

The frontend is a lightweight, dependency-free dashboard that reads live model
availability and saved analysis results from the FastAPI backend. It does not
include sample predictions.

## Run locally

Start the backend from the `backend` directory as described in `backend/README.md`.
Then, from `frontend`, serve the files on a CORS-allowed local origin:

```powershell
python -m http.server 5173
```

Open <http://localhost:5173>. The API defaults to `http://localhost:8000`.
Change it in the API connection panel; the URL is saved in browser local
storage. The backend's default CORS configuration permits `localhost:5173`.

The frontend uses only browser APIs and can also be hosted as static files.