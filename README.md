# VeriFinVerse

Explainable financial-claim classification: claim/headline text in, REAL/FAKE
class and confidence out, with SHAP word contributions. This research/demo
classifier learns from fact-checker claim verdicts; it does not verify complete
articles and is not financial advice or an authoritative fact-checking service.

## Dataset and labeling assumptions

Training data comes from [Fin-Fact](https://github.com/IIT-DM/Fin-Fact), a
finance-focused fact-checking dataset of claims/headlines, not complete news
articles. Its official source contains `True`, `False`, and `NEI` labels.
[`src/prepare_finfact.py`](src/prepare_finfact.py) downloads the official
dataset, maps `True` to `REAL` and `False` to `FAKE`, and excludes `NEI`
examples because this application is binary. Each exported row retains its
original fact-check URL. See the [Fin-Fact license and attribution](data/raw/FINFACT_LICENSE.txt).
The official GitHub repository says MIT; the Hugging Face dataset card metadata
says Apache-2.0, so that discrepancy is disclosed here and in the attribution
file. Do not interpret claim classification as verification of a full article
or a current financial claim.

FiQA and Financial PhraseBank provide sentiment or related financial language,
not fake/real truth labels. This project intentionally does not relabel them.
They can be used separately for vocabulary/EDA or a distinct sentiment task,
but do not enter the fake/real supervised train/validation/test splits.

## Setup

Run commands from this `verifinverse` directory. Python 3.10+ is recommended.
Create and activate a virtual environment, then install dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the web application, copy `.env.example` to `.env` and set your MySQL
connection values. Create the configured database/schema in MySQL before
starting the API; on startup, the service creates its `predictions` table.
The `.env` file is ignored by Git.

```powershell
Copy-Item .env.example .env
```

For the example configuration, create the database once in MySQL:

```sql
CREATE DATABASE verifinverse CHARACTER SET utf8mb4;
```

For GPU training, install the PyTorch build appropriate for your CUDA version
using the official PyTorch installation selector, then install the remaining
requirements. CPU is supported. First FinBERT run downloads
`ProsusAI/finbert`, so internet access is required at that point.

To reproduce the labeled input CSV, run the preparation script:

```powershell
python -m src.prepare_finfact
```

It creates [data/raw/finfact_claims.csv](data/raw/finfact_claims.csv) with
`article`, `label`, and `source_url` columns. The `article` column contains the
dataset's claim/headline text for compatibility with the current text-model
pipeline. The older [header-only template](data/raw/financial_news_template.csv)
is for user-provided data; it is not training data. Do not put
FiQA/PhraseBank sentiment labels in the fake/real label column.

## Preprocess

If your data uses different column names, replace `article` and `label` below
with the real column names. Omit `--label-map` for recognized text labels such
as `REAL` and `FAKE`; specify a mapping only when the source's documentation
confirms the numeric label meanings.

```powershell
python -m src.preprocess `
  --input data/raw/finfact_claims.csv `
  --text-column article `
  --label-column label `
  --output-dir data/processed `
  --seed 42
```

The [preprocessing pipeline](src/preprocess.py) reads the labeled source through
the [CSV loader](src/data_loader.py) and creates
`data/processed/train.csv`, `data/processed/val.csv`, and
`data/processed/test.csv`. Each generated split contains `text`, normalized
`label`, and `tfidf_text` columns. Splits are stratified and use a fixed seed;
duplicates and empty records are removed before splitting. These files are
generated from your populated source and are intentionally not checked in.
FinBERT reads `text`; the classical models read `tfidf_text`.

## Train all models

Classical TF-IDF Logistic Regression and XGBoost read
`data/processed/train.csv`:

```powershell
python -m src.train_classical --model all
```

The [FinBERT training script](src/train_finbert.py) reads the same
`train.csv` plus `val.csv`. Adjust epochs, maximum token length, and batch size
for your hardware:

```powershell
python -m src.train_finbert --epochs 2 --max-length 128 --batch-size 8
```

Artifacts are written to `models/`. FinBERT automatically uses CUDA when
available and CPU otherwise.

## Evaluate

The [evaluation script](src/evaluate.py) reads `data/processed/test.csv` by
default and computes accuracy, precision, recall, F1, ROC-AUC, confusion
matrices, and measured mean per-article inference latency. Latency uses warmups
and repeated model calls on up to 100 test articles; it excludes SHAP
computation. Metrics are derived from the supplied test split and saved as
CSV/JSON; no results are pre-populated.

```powershell
python -m src.evaluate --models logistic xgboost finbert --warmups 3 --latency-repeats 3
```

Reports are saved to `reports/`: `metrics.csv`, `metrics.json`, one confusion
matrix per model, and `f1_vs_latency.png` (F1/latency trade-off).

## Run the application

Start FastAPI (from this directory):

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000` for the HTML/CSS/JavaScript interface or
`http://127.0.0.1:8000/docs` for the API documentation. The interface and API
are served from the same Python/FastAPI process. `GET /models` lists trained
artifacts, `POST /predict` accepts `{"text":"...","model":"logistic"}`, and
`GET /history?limit=20` reads recent analyses from MySQL. Prediction results
include the verdict, confidence, SHAP signals, and latency. This model predicts
from claim/headline text and does not independently verify sources or current
facts.

## Notes

- `REAL` is class 0 and `FAKE` is class 1 throughout training and evaluation.
- Use a source-appropriate split or group by publisher/time if near-duplicate
  articles or source leakage are concerns; a random stratified split alone
  cannot guarantee independent sources.
- SHAP feature contributions explain model behavior, not factual verification.
- XGBoost and SHAP performance can vary by platform and package version. FinBERT
  fine-tuning on CPU may be slow; reduce batch size or epochs for a laptop.
