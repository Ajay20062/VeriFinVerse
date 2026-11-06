# VeriFinVerse

Explainable financial-news classification: article text in, REAL/FAKE class and
confidence out, with SHAP word contributions. This is a research/demo classifier,
not financial advice or an authoritative fact-checking service.

## Dataset and labeling assumptions

The primary training CSV must contain a financial-news text column and a
human-provided binary fake/real label column. Use a finance-focused labeled
fake/real dataset whose provenance and license you have checked. The project
does not bundle or fabricate a dataset. The loader accepts column names and an
explicit source-label mapping. Text values such as `REAL`, `FAKE`, `true`, and
`false` are recognized; numeric labels must be mapped explicitly, for example
`0=FAKE,1=REAL` only after checking the dataset's documentation.

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

For GPU training, install the PyTorch build appropriate for your CUDA version
using the official PyTorch installation selector, then install the remaining
requirements. CPU is supported. First FinBERT run downloads
`ProsusAI/finbert`, so internet access is required at that point.

Start with the [raw CSV template](data/raw/financial_news_template.csv). Before
preprocessing, populate it with one article per row and set `label` to a
human-verified `REAL` or `FAKE` value; a headers-only file is not trainable.
Keep the source dataset's provenance and license with your project records.
Do not put FiQA/PhraseBank sentiment labels in the fake/real label column.

## Preprocess

If your data uses different column names, replace `article` and `label` below
with the real column names. Omit `--label-map` for recognized text labels such
as `REAL` and `FAKE`; specify a mapping only when the source's documentation
confirms the numeric label meanings.

```powershell
python -m src.preprocess `
  --input data/raw/financial_news_template.csv `
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

`POST /predict` accepts `{"text":"...","model":"finbert"}`; model may be
`finbert`, `logistic`, or `xgboost`. Response includes verdict, confidence,
top signed SHAP signals, and prediction latency in milliseconds. First-time
SHAP initialization may take a little while.

In another terminal, with the same virtual environment active:

```powershell
streamlit run app/streamlit_app.py
```

The app includes article verification, confidence, SHAP highlights and bar
chart, a SQLite-backed prediction history page, a recommendation to consult
official sources such as SEBI/RBI, and a clear notice that source evidence
cross-checking is planned future work.

## Notes

- `REAL` is class 0 and `FAKE` is class 1 throughout training and evaluation.
- Use a source-appropriate split or group by publisher/time if near-duplicate
  articles or source leakage are concerns; a random stratified split alone
  cannot guarantee independent sources.
- SHAP feature contributions explain model behavior, not factual verification.
- XGBoost and SHAP performance can vary by platform and package version. FinBERT
  fine-tuning on CPU may be slow; reduce batch size or epochs for a laptop.
