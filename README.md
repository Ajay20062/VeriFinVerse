# VeriFinVerse

VeriFinVerse is an explainable, wording-based classifier for **short financial claims**. It learns from Fin-Fact fact-checker verdicts and returns a REAL/FAKE prediction, model confidence, class probabilities, and the TF-IDF features that influenced that prediction. It uses Flask, CSV files, and scikit-learn; it does not use a database, FastAPI, or a deep-learning framework.

> **Limitations:** This is a machine-learning classification demo, not fact verification. It does not verify full articles or current facts, is not financial advice, and is not an authoritative fact-checking service. Feature explanations describe the model's learned weights, not whether a claim is factually true. The model was trained on short claims and may not generalize to other writing styles or topics. Near-duplicate wording may remain across splits. Confidence near 50% means the model is unsure; confidence is not a guarantee of correctness.

## Dataset and license note

The preparation script uses the official [Fin-Fact JSON dataset](https://raw.githubusercontent.com/IIT-DM/Fin-Fact/FinFact/finfact.json). Source `true` labels become REAL, `false` labels become FAKE, and `NEI` records are excluded. The fact-check URL is preserved as `source_url` in `data/raw/finfact_claims.csv`. FiQA and Financial PhraseBank are not used because they contain sentiment labels, not truth-verdict labels.

There is a license discrepancy: the Fin-Fact GitHub repository states MIT, while its Hugging Face dataset card states Apache-2.0. Review the dataset source and applicable terms before redistribution or commercial use.

## Setup

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build the dataset and models

Run these commands from the repository root:

```bash
python -m src.prepare_finfact
python -m src.preprocess
python -m src.train --model all
python -m src.evaluate
python app.py
```

The data-preparation command downloads Fin-Fact JSON to `data/raw/finfact.json` when it is missing, then writes `data/raw/finfact_claims.csv` with `article,label,source_url` columns. It accepts `--input` and `--output` paths.

Preprocessing normalizes labels, lowercases and cleans claim text into `tfidf_text`, drops empty/duplicate claims and claims with conflicting verdicts, and creates stratified 70/15/15 train/validation/test splits using seed 42. Each split has `text,label,tfidf_text` columns. Override paths and the seed using `python -m src.preprocess --help`.

Training fits TF-IDF (unigrams and bigrams) pipelines with Logistic Regression, Complement Naive Bayes, Bayesian Logistic Regression (MAP with a diagonal Laplace posterior approximation), a linear SVM, CART, AdaBoost, and a soft-voting ensemble. Validation macro-F1 is printed and models are stored in `models/*.joblib`. Select one with `--model logistic`, `--model svm`, `--model cart`, `--model boosting`, `--model ensemble`, or another listed by `python -m src.train --help`.

Evaluation computes test accuracy, FAKE precision/recall/F1, macro-F1, ROC-AUC, confusion matrices, and mean per-claim inference latency after warmups and repeated measurements (up to 100 test claims). It writes `reports/metrics.csv`, `reports/metrics.json`, `reports/confusion_<model>.csv`, and `reports/f1_vs_latency.png`. Metrics are produced from the data and trained models; no results are hard-coded.

The Flask app runs at [http://127.0.0.1:5000](http://127.0.0.1:5000). A trained model is required for predictions. It can also be started through a WSGI server by importing `app` from `app.py`; `create_app()` provides an independently configured application instance for tests and deployments.

The startup settings can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `VERIFINVERSE_HOST` | `127.0.0.1` | Address used by the built-in development server. |
| `VERIFINVERSE_PORT` | `5000` | Port used by the built-in development server (1–65535). |
| `VERIFINVERSE_DEBUG` | `false` | Enables Flask debug mode when set to `true`, `1`, `yes`, or `on`. |
| `VERIFINVERSE_LOG_LEVEL` | `INFO` | Python logging level, such as `INFO` or `WARNING`. |
| `VERIFINVERSE_MODEL_DIR` | `models` | Directory containing trained `.joblib` models. |
| `VERIFINVERSE_DATASET_PATH` | `data/raw/finfact_claims.csv` | Dataset used by the API and workbench. |
| `VERIFINVERSE_HISTORY_PATH` | `data/history.csv` | CSV file used to store successful predictions. |

Relative file paths are resolved from the repository root. Request logs include the HTTP method, path, status code, and elapsed time; query parameters and request bodies are not logged.

## API

- `GET /` — serves the browser interface.
- `GET /api/v1/models` — returns the model names present in the configured model directory.
- `POST /api/v1/predict` — accepts `{"text":"...","model":"logistic"}` and returns `verdict`, `confidence`, `probabilities`, `signals`, `known_words`, and `latency_ms`. Empty text or text longer than 1,000 characters returns HTTP 400; an unavailable model name returns HTTP 404. Successful predictions are appended to the configured history CSV.
- `POST /api/v1/predict/all` — accepts `{"text":"..."}` and returns the same prediction and explanation fields for every available model in a `results` array. The browser uses this route to compare all model verdicts and confidence values; successful results are recorded individually in history.
- `GET /api/v1/history?limit=20` — returns the newest history rows first; `limit` is capped at 200.

The versioned API also includes `/api/v1/algorithms`, `/api/v1/algorithms/run`, and `/api/v1/dataset/summary`. The previous unversioned JSON routes remain available as compatibility aliases.

For linear models, a signal is a vectorizer feature's TF-IDF value multiplied by its model weight (for Complement Naive Bayes, the difference between FAKE and REAL log-probability weights). Positive contributions push toward FAKE; negative contributions push toward REAL. For CART, boosting, and the voting ensemble, signals are single-feature probability changes relative to an empty feature vector and do not capture interactions. Features may include bigrams as well as individual words.

## Algorithm workbench

Open `/workbench` to inspect the loaded Fin-Fact CSV and run the algorithm demonstrations. `GET /dataset/summary` reports dataset counts and descriptive text statistics. `GET /algorithms` lists demonstrations; `POST /algorithms/run` accepts `{"algorithm":"kmeans"}` and returns computed output. The workbench includes polynomial curve fitting, linear regression, Bayesian model averaging, boosting, CART, GAM, soft-voting ensemble learning, K-Means, Gaussian mixture, hierarchical clustering, and PCA. Regression and unsupervised demonstrations use numeric features derived from claim length, token diversity, digits, and punctuation; clustering does not use verdict labels.

The HMM demo runs the forward, backward, and Viterbi algorithms on a small illustrative two-state HMM because Fin-Fact claims are independent records, not time-ordered observations. The linear regression and polynomial fit use binary labels as a numeric target and are demonstrations, not suitable REAL/FAKE classifiers. Bayesian model averaging uses a simple equal-prior posterior approximation based on validation log scores, not a fully specified Bayesian prior over data-generating models. Dataset loading is available from CSV with `python -m src.algorithms --input path/to/claims.csv --algorithm all`.
