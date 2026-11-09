"""Runnable machine-learning algorithm workbench for Fin-Fact claims."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.common import normalize_label

DEFAULT_DATASET = Path("data/raw/finfact_claims.csv")

ALGORITHMS = {
    "polynomial_curve_fitting": "Fit a degree-two polynomial to claim length and the binary verdict. This is a regression demonstration, not a fact-check.",
    "linear_regression": "Fit a least-squares baseline to numeric claim features and the binary verdict.",
    "kmeans": "Cluster standardized numeric claim features without using verdict labels.",
    "gaussian_mixture": "Fit a two-component Gaussian mixture to standardized numeric claim features.",
    "hierarchical_clustering": "Agglomerative clustering of a bounded sample of numeric claim features.",
    "pca": "Project standardized claim features to two principal components.",
    "hmm": "Run forward, backward, and Viterbi on a documented illustrative two-state HMM.",
    "bayesian_model_averaging": "Average classifier predictions using a validation log-score based posterior weight.",
    "boosting": "Train an AdaBoost CART ensemble on numeric claim features.",
    "cart": "Train a CART decision tree on numeric claim features.",
    "generalized_additive_model": "Fit a spline-based logistic GAM to claim word count.",
    "ensemble_learning": "Combine logistic regression, SVM, and CART probabilities by soft voting.",
}


class BayesianLogisticRegression(ClassifierMixin, BaseEstimator):
    """Logistic MAP estimator with a Gaussian prior and diagonal Laplace posterior."""

    def __init__(self, prior_precision: float = 1.0, max_iter: int = 100):
        self.prior_precision = prior_precision
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("Bayesian logistic regression requires labels 0 and 1.")
        n_features = X.shape[1]

        def objective(parameters):
            logits = np.asarray(X @ parameters[:-1]).ravel() + parameters[-1]
            weights = parameters[:-1]
            loss = np.logaddexp(0, logits).sum() - np.dot(y, logits)
            loss += 0.5 * self.prior_precision * np.dot(weights, weights)
            residual = expit(logits) - y
            gradient = np.concatenate(
                [
                    np.asarray(X.T @ residual).ravel() + self.prior_precision * weights,
                    [residual.sum()],
                ]
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(n_features + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        if not result.success and result.status != 1:
            raise RuntimeError(f"Bayesian logistic regression failed: {result.message}")
        self.n_iter_ = int(result.nit)
        self.converged_ = bool(result.success)
        if not self.converged_:
            warnings.warn(
                f"Bayesian logistic regression did not converge in {self.max_iter} iterations.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.coef_ = result.x[:-1].reshape(1, -1)
        self.intercept_ = np.asarray([result.x[-1]])
        self.classes_ = np.asarray([0, 1])
        logits = np.asarray(X @ result.x[:-1]).ravel() + result.x[-1]
        variance = expit(logits) * (1 - expit(logits))
        diagonal_hessian = np.asarray(X.multiply(X).T @ variance).ravel()
        self.posterior_variance_ = 1.0 / (
            diagonal_hessian + self.prior_precision + 1e-8
        )
        self.intercept_variance_ = 1.0 / (variance.sum() + 1e-8)
        return self

    def predict_proba(self, X):
        mean = np.asarray(X @ self.coef_[0]).ravel() + self.intercept_[0]
        variance = np.asarray(X.multiply(X) @ self.posterior_variance_).ravel()
        variance += self.intercept_variance_
        probability_fake = expit(mean / np.sqrt(1 + np.pi * variance / 8))
        return np.column_stack([1 - probability_fake, probability_fake])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def build_models() -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer_args = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
        "stop_words": "english",
    }

    def text_pipeline(classifier):
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_args)),
                ("classifier", classifier),
            ]
        )

    def calibrated_svm():
        return CalibratedClassifierCV(
            estimator=SVC(kernel="linear", class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3,
            ensemble=False,
        )

    base_estimators = [
        ("logistic", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ("svm", calibrated_svm()),
        ("cart", DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=42)),
    ]
    return {
        "logistic": text_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000)),
        "naive_bayes": text_pipeline(ComplementNB()),
        "bayesian_logistic": text_pipeline(BayesianLogisticRegression()),
        "svm": text_pipeline(calibrated_svm()),
        "cart": text_pipeline(
            DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=42)
        ),
        "boosting": text_pipeline(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
                n_estimators=50,
                random_state=42,
            )
        ),
        "ensemble": text_pipeline(VotingClassifier(estimators=base_estimators, voting="soft")),
    }


def load_dataset(path: Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load a labeled claim CSV, accepting either source or processed column names."""
    frame = pd.read_csv(path)
    text_column = next((name for name in ("article", "text", "claim") if name in frame.columns), None)
    if text_column is None or "label" not in frame.columns:
        raise ValueError("Dataset CSV must contain a claim column (article/text/claim) and label.")
    loaded = frame[[text_column, "label"]].rename(columns={text_column: "text"}).copy()
    loaded["text"] = loaded["text"].fillna("").astype(str).str.strip()
    loaded["label"] = loaded["label"].map(normalize_label)
    loaded = loaded.dropna(subset=["label"])
    loaded = loaded[loaded["text"].ne("")].reset_index(drop=True)
    if loaded.empty or loaded["label"].nunique() != 2:
        raise ValueError("Dataset must contain non-empty claims from both REAL and FAKE classes.")
    loaded["label"] = loaded["label"].astype(int)
    loaded.attrs["source_columns"] = list(frame.columns)
    return loaded


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    word_counts = frame["text"].map(lambda text: len(text.split()))
    return {
        "rows": int(len(frame)),
        "columns": frame.attrs.get("source_columns", list(frame.columns)),
        "label_counts": {
            "REAL": int((frame["label"] == 0).sum()),
            "FAKE": int((frame["label"] == 1).sum()),
        },
        "mean_words": float(word_counts.mean()),
        "median_words": float(word_counts.median()),
        "mean_characters": float(frame["text"].str.len().mean()),
        "min_words": int(word_counts.min()),
        "max_words": int(word_counts.max()),
        "unique_claims": int(frame["text"].nunique()),
        "empty_claims": int(frame["text"].str.strip().eq("").sum()),
    }


def _numeric_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    texts = frame["text"].astype(str)
    words = texts.map(str.split)
    matrix = np.column_stack(
        [
            words.map(len).to_numpy(dtype=float),
            texts.str.len().to_numpy(dtype=float),
            words.map(lambda items: len(set(items)) / max(1, len(items))).to_numpy(dtype=float),
            texts.str.count(r"[!?$%]").to_numpy(dtype=float),
            texts.str.count(r"\d").to_numpy(dtype=float),
        ]
    )
    return matrix, frame["label"].to_numpy(dtype=int)


def _split_numeric(frame: pd.DataFrame):
    features, labels = _numeric_features(frame)
    return train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )


def _regression_summary(actual, predicted) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "r2": float(1 - np.sum((actual - predicted) ** 2) / max(np.sum((actual - actual.mean()) ** 2), 1e-12)),
    }


def run_hmm_algorithms() -> dict[str, Any]:
    """Calculate exact forward/backward likelihood and most-likely hidden path."""
    initial = np.asarray([0.6, 0.4])
    transition = np.asarray([[0.7, 0.3], [0.4, 0.6]])
    emission = np.asarray([[0.5, 0.5], [0.1, 0.9]])
    observations = [0, 1, 0]

    forward = np.zeros((len(observations), 2))
    forward[0] = initial * emission[:, observations[0]]
    for t in range(1, len(observations)):
        forward[t] = (forward[t - 1] @ transition) * emission[:, observations[t]]

    backward = np.zeros_like(forward)
    backward[-1] = 1.0
    for t in range(len(observations) - 2, -1, -1):
        backward[t] = transition @ (emission[:, observations[t + 1]] * backward[t + 1])

    log_probability = np.log(initial) + np.log(emission[:, observations[0]])
    path_scores = np.zeros_like(forward)
    backpointers = np.zeros_like(forward, dtype=int)
    path_scores[0] = log_probability
    log_transition = np.log(transition)
    log_emission = np.log(emission)
    for t in range(1, len(observations)):
        scores = path_scores[t - 1][:, None] + log_transition
        backpointers[t] = np.argmax(scores, axis=0)
        path_scores[t] = np.max(scores, axis=0) + log_emission[:, observations[t]]
    path = [int(np.argmax(path_scores[-1]))]
    for t in range(len(observations) - 1, 0, -1):
        path.append(int(backpointers[t, path[-1]]))
    path.reverse()

    return {
        "observations": observations,
        "forward_likelihood": float(forward[-1].sum()),
        "forward_state_probabilities": forward.tolist(),
        "backward_start_likelihood": float(np.dot(initial * emission[:, observations[0]], backward[0])),
        "backward_state_values": backward.tolist(),
        "viterbi_hidden_state_path": path,
        "note": "Illustrative two-state HMM parameters; Fin-Fact claims are not sequential observations.",
    }


def run_algorithm(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}")
    if name == "hmm":
        return run_hmm_algorithms()

    features, labels = _numeric_features(frame)
    if name == "polynomial_curve_fitting":
        word_counts = features[:, 0].reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            word_counts, labels, test_size=0.25, random_state=42, stratify=labels
        )
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        return {"degree": 2, **_regression_summary(y_test, predicted), "sample_predictions": predicted[:10].tolist()}

    if name == "linear_regression":
        x_train, x_test, y_train, y_test = _split_numeric(frame)
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(x_train, y_train)
        return _regression_summary(y_test, model.predict(x_test))

    scaled = StandardScaler().fit_transform(features)
    if name == "pca":
        components = PCA(n_components=2, random_state=42).fit(scaled)
        projection = components.transform(scaled)
        sample = projection[:50]
        return {
            "explained_variance_ratio": components.explained_variance_ratio_.tolist(),
            "points": sample.tolist(),
            "labels": labels[: len(sample)].tolist(),
        }

    if name in {"kmeans", "gaussian_mixture", "hierarchical_clustering"}:
        if len(frame) < 3:
            raise ValueError("At least three claims are needed for clustering.")
        sample = scaled[: min(500, len(scaled))]
        if name == "kmeans":
            assignments = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(sample)
        elif name == "gaussian_mixture":
            assignments = GaussianMixture(n_components=2, random_state=42).fit_predict(sample)
        else:
            assignments = AgglomerativeClustering(n_clusters=2).fit_predict(sample)
        return {
            "clusters": int(len(np.unique(assignments))),
            "cluster_sizes": np.bincount(assignments).tolist(),
            "silhouette": float(silhouette_score(sample, assignments)),
            "sample_size": int(len(sample)),
            "note": "Clustering is unsupervised; label values were not used to fit cluster assignments.",
        }

    if name in {"cart", "boosting", "ensemble_learning"}:
        x_train, x_test, y_train, y_test = _split_numeric(frame)
        if name == "cart":
            model = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
        elif name == "boosting":
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
                n_estimators=50,
                random_state=42,
            )
        else:
            model = VotingClassifier(
                [
                    ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                    (
                        "svm",
                        make_pipeline(
                            StandardScaler(),
                            CalibratedClassifierCV(
                                estimator=SVC(random_state=42),
                                method="sigmoid",
                                cv=3,
                                ensemble=False,
                            ),
                        ),
                    ),
                    ("cart", DecisionTreeClassifier(max_depth=8, random_state=42)),
                ],
                voting="soft",
            )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "test_size": int(len(y_test)),
        }

    if name == "generalized_additive_model":
        word_counts = features[:, :1]
        x_train, x_test, y_train, y_test = train_test_split(
            word_counts, labels, test_size=0.25, random_state=42, stratify=labels
        )
        model = make_pipeline(
            SplineTransformer(n_knots=5, degree=3),
            LogisticRegression(max_iter=1000),
        )
        model.fit(x_train, y_train)
        return {
            "accuracy": float(accuracy_score(y_test, model.predict(x_test))),
            "test_size": int(len(y_test)),
            "feature": "claim word count",
        }

    if name == "bayesian_model_averaging":
        x_train, x_test, y_train, y_test = _split_numeric(frame)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        candidates = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "svm": make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    estimator=SVC(random_state=42),
                    method="sigmoid",
                    cv=3,
                    ensemble=False,
                ),
            ),
            "cart": DecisionTreeClassifier(max_depth=8, random_state=42),
        }
        fitted = {}
        log_weights = []
        for model_name, model in candidates.items():
            model.fit(x_train, y_train)
            valid_probs = model.predict_proba(x_valid)
            log_weights.append(-log_loss(y_valid, valid_probs, labels=[0, 1]))
            fitted[model_name] = model
        log_weights_array = np.asarray(log_weights)
        weights = np.exp(log_weights_array - log_weights_array.max())
        weights /= weights.sum()
        averaged = sum(
            weight * fitted[model_name].predict_proba(x_test)
            for weight, model_name in zip(weights, candidates)
        )
        return {
            "accuracy": float(accuracy_score(y_test, averaged.argmax(axis=1))),
            "posterior_weights": dict(zip(candidates, weights.tolist())),
            "note": "Approximate model weights use validation log score and equal prior model probabilities.",
        }

    raise ValueError(f"Algorithm is not implemented: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--algorithm", choices=["all", *ALGORITHMS], default="all")
    args = parser.parse_args()
    frame = load_dataset(args.input)
    print(json.dumps({"dataset": dataset_summary(frame)}, indent=2))
    selected = ALGORITHMS if args.algorithm == "all" else [args.algorithm]
    for name in selected:
        print(json.dumps({"algorithm": name, "result": run_algorithm(name, frame)}, indent=2))
