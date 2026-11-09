import unittest

import numpy as np
import pandas as pd

from src.algorithms import (
    ALGORITHMS,
    BayesianLogisticRegression,
    dataset_summary,
    run_algorithm,
    run_hmm_algorithms,
)


def sample_frame() -> pd.DataFrame:
    records = []
    for label in (0, 1):
        for index in range(24):
            text = (
                f"market report class{label} company earns {index + 10} percent "
                + ("verified revenue " if label == 0 else "claimed losses!!! ")
                + "stock"
                * (index % 4 + 1)
            )
            records.append({"text": text, "label": label})
    return pd.DataFrame(records)


class AlgorithmTests(unittest.TestCase):
    def test_hmm_forward_and_backward_likelihoods_match(self):
        result = run_hmm_algorithms()
        self.assertAlmostEqual(result["forward_likelihood"], result["backward_start_likelihood"])
        self.assertEqual(len(result["viterbi_hidden_state_path"]), 3)

    def test_dataset_summary_counts_both_classes(self):
        summary = dataset_summary(sample_frame())
        self.assertEqual(summary["rows"], 48)
        self.assertEqual(summary["label_counts"], {"REAL": 24, "FAKE": 24})
        self.assertEqual(summary["empty_claims"], 0)

    def test_every_workbench_algorithm_runs(self):
        frame = sample_frame()
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                result = run_algorithm(name, frame)
                self.assertIsInstance(result, dict)

    def test_bayesian_logistic_probabilities_are_valid(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = ["stable earnings report", "steady company revenue"] * 8 + [
            "fraudulent fake market claim",
            "false invented losses",
        ] * 8
        labels = np.asarray([0] * 16 + [1] * 16)
        matrix = TfidfVectorizer(min_df=1).fit_transform(texts)
        model = BayesianLogisticRegression().fit(matrix, labels)
        probabilities = model.predict_proba(matrix)
        self.assertEqual(probabilities.shape, (32, 2))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
