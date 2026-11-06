"""Shared text features and label ordering."""

from sklearn.feature_extraction.text import TfidfVectorizer

LABEL_TO_ID = {"REAL": 0, "FAKE": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def build_tfidf(
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
        min_df=1,
    )
