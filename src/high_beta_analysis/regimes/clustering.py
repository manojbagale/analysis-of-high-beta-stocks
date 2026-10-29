"""Fit PCA and K-means on training data; transform unseen sessions separately."""
from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_regimes(train, test, numeric_columns: list[str], *, n_clusters: int = 3,
                components: int = 2, seed: int = 42):
    if not numeric_columns:
        raise ValueError("Provide numeric features for regime detection")
    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=min(components, len(numeric_columns)))),
    ])
    train_emb = prep.fit_transform(train[numeric_columns])
    test_emb = prep.transform(test[numeric_columns])
    cluster = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(train_emb)
    return {
        "train_labels": cluster.labels_,
        "test_labels": cluster.predict(test_emb),
        "train_embedding": train_emb, "test_embedding": test_emb,
        "explained_variance_ratio": prep.named_steps["pca"].explained_variance_ratio_,
        "transform": prep, "cluster": cluster,
    }
