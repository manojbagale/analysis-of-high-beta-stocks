"""Classical models with preprocessing fitted entirely inside training folds."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def preprocessing(numeric_features: list[str], *, scale: bool = True) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_features),
        ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"]),
    ], remainder="drop", sparse_threshold=0)


def make_classifier(name: str, numeric_features: list[str], seed: int = 42) -> Pipeline:
    if name == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
    elif name == "decision_tree":
        estimator = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=15, class_weight="balanced", random_state=seed
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        )
    elif name == "svm_rbf":
        # sklearn's SVC(probability=True) performs internal CV, not date-aware CV.
        # Keep probability=False and report ranking metrics from decision_function.
        estimator = SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=False, random_state=seed
        )
    elif name == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=300,
            early_stopping=False, random_state=seed
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("preprocess", preprocessing(
            numeric_features, scale=name not in {"dummy", "decision_tree", "random_forest"}
        )),
        ("model", estimator),
    ])


def make_regression(numeric_features: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessing(numeric_features)),
        ("regression", LinearRegression()),
    ])
