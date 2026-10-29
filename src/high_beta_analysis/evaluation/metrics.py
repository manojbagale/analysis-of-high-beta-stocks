"""Class-imbalance metrics; distinguish raw decision scores from probabilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def classifier_metrics(y_true, y_pred, scores, *, probabilities: bool) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    result = {
        "samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, scores))
                  if len(np.unique(y_true)) == 2 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores))
                   if len(np.unique(y_true)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y_true, scores))
                 if probabilities else float("nan"),
    }
    return result
