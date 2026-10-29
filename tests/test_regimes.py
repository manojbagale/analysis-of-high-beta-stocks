import numpy as np
import pandas as pd

from high_beta_analysis.regimes.clustering import fit_regimes


def test_pca_regimes_fit_train_only():
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(size=(35, 3)), columns=["a", "b", "c"])
    test = pd.DataFrame(rng.normal(size=(4, 3)), columns=["a", "b", "c"])
    out = fit_regimes(train, test, ["a", "b", "c"], n_clusters=2)
    assert len(out["train_labels"]) == 35
    assert len(out["test_labels"]) == 4
    assert out["test_embedding"].shape == (4, 2)
