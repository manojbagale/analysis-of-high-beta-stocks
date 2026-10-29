"""Optional PyTorch MLP to illustrate logits, BCE loss and minibatches.

Input must already be train-only preprocessed; sigmoid outputs are uncalibrated.
"""
from __future__ import annotations


def train_mlp(x_train, y_train, x_test, *, epochs: int = 30, seed: int = 42,
              hidden: int = 32, lr: float = 1e-3):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is optional: pip install -e '.[nlp]'") from exc
    import numpy as np

    torch.manual_seed(seed)
    x = torch.tensor(np.asarray(x_train, dtype=np.float32))
    y = torch.tensor(np.asarray(y_train, dtype=np.float32).reshape(-1, 1))
    x_out = torch.tensor(np.asarray(x_test, dtype=np.float32))
    network = nn.Sequential(
        nn.Linear(x.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        network.train()
        for inds in torch.randperm(len(x)).split(64):
            optimizer.zero_grad()
            loss = loss_fn(network(x[inds]), y[inds])
            loss.backward()
            optimizer.step()
    network.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(network(x_out)).flatten().numpy()
    return network, probabilities
