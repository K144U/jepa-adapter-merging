"""Evaluation protocols (plan 1.5).

P1 (primary): logistic-regression linear probe re-fit per task on the frozen
merged encoder, fixed budget (sample cap + iteration cap, identical for every
cell). P2 (secondary): reuse the task's original fine-tuned head.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

P1_TRAIN_CAP = 10_000   # fixed probe budget (samples)
P1_MAX_ITER = 1_000
P1_C = 1.0


@torch.no_grad()
def extract_features(encoder, ds, batch_size=256, num_workers=4,
                     max_n=None) -> tuple:
    dev = next(encoder.parameters()).device
    encoder.eval()
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    feats, ys = [], []
    n = 0
    for x, y in loader:
        feats.append(encoder(x.to(dev)).float().cpu())
        ys.append(y)
        n += len(y)
        if max_n and n >= max_n:
            break
    X = torch.cat(feats)
    Y = torch.cat(ys)
    if max_n:
        X, Y = X[:max_n], Y[:max_n]
    return X, Y


def p1_probe(train_X, train_y, test_X, test_y, seed: int = 0) -> float:
    """Fixed-budget logistic-regression probe accuracy."""
    from sklearn.linear_model import LogisticRegression
    Xtr = train_X.numpy()
    ytr = train_y.numpy()
    if len(ytr) > P1_TRAIN_CAP:
        idx = np.random.RandomState(seed).permutation(len(ytr))[:P1_TRAIN_CAP]
        Xtr, ytr = Xtr[idx], ytr[idx]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    clf = LogisticRegression(max_iter=P1_MAX_ITER, C=P1_C, n_jobs=-1)
    clf.fit((Xtr - mu) / sd, ytr)
    pred = clf.predict((test_X.numpy() - mu) / sd)
    return float((pred == test_y.numpy()).mean())


def p2_head(head_w: torch.Tensor, head_b: torch.Tensor, test_X, test_y) -> float:
    """Accuracy reusing the task's original fine-tuned head."""
    logits = test_X @ head_w.T.float() + head_b.float()
    return float((logits.argmax(dim=1) == test_y).float().mean())
