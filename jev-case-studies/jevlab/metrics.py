"""Evaluation metrics: discrimination, calibration, selective prediction, bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _arr(x):
    return np.asarray(x, dtype=float)


def auc(y, p) -> float:
    y, p = _arr(y), _arr(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def aupr(y, p) -> float:
    return float(average_precision_score(_arr(y), _arr(p)))


def brier(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p, eps=1e-6) -> float:
    y, p = _arr(y), np.clip(_arr(p), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(y, p, n_bins: int = 10, strategy: str = "uniform"):
    """Rows of (bin_lo, bin_hi, mean_pred, frac_pos, count)."""
    y, p = _arr(y), _arr(p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & ((p < hi) if i < len(edges) - 2 else (p <= hi))
        if m.sum() == 0:
            continue
        rows.append((float(lo), float(hi), float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def ece(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    rows = reliability(y, p, n_bins, strategy)
    n = sum(r[4] for r in rows)
    return float(sum(abs(r[2] - r[3]) * r[4] for r in rows) / n) if n else float("nan")


def binary_at(y, p, thr: float) -> dict:
    y, p = _arr(y).astype(int), _arr(p)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(thr),
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "recall": rec,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def best_threshold(y, p, metric: str = "accuracy") -> float:
    """Threshold that maximises `metric` (ties -> the one closest to 0.5). Used on dev splits only."""
    p = _arr(p)
    cands = np.unique(np.concatenate([p, [0.5]]))
    best, best_v = 0.5, -1.0
    for t in cands:
        v = binary_at(y, p, t)[metric]
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.5) < abs(best - 0.5)):
            best, best_v = float(t), v
    return best


def selective_curve(y, p, n_points: int = 20):
    """Accuracy when acting only on the most confident fraction of cases (confidence=|p-0.5|)."""
    y, p = _arr(y), _arr(p)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(float)[order]
    out = []
    for cov in np.linspace(1 / n_points, 1.0, n_points):
        k = max(1, int(round(cov * len(y))))
        out.append((float(cov), float(correct[:k].mean())))
    return out


def bootstrap_ci(fn, *arrays, n: int = 2000, seed: int = 7, alpha: float = 0.05):
    """Percentile CI of fn(*resampled arrays); arrays are resampled jointly by row."""
    rng = np.random.default_rng(seed)
    arrays = [_arr(a) for a in arrays]
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = fn(*(a[idx] for a in arrays))
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(fn, y, p_a, p_b, n: int = 2000, seed: int = 11):
    """CI and one-sided p-value for fn(y,p_a) - fn(y,p_b)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = _arr(y), _arr(p_a), _arr(p_b)
    m = len(y)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            diffs.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(fn(y, p_a) - fn(y, p_b)), "ci": [float(lo), float(hi)],
            "p_le_0": float((diffs <= 0).mean())}


def summarize_binary(y, p, thr: float | None = None, with_ci: bool = True) -> dict:
    y, p = _arr(y), _arr(p)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "auc": auc(y, p),
        "aupr": aupr(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece": ece(y, p),
        "ece_quantile": ece(y, p, strategy="quantile"),
        "reliability": reliability(y, p),
        "selective": selective_curve(y, p),
    }
    if thr is not None:
        out["at_threshold"] = binary_at(y, p, thr)
    if with_ci and len(np.unique(y)) == 2:
        out["auc_ci"] = bootstrap_ci(auc, y, p)
        out["ece_ci"] = bootstrap_ci(ece, y, p)
    return out
