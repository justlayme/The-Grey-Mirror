"""Single change-point detection with a circular-block-permutation null.

Mirrors the protocol Grey Mirror publishes for its turning-point benchmark: a candidate is reported
only when it stands out from the thread's own normal rhythm, and the null is tested by circular
block permutation. The same detector (same pre-registered settings) is applied to every signal layer
so that differences in results come from the signal, not the detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Pre-registered settings. Fixed before any Jev output was seen; not tuned on the benchmark.
ALPHA = 0.01          # permutation p-value threshold
MIN_EFFECT = 0.8      # |mean shift| / pooled within-segment SD (Cohen's d)
MIN_SEGMENT = 10      # days on each side of a candidate split
BLOCK = 7             # block length (days) for the circular block permutation
N_PERM = 999


@dataclass
class Detection:
    detected: bool
    day: int | None
    direction: int | None  # +1 warmer after the change, -1 cooler
    p_value: float
    effect_size: float
    stat: float


def _interp_nans(x: np.ndarray) -> np.ndarray:
    x = x.astype(float).copy()
    bad = np.isnan(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _max_t(x: np.ndarray, m: int) -> tuple[float, int]:
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    best, arg = -1.0, m
    for k in range(m, n - m + 1):
        n1, n2 = k, n - k
        s1, s2 = cs[k], cs[n] - cs[k]
        q1, q2 = cs2[k], cs2[n] - cs2[k]
        mu1, mu2 = s1 / n1, s2 / n2
        ss = (q1 - n1 * mu1 * mu1) + (q2 - n2 * mu2 * mu2)
        var = max(ss / (n - 2), 1e-12)
        t = abs(mu1 - mu2) / np.sqrt(var * (1 / n1 + 1 / n2))
        if t > best:
            best, arg = t, k
    return float(best), int(arg)


def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    shifted = np.roll(x, -int(rng.integers(0, n)))
    blocks = [shifted[i:i + block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def detect(series, *, seed: int = 0, alpha: float = ALPHA, min_effect: float = MIN_EFFECT,
           min_segment: int = MIN_SEGMENT, block: int = BLOCK, n_perm: int = N_PERM) -> Detection:
    x = _interp_nans(np.asarray(series, dtype=float))
    if np.allclose(x, x[0]):
        return Detection(False, None, None, 1.0, 0.0, 0.0)
    t_obs, k = _max_t(x, min_segment)
    rng = np.random.default_rng(seed)
    exceed = sum(_max_t(_block_permute(x, block, rng), min_segment)[0] >= t_obs for _ in range(n_perm))
    p = (1 + exceed) / (1 + n_perm)
    before, after = x[:k], x[k:]
    pooled = np.sqrt(((len(before) - 1) * before.var(ddof=1) + (len(after) - 1) * after.var(ddof=1))
                     / (len(x) - 2))
    d = float((after.mean() - before.mean()) / max(pooled, 1e-9))
    ok = p < alpha and abs(d) >= min_effect
    return Detection(bool(ok), int(k) if ok else None, (1 if d > 0 else -1) if ok else None,
                     float(p), d, t_obs)


def composite(columns: dict[str, np.ndarray], signs: dict[str, int]) -> np.ndarray:
    """Average of per-thread z-scored signals with pre-specified signs (+1 = higher is warmer)."""
    zs = []
    for name, sgn in signs.items():
        v = _interp_nans(np.asarray(columns[name], dtype=float))
        sd = v.std()
        zs.append(sgn * (v - v.mean()) / (sd if sd > 1e-9 else 1.0))
    return np.mean(zs, axis=0)


def as_dict(d: Detection) -> dict:
    return asdict(d)
