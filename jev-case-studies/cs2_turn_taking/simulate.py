"""Latency-aware simulation of a voice agent's endpointing policy.

Timeline of one pause (t = 0 is the moment the caller stops speaking):
  * the agent's voice-activity detector confirms silence at t = VAD_S and, for model-gated
    policies, sends the transcript to the model then
  * the model answer arrives at VAD_S + latency (measured per request, not assumed)
  * the agent speaks at its decision time t_d, unless the caller has already resumed

For a HOLD event with pause s: the agent interrupts iff t_d < s.
For a SHIFT event: the agent's response latency is t_d (the caller is waiting for it).

Policies (model-gated families are pooled; the frontier and operating points use the union)
  silence(T):              t_d = T
  gate(theta, T_fb):       t_d = VAD_S + latency  if p >= theta  else  T_fb
  adaptive(a, b):          t_d = max(VAD_S + latency, a + b * (1 - p))   (timeout shrinks as p grows)
Operating points are chosen on the dev sample (minimise mean response latency subject to an
interruption-rate ceiling) and then reported on the held-out eval sample.
"""
from __future__ import annotations

import numpy as np

VAD_S = 0.2
SILENCE_GRID = np.round(np.arange(0.2, 3.01, 0.05), 3)
THETA_GRID = np.round(np.arange(0.05, 0.99, 0.01), 3)
FALLBACK_GRID = np.round(np.arange(0.6, 3.01, 0.1), 3)
A_GRID = np.round(np.arange(0.2, 1.21, 0.1), 3)
B_GRID = np.round(np.arange(0.0, 4.01, 0.2), 3)


def outcomes(td: np.ndarray, is_shift: np.ndarray, silence: np.ndarray) -> dict:
    hold = ~is_shift
    interrupts = (td < silence) & hold
    lat = td[is_shift]
    return {
        "interruption_rate": float(interrupts.sum() / max(hold.sum(), 1)),
        "median_latency_s": float(np.median(lat)) if len(lat) else float("nan"),
        "mean_latency_s": float(np.mean(lat)) if len(lat) else float("nan"),
        "p90_latency_s": float(np.percentile(lat, 90)) if len(lat) else float("nan"),
    }


def silence_curve(is_shift, silence):
    return [{"T": float(T), **outcomes(np.full(len(silence), T), is_shift, silence)} for T in SILENCE_GRID]


def decision_times(point, p, latency_s, n=None):
    if "T" in point:
        return np.full(n if n is not None else len(p), point["T"])
    if point.get("family") == "adaptive":
        return np.maximum(VAD_S + latency_s, point["a"] + point["b"] * (1 - p))
    return np.where(p >= point["theta"], VAD_S + latency_s, point["fallback"])


def gated_points(p, latency_s, is_shift, silence):
    pts = []
    for fb in FALLBACK_GRID:
        for th in THETA_GRID:
            pt = {"family": "gate", "theta": float(th), "fallback": float(fb)}
            pts.append({**pt, **outcomes(decision_times(pt, p, latency_s), is_shift, silence)})
    for a in A_GRID:
        for b in B_GRID:
            pt = {"family": "adaptive", "a": float(a), "b": float(b)}
            pts.append({**pt, **outcomes(decision_times(pt, p, latency_s), is_shift, silence)})
    return pts


def frontier(points, x="interruption_rate", y="mean_latency_s"):
    """Lower-left Pareto frontier (minimise both)."""
    pts = sorted(points, key=lambda r: (r[x], r[y]))
    out, best = [], float("inf")
    for r in pts:
        if r[y] < best - 1e-9:
            out.append(r)
            best = r[y]
    return out


def best_under(points, max_interrupt, y="mean_latency_s"):
    ok = [r for r in points if r["interruption_rate"] <= max_interrupt]
    return min(ok, key=lambda r: r[y]) if ok else None


def apply_policy(point, p, latency_s, is_shift, silence):
    return outcomes(decision_times(point, p, latency_s, n=len(silence)), is_shift, silence)
