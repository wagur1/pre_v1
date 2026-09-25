"""Bjøntegaard Delta Rate (BD-Rate) computation for Task Accuracy (mAP / Top-1).

Computes the percentage bitrate savings at equivalent downstream task accuracy.
Negative BD-Rate indicates superior compression efficiency.
"""
from __future__ import annotations

import numpy as np


def bd_rate(
    rate_anchor: list[float] | np.ndarray,
    metric_anchor: list[float] | np.ndarray,
    rate_test: list[float] | np.ndarray,
    metric_test: list[float] | np.ndarray,
) -> float:
    """Compute BD-Rate between Anchor and Test curves.

    Args:
        rate_anchor: List of bitrates (or bpp) for anchor codec.
        metric_anchor: List of task accuracy values (e.g. mAP) for anchor.
        rate_test: List of bitrates for proposed preprocessed pipeline.
        metric_test: List of task accuracy values for proposed pipeline.

    Returns:
        BD-Rate in percent (e.g. -18.5 means 18.5% bitrate reduction).
    """
    ra = np.asarray(rate_anchor, dtype=np.float64)
    ma = np.asarray(metric_anchor, dtype=np.float64)
    rt = np.asarray(rate_test, dtype=np.float64)
    mt = np.asarray(metric_test, dtype=np.float64)

    # Convert rate to log scale
    lra = np.log(np.maximum(ra, 1e-8))
    lrt = np.log(np.maximum(rt, 1e-8))

    # Fit 3rd order polynomials: log(rate) = f(metric)
    deg = min(3, len(ma) - 1)
    if deg < 1:
        return 0.0

    # Ensure strictly increasing order for integration
    idx_a = np.argsort(ma)
    idx_t = np.argsort(mt)
    ma, lra = ma[idx_a], lra[idx_a]
    mt, lrt = mt[idx_t], lrt[idx_t]

    poly_a = np.polyfit(ma, lra, deg)
    poly_t = np.polyfit(mt, lrt, deg)

    # Integration interval: common overlap of task metrics
    min_m = max(ma[0], mt[0])
    max_m = min(ma[-1], mt[-1])

    if max_m <= min_m:
        # No overlap in accuracy range
        return 0.0

    # Integrate polynomial curves
    poly_int_a = np.polyint(poly_a)
    poly_int_t = np.polyint(poly_t)

    int_a = np.polyval(poly_int_a, max_m) - np.polyval(poly_int_a, min_m)
    int_t = np.polyval(poly_int_t, max_m) - np.polyval(poly_int_t, min_m)

    avg_diff = (int_t - int_a) / (max_m - min_m)
    bd = (np.exp(avg_diff) - 1.0) * 100.0
    return float(bd)
