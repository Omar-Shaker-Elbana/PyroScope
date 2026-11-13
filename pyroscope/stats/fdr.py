"""
Multiple-testing correction for the daily anomaly z-scores.

Context
-------
The anomaly detector computes one z-score per AOI per day against a
seasonal-naive (or, once shipped, harmonic-regression) climatology and
flags a day "elevated" at z >= PYRO_ANOM_Z_THRESHOLD (default 2.0). Over
a multi-year run that's on the order of ~4,000 independent tests, so at a
nominal two-sided alpha of ~0.045 (|z| >= 2 under a standard normal null)
roughly 4,000 * 0.045 ~= 180 days are expected to cross that bar by chance
alone, with no real fire-activity signal behind them.

This module does two things about that, matching the two options raised
in the review:

1. ``benjamini_hochberg`` — the actual Benjamini-Hochberg FDR procedure,
   which turns the fixed z >= 2 cutoff into a cutoff that adapts to how
   many tests are run, controlling the *expected proportion* of flagged
   days that are false positives (the FDR) at a chosen level ``q``
   (default 0.05, i.e. "at most 5% of flagged days are expected to be
   false alarms").

2. ``expected_false_positive_count`` — the cheaper, purely descriptive
   fallback: given the naive z threshold and the number of tests, report
   how many "elevated" days are expected by chance alone. This does not
   change which days are flagged; it just makes the naive threshold's
   output honestly labeled.

Both are independent of the null model used upstream (seasonal-naive
MAD/median today, harmonic regression + changepoint detection later,
block-bootstrap residuals later still) — this module only consumes the
z-scores (or p-values) those methods already produce, so it does not need
to change when item 3 ships. It also accepts an optional ``group_col`` so
that once fire-type stratification (item 2) lands, FDR can be (and
should be) applied separately within each stratum (e.g. vegetation fires
only) rather than pooled across static sources and vegetation fires,
without any change to this module.

No dependency beyond numpy; pandas is optional and only used by the
``apply_fdr_to_frame`` convenience wrapper.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "z_to_pvalue",
    "benjamini_hochberg",
    "BHResult",
    "expected_false_positive_count",
    "apply_fdr_to_frame",
]

# Mirrors the PYRO_ANOM_WINDOW-style env-var knobs already used elsewhere
# in the pipeline. PYRO_FDR_Q is the target false discovery rate; rename
# here (and in the one call site) if the project's convention differs.
DEFAULT_FDR_Q = float(os.environ.get("PYRO_FDR_Q", "0.05"))


def z_to_pvalue(z: Sequence[float], tails: str = "two-sided") -> np.ndarray:
    """Convert z-scores to p-values under a standard normal null.

    This assumes the upstream z-score (z = (x - median) / (1.4826 * MAD),
    or whatever the climatology null currently produces) is already
    standardized, which is the same assumption the z >= 2 cutoff already
    makes implicitly.

    Parameters
    ----------
    z : array-like of float. NaN entries (e.g. cloud-obscured /
        not-observed days) pass through as NaN p-values.
    tails : "two-sided" (default) or "greater" (one-sided, for a
        detector that only cares about unusually *high* activity, which
        is the actual use case here — nobody flags an anomalously quiet
        day as a fire alarm).
    """
    z = np.asarray(z, dtype=float)
    if tails == "two-sided":
        p = np.array(
            [math.erfc(abs(v) / math.sqrt(2)) if not math.isnan(v) else np.nan for v in z]
        )
    elif tails == "greater":
        p = np.array(
            [0.5 * math.erfc(v / math.sqrt(2)) if not math.isnan(v) else np.nan for v in z]
        )
    else:
        raise ValueError("tails must be 'two-sided' or 'greater'")
    return p


@dataclass
class BHResult:
    reject: np.ndarray        # bool mask, same order as input, True = significant after FDR control
    q_values: np.ndarray      # BH-adjusted p-values ("q-values"), same order as input
    n_tested: int             # number of non-NaN tests actually considered
    n_significant: int
    critical_p: float         # the largest raw p-value that still passed BH
    q: float                  # the target FDR level that was used


def benjamini_hochberg(pvalues: Sequence[float], q: Optional[float] = None) -> BHResult:
    """Standard Benjamini-Hochberg step-up FDR procedure.

    NaN p-values (missing/not-yet-observed days) are excluded from the
    test and come back with reject=False, q_value=NaN — they were never
    a candidate "elevated" day in the first place.

    Procedure: sort p-values ascending as p_(1) <= ... <= p_(n); find the
    largest k such that p_(k) <= (k/n) * q; reject all hypotheses with
    rank <= k. This controls E[false positives / flagged days] <= q
    under independence or positive dependence among the tests, which is
    the relevant regime here (nearby days in a fire season are
    positively correlated, not adversarially anti-correlated).
    """
    if q is None:
        q = DEFAULT_FDR_Q

    p = np.asarray(pvalues, dtype=float)
    n_total = p.shape[0]
    valid = ~np.isnan(p)
    n = int(valid.sum())

    reject = np.zeros(n_total, dtype=bool)
    qvals = np.full(n_total, np.nan)

    if n == 0:
        return BHResult(reject, qvals, 0, 0, float("nan"), q)

    idx_valid = np.where(valid)[0]
    pv = p[idx_valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]

    thresh = (np.arange(1, n + 1) / n) * q
    below = ranked <= thresh
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    critical_p = float(ranked[k - 1]) if k > 0 else float("nan")

    # Monotone BH-adjusted p-values ("q-values"): q_(i) = min_{j>=i} (p_(j) * n / j)
    adj_sorted = np.empty(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        candidate = ranked[i] * n / (i + 1)
        running_min = min(running_min, candidate)
        adj_sorted[i] = min(running_min, 1.0)

    adj = np.empty(n)
    adj[order] = adj_sorted
    qvals[idx_valid] = adj

    if k > 0:
        sig_positions = idx_valid[order[:k]]
        reject[sig_positions] = True

    return BHResult(
        reject=reject,
        q_values=qvals,
        n_tested=n,
        n_significant=int(reject.sum()),
        critical_p=critical_p,
        q=q,
    )


def expected_false_positive_count(
    n_tests: int, z_threshold: float = 2.0, tails: str = "two-sided"
) -> float:
    """Expected number of 'elevated' days under the null, at the naive
    fixed-threshold rule, with no FDR correction applied.

    This is the minimum fix mentioned in the review for anyone who keeps
    the fixed z >= threshold rule instead of switching to BH: report this
    number next to the "N days elevated" figure so "anomalous" is read
    against the right baseline.
    """
    if tails == "two-sided":
        alpha = math.erfc(z_threshold / math.sqrt(2))
    elif tails == "greater":
        alpha = 0.5 * math.erfc(z_threshold / math.sqrt(2))
    else:
        raise ValueError("tails must be 'two-sided' or 'greater'")
    return n_tests * alpha


def apply_fdr_to_frame(
    df,
    z_col: str = "z_score",
    q: Optional[float] = None,
    naive_z_threshold: float = 2.0,
    tails: str = "greater",
    group_col: Optional[str] = None,
):
    """Convenience wrapper for a pandas DataFrame of daily anomaly rows.

    Adds four columns (returns a copy; does not mutate ``df``):

    - ``p_value``: p-value for that day's z-score under the standard
      normal null.
    - ``q_value``: BH-adjusted p-value.
    - ``fdr_significant``: bool, the FDR-controlled replacement for the
      current "elevated" flag.
    - ``expected_false_positives_naive``: scalar (repeated per row, or
      per group if ``group_col`` is given) — how many of the *naive*
      z >= naive_z_threshold days in that run/group are expected by
      chance. Kept alongside fdr_significant so a dashboard can show
      both "here's the honest count under the old rule" and "here's the
      corrected flag" during the transition.

    ``group_col``: once fire-type stratification (item 2) ships, pass the
    fire-type column here so FDR is controlled separately within each
    stratum (e.g. vegetation fires only) instead of pooled with static
    sources like flares, which would otherwise dilute the correction.
    Left as None today applies one correction across the whole frame, matching
    the current unstratified pipeline.
    """
    import pandas as pd  # local import: this wrapper is the only part that needs pandas

    if q is None:
        q = DEFAULT_FDR_Q

    out = df.copy()
    out["p_value"] = z_to_pvalue(out[z_col].to_numpy(), tails=tails)

    if group_col is None:
        groups = [(None, out.index)]
    else:
        groups = list(out.groupby(group_col).groups.items())

    out["q_value"] = np.nan
    out["fdr_significant"] = False
    out["expected_false_positives_naive"] = np.nan

    for _, row_index in groups:
        sub = out.loc[row_index]
        res = benjamini_hochberg(sub["p_value"].to_numpy(), q=q)
        out.loc[row_index, "q_value"] = res.q_values
        out.loc[row_index, "fdr_significant"] = res.reject
        out.loc[row_index, "expected_false_positives_naive"] = expected_false_positive_count(
            n_tests=res.n_tested, z_threshold=naive_z_threshold, tails=tails
        )

    return out


if __name__ == "__main__":
    # Self-test / usage demo — run with `python -m pyroscope.stats.fdr`.
    # No pytest dependency; this is meant to double as a readable example.
    rng = np.random.default_rng(0)
    n_null, n_signal = 3950, 50
    z_null = rng.normal(0, 1, n_null)
    z_signal = rng.normal(4.5, 1, n_signal)
    z_all = np.concatenate([z_null, z_signal])

    naive_flagged = int((z_all >= 2.0).sum())
    expected_fp = expected_false_positive_count(len(z_all), 2.0, tails="greater")
    print(f"naive z>=2 flagged: {naive_flagged} days "
          f"(expected false positives among these by chance: {expected_fp:.0f})")

    p = z_to_pvalue(z_all, tails="greater")
    res = benjamini_hochberg(p, q=0.05)
    true_positive_hits = int(res.reject[n_null:].sum())
    false_positive_hits = int(res.reject[:n_null].sum())
    print(f"BH-FDR (q=0.05) flagged: {res.n_significant} days "
          f"({true_positive_hits}/{n_signal} real signals caught, "
          f"{false_positive_hits} false positives among {n_null} null days)")

    assert res.n_significant < naive_flagged
    assert false_positive_hits < expected_fp
    print("self-test OK")
