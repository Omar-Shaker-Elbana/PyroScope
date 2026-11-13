"""
Fire-type stratification for the Burning Activity Calendar and anomaly
detector.

Context
-------
MODIS/VIIRS active-fire records carry a ``type`` field (already present
in the harmonized data, since it's what the k_sensitivity diagnostic
uses) with the standard FIRMS encoding:

    0 = presumed vegetation fire
    1 = active volcano
    2 = other static land source (gas flares, industrial sites, etc.)
    3 = offshore

Today the calendar and anomaly detector sum every type together into one
daily number. Type 2 sources burn steadily year-round and carry no
wildfire-activity signal — mixing them in drags the baseline up and can
both hide a real vegetation-fire anomaly (buried under a steady flare
count) and, less often, manufacture a fake one (a flare coming online or
going dark looks like a step change in the pooled series). The fix below
is the one the review asks for: split the calendar by type, run anomaly
detection on vegetation fires only, and report static sources separately
rather than discarding them.

This module only splits and aggregates; it does not compute z-scores or
climatology itself; the existing (or the harmonic-regression replacement
once item 3 ships) anomaly code should run on the output of
``vegetation_fires_only`` instead of on the raw unstratified hotspot
table. The FDR module added earlier (``pyroscope.stats.fdr``) composes
with this directly: once you have a per-day z-score for the
vegetation-only series, feed it straight into
``apply_fdr_to_frame(veg_df, z_col=...)`` as before — nothing about that
call needs to change, it's just now being fed a cleaner input.

No dependency beyond pandas.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

__all__ = [
    "FIRE_TYPE_LABELS",
    "DEFAULT_ANOMALY_STRATA",
    "classify_fire_type",
    "split_by_fire_type",
    "vegetation_fires_only",
    "build_stratified_calendar",
    "summarize_static_sources",
]

# Standard FIRMS `type` field encoding (MODIS Collection 6.1 / VIIRS).
FIRE_TYPE_LABELS = {
    0: "vegetation",
    1: "volcano",
    2: "static_other",  # flares, industrial — the diagnostic k_sensitivity already uses
    3: "offshore",
}

# Strata that feed the wildfire-activity anomaly signal. Volcanoes and
# offshore detections are left out of both the anomaly feed and the
# "static_other" report by default since neither is what k_sensitivity
# was diagnosing nor what the review flagged — surface them via
# `extra_strata` below if your AOI needs them tracked too.
DEFAULT_ANOMALY_STRATA = ("vegetation",)

# Strata reported separately rather than discarded outright.
DEFAULT_REPORT_STRATA = ("vegetation", "static_other")


def classify_fire_type(type_codes: pd.Series) -> pd.Series:
    """Map raw FIRMS `type` codes to the string labels in FIRE_TYPE_LABELS.

    Unrecognized or missing codes map to "unknown" rather than raising,
    so a malformed row doesn't take down the whole run — but check the
    "unknown" bucket's size before trusting a calendar built from data
    with a lot of them; a large unknown count usually means the type
    column isn't in the FIRMS 0-3 encoding you expect.
    """
    return type_codes.map(FIRE_TYPE_LABELS).fillna("unknown")


def split_by_fire_type(
    df: pd.DataFrame, type_col: str = "type"
) -> Dict[str, pd.DataFrame]:
    """Split a hotspot table into one DataFrame per stratum label.

    Returns a dict keyed by the labels in FIRE_TYPE_LABELS plus
    "unknown" for anything that didn't map. Each value is a view-safe
    copy; downstream mutation of one stratum never touches another.
    """
    labels = classify_fire_type(df[type_col])
    return {label: df.loc[labels == label].copy() for label in labels.unique()}


def vegetation_fires_only(df: pd.DataFrame, type_col: str = "type") -> pd.DataFrame:
    """The input the anomaly detector should actually run on.

    Equivalent to ``split_by_fire_type(df, type_col)["vegetation"]`` but
    cheaper (no need to build every other stratum just to discard them),
    and the one-line call site this is meant to replace
    ``df`` with at the top of the anomaly step.
    """
    labels = classify_fire_type(df[type_col])
    return df.loc[labels == "vegetation"].copy()


def build_stratified_calendar(
    df: pd.DataFrame,
    date_col: str = "acq_date",
    type_col: str = "type",
    frp_col: Optional[str] = "frp",
    strata: Sequence[str] = DEFAULT_REPORT_STRATA,
) -> pd.DataFrame:
    """Daily calendar with one count (and, if frp_col is given, one FRP-sum)
    column per requested stratum, plus a `vegetation_share` column.

    This is the artifact the review means by "split the calendar by
    type": a single date-indexed table where vegetation and static
    sources are columns, not a pooled sum, so a dashboard or the
    anomaly step can read off exactly the series it needs instead of
    re-deriving it from the raw hotspot rows each time.

    `vegetation_share` = vegetation count / (vegetation + static_other
    count) for that day, which is a cheap sanity check: a day where it
    suddenly drops usually means a static source changed behavior, not
    that vegetation fire activity fell.
    """
    labels = classify_fire_type(df[type_col])
    dates = pd.to_datetime(df[date_col])

    out = pd.DataFrame(index=pd.Index(sorted(dates.dt.normalize().unique()), name=date_col))

    for stratum in strata:
        mask = labels == stratum
        day = dates[mask].dt.normalize()
        counts = day.value_counts().reindex(out.index, fill_value=0)
        out[f"{stratum}_count"] = counts.astype(int)

        if frp_col is not None and frp_col in df.columns:
            frp_sum = df.loc[mask, frp_col].groupby(day).sum().reindex(out.index, fill_value=0.0)
            out[f"{stratum}_frp_sum"] = frp_sum

    if "vegetation_count" in out.columns and "static_other_count" in out.columns:
        denom = out["vegetation_count"] + out["static_other_count"]
        out["vegetation_share"] = (out["vegetation_count"] / denom).where(denom > 0)

    return out.reset_index()


def summarize_static_sources(
    df: pd.DataFrame,
    date_col: str = "acq_date",
    type_col: str = "type",
    frp_col: Optional[str] = "frp",
) -> pd.DataFrame:
    """Separate daily report for type==2 (flares/industrial) detections —
    the "report static sources separately" half of the fix, so they're
    visible rather than silently dropped once they're excluded from the
    anomaly feed.
    """
    static_df = split_by_fire_type(df, type_col=type_col).get(
        "static_other", df.iloc[0:0]
    )
    if static_df.empty:
        cols = [date_col, "static_other_count"] + (
            ["static_other_frp_sum"] if frp_col else []
        )
        return pd.DataFrame(columns=cols)

    day = pd.to_datetime(static_df[date_col]).dt.normalize()
    report = day.value_counts().sort_index().rename("static_other_count").reset_index()
    report = report.rename(columns={"index": date_col, "day": date_col})

    if frp_col is not None and frp_col in static_df.columns:
        frp = static_df[frp_col].groupby(day).sum().rename("static_other_frp_sum")
        report = report.merge(frp.reset_index().rename(columns={"index": date_col, "day": date_col}),
                               on=date_col, how="left")

    return report.sort_values(date_col).reset_index(drop=True)


if __name__ == "__main__":
    # Self-test / usage demo — run with `python -m pyroscope.pipeline.fire_type`.
    import numpy as np

    rng = np.random.default_rng(0)
    n_days = 60
    dates = pd.date_range("2024-06-01", periods=n_days, freq="D")

    rows = []
    for d in dates:
        # steady flare, every day, roughly constant FRP
        rows.append({"acq_date": d, "type": 2, "frp": 12 + rng.normal(0, 1)})
        # background vegetation activity, with a spike mid-period
        n_veg = rng.poisson(3 if d < dates[40] or d > dates[45] else 25)
        for _ in range(n_veg):
            rows.append({"acq_date": d, "type": 0, "frp": rng.exponential(5)})
    raw = pd.DataFrame(rows)

    veg = vegetation_fires_only(raw)
    assert (veg["type"] == 0).all()

    cal = build_stratified_calendar(raw)
    assert {"vegetation_count", "static_other_count", "vegetation_share"} <= set(cal.columns)
    # the flare's own count should barely move day to day...
    assert cal["static_other_count"].std() == 0  # exactly one flare detection/day in this synthetic set
    # ...while vegetation clearly spikes mid-period
    spike_mean = cal.loc[40:45, "vegetation_count"].mean()
    background_mean = cal.loc[cal.index.difference(range(40, 46)), "vegetation_count"].mean()
    assert spike_mean > 3 * background_mean

    static_report = summarize_static_sources(raw)
    assert len(static_report) == n_days

    print(cal.head())
    print("vegetation-only rows:", len(veg), "of", len(raw), "total")
    print("static source report rows:", len(static_report))
    print("self-test OK")
