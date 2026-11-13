"""HotSync-style active-cell anomaly scoring for PyroScope.

This is a companion to ``src.anomaly``'s robust FRP z-score. It ports
HotSync's activity-count signal and its mean/standard-deviation Z-score
without replacing PyroScope's existing FRP detector or validation.

The activity series is anchored to valid VIIRS days. When both sensors
were observed, it uses the union of MODIS and VIIRS active cells. On
VIIRS-observed days without MODIS coverage, it can use a VIIRS-to-union
scale learned from overlap days. MODIS-only days are never used to fill a
missing VIIRS day.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    HOTSYNC_ANOM_MIN_ACTIVITY,
    HOTSYNC_ANOM_MIN_OVERLAP_DAYS,
    HOTSYNC_ANOM_MIN_POOL,
    HOTSYNC_ANOM_WINDOW_DAYS,
    HOTSYNC_ANOM_Z,
)


_SCORE_COLUMNS = (
    "hotsync_activity_count",
    "hotsync_activity_source",
    "hotsync_activity_baseline",
    "hotsync_activity_baseline_sd",
    "hotsync_activity_pool_n",
    "hotsync_activity_ratio",
    "hotsync_activity_z",
    "hotsync_activity_flag",
)
_EVENT_COLUMNS = (
    "date",
    "activity_count",
    "typical_activity",
    "ratio_to_typical",
    "z_score",
    "activity_source",
)


def _float_values(series: pd.Series) -> np.ndarray:
    """Convert nullable pandas numeric columns to float arrays with NaNs."""
    return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()


def _activity_series(cal: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """Build HotSync's sensor-union activity measure, anchored to valid VIIRS days."""
    if not isinstance(cal.index, pd.DatetimeIndex):
        raise ValueError("activity scoring requires the DatetimeIndex produced by anomaly.load_series()")

    index = cal.index
    n = len(cal)
    values = np.full(n, np.nan, dtype=float)
    source = np.full(n, "not scored", dtype=object)

    if "valid" in cal.columns:
        valid = cal["valid"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    elif "status" in cal.columns:
        valid = cal["status"].isin(("measured", "measured_zero")).to_numpy(dtype=bool)
    elif "harmonized_status" in cal.columns:
        valid = cal["harmonized_status"].isin(("measured", "measured_zero")).to_numpy(dtype=bool)
    else:
        valid = np.ones(n, dtype=bool)

    if "viirs_active_cells" not in cal.columns:
        info = {
            "available": False,
            "reason": "calendar_daily.csv has no viirs_active_cells column; re-run preprocess and harmonize",
            "calibrated_to_sensor_union": False,
            "overlap_days": 0,
            "viirs_to_union_scale": None,
        }
        return (pd.Series(values, index=index), pd.Series(source, index=index), info)

    viirs = _float_values(cal["viirs_active_cells"])
    valid &= np.isfinite(viirs)

    calibrated = False
    overlap_days = 0
    scale_v = None
    calibration_reason = "MODIS overlap columns are unavailable; using VIIRS active cells only"

    can_union = all(c in cal.columns for c in ("modis_active_cells", "cells_active_in_both"))
    if can_union:
        modis = _float_values(cal["modis_active_cells"])
        both = _float_values(cal["cells_active_in_both"])
        if "modis_observed" in cal.columns:
            modis_observed = cal["modis_observed"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        else:
            # Older calendars may not have the explicit flag. A finite MODIS
            # count and overlap count are the conservative observation proxy.
            modis_observed = np.isfinite(modis) & np.isfinite(both)

        joint = valid & modis_observed & np.isfinite(modis) & np.isfinite(both)
        overlap_days = int(joint.sum())
        union = np.maximum(modis + viirs - both, 0.0)
        total_viirs = float(np.sum(viirs[joint])) if overlap_days else 0.0
        total_union = float(np.sum(union[joint])) if overlap_days else 0.0

        if (overlap_days >= HOTSYNC_ANOM_MIN_OVERLAP_DAYS
                and total_viirs > 0.0 and total_union > 0.0):
            scale_v = total_union / total_viirs
            values[joint] = union[joint]
            source[joint] = "MODIS+VIIRS union"
            viirs_only = valid & ~joint
            values[viirs_only] = viirs[viirs_only] * scale_v
            source[viirs_only] = "VIIRS scaled to union"
            calibrated = True
            calibration_reason = "VIIRS active-cell totals calibrated to the MODIS+VIIRS union on overlap days"

    if not calibrated:
        use_viirs = valid & np.isfinite(viirs)
        values[use_viirs] = viirs[use_viirs]
        source[use_viirs] = "VIIRS only (uncalibrated)"

    info = {
        "available": bool(np.isfinite(values).any()),
        "reason": calibration_reason,
        "calibrated_to_sensor_union": bool(calibrated),
        "overlap_days": int(overlap_days),
        "viirs_to_union_scale": None if scale_v is None else float(scale_v),
        "grid_note": "Uses PyroScope calendar_cells grid (PYRO_CELL_DEG; default 0.5 degrees), not HotSync's ~2 km grid",
    }
    return pd.Series(values, index=index), pd.Series(source, index=index), info


def score_activity(cal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculate HotSync-style same-season activity Z-scores for each valid day.

    The baseline is the mean and population standard deviation of activity
    in the same +/- day-of-year window across all OTHER years (leave-one-year-out).
    The denominator follows HotSync: max(std, 1 + sqrt(mean)).
    """
    activity, source, info = _activity_series(cal)
    index = cal.index
    result = pd.DataFrame(index=index)

    values = activity.to_numpy(dtype=float)
    valid = np.isfinite(values)
    if "doy" in cal.columns:
        doy = pd.to_numeric(cal["doy"], errors="coerce").to_numpy(dtype=float)
    else:
        dates = pd.DatetimeIndex(index)
        doy = dates.dayofyear.to_numpy(dtype=float)
        after_feb = dates.is_leap_year & (dates.month > 2)
        doy -= after_feb.astype(float)
        doy = np.minimum(doy, 365.0)
    if "year" in cal.columns:
        years = pd.to_numeric(cal["year"], errors="coerce").to_numpy(dtype=float)
    else:
        years = pd.DatetimeIndex(index).year.to_numpy(dtype=float)

    baseline = np.full(len(index), np.nan, dtype=float)
    baseline_sd = np.full(len(index), np.nan, dtype=float)
    pool_n = np.zeros(len(index), dtype=int)
    usable_doy = np.isfinite(doy) & np.isfinite(years)
    window = int(HOTSYNC_ANOM_WINDOW_DAYS)

    for day in range(1, 366):
        target_rows = np.flatnonzero(doy == day)
        if not len(target_rows):
            continue
        distance = np.abs(doy - day)
        distance = np.minimum(distance, 365.0 - distance)
        in_window = usable_doy & valid & (distance <= window)
        for year in np.unique(years[target_rows]):
            rows = target_rows[years[target_rows] == year]
            pool = values[in_window & (years != year)]
            pool_n[rows] = len(pool)
            if len(pool) < HOTSYNC_ANOM_MIN_POOL:
                continue
            mean = float(np.mean(pool))
            sd = float(np.std(pool, ddof=0))
            baseline[rows] = mean
            baseline_sd[rows] = sd

    denominator = np.maximum(baseline_sd, 1.0 + np.sqrt(np.maximum(baseline, 0.0)))
    z = np.full(len(index), np.nan, dtype=float)
    scoreable = (valid & (values >= HOTSYNC_ANOM_MIN_ACTIVITY)
                 & np.isfinite(baseline) & (pool_n >= HOTSYNC_ANOM_MIN_POOL))
    z[scoreable] = (values[scoreable] - baseline[scoreable]) / denominator[scoreable]

    ratio = np.full(len(index), np.nan, dtype=float)
    comparable = valid & np.isfinite(baseline)
    ratio[comparable] = (values[comparable] + 1.0) / (baseline[comparable] + 1.0)

    flag = np.full(len(index), "", dtype=object)
    flag[np.isfinite(z) & (z >= HOTSYNC_ANOM_Z)] = "high"

    result["hotsync_activity_count"] = values
    result["hotsync_activity_source"] = source.to_numpy(dtype=object)
    result["hotsync_activity_baseline"] = baseline
    result["hotsync_activity_baseline_sd"] = baseline_sd
    result["hotsync_activity_pool_n"] = pool_n
    result["hotsync_activity_ratio"] = ratio
    result["hotsync_activity_z"] = z
    result["hotsync_activity_flag"] = flag

    info.update({
        "z_threshold": float(HOTSYNC_ANOM_Z),
        "minimum_activity_to_score": float(HOTSYNC_ANOM_MIN_ACTIVITY),
        "baseline_window_days_each_side": int(HOTSYNC_ANOM_WINDOW_DAYS),
        "minimum_baseline_values": int(HOTSYNC_ANOM_MIN_POOL),
        "minimum_overlap_days_for_union_calibration": int(HOTSYNC_ANOM_MIN_OVERLAP_DAYS),
        "z_formula": "(observed - seasonal_mean) / max(seasonal_population_sd, 1 + sqrt(seasonal_mean))",
        "baseline": "same day-of-year +/- window in all other years (leave-one-year-out)",
    })
    return result, info


def top_anomalies(scored: pd.DataFrame, max_items: int = 8, separation_days: int = 7) -> pd.DataFrame:
    """Return the strongest non-neighbouring high-activity days, like HotSync's list."""
    if "hotsync_activity_flag" not in scored.columns:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    candidates = scored.loc[scored["hotsync_activity_flag"] == "high"].sort_values(
        "hotsync_activity_z", ascending=False
    )
    rows = []
    selected_dates = []
    for date, row in candidates.iterrows():
        date = pd.Timestamp(date)
        if any(date.year == previous.year and abs((date - previous).days) < separation_days
               for previous in selected_dates):
            continue
        rows.append({
            "date": date.date().isoformat(),
            "activity_count": float(row["hotsync_activity_count"]),
            "typical_activity": float(row["hotsync_activity_baseline"]),
            "ratio_to_typical": float(row["hotsync_activity_ratio"]),
            "z_score": float(row["hotsync_activity_z"]),
            "activity_source": str(row["hotsync_activity_source"]),
        })
        selected_dates.append(date)
        if len(rows) >= max_items:
            break
    return pd.DataFrame(rows, columns=_EVENT_COLUMNS)


def plot_activity(scored: pd.DataFrame, output_path: Path) -> bool:
    """Write a separate activity-count plot; return False when the metric is unavailable."""
    if ("hotsync_activity_count" not in scored.columns
            or not np.isfinite(pd.to_numeric(scored["hotsync_activity_count"], errors="coerce")).any()):
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    values = scored["hotsync_activity_count"].astype(float)
    baseline = scored["hotsync_activity_baseline"].astype(float)
    ax.plot(scored.index, values, lw=0.8, label="Observed active-cell activity")
    ax.plot(scored.index, baseline, lw=1.0, label="Same-season baseline (other years)")
    high = scored[scored["hotsync_activity_flag"] == "high"]
    if not high.empty:
        ax.scatter(high.index, high["hotsync_activity_count"], s=18, label="High activity anomaly")
    ax.set_ylabel("Active cells per day (calendar grid)")
    ax.set_title("HotSync-style activity anomalies (mean / standard-deviation Z-score)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
