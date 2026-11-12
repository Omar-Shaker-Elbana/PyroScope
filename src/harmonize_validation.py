"""Diagnostics for the MODIS -> VIIRS scale k and for the spatial aggregation grid (called by harmonize.py).

 - k_sensitivity            : how k moves with the choices made to estimate it (time gap, day/night, source type,
                              confidence, fire size, trimming, estimator). A constant k is a simplification; this
                              table says how much it simplifies.
 - cluster_unit_sensitivity : width of the bootstrap CI of k for different resampling units (pixel, cells of several
                              sizes, site, day, month). Justifies the unit used for the headline CI.
 - grid_sensitivity         : what the 0.5 deg aggregation grid changes and what it does not (point estimate of k and
                              daily FRP totals: nothing; active-cell ratios, cell-level correlations, the CI: yes).
 - plots and NaN-safe JSON helpers.

Deliberately free of any import from harmonize.py (harmonize.py imports this module).
"""
import json

import numpy as np
import pandas as pd

from src.kstats import cell_area_km2, cell_id, edge_split_probability, ratio_ci

SMALL_CLUSTER_N = 30   # percentile bootstrap understates uncertainty with fewer clusters than this


# --------------------------------------------------------------------------- json
def sanitize(obj):
    """Recursively turn numpy / pandas scalars into plain Python and NaN / inf into None (valid JSON)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if obj is pd.NA:
        return None
    return obj


def dumps(obj) -> str:
    return json.dumps(sanitize(obj), indent=2)


# --------------------------------------------------------------------------- k sensitivity
def _k(g):
    return float(g["viirs_frp_sum"].sum() / g["frp"].sum())


def k_sensitivity(use_all, max_dt, min_pairs):
    """k for every sensible alternative way of estimating it, from the usable matched pixels of ONE MODIS satellite
    (`use_all`: any time gap within the matching window). The primary estimate is the ratio of sums over pixels within
    `max_dt` minutes. Variants that change the ESTIMAND (median / geometric mean of pixel ratios describe a typical
    pixel, not the total) are labelled as such. A variant with fewer than `min_pairs` pixels is reported without k."""
    base = use_all[use_all["abs_dt_min"] <= max_dt]
    k0 = _k(base)
    out = {}

    def add(name, g, estimand="total FRP (ratio of sums)"):
        if len(g) < min_pairs:
            out[name] = {"n_pairs": int(len(g)), "k": None, "rel_to_primary": None, "estimand": estimand,
                         "note": f"fewer than {min_pairs} pixels"}
            return
        k = _k(g)
        out[name] = {"n_pairs": int(len(g)), "k": k, "rel_to_primary": k / k0 - 1.0, "estimand": estimand}

    add(f"primary (|dt| <= {max_dt:g} min)", base)
    for dt in (10, 15, 60, 120):
        add(f"|dt| <= {dt} min", use_all[use_all["abs_dt_min"] <= dt])
    if "daynight" in base.columns:
        add("daytime pixels only", base[base["daynight"].astype(str) == "D"])
        add("night-time pixels only", base[base["daynight"].astype(str) == "N"])
    if "type" in base.columns:
        t = pd.to_numeric(base["type"], errors="coerce")
        add("presumed vegetation fires only (FIRMS type 0)", base[t == 0])
        add("static land sources only (type 2: flares, industry)", base[t == 2])
    if "confidence_class" in base.columns:
        add("MODIS high-confidence pixels only", base[base["confidence_class"] == "high"])
    if "version" in base.columns:
        for ver, g in base.groupby(base["version"].astype(str)):
            add(f"MODIS product version {ver}", g)
    med = base["frp"].median()
    add("MODIS FRP below the median", base[base["frp"] <= med])
    add("MODIS FRP above the median", base[base["frp"] > med])
    add("drop the largest 1 % of MODIS pixels", base[base["frp"] <= base["frp"].quantile(0.99)])
    add("drop the largest 5 % of MODIS pixels", base[base["frp"] <= base["frp"].quantile(0.95)])
    ratio = (base["viirs_frp_sum"] / base["frp"]).to_numpy()
    out["median of pixel ratios (typical pixel)"] = {
        "n_pairs": int(len(base)), "k": float(np.median(ratio)), "rel_to_primary": float(np.median(ratio) / k0 - 1),
        "estimand": "typical pixel (NOT total FRP): different quantity, shown for reference"}
    out["geometric mean of pixel ratios"] = {
        "n_pairs": int(len(base)), "k": float(np.exp(np.mean(np.log(ratio)))),
        "rel_to_primary": float(np.exp(np.mean(np.log(ratio))) / k0 - 1),
        "estimand": "typical pixel (NOT total FRP): different quantity, shown for reference"}
    same = [d["rel_to_primary"] for d in out.values()
            if d["rel_to_primary"] is not None and d["estimand"].startswith("total")]
    return {"primary_k": k0, "max_abs_rel_deviation_same_estimand": float(max(abs(x) for x in same)) if same else None,
            "variants": out,
            "note": "k is an aggregate ratio conditional on a MODIS detection; it varies with fire size and "
                    "time gap, so a single constant is an approximation whose size is read off this table."}


# --------------------------------------------------------------------------- cluster units
def cluster_unit_sensitivity(use, n_boot):
    """Bootstrap CI of k when the resampling unit is a pixel, a cell of several sizes, a 0.01 deg site, a day or a month.
    Too small a unit (pixel) ignores that neighbouring pixels of one fire move together; too large a unit leaves too
    few clusters (< SMALL_CLUSTER_N) and the percentile bootstrap becomes unreliable (it under-covers)."""
    rng = np.random.default_rng(1)
    k0 = _k(use)
    lat, lon = use["latitude"].to_numpy(), use["longitude"].to_numpy()
    keys = {"pixel (no clustering)": np.arange(len(use))}
    for d in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        keys[f"{d:g} deg cell"] = cell_id(lat, lon, d)
    keys["site (0.01 deg)"] = pd.factorize(pd.Series(np.round(lat, 2).astype(str)) + "_" + pd.Series(np.round(lon, 2).astype(str)))[0]
    keys["day"] = use["acq_datetime_utc"].dt.strftime("%Y%m%d").astype(int).to_numpy()
    keys["calendar month"] = use["acq_datetime_utc"].dt.strftime("%Y%m").astype(int).to_numpy()
    rows = []
    for name, key in keys.items():
        arr = pd.DataFrame({"k": key, "f": use["frp"].to_numpy(), "v": use["viirs_frp_sum"].to_numpy()}).groupby("k")[["f", "v"]].sum().to_numpy()
        _, lo, hi, se = ratio_ci(arr, rng, n_boot)
        rows.append({"unit": name, "n_clusters": int(len(arr)), "k_ci95": [lo, hi], "rel_width": (hi - lo) / k0,
                     "reliable": bool(len(arr) >= SMALL_CLUSTER_N)})
    return rows


# --------------------------------------------------------------------------- grid sensitivity
def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(np.log(a)) == 0 or np.std(np.log(b)) == 0:
        return float("nan")
    return float(np.corrcoef(np.log(a), np.log(b))[0, 1])


def grid_sensitivity(m_like, v_ref, valid_days, use, grids, primary_deg, n_boot, sat_label):
    """Re-do every cell-level quantity of the pipeline on grids of several sizes.

    m_like / v_ref : MODIS (the satellite whose k is used: like-for-like overpass with the VIIRS reference) and
                     VIIRS-reference detections; valid_days: complete, observed UTC days; use: matched pixels in the
                     k fit. NOT affected by the grid (by construction, checked here): k itself and every daily FRP
                     total in the calendar. Affected: active-cell counts and overlap (modifiable areal unit problem),
                     cell-level correlations, the clustered CI of k, and the 'same fire in two cells' error."""
    days = set(pd.DatetimeIndex(valid_days).tolist())
    m = m_like[m_like["date"].isin(days)]
    v = v_ref[v_ref["date"].isin(days)]
    rng = np.random.default_rng(2)
    k0 = _k(use)
    lat0 = float(np.nanmedian(np.r_[m["latitude"].to_numpy(), v["latitude"].to_numpy()])) if len(m) + len(v) else 0.0
    date_u = use["acq_datetime_utc"].dt.strftime("%Y%m%d").astype(int)
    rows = []
    for g in sorted(set(grids)):
        mc = pd.DataFrame({"date": m["date"].to_numpy(), "c": cell_id(m["latitude"], m["longitude"], g)}).drop_duplicates()
        vc = pd.DataFrame({"date": v["date"].to_numpy(), "c": cell_id(v["latitude"], v["longitude"], g)}).drop_duplicates()
        both = len(mc.merge(vc, on=["date", "c"]))
        uc = cell_id(use["latitude"], use["longitude"], g)
        frame = pd.DataFrame({"c": uc, "d": date_u.to_numpy(), "f": use["frp"].to_numpy(), "v": use["viirs_frp_sum"].to_numpy()})
        tot = frame.groupby("c")[["f", "v"]].sum()
        cd = frame.groupby(["c", "d"])[["f", "v"]].sum()
        k_g, lo, hi, _ = ratio_ci(tot.to_numpy(), rng, n_boot)
        rows.append({
            "cell_deg": g, "cell_area_km2": float(cell_area_km2(lat0, g)),
            "edge_split_probability_1km": edge_split_probability(1.0, g, lat0),
            "modis_active_cell_days": int(len(mc)), "viirs_active_cell_days": int(len(vc)),
            "viirs_over_modis_active_cells": len(vc) / max(len(mc), 1),
            "share_of_modis_active_cells_also_viirs": both / max(len(mc), 1),
            "viirs_detections_per_active_cell": len(v) / max(len(vc), 1),
            "k_point_estimate": k_g, "k_ci95_low": lo, "k_ci95_high": hi, "k_ci_rel_width": (hi - lo) / k_g,
            "n_cells_in_fit": int(len(tot)), "ci_reliable": bool(len(tot) >= SMALL_CLUSTER_N),
            "cell_total_corr_log": _corr(tot["f"], tot["v"]), "cell_day_corr_log": _corr(cd["f"], cd["v"]),
        })
    df = pd.DataFrame(rows)
    ref = df.loc[(df["cell_deg"] - primary_deg).abs().idxmin()]
    n_days = len(days)
    like = {
        "modis_satellite": sat_label,
        "basis": f"{sat_label} x VIIRS reference (same overpass time), days observed and complete",
        "complete_days_used": int(n_days),
        "viirs_over_modis_detections": float(len(v) / max(len(m), 1)),
        "viirs_over_modis_scaled_total_frp": float(v["frp"].sum() / max(m["frp_equiv"].sum(), 1e-9)),
        "viirs_over_modis_active_cells": float(ref["viirs_over_modis_active_cells"]),
        "share_of_modis_active_cells_also_viirs": float(ref["share_of_modis_active_cells_also_viirs"]),
        "cell_deg": float(primary_deg),
    }
    info = {
        "grids_deg": [float(x) for x in df["cell_deg"]],
        "k_point_estimate_identical_across_grids": bool(np.allclose(df["k_point_estimate"], k0, rtol=1e-9)),
        "calendar_daily_frp_totals_depend_on_grid": False,
        "at_calendar_grid": like,
        "table": "grid_sensitivity.csv",
        "median_latitude_of_detections": lat0,
        "area_distortion": ("a lat/lon cell shrinks with cos(latitude): per-cell totals are comparable only between cells "
                            "at similar latitudes; calendar_cells.csv carries cell_area_km2 for density normalisation. "
                            "For a global run use an equal-area grid."),
        "reading_guide": [
            "k and daily FRP totals: identical for every grid (ratio of sums over pixels).",
            "viirs_over_modis_active_cells and share_of_modis_active_cells_also_viirs: change with the grid (coarser cells "
            "merge several fires into one 'active cell', so the ratio falls towards 1 and the overlap towards 100 %): "
            "quote them together with cell_deg, never as grid-free properties of the sensors.",
            "k_ci_rel_width: grows from pixel to ~0.25-0.5 deg cells (spatial dependence is captured) and collapses for "
            "cells that leave fewer than 30 clusters (ci_reliable = False).",
            "cell_total_corr_log (per-cell totals over the whole record) is inflated by aggregation; cell_day_corr_log is "
            "the fairer cell-level agreement.",
            "edge_split_probability_1km: chance that two detections 1 km apart (one fire seen by both sensors) land in "
            "different cells; it is the matching error of the grid, small at 0.5 deg, large at 0.05 deg.",
        ],
    }
    return df, info


# --------------------------------------------------------------------------- plots
def plot_grid(df, primary_deg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    g = df["cell_deg"]
    ax = axes[0]
    ok = df["ci_reliable"]
    ax.plot(g, df["k_ci_rel_width"], "o-", color=BLUE)
    ax.scatter(g[~ok], df["k_ci_rel_width"][~ok], s=90, facecolors="none", edgecolors=ORANGE, label="< 30 clusters (unreliable)")
    ax.set_title("Width of the 95% CI of k\n(cluster bootstrap, cells of this size)")
    ax.set_ylabel("CI width / k")
    if (~ok).any():
        ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(g, df["viirs_over_modis_active_cells"], "o-", color=BLUE, label="VIIRS / MODIS active cells")
    ax2 = ax.twinx()
    ax2.plot(g, df["share_of_modis_active_cells_also_viirs"], "s--", color=ORANGE, label="share of MODIS cells also VIIRS")
    ax2.set_ylim(0, 1.02)
    ax.set_title("Cell metrics are properties of the grid")
    ax.set_ylabel("ratio", color=BLUE); ax2.set_ylabel("share", color=ORANGE)
    ax = axes[2]
    ax.plot(g, df["cell_day_corr_log"], "o-", color=BLUE, label="cell x day")
    ax.plot(g, df["cell_total_corr_log"], "s--", color=ORANGE, label="cell totals, whole record")
    ax.plot(g, df["edge_split_probability_1km"], "^:", color=GREY, label="P(same fire split by cell edge)")
    ax.set_title("Cell-level agreement and grid-edge error")
    ax.legend(fontsize=8)
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("cell size (degrees)"); a.grid(True, alpha=0.3)
        a.axvline(primary_deg, color=GREY, lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_k_by_year(info, sat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = info.get("k_by_year") or {}
    if not by:
        return
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
    yrs = list(by)
    k = np.array([by[y]["k"] for y in yrs])
    lo = np.array([by[y]["k_ci95"][0] for y in yrs])
    hi = np.array([by[y]["k_ci95"][1] for y in yrs])
    fig, ax = plt.subplots(figsize=(8, 5))
    pi = info["k_prediction_interval95_new_year"]
    ax.axhspan(pi[0], pi[1], color=ORANGE, alpha=0.12, label="95% prediction interval for an unseen year")
    ax.axhspan(*info["k_ci95"], color=BLUE, alpha=0.18, label="95% CI of pooled k")
    ax.axhline(info["k"], color=BLUE, lw=1.5)
    ax.errorbar(range(len(yrs)), k, yerr=[k - lo, hi - k], fmt="o", color="#33322f", capsize=3, label="per-year k (95% CI)")
    ax.set_xticks(range(len(yrs)), yrs)
    het = info.get("k_year_heterogeneity")
    sub = (f"Cochran Q p = {het['p_Q']:.2f}, I² = {het['I2']:.0%}, tau = {het['tau']:.0%} "
           f"(95% upper limit {het['tau_ci95'][1]:.0%})") if het else "fewer than 3 years"
    ax.set_title(f"MODIS {sat} x S-NPP: k by year\n{sub}")
    ax.set_ylabel("k (VIIRS-equivalent FRP / MODIS FRP)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
