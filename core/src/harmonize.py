"""Phase B: MODIS <-> VIIRS harmonization and a unified burning-activity calendar.

Steps
 1. Load the standardized CSVs and clamp both sensors to their COMMON time window.
 2. Footprint-aware matching: every VIIRS 375 m pixel whose centre falls inside a MODIS pixel footprint
    (scan x track) within +-MATCH_MAX_MIN is assigned to ONE MODIS pixel (nearest centre), so overlapping MODIS
    footprints at the swath edge never double-count VIIRS FRP. Each MODIS satellite is matched separately.
 3. Estimate a constant aggregate FRP scale  k = sum(VIIRS footprint FRP) / sum(MODIS FRP)  FOR EACH MODIS
    SATELLITE SEPARATELY, from matches within K_FIT_MAX_DT_MIN minutes only, with a cell-clustered bootstrap CI,
    per-year estimates, a between-year heterogeneity analysis (Cochran Q, I^2, tau with a Q-profile interval), a
    prediction interval for an unseen year and a chronological hold-out. A satellite that has no close-in-time
    matches (Terra vs S-NPP: ~150 min apart) gets NO k and NO VIIRS-equivalent FRP; its k is never borrowed from
    another satellite unless PYRO_APPLY_UNVALIDATED_K=1, and then every value is labelled.
    The constant k is the headline number, but the scale that is APPLIED can depend on fire size (and day/night, time
    gap) when cross-validation shows it should (kmodel.py; PYRO_K_MODEL=0 switches this off).
 4. Build the calendar on the VIIRS reference scale. A day with no detections is NOT a day without observations:
    every sensor-level number is NaN (not 0) on a day the sensor did not deliver data, and `harmonized_status` says
    exactly what a row is (measured / measured_zero / measured_low_coverage / observed_obscured / viirs_not_observed /
    partial_day). `observed_obscured` (needs PYRO_CLOUD_FILE, see cloud_mask.py) is a day the sensor delivered but whose
    clear-sky share is too low for "0 detections" to mean "no fire".
    MODIS is never substituted for a missing VIIRS day (it sees ~3-5x less FRP than VIIRS and predicts VIIRS daily
    totals poorly: see `daily_total_relationship` in frp_mapping.json) and sensors are never summed.
 5. Diagnostics (harmonize_validation.py): what k depends on, and how the 0.5 deg aggregation grid changes the
    cell-level numbers.

Run:  python -m core.src.harmonize
"""
import json
import re
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from core.src import cloud_mask
from core.src import harmonize_validation as hv
from core.src import kmodel
from core.src.config import ensure_dirs
from core.src.config import (
    PROCESSED_DIR, BBOX, START_DATE, END_DATE, CELL_DEG, MATCH_MAX_MIN, DAY_TOL_MIN, MIN_PAIRS_FIT,
    K_FIT_MAX_DT_MIN, K_FIT_AUTO_WIDEN, K_FIT_WIDEN_STEPS, SINGLE_DAY_DEMO, K_TRUST_MIN_PAIRS, REF_SAT, AOI_FILE, EXCLUDE_STATIC, OUT_OF_CORE, OUT_OF_CORE_ROWS,
    APPLY_UNVALIDATED_K, N_BOOT, K_TAU_MAX, GRID_SENS_DEG, COV_MIN_FRAC, COV_EXTREME_FRAC, COV_MIN_EXPECTED, COV_CLUSTER_GAP_MIN,
    K_MODEL, K_MODEL_BOOT, K_MODEL_FOLDS, K_MODEL_CLIP_Q,
    CLOUD_FILE, CLOUD_MIN_CLEAR, CLOUD_OBSCURE_NONZERO,
)
from core.src.kstats import (
    KM_PER_DEG, R_KM, cell_area_km2, heterogeneity, infer_missing_days, prediction_interval, ratio_ci,
)

VIIRS_I4_SATURATION_K = 367.0
MIN_CELLS_FIT = 10          # fewer clusters than this: the cluster bootstrap is meaningless
MIN_YEAR_CELLS = 10

WANTED = {
    "sensor_family", "source_satellite", "latitude", "longitude", "acq_datetime_utc",
    "frp", "scan", "track", "confidence_raw", "daynight", "brightness", "bright_ti4", "type", "version",
}


def slug(name) -> str:
    """'S-NPP' -> 'snpp', 'Aqua' -> 'aqua' (column-name prefix)."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


# --------------------------------------------------------------------------- loading
def load(path):
    """Load a standardized file for the legacy in-memory path (small datasets only)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Run `python -m core.src.preprocess` first, and check that "
                         "PYRO_BBOX / PYRO_START / PYRO_END leave rows for BOTH sensors.")
    df = pd.read_csv(path, usecols=lambda c: c in WANTED, parse_dates=["acq_datetime_utc"])
    df["source_satellite"] = df["source_satellite"].astype(str)
    # Downcast high-volume numeric columns without changing timestamps or categorical labels.
    for col in ("latitude", "longitude", "frp", "scan", "track", "brightness", "bright_ti4"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    return df.sort_values("acq_datetime_utc").reset_index(drop=True)


def coverage(label, df):
    """Observation window of the RAW file (before any AOI filter), written by preprocess.
    Falls back to the first/last detection, which is only reliable for global data."""
    p = PROCESSED_DIR / f"{label}_coverage.json"
    if p.exists():
        d = json.loads(p.read_text())
        return pd.Timestamp(d["start_utc"]), pd.Timestamp(d["end_utc"])
    return df["acq_datetime_utc"].min(), df["acq_datetime_utc"].max()


def observed_days(label, satellite=None):
    """(set of UTC days PRESENT in the raw file, basis) or (None, 'unavailable') for runs made before this feature.

    A day with no rows is MISSING (not 'no fire') when an empty day is implausible for that file at that time
    (kstats.infer_missing_days: negative-binomial model of a normal day from the +-30 surrounding days). When the
    coverage file holds per-satellite counts and `satellite` is given, the test is run on THAT satellite alone, so
    an outage of one satellite is not hidden by another satellite's rows."""
    p = PROCESSED_DIR / f"{label}_coverage.json"
    if not p.exists():
        return None, "unavailable"
    d = json.loads(p.read_text())
    if "raw_rows_per_day" not in d:
        return None, "unavailable"
    rows, basis = d["raw_rows_per_day"], "sensor-level raw rows per day"
    if satellite is not None and "raw_rows_per_day_by_satellite" in d:
        rows = d["raw_rows_per_day_by_satellite"].get(satellite, {})
        basis = f"{satellite} raw rows per day"
    full = pd.date_range(pd.Timestamp(d["start_utc"]).tz_localize(None).floor("D"),
                         pd.Timestamp(d["end_utc"]).tz_localize(None).floor("D"), freq="D")
    s = pd.Series(rows, dtype=float)
    s.index = pd.to_datetime(s.index)
    cnt = s.reindex(full, fill_value=0).to_numpy(float)
    missing = infer_missing_days(cnt)
    return set(full[~missing]), basis


def clamp_to_common_window(m, v):
    (ms, me), (vs, ve) = coverage("modis", m), coverage("viirs", v)
    t0, t1 = max(ms, vs), min(me, ve)
    # A day the user filtered out is MISSING, not "zero fire activity": honour PYRO_START / PYRO_END.
    if START_DATE:
        t0 = max(t0, pd.Timestamp(START_DATE, tz="UTC"))
    if END_DATE:
        t1 = min(t1, pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1))
    if t0 >= t1:
        raise SystemExit("MODIS and VIIRS files have no time overlap - nothing to harmonize.")
    m = m[m["acq_datetime_utc"].between(t0, t1)].reset_index(drop=True)
    v = v[v["acq_datetime_utc"].between(t0, t1)].reset_index(drop=True)
    return m, v, t0, t1


def pick_ref_sat(v):
    """Calendar reference = the VIIRS satellite observed on the most days (ties: most detections), so a
    multi-year series stays on ONE instrument instead of switching when a newer satellite launches."""
    if REF_SAT:
        if REF_SAT not in set(v["source_satellite"]):
            raise SystemExit(f"PYRO_REF_SAT={REF_SAT!r} is not in the VIIRS data: {sorted(v['source_satellite'].unique())}")
        return REF_SAT
    s = v.groupby("source_satellite").agg(days=("date", "nunique"), n=("frp", "size")).sort_values(["days", "n"])
    return s.index[-1]


def add_features(df, cell_deg=None):
    cell_deg = CELL_DEG if cell_deg is None else cell_deg
    df = df.copy()
    t = df["acq_datetime_utc"]
    df["date"] = t.dt.floor("D").dt.tz_localize(None)
    df["lst_h"] = ((t.dt.hour + t.dt.minute / 60.0) + df["longitude"] / 15.0) % 24
    df["area_km2"] = df["scan"] * df["track"]
    cell_i = np.floor((df["latitude"] + 90.0) / cell_deg).astype("int64")
    cell_j = np.floor((df["longitude"] + 180.0) / cell_deg).astype("int64")
    df["cell_id"] = cell_i * 1_000_000 + cell_j
    df["cell_lat"] = -90.0 + (cell_i + 0.5) * cell_deg
    df["cell_lon"] = -180.0 + (cell_j + 0.5) * cell_deg
    return df


def confidence_class(df):
    """Derived low/nominal/high class; the raw value stays in confidence_raw.
    MODIS numeric thresholds follow the FIRMS documentation (<30 low, 30-79 nominal, >=80 high)."""
    raw = df["confidence_raw"].astype(str).str.strip().str.lower()
    num = pd.to_numeric(raw, errors="coerce")
    out = pd.Series("unknown", index=df.index)
    out[num < 30] = "low"
    out[(num >= 30) & (num < 80)] = "nominal"
    out[num >= 80] = "high"
    out[raw.isin(["l", "low"])] = "low"
    out[raw.isin(["n", "nominal"])] = "nominal"
    out[raw.isin(["h", "high"])] = "high"
    return out


# --------------------------------------------------------------------------- matching
def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[R_KM * np.cos(la) * np.cos(lo), R_KM * np.cos(la) * np.sin(lo), R_KM * np.sin(la)]


def match_pair(m, v, max_min):
    """One MODIS satellite vs ONE VIIRS satellite.
    Adds n_viirs / viirs_frp_sum / viirs_frp_max / abs_dt_min to the MODIS pixels.
    Footprint = box |dx| <= scan/2 (east-west), |dy| <= track/2 (north-south): an approximation
    that ignores the small rotation of the swath at high latitude."""
    m = m.reset_index(drop=True).copy()
    for c, fill in [("n_viirs", 0), ("viirs_frp_sum", 0.0), ("viirs_frp_max", 0.0), ("abs_dt_min", np.nan)]:
        m[c] = fill
    v = v.sort_values("acq_datetime_utc").reset_index(drop=True)
    vt = v["acq_datetime_utc"].values.astype("datetime64[ns]").astype("int64")
    v_lat, v_lon, v_frp = v["latitude"].values, v["longitude"].values, v["frp"].values
    margin = int(max_min * 60 * 1e9)

    for _, idx in m.groupby("date").groups.items():
        idx = np.asarray(idx)
        g = m.loc[idx]
        mt = g["acq_datetime_utc"].values.astype("datetime64[ns]").astype("int64")
        lo, hi = np.searchsorted(vt, [mt.min() - margin, mt.max() + margin])
        if hi <= lo:
            continue
        tree = cKDTree(_xyz(v_lat[lo:hi], v_lon[lo:hi]))
        half_x, half_y = g["scan"].values / 2.0, g["track"].values / 2.0
        lists = tree.query_ball_point(
            _xyz(g["latitude"].values, g["longitude"].values), r=np.hypot(half_x, half_y) + 0.05
        )
        counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        if counts.sum() == 0:
            continue
        mi = np.repeat(np.arange(len(g)), counts)
        vi = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(counts.sum())) + lo

        glat, glon = g["latitude"].values[mi], g["longitude"].values[mi]
        dlon = (v_lon[vi] - glon + 180.0) % 360.0 - 180.0
        dx = dlon * KM_PER_DEG * np.cos(np.radians(glat))
        dy = (v_lat[vi] - glat) * KM_PER_DEG
        dt_min = np.abs(vt[vi] - mt[mi]) / 6e10
        ok = (np.abs(dx) <= half_x[mi]) & (np.abs(dy) <= half_y[mi]) & (dt_min <= max_min)
        if not ok.any():
            continue
        # one VIIRS pixel -> one MODIS pixel (nearest centre in footprint-normalised units)
        d2 = (dx[ok] / half_x[mi[ok]]) ** 2 + (dy[ok] / half_y[mi[ok]]) ** 2
        cand = pd.DataFrame({"mi": mi[ok], "vi": vi[ok], "d2": d2, "dt": dt_min[ok]})
        cand = cand.sort_values("d2").drop_duplicates("vi", keep="first")
        cand["frp"] = v_frp[cand["vi"].values]
        agg = cand.groupby("mi").agg(n=("frp", "size"), s=("frp", "sum"), mx=("frp", "max"), dt=("dt", "median"))
        rows = idx[agg.index.values]
        m.loc[rows, "n_viirs"] = agg["n"].values
        m.loc[rows, "viirs_frp_sum"] = agg["s"].values
        m.loc[rows, "viirs_frp_max"] = agg["mx"].values
        m.loc[rows, "abs_dt_min"] = agg["dt"].values
    return m


# --------------------------------------------------------------------------- FRP scale
def _usable(pairs):
    u = pairs[(pairs["n_viirs"] > 0) & (pairs["frp"] > 0) & (pairs["viirs_frp_sum"] > 0)]
    return u.sort_values("acq_datetime_utc")


def _cluster_arr(df, key="cell_id"):
    """(n_clusters, 2) = [MODIS FRP, VIIRS footprint FRP] summed per cluster."""
    return df.groupby(key)[["frp", "viirs_frp_sum"]].sum().to_numpy()


def _log_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(np.log(a)) == 0 or np.std(np.log(b)) == 0:
        return float("nan")
    return float(np.corrcoef(np.log(a), np.log(b))[0, 1])


def per_year_fits(use, rng):
    """k per calendar year with a cell-clustered bootstrap SE, plus the leave-one-year-out check judged against
    SAMPLING error: z = ln(k_other_years / k_year) / sqrt(se_year^2 + se_other_years^2). A year whose k differs by
    30 % but whose own SE is 25 % is NOT evidence of instability."""
    out = {}
    yrs = use["acq_datetime_utc"].dt.year
    for y in sorted(yrs.unique()):
        g, rest = use[yrs == y], use[yrs != y]
        ay, ar = _cluster_arr(g), _cluster_arr(rest)
        if len(g) < MIN_PAIRS_FIT or len(ay) < MIN_YEAR_CELLS:
            continue
        k_y, lo, hi, se_y = ratio_ci(ay, rng, N_BOOT)
        d = {"n_pairs": int(len(g)), "n_cells": int(len(ay)), "k": k_y, "k_ci95": [lo, hi], "se_log_k": se_y,
             "k_from_other_years_pred_over_actual": None, "loyo_z": None}
        if len(ar) >= MIN_YEAR_CELLS:
            k_r, _, _, se_r = ratio_ci(ar, rng, N_BOOT)
            if k_y > 0 and k_r > 0:
                d["k_from_other_years_pred_over_actual"] = float(k_r / k_y)
                den = float(np.sqrt(se_y ** 2 + se_r ** 2))
                if den > 0:                      # a zero standard error gives no z (and no divide-by-zero warning)
                    d["loyo_z"] = float(np.log(k_r / k_y) / den)
        out[str(y)] = d
    return out


def k_trust_grade(use, k, ci, by_year, het):
    """Grade k (HIGH / MEDIUM / LOW) and list exactly what is missing for HIGH.
    Stability is judged against sampling error, not by the max-min spread of per-year k (see kstats.heterogeneity):
    HIGH needs no significant between-year heterogeneity (Cochran Q p >= 0.05) AND a 95 % upper limit of the
    between-year SD of log k (tau) of at most PYRO_K_TAU_MAX, so a 10-year record that merely CANNOT exclude large
    year-to-year variation is not certified as stable."""
    n_days = int(use["acq_datetime_utc"].dt.floor("D").nunique())
    ci_rel = float((ci[1] - ci[0]) / k)
    problems = []
    if len(use) < K_TRUST_MIN_PAIRS:
        problems.append(f"{len(use)} matched pixels (need >= {K_TRUST_MIN_PAIRS})")
    if n_days < 30:
        problems.append(f"matches come from only {n_days} distinct days (need >= 30)")
    if len(by_year) < 3:
        problems.append(f"only {len(by_year)} year(s) with >= {MIN_PAIRS_FIT} matched pixels (need >= 3)")
    if ci_rel > 0.2:
        problems.append(f"bootstrap 95% CI is {ci_rel:.1%} of k wide (need <= 20%)")
    if het is not None:
        if het["p_Q"] < 0.05:
            problems.append(f"k differs between years by more than sampling error (Cochran Q p = {het['p_Q']:.3f})")
        if het["tau_ci95"][1] > K_TAU_MAX:
            problems.append(f"between-year variability of k cannot be bounded: tau = {het['tau']:.0%}, 95% upper "
                            f"limit {het['tau_ci95'][1]:.0%} (need <= {K_TAU_MAX:.0%})")
    z_crit = float(stats.norm.isf(0.025 / max(len(by_year), 1)))   # Bonferroni: one test per year at 5 % overall
    bad = [y for y, d in by_year.items() if d.get("loyo_z") is not None and abs(d["loyo_z"]) > z_crit]
    if bad:
        problems.append(f"k from the other years misses year(s) {bad} by more than sampling error "
                        f"(|z| > {z_crit:.2f}, Bonferroni over {len(by_year)} years)")
    if K_WINDOW_NOTE["widened"]:
        problems.append(f"k fitted with the time-gap window widened from {K_WINDOW_NOTE['strict_min']:g} to "
                        f"{K_WINDOW_NOTE['used_min']:g} min: part of the MODIS/VIIRS ratio is fire evolution between "
                        f"the two overpasses, not sensor difference")
    grade = "HIGH" if not problems else ("LOW" if len(use) < K_TRUST_MIN_PAIRS else "MEDIUM")
    return grade, problems, n_days, ci_rel


K_WINDOW_NOTE = {"widened": False, "strict_min": None, "used_min": None}


def resolve_k_window(window_ok):
    """Choose the time-gap window (minutes) inside which matched pixels are used to fit k.
    window_ok(w) -> True when at least one MODIS satellite has >= MIN_PAIRS_FIT matched pixels in >= MIN_CELLS_FIT
    cells with |dt| <= w. The configured window is kept whenever it works. Only if it does not, and PYRO_K_MAX_DT_MIN
    is "auto", the smallest step of K_FIT_WIDEN_STEPS that works is used. The choice is recorded in K_WINDOW_NOTE
    (-> frp_mapping.json) and a widened window can never be graded HIGH (k_trust_grade)."""
    global K_FIT_MAX_DT_MIN
    strict = float(K_FIT_MAX_DT_MIN)
    K_WINDOW_NOTE.update(widened=False, strict_min=strict, used_min=strict)
    if not K_FIT_AUTO_WIDEN or window_ok(strict):
        return K_FIT_MAX_DT_MIN
    for w in K_FIT_WIDEN_STEPS:
        if w > strict and window_ok(float(w)):
            K_FIT_MAX_DT_MIN = float(w)
            K_WINDOW_NOTE.update(widened=True, used_min=float(w))
            print(f"  k window: no MODIS satellite has enough matches within {strict:g} min of VIIRS; widened to "
                  f"{w:g} min (k_trust cannot be HIGH; set PYRO_K_MAX_DT_MIN=<minutes> to fix the window)")
            break
    return K_FIT_MAX_DT_MIN


def apply_single_day_demo(cal, ref_sat):
    """PYRO_SINGLE_DAY_DEMO=1: a partial day on which the reference VIIRS satellite delivered data becomes
    `demo_partial_window` with the VIIRS total as its harmonized value (a lower bound for that day). It is NOT a
    measured day: harmonized_is_measured stays False and anomaly.py never scores it."""
    if not SINGLE_DAY_DEMO:
        return cal
    demo = ((cal["harmonized_status"] == "partial_day") & cal["viirs_observed"] & cal["viirs_detections"].notna()).to_numpy()
    if demo.any():
        cal.loc[demo, "harmonized_status"] = "demo_partial_window"
        cal.loc[demo, "harmonized_frp_mw_viirs_equiv"] = cal.loc[demo, "viirs_frp_mw"]
        cal.loc[demo, "harmonized_source"] = f"VIIRS {ref_sat} (demo, partial window)"
    return cal


def fit_scale(pairs, label=""):
    """Constant aggregate scale k (VIIRS-equivalent FRP = k * MODIS FRP) for ONE MODIS satellite, plus (info["k_model"])
    the conditional version k(fire size, day/night, time gap) that kmodel.py fits and cross-validates on the same pixels.
    The returned k is always the constant ratio of sums; whether the conditional model is applied is decided in
    decide_scales. Estimated over footprints where BOTH sensors detected fire (conditional on a MODIS detection), so it is not a
    detection probability and it ignores the many small fires only VIIRS sees.

    Returns (k, info). k is None when the satellite has fewer than MIN_PAIRS_FIT matches within K_FIT_MAX_DT_MIN
    minutes of S-NPP: overpasses further apart sample different fire states, so a ratio over all gaps is reported
    only as `k_diagnostic_all_gaps` and is NOT a calibration."""
    use_all = _usable(pairs)
    close = use_all[use_all["abs_dt_min"] <= K_FIT_MAX_DT_MIN]
    info = {"modis_satellite": label, "k": None, "k_trust": "NOT_VALIDATED", "k_trust_problems": [],
            "n_pairs": int(len(close)), "n_pairs_any_time_gap": int(len(use_all)), "fit_max_dt_min": K_FIT_MAX_DT_MIN}
    rng = np.random.default_rng(0)
    cells_close = close.groupby("cell_id").ngroups if len(close) else 0
    if len(close) < MIN_PAIRS_FIT or cells_close < MIN_CELLS_FIT:
        info["k_trust_problems"] = [f"only {len(close)} matched pixels within {K_FIT_MAX_DT_MIN:g} min of VIIRS "
                                    f"(need >= {MIN_PAIRS_FIT} in >= {MIN_CELLS_FIT} cells)"]
        info["note"] = ("no close-in-time overpass pairs: k cannot be validated for this satellite and it receives "
                        "NO VIIRS-equivalent FRP unless PYRO_APPLY_UNVALIDATED_K=1")
        if len(use_all) >= MIN_PAIRS_FIT and use_all.groupby("cell_id").ngroups >= MIN_CELLS_FIT:
            kd, lo, hi, _ = ratio_ci(_cluster_arr(use_all), rng, N_BOOT)
            info["k_diagnostic_all_gaps"] = {
                "k": kd, "k_ci95": [lo, hi], "n_pairs": int(len(use_all)),
                "median_abs_dt_min": float(use_all["abs_dt_min"].median()),
                "warning": "overpasses this far apart see different fire states (diurnal cycle): NOT a calibration"}
        return None, info

    use = close
    arr = _cluster_arr(use)
    k, lo, hi, se_log = ratio_ci(arr, rng, N_BOOT)
    cut = int(len(use) * 0.7)  # chronological split, never random
    tr, te = use.iloc[:cut], use.iloc[cut:]
    k_tr = tr["viirs_frp_sum"].sum() / tr["frp"].sum()
    date = use["acq_datetime_utc"].dt.floor("D")
    day_cell = use.groupby([use["cell_id"], date])[["frp", "viirs_frp_sum"]].sum()
    cells = use.groupby("cell_id")[["frp", "viirs_frp_sum"]].sum()
    by_year = per_year_fits(use, rng)
    het = None
    if len(by_year) >= 3:
        het = heterogeneity([np.log(d["k"]) for d in by_year.values()], [d["se_log_k"] for d in by_year.values()])
    ks = [d["k"] for d in by_year.values()]
    grade, problems, n_days, ci_rel = k_trust_grade(use, k, [lo, hi], by_year, het)
    pi = prediction_interval(k, se_log, None if het is None else het["tau"], len(by_year))
    sites = use[["latitude", "longitude"]].round(2).drop_duplicates()
    info.update({
        "k": float(k), "k_ci95": [lo, hi], "k_ci_includes_1": bool(lo <= 1.0 <= hi), "k_se_log": se_log,
        "k_prediction_interval95_new_year": pi,
        "n_cells": int(len(cells)), "n_sites_0p01deg": int(len(sites)),
        "holdout": {
            "n_test": int(len(te)),
            "total_pred_over_actual_with_k": float(k_tr * te["frp"].sum() / te["viirs_frp_sum"].sum()),
            "total_pred_over_actual_identity": float(te["frp"].sum() / te["viirs_frp_sum"].sum()),
        },
        "pixel_corr_log": _log_corr(use["frp"], use["viirs_frp_sum"]),
        # NOTE: cell_total_corr_log correlates per-cell TOTALS over the whole record (a spatial climatology, inflated
        # by aggregation); cell_day_corr_log correlates cell x day totals, the fairer cell-level agreement.
        "cell_total_corr_log": _log_corr(cells["frp"], cells["viirs_frp_sum"]),
        "cell_corr_log": _log_corr(cells["frp"], cells["viirs_frp_sum"]),
        "cell_day_corr_log": _log_corr(day_cell["frp"], day_cell["viirs_frp_sum"]),
        "median_abs_dt_min": float(use["abs_dt_min"].median()),
        "k_trust": grade, "k_trust_problems": problems, "k_by_year": by_year,
        "k_year_spread_max_minus_min_over_median": float((max(ks) - min(ks)) / np.median(ks)) if len(ks) >= 2 else None,
        "k_year_heterogeneity": het, "n_days_used": n_days, "k_ci_rel_width": ci_rel,
        "interpretation": _interpret(by_year, het, ks),
    })
    # separate generator: leaves every number above bit-identical to a run without the model
    info["k_model"] = (kmodel.fit_k_model(use, np.random.default_rng(1), K_MODEL_BOOT, K_MODEL_CLIP_Q, K_MODEL_FOLDS)
                       if K_MODEL else {"available": False, "adopted": False, "model": None,
                                        "reason": "disabled (PYRO_K_MODEL=0)"})
    return float(k), info


def _interpret(by_year, het, ks):
    if het is None or len(ks) < 2:
        return "fewer than 3 years with enough matches: between-year stability cannot be assessed"
    spread = (max(ks) - min(ks)) / np.median(ks)
    se_med = float(np.median([d["se_log_k"] for d in by_year.values()]))
    verdict = ("per-year k differ by more than their sampling error" if het["p_Q"] < 0.05
               else "the year-to-year scatter is compatible with sampling error alone")
    return (f"per-year k spans {spread:.0%} (max-min / median) but the median per-year standard error is "
            f"{se_med:.0%}: {verdict} (Cochran Q p = {het['p_Q']:.2f}, I2 = {het['I2']:.0%}). Estimated between-year SD "
            f"tau = {het['tau']:.0%} (95% interval {het['tau_ci95'][0]:.0%}-{het['tau_ci95'][1]:.0%}): the data cannot "
            f"exclude a real year-to-year variation of that size, which is carried into k_prediction_interval95_new_year.")


def k_by_group(pairs, column):
    """k recomputed inside each value of `column` (MODIS product version, day/night, ...), on the same
    close-in-time pixels as the main fit. Groups with fewer than MIN_PAIRS_FIT pixels are left out."""
    if pairs.empty or column not in pairs.columns:
        return {}
    use = _usable(pairs)
    use = use[use["abs_dt_min"] <= K_FIT_MAX_DT_MIN]
    out = {}
    for key, g in use.groupby(use[column].astype(str)):
        if len(g) >= MIN_PAIRS_FIT:
            out[key] = {"n_pairs": int(len(g)), "k": float(g["viirs_frp_sum"].sum() / g["frp"].sum())}
    return out


def decide_scales(fits):
    """Which satellites receive a VIIRS-equivalent FRP, and on what basis.
    Default: only satellites whose k was fitted on close-in-time matches; every other satellite gets NaN.
      'own'              constant k (no covariate beat it in cross-validation, or PYRO_K_MODEL=0)
      'own_conditional'  k depends on fire size (and day/night, time gap) as selected by cross-validation (kmodel.py)
    PYRO_APPLY_UNVALIDATED_K=1: also apply a low-trust own fit ('own_low_trust') or, failing that, the CONSTANT k of the
    best-validated satellite ('transferred:<sat>'), all labelled. A conditional model is never transferred."""
    scale, basis = {}, {}
    for s, (k, info) in fits.items():
        if k is not None:
            scale[s] = {"k": k, "k_lo": info["k_prediction_interval95_new_year"][0],
                        "k_hi": info["k_prediction_interval95_new_year"][1]}
            basis[s] = "own"
            km = info.get("k_model") or {}
            if km.get("adopted") and km.get("model"):
                scale[s]["model"] = km["model"]
                basis[s] = "own_conditional"
    donors = sorted(scale, key=lambda s: -fits[s][1]["n_pairs"])
    for s, (k, info) in fits.items():
        if s in scale:
            continue
        basis[s] = "none"
        if not APPLY_UNVALIDATED_K:
            continue
        diag = info.get("k_diagnostic_all_gaps")
        if diag is not None:
            scale[s] = {"k": diag["k"], "k_lo": diag["k_ci95"][0], "k_hi": diag["k_ci95"][1]}
            basis[s] = "own_low_trust"
        elif donors:
            d = scale[donors[0]]
            scale[s] = {c: d[c] for c in ("k", "k_lo", "k_hi")}
            basis[s] = f"transferred:{donors[0]}"
    return scale, basis


def apply_scale(df, scale, basis=None):
    """Adds frp_equiv (+ lo/hi from the prediction interval of k) and scale_basis. NaN where the satellite has no
    scale: a missing k is never replaced by 1 or by another satellite's k unless decide_scales allowed it.
    A satellite with a conditional model gets frp * k(frp, day/night) evaluated at dt = 0 (simultaneous overpass); its
    lo / hi keep the RELATIVE width of the constant-k prediction interval (aggregate sampling error + between-year tau)."""
    df = df.copy()
    if (df["sensor_family"] == "VIIRS").all():
        df["frp_equiv"] = df["frp"]
        df["frp_equiv_lo"], df["frp_equiv_hi"] = df["frp"], df["frp"]
        df["scale_basis"] = "reference"
        return df
    sat = df["source_satellite"]
    frp = df["frp"].to_numpy(float)
    night = kmodel.night_flag(df["daynight"]) if "daynight" in df.columns else None
    eq, lo, hi = (np.full(len(df), np.nan) for _ in range(3))
    for s, d in scale.items():
        sel = (sat == s).to_numpy()
        if not sel.any():
            continue
        if d.get("model"):
            r = kmodel.predict_ratio(d["model"], frp[sel], None if night is None else night[sel])
            eq[sel] = frp[sel] * r
            lo[sel], hi[sel] = eq[sel] * (d["k_lo"] / d["k"]), eq[sel] * (d["k_hi"] / d["k"])
        else:
            eq[sel], lo[sel], hi[sel] = frp[sel] * d["k"], frp[sel] * d["k_lo"], frp[sel] * d["k_hi"]
    df["frp_equiv"], df["frp_equiv_lo"], df["frp_equiv_hi"] = eq, lo, hi
    df["scale_basis"] = sat.map(basis or {}).fillna("none")
    return df


def applied_scale(m, scale):
    """The aggregate scale actually applied to each MODIS satellite: sum(frp_equiv) / sum(frp). For a constant k this
    is k; for a conditional model it is the FRP-weighted mean of k(frp, ...) over that satellite's detections."""
    out = {}
    for s in sorted(m["source_satellite"].unique()):
        g = m[m["source_satellite"] == s]
        if s not in scale or g["frp"].sum() <= 0:
            out[s] = None
        elif not scale[s].get("model"):
            out[s] = float(scale[s]["k"])
        else:
            out[s] = float(g["frp_equiv"].sum() / g["frp"].sum())
    return out


# --------------------------------------------------------------------------- calendar
def day_complete_flags(dates, t0, t1):
    tol = pd.Timedelta(minutes=DAY_TOL_MIN)
    t0n, t1n = t0.tz_localize(None), t1.tz_localize(None)
    return {d: (t0n <= d + tol) and (t1n >= d + pd.Timedelta(days=1) - tol) for d in dates}


def overpass_clusters(df, gap_min=None):
    """Distinct overpass clusters per UTC day, or reuse exact precomputed counts on cell/day aggregates."""
    gap_min = COV_CLUSTER_GAP_MIN if gap_min is None else gap_min
    if df.empty:
        return pd.Series(dtype=float)
    if "overpass_cluster_count" in df.columns:
        return df.groupby("date")["overpass_cluster_count"].max().astype(float)
    t = df["acq_datetime_utc"]
    mins = ((t - t.dt.floor("D")).dt.total_seconds() / 60.0).to_numpy()
    s = pd.DataFrame({"date": df["date"].to_numpy(), "m": mins}).sort_values(["date", "m"])
    new = s.groupby("date")["m"].diff().fillna(np.inf) > gap_min
    return new.groupby(s["date"]).sum().astype(float)


def flag_low_coverage(det, clusters, min_frac=None, min_expected=None, extreme_frac=None):
    """Partial-outage detector. det / clusters: Series indexed by day (det NaN where the sensor was not observed).
    expected = local (+-15 d) median number of detections. A day is flagged when expected >= `min_expected` (a sparse
    series cannot reveal an outage this way) AND either
      (a) it has <= 1 overpass cluster and fewer than `min_frac` x expected detections (a pass is missing), or
      (b) it has fewer than `extreme_frac` x expected detections (a level reached by < 1 % of days; real fire variability
          does not remove 80 % of the detections of a region with persistent static sources).
    A flagged day keeps its number but is labelled measured_low_coverage: it is a lower bound, not a measurement."""
    min_frac = COV_MIN_FRAC if min_frac is None else min_frac
    extreme_frac = COV_EXTREME_FRAC if extreme_frac is None else extreme_frac
    min_expected = COV_MIN_EXPECTED if min_expected is None else min_expected
    expected = det.rolling(31, center=True, min_periods=10).median()
    low = (expected >= min_expected) & ((det < extreme_frac * expected) | ((clusters <= 1) & (det < min_frac * expected)))
    return low.fillna(False).astype(bool), expected


STATUS_MEASURED = ("measured", "measured_zero", "measured_low_coverage")


def build_calendar(m, v, ref_sat, t0, t1, observed=None, scale_basis=None, clear=None, min_clear=None,
                   obscure_nonzero=None):
    """Daily calendar + cell-day table.

    observed : {"viirs": set-of-days | None, "<MODIS satellite>": set-of-days | None}; None / missing key = unknown,
               treated as observed (and said so in frp_mapping.json).
    clear    : optional Series (index = day) with the clear-sky share 0..1 of the area on that day for the reference VIIRS
               satellite (cloud_mask.clear_series); NaN / a missing day = cloud cover unknown (NOT clear). None = no cloud
               information: the calendar is then exactly what it was before this option existed (no extra column).
    min_clear / obscure_nonzero : a day with clear share < min_clear (default PYRO_CLOUD_MIN_CLEAR) and 0 detections is
               `observed_obscured`; with obscure_nonzero (default PYRO_CLOUD_OBSCURE_NONZERO) so is a cloud-limited day that
               had detections.
    Every sensor-level number is NaN on a day that sensor did not deliver data; `harmonized_status` is explicit."""
    observed = observed or {}
    scale_basis = scale_basis or {}
    min_clear = CLOUD_MIN_CLEAR if min_clear is None else min_clear
    obscure_nonzero = CLOUD_OBSCURE_NONZERO if obscure_nonzero is None else obscure_nonzero
    days = pd.date_range(t0.tz_localize(None).floor("D"), t1.tz_localize(None).floor("D"), freq="D")
    cal = pd.DataFrame(index=days)
    cal.index.name = "date"
    vr = v[v["source_satellite"] == ref_sat]
    mod_sats = sorted(m["source_satellite"].unique())

    def obs_mask(key):
        s = observed.get(key)
        return np.ones(len(days), bool) if s is None else days.isin(list(s))

    complete = np.array([f for f in day_complete_flags(days, t0, t1).values()])
    v_obs = obs_mask("viirs")
    sat_obs = {s: obs_mask(s) for s in mod_sats}
    m_obs = np.logical_and.reduce(list(sat_obs.values())) if sat_obs else np.ones(len(days), bool)

    def block(df, mask, name):
        g = df.groupby("date")
        det = g["detection_count"].sum() if "detection_count" in df.columns else g.size()
        out = pd.DataFrame({"det": det, "frp": g["frp"].sum(), "cells": g["cell_id"].nunique()}).reindex(days)
        out[["det", "frp", "cells"]] = out[["det", "frp", "cells"]].fillna(0.0)
        out.loc[~mask, :] = np.nan
        cal[f"{name}_detections"] = out["det"].astype("Int64")
        cal[f"{name}_frp_mw"] = out["frp"]
        return out

    mb = block(m, m_obs, "modis")
    cal["modis_active_cells"] = mb["cells"].astype("Int64")
    for s in mod_sats:                       # Aqua and Terra are reported separately, never only as "MODIS"
        block(m[m["source_satellite"] == s], sat_obs[s], slug(s))
        cal[f"{slug(s)}_observed"] = sat_obs[s]
    # VIIRS-equivalent MODIS FRP: only satellites that HAVE a scale; NaN (not 0) when none does
    eq = m.dropna(subset=["frp_equiv"])
    scaled = sorted(eq["source_satellite"].unique())
    if scaled:
        eq_obs = np.logical_and.reduce([sat_obs[s] for s in scaled])
        g = eq.groupby("date")
        for col, src in [("modis_frp_viirs_equiv_mw", "frp_equiv"), ("modis_frp_viirs_equiv_lo_mw", "frp_equiv_lo"),
                         ("modis_frp_viirs_equiv_hi_mw", "frp_equiv_hi")]:
            x = g[src].sum().reindex(days).fillna(0.0)
            x[~eq_obs] = np.nan
            cal[col] = x
    else:
        for col in ("modis_frp_viirs_equiv_mw", "modis_frp_viirs_equiv_lo_mw", "modis_frp_viirs_equiv_hi_mw"):
            cal[col] = np.nan
    cal["modis_frp_viirs_equiv_satellites"] = ",".join(scaled) if scaled else "none"

    vb = block(vr, v_obs, "viirs")
    cal["viirs_active_cells"] = vb["cells"].astype("Int64")
    if len(vr) and "i4_saturated_count" in vr.columns and "detection_count" in vr.columns:
        sat_share = (vr.groupby("date")["i4_saturated_count"].sum() /
                     vr.groupby("date")["detection_count"].sum()).reindex(days)
    else:
        sat_share = vr.groupby("date")["i4_saturated"].mean().reindex(days) if len(vr) else pd.Series(np.nan, index=days)
    sat_share[~v_obs] = np.nan
    cal["viirs_i4_saturated_share"] = sat_share
    clus = overpass_clusters(vr).reindex(days).fillna(0.0)
    clus[~v_obs] = np.nan
    cal["viirs_overpass_clusters"] = clus
    low, expected = flag_low_coverage(vb["det"], clus)
    cal["viirs_expected_detections"] = expected
    cal["viirs_low_coverage"] = low.to_numpy()

    mc = m[["date", "cell_id"]].drop_duplicates()
    vc = vr[["date", "cell_id"]].drop_duplicates()
    both = mc.merge(vc, on=["date", "cell_id"]).groupby("date").size().reindex(days).fillna(0.0)
    both[~(v_obs & m_obs)] = np.nan
    cal["cells_active_in_both"] = both.astype("Int64")
    mac = mb["cells"]
    cal["modis_cells_also_viirs"] = np.where(mac > 0, both / mac.where(mac > 0), np.nan)

    # --- what each row IS: measured / zero / outage-suspect / obscured / not observed / partial day
    det = cal["viirs_detections"].fillna(-1).to_numpy()
    obscured = np.zeros(len(days), bool)
    if clear is not None:
        c = pd.Series(clear).reindex(days).to_numpy(float)       # unknown (NaN) is never clear and never cloudy
        limited = np.isfinite(c) & (c < min_clear)
        # an empty observed day under cloud is not "no fire"; the cloud explanation also wins over the outage detector,
        # because few detections on a cloudy day are expected, whereas few detections on a clear day are suspicious
        obscured = limited & ((det == 0) | bool(obscure_nonzero))
        cal["viirs_clear_fraction"] = c
        cal["viirs_cloud_limited"] = limited & v_obs & complete
    status = np.select(
        [~complete, ~v_obs, obscured, cal["viirs_low_coverage"].to_numpy(), det == 0],
        ["partial_day", "viirs_not_observed", "observed_obscured", "measured_low_coverage", "measured_zero"],
        default="measured")
    cal["day_complete"] = complete
    cal["viirs_observed"] = v_obs
    cal["modis_observed"] = m_obs
    cal["day_observed"] = v_obs & m_obs          # legacy: both sensors present in their raw files
    cal["harmonized_status"] = status
    is_meas = np.isin(status, STATUS_MEASURED)
    cal["harmonized_is_measured"] = is_meas
    cal["harmonized_frp_mw_viirs_equiv"] = np.where(is_meas, cal["viirs_frp_mw"], np.nan)
    cal["harmonized_source"] = np.where(is_meas, f"VIIRS {ref_sat}", "none")
    apply_single_day_demo(cal, ref_sat)
    eq_mw = cal["modis_frp_viirs_equiv_mw"].to_numpy()
    cal["modis_only_evidence"] = (status == "viirs_not_observed") & (np.nan_to_num(eq_mw, nan=0.0) > 0)
    cal["aoi"] = "global" if BBOX is None else ",".join(str(x) for x in BBOX)
    if AOI_FILE is not None:
        cal["aoi"] += f" | mask {AOI_FILE.name}"
    if EXCLUDE_STATIC:
        cal["aoi"] += " | non-static only (type 0)"

    rows = pd.concat([m, v])
    rows["has_eq"] = rows["frp_equiv"].notna()
    rows["eq0"] = rows["frp_equiv"].fillna(0.0)
    detections_agg = ("detection_count", "sum") if "detection_count" in rows.columns else ("frp", "size")
    cells = rows.groupby(["sensor_family", "source_satellite", "date", "cell_id", "cell_lat", "cell_lon"], as_index=False).agg(
        detections=detections_agg, frp_mw=("frp", "sum"), frp_viirs_equiv_mw=("eq0", "sum"), n_eq=("has_eq", "sum"))
    cells.loc[cells["n_eq"] == 0, "frp_viirs_equiv_mw"] = np.nan   # no scale != zero FRP
    cells = cells.drop(columns="n_eq")
    cells["cell_area_km2"] = cell_area_km2(cells["cell_lat"], CELL_DEG)
    cells["scale_basis"] = np.where(cells["sensor_family"] == "VIIRS", "reference",
                                    cells["source_satellite"].map(scale_basis).fillna("none"))
    return cal.reset_index(), cells


def sensitivity_summary(cal):
    """Detection-sensitivity ratios on days where both sensors were observed and complete (VIIRS 375 m sees many
    more, smaller fires). ALL-MODIS basis (Terra + Aqua), kept for continuity; the like-for-like Aqua x S-NPP block
    (same overpass time) is added by harmonize_validation.grid_sensitivity for the grid in use."""
    full = cal[cal["day_complete"] & cal["day_observed"]]
    if full.empty:
        return {"note": "no complete, observed UTC day in the common window"}
    s = lambda c: float(full[c].astype(float).sum())
    return {
        "basis": "all MODIS satellites (Terra + Aqua) vs the VIIRS reference",
        "complete_days_used": int(len(full)),
        "viirs_over_modis_detections": s("viirs_detections") / max(s("modis_detections"), 1),
        "viirs_over_modis_active_cells": s("viirs_active_cells") / max(s("modis_active_cells"), 1),
        "share_of_modis_active_cells_also_viirs": s("cells_active_in_both") / max(s("modis_active_cells"), 1),
        "cell_deg": CELL_DEG,
    }


def daily_total_relationship(cal):
    """Why MODIS cannot stand in for a missing VIIRS day: on days where VIIRS was measured, how well do the
    k-scaled MODIS daily FRP totals reproduce the VIIRS totals?"""
    if cal["modis_frp_viirs_equiv_mw"].notna().sum() == 0:
        return {"note": "no MODIS satellite has a scale"}
    d = cal[cal["harmonized_is_measured"] & cal["modis_frp_viirs_equiv_mw"].notna()]
    if len(d) < 30:
        return {"note": f"only {len(d)} usable days"}
    x, y = d["modis_frp_viirs_equiv_mw"].to_numpy(float), d["viirs_frp_mw"].to_numpy(float)
    nz = x > 0
    r = float(np.corrcoef(np.log1p(x), np.log1p(y))[0, 1])
    return {
        "modis_satellites": cal["modis_frp_viirs_equiv_satellites"].iloc[0],
        "n_days": int(len(d)), "share_days_modis_zero": float(1 - nz.mean()),
        "corr_log1p": r, "r2_log1p": r * r,
        "total_viirs_over_scaled_modis": float(y.sum() / max(x.sum(), 1e-9)),
        "median_daily_viirs_over_scaled_modis_when_modis_detects": float(np.median(y[nz] / x[nz])) if nz.any() else None,
        "conclusion": ("k aligns FRP inside co-detected footprints only; MODIS misses most of the small fires VIIRS "
                       "375 m sees, so its scaled daily total is biased low and weakly predictive. A missing VIIRS day "
                       "is therefore left missing (modis_only_evidence marks days where MODIS alone saw fire)."),
    }


# --------------------------------------------------------------------------- plot
def plot_matchup(pairs, k, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allp = _usable(pairs)
    close = allp[allp["abs_dt_min"] <= K_FIT_MAX_DT_MIN]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    far = allp.drop(close.index, errors="ignore")
    if len(far):
        ax.scatter(far["frp"], far["viirs_frp_sum"], s=4, alpha=0.15, color="#8a8a85", label=f"|dt| > {K_FIT_MAX_DT_MIN:g} min (not used)")
    ax.scatter(close["frp"], close["viirs_frp_sum"], s=6, alpha=0.35, label=f"|dt| <= {K_FIT_MAX_DT_MIN:g} min (fit)")
    lim = np.array([max(allp["frp"].min(), 0.1), allp["frp"].max()])
    ax.plot(lim, lim, "k--", lw=1, label="identity (1:1)")
    if k:
        ax.plot(lim, k * lim, "r-", lw=2, label=f"constant k = {k:.2f} x MODIS")
    km = info.get("k_model") or {}
    if km.get("adopted") and km.get("curve"):
        c = km["curve"]
        f = np.array([r["frp"] for r in c])
        ax.fill_between(f, f * np.array([r["k_lo"] for r in c]), f * np.array([r["k_hi"] for r in c]), color="#1f77b4", alpha=0.18)
        ax.plot(f, f * np.array([r["k"] for r in c]), color="#1f77b4", lw=2, label="conditional k(FRP) at dt = 0 (95% band)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"MODIS {info.get('modis_satellite', '')} pixel FRP (MW)")
    ax.set_ylabel("Sum of VIIRS 375 m FRP inside the MODIS footprint (MW)")
    ax.set_title(f"{info['n_pairs']:,} matched pixels, median |dt| {info.get('median_abs_dt_min', float('nan')):.0f} min\n"
                 f"pixel r(log)={info.get('pixel_corr_log', float('nan')):.2f}, "
                 f"{CELL_DEG:g}° cell x day r(log)={info.get('cell_day_corr_log', float('nan')):.2f}")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def cloud_summary(cal, info, min_clear, obscure_nonzero):
    """What the cloud information did to the calendar (counts), added to the info of cloud_mask.clear_series."""
    status = cal["harmonized_status"]
    limited = cal["viirs_cloud_limited"].astype(bool)
    det = cal["viirs_detections"].fillna(-1)
    info = dict(info, enabled=True, min_clear_fraction=float(min_clear), obscure_nonzero=bool(obscure_nonzero))
    info.update({
        "days_cloud_limited": int(limited.sum()),
        "days_marked_observed_obscured": int((status == "observed_obscured").sum()),
        "cloud_limited_days_with_detections_kept": int((limited & (det > 0) & status.isin(STATUS_MEASURED)).sum()),
        "reading": ("observed_obscured: the sensor delivered data but the clear-sky share is below the threshold and "
                    "there was no detection, so 'no fire' cannot be claimed; the harmonized value is NaN and the day is "
                    "not scored by the anomaly detector. Days without cloud information keep their old status."),
    })
    return info


def _estimate_csv_rows(path, sample_bytes=1_048_576):
    """Estimate CSV rows from a small byte sample (rather than assuming a fixed row width)."""
    size = path.stat().st_size
    if size <= 0:
        return 0
    with path.open("rb") as stream:
        sample = stream.read(sample_bytes)
    line_count = sample.count(b"\n")
    if line_count < 2:
        return max(0, int(size / 120) - 1)
    mean_bytes_per_line = len(sample) / line_count
    return max(0, int(size / max(mean_bytes_per_line, 1.0)) - 1)


def main():
    ensure_dirs()
    modis_path = PROCESSED_DIR / "modis_standardized.csv"
    viirs_path = PROCESSED_DIR / "viirs_standardized.csv"
    if not modis_path.exists() or not viirs_path.exists():
        raise SystemExit("Run `python -m core.src.preprocess` first; both standardized sensor files are required.")
    estimated_rows = _estimate_csv_rows(modis_path) + _estimate_csv_rows(viirs_path)
    force = OUT_OF_CORE in {"1", "true", "yes", "on"}
    disable = OUT_OF_CORE in {"0", "false", "no", "off"}
    if force or (not disable and estimated_rows >= OUT_OF_CORE_ROWS):
        from core.src.harmonize_streaming import run as run_streaming
        run_streaming()
        return
    cloud_table = cloud_mask.read_table(CLOUD_FILE) if CLOUD_FILE is not None else None   # fail before the slow steps
    m = load(modis_path)
    v = load(viirs_path)
    print(f"MODIS {len(m):,} rows, VIIRS {len(v):,} rows before clamping")
    m, v, t0, t1 = clamp_to_common_window(m, v)
    print(f"Common window: {t0} -> {t1}  (MODIS {len(m):,}, VIIRS {len(v):,} rows kept)")
    m, v = add_features(m), add_features(v)
    m["confidence_class"], v["confidence_class"] = confidence_class(m), confidence_class(v)
    lst = {f"{s}": g.loc[g["daynight"] == "D", "lst_h"].median()
           for s, g in pd.concat([m, v]).groupby("source_satellite")}
    print("Median local solar time of DAYTIME detections (h): "
          + ", ".join(f"{s} {t:.1f}" for s, t in lst.items()))
    v["i4_saturated"] = v.get("bright_ti4", pd.Series(0.0, index=v.index)) >= VIIRS_I4_SATURATION_K
    print(f"VIIRS pixels at/above the I4 saturation level (~{VIIRS_I4_SATURATION_K:.0f} K): "
          f"{int(v['i4_saturated'].sum()):,} ({v['i4_saturated'].mean():.1%}); their FRP is a lower-quality estimate")

    # 1) matching: one MODIS satellite x one VIIRS satellite at a time (VIIRS satellites are never pooled)
    parts = []
    for ms in sorted(m["source_satellite"].unique()):
        for vs in sorted(v["source_satellite"].unique()):
            res = match_pair(m[m["source_satellite"] == ms], v[v["source_satellite"] == vs], MATCH_MAX_MIN)
            res["viirs_satellite"] = vs
            parts.append(res)
            hit = res["n_viirs"] > 0
            if hit.any():
                print(f"  {ms} x {vs}: {hit.sum():,}/{len(res):,} MODIS pixels contain >=1 VIIRS pixel within "
                      f"{MATCH_MAX_MIN:.0f} min (median |dt| {res.loc[hit, 'abs_dt_min'].median():.0f} min)")
            else:
                print(f"  {ms} x {vs}: no matches within {MATCH_MAX_MIN:.0f} min")
    pairs = pd.concat(parts, ignore_index=True)

    # 2) FRP scale: ONE k PER MODIS SATELLITE, each against the VIIRS satellite the calendar uses
    ref_sat = pick_ref_sat(v)
    matched_by_vs = pairs[pairs["n_viirs"] > 0].groupby("viirs_satellite").size()
    fit_vs = ref_sat if ref_sat in matched_by_vs.index else (matched_by_vs.idxmax() if len(matched_by_vs) else None)
    mod_sats = sorted(m["source_satellite"].unique())

    def _window_ok(w):
        for ms_ in mod_sats:
            sub_ = pairs[(pairs["source_satellite"] == ms_) & (pairs["viirs_satellite"] == fit_vs)] if fit_vs else pairs.iloc[0:0]
            u_ = _usable(sub_)
            u_ = u_[u_["abs_dt_min"] <= w]
            if len(u_) >= MIN_PAIRS_FIT and u_.groupby("cell_id").ngroups >= MIN_CELLS_FIT:
                return True
        return False
    resolve_k_window(_window_ok)
    fits = {}
    for ms in mod_sats:
        sub = pairs[(pairs["source_satellite"] == ms) & (pairs["viirs_satellite"] == fit_vs)] if fit_vs else pairs.iloc[0:0]
        fits[ms] = fit_scale(sub, ms) if len(sub) else (None, {"modis_satellite": ms, "k": None, "k_trust": "NOT_VALIDATED",
                                                               "k_trust_problems": ["no matches with VIIRS"], "n_pairs": 0})
        k_s, info_s = fits[ms]
        if k_s is None:
            print(f"  k {ms}: NOT VALIDATED ({'; '.join(info_s['k_trust_problems'])})")
        else:
            print(f"  k {ms}: {k_s:.3f} (95% CI {info_s['k_ci95'][0]:.3f}-{info_s['k_ci95'][1]:.3f}, "
                  f"new-year PI {info_s['k_prediction_interval95_new_year'][0]:.3f}-{info_s['k_prediction_interval95_new_year'][1]:.3f}, "
                  f"trust {info_s['k_trust']}, {info_s['n_pairs']:,} pixels)")
    scale, basis = decide_scales(fits)
    for s in mod_sats:
        if basis.get(s) == "none":
            print(f"WARNING: MODIS {s} has no validated k: it gets NO VIIRS-equivalent FRP (NaN). "
                  "Set PYRO_APPLY_UNVALIDATED_K=1 only if you accept an unverified assumption.")
        elif basis.get(s, "").startswith(("transferred", "own_low")):
            print(f"WARNING: MODIS {s} uses an UNVALIDATED scale ({basis[s]}); its values are labelled in scale_basis.")
    versions = sorted(m["version"].astype(str).unique()) if "version" in m.columns else []
    if len(versions) > 1:
        print(f"WARNING: the MODIS files mix product versions {versions}; compare k_by_modis_version in "
              "frp_mapping.json before treating the MODIS series as homogeneous.")
    for s in mod_sats:
        k_s, info_s = fits[s]
        sub = pairs[(pairs["source_satellite"] == s) & (pairs["viirs_satellite"] == fit_vs)] if fit_vs else pairs.iloc[0:0]
        info_s["k_by_modis_version"] = k_by_group(sub, "version") if k_s is not None else {}
        info_s["k_by_daynight"] = k_by_group(sub, "daynight") if k_s is not None else {}
        info_s["scale_basis"] = basis.get(s, "none")
        if k_s is not None:
            info_s["k_sensitivity"] = hv.k_sensitivity(_usable(sub), K_FIT_MAX_DT_MIN, MIN_PAIRS_FIT)
            info_s["k_ci_by_cluster_unit"] = hv.cluster_unit_sensitivity(
                _usable(sub)[lambda d: d["abs_dt_min"] <= K_FIT_MAX_DT_MIN], N_BOOT)

    # 3) calendar. Observation status is inferred PER SATELLITE; a day absent from a raw file is not "zero fire".
    m, v = apply_scale(m, scale, basis), apply_scale(v, scale, basis)
    observed, obs_basis = {}, {}
    observed["viirs"], obs_basis["viirs"] = observed_days("viirs", ref_sat)
    for s in mod_sats:
        observed[s], obs_basis[s] = observed_days("modis", s)
    clear = cloud_info = None
    if cloud_table is not None:
        clear, cloud_info = cloud_mask.clear_series(
            cloud_table, ref_sat, pd.date_range(t0.tz_localize(None).floor("D"), t1.tz_localize(None).floor("D"), freq="D"))
    cal, cells = build_calendar(m, v, ref_sat, t0, t1, observed, basis, clear)
    if cloud_info is not None:
        cloud_info = cloud_summary(cal, cloud_info, CLOUD_MIN_CLEAR, CLOUD_OBSCURE_NONZERO)
        print(cloud_mask.describe(cloud_info))

    primary = max(mod_sats, key=lambda s: fits[s][1].get("n_pairs", 0)) if mod_sats else None
    info = {
        "window_utc": [str(t0), str(t1)], "match_max_min": MATCH_MAX_MIN,
        "reference_viirs_satellite": ref_sat, "fit_viirs_satellite": fit_vs,
        "primary_modis_satellite": primary,
        "k_by_modis_satellite": {s: fits[s][0] for s in mod_sats},      # fitted on close-in-time matches; null = not validated
        "k_applied_to_calendar": applied_scale(m, scale),          # sum(frp_equiv) / sum(frp): equals k for a constant scale
        "scale_form": {s: ("conditional" if scale.get(s, {}).get("model") else ("constant" if s in scale else None))
                       for s in mod_sats},
        "scale_basis": basis,
        "scale_policy": {
            "rule": "a k is applied only to the satellite it was fitted on, from matches within "
                    f"{K_FIT_MAX_DT_MIN:g} min of VIIRS; a satellite without such matches gets NaN",
            "conditional_scale_enabled": K_MODEL,
            "apply_unvalidated_k": APPLY_UNVALIDATED_K,
            "k_window": dict(K_WINDOW_NOTE),
            "satellites_without_scale": [s for s in mod_sats if s not in scale]},
        "modis_satellites": {s: fits[s][1] for s in mod_sats},
        "modis_versions": versions,
        "observation_inference": {k_: (b if observed.get(k_) is not None else "unavailable (assumed observed)")
                                  for k_, b in obs_basis.items()},
        "harmonized_status_counts": {k_: int(c) for k_, c in cal["harmonized_status"].value_counts().items()},
        "daily_total_relationship": daily_total_relationship(cal),
        "detection_sensitivity": sensitivity_summary(cal),
        "caveat": "k is an aggregate statistical mapping fitted on footprints where BOTH sensors detected fire, "
                  "conditional on a MODIS detection; it is not a physical calibration, not a detection-probability "
                  "correction, and it is specific to the MODIS satellite it was fitted on.",
    }

    if cloud_info is not None:
        info["cloud_obscuration"] = cloud_info

    # 4) spatial-aggregation diagnostics (what the 0.5 deg grid does and does not change)
    if primary is not None and fits[primary][0] is not None:
        sub = pairs[(pairs["source_satellite"] == primary) & (pairs["viirs_satellite"] == fit_vs)]
        use_primary = _usable(sub)[lambda d: d["abs_dt_min"] <= K_FIT_MAX_DT_MIN]
        okdays = cal.loc[cal["day_complete"] & cal["viirs_observed"] & cal[f"{slug(primary)}_observed"], "date"]
        grid_df, grid_info = hv.grid_sensitivity(
            m[m["source_satellite"] == primary], v[v["source_satellite"] == ref_sat], pd.DatetimeIndex(okdays),
            use_primary, sorted(set(GRID_SENS_DEG) | {CELL_DEG}), CELL_DEG, N_BOOT, primary)
        grid_df.to_csv(PROCESSED_DIR / "grid_sensitivity.csv", index=False)
        info["grid_sensitivity"] = grid_info
        info["detection_sensitivity"]["like_for_like"] = grid_info["at_calendar_grid"]
        hv.plot_grid(grid_df, CELL_DEG, PROCESSED_DIR / "grid_sensitivity.png")
        hv.plot_k_by_year(fits[primary][1], primary, PROCESSED_DIR / "k_by_year.png")
        if (fits[primary][1].get("k_model") or {}).get("available"):
            hv.plot_k_model(fits[primary][1], primary, PROCESSED_DIR / "k_model.png")
        plot_matchup(sub, fits[primary][0], fits[primary][1], PROCESSED_DIR / "frp_matchup.png")

    (PROCESSED_DIR / "frp_mapping.json").write_text(hv.dumps(info))
    pairs.drop(columns=["date"], errors="ignore").to_csv(PROCESSED_DIR / "matched_pairs.csv", index=False)
    cal.to_csv(PROCESSED_DIR / "calendar_daily.csv", index=False)
    cells.to_csv(PROCESSED_DIR / "calendar_cells.csv", index=False)

    print("\nScale per MODIS satellite:", json.dumps(info["k_by_modis_satellite"]), "| basis:", json.dumps(basis))
    for s in mod_sats:
        i_ = fits[s][1]
        if i_.get("interpretation"):
            print(f"  {s}: {i_['interpretation']}")
        if fits[s][0] is not None:
            print(f"  {s}: {kmodel.describe(i_.get('k_model'))}")
        for p_ in i_.get("k_trust_problems", []):
            print(f"  {s}: - {p_}")
    print("Calendar rows by harmonized_status:", json.dumps(info["harmonized_status_counts"]))
    print(f"\nReference series: VIIRS {ref_sat}. First / last rows of the calendar "
          "(see harmonized_status; NaN = not observed, never zero):")
    cols = ["date", "aqua_detections", "viirs_detections", "viirs_frp_mw", "viirs_overpass_clusters",
            "harmonized_status", "harmonized_frp_mw_viirs_equiv", "modis_only_evidence"]
    show = cal[cols].copy() if len(cal) <= 10 else pd.concat([cal[cols].head(5), cal[cols].tail(5)])
    num = show.select_dtypes("number").columns   # round only numeric columns; "date" is datetime and .round() on it just warns
    show[num] = show[num].round(2)
    print(show.to_string(index=False))
    print_demo_summary(cal, fits, ref_sat)
    print("\nSaved matched_pairs.csv, calendar_daily.csv, calendar_cells.csv, frp_mapping.json, grid_sensitivity.csv, "
      "grid_sensitivity.png, k_by_year.png, k_model.png, frp_matchup.png in", PROCESSED_DIR)


def print_demo_summary(cal, fits, ref_sat):
    """Console summary for PYRO_SINGLE_DAY_DEMO=1 (one line per demo day, plus where MODIS enters)."""
    if not SINGLE_DAY_DEMO:
        return
    d = cal[cal["harmonized_status"] == "demo_partial_window"]
    print("\n=== SINGLE-DAY DEMO (PYRO_SINGLE_DAY_DEMO=1) ===")
    print("Input covers less than one UTC day: totals are LOWER BOUNDS for that day, not a validated record.")
    if d.empty:
        print("No partial day with reference-VIIRS data was found.")
        return
    for _, r in d.iterrows():
        eq = r.get("modis_frp_viirs_equiv_mw")
        eq_txt = "n/a (no k for any MODIS satellite)" if pd.isna(eq) else f"{eq:,.0f} MW"
        print(f"  {pd.Timestamp(r['date']).date()}: VIIRS {ref_sat} {r['viirs_frp_mw']:,.0f} MW "
              f"({int(r['viirs_detections']):,} detections); MODIS expressed on the VIIRS scale {eq_txt} "
              f"({int(r['modis_detections']) if pd.notna(r['modis_detections']) else 0:,} MODIS detections)")
    for s, (k, i) in fits.items():
        if k is not None:
            print(f"  {s}: k = {k:.3f} (trust {i['k_trust']}, {i['n_pairs']:,} matched pixels, window {i['fit_max_dt_min']:g} min)")
        else:
            print(f"  {s}: no k (it contributes only as a detection count)")


if __name__ == "__main__":
    main()
