"""Disk-backed, day-batched harmonization for large FIRMS archives.

The standard harmonizer remains the reference implementation for small runs.  This
module stages standardized CSVs in a temporary SQLite database, matches one UTC day
at a time, spills matched pixels to temporary CSVs, and builds the calendar from
cell/day aggregates.  Pixel-level model/plot diagnostics use a deterministic bounded
sample; the headline k and its cell-cluster bootstrap interval use every qualifying
matched pixel via exact per-cell sufficient statistics.

SQLite is part of Python's standard library; no new runtime dependency is required.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import cloud_mask
from src import harmonize as h
from src import harmonize_validation as hv
from src import kmodel
from src.config import (
    PROCESSED_DIR, CHUNK_ROWS, MATCH_MAX_MIN, K_FIT_MAX_DT_MIN, MIN_PAIRS_FIT,
    K_TRUST_MIN_PAIRS, K_TAU_MAX, REF_SAT, START_DATE, END_DATE, CLOUD_FILE,
    CLOUD_MIN_CLEAR, CLOUD_OBSCURE_NONZERO, N_BOOT, K_MODEL, K_MODEL_BOOT,
    K_MODEL_FOLDS, K_MODEL_CLIP_Q, CELL_DEG, GRID_SENS_DEG, DAY_TOL_MIN,
    COV_CLUSTER_GAP_MIN,
)
from src.io_utils import TEXT_COLS
from src.kstats import ratio_ci, heterogeneity, prediction_interval

STREAM_SAMPLE_ROWS = int(os.getenv("PYRO_STREAM_SAMPLE_ROWS", "100000"))
STREAM_DIAGNOSTIC_ROWS = int(os.getenv("PYRO_STREAM_DIAGNOSTIC_ROWS", "150000"))
STREAM_CHUNK_ROWS = int(os.getenv("PYRO_STREAM_CHUNK_ROWS", str(CHUNK_ROWS)))
SQL_COLUMNS = [
    "sensor_family", "source_satellite", "instrument", "latitude", "longitude",
    "acq_datetime_utc", "frp", "scan", "track", "confidence_raw", "daynight",
    "brightness", "bright_ti4", "type", "version",
]


def _stage_csv(conn: sqlite3.Connection, table: str, path: Path) -> int:
    """Import one standardized CSV in bounded chunks; store UTC timestamps as integer ns."""
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Run `python -m src.preprocess` first.")
    n = 0
    usecols = lambda c: str(c).strip().lower() in set(SQL_COLUMNS)
    for chunk in pd.read_csv(path, chunksize=STREAM_CHUNK_ROWS, low_memory=False, dtype=TEXT_COLS,
                             usecols=usecols):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        for col in SQL_COLUMNS:
            if col not in chunk:
                chunk[col] = np.nan
        ts = pd.to_datetime(chunk["acq_datetime_utc"], errors="coerce", utc=True)
        ok = ts.notna()
        chunk = chunk.loc[ok, SQL_COLUMNS].copy()
        ts = ts.loc[ok]
        if chunk.empty:
            continue
        chunk["ts_ns"] = ts.astype("int64").to_numpy()
        chunk["date"] = ts.dt.strftime("%Y-%m-%d").to_numpy()
        for col in ("latitude", "longitude", "frp", "scan", "track", "brightness", "bright_ti4"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["source_satellite"] = chunk["source_satellite"].astype(str)
        chunk = chunk.drop(columns="acq_datetime_utc")
        chunk.to_sql(table, conn, if_exists="append", index=False, chunksize=10_000)
        n += len(chunk)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_time ON {table}(ts_ns)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_sat_time ON {table}(source_satellite, ts_ns)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table}(date)")
    conn.commit()
    return n


def _query(conn: sqlite3.Connection, table: str, lo_ns: int, hi_ns: int,
           satellite: str | None = None) -> pd.DataFrame:
    cols = ",".join([c for c in SQL_COLUMNS if c != "acq_datetime_utc"] + ["date", "ts_ns"])
    sql = f"SELECT {cols} FROM {table} WHERE ts_ns >= ? AND ts_ns < ?"
    args: list[object] = [int(lo_ns), int(hi_ns)]
    if satellite is not None:
        sql += " AND source_satellite = ?"
        args.append(str(satellite))
    sql += " ORDER BY ts_ns"
    df = pd.read_sql_query(sql, conn, params=args)
    if df.empty:
        df["acq_datetime_utc"] = pd.to_datetime([], utc=True)
        df["date"] = pd.to_datetime([])
        return df
    df["acq_datetime_utc"] = pd.to_datetime(df.pop("ts_ns"), unit="ns", utc=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _coverage(conn, table: str, label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    p = PROCESSED_DIR / f"{label}_coverage.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return pd.Timestamp(data["start_utc"]), pd.Timestamp(data["end_utc"])
        except (ValueError, KeyError, TypeError):
            pass
    row = conn.execute(f"SELECT MIN(ts_ns), MAX(ts_ns) FROM {table}").fetchone()
    if row[0] is None:
        raise SystemExit(f"No usable {label.upper()} rows in standardized file.")
    return (pd.to_datetime(row[0], unit="ns", utc=True),
            pd.to_datetime(row[1], unit="ns", utc=True))


def _sample_append(current: pd.DataFrame, incoming: pd.DataFrame, cap: int,
                   columns: list[str]) -> pd.DataFrame:
    """Keep the smallest stable 64-bit row hashes: deterministic, bounded reservoir."""
    if incoming.empty or cap <= 0:
        return current
    use_cols = [c for c in columns if c in incoming.columns]
    keys = pd.util.hash_pandas_object(incoming[use_cols], index=False).to_numpy(dtype=np.uint64)
    add = incoming.copy()
    add["_sample_hash"] = keys
    if current.empty:
        merged = add
    else:
        merged = pd.concat([current, add], ignore_index=True)
    if len(merged) > cap:
        idx = np.argpartition(merged["_sample_hash"].to_numpy(dtype=np.uint64), cap - 1)[:cap]
        merged = merged.iloc[idx]
    return merged.reset_index(drop=True)


def _aggregate_cells(df: pd.DataFrame, overpass_count: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = ["sensor_family", "source_satellite", "date", "cell_id", "cell_lat", "cell_lon"]
    agg = {
        "detection_count": ("frp", "size"),
        "frp": ("frp", "sum"),
        "frp_equiv": ("frp_equiv", "sum"),
        "frp_equiv_lo": ("frp_equiv_lo", "sum"),
        "frp_equiv_hi": ("frp_equiv_hi", "sum"),
        "n_eq": ("frp_equiv", "count"),
    }
    if "i4_saturated" in df:
        agg["i4_saturated_count"] = ("i4_saturated", "sum")
    out = df.groupby(group_cols, as_index=False).agg(**agg)
    out.loc[out["n_eq"] == 0, ["frp_equiv", "frp_equiv_lo", "frp_equiv_hi"]] = np.nan
    out = out.drop(columns="n_eq")
    if "sensor_family" in df and df["sensor_family"].astype(str).str.upper().eq("VIIRS").any():
        out["overpass_cluster_count"] = (np.nan if overpass_count is None else int(overpass_count))
    return out


def _exact_fit_summary(conn: sqlite3.Connection, pair_path: Path, sample: pd.DataFrame,
                       label: str) -> tuple[float | None, dict]:
    """Exact full-stream constant k / cell bootstrap; sample-based ancillary/model diagnostics.

    The year-by-cell sufficient statistics are stored in SQLite, not retained as a Python
    dictionary whose size grows with years x active cells.
    """
    if sample.empty:
        return None, {"modis_satellite": label, "k": None, "k_trust": "NOT_VALIDATED",
                      "k_trust_problems": ["no matched pixels with the selected VIIRS satellite"], "n_pairs": 0}

    conn.execute("DROP TABLE IF EXISTS fit_summary")
    conn.execute("CREATE TABLE fit_summary (year INTEGER, cell INTEGER, frp REAL, viirs REAL, pairs INTEGER)")
    n_any = n_close = 0
    days: set[str] = set()
    close_frp_sum = close_viirs_sum = 0.0
    # Each CSV chunk contributes compact year x cell sufficient statistics to disk.
    for chunk in pd.read_csv(pair_path, chunksize=STREAM_CHUNK_ROWS, low_memory=False,
                             parse_dates=["acq_datetime_utc"]):
        if chunk.empty:
            continue
        hit = pd.to_numeric(chunk.get("n_viirs", 0), errors="coerce").fillna(0).to_numpy() > 0
        n_any += int(hit.sum())
        f = pd.to_numeric(chunk["frp"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(chunk["viirs_frp_sum"], errors="coerce").to_numpy(float)
        dt = pd.to_numeric(chunk["abs_dt_min"], errors="coerce").to_numpy(float)
        good = hit & np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0) & np.isfinite(dt) & (dt <= K_FIT_MAX_DT_MIN)
        if not good.any():
            continue
        selected = chunk.loc[good]
        yy = selected["acq_datetime_utc"].dt.year.to_numpy(dtype=np.int32)
        dd = selected["acq_datetime_utc"].dt.strftime("%Y-%m-%d").to_numpy()
        cc = pd.to_numeric(selected["cell_id"], errors="coerce").to_numpy(dtype=np.int64)
        ff, vv = f[good], y[good]
        n_close += len(ff)
        close_frp_sum += float(ff.sum())
        close_viirs_sum += float(vv.sum())
        days.update(dd.tolist())
        compact = pd.DataFrame({"year": yy, "cell": cc, "frp": ff, "viirs": vv})
        compact = compact.groupby(["year", "cell"], as_index=False).agg(
            frp=("frp", "sum"), viirs=("viirs", "sum"), pairs=("frp", "size"))
        compact.to_sql("fit_summary", conn, if_exists="append", index=False, chunksize=10_000)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fit_summary_year_cell ON fit_summary(year, cell)")
    conn.commit()

    if n_close == 0 or close_frp_sum <= 0:
        conn.execute("DROP TABLE fit_summary")
        return None, {"modis_satellite": label, "k": None, "k_trust": "NOT_VALIDATED",
                      "k_trust_problems": ["no positive, close-in-time matched pixels"],
                      "n_pairs": 0, "n_pairs_any_time_gap": int(n_any)}

    full = pd.read_sql_query(
        "SELECT cell, SUM(frp) AS frp, SUM(viirs) AS viirs FROM fit_summary GROUP BY cell ORDER BY cell", conn)
    arr = full[["frp", "viirs"]].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    k, lo, hi, se_log = ratio_ci(arr, rng, N_BOOT)

    by_year = {}
    years = [int(r[0]) for r in conn.execute("SELECT DISTINCT year FROM fit_summary ORDER BY year")]
    for year in years:
        year_count = int(conn.execute("SELECT SUM(pairs) FROM fit_summary WHERE year=?", (year,)).fetchone()[0] or 0)
        ydf = pd.read_sql_query(
            "SELECT cell, SUM(frp) AS frp, SUM(viirs) AS viirs FROM fit_summary WHERE year=? GROUP BY cell ORDER BY cell",
            conn, params=(year,))
        ya = ydf[["frp", "viirs"]].to_numpy(dtype=float)
        if year_count < MIN_PAIRS_FIT or len(ya) < h.MIN_YEAR_CELLS:
            continue
        ky, ylo, yhi, yse = ratio_ci(ya, rng, N_BOOT)
        # Align the per-year clusters to the full-period cell order before leave-one-year-out subtraction.
        year_aligned = full[["cell"]].merge(ydf, on="cell", how="left")[["frp", "viirs"]].fillna(0.0).to_numpy(float)
        rest = arr - year_aligned
        ra = rest[(rest[:, 0] > 0) & (rest[:, 1] > 0)]
        loyo, loyo_z = None, None
        if len(ra) >= h.MIN_YEAR_CELLS and ra[:, 0].sum() > 0:
            kr, _, _, rse = ratio_ci(ra, rng, N_BOOT)
            se_sum = float(np.sqrt(yse ** 2 + rse ** 2))
            loyo, loyo_z = float(kr / ky), (float(np.log(kr / ky) / se_sum) if se_sum > 0 else None)
        by_year[str(year)] = {
            "n_pairs": year_count, "n_cells": int(len(ya)), "k": float(ky),
            "k_ci95": [float(ylo), float(yhi)], "se_log_k": float(yse),
            "k_from_other_years_pred_over_actual": loyo, "loyo_z": loyo_z,
        }

    conn.execute("DROP TABLE fit_summary")
    het = None
    if len(by_year) >= 3:
        het = heterogeneity([np.log(d["k"]) for d in by_year.values()], [d["se_log_k"] for d in by_year.values()])
    tau = None if het is None else het["tau"]
    pi = prediction_interval(k, se_log, tau, len(by_year))
    problems = []
    if n_close < K_TRUST_MIN_PAIRS:
        problems.append(f"{n_close} matched pixels (need >= {K_TRUST_MIN_PAIRS})")
    if len(days) < 30:
        problems.append(f"matches come from only {len(days)} distinct days (need >= 30)")
    if len(by_year) < 3:
        problems.append(f"only {len(by_year)} year(s) with >= {MIN_PAIRS_FIT} matched pixels (need >= 3)")
    ci_rel = float((hi - lo) / k)
    if ci_rel > 0.2:
        problems.append(f"bootstrap 95% CI is {ci_rel:.1%} of k wide (need <= 20%)")
    if het is not None:
        if het["p_Q"] < 0.05:
            problems.append(f"k differs between years by more than sampling error (Cochran Q p = {het['p_Q']:.3f})")
        if het["tau_ci95"][1] > K_TAU_MAX:
            problems.append(f"between-year variability of k cannot be bounded: tau = {het['tau']:.0%}, 95% upper limit "
                            f"{het['tau_ci95'][1]:.0%} (need <= {K_TAU_MAX:.0%})")
    zcrit = float(stats.norm.isf(0.025 / max(len(by_year), 1)))
    bad = [year for year, data in by_year.items() if data.get("loyo_z") is not None and abs(data["loyo_z"]) > zcrit]
    if bad:
        problems.append(f"k from the other years misses year(s) {bad} by more than sampling error "
                        f"(|z| > {zcrit:.2f}, Bonferroni over {len(by_year)} years)")
    grade = "HIGH" if not problems else ("LOW" if n_close < K_TRUST_MIN_PAIRS else "MEDIUM")

    # Ancillary diagnostics and the conditional k model are deliberately bounded to a stable sample.
    model_df = sample.drop(columns=["_sample_hash"], errors="ignore").copy()
    model_df["acq_datetime_utc"] = pd.to_datetime(model_df["acq_datetime_utc"], utc=True, errors="coerce")
    old_boot, old_model_boot = h.N_BOOT, h.K_MODEL_BOOT
    try:
        h.N_BOOT = min(int(old_boot), 400)
        h.K_MODEL_BOOT = min(int(old_model_boot), 500)
        _, sampled = h.fit_scale(model_df, label)
    finally:
        h.N_BOOT, h.K_MODEL_BOOT = old_boot, old_model_boot
    info = sampled
    info.update({
        "k": float(k), "k_ci95": [float(lo), float(hi)], "k_ci_includes_1": bool(lo <= 1.0 <= hi),
        "k_se_log": float(se_log), "k_prediction_interval95_new_year": pi,
        "n_pairs": int(n_close), "n_pairs_any_time_gap": int(n_any), "n_cells": int(len(arr)),
        "n_days_used": int(len(days)), "k_by_year": by_year, "k_year_heterogeneity": het,
        "k_year_spread_max_minus_min_over_median": (float((max(d["k"] for d in by_year.values()) -
             min(d["k"] for d in by_year.values())) / np.median([d["k"] for d in by_year.values()])) if len(by_year) >= 2 else None),
        "k_trust": grade, "k_trust_problems": problems, "k_ci_rel_width": ci_rel,
        "streaming": {"exact_k_and_cell_bootstrap": True, "sampled_diagnostics": True,
                      "sample_rows": int(len(sample)), "sample_cap": int(STREAM_SAMPLE_ROWS),
                      "n_close_pairs_full": int(n_close), "n_any_gap_pairs_full": int(n_any)},
    })
    if K_MODEL and info.get("k_model"):
        info["k_model"]["streaming_sample_bootstrap_note"] = "conditional-model bootstrap is bounded to at most 500 replicates in out-of-core mode"
    return float(k), info


def _append_csv(frame: pd.DataFrame, path: Path, first: bool) -> bool:
    if frame.empty:
        return first
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)
    return False


def _daily_sql(conn: sqlite3.Connection, table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
    clause = f" WHERE {where}" if where else ""
    return pd.read_sql_query(
        f"SELECT date, SUM(detection_count) AS det, SUM(frp) AS frp, "
        f"COUNT(DISTINCT cell_id) AS cells FROM {table}{clause} GROUP BY date ORDER BY date",
        conn, params=params, parse_dates=["date"])


def _calendar_from_sql(conn: sqlite3.Connection, ref_sat: str, m_sats: list[str], t0: pd.Timestamp,
                       t1: pd.Timestamp, observed: dict, scale_basis: dict, clear=None) -> pd.DataFrame:
    """Build the full daily calendar from SQL aggregates without loading all cell-days into RAM."""
    days = pd.date_range(t0.tz_localize(None).floor("D"), t1.tz_localize(None).floor("D"), freq="D")
    cal = pd.DataFrame(index=days)
    cal.index.name = "date"

    def obs_mask(key):
        values = observed.get(key)
        return np.ones(len(days), bool) if values is None else days.isin(list(values))

    complete = np.array([f for f in h.day_complete_flags(days, t0, t1).values()])
    v_obs = obs_mask("viirs")
    sat_obs = {sat: obs_mask(sat) for sat in m_sats}
    m_obs = np.logical_and.reduce(list(sat_obs.values())) if sat_obs else np.ones(len(days), bool)

    def block(table: str, name: str, mask: np.ndarray, where: str = "", params: tuple = ()):
        frame = _daily_sql(conn, table, where, params)
        frame = frame.set_index("date").reindex(days)
        frame[["det", "frp", "cells"]] = frame[["det", "frp", "cells"]].fillna(0.0)
        frame.loc[~mask, :] = np.nan
        cal[f"{name}_detections"] = frame["det"].round().astype("Int64")
        cal[f"{name}_frp_mw"] = frame["frp"]
        return frame

    mb = block("modis_cells", "modis", m_obs)
    cal["modis_active_cells"] = mb["cells"].round().astype("Int64")
    for sat in m_sats:
        block("modis_cells", h.slug(sat), sat_obs[sat], "source_satellite = ?", (str(sat),))
        cal[f"{h.slug(sat)}_observed"] = sat_obs[sat]

    eq_sats = [str(r[0]) for r in conn.execute(
        "SELECT DISTINCT source_satellite FROM modis_cells WHERE frp_equiv IS NOT NULL ORDER BY source_satellite")]
    if eq_sats:
        eq = pd.read_sql_query(
            "SELECT date, SUM(frp_equiv) AS frp_equiv, SUM(frp_equiv_lo) AS frp_equiv_lo, "
            "SUM(frp_equiv_hi) AS frp_equiv_hi FROM modis_cells WHERE frp_equiv IS NOT NULL GROUP BY date",
            conn, parse_dates=["date"]).set_index("date").reindex(days)
        eq_obs = np.logical_and.reduce([sat_obs[s] for s in eq_sats if s in sat_obs]) if eq_sats else np.ones(len(days), bool)
        for out_col, source_col in [
            ("modis_frp_viirs_equiv_mw", "frp_equiv"),
            ("modis_frp_viirs_equiv_lo_mw", "frp_equiv_lo"),
            ("modis_frp_viirs_equiv_hi_mw", "frp_equiv_hi"),
        ]:
            vals = eq[source_col].fillna(0.0)
            vals.iloc[~eq_obs] = np.nan
            cal[out_col] = vals
    else:
        for col in ("modis_frp_viirs_equiv_mw", "modis_frp_viirs_equiv_lo_mw", "modis_frp_viirs_equiv_hi_mw"):
            cal[col] = np.nan
    cal["modis_frp_viirs_equiv_satellites"] = ",".join(eq_sats) if eq_sats else "none"

    vb = block("viirs_cells", "viirs", v_obs, "source_satellite = ?", (ref_sat,))
    cal["viirs_active_cells"] = vb["cells"].round().astype("Int64")
    sat = pd.read_sql_query(
        "SELECT date, SUM(i4_saturated_count) AS saturated, SUM(detection_count) AS detections "
        "FROM viirs_cells WHERE source_satellite = ? GROUP BY date", conn,
        params=(ref_sat,), parse_dates=["date"]).set_index("date").reindex(days)
    sat_share = (sat["saturated"] / sat["detections"]).replace([np.inf, -np.inf], np.nan)
    sat_share[~v_obs] = np.nan
    cal["viirs_i4_saturated_share"] = sat_share
    cluster = pd.read_sql_query(
        "SELECT date, MAX(overpass_cluster_count) AS clusters FROM viirs_cells "
        "WHERE source_satellite = ? GROUP BY date", conn, params=(ref_sat,), parse_dates=["date"]
    ).set_index("date")["clusters"].reindex(days).fillna(0.0)
    cluster[~v_obs] = np.nan
    cal["viirs_overpass_clusters"] = cluster
    low, expected = h.flag_low_coverage(vb["det"], cluster)
    cal["viirs_expected_detections"] = expected
    cal["viirs_low_coverage"] = low.to_numpy()

    both = pd.read_sql_query(
        "SELECT m.date, COUNT(*) AS cells FROM "
        "(SELECT DISTINCT date, cell_id FROM modis_cells) AS m "
        "INNER JOIN (SELECT DISTINCT date, cell_id FROM viirs_cells WHERE source_satellite = ?) AS v "
        "ON m.date = v.date AND m.cell_id = v.cell_id GROUP BY m.date",
        conn, params=(ref_sat,), parse_dates=["date"]).set_index("date")["cells"].reindex(days).fillna(0.0)
    both[~(v_obs & m_obs)] = np.nan
    cal["cells_active_in_both"] = both.round().astype("Int64")
    mac = mb["cells"]
    cal["modis_cells_also_viirs"] = np.where(mac > 0, both / mac.where(mac > 0), np.nan)

    det = cal["viirs_detections"].fillna(-1).to_numpy()
    obscured = np.zeros(len(days), bool)
    if clear is not None:
        c = pd.Series(clear).reindex(days).to_numpy(float)
        limited = np.isfinite(c) & (c < h.CLOUD_MIN_CLEAR)
        obscured = limited & ((det == 0) | bool(h.CLOUD_OBSCURE_NONZERO))
        cal["viirs_clear_fraction"] = c
        cal["viirs_cloud_limited"] = limited & v_obs & complete
    status = np.select(
        [~complete, ~v_obs, obscured, cal["viirs_low_coverage"].to_numpy(), det == 0],
        ["partial_day", "viirs_not_observed", "observed_obscured", "measured_low_coverage", "measured_zero"],
        default="measured")
    cal["day_complete"] = complete
    cal["viirs_observed"] = v_obs
    cal["modis_observed"] = m_obs
    cal["day_observed"] = v_obs & m_obs
    cal["harmonized_status"] = status
    is_meas = np.isin(status, h.STATUS_MEASURED)
    cal["harmonized_is_measured"] = is_meas
    cal["harmonized_frp_mw_viirs_equiv"] = np.where(is_meas, cal["viirs_frp_mw"], np.nan)
    cal["harmonized_source"] = np.where(is_meas, f"VIIRS {ref_sat}", "none")
    eq_mw = cal["modis_frp_viirs_equiv_mw"].to_numpy()
    cal["modis_only_evidence"] = (status == "viirs_not_observed") & (np.nan_to_num(eq_mw, nan=0.0) > 0)
    cal["aoi"] = "global" if h.BBOX is None else ",".join(str(x) for x in h.BBOX)
    if h.AOI_FILE is not None:
        cal["aoi"] += f" | mask {h.AOI_FILE.name}"
    if h.EXCLUDE_STATIC:
        cal["aoi"] += " | non-static only (type 0)"
    return cal.reset_index()


def _write_calendar_cells(conn: sqlite3.Connection, path: Path, scale_basis: dict) -> None:
    """Export the disk-backed cell/day table in bounded chunks, preserving the legacy CSV schema."""
    query = (
        "SELECT sensor_family, source_satellite, date, cell_id, cell_lat, cell_lon, "
        "detection_count AS detections, frp AS frp_mw, frp_equiv AS frp_viirs_equiv_mw "
        "FROM modis_cells UNION ALL "
        "SELECT sensor_family, source_satellite, date, cell_id, cell_lat, cell_lon, "
        "detection_count AS detections, frp AS frp_mw, frp_equiv AS frp_viirs_equiv_mw "
        "FROM viirs_cells ORDER BY date, sensor_family, source_satellite, cell_id"
    )
    first = True
    for chunk in pd.read_sql_query(query, conn, chunksize=500_000, parse_dates=["date"]):
        chunk["cell_area_km2"] = h.cell_area_km2(chunk["cell_lat"].to_numpy(float), h.CELL_DEG)
        chunk["scale_basis"] = np.where(
            chunk["sensor_family"].eq("VIIRS"), "reference",
            chunk["source_satellite"].map(scale_basis).fillna("none"))
        chunk.to_csv(path, mode="w" if first else "a", header=first, index=False)
        first = False
    if first:
        pd.DataFrame(columns=["sensor_family", "source_satellite", "date", "cell_id", "cell_lat", "cell_lon",
                              "detections", "frp_mw", "frp_viirs_equiv_mw", "cell_area_km2", "scale_basis"]
                     ).to_csv(path, index=False)


def run() -> None:
    cloud_table = cloud_mask.read_table(CLOUD_FILE) if CLOUD_FILE is not None else None
    modis_path = PROCESSED_DIR / "modis_standardized.csv"
    viirs_path = PROCESSED_DIR / "viirs_standardized.csv"
    with tempfile.TemporaryDirectory(prefix="pyroscope_stream_", dir=PROCESSED_DIR) as td:
        temp = Path(td)
        db_path = temp / "stage.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=FILE")
            print("Out-of-core harmonization: staging standardized CSVs in temporary SQLite tables...")
            nm = _stage_csv(conn, "modis", modis_path)
            nv = _stage_csv(conn, "viirs", viirs_path)
            if not nm or not nv:
                raise SystemExit("Both MODIS and VIIRS standardized files must contain usable rows.")
            (ms, me), (vs, ve) = (_coverage(conn, "modis", "modis"), _coverage(conn, "viirs", "viirs"))
            t0, t1 = max(ms, vs), min(me, ve)
            if START_DATE:
                t0 = max(t0, pd.Timestamp(START_DATE, tz="UTC"))
            if END_DATE:
                t1 = min(t1, pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1))
            if t0 >= t1:
                raise SystemExit("MODIS and VIIRS files have no time overlap - nothing to harmonize.")
            # The in-memory reference uses Series.between(t0, t1), inclusive on both
            # ends. Add one nanosecond to the SQL exclusive upper bound so a detection
            # exactly at the common-window endpoint is not silently dropped.
            lo_ns, hi_ns = int(t0.value), int(t1.value) + 1
            m_sats = [r[0] for r in conn.execute("SELECT DISTINCT source_satellite FROM modis ORDER BY source_satellite")]
            v_stats = pd.read_sql_query(
                "SELECT source_satellite, COUNT(DISTINCT date) AS days, COUNT(*) AS n FROM viirs "
                "WHERE ts_ns >= ? AND ts_ns < ? GROUP BY source_satellite ORDER BY days, n",
                conn, params=(lo_ns, hi_ns))
            if v_stats.empty:
                raise SystemExit("No VIIRS rows in the common time window.")
            ref_sat = REF_SAT or str(v_stats.iloc[-1]["source_satellite"])
            if ref_sat not in set(v_stats["source_satellite"].astype(str)):
                raise SystemExit(f"PYRO_REF_SAT={ref_sat!r} is not in the VIIRS data: {sorted(v_stats.source_satellite.astype(str))}")
            v_sats = sorted(v_stats["source_satellite"].astype(str).unique())
            dates = [pd.Timestamp(r[0]) for r in conn.execute(
                "SELECT DISTINCT date FROM modis WHERE ts_ns >= ? AND ts_ns < ? UNION "
                "SELECT DISTINCT date FROM viirs WHERE ts_ns >= ? AND ts_ns < ? ORDER BY 1",
                (lo_ns, hi_ns, lo_ns, hi_ns))]
            print(f"Streaming {nm:,} MODIS / {nv:,} VIIRS staged rows across {len(dates):,} UTC dates; "
                  f"common window {t0} -> {t1}; reference VIIRS {ref_sat}.")

            # First pass: match one UTC day at a time, with VIIRS time padding for overpass gaps.
            # Never leave a stale match table behind if this run finds no matchable dates.
            pair_files: dict[tuple[str, str], Path] = {}
            pair_first: dict[tuple[str, str], bool] = {}
            matched_counts = {s: 0 for s in v_sats}
            margin_ns = int(MATCH_MAX_MIN * 60 * 1e9)
            for day_i, day in enumerate(dates, 1):
                day0 = pd.Timestamp(day, tz="UTC")
                day1 = day0 + pd.Timedelta(days=1)
                vlo, vhi = max(lo_ns, day0.value - margin_ns), min(hi_ns, day1.value + margin_ns)
                # Match one MODIS / VIIRS satellite pair at a time. This bounds working RAM by
                # one instrument-day and its nearby VIIRS time window instead of all sensors together.
                for msat in m_sats:
                    mm = _query(conn, "modis", max(lo_ns, day0.value), min(hi_ns, day1.value), str(msat))
                    if mm.empty:
                        continue
                    mm = h.add_features(mm)
                    mm["confidence_class"] = h.confidence_class(mm)
                    for vsat in v_sats:
                        vv = _query(conn, "viirs", vlo, vhi, str(vsat))
                        if vv.empty:
                            continue
                        vv = h.add_features(vv)
                        res = h.match_pair(mm, vv, MATCH_MAX_MIN)
                        res["viirs_satellite"] = vsat
                        matched_counts[vsat] += int((res["n_viirs"] > 0).sum())
                        key = (str(msat), str(vsat))
                        path = pair_files.setdefault(key, temp / f"pairs_{h.slug(msat)}_{h.slug(vsat)}.csv")
                        first = pair_first.setdefault(key, True)
                        res = res.drop(columns=["date"], errors="ignore")
                        pair_first[key] = _append_csv(res, path, first)
                if day_i % 100 == 0:
                    print(f"  matched {day_i:,}/{len(dates):,} dates...")

            fit_vs = ref_sat if matched_counts.get(ref_sat, 0) > 0 else (
                max(matched_counts, key=matched_counts.get) if max(matched_counts.values(), default=0) > 0 else None)
            # Preserve matched_pairs.csv as a streamed full output (not just the fit sample).
            matched_out = temp / "matched_pairs.csv"
            first_out = True
            for key in sorted(pair_files):
                pth = pair_files[key]
                if not pth.exists():
                    continue
                with pth.open("r", encoding="utf-8", newline="") as src, matched_out.open(
                        "a" if not first_out else "w", encoding="utf-8", newline="") as dst:
                    if first_out:
                        shutil.copyfileobj(src, dst)
                        first_out = False
                    else:
                        next(src, None)
                        shutil.copyfileobj(src, dst)

            # Stream one bounded sample per MODIS satellite; k and the cell bootstrap are recomputed exactly from disk.
            fits: dict[str, tuple[float | None, dict]] = {}
            pair_samples = {}
            for msat in m_sats:
                pth = pair_files.get((str(msat), str(fit_vs))) if fit_vs else None
                if pth is None or not pth.exists():
                    fits[str(msat)] = (None, {"modis_satellite": str(msat), "k": None, "k_trust": "NOT_VALIDATED",
                                             "k_trust_problems": ["no matches with selected VIIRS satellite"], "n_pairs": 0})
                    continue
                # Deterministic bounded sample for ancillary diagnostics and conditional model fitting.
                sample = pd.DataFrame()
                sample_keys = ["source_satellite", "acq_datetime_utc", "latitude", "longitude", "frp", "viirs_frp_sum"]
                # Pair CSV includes only standard harmonizer columns; hash-based top-k is stable across chunk sizes.
                for chunk in pd.read_csv(pth, chunksize=STREAM_CHUNK_ROWS, low_memory=False,
                                         parse_dates=["acq_datetime_utc"]):
                    hit = chunk[pd.to_numeric(chunk["n_viirs"], errors="coerce").fillna(0) > 0]
                    sample = _sample_append(sample, hit, STREAM_SAMPLE_ROWS, sample_keys)
                if not sample.empty:
                    sample = sample.drop(columns=["_sample_hash"], errors="ignore")
                k_s, info_s = _exact_fit_summary(conn, pth, sample, str(msat))
                if k_s is not None and not sample.empty:
                    sample_use = h._usable(sample)
                    sample_close = sample_use[sample_use["abs_dt_min"] <= K_FIT_MAX_DT_MIN]
                    info_s["k_by_modis_version"] = h.k_by_group(sample, "version")
                    info_s["k_by_daynight"] = h.k_by_group(sample, "daynight")
                    info_s["k_sensitivity"] = hv.k_sensitivity(sample_use, K_FIT_MAX_DT_MIN, MIN_PAIRS_FIT)
                    # Bootstrap cluster-unit diagnostics on a further bounded subset: these are
                    # descriptive only, while the main cell-cluster CI above uses every match.
                    cluster_sample = (sample_close if len(sample_close) <= 10_000 else
                                      sample_close.sample(n=10_000, random_state=0))
                    if len(cluster_sample) >= 3:
                        info_s["k_ci_by_cluster_unit"] = hv.cluster_unit_sensitivity(
                            cluster_sample, min(int(N_BOOT), 200))
                    info_s["streaming"]["ancillary_diagnostics_basis"] = "deterministic bounded pixel sample"
                    info_s["streaming"]["cluster_unit_diagnostic_sample_rows"] = int(len(cluster_sample))
                fits[str(msat)] = (k_s, info_s)
                pair_samples[str(msat)] = sample
                if k_s is None:
                    print(f"  k {msat}: NOT VALIDATED ({'; '.join(info_s.get('k_trust_problems', []))})")
                else:
                    print(f"  k {msat}: {k_s:.3f} (exact full-stream 95% CI {info_s['k_ci95'][0]:.3f}-"
                          f"{info_s['k_ci95'][1]:.3f}, trust {info_s['k_trust']}, {info_s['n_pairs']:,} close matches; "
                          f"diagnostics sample {info_s['streaming']['sample_rows']:,})")

            scale, basis = h.decide_scales(fits)
            for s in m_sats:
                fits[str(s)][1]["scale_basis"] = basis.get(str(s), "none")
                if basis.get(s) == "none":
                    print(f"WARNING: MODIS {s} has no validated k and gets no VIIRS-equivalent FRP.")
            versions = [str(r[0]) for r in conn.execute(
                "SELECT DISTINCT version FROM modis WHERE version IS NOT NULL ORDER BY version")]

            # Second pass: aggregate one instrument/day and persist cell/day rows in SQLite.
            # Neither raw detections nor the multi-year cell/day table is retained in RAM.
            conn.execute("DROP TABLE IF EXISTS modis_cells")
            conn.execute("DROP TABLE IF EXISTS viirs_cells")
            conn.commit()
            diag_m, diag_v = pd.DataFrame(), pd.DataFrame()
            diag_keys = ["source_satellite", "acq_datetime_utc", "latitude", "longitude", "frp"]
            for i, day in enumerate(dates, 1):
                day0 = pd.Timestamp(day, tz="UTC")
                day1 = day0 + pd.Timedelta(days=1)
                primary = max(m_sats, key=lambda s: fits[str(s)][1].get("n_pairs", 0)) if m_sats else None
                for msat in m_sats:
                    md = _query(conn, "modis", max(lo_ns, day0.value), min(hi_ns, day1.value), str(msat))
                    if md.empty:
                        continue
                    md = h.add_features(md)
                    md["confidence_class"] = h.confidence_class(md)
                    md = h.apply_scale(md, scale, basis)
                    _aggregate_cells(md).to_sql("modis_cells", conn, if_exists="append", index=False, chunksize=10_000)
                    if str(msat) == str(primary):
                        diag_m = _sample_append(diag_m, md, STREAM_DIAGNOSTIC_ROWS, diag_keys)
                for vsat in v_sats:
                    vd = _query(conn, "viirs", max(lo_ns, day0.value), min(hi_ns, day1.value), str(vsat))
                    if vd.empty:
                        continue
                    vd = h.add_features(vd)
                    vd = h.apply_scale(vd, scale, basis)
                    vd["i4_saturated"] = vd.get("bright_ti4", pd.Series(0.0, index=vd.index)) >= h.VIIRS_I4_SATURATION_K
                    cluster_n = None
                    if str(vsat) == str(ref_sat):
                        clus = h.overpass_clusters(vd)
                        cluster_n = int(clus.iloc[0]) if len(clus) else 0
                        diag_v = _sample_append(diag_v, vd, STREAM_DIAGNOSTIC_ROWS, diag_keys)
                    _aggregate_cells(vd, cluster_n).to_sql("viirs_cells", conn, if_exists="append", index=False, chunksize=10_000)
                if i % 100 == 0:
                    print(f"  aggregated {i:,}/{len(dates):,} dates...")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_modis_cells_date_cell ON modis_cells(date, cell_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_modis_cells_sat_date ON modis_cells(source_satellite, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viirs_cells_date_cell ON viirs_cells(date, cell_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viirs_cells_sat_date ON viirs_cells(source_satellite, date)")
            conn.commit()
            observed, obs_basis = {}, {}
            observed["viirs"], obs_basis["viirs"] = h.observed_days("viirs", ref_sat)
            for sat in m_sats:
                observed[sat], obs_basis[sat] = h.observed_days("modis", sat)
            clear = cloud_info = None
            if cloud_table is not None:
                clear, cloud_info = cloud_mask.clear_series(
                    cloud_table, ref_sat, pd.date_range(t0.tz_localize(None).floor("D"),
                                                        t1.tz_localize(None).floor("D"), freq="D"))
            cal = _calendar_from_sql(conn, ref_sat, [str(x) for x in m_sats], t0, t1, observed, basis, clear)
            cells_stage = temp / "calendar_cells.csv"
            _write_calendar_cells(conn, cells_stage, basis)
            if cloud_info is not None:
                cloud_info = h.cloud_summary(cal, cloud_info, CLOUD_MIN_CLEAR, CLOUD_OBSCURE_NONZERO)
                print(cloud_mask.describe(cloud_info))

            # Matchup summary and grid sensitivity use a deterministic bounded pixel sample only.
            primary = max(m_sats, key=lambda s: fits[str(s)][1].get("n_pairs", 0)) if m_sats else None
            info = {
                "window_utc": [str(t0), str(t1)], "match_max_min": MATCH_MAX_MIN,
                "reference_viirs_satellite": ref_sat, "fit_viirs_satellite": fit_vs,
                "primary_modis_satellite": primary,
                "k_by_modis_satellite": {str(s): fits[str(s)][0] for s in m_sats},
                "k_applied_to_calendar": {
                    str(s): (None if str(s) not in scale else
                             float((lambda row: (row[0] or 0.0) / max(row[1] or 0.0, 1e-12))(
                                 conn.execute("SELECT SUM(frp_equiv), SUM(frp) FROM modis_cells WHERE source_satellite = ?",
                                              (str(s),)).fetchone())))
                    for s in m_sats},
                "scale_form": {str(s): ("conditional" if scale.get(str(s), {}).get("model") else
                                          ("constant" if str(s) in scale else None)) for s in m_sats},
                "scale_basis": basis,
                "scale_policy": {"rule": "a k is applied only to its own MODIS satellite, using close-in-time matches",
                                 "conditional_scale_enabled": K_MODEL, "apply_unvalidated_k": h.APPLY_UNVALIDATED_K,
                                 "satellites_without_scale": [str(s) for s in m_sats if str(s) not in scale]},
                "modis_satellites": {str(s): fits[str(s)][1] for s in m_sats},
                "modis_versions": versions,
                "observation_inference": {key: (val if observed.get(key) is not None else "unavailable (assumed observed)")
                                           for key, val in obs_basis.items()},
                "harmonized_status_counts": {str(k): int(v) for k, v in cal["harmonized_status"].value_counts().items()},
                "daily_total_relationship": h.daily_total_relationship(cal),
                "detection_sensitivity": h.sensitivity_summary(cal),
                "out_of_core": {"enabled": True,
                                 "storage": "temporary SQLite staging and cell/day aggregate tables; CSV exports are chunked",
                                 "stream_chunk_rows": STREAM_CHUNK_ROWS, "fit_sample_cap": STREAM_SAMPLE_ROWS,
                                 "diagnostic_sample_cap": STREAM_DIAGNOSTIC_ROWS,
                                 "cluster_unit_diagnostic_sample_cap": 10000,
                                 "headline_k_and_cell_bootstrap_use_all_qualifying_pairs": True,
                                 "year_cell_sufficient_statistics_spilled_to_disk": True,
                                 "calendar_cell_table_written_in_chunks": True,
                                 "conditional_k_and_pixel_level_diagnostics_use_bounded_sample": True},
                "caveat": "k is an aggregate statistical mapping fitted on footprints where BOTH sensors detected fire; "
                          "the streaming mode computes the headline k and cell-cluster CI from all qualifying matches, "
                          "while conditional-model selection and pixel-level diagnostics use a deterministic bounded sample.",
            }
            if cloud_info is not None:
                info["cloud_obscuration"] = cloud_info

            if primary is not None and fits[str(primary)][0] is not None and not diag_m.empty and not diag_v.empty:
                sample_use = pair_samples.get(str(primary), pd.DataFrame())
                sample_use = h._usable(sample_use)
                sample_use = sample_use[sample_use["abs_dt_min"] <= K_FIT_MAX_DT_MIN]
                okdays = cal.loc[cal["day_complete"] & cal["viirs_observed"] & cal[f"{h.slug(primary)}_observed"], "date"]
                if len(sample_use) >= 3:
                    grid_df, grid_info = hv.grid_sensitivity(
                        diag_m.drop(columns=["_sample_hash"], errors="ignore"),
                        diag_v.drop(columns=["_sample_hash"], errors="ignore"),
                        pd.DatetimeIndex(okdays), sample_use,
                        sorted(set(GRID_SENS_DEG) | {CELL_DEG}), CELL_DEG, N_BOOT, str(primary))
                    grid_info["sampling"] = "deterministic bounded pixel sample; descriptive only in out-of-core mode"
                    grid_df.to_csv(PROCESSED_DIR / "grid_sensitivity.csv", index=False)
                    info["grid_sensitivity"] = grid_info
                    info["detection_sensitivity"]["like_for_like"] = grid_info["at_calendar_grid"]
                    hv.plot_grid(grid_df, CELL_DEG, PROCESSED_DIR / "grid_sensitivity.png")
                p_sample = pair_samples[str(primary)]
                info["modis_satellites"][str(primary)]["streaming_sample_note"] = (
                    "k, confidence interval, pair counts and year-cell summaries use the full stream; k-model and pixel-level "
                    "diagnostics use the deterministic bounded sample.")
                h.plot_matchup(p_sample, fits[str(primary)][0], fits[str(primary)][1], PROCESSED_DIR / "frp_matchup.png")
                hv.plot_k_by_year(fits[str(primary)][1], str(primary), PROCESSED_DIR / "k_by_year.png")
                for sat in m_sats:
                    inf = fits[str(sat)][1]
                    if inf.get("k_model", {}).get("available"):
                        hv.plot_k_model(inf, str(sat), PROCESSED_DIR / f"k_model_{h.slug(sat)}.png")

            # Stage every main artifact on the same filesystem, then publish only after the
            # full computation succeeds. Individual replaces are atomic on the target filesystem.
            mapping_stage = temp / "frp_mapping.json"
            daily_stage = temp / "calendar_daily.csv"
            if first_out:
                matched_out.write_text("", encoding="utf-8")
            mapping_stage.write_text(hv.dumps(info), encoding="utf-8")
            cal.to_csv(daily_stage, index=False)
            matched_out.replace(PROCESSED_DIR / "matched_pairs.csv")
            daily_stage.replace(PROCESSED_DIR / "calendar_daily.csv")
            cells_stage.replace(PROCESSED_DIR / "calendar_cells.csv")
            mapping_stage.replace(PROCESSED_DIR / "frp_mapping.json")
            print("Out-of-core harmonization complete. Wrote matched_pairs.csv, calendar_daily.csv, calendar_cells.csv, "
                  "frp_mapping.json and available diagnostics to", PROCESSED_DIR)
            print("Calendar rows by harmonized_status:", json.dumps(info["harmonized_status_counts"]))
        finally:
            conn.close()
