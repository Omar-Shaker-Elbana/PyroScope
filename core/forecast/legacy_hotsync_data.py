"""LEGACY harmonizer — kept only so train_countries/predict_countries still run.

This is worldmap_forcase's original harmonization: ONE global kM/kV scale
factor fit over the whole overlap period, no trust grading, no per-satellite
split. core.src.harmonize (ported from PyroScope) replaces this with
per-satellite k, bootstrap CIs and trust grades.

TODO (validation step 2/5 of the plan): once core.validation wraps
core.src.harmonize into a per-country wide-table output shaped like
harmonize_by_country() below, retrain core.forecast's model on THAT instead
of this file, and delete this module. Until then, train_countries.py below
still calls this file's prepare()/harmonize_by_country() so the existing
model keeps working unchanged.

Original docstring:
Load NASA FIRMS CSVs and harmonize MODIS + VIIRS the way HotSync does.

Pipeline (same as the JS app):
  1. drop low-confidence detections (MODIS < 30, VIIRS 'l')
  2. snap detections to a ~2 km grid (0.02 deg) and a UTC day -> "active cell-days"
  3. during the sensors' overlap, the union of active cells is the reference;
     kM / kV scale MODIS-only or VIIRS-only counts to that reference
  4. FRP: per cell-day keep the peak 90-minute bin sum, convert VIIRS to MODIS
     scale with a robust median ratio, average where both sensors saw the cell
"""
from __future__ import annotations

import io
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

GRID_DEG = 0.02
FRP_BIN_MIN = 90
FRP_MATCH_MIN = 180


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_raw(paths) -> pd.DataFrame:
    """Read one or more FIRMS CSV exports into a single raw DataFrame."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"acq_time": str, "confidence": str}, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare(raw: pd.DataFrame, bbox=None) -> pd.DataFrame:
    """Filter, de-duplicate and grid raw FIRMS rows.

    bbox = (lat_min, lat_max, lon_min, lon_max) or None.
    """
    df = raw.copy()
    missing = {"latitude", "longitude", "acq_date"} - set(df.columns)
    if missing:
        raise ValueError(f"Not a FIRMS CSV, missing columns: {sorted(missing)}")

    keys = [c for c in ("latitude", "longitude", "acq_date", "acq_time") if c in df.columns]
    df = df.drop_duplicates(subset=keys)

    if "confidence" in df.columns:
        c = df["confidence"].astype(str).str.strip().str.lower()
        low = c.isin(["l", "low"]) | (pd.to_numeric(c, errors="coerce") < 30)
        df = df[~low]

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if bbox:
        la0, la1, lo0, lo1 = bbox
        df = df[df["latitude"].between(la0, la1) & df["longitude"].between(lo0, lo1)]

    if "acq_time" in df.columns:
        t = df["acq_time"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
        empty = t == ""
        t = t.str.zfill(4)
        hh = pd.to_numeric(t.str[:-2], errors="coerce")
        mm = pd.to_numeric(t.str[-2:], errors="coerce")
        minute = (hh * 60 + mm).where((hh < 24) & (mm < 60) & ~empty)
    else:
        minute = pd.Series(np.nan, index=df.index)

    frp = pd.to_numeric(df["frp"], errors="coerce").clip(lower=0) if "frp" in df.columns else np.nan
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["acq_date"]).dt.normalize(),
            "ci": np.floor(df["latitude"] / GRID_DEG).astype(np.int32),
            "cj": np.floor(df["longitude"] / GRID_DEG).astype(np.int32),
            "minute": minute,
            "frp": frp,
        }
    )
    out["bin"] = (out["minute"] // FRP_BIN_MIN).fillna(0).astype(np.int16)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Harmonization
# --------------------------------------------------------------------------- #
def cell_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, cell): peak 90-min-bin FRP sum and that bin's mean minute."""
    g = (
        d.groupby(["date", "ci", "cj", "bin"], sort=False)
        .agg(frp=("frp", "sum"), minute=("minute", "mean"))
        .reset_index()
    )
    top = g.sort_values("frp", ascending=False).drop_duplicates(["date", "ci", "cj"])
    return top.set_index(["date", "ci", "cj"])[["frp", "minute"]]


def harmonize(modis: pd.DataFrame, viirs: pd.DataFrame):
    """Return (daily DataFrame, info dict).

    daily columns: modis, viirs (cell-days per sensor), activity (harmonized
    active cell-days), frp (harmonized daily sum of cell-day peak FRP, MW).
    Days outside a sensor's file span are NaN for that sensor, not zero.
    """
    if modis.empty or viirs.empty:
        raise ValueError("Need detections from both MODIS and VIIRS after filtering.")

    m0, m1 = modis["date"].min(), modis["date"].max()
    v0, v1 = viirs["date"].min(), viirs["date"].max()
    days = pd.date_range(min(m0, v0), max(m1, v1), freq="D")
    has_m = (days >= m0) & (days <= m1)
    has_v = (days >= v0) & (days <= v1)

    tm, tv = cell_table(modis), cell_table(viirs)
    j = tm.join(tv, how="outer", lsuffix="_m", rsuffix="_v")

    def per_day(frame):
        return frame.groupby(level="date").size().reindex(days).fillna(0)

    mc, vc, uc = per_day(tm), per_day(tv), per_day(j)

    o0, o1 = max(m0, v0), min(m1, v1)
    ov = (days >= o0) & (days <= o1)
    k_m = uc[ov].sum() / mc[ov].sum() if ov.any() and mc[ov].sum() > 0 else np.nan
    k_v = uc[ov].sum() / vc[ov].sum() if ov.any() and vc[ov].sum() > 0 else np.nan

    activity = np.select(
        [has_m & has_v, has_m, has_v],
        [uc.values, mc.values * k_m, vc.values * k_v],
        default=np.nan,
    )

    # ---- FRP ----
    fm, fv = j["frp_m"], j["frp_v"]
    dt = (j["minute_m"] - j["minute_v"]).abs().fillna(0)  # unknown time -> comparable
    paired = fm.notna() & fv.notna() & (fm > 0) & (fv > 0) & (dt <= FRP_MATCH_MIN)
    frp_scale = float(np.exp(np.median(np.log(fm[paired] / fv[paired])))) if paired.any() else np.nan
    fv_s = fv * frp_scale
    both = fm.notna() & fv_s.notna() & (dt <= FRP_MATCH_MIN)
    comb = np.where(both, (fm + fv_s) / 2, np.where(fm.notna(), fm, fv_s))
    frp = (
        pd.Series(comb, index=j.index).groupby(level="date").sum()
        .reindex(days).fillna(0).where(has_m | has_v)
    )

    daily = pd.DataFrame(
        {
            "modis": mc.where(has_m),
            "viirs": vc.where(has_v),
            "activity": activity,
            "frp": frp.values,
        },
        index=days,
    )
    daily.index.name = "date"
    info = {
        "kM": float(k_m), "kV": float(k_v), "frp_scale": frp_scale,
        "modis_span": [str(m0.date()), str(m1.date())],
        "viirs_span": [str(v0.date()), str(v1.date())],
    }
    return daily, info


# --------------------------------------------------------------------------- #
# Optional: pull the latest near-real-time detections (needs a free MAP_KEY)
# --------------------------------------------------------------------------- #
def fetch_live(key: str, bbox, days: int = 40) -> dict[str, pd.DataFrame]:
    """Download the last `days` days of NRT MODIS + VIIRS(SNPP) detections.

    Returns {"M": raw_df, "V": raw_df}, ready to be concatenated with archive
    raw frames before calling prepare().
    """
    la0, la1, lo0, lo1 = bbox
    area = f"{lo0},{la0},{lo1},{la1}"  # west,south,east,north
    today = datetime.now(timezone.utc).date()
    out = {}
    for sensor, src in (("M", "MODIS_NRT"), ("V", "VIIRS_SNPP_NRT")):
        parts, cur = [], today - timedelta(days=days - 1)
        while cur <= today:
            n = min(10, (today - cur).days + 1)
            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{key}/{src}/{area}/{n}/{cur.isoformat()}"
            )
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8", "replace").lstrip()
            if not text.lower().startswith("latitude"):
                raise RuntimeError(f"FIRMS error for {src}: {text[:200]}")
            df = pd.read_csv(io.StringIO(text), dtype={"acq_time": str, "confidence": str})
            df.columns = [c.strip().lower() for c in df.columns]
            parts.append(df)
            cur += timedelta(days=n)
        out[sensor] = pd.concat(parts, ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# Country-level processing
# --------------------------------------------------------------------------- #
COUNTRY_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
               "master/geojson/ne_50m_admin_0_countries.geojson")


def load_countries(path: str | None = None, cache_dir: str | None = None):
    """Natural Earth country borders (downloaded once, then cached).

    Pass --countries to use your own GeoJSON/Shapefile (needs NAME, ADM0_A3,
    ISO_N3_EH columns, i.e. Natural Earth admin-0 format).
    """
    import geopandas as gpd
    from pathlib import Path

    if path is None:
        from core.src.config import CACHE_DIR

        p = Path(cache_dir or CACHE_DIR) / "ne_50m_admin_0_countries.geojson"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading country borders (Natural Earth, ~3 MB) ...")
            urllib.request.urlretrieve(COUNTRY_URL, p)
        path = str(p)
    g = gpd.read_file(path)
    iso = g["ISO_N3_EH"] if "ISO_N3_EH" in g.columns else g["ISO_N3"]
    out = gpd.GeoDataFrame(
        {
            "country": g["ADM0_A3"].astype(str),
            "name": g["NAME"].astype(str),
            "iso_n3": pd.to_numeric(iso, errors="coerce").fillna(-99).astype(int),
            "geometry": g.geometry,
        },
        crs="EPSG:4326",
    )
    return out


def _cell_key(ci, cj):
    return (np.asarray(ci, dtype=np.int64) + 5000) * 20000 + (np.asarray(cj, dtype=np.int64) + 10000)


def build_cell_country_map(frames, countries, max_snap_deg: float = 0.1) -> pd.Series:
    """Map every 0.02-degree grid cell that has a detection to a country code.

    Done on UNIQUE cells (not detections), so it stays fast for millions of rows.
    Cells just offshore are snapped to the nearest coast within `max_snap_deg`.
    """
    import geopandas as gpd
    import warnings

    cells = pd.concat([f[["ci", "cj"]] for f in frames]).drop_duplicates()
    pts = gpd.GeoDataFrame(
        {"key": _cell_key(cells["ci"], cells["cj"])},
        geometry=gpd.points_from_xy((cells["cj"].values + 0.5) * GRID_DEG, (cells["ci"].values + 0.5) * GRID_DEG),
        crs="EPSG:4326",
    )
    ref = countries[["country", "geometry"]]
    j = gpd.sjoin(pts, ref, how="left", predicate="within").drop_duplicates("key")
    miss = j["country"].isna()
    if miss.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            near = gpd.sjoin_nearest(pts[pts["key"].isin(j.loc[miss, "key"])], ref,
                                     how="left", max_distance=max_snap_deg).drop_duplicates("key")
        j = j.set_index("key")
        j.loc[near["key"].values, "country"] = near["country"].values
        j = j.reset_index()
    return j.dropna(subset=["country"]).set_index("key")["country"]


def attach_country(df: pd.DataFrame, cell_country: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["country"] = cell_country.reindex(_cell_key(df["ci"], df["cj"])).values
    return df.dropna(subset=["country"])


def harmonize_by_country(modis: pd.DataFrame, viirs: pd.DataFrame):
    """Per-country version of harmonize(). Inputs must carry a 'country' column.

    Returns (activity, frp, info): two wide DataFrames (date x country). The
    sensor scale factors (kM, kV, FRP scale) are estimated ONCE over all data,
    so small countries do not get noisy calibrations of their own.
    """
    m0, m1 = modis["date"].min(), modis["date"].max()
    v0, v1 = viirs["date"].min(), viirs["date"].max()
    days = pd.date_range(min(m0, v0), max(m1, v1), freq="D")
    has_m = ((days >= m0) & (days <= m1))[:, None]
    has_v = ((days >= v0) & (days <= v1))[:, None]

    tm, tv = cell_table(modis), cell_table(viirs)
    def _cc(frame):
        u = frame.drop_duplicates(["ci", "cj"])
        return pd.Series(u["country"].values, index=_cell_key(u["ci"], u["cj"]))

    cc = pd.concat([_cc(modis), _cc(viirs)])
    cc = cc[~cc.index.duplicated()]

    j = tm.join(tv, how="outer", lsuffix="_m", rsuffix="_v")

    def country_of(frame):
        ci = frame.index.get_level_values("ci"); cj = frame.index.get_level_values("cj")
        return cc.reindex(_cell_key(ci, cj)).values

    def wide_counts(frame):
        s = pd.Series(1, index=pd.MultiIndex.from_arrays(
            [frame.index.get_level_values("date"), country_of(frame)], names=["date", "country"]))
        return s.groupby(level=["date", "country"]).sum().unstack("country").reindex(days).fillna(0)

    mc, vc, uc = wide_counts(tm), wide_counts(tv), wide_counts(j)
    cols = sorted(set(mc.columns) | set(vc.columns) | set(uc.columns))
    mc, vc, uc = (x.reindex(columns=cols, fill_value=0) for x in (mc, vc, uc))

    o0, o1 = max(m0, v0), min(m1, v1)
    ov = ((days >= o0) & (days <= o1))
    k_m = uc.values[ov].sum() / max(mc.values[ov].sum(), 1)
    k_v = uc.values[ov].sum() / max(vc.values[ov].sum(), 1)
    act = np.where(has_m & has_v, uc.values,
          np.where(has_m, mc.values * k_m, np.where(has_v, vc.values * k_v, np.nan)))
    activity = pd.DataFrame(act, index=days, columns=cols)

    fm, fv = j["frp_m"], j["frp_v"]
    dt = (j["minute_m"] - j["minute_v"]).abs().fillna(0)
    paired = fm.notna() & fv.notna() & (fm > 0) & (fv > 0) & (dt <= FRP_MATCH_MIN)
    frp_scale = float(np.exp(np.median(np.log(fm[paired] / fv[paired])))) if paired.any() else np.nan
    fv_s = fv * frp_scale
    both = fm.notna() & fv_s.notna() & (dt <= FRP_MATCH_MIN)
    comb = np.where(both, (fm + fv_s) / 2, np.where(fm.notna(), fm, fv_s))
    s = pd.Series(comb, index=pd.MultiIndex.from_arrays(
        [j.index.get_level_values("date"), country_of(j)], names=["date", "country"]))
    fw = s.groupby(level=["date", "country"]).sum().unstack("country").reindex(days).reindex(columns=cols).fillna(0)
    frp = fw.where(np.broadcast_to(has_m | has_v, fw.shape))

    info = {"kM": float(k_m), "kV": float(k_v), "frp_scale": frp_scale,
            "modis_span": [str(m0.date()), str(m1.date())], "viirs_span": [str(v0.date()), str(v1.date())]}
    return activity, frp, info
