import json

import pandas as pd
import numpy as np

from src.config import (
    RAW_DIR, PROCESSED_DIR, START_DATE, END_DATE, BBOX, CHUNK_ROWS, DEDUPE, AOI_FILE, EXCLUDE_STATIC
)
from src.io_utils import find_files, iter_firms_csv

SATELLITE_NAMES = {
    "T": "Terra", "A": "Aqua",
    "N": "S-NPP", "N20": "NOAA-20", "N21": "NOAA-21",
}

COMMON_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "frp",
    "daynight", "brightness", "scan", "track", "version", "type"
]

_AOI_RINGS = None  # [(exterior Path, [hole Paths])], read once from PYRO_AOI_FILE


def _load_aoi():
    """Read the PYRO_AOI_FILE GeoJSON into matplotlib Paths (no extra dependency)."""
    global _AOI_RINGS
    if _AOI_RINGS is None:
        from matplotlib.path import Path as MplPath
        gj = json.loads(AOI_FILE.read_text(encoding="utf-8"))
        if gj.get("type") == "FeatureCollection":
            geoms = [f["geometry"] for f in gj["features"]]
        else:
            geoms = [gj.get("geometry", gj)]
        _AOI_RINGS = []
        for g in geoms:
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                _AOI_RINGS.append((MplPath(np.asarray(poly[0], float)[:, :2]),
                                   [MplPath(np.asarray(h, float)[:, :2]) for h in poly[1:]]))
    return _AOI_RINGS


def inside_aoi(lat, lon) -> np.ndarray:
    """Boolean mask: point lies inside the AOI polygon(s), holes excluded."""
    pts = np.c_[np.asarray(lon, float), np.asarray(lat, float)]
    ok = np.zeros(len(pts), dtype=bool)
    for outer, holes in _load_aoi():
        inside = outer.contains_points(pts)
        for h in holes:
            inside &= ~h.contains_points(pts)
        ok |= inside
    return ok


def standardize(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Keep original brightness fields but also expose one generic field.
    brightness_candidates = [
        "brightness", "bright_ti4", "bright_t31", "bright_ti5"
    ]
    available_brightness = [c for c in brightness_candidates if c in df.columns]
    if "brightness" not in df.columns and available_brightness:
        df["brightness"] = df[available_brightness[0]]

    required = ["latitude", "longitude", "acq_date", "acq_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{sensor_name}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Preserve the raw confidence value; it is not comparable across sensors.
    df["confidence_raw"] = df["confidence"].astype("string")

    # FIRMS acquisition time is HHMM in UTC. Normalize values such as 5 -> 0005.
    time_text = (
        df["acq_time"].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(4)
    )
    date_text = pd.to_datetime(df["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["acq_datetime_utc"] = pd.to_datetime(
        date_text + " " + time_text,
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True
    )

    for col in ["latitude", "longitude", "frp", "scan", "track", "brightness"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sensor_family"] = sensor_name
    df["source_satellite"] = (
        df["satellite"].astype("string").str.strip().replace(SATELLITE_NAMES)
    )
    df["confidence_raw"] = df["confidence_raw"].fillna("unknown")

    # Basic coordinate and timestamp validity checks.
    df = df.dropna(subset=[
        "latitude", "longitude", "acq_datetime_utc"
    ]).copy()
    df = df[
        df["latitude"].between(-90, 90) &
        df["longitude"].between(-180, 180)
    ].copy()

    # Optional region filter (default: global, nothing removed).
    if BBOX is not None:
        west, south, east, north = BBOX
        lat_ok = df["latitude"].between(south, north)
        if west <= east:
            lon_ok = df["longitude"].between(west, east)
        else:  # box crosses the antimeridian
            lon_ok = (df["longitude"] >= west) | (df["longitude"] <= east)
        df = df[lat_ok & lon_ok].copy()

    # Optional country mask (PYRO_AOI_FILE): the bbox above is only a rectangle around the country.
    if AOI_FILE is not None and len(df):
        df = df[inside_aoi(df["latitude"].to_numpy(), df["longitude"].to_numpy())].copy()

    # Optional: drop static / offshore / volcano sources (PYRO_EXCLUDE_STATIC=1); unknown type is kept.
    if EXCLUDE_STATIC:
        ftype = pd.to_numeric(df["type"], errors="coerce")
        df = df[ftype.isna() | (ftype == 0)].copy()

    # Optional date filter (inclusive).
    if START_DATE:
        df = df[df["acq_datetime_utc"] >= pd.Timestamp(START_DATE, tz="UTC")]
    if END_DATE:
        end_exclusive = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["acq_datetime_utc"] < end_exclusive]
    df = df.copy()

    # Retain original source-specific columns for traceability.
    preferred = [
        "sensor_family", "source_satellite", "instrument",
        "latitude", "longitude", "acq_date", "acq_time",
        "acq_datetime_utc", "frp", "brightness",
        "bright_ti4", "bright_ti5", "bright_t31",
        "scan", "track", "confidence_raw", "daynight", "version", "type"
    ]
    keep = [c for c in preferred if c in df.columns]
    df = df[keep].drop_duplicates().sort_values("acq_datetime_utc")
    return df.reset_index(drop=True)

def process_sensor(label, keyword, out_path):
    """Stream every file for one sensor through standardize(), appending to out_path."""
    out_path.unlink(missing_ok=True)  # never leave a stale file from an earlier run
    n_raw = n_kept = n_dup = 0
    first = True
    daily = []
    seen = np.empty(0, dtype=np.uint64)  # sorted hashes of (satellite, lat, lon, UTC time) already written
    raw_rows_per_day = {}  # UTC day -> raw rows in the file, before any AOI/date filter
    raw_rows_by_sat = {}   # satellite name -> {UTC day -> raw rows}: an outage of ONE satellite must stay visible
    t_min = t_max = None  # observation coverage of the raw files, before any AOI/date filter
    for path in find_files(RAW_DIR, keyword):
        print(f"{label}: reading {path.name}")
        for chunk in iter_firms_csv(path, CHUNK_ROWS):
            n_raw += len(chunk)
            std = standardize(chunk, label)

            raw_t = pd.to_datetime(
                chunk["acq_date"].astype(str) + " "
                + chunk["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True)
                .str.replace(r"\D", "", regex=True).str.zfill(4),
                format="%Y-%m-%d %H%M", errors="coerce", utc=True,
            )
            if raw_t.notna().any():
                lo, hi = raw_t.min(), raw_t.max()
                t_min = lo if t_min is None else min(t_min, lo)
                t_max = hi if t_max is None else max(t_max, hi)
                for day, cnt in raw_t.dropna().dt.strftime("%Y-%m-%d").value_counts().items():
                    raw_rows_per_day[day] = raw_rows_per_day.get(day, 0) + int(cnt)
                raw_sat = (chunk["satellite"].astype("string").str.strip().replace(SATELLITE_NAMES).fillna("unknown")
                           if "satellite" in chunk.columns else pd.Series("unknown", index=chunk.index))
                by_sat = pd.DataFrame({"day": raw_t.dt.strftime("%Y-%m-%d"), "sat": raw_sat.astype(str)}).dropna()
                for (sat, day), cnt in by_sat.groupby(["sat", "day"]).size().items():
                    d = raw_rows_by_sat.setdefault(str(sat), {})
                    d[day] = d.get(day, 0) + int(cnt)

            if len(std):
                # Drop detections already written from an earlier chunk or file
                # (e.g. overlapping 24h / 48h / 7d downloads).
                keys = pd.util.hash_pandas_object(
                    std[["source_satellite", "latitude", "longitude", "acq_datetime_utc"]],
                    index=False,
                ).to_numpy()
                if DEDUPE:
                    fresh = np.zeros(len(keys), dtype=bool)
                    fresh[np.unique(keys, return_index=True)[1]] = True  # first occurrence inside this chunk
                    if len(seen):  # ... and not written by an earlier chunk or file
                        pos = np.minimum(np.searchsorted(seen, keys), len(seen) - 1)
                        fresh &= seen[pos] != keys
                    seen = np.union1d(seen, keys[fresh])
                else:
                    fresh = np.ones(len(keys), dtype=bool)
                n_dup += int((~fresh).sum())
                std = std[fresh]
            n_kept += len(std)
            if len(std) == 0:
                continue
            std.to_csv(out_path, mode="w" if first else "a",
                       header=first, index=False)
            first = False
            day = std["acq_datetime_utc"].dt.floor("D").dt.tz_localize(None)
            daily.append(
                std.assign(date=day)
                .groupby(["sensor_family", "source_satellite", "date"])
                .size().rename("detections").reset_index()
            )
    if t_min is not None:
        (PROCESSED_DIR / f"{label.lower()}_coverage.json").write_text(
            json.dumps({"start_utc": str(t_min), "end_utc": str(t_max),
                        "raw_rows_per_day": dict(sorted(raw_rows_per_day.items())),
                        "raw_rows_per_day_by_satellite": {s: dict(sorted(d.items())) for s, d in sorted(raw_rows_by_sat.items())}}))
    print(f"{label} raw rows: {n_raw:,} | kept: {n_kept:,} | "
          f"duplicates removed across chunks/files: {n_dup:,}")
    if first:
        print(f"WARNING: no {label} rows survived filtering "
              "(check PYRO_START / PYRO_END / PYRO_BBOX).")
    return daily

def main():
    scope = "global" if BBOX is None else f"bbox {BBOX}"
    if AOI_FILE is not None:
        scope += f" masked by {AOI_FILE.name}"
    if EXCLUDE_STATIC:
        scope += ", non-static sources only (type 0)"
    dates = f"{START_DATE or 'any'} to {END_DATE or 'any'}"
    print(f"Scope: {scope}; dates: {dates}")
    (PROCESSED_DIR / "daily_detection_counts.csv").unlink(missing_ok=True)

    daily = []
    daily += process_sensor("MODIS", "modis", PROCESSED_DIR / "modis_standardized.csv")
    daily += process_sensor("VIIRS", "viirs", PROCESSED_DIR / "viirs_standardized.csv")

    # Chunks can split a day, so sum the partial counts. Descriptive only: do not
    # read sensor differences as real changes in fire activity without accounting
    # for overpass times and detection sensitivity.
    if daily:
        counts = (
            pd.concat(daily)
            .groupby(["sensor_family", "source_satellite", "date"], as_index=False)
            ["detections"].sum()
            .sort_values(["sensor_family", "source_satellite", "date"])
        )
        counts.to_csv(PROCESSED_DIR / "daily_detection_counts.csv", index=False)
        print("Saved daily counts and standardized CSVs in", PROCESSED_DIR)

if __name__ == "__main__":
    main()
