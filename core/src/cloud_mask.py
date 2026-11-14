"""Cloud / obscuration input for the observation status: "observed, but obscured" (review point 6).

Why
---
A satellite can pass over an area and see nothing because cloud hides the ground. That is neither "there was no fire"
nor "the sensor did not look". FIRMS detection files carry no cloud information, so an observed day with 0 detections
was always read as "no fire", even when most of the area was under cloud. With a daily clear-sky share per day this
module lets `harmonize.py` add a third state (`harmonized_status = observed_obscured`) and stop counting such days as
confirmed zero-fire days.

Input (optional): PYRO_CLOUD_FILE -> a CSV with one row per UTC day (and optionally per satellite)
    date             YYYY-MM-DD (UTC day; a full timestamp is floored to its day)
    clear_fraction   share of the area of interest that was observed clear, 0..1
                     (alternatives, first one present wins: cloud_fraction = 1 - clear, 0..1;
                      clear_percent / cloud_percent, 0..100. Values are never guessed to be percentages.)
    satellite        optional: S-NPP, NOAA-20, ... A row with a satellite is used only for that satellite; a row without
                     one (or "viirs" / "all") applies to the reference VIIRS satellite when it has no rows of its own.
Several rows for the same day and satellite (day and night overpass, several granules) are averaged. A day that is not in
the file has UNKNOWN cloud cover: it is never treated as clear and never as cloudy.

How to produce the file from a cloud-mask product (MOD35_L2 / MYD35_L2, or a VIIRS cloud mask): clip every granule to
the same area of interest as the fire data, take the share of area pixels that are confidently / probably clear
(`clear_share_from_mod35` below decodes the MOD35 `Cloud_Mask` first byte), and average per UTC day. Downloading and
clipping the granules is outside this module. `python -m src.cloud_mask file.csv` checks a file before a long run.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GENERIC_KEYS = {"", "viirs", "all", "any", "nan", "none"}
_FRACTION_COLUMNS = (("clear_fraction", False, 1.0), ("cloud_fraction", True, 1.0),
                     ("clear_percent", False, 100.0), ("cloud_percent", True, 100.0))


def _slug(name) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


# --------------------------------------------------------------------------- decoding a cloud-mask product
def clear_share_from_mod35(cloud_mask_byte0, in_area=None, probably_clear: bool = True) -> float:
    """Share of the area's pixels that are clear in a MOD35_L2 / MYD35_L2 `Cloud_Mask` (its first byte).

    Bit 0 = cloud mask determined; bits 1-2 = unobstructed-FOV quality: 0 cloudy, 1 uncertain, 2 probably clear,
    3 confident clear. A pixel counts as clear when the mask was determined AND it is confidently clear (or probably
    clear, unless `probably_clear=False`). Pixels where the mask was not determined or is uncertain count as NOT
    clear: the sensor could not be shown to have seen the ground there, so a zero cannot be claimed.
    in_area : boolean array, the pixels that belong to the area of interest AND are inside the swath (default: all).
    Returns NaN when the area has no pixel."""
    b = np.asarray(cloud_mask_byte0)
    if b.size and (np.nanmin(b) < 0 or np.nanmax(b) > 255):
        raise ValueError("Cloud_Mask byte values must be in 0..255")
    b = b.astype(np.int64)
    area = np.ones(b.shape, bool) if in_area is None else np.asarray(in_area, bool)
    if area.shape != b.shape:
        raise ValueError("in_area must have the same shape as the cloud-mask array")
    n = int(area.sum())
    if n == 0:
        return float("nan")
    quality = (b >> 1) & 3
    clear = ((b & 1) == 1) & ((quality == 3) | ((quality == 2) & probably_clear))
    return float(clear[area].sum() / n)


# --------------------------------------------------------------------------- reading the daily table
def read_table(path) -> pd.DataFrame:
    """Read and validate the daily cloud table. Returns columns date (naive UTC day), key (satellite slug, "" = none),
    clear (0..1). A wrong file stops the run with a message: a cloud file that is silently ignored would look like
    clear skies."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"PYRO_CLOUD_FILE={str(path)!r} does not exist. Unset PYRO_CLOUD_FILE to run without cloud "
                         "information (every observed empty day is then a measured zero, as before).")
    try:
        raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig", skipinitialspace=True)
    except Exception as e:  # unreadable / empty / not a CSV
        raise SystemExit(f"{path.name}: cannot be read as a CSV ({e}).") from e
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "date" not in raw.columns:
        raise SystemExit(f"{path.name}: needs a 'date' column (found {list(raw.columns)}).")
    spec = next(((c, inv, scale) for c, inv, scale in _FRACTION_COLUMNS if c in raw.columns), None)
    if spec is None:
        raise SystemExit(f"{path.name}: needs one of the columns clear_fraction, cloud_fraction, clear_percent, "
                         f"cloud_percent (found {list(raw.columns)}).")
    col, inverted, scale = spec

    dates = pd.to_datetime(raw["date"].str.strip(), errors="coerce", utc=True, format="mixed")   # "2016-03-02" and a timestamp may share a file
    if dates.isna().any():
        bad = raw.loc[dates.isna(), "date"].iloc[0]
        raise SystemExit(f"{path.name}: {int(dates.isna().sum())} row(s) with an unreadable date, e.g. {bad!r}.")
    text = raw[col].str.strip()
    value = pd.to_numeric(text, errors="coerce")
    unreadable = value.isna() & text.notna() & (text != "")
    if unreadable.any():
        raise SystemExit(f"{path.name}: {int(unreadable.sum())} non-numeric value(s) in '{col}', e.g. "
                         f"{text[unreadable].iloc[0]!r}.")
    value = value / scale
    if ((value < -1e-9) | (value > 1 + 1e-9)).any():
        raise SystemExit(f"{path.name}: '{col}' must lie between 0 and {scale:g} "
                         f"(found {value.min() * scale:g} .. {value.max() * scale:g}). Use clear_percent / cloud_percent "
                         "for percentages; they are not guessed.")
    clear = (1.0 - value if inverted else value).clip(0.0, 1.0)
    sat = raw["satellite"].fillna("").map(_slug) if "satellite" in raw.columns else pd.Series("", index=raw.index)
    out = pd.DataFrame({"date": dates.dt.tz_localize(None).dt.floor("D"), "key": sat.to_numpy(), "clear": clear.to_numpy()})
    out = out.dropna(subset=["clear"])          # an empty value = unknown, same as an absent day
    if out.empty:
        raise SystemExit(f"{path.name}: no row with a value in '{col}'.")
    return out.reset_index(drop=True)


def clear_series(table: pd.DataFrame, satellite, days):
    """Clear-sky share for each day of `days` for the calendar's reference satellite: (Series over `days`, info).
    NaN = no cloud information for that day (unknown, not clear)."""
    days = pd.DatetimeIndex(days)
    key = _slug(satellite)
    t = table[table["key"] == key]
    basis = f"{satellite} rows"
    if t.empty:
        t = table[table["key"].isin(GENERIC_KEYS)]
        basis = "rows without a satellite"
    problems = []
    if t.empty:
        problems.append(f"the file has no rows for {satellite} and none without a satellite "
                        f"(it has: {sorted(set(table['key']))}); cloud information is NOT applied")
        basis = "none"
    daily = t.groupby("date")["clear"].mean()
    s = daily.reindex(days).rename("clear_fraction")
    if len(daily) and not t["date"].between(days.min(), days.max()).any():
        problems.append("the file has no day inside the calendar window; cloud information is NOT applied")
    info = {
        "applied_to_satellite": str(satellite), "rows_used": basis, "n_rows": int(len(t)),
        "file_first_day": None if t.empty else str(t["date"].min().date()),
        "file_last_day": None if t.empty else str(t["date"].max().date()),
        "calendar_days": int(len(days)), "days_with_cloud_info": int(s.notna().sum()),
        "days_without_cloud_info": int(s.isna().sum()),
        "problems": problems,
    }
    return s, info


def describe(info) -> str:
    if not info or not info.get("enabled"):
        return "cloud information: not used"
    s = (f"cloud information ({info.get('rows_used')}): {info['days_with_cloud_info']:,} of {info['calendar_days']:,} days "
         f"known, {info['days_without_cloud_info']:,} unknown (never treated as clear)")
    if "days_marked_observed_obscured" in info:
        s += (f"; clear share < {info['min_clear_fraction']:g} on {info['days_cloud_limited']:,} observed days, "
              f"{info['days_marked_observed_obscured']:,} marked observed_obscured")
    return s + "".join(f"\n  WARNING: {p}" for p in info.get("problems", []))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.cloud_mask <cloud_file.csv>")
    tab = read_table(sys.argv[1])
    from core.src.config import CLOUD_MIN_CLEAR
    print(f"{len(tab):,} rows, {tab['date'].min().date()} -> {tab['date'].max().date()}, "
          f"satellites: {sorted(set(tab['key'])) or ['(none)']}")
    daily = tab.groupby(["key", "date"])["clear"].mean()
    print(f"median clear share {daily.median():.2f}; {(daily < CLOUD_MIN_CLEAR).mean():.1%} of day x satellite values "
          f"are below PYRO_CLOUD_MIN_CLEAR={CLOUD_MIN_CLEAR:g}")
