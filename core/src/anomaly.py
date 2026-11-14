"""Phase C: anomaly detection and "critical burning windows" on the harmonized calendar.

Needs MULTI-YEAR history (ideally >= 5 years; see PYRO_ANOM_MIN_YEARS). Input is
data/processed/calendar_daily.csv from `python -m src.harmonize`.

What it does
 1. Builds one daily series on the VIIRS reference scale from days whose `harmonized_status` is `measured` or
    `measured_zero`. Everything else is MISSING, never zero: days the VIIRS file does not contain
    (`viirs_not_observed`), the two partial days at the window edges (`partial_day`), days that look like
    partial outages (`measured_low_coverage`, excluded unless PYRO_ANOM_INCLUDE_LOW_COVERAGE=1, because a day that lost
    overpasses is biased low and would be flagged as a 'low' anomaly) and, when the calendar was built with a cloud file
    (PYRO_CLOUD_FILE), days that were observed but obscured by cloud with no detection (`observed_obscured`: not a
    confirmed zero, so never scored). MODIS is never substituted for VIIRS.
 2. Anomalies: for every day, a robust baseline (median and MAD of log1p FRP) taken from the
    same time of year (+- PYRO_ANOM_WINDOW days) in ALL OTHER YEARS (leave-one-year-out, so an
    extreme year cannot hide itself). robust z = (x - median) / (1.4826 * MAD).
 3. Anomalous episodes: runs of elevated days (z >= 2, gaps <= 2 days merged, >= 3 days,
    peak z >= PYRO_ANOM_Z).
 4. Validation (anomaly_validation.py, on by default): every episode is re-tested under alternative baselines, scale
    estimators, baseline periods, random data loss and a null model; see anomaly_validation.json. An episode is a
    statistical candidate, not a proven exceptional event: read its `robustness` column.
 5. Critical burning windows:
      - climatological: the smoothed multi-year median FRP by day of year; days at or above the
        PYRO_CRIT_PCT percentile form the typical critical window(s), with their share of
        annual FRP;
      - realised, per season-year: onset (10 % of the season's FRP), peak, end (90 %), so the
        window's shift between years is visible.

Run:  python -m src.anomaly
Outputs (data/processed): anomalies_daily.csv, anomaly_events.csv, critical_windows.csv,
season_by_year.csv, anomaly_report.json, anomaly_overview.png and, with validation, anomaly_validation.json,
anomaly_variants.csv, anomaly_event_variants.csv, anomaly_validation.png
"""
import json

import numpy as np
import pandas as pd

from core.src.config import (
    PROCESSED_DIR, ANOM_MIN_YEARS, ANOM_Z, ANOM_WINDOW_DAYS, CRIT_PCT, ANOM_VALIDATE, ANOM_INCLUDE_LOW_COVERAGE,
    ANOM_DETREND,
)
from core.src import hotsync_anomaly

VALUE_COL = "harmonized_frp_mw_viirs_equiv"
Z_EVENT = 2.0          # a day is "elevated" at or above this robust z
MIN_EVENT_DAYS = 3
MERGE_GAP_DAYS = 2
MAD_FLOOR = 0.10       # log units; stops a near-constant baseline from making z explode
MIN_POOL = 10          # minimum baseline samples for a day to get a z at all
MIN_SEASON_VALID_DAYS = 330
SCORING_STATUS = ("measured", "measured_zero")   # + measured_low_coverage with PYRO_ANOM_INCLUDE_LOW_COVERAGE=1


def doy365(dates: pd.Series) -> np.ndarray:
    """Day of year on a 365-day calendar (29 Feb shares a slot with 1 Mar)."""
    d = dates.dt.dayofyear.to_numpy()
    d = d - (dates.dt.is_leap_year.to_numpy() & (dates.dt.month.to_numpy() > 2)).astype(int)
    return np.minimum(d, 365)


def md_label(doy: int) -> str:
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(doy) - 1)).strftime("%d %b")


# --------------------------------------------------------------------------- series
def load_series() -> pd.DataFrame:
    p = PROCESSED_DIR / "calendar_daily.csv"
    if not p.exists():
        raise SystemExit("calendar_daily.csv not found. Run `python -m src.harmonize` first.")
    cal = pd.read_csv(p, parse_dates=["date"])
    cal = cal.copy()
    if "harmonized_status" in cal.columns:
        cal["status"] = cal["harmonized_status"].astype(str)
        measured = cal["status"].isin(("measured", "measured_zero", "measured_low_coverage"))
        ok = cal["status"].isin(SCORING_STATUS + (("measured_low_coverage",) if ANOM_INCLUDE_LOW_COVERAGE else ()))
    else:  # calendar written before observation status existed: keep the old rule, never score MODIS fallbacks
        ok = cal["day_complete"].astype(bool)
        if "day_observed" in cal.columns:
            ok &= cal["day_observed"].astype(bool)
        if "harmonized_source" in cal.columns:
            ok &= cal["harmonized_source"].astype(str).str.startswith("VIIRS")
        cal["status"] = np.where(ok, "measured", "legacy_excluded")
        measured = ok
    cal["valid"] = ok
    cal["value_measured"] = cal[VALUE_COL].where(measured)   # kept so validation can re-admit outage-suspect days
    cal["value"] = cal[VALUE_COL].where(ok)
    full = pd.date_range(cal["date"].min(), cal["date"].max(), freq="D")
    cal = cal.set_index("date").reindex(full)
    cal.index.name = "date"
    cal["valid"] = cal["valid"].fillna(False).astype(bool)
    cal["status"] = cal["status"].fillna("absent")
    cal["doy"] = doy365(cal.index.to_series())
    cal["year"] = cal.index.year
    return cal


# --------------------------------------------------------------------------- anomalies
def _pool_mask(rule, yr, years):
    """Which baseline years may be used for target year `yr`."""
    if rule == "loyo":                       # all other years (default)
        return years != yr
    if rule == "past":                       # causal: only earlier years
        return years < yr
    if rule == "skip_neighbours":            # other years except the two adjacent ones (multi-year persistence)
        return np.abs(years - yr) > 1
    if isinstance(rule, (set, frozenset, tuple, list)):   # a fixed baseline period
        return np.isin(years, list(rule)) & (years != yr)
    raise ValueError(f"unknown pool rule {rule!r}")


def _centre_scale(pool, method):
    if method == "median_mad":
        m = np.median(pool)
        return m, 1.4826 * np.median(np.abs(pool - m))
    if method == "median_iqr":
        q1, m, q3 = np.percentile(pool, [25, 50, 75])
        return m, (q3 - q1) / 1.349
    if method == "mean_sd":                  # classical z-score: NOT robust to the extremes it is meant to find
        return pool.mean(), pool.std(ddof=1)
    raise ValueError(f"unknown method {method!r}")


def baseline(x, doy, year, valid, window=None, method="median_mad", pool="loyo"):
    """Seasonal baseline (centre, scale, pool size) of x for every day, from the same time of year (+-window days) in
    the years allowed by `pool` (default: leave-one-year-out). Centre / scale are NaN where the pool is smaller than
    MIN_POOL; the pool size is returned so a variant can be judged 'testable' only when its baseline is not tiny."""
    window = ANOM_WINDOW_DAYS if window is None else window
    cen = np.full(len(x), np.nan)
    sca = np.full(len(x), np.nan)
    npool = np.zeros(len(x))
    usable = valid & np.isfinite(x)
    for d in range(1, 366):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        in_win = usable & (dist <= window)
        targets = np.flatnonzero(doy == d)
        for yr in np.unique(year[targets]):
            pool_x = x[in_win & _pool_mask(pool, yr, year)]
            rows = targets[year[targets] == yr]
            npool[rows] = len(pool_x)
            if len(pool_x) < MIN_POOL:
                continue
            c, sc = _centre_scale(pool_x, method)
            cen[rows] = c
            sca[rows] = sc
    return cen, np.maximum(sca, MAD_FLOOR), npool


def score(cal: pd.DataFrame, window=None, method="median_mad", pool="loyo", level_adjust=None, valid=None) -> pd.DataFrame:
    """Robust z-scores of the daily series. The keyword arguments exist for the validation: every one of them has a
    default that reproduces the production analysis.
      valid        : override of the validity mask (e.g. to re-admit outage-suspect days or to thin the data)
      level_adjust : remove each year's median level (guards against a multi-year trend contaminating the baseline);
                     default PYRO_ANOM_DETREND (off)"""
    level_adjust = ANOM_DETREND if level_adjust is None else level_adjust
    df = cal.copy()
    if valid is not None:
        df["valid"] = np.asarray(valid, bool)
        base_col = "value_measured" if "value_measured" in df.columns else "value"
        df["value"] = df[base_col].where(df["valid"])
    x = np.log1p(df["value"].to_numpy(float))
    ok = df["valid"].to_numpy() & np.isfinite(x)
    if level_adjust:
        lvl = pd.Series(np.where(ok, x, np.nan)).groupby(df["year"].to_numpy()).transform("median").to_numpy()
        x = x - (lvl - np.nanmedian(lvl))
    med, mad, npool = baseline(x, df["doy"].to_numpy(), df["year"].to_numpy(), ok, window, method, pool)
    df["pool_n"] = npool
    df["baseline_mw"] = np.expm1(med)
    df["band_low_mw"] = np.expm1(med - ANOM_Z * mad)
    df["band_high_mw"] = np.expm1(med + ANOM_Z * mad)
    df["ratio_to_baseline"] = (df["value"] + 1.0) / (df["baseline_mw"] + 1.0)
    df["z"] = (x - med) / mad
    df["flag"] = np.where(df["z"] >= ANOM_Z, "high", np.where(df["z"] <= -ANOM_Z, "low", ""))
    return df


def episode_slices(z, z_event=Z_EVENT, peak_z=None, min_days=MIN_EVENT_DAYS, merge_gap=MERGE_GAP_DAYS):
    """Episodes of an array of z-scores as (first_pos, last_pos) pairs. Elevated days (z >= z_event) closer than
    merge_gap + 1 are merged; an episode needs >= min_days elevated days and a peak >= peak_z (default ANOM_Z).
    The one rule shared by the production analysis, every validation variant and the Monte-Carlo null."""
    peak_z = ANOM_Z if peak_z is None else peak_z
    z = np.asarray(z, float)
    idx = np.flatnonzero(z >= z_event)
    if len(idx) == 0:
        return []
    out = []
    for g in np.split(idx, np.flatnonzero(np.diff(idx) > merge_gap + 1) + 1):
        if len(g) < min_days or np.nanmax(z[g[0]: g[-1] + 1]) < peak_z:
            continue
        out.append((int(g[0]), int(g[-1])))
    return out


def find_events(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["start", "end", "duration_days", "peak_date", "peak_z", "total_frp_mw", "excess_frp_mw"]
    rows = []
    for a, b in episode_slices(df["z"].to_numpy()):
        seg = df.iloc[a: b + 1]
        rows.append({
            "start": seg.index[0].date(), "end": seg.index[-1].date(),
            "duration_days": len(seg), "peak_date": seg["z"].idxmax().date(),
            "peak_z": float(seg["z"].max()),
            "total_frp_mw": float(seg["value"].sum()),
            "excess_frp_mw": float((seg["value"] - seg["baseline_mw"]).sum()),
        })
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- windows
def climatology(df: pd.DataFrame) -> np.ndarray:
    """Smoothed multi-year median FRP for day of year 1..365 (circular moving average)."""
    by = df["value"].groupby(df["doy"]).median().reindex(range(1, 366))
    clim = by.interpolate(limit_direction="both").to_numpy(float)
    w = ANOM_WINDOW_DAYS
    ext = np.r_[clim[-w:], clim, clim[:w]]
    return np.convolve(ext, np.ones(2 * w + 1) / (2 * w + 1), mode="valid")


def critical_windows(sm: np.ndarray):
    """Contiguous runs of days at/above the CRIT_PCT percentile, on a year that starts at the
    quietest day so a season crossing 31 Dec is not split in two."""
    start0 = int(np.argmin(sm))
    rot = np.roll(sm, -start0)
    thr = float(np.percentile(rot, CRIT_PCT))
    crit = rot >= thr
    edges = np.flatnonzero(np.diff(np.r_[0, crit.astype(int), 0]))
    rows = []
    for a, b in zip(edges[::2], edges[1::2] - 1):
        peak = a + int(np.argmax(rot[a:b + 1]))
        s, e, p = ((a + start0) % 365) + 1, ((b + start0) % 365) + 1, ((peak + start0) % 365) + 1
        rows.append({
            "start_doy": s, "end_doy": e, "peak_doy": p,
            "start": md_label(s), "end": md_label(e), "peak": md_label(p),
            "length_days": int(b - a + 1),
            "share_of_annual_frp": float(rot[a:b + 1].sum() / rot.sum()),
            "threshold_mw": thr,
        })
    return pd.DataFrame(rows), start0 + 1


def season_by_year(df: pd.DataFrame, season_start_doy: int) -> pd.DataFrame:
    shift = pd.Timedelta(days=season_start_doy - 1)
    sid = (df.index - shift).year
    rows = []
    for sy, g in df.groupby(sid):
        v = g["value"]
        n_valid = int(g["valid"].sum())
        row = {"season_year": int(sy), "season_start": g.index[0].date(), "valid_days": n_valid,
               "complete": bool(n_valid >= MIN_SEASON_VALID_DAYS and len(g) >= 360)}
        total = float(v.sum())
        smooth = v.rolling(15, center=True, min_periods=8).mean()
        # A stub season (e.g. the last 4 days of the record) has no smoothed value: leave its dates empty.
        if total > 0 and n_valid > 0 and smooth.notna().any():
            c = v.fillna(0).cumsum() / total
            row.update({
                "onset_date": c[c >= 0.10].index[0].date(),
                "end_date": c[c >= 0.90].index[0].date(),
                "peak_date": smooth.idxmax().date(),
                "total_frp_mw": total,
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    if "onset_date" in out:
        out["window_days"] = (pd.to_datetime(out["end_date"]) - pd.to_datetime(out["onset_date"])).dt.days + 1
    return out


# --------------------------------------------------------------------------- plot
def plot(df, events, windows, sm, season_start_doy, seasons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#33322f"
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [1.2, 1, 1]})

    ax = axes[0]
    ax.fill_between(df.index, df["band_low_mw"].clip(lower=0.1), df["band_high_mw"].clip(lower=0.1),
                    color=GREY, alpha=0.25, lw=0, label=f"expected range (robust z within ±{ANOM_Z:g})")
    ax.plot(df.index, df["baseline_mw"].clip(lower=0.1), color=GREY, lw=1, label="baseline (other years)")
    ax.plot(df.index, df["value"].where(df["value"] > 0), color=BLUE, lw=1, label="daily FRP")
    hi = df[df["flag"] == "high"]
    ax.scatter(hi.index, hi["value"], s=16, color=ORANGE, zorder=3, label=f"anomalous day (z ≥ {ANOM_Z:g})")
    for _, e in events.iterrows():
        ax.axvspan(pd.Timestamp(e["start"]), pd.Timestamp(e["end"]) + pd.Timedelta(days=1), color=ORANGE, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("Daily FRP (MW, VIIRS-equivalent, log scale)")
    ax.set_title("Daily burning activity vs. same-season baseline from the other years")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[1]
    doy = np.arange(1, 366)
    ax.plot(doy, sm, color=BLUE, lw=2, label="multi-year median (smoothed)")
    for _, w in windows.iterrows():
        a, b = w["start_doy"], w["end_doy"]
        if a <= b:
            ax.axvspan(a, b + 1, color=ORANGE, alpha=0.2, lw=0)
        else:
            ax.axvspan(a, 366, color=ORANGE, alpha=0.2, lw=0)
            ax.axvspan(1, b + 1, color=ORANGE, alpha=0.2, lw=0)
    ax.axhline(windows["threshold_mw"].iloc[0], color=GREY, ls="--", lw=1,
               label=f"{CRIT_PCT:g}th percentile threshold")
    ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set_xticks(ticks, [md_label(t)[3:] for t in ticks])
    ax.set_ylabel("Typical daily FRP (MW)")
    ax.set_title("Typical critical burning window (shaded)")
    ax.legend(fontsize=8)

    ax = axes[2]
    done = seasons[seasons["complete"]] if "onset_date" in seasons else seasons.iloc[0:0]
    for _, s in done.iterrows():
        x0 = (pd.Timestamp(s["onset_date"]) - pd.Timestamp(s["season_start"])).days
        x1 = (pd.Timestamp(s["end_date"]) - pd.Timestamp(s["season_start"])).days
        xp = (pd.Timestamp(s["peak_date"]) - pd.Timestamp(s["season_start"])).days
        ax.plot([x0, x1], [s["season_year"]] * 2, color=BLUE, lw=6, solid_capstyle="round")
        ax.scatter([xp], [s["season_year"]], color=ORANGE, s=40, zorder=3, edgecolor="white", linewidth=1)
    tick_days = np.arange(0, 365, 30)
    ax.set_xticks(tick_days, [md_label(((season_start_doy - 1 + t) % 365) + 1)[3:] + " " + md_label(((season_start_doy - 1 + t) % 365) + 1)[:2]
                              for t in tick_days], rotation=45, fontsize=8)
    ax.set_ylabel("Season year")
    ax.set_title("Realised burning window per season: 10 % to 90 % of FRP (bar), peak (dot)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PROCESSED_DIR / "anomaly_overview.png", dpi=150)


# --------------------------------------------------------------------------- main
def main():
    cal = load_series()
    n_valid = int(cal["valid"].sum())
    n_years = n_valid / 365.25
    years_present = int(cal.loc[cal["valid"], "year"].nunique())
    if years_present < 2:
        raise SystemExit(
            f"Only {n_valid} valid day(s) in {years_present} calendar year(s). Anomaly detection and critical "
            f"windows need multi-year history (>= 2 years to run, >= {ANOM_MIN_YEARS} to trust). "
            "Download a multi-year archive (see README, section 'Multi-year archive') and re-run "
            "preprocess + harmonize first.")
    reasons = []
    if n_years < ANOM_MIN_YEARS:
        reasons.append(f"only {n_years:.1f} years of valid days (< {ANOM_MIN_YEARS})")
    by_status = {k: int(v) for k, v in cal["status"].value_counts().items()}
    excluded = int(len(cal) - n_valid)
    if excluded > 0.05 * len(cal):
        reasons.append(f"{excluded / len(cal):.0%} of days are not usable (not observed, partial, outage-suspect or cloud-obscured)")
    reliable = not reasons

    df = score(cal)
    # Keep PyroScope's robust FRP detector as the primary result, and add a
    # separate HotSync-style active-cell Z-score for count-based comparison.
    hotsync_scores, hotsync_info = hotsync_anomaly.score_activity(cal)
    hotsync_events = hotsync_anomaly.top_anomalies(hotsync_scores)
    events = find_events(df)
    validation = None
    if ANOM_VALIDATE:
        from core.src import anomaly_validation
        validation = anomaly_validation.validate(cal, df, events)
        events = validation["events"]
    sm = climatology(df)
    windows, season_start_doy = critical_windows(sm)
    seasons = season_by_year(df, season_start_doy)

    out = df[["value", "valid", "status", "baseline_mw", "ratio_to_baseline", "z", "flag"]].rename(columns={"value": "frp_mw"})
    out = out.join(hotsync_scores)
    out.to_csv(PROCESSED_DIR / "anomalies_daily.csv")
    events.to_csv(PROCESSED_DIR / "anomaly_events.csv", index=False)
    hotsync_events.to_csv(PROCESSED_DIR / "hotsync_activity_anomalies.csv", index=False)
    hotsync_anomaly.plot_activity(hotsync_scores, PROCESSED_DIR / "hotsync_activity_overview.png")
    windows.to_csv(PROCESSED_DIR / "critical_windows.csv", index=False)
    seasons.to_csv(PROCESSED_DIR / "season_by_year.csv", index=False)
    plot(df, events, windows, sm, season_start_doy, seasons)
    if validation is not None:
        anomaly_validation.write_outputs(validation, df)

    done = seasons[seasons["complete"]] if "window_days" in seasons else seasons.iloc[0:0]
    report = {
        "reliable": reliable, "reasons_unreliable": reasons,
        "valid_days": n_valid, "missing_days": excluded, "years_of_valid_days": round(n_years, 2),
        "days_by_status": by_status,
        "outage_suspect_days_excluded": 0 if ANOM_INCLUDE_LOW_COVERAGE else by_status.get("measured_low_coverage", 0),
        **({"cloud_obscured_days_excluded": by_status["observed_obscured"]} if by_status.get("observed_obscured") else {}),
        "anomalous_days_high": int((df["flag"] == "high").sum()),
        "anomalous_days_low": int((df["flag"] == "low").sum()),
        "episodes": int(len(events)), "season_starts_at_doy": season_start_doy,
        "complete_seasons": int(len(done)),
        "median_window_days": None if done.empty else float(done["window_days"].median()),
        "episode_robustness": ({} if validation is None else
                               {k: int(v) for k, v in events["robustness"].value_counts().items()}),
        "validation_file": None if validation is None else "anomaly_validation.json",
        "hotsync_activity": {
            **hotsync_info,
            "flagged_days": int((hotsync_scores["hotsync_activity_flag"] == "high").sum()),
            "reported_days": int(len(hotsync_events)),
            "events_file": "hotsync_activity_anomalies.csv",
        },
        "caveat": ("An episode is a statistical candidate: a z-score does not prove that an event is scientifically "
                   "exceptional (see `robustness` and anomaly_validation.json). 'low' flags can be satellite or "
                   "ground-segment outages, not less fire; days that look like partial outages are excluded. A single "
                   "reference sensor family is assumed; the trend is not removed."),
    }
    (PROCESSED_DIR / "anomaly_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nTypical critical burning window(s):")
    print(windows[["start", "end", "peak", "length_days", "share_of_annual_frp"]].round(3).to_string(index=False))
    print("\nAnomalous episodes:")
    print(events.round(2).to_string(index=False) if len(events) else "none")
    if validation is not None:
        print("\nValidation:", validation["summary_text"])
    if not reliable:
        print("\nUNRELIABLE:", "; ".join(reasons))
    print("\nSaved anomalies_daily.csv, anomaly_events.csv, critical_windows.csv, season_by_year.csv, "
          "anomaly_report.json, anomaly_overview.png" + (", anomaly_validation.json/.png, anomaly_variants.csv, "
                                                         "anomaly_event_variants.csv" if validation else "") + " in", PROCESSED_DIR)


if __name__ == "__main__":
    main()
