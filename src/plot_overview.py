import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, CHUNK_ROWS, DAY_TOL_MIN

FILES = {
    "MODIS C6.1": PROCESSED_DIR / "modis_standardized.csv",
    "VIIRS 375 m": PROCESSED_DIR / "viirs_standardized.csv",
}
FRP_BINS = np.linspace(0, 200, 81)       # MW; detections above 200 MW not shown
LON_EDGES = np.arange(-180, 181, 1.0)
LAT_EDGES = np.arange(-90, 91, 1.0)

def drop_partial_days(counts):
    """Hide UTC days the raw file does not fully cover: a partial day looks like a fake drop in activity."""
    tol = pd.Timedelta(minutes=DAY_TOL_MIN)
    keep = pd.Series(True, index=counts.index)
    for family in counts["sensor_family"].unique():
        f = PROCESSED_DIR / f"{family.lower()}_coverage.json"
        if not f.exists():
            continue
        cov = json.loads(f.read_text())
        start = pd.Timestamp(cov["start_utc"]).tz_localize(None)
        end = pd.Timestamp(cov["end_utc"]).tz_localize(None)
        full = ((counts["date"] + tol >= start)
                & (counts["date"] + pd.Timedelta(days=1) - tol <= end))
        keep &= (counts["sensor_family"] != family) | full
    if not keep.any():
        return counts, " (all days partial)"
    note = f" ({int((~keep).sum())} partial day(s) hidden)" if (~keep).any() else ""
    return counts[keep], note

def plot_counts():
    counts = pd.read_csv(PROCESSED_DIR / "daily_detection_counts.csv",
                         parse_dates=["date"])
    counts, partial_note = drop_partial_days(counts)
    span_days = (counts["date"].max() - counts["date"].min()).days + 1
    monthly = span_days > 90
    fig, ax = plt.subplots(figsize=(12, 5))
    for (family, sat), g in counts.groupby(["sensor_family", "source_satellite"]):
        s = g.set_index("date")["detections"]
        if monthly:
            s = s.resample("MS").sum()
        ax.plot(s.index, s.values, marker="o" if len(s) < 40 else None,
                label=f"{family} {sat}")
    ax.set_title("Global active-fire detections per "
                 + ("month" if monthly else "day") + " (by satellite)" + partial_note)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Number of detections")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PROCESSED_DIR / "detection_counts.png", dpi=160)

def scan_files():
    """One pass over each standardized file: FRP histogram + 1-degree density grid."""
    hist, grid = {}, {}
    for label, path in FILES.items():
        h = np.zeros(len(FRP_BINS) - 1)
        g = np.zeros((len(LAT_EDGES) - 1, len(LON_EDGES) - 1))
        for chunk in pd.read_csv(path, usecols=["frp", "latitude", "longitude"],
                                 chunksize=CHUNK_ROWS):
            frp = chunk["frp"].dropna().to_numpy()
            h += np.histogram(frp, bins=FRP_BINS)[0]
            g += np.histogram2d(chunk["latitude"], chunk["longitude"],
                                bins=[LAT_EDGES, LON_EDGES])[0]
        hist[label], grid[label] = h, g
    return hist, grid

def plot_frp(hist):
    fig, ax = plt.subplots(figsize=(10, 5))
    centers = (FRP_BINS[:-1] + FRP_BINS[1:]) / 2
    width = FRP_BINS[1] - FRP_BINS[0]
    for label, h in hist.items():
        ax.bar(centers, h / h.sum() / width, width=width, alpha=0.55, label=label)
    ax.set_title("FRP distributions (exploratory, not harmonized; >200 MW not shown)")
    ax.set_xlabel("Fire Radiative Power (MW)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PROCESSED_DIR / "frp_distributions.png", dpi=160)

def plot_maps(grid):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, (label, g) in zip(axes, grid.items()):
        masked = np.ma.masked_where(g == 0, g)
        im = ax.pcolormesh(LON_EDGES, LAT_EDGES, masked, cmap="inferno",
                           norm="log", shading="flat")
        ax.set_title(f"{label}: detections per 1° cell (log scale)")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, pad=0.01)
        ax.set_aspect("equal")
    axes[-1].set_xlabel("Longitude")
    fig.tight_layout()
    fig.savefig(PROCESSED_DIR / "detection_density_map.png", dpi=140)

def main():
    plot_counts()
    hist, grid = scan_files()
    plot_frp(hist)
    plot_maps(grid)
    print("Plots saved in", PROCESSED_DIR)

if __name__ == "__main__":
    main()
