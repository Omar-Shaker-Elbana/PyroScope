"""Country-by-country 14-day forecast, ready to view on the HotSync map.

The forecast starts the day after the LAST day in your data, so feed it fresh
data: re-export recent FIRMS CSVs, or add --live-key to top up with the last ~40
days of near-real-time detections.

Example:
    python manage.py predict_countries --modis modis.csv --viirs viirs.csv
    python manage.py predict_countries --modis modis.csv --viirs viirs.csv --live-key $FIRMS_MAP_KEY
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from core.src.config import OUTPUTS_DIR

from . import hotsync_country_model as cm
from . import legacy_hotsync_data as hd  # TODO: swap for core.src.harmonize, see legacy_hotsync_data.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modis", nargs="+", required=True)
    ap.add_argument("--viirs", nargs="+", required=True)
    ap.add_argument("--model", default=str(OUTPUTS_DIR / "model_countries.joblib"))
    ap.add_argument("--live-key", help="FIRMS MAP_KEY: append the last ~40 days of NRT detections")
    ap.add_argument("--out-dir", default=str(OUTPUTS_DIR))
    args = ap.parse_args()

    art = joblib.load(args.model)
    raw_m, raw_v = hd.read_raw(args.modis), hd.read_raw(args.viirs)
    if args.live_key:
        b = art["bbox"] or (float(raw_m["latitude"].min()), float(raw_m["latitude"].max()),
                            float(raw_m["longitude"].min()), float(raw_m["longitude"].max()))
        print("Fetching latest NASA FIRMS detections ...")
        live = hd.fetch_live(args.live_key, b)
        raw_m = pd.concat([raw_m, live["M"]], ignore_index=True)
        raw_v = pd.concat([raw_v, live["V"]], ignore_index=True)

    wide, _, _ = cm.prepare_country_series(raw_m, raw_v, art["target"], art["countries_path"], art["bbox"],
                                           min_active_days=0, max_countries=10_000)
    wide = wide[[c for c in art["countries"] if c in wide.columns]]
    last = wide.index[-1]
    age = (pd.Timestamp.now().normalize() - last).days
    if age > 14:
        print(f"WARNING: newest data is {last.date()} ({age} days old). The forecast covers the 14 days "
              f"after that date, not today. Use fresher exports or --live-key.")

    fc = cm.forecast_latest(art["models"], wide, art["meta"], art["thresholds"], art["target"], art["horizons"])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out / "forecast_by_country.csv", index=False)

    recent = wide.loc[wide.index >= last - pd.Timedelta(days=59)].stack().rename("observed").reset_index()
    recent.columns = ["date", "country", "observed"]
    recent["date"] = recent["date"].dt.date
    recent.to_csv(out / "recent_by_country.csv", index=False)

    unit = "active cell-days" if art["target"] == "activity" else "MW"
    print(f"\nLast observed day: {last.date()} | forecast unit: {unit}")
    top = fc[fc["horizon_days"] == 7].sort_values("predicted", ascending=False).head(10)
    print("Top 10 countries, predicted value 7 days ahead:")
    print(top[["name", "date", "predicted", "low_p10", "high_p90", "typical", "busy_day"]].to_string(index=False))
    print(f"\nSaved forecast_by_country.csv and recent_by_country.csv to {out.resolve()}")


if __name__ == "__main__":
    main()
