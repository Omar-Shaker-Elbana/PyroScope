"""Country-by-country 14-day forecast, ready to view on the HotSync map.

STEP 4 FOLLOW-UP: this no longer takes raw --modis/--viirs CSVs or a
--live-key. It reads core.src.harmonize's calendar the exact same way
train_countries.py does (via hotsync_country_model.prepare_country_series),
so forecasting and training are always on the same per-satellite,
trust-graded data -- not two different harmonization paths.

The forecast starts the day after the LAST day currently in that calendar.
To get a fresh forecast, refresh the calendar first, then run this:

    python manage.py download_firms --start <last_date> --end <today> [--bbox ...]
    python manage.py harmonize
    python manage.py predict_countries

This script does not fetch FIRMS data or re-run harmonize itself -- same
"pipelines run async, this only reads whatever they last wrote" rule as
core.validation.confidence and train_countries.py.

Example:
    python manage.py predict_countries
    python manage.py predict_countries --model /path/to/model_countries.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from core.src.config import OUTPUTS_DIR

from . import hotsync_country_model as cm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(OUTPUTS_DIR / "model_countries.joblib"))
    ap.add_argument("--out-dir", default=str(OUTPUTS_DIR))
    args = ap.parse_args()

    art = joblib.load(args.model)

    # Same call shape as train_countries.py: (target, countries_path, start, end, min_active_days, max_countries).
    # start/end are left open so this always reads up through the newest day the calendar has.
    wide, _, _ = cm.prepare_country_series(
        art["target"], art["countries_path"], None, None, min_active_days=0, max_countries=10_000
    )
    wide = wide[[c for c in art["countries"] if c in wide.columns]]
    if wide.empty:
        raise SystemExit(
            "No harmonized data available for this model's countries. Run `python manage.py "
            "preprocess` and `python manage.py harmonize` first."
        )

    last = wide.index[-1]
    age = (pd.Timestamp.now().normalize() - last).days
    if age > 14:
        print(
            f"WARNING: newest harmonized day is {last.date()} ({age} days old). The forecast covers "
            f"the 14 days after that date, not today. Run `python manage.py download_firms` then "
            f"`python manage.py harmonize` to bring the calendar up to date before forecasting."
        )

    fc = cm.forecast_latest(art["models"], wide, art["meta"], art["thresholds"], art["target"], art["horizons"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out / "forecast_by_country.csv", index=False)

    recent = wide.loc[wide.index >= last - pd.Timedelta(days=59)].stack().rename("observed").reset_index()
    recent.columns = ["date", "country", "observed"]
    recent["date"] = recent["date"].dt.date
    recent.to_csv(out / "recent_by_country.csv", index=False)

    unit = "active cell-days" if art["target"] == "activity" else "MW"
    print(f"\nLast harmonized day: {last.date()} | forecast unit: {unit}")
    top = fc[fc["horizon_days"] == 7].sort_values("predicted", ascending=False).head(10)
    print("Top 10 countries, predicted value 7 days ahead:")
    print(top[["name", "date", "predicted", "low_p10", "high_p90", "typical", "busy_day"]].to_string(index=False))
    print(f"\nSaved forecast_by_country.csv and recent_by_country.csv to {out.resolve()}")


if __name__ == "__main__":
    main()