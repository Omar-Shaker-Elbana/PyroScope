"""Load the trained country model and write a fresh forecast_by_country.csv.

RECONSTRUCTED FILE — see core/forecast/__init__.py docstring. The original
predict_countries.py failed to extract from the uploaded .rar (0 bytes).
This rebuild is based on exactly what train_countries.py saves into
model_countries.joblib and what hotsync_country_model.forecast_latest()
returns/expects, and it reproduces the header seen in the real
outputs/forecast_by_country.csv:

    country,name,iso_n3,date,horizon_days,predicted,low_p10,high_p90,
    typical,busy_day,target

Verify against your own copy if you still have one, and replace this file
if anything here doesn't match.

Example:
    python predict_countries.py --modis modis.csv --viirs viirs.csv \
        --model outputs/model_countries.joblib --out-dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from . import hotsync_country_model as cm
from . import legacy_hotsync_data as hd  # TODO: swap for core.src.harmonize, see legacy_hotsync_data.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modis", nargs="+", required=True)
    ap.add_argument("--viirs", nargs="+", required=True)
    ap.add_argument("--model", default="outputs/model_countries.joblib")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    models, target = bundle["models"], bundle["target"]
    countries_path, bbox = bundle["countries_path"], bundle["bbox"]

    # Re-harmonize the freshest raw FIRMS data (same path train_countries.py
    # used) so `wide` carries whatever days are newest, then restrict to the
    # countries the model was actually trained on.
    wide, meta, info = cm.prepare_country_series(
        hd.read_raw(args.modis), hd.read_raw(args.viirs), target, countries_path, bbox,
    )
    trained_cols = [c for c in bundle["countries"] if c in wide.columns]
    wide, meta = wide[trained_cols], {c: bundle["meta"][c] for c in trained_cols}

    forecast = cm.forecast_latest(models, wide, meta, bundle["thresholds"], target)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out / "forecast_by_country.csv", index=False)
    print(f"Wrote {len(forecast)} forecast rows for {len(trained_cols)} countries to "
          f"{(out / 'forecast_by_country.csv').resolve()}")
    print(f"Model trained through {bundle['trained_through']}; harmonization kM="
          f"{info['kM']:.2f} kV={info['kV']:.2f} (this run's re-harmonization, "
          f"for reference against the model's own training-time values)")


if __name__ == "__main__":
    main()
