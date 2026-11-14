"""Train the country-by-country HotSync forecaster on real FIRMS exports.

Example:
    python train_countries.py --modis modis.csv --viirs viirs.csv
    python train_countries.py --modis m1.csv m2.csv --viirs v1.csv --max-countries 80 --stride 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from . import hotsync_country_model as cm
from . import hotsync_model as hm
from . import legacy_hotsync_data as hd  # TODO: swap for core.src.harmonize, see legacy_hotsync_data.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modis", nargs="+", required=True)
    ap.add_argument("--viirs", nargs="+", required=True)
    ap.add_argument("--countries", help="own Natural-Earth-style borders file (default: download once)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--target", choices=["activity", "frp"], default="activity")
    ap.add_argument("--test-days", type=int, default=365)
    ap.add_argument("--max-countries", type=int, default=120, help="model the N most fire-active countries")
    ap.add_argument("--min-active-days", type=int, default=60, help="skip countries with fewer fire days")
    ap.add_argument("--stride", type=int, default=3, help="use every n-th day for training (speed/memory)")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox) if args.bbox else None

    wide, meta, info = cm.prepare_country_series(
        hd.read_raw(args.modis), hd.read_raw(args.viirs), args.target, args.countries, bbox,
        args.min_active_days, args.max_countries)
    print(f"Modelling {wide.shape[1]} countries, {wide.index[0].date()} -> {wide.index[-1].date()} | "
          f"kM={info['kM']:.2f} kV={info['kV']:.2f}")

    print(f"Evaluating on the last {args.test_days} days (trained only on earlier data) ...")
    metrics, test, _ = cm.evaluate_panel(wide, meta, args.target, args.test_days, args.stride)
    metrics["harmonization"] = info
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    o = metrics["overall"]
    print("\nHeld-out results, all countries pooled (lower is better):")
    for k in ("model", "persistence_baseline", "same_season_baseline"):
        print(f"  {k:<22} MAE {o[k]['MAE']:8.2f}   RMSE {o[k]['RMSE']:8.2f}   corr {o[k]['corr']:.2f}")
    print(f"  80% interval covered {metrics['interval_80pct_coverage']:.0%} of test rows")
    wins = sum(r["MAE_model"] < r["MAE_same_season"] for r in metrics["by_country"])
    print(f"  model beats the same-season baseline in {wins}/{len(metrics['by_country'])} of the biggest countries")

    print("\nFitting final model on ALL data ...")
    rows = cm.build_panel(wide, meta, stride=args.stride).dropna(subset=["target"])
    models = hm.fit(rows, cm.FEATURES_C)
    joblib.dump({
        "models": models, "features": cm.FEATURES_C, "horizons": hm.HORIZONS, "target": args.target,
        "bbox": bbox, "countries": list(wide.columns), "meta": {c: meta[c] for c in wide.columns},
        "thresholds": cm.thresholds(wide, args.target), "countries_path": args.countries,
        "trained_through": str(wide.index[-1].date()), "harmonization": info,
    }, out / "model_countries.joblib")
    print(f"Saved model + metrics to {out.resolve()}")


if __name__ == "__main__":
    main()
