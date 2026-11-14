"""core.validation: forecast-vs-real-data confidence scoring (plan steps 4-6).

Step 2 of the plan is implemented in confidence.py:

  1. load_observed_activity(): per-country harmonized activity, read from
     whatever `python manage.py harmonize` last wrote to
     core.src.config.PROCESSED_DIR (calendar_cells.csv / calendar_daily.csv) --
     core.src.harmonize's per-satellite k, bootstrap CIs and trust grades,
     not legacy_hotsync_data's single global kM/kV.
  2. backtest_forecast(): joins core.forecast's forecast_by_country.csv
     against that observed activity by (country, date) -- the same predicted
     vs. observed comparison worldmap_forcase/jspyro drew as a chart
     (HotSync_harmonized_Final.html's forecast-section: orange predicted band
     vs. teal observed), but scored as a real backtest against PyroScope's
     more accurate harmonization instead of eyeballed on a plot.
  3. harmonization_confidence(): attaches a confidence flag per country from
     core.src.harmonize's k_trust / bootstrap CI-width / k-window-widened
     output (frp_mapping.json), independent of the forecast model's own
     p10/p90 band.
  4. confidence_report(): merges 2 + 3 into the table the `validation` app's
     view renders.

build_report() in confidence.py wires all four together and is what the
Django view/management command call.
"""
