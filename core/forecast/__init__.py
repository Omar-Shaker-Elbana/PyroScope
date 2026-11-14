"""core.forecast: worldmap_forcase's per-country fire-activity forecaster.

hotsync_model.py          - the pooled gradient-boosting model (mean/p10/p90)
hotsync_country_model.py  - stacks per-country rows, evaluation, forecasting
legacy_hotsync_data.py    - LEGACY global kM/kV harmonizer (see its docstring
                             — slated for replacement by core.src.harmonize)
train_countries.py        - trains the model, writes outputs/ (joblib + metrics)
predict_countries.py      - loads the trained model, writes forecast_by_country.csv

NOTE: predict_countries.py below is a RECONSTRUCTION. The original file
failed to extract from the uploaded .rar (came through as 0 bytes), so this
version was rebuilt from train_countries.py's joblib schema and
hotsync_country_model.forecast_latest(). Check it against your actual
script if you still have a good copy, and replace it if it differs.
"""
