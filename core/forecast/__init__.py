"""core.forecast: worldmap_forcase's per-country fire-activity forecaster.

hotsync_model.py          - the pooled gradient-boosting model (mean/p10/p90)
hotsync_country_model.py  - stacks per-country rows, evaluation, forecasting
legacy_hotsync_data.py    - LEGACY global kM/kV harmonizer (see its docstring
                             — slated for replacement by core.src.harmonize)
train_countries.py        - trains the model, writes data/outputs/ (joblib + metrics)
predict_countries.py      - loads the trained model, writes forecast_by_country.csv

predict_countries.py is the original script from the jspyro repo (including --live-key for
near-real-time FIRMS top-up), with imports switched to the package layout and
output paths pointed at DATA_DIR/outputs.
"""
