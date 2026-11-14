"""core: the engine. Plain Python package, NOT a Django app.

Submodules:
  core.src        - PyroScope harmonization/anomaly pipeline (unchanged logic)
  core.forecast   - worldmap_forcase's per-country forecasting model
  core.validation - forecast-vs-real-data confidence scoring (new)

No Django wiring belongs here (no apps.py, no models.py, no
INSTALLED_APPS entry). Django apps (ingestion/, forecast/, validation/)
import from this package and expose it via management commands + views.
"""
