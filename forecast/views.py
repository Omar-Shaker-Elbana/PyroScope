import csv

from django.conf import settings
from django.shortcuts import render


def forecast_map(request):
    """Render forecast_by_country.csv — the same data the standalone
    HotSync_harmonized_Final.html / scripts.js map showed, now served by
    Django instead of a static file.
    """
    path = settings.DATA_DIR / "outputs" / "forecast_by_country.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    return render(request, "forecast/map.html", {"rows": rows})
