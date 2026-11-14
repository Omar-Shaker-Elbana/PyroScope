from django.http import HttpResponse
from django.shortcuts import render

from core.validation import confidence


def report(request):
    """The confidence report from the original plan's steps 4-6: forecast
    error against real data, alongside whether weak harmonization trust
    explains it. core.validation.confidence.build_report() reads whatever
    `predict_countries` / `harmonize` last wrote to disk -- it runs neither
    pipeline step itself.

    ?countries=EGY,SDN restricts the report to those country codes; left out,
    every country in the current forecast_by_country.csv is reported.
    """
    countries = request.GET.get("countries")
    countries = [c.strip() for c in countries.split(",") if c.strip()] if countries else None

    try:
        data = confidence.build_report(countries=countries)
    except FileNotFoundError as exc:
        return HttpResponse(str(exc), status=501)

    rows = data.to_dict("records")
    summary = {
        "countries": data["country"].nunique(),
        "rows": len(data),
        "scored": int(data["in_band"].notna().sum()),
        "in_band": int(data["in_band"].fillna(False).sum()),
        "flagged": int(data["confidence_flag"].isin(["CAUTION", "LOW_TRUST", "UNKNOWN"]).sum()),
    }
    return render(request, "validation/report.html", {"rows": rows, "summary": summary})
