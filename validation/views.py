from django.http import HttpResponse
from django.shortcuts import render

from core.validation import confidence


def report(request):
    """The confidence report from the original plan's steps 4-6: forecast
    error against real data, alongside whether weak harmonization trust
    explains it. core.validation.confidence isn't implemented yet (every
    function there raises NotImplementedError) — this view exists so the
    URL/template wiring is in place once it is.
    """
    try:
        data = confidence.confidence_report(backtest=None, confidence=None)  # TODO: real inputs
    except NotImplementedError:
        return HttpResponse(
            "Validation logic not implemented yet — see core/validation/confidence.py "
            "(plan steps 4-6).",
            status=501,
        )
    return render(request, "validation/report.html", {"data": data})
