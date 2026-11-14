import json

from django.conf import settings
from django.shortcuts import render


def status(request):
    """Show the last harmonization run's k_trust grades + coverage, read
    straight off disk (no model import needed here — ingestion only
    displays what core.src.harmonize already wrote out).
    """
    mapping_path = settings.DATA_DIR / "processed" / "frp_mapping.json"
    frp_mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else None
    return render(request, "ingestion/status.html", {"frp_mapping": frp_mapping})
