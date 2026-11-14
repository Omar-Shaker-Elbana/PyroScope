"""python manage.py harmonize  ->  core.src.harmonize.main()

Writes calendar_daily.csv, calendar_cells.csv, frp_mapping.json (the k_trust
grades core.validation.confidence.harmonization_confidence() will read) into
PROCESSED_DIR. Requires preprocess to have already run.
"""
from django.core.management.base import BaseCommand

from core.src import harmonize


class Command(BaseCommand):
    help = "Harmonize MODIS + VIIRS into the trust-graded calendar (core.src.harmonize)."

    def handle(self, *args, **options):
        harmonize.main()
