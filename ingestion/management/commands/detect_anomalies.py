"""python manage.py detect_anomalies  ->  core.src.anomaly.main()

Named detect_anomalies (not anomaly) to avoid shadowing the core.src.anomaly
module name on sys.path. Requires harmonize to have already run.
"""
from django.core.management.base import BaseCommand

from core.src import anomaly


class Command(BaseCommand):
    help = "Leave-one-year-out anomaly detection + critical windows (core.src.anomaly)."

    def handle(self, *args, **options):
        anomaly.main()
