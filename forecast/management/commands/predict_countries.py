"""python manage.py predict_countries -- --modis ... --viirs ... --model ...

Forwards everything after `--` to core.forecast.predict_countries's own
argparse. See core/forecast/predict_countries.py — it's a reconstruction,
verify it against your original script if you still have one.
"""
import sys

from django.core.management.base import BaseCommand

from core.forecast import predict_countries


class Command(BaseCommand):
    help = "Write a fresh forecast_by_country.csv from the trained model (core.forecast.predict_countries)."

    def run_from_argv(self, argv):
        sys.argv = [argv[0]] + argv[2:]
        self.execute()

    def handle(self, *args, **options):
        predict_countries.main()
