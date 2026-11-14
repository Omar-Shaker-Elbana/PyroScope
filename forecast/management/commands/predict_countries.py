"""python manage.py predict_countries --modis ... --viirs ... [--live-key KEY]

Forwards every argument after the command name to core.forecast.predict_countries's
own argparse (Django's option parser is bypassed), so no `--` separator is needed.
"""
import sys

from django.core.management.base import BaseCommand

from core.forecast import predict_countries


class Command(BaseCommand):
    help = "Write a fresh forecast_by_country.csv from the trained model (core.forecast.predict_countries)."

    def run_from_argv(self, argv):
        sys.argv = [argv[0]] + argv[2:]
        # Django's execute() reads these options, which its own parser would normally supply.
        self.execute(force_color=False, no_color=False, skip_checks=True)

    def handle(self, *args, **options):
        predict_countries.main()
