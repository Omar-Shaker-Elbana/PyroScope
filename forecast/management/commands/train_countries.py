"""python manage.py train_countries -- --modis ... --viirs ... --out-dir ...

Forwards everything after `--` to core.forecast.train_countries's own
argparse, same pattern as ingestion's download_firms command.
"""
import sys

from django.core.management.base import BaseCommand

from core.forecast import train_countries


class Command(BaseCommand):
    help = "Train the pooled per-country forecasting model (core.forecast.train_countries)."

    def run_from_argv(self, argv):
        sys.argv = [argv[0]] + argv[2:]
        self.execute()

    def handle(self, *args, **options):
        train_countries.main()
