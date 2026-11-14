"""python manage.py train_countries --modis ... --viirs ... [--out-dir ...]

Forwards every argument after the command name to core.forecast.train_countries's
own argparse (Django's option parser is bypassed), so no `--` separator is needed.
"""
import sys

from django.core.management.base import BaseCommand

from core.forecast import train_countries


class Command(BaseCommand):
    help = "Train the pooled per-country forecasting model (core.forecast.train_countries)."

    def run_from_argv(self, argv):
        sys.argv = [argv[0]] + argv[2:]
        # Django's execute() reads these options, which its own parser would normally supply.
        self.execute(force_color=False, no_color=False, skip_checks=True)

    def handle(self, *args, **options):
        train_countries.main()
