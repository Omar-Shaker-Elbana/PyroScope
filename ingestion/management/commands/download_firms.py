"""python manage.py download_firms -- --start 2015-01-01 --end 2025-12-31 --bbox ...

core.src.download_firms.main() does its own argparse on sys.argv, so this
command forwards whatever comes after `--` straight through. Needs
FIRMS_MAP_KEY set in the environment.
"""
import sys

from django.core.management.base import BaseCommand

from core.src import download_firms


class Command(BaseCommand):
    help = "Download FIRMS MODIS/VIIRS archive CSVs into data/raw (core.src.download_firms)."

    def run_from_argv(self, argv):
        # Hand everything after `manage.py download_firms` to the real argparse
        # untouched, instead of Django's option parser intercepting it.
        sys.argv = [argv[0]] + argv[2:]
        self.execute()

    def handle(self, *args, **options):
        download_firms.main()
