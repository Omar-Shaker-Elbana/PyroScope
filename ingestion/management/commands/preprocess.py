"""python manage.py preprocess  ->  core.src.preprocess.main()

All the real options (PYRO_BBOX, PYRO_START/END, PYRO_AOI_FILE, ...) are
still env vars read by core.src.config — this command does not re-expose
them as Django args, it just gives the step a manage.py entry point.
"""
from django.core.management.base import BaseCommand

from core.src import preprocess


class Command(BaseCommand):
    help = "Standardize raw FIRMS CSVs (core.src.preprocess). Reads PYRO_* env vars."

    def handle(self, *args, **options):
        preprocess.main()
