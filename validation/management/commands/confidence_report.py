"""python manage.py confidence_report [--countries EGY,SDN] [--forecast-csv PATH] [--out PATH]

Runs core.validation.confidence.build_report() and writes confidence_report.csv into
core.src.config.OUTPUTS_DIR (next to forecast_by_country.csv), the way predict_countries writes
forecast_by_country.csv and harmonize writes frp_mapping.json: a plain file the `validation` app's
view then just reads/recomputes on request, and that a cron job can run on a schedule.
"""
from django.core.management.base import BaseCommand, CommandError

from core.src.config import OUTPUTS_DIR
from core.validation import confidence


class Command(BaseCommand):
    help = "Backtest the forecast against harmonized real-world activity (core.validation.confidence)."

    def add_arguments(self, parser):
        parser.add_argument("--countries", help="Comma-separated country codes; default: every country "
                                                  "in forecast_by_country.csv")
        parser.add_argument("--forecast-csv", default=None, help="default: OUTPUTS_DIR/forecast_by_country.csv")
        parser.add_argument("--out", default=None, help="default: OUTPUTS_DIR/confidence_report.csv")

    def handle(self, *args, **options):
        countries = [c.strip() for c in options["countries"].split(",") if c.strip()] if options["countries"] else None
        try:
            report = confidence.build_report(countries=countries, forecast_csv=options["forecast_csv"])
        except (FileNotFoundError, ValueError) as exc:
            raise CommandError(str(exc))

        out = options["out"] or (OUTPUTS_DIR / "confidence_report.csv")
        report.to_csv(out, index=False)

        scored = report["in_band"].notna()
        flagged = report["confidence_flag"].isin(["CAUTION", "LOW_TRUST", "UNKNOWN"])
        overall = confidence.backtest_metrics(report).iloc[0]
        self.stdout.write(
            f"{report['country'].nunique()} countries, {len(report)} rows, "
            f"{int(scored.sum())} scored against observed activity so far "
            f"({int((report.loc[scored, 'in_band']).sum())} inside the 10-90% band), "
            f"{int(flagged.sum())} rows flagged for weak harmonization trust.\n"
            f"MAE={overall['mae']:.3g}  RMSE={overall['rmse']:.3g}  "
            f"coverage={overall['coverage']:.1%}  (n={int(overall['n_scored'])})\n"
            f"Saved {out}"
        )
