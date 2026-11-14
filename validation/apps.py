from django.apps import AppConfig


class ValidationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "validation"
    verbose_name = "Validation (forecast-vs-real-data confidence, core.validation wrapper)"
