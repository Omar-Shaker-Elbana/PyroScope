from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),

    path("", include("dashboard.urls")),
    path("ingestion/", include("ingestion.urls")),
    path("forecast/", include("forecast.urls")),
    path("validation/", include("validation.urls")),
]