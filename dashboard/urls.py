from django.urls import path
from . import views
from . import api_views

app_name = "dashboard"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("dashboard/", views.workspace, name="workspace"),
    path(
    "api/dashboard/status/",
    api_views.api_status,
    name="api-status",
),

path(
    "api/dashboard/map/",
    api_views.api_map,
    name="api-map",
),

path(
    "api/dashboard/region/",
    api_views.api_region,
    name="api-region",
),

path(
    "api/dashboard/forecast/",
    api_views.api_forecast,
    name="api-forecast",
),
]