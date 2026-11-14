from django.urls import path

from . import views

app_name = "forecast"
urlpatterns = [
    path("map/", views.forecast_map, name="map"),
]
