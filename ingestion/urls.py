from django.urls import path

from . import views

app_name = "ingestion"
urlpatterns = [
    path("status/", views.status, name="status"),
]
