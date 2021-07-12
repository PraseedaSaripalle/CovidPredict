from django.urls import path
from projects import views

urlpatterns = [
    path("projects/", views.project_index, name='project_index'),
    path("covid/", views.home_view, name='home_view'),
    path("corona_view/", views.corona_view, name='corona_view'),
]