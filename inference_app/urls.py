from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_view, name='upload'),
    path('gallery/', views.gallery, name='gallery'),
    path('run_eval/', views.run_eval, name='run_eval'),
]
