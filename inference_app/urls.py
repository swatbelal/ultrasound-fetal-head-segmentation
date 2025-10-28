from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_view, name='upload'),
    path('gallery/', views.gallery, name='gallery'),
    path('upload_gallery/', views.upload_gallery, name='upload_gallery'),
    path("run_eval_stream/", views.run_eval_stream, name="run_eval_stream"),
    path("status_eval/", views.status_eval, name="status_eval"),
    path("view_log/", views.view_log, name="view_log"),
    path("cancel_eval/", views.cancel_eval, name="cancel_eval"),
]
