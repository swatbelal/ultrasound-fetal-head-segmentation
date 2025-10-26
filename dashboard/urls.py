from django.urls import path, include

urlpatterns = [
    path('', include('inference_app.urls')),
]
