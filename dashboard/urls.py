from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

urlpatterns = [
    path('', include('inference_app.urls')),
]

# Serve /media/ -> MEDIA_ROOT
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Serve /results/ -> RESULTS_ROOT
urlpatterns += static(settings.RESULTS_URL, document_root=settings.RESULTS_ROOT)
