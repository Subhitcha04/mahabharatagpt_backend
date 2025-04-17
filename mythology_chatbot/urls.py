from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from api.views import (
    home_view, register, login_view, save_query, get_user_queries, get_logged_in_username,handle_query
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),  # Home view
    path('register/', register, name='register'),  
    path('login/', login_view, name='login'), 
    path('handle_query/', handle_query, name='handle_query'),
    path('save_query/', save_query, name='save_query'),  # Save user query
    path('user_queries/<str:username>/', get_user_queries, name='get_user_queries'),  # Fetch queries by user
    path('get-username/', get_logged_in_username, name='get_logged_in_username'),  # Fetch logged-in user's username
]

# Serving media files during development when DEBUG=True
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
