"""
Django settings for mythology_chatbot project.
Generated by 'django-admin startproject' using Django 5.1.1.
"""

import os
from pathlib import Path
from pymongo import MongoClient
from corsheaders.defaults import default_headers

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Port Setup
PORT = os.environ.get('PORT', 8000)

# MongoDB Client (optional if using djongo directly in DATABASES)
client = MongoClient('mongodb://localhost:27017/')
db = client['mythology_db']

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-vuoec2z2we+kpv524_aizc=s05!z4(k!v*5f1y@gtlnw(8j9w1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']  # Allow all for dev; restrict in production

# Installed Applications
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'mythology_chatbot',
    'api',
]

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# URL Routing
ROOT_URLCONF = 'mythology_chatbot.urls'

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI
WSGI_APPLICATION = 'mythology_chatbot.wsgi.application'

# Database: Using djongo for MongoDB
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'mahabharata-chatbotdb',
        'CLIENT': {
            'host': 'mongodb://localhost:27017/',
        }
    }
}

# Password Validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static Files
STATIC_URL = '/static/'

# Media Files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default Primary Key Field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework Settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
}

# CORS Configuration
CORS_ALLOW_ALL_ORIGINS = True

CSRF_TRUSTED_ORIGINS = [
    'http://localhost:3000',  # frontend URL
]

CORS_ALLOW_HEADERS = list(default_headers) + [
    'X-CSRFToken',
]

