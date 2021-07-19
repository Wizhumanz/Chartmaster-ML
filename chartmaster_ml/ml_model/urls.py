from django.urls import path
from . import views
urlpatterns = [
    path('', views.apiOverview, name="api-overview"),
    path('model-name/', views.modelName, name="model-name"),
    path('send-input/', views.sendInput, name="send-input"),
  ]