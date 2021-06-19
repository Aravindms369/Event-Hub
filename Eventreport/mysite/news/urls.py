from django.urls import path,include
from . import views

from django.conf import settings
from django.conf.urls.static import static



app_name = "news"
urlpatterns = [
    path('', views.report, name='report'),
    path('report/',views.report,name="report"),


]
