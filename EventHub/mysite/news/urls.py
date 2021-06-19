from django.urls import path,include
from . import views

from django.conf import settings
from django.conf.urls.static import static



app_name = "news"
urlpatterns = [
    path('', views.home, name='home'),
    path('map', views.map, name='map'),
    path("logout/", views.logout_request, name="logout"),
    path('loginform/', views.loginform, name='loginform'),
    path('signupform/', views.signupform, name='signupform'),
    path('userhome/',views.userhome,name="userhome"),
    path('usermap/',views.usermap,name="usermap"),
    path('chat/',include('custom_app.urls')),
    path('analysis/',views.analysis,name="analysis"),
    path('report/',views.report,name="report"),


]
