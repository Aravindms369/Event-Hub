from django.db import models
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from django_random_queryset import RandomManager

# Create your models here.

 
class location(models.Model):
    objects=RandomManager()

    country = models.CharField(max_length=200)
    place = models.CharField(max_length=200)
    lat = models.FloatField(max_length=100)
    lon = models.FloatField(max_length=100)
    text = models.CharField(max_length=500, null=True)
    event = models.CharField(max_length=20, null=True)