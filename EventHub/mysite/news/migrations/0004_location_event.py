# Generated by Django 2.2.13 on 2021-06-06 16:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('news', '0003_auto_20210605_1733'),
    ]

    operations = [
        migrations.AddField(
            model_name='location',
            name='event',
            field=models.CharField(max_length=20, null=True),
        ),
    ]