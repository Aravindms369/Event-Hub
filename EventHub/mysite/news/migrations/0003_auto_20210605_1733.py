# Generated by Django 2.2.13 on 2021-06-05 12:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('news', '0002_tweet'),
    ]

    operations = [
        migrations.DeleteModel(
            name='tweet',
        ),
        migrations.AddField(
            model_name='location',
            name='text',
            field=models.CharField(max_length=500, null=True),
        ),
    ]
