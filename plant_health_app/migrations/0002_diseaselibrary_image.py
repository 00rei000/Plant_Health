# Generated by Django 4.2.13 on 2025-05-25 12:28

from django.db import migrations, models
import plant_health_app.models


class Migration(migrations.Migration):

    dependencies = [
        ('plant_health_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='diseaselibrary',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to=plant_health_app.models.get_image_path),
        ),
    ]
