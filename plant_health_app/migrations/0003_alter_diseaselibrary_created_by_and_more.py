# Generated by Django 4.2.13 on 2025-05-25 12:43

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import plant_health_app.models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('plant_health_app', '0002_diseaselibrary_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diseaselibrary',
            name='created_by',
            field=models.ForeignKey(db_column='created_by_id', null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='diseaselibrary',
            name='image',
            field=models.ImageField(blank=True, max_length=500, null=True, upload_to=plant_health_app.models.get_image_path),
        ),
        migrations.AlterModelTable(
            name='diseaselibrary',
            table='plant_health_app_diseaselibrary',
        ),
    ]
