# Generated by Django 4.2.13 on 2025-05-25 13:48

from django.db import migrations, models
import plant_health_app.models


class Migration(migrations.Migration):

    dependencies = [
        ('plant_health_app', '0003_alter_diseaselibrary_created_by_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='plantimage',
            name='image',
            field=models.ImageField(max_length=255, upload_to=plant_health_app.models.get_image_path),
        ),
    ]
