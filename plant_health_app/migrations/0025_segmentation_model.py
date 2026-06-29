from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('plant_health_app', '0024_predictionhistory_cropped_image_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='SegmentationModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('description', models.TextField(blank=True)),
                ('file_path', models.CharField(max_length=500, unique=True)),
                ('is_active', models.BooleanField(db_index=True, default=True)),
                ('is_default', models.BooleanField(default=False, help_text='Model đang được sử dụng cho segmentation')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Segmentation Model',
                'verbose_name_plural': 'Segmentation Models',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddField(
            model_name='predictionhistory',
            name='segmentation_model',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='predictions', to='plant_health_app.segmentationmodel'),
        ),
    ]
