from plant_health_app.models import DiseaseLibrary
DiseaseLibrary.objects.all().delete()
print("Đã xóa tất cả dữ liệu trong DiseaseLibrary.")

from plant_health_app.models import DiseaseLibrary
print(DiseaseLibrary.objects.all().count())
for disease in DiseaseLibrary.objects.all():
    print(f"{disease.name}: {disease.image.name if disease.image else 'No image'}")