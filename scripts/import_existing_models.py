"""
Script to import existing model files into PlantTypeModel and DiseaseModel tables.

Usage in Django shell:
    python manage.py shell
    
Then paste:
    exec(open('scripts/import_existing_models.py').read())
"""

import os
from django.conf import settings
from django.contrib.auth.models import User
from plant_health_app.models import PlantTypeModel, DiseaseModel

def import_models():
    """Import existing models from notebook/*/models/best/ into database."""
    
    # Get admin user (or create one)
    admin_user = User.objects.filter(is_staff=True).first()
    if not admin_user:
        print("Warning: No admin user found. Models will be created without creator.")
    
    print("=" * 60)
    print("IMPORTING EXISTING MODELS TO DATABASE")
    print("=" * 60)
    
    # ==================== PLANT TYPE MODEL ====================
    plant_type_path = os.path.join(
        settings.BASE_DIR,
        'plant_health_app',
        'notebook',
        'plant_type',
        'models',
        'best',
        'plant_type_model.pth'
    )
    
    if os.path.exists(plant_type_path):
        # Check if already exists
        existing = PlantTypeModel.objects.filter(
            name="EfficientNet-B0 (Original)"
        ).first()
        
        if not existing:
            plant_model = PlantTypeModel.objects.create(
                name="EfficientNet-B0 (Original)",
                description="Model gốc đã train sẵn từ notebook",
                architecture="EfficientNet-B0",
                optimizer="AdamW",  # Default, có thể sửa
                file_path="plant_health_app/notebook/plant_type/models/best/plant_type_model.pth",
                num_classes=9,
                is_default=True,
                is_active=True,
                created_by=admin_user
            )
            print(f"✓ Created Plant Type Model: {plant_model.name}")
        else:
            print(f"⊘ Plant Type Model already exists: {existing.name}")
    else:
        print(f"✗ Plant Type Model file not found: {plant_type_path}")
    
    print()
    
    # ==================== DISEASE MODELS ====================
    plant_types = [
        'Apple', 'Cherry', 'Corn', 'Grape', 
        'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato'
    ]
    
    disease_base_path = os.path.join(
        settings.BASE_DIR,
        'plant_health_app',
        'notebook',
        'plant_disease',
        'models',
        'best'
    )
    
    for plant_type in plant_types:
        model_file = f"{plant_type.lower()}_model.pth"
        model_path = os.path.join(disease_base_path, model_file)
        
        if os.path.exists(model_path):
            # Check file size to see if it's a real model or placeholder
            file_size = os.path.getsize(model_path)
            
            if file_size < 1000:  # Less than 1KB = likely placeholder text
                print(f"⊘ {plant_type}: Placeholder file detected, skipping")
                continue
            
            # Check if already exists
            existing = DiseaseModel.objects.filter(
                plant_type=plant_type,
                name=f"{plant_type} EfficientNet-B0 (Original)"
            ).first()
            
            if not existing:
                disease_model = DiseaseModel.objects.create(
                    name=f"{plant_type} EfficientNet-B0 (Original)",
                    description=f"Model gốc đã train cho {plant_type}",
                    plant_type=plant_type,
                    architecture="EfficientNet-B0",
                    optimizer="AdamW",  # Default
                    file_path=f"plant_health_app/notebook/plant_disease/models/best/{model_file}",
                    is_default=True,
                    is_active=True,
                    created_by=admin_user
                )
                print(f"✓ Created {plant_type} Disease Model: {disease_model.name}")
            else:
                print(f"⊘ {plant_type} Disease Model already exists: {existing.name}")
        else:
            print(f"✗ {plant_type}: Model file not found at {model_path}")
    
    print()
    print("=" * 60)
    print("IMPORT COMPLETED!")
    print("=" * 60)
    
    # Summary
    plant_count = PlantTypeModel.objects.filter(is_active=True).count()
    disease_count = DiseaseModel.objects.filter(is_active=True).count()
    
    print(f"\nTotal Plant Type Models: {plant_count}")
    print(f"Total Disease Models: {disease_count}")
    print(f"\nDefault Plant Type Model: {PlantTypeModel.objects.filter(is_default=True).first()}")
    print(f"\nDefault Disease Models:")
    for plant_type in plant_types:
        default = DiseaseModel.objects.filter(
            plant_type=plant_type,
            is_default=True
        ).first()
        if default:
            print(f"  - {plant_type}: {default.name}")

# Auto-run when script is executed
import_models()
