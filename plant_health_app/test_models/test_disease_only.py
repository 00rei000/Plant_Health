"""
Script test chỉ mô hình disease, lấy loại cây từ tên file ảnh (PlantType_xxx.jpg)
"""
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==================== CẤU HÌNH ====================
TEST_DIR = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\test\test"
DISEASE_MODELS_DIR = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\plant_health_app\notebook\plant_disease\models\best"
GROUP_CLASSES_JSON = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\plant_health_app\notebook\plant_disease\group_classes.json"
OUTPUT_CSV = "test_results_disease_only.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# ==================== TRANSFORMS ====================
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== LOAD JSON ====================
with open(GROUP_CLASSES_JSON, 'r', encoding='utf-8') as f:
    GROUPS = json.load(f)
print(f"Số nhóm bệnh: {len(GROUPS)}")

# ==================== LOAD DISEASE MODEL ====================
def load_disease_model(plant_type):
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    num_classes = len(GROUPS.get(plant_type, []))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model_path = os.path.join(DISEASE_MODELS_DIR, f'{plant_type.lower()}_model.pth')
    if not os.path.exists(model_path):
        print(f"⚠ Không tìm thấy disease model cho {plant_type}: {model_path}")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Đã load disease model cho {plant_type}: EfficientNet-B0")
    return model

# ==================== PREDICT FUNCTION ====================
def predict_image(image_path, disease_models_cache):
    """Dự đoán bệnh cho 1 ảnh, loại cây lấy từ tên file."""
    try:
        filename = os.path.basename(image_path)
        plant_type = filename.split('_')[0]
        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        if plant_type not in GROUPS:
            return {
                'plant_type': plant_type,
                'disease': 'N/A',
                'disease_confidence': 0.0,
                'status': 'Plant type not in GROUPS'
            }
        if plant_type not in disease_models_cache:
            disease_models_cache[plant_type] = load_disease_model(plant_type)
        disease_model = disease_models_cache[plant_type]
        if disease_model is None:
            return {
                'plant_type': plant_type,
                'disease': 'N/A',
                'disease_confidence': 0.0,
                'status': 'Disease model not found'
            }
        with torch.no_grad():
            outputs = disease_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            disease = GROUPS[plant_type][predicted.item()]
            disease_confidence = confidence.item() * 100
        return {
            'plant_type': plant_type,
            'disease': disease,
            'disease_confidence': disease_confidence,
            'status': 'Success'
        }
    except Exception as e:
        return {
            'plant_type': 'Error',
            'disease': 'Error',
            'disease_confidence': 0.0,
            'status': str(e)
        }

# ==================== MAIN ====================
def main():
    print("\n" + "="*60)
    print("BẮT ĐẦU TEST DISEASE MODEL (DỰA VÀO TÊN FILE)")
    print("="*60 + "\n")
    if not os.path.exists(TEST_DIR):
        print(f"❌ Không tìm thấy thư mục test: {TEST_DIR}")
        return
    print("\n--- LOAD DISEASE MODELS (on demand) ---")
    disease_models_cache = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    print(f"\n✓ Tìm thấy {len(image_files)} ảnh trong {TEST_DIR}")
    if len(image_files) == 0:
        print("❌ Không có ảnh nào để test!")
        return
    print("\n--- BẮT ĐẦU TEST ---")
    results = []
    for image_path in tqdm(image_files, desc="Testing images"):
        filename = os.path.basename(image_path)
        relative_path = os.path.relpath(image_path, TEST_DIR)
        result = predict_image(image_path, disease_models_cache)
        results.append({
            'Filename': filename,
            'Relative Path': relative_path,
            'Plant Type': result['plant_type'],
            'Disease': result['disease'],
            'Disease Confidence (%)': f"{result['disease_confidence']:.2f}",
            'Status': result['status']
        })
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✓ Đã lưu kết quả vào: {OUTPUT_CSV}")
    print(f"✓ Tổng số ảnh đã test: {len(results)}")
    print("\n--- THỐNG KÊ ---")
    print(f"Số ảnh thành công: {df[df['Status'] == 'Success'].shape[0]}")
    print(f"Số ảnh lỗi: {df[df['Status'] != 'Success'].shape[0]}")
    print("\n--- PHÂN BỐ LOẠI CÂY ---")
    print(df['Plant Type'].value_counts())
    print("\n--- MẪU KẾT QUẢ (5 ảnh đầu) ---")
    print(df.head().to_string(index=False))
    print("\n" + "="*60)
    print("HOÀN TẤT!")
    print("="*60)

if __name__ == "__main__":
    main()
