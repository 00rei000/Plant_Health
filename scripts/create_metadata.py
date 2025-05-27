import os
import pandas as pd

# Cập nhật đường dẫn đến thư mục thực tế trong OneDrive
data_dir = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\dataset_plant\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
folders = ["train", "valid"]

# Kiểm tra data_dir
if not os.path.exists(data_dir):
    print(f"Thư mục không tồn tại: {data_dir}")
    exit()

data = []
for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    # Kiểm tra folder_path
    if not os.path.exists(folder_path):
        print(f"Thư mục không tồn tại: {folder_path}")
        continue
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            plant_type = class_name.split('___')[0]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                data.append({
                    "image_path": img_path,
                    "class_name": class_name,
                    "plant_type": plant_type,
                    "dataset_type": folder
                })

# Tạo CSV
df = pd.DataFrame(data)
# Lưu file CSV vào thư mục của bạn
output_path = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\data\plant_diseases_metadata.csv"
df.to_csv(output_path, index=False)
print("Đã tạo file CSV tại:", output_path)