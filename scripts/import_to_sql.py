import os
import pyodbc
from datetime import datetime

# Đường dẫn thư mục media
media_root = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\media\plant_images"

def extract_plant_and_disease(disease_name):
    """
    Tách plant_type và disease từ tên thư mục.
    Nếu không có '__', đặt plant_type là 'Unknown'.
    """
    if '__' in disease_name:
        plant_type, disease = disease_name.split('__', 1)
        disease = disease.replace('_', ' ').lower()  # Chuẩn hóa disease
    else:
        plant_type = 'Unknown'
        disease = disease_name.replace('_', ' ').lower()
    return plant_type, disease

def get_image_info(image_path, disease_name, dataset_type):
    """
    Tạo thông tin ảnh từ đường dẫn, tên bệnh, và loại dữ liệu.
    """
    # Tách plant_type và disease
    plant_type, disease = extract_plant_and_disease(disease_name)
    # Tạo đường dẫn tương đối
    relative_path = os.path.join('images', dataset_type, disease_name, os.path.basename(image_path))
    return (relative_path, disease, plant_type, dataset_type)

def scan_images(data_dir, dataset_type):
    """
    Quét thư mục để lấy danh sách ảnh và nhãn bệnh.
    """
    image_info = []
    # Quét các thư mục disease
    for disease_name in os.listdir(data_dir):
        disease_dir = os.path.join(data_dir, disease_name)
        if not os.path.isdir(disease_dir):
            continue
        # Quét các file ảnh trong thư mục disease
        for img_name in os.listdir(disease_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.JPG'):
                img_path = os.path.join(disease_dir, img_name)
                image_info.append((img_path, disease_name, dataset_type))
    return image_info

def import_to_sql():
    # Đường dẫn thư mục train và valid
    train_dir = os.path.join(media_root, 'images', 'train')
    valid_dir = os.path.join(media_root, 'images', 'valid')

    # Kiểm tra thư mục
    if not os.path.exists(train_dir):
        print(f"Thư mục train không tồn tại: {train_dir}")
        return
    if not os.path.exists(valid_dir):
        print(f"Thư mục valid không tồn tại: {valid_dir}")
        return

    # Quét dữ liệu train và valid
    train_images = scan_images(train_dir, 'train')
    valid_images = scan_images(valid_dir, 'valid')
    all_images = train_images + valid_images

    print(f"Tổng số ảnh tìm thấy: {len(all_images)} (train: {len(train_images)}, valid: {len(valid_images)})")

    # Kết nối SQL Server
    server = 'DESKTOP-NDJJABF\SQLEXPRESS'
    database = 'plant_disease'
    connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Chèn dữ liệu theo batch
        batch_size = 1000
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            # Tạo thông tin cho batch
            formatted_batch = [get_image_info(img_path, disease_name, dataset_type) for img_path, disease_name, dataset_type in batch]
            
            cursor.executemany(
                """
                INSERT INTO plant_health_app_plantmodel (image, disease, plant_type, dataset_type)
                VALUES (?, ?, ?, ?)
                """,
                formatted_batch
            )
            conn.commit()
            print(f"Đã chèn {i + len(batch)} bản ghi")

        print("Dữ liệu đã được nhập vào SQL Server!")

    except Exception as e:
        print(f"Lỗi: {e}")

    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    import_to_sql()