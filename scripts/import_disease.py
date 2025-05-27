import os
import pyodbc
from datetime import datetime

# Đường dẫn thư mục media
media_root = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\media"

def extract_disease(disease_name):
    """
    Tách tên bệnh từ tên thư mục.
    Loại bỏ tiền tố (nếu có) và chuẩn hóa tên.
    """
    disease = disease_name.split('___')[-1] if '___' in disease_name else disease_name
    disease = disease.replace('_', ' ').lower()
    return disease

def get_image_info(disease_name, image_path, dataset_type):
    """
    Tạo thông tin bệnh từ tên bệnh và đường dẫn ảnh.
    """
    disease = extract_disease(disease_name)
    # Tạo đường dẫn tương đối đúng định dạng plant_images/images/valid/<disease_folder>/<filename>
    relative_path = os.path.join('plant_images', 'images', dataset_type, disease_name, os.path.basename(image_path))
    description = "No description available."
    symptoms = "No symptoms available."
    treatment = "No treatment available."
    return (disease, description, symptoms, treatment, None, datetime.now(), relative_path)

def scan_diseases(data_dir, dataset_type):
    """
    Quét thư mục để lấy danh sách bệnh và chọn một ảnh đại diện cho mỗi bệnh.
    """
    disease_info = []
    for disease_name in os.listdir(data_dir):
        disease_dir = os.path.join(data_dir, disease_name)
        if not os.path.isdir(disease_dir):
            continue
        for img_name in os.listdir(disease_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(disease_dir, img_name)
                disease_info.append((disease_name, img_path, dataset_type))
                break  # Chỉ lấy ảnh đầu tiên làm đại diện
    return disease_info

def import_to_sql():
    # Đường dẫn thư mục valid
    valid_dir = os.path.join(media_root, 'plant_images', 'images', 'valid')

    # Kiểm tra thư mục
    if not os.path.exists(valid_dir):
        print(f"Thư mục valid không tồn tại: {valid_dir}")
        return

    # Quét dữ liệu valid
    valid_diseases = scan_diseases(valid_dir, 'valid')
    all_diseases = valid_diseases

    if not all_diseases:
        print("Không tìm thấy bệnh nào trong thư mục.")
        return

    print(f"Tổng số bệnh tìm thấy: {len(all_diseases)}")

    # Kết nối SQL Server
    server = 'DESKTOP-NDJJABF\SQLEXPRESS'
    database = 'plant_disease'
    connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Kiểm tra bảng plant_health_app_diseaselibrary
        cursor.execute("SELECT OBJECT_ID('plant_health_app_diseaselibrary', 'U')")
        if cursor.fetchone()[0] is None:
            print("Bảng plant_health_app_diseaselibrary không tồn tại. Vui lòng chạy 'python manage.py migrate' để tạo bảng.")
            return

        # Xóa dữ liệu cũ trong bảng
        cursor.execute("DELETE FROM plant_health_app_diseaselibrary")
        conn.commit()
        print("Đã xóa dữ liệu cũ trong bảng plant_health_app_diseaselibrary.")

        # Chèn dữ liệu theo batch
        batch_size = 100
        for i in range(0, len(all_diseases), batch_size):
            batch = all_diseases[i:i + batch_size]
            formatted_batch = [get_image_info(disease_name, img_path, dataset_type) for disease_name, img_path, dataset_type in batch]
            
            cursor.executemany(
                """
                INSERT INTO plant_health_app_diseaselibrary (name, description, symptoms, treatment, created_by_id, created_at, image)
                VALUES (?, ?, ?, ?, ?, ?, ?)
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