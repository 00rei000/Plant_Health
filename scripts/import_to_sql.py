import pyodbc
import pandas as pd
import os
from tqdm import tqdm  # pip install tqdm nếu chưa có

# Thiết lập kết nối SQL Server
server = 'DESKTOP-NDJJABF\SQLEXPRESS'
database = 'plant_disease'
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# Đường dẫn file CSV
csv_file_path = r'C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\plant_data.csv'

# Đường dẫn thư mục chứa ảnh
media_root = r'C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\media'

try:
    # Đọc file CSV
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    total_rows = len(df)
    print(f"Tổng số dòng CSV: {total_rows}")

    # Kết nối SQL Server
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    # Chuẩn bị danh sách bản ghi
    records = []
    successful = 0
    skipped = 0

    # Hiển thị tiến độ
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Importing"):
        image_path = row['image']  # plant_images/images/train/...
        image_path = image_path.replace('\\', '/').strip()
        
        # Kiểm tra file ảnh
        full_image_path = os.path.join(media_root, image_path)
        if not os.path.exists(full_image_path):
            skipped += 1
            if skipped <= 10:  # Chỉ in 10 lỗi đầu
                print(f"Image not found at row {index + 2}: {full_image_path}")
            continue

        # Chuẩn hóa đường dẫn cho SQL Server
        image_path_sql = image_path.replace('/', '\\')

        records.append((image_path_sql, row['disease'], row['plant_type'], row['dataset_type']))
        successful += 1

    # Insert hàng loạt
    if records:
        cursor.executemany(
            """
            INSERT INTO plant_health_app_plantmodel (image, disease, plant_type, dataset_type)
            VALUES (?, ?, ?, ?)
            """,
            records
        )
        conn.commit()
        print(f"\n✅ Import thành công {len(records)} records!")
        print(f"📊 Tiến độ: {successful}/{total_rows} thành công, {skipped} bị bỏ qua")
    else:
        print("❌ Không có records hợp lệ để import.")

except Exception as e:
    print(f"❌ Lỗi khi import: {e}")

finally:
    cursor.close()
    conn.close()