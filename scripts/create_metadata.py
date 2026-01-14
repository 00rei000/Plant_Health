import pyodbc
import pandas as pd
import os

# Thiết lập kết nối SQL Server
server = 'DESKTOP-NDJJABF\SQLEXPRESS'
database = 'plant_disease'
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

try:
    # Kết nối
    conn = pyodbc.connect(connection_string)
    
    # Truy vấn dữ liệu
    query = "SELECT image, disease, plant_type, dataset_type FROM plant_health_app_plantmodel WHERE dataset_type = 'train' OR dataset_type = 'valid'"
    df = pd.read_sql(query, conn)
    
    # Đóng kết nối
    conn.close()
    
    # Chuẩn hóa đường dẫn ảnh (thay \ thành / để tương thích với Colab)
    df['image'] = df['image'].apply(lambda x: x.replace('\\', '/'))
    
    # Lưu dữ liệu vào file CSV
    output_path = 'plant_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dữ liệu đã được xuất ra file: {output_path}")
    
    # Kiểm tra nội dung file CSV
    print("Dữ liệu mẫu từ file CSV:")
    print(df.head())

except Exception as e:
    print(f"Lỗi khi xuất dữ liệu: {e}")