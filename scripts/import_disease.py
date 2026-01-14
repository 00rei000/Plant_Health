import pandas as pd

# Đọc file CSV
csv_path = r'C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\plant_data.csv'
df = pd.read_csv(csv_path)

# Tạo cột disease mới bằng cách kết hợp plant_type và disease
df['disease'] = df['plant_type'] + '___' + df['disease']

# Chuẩn hóa đường dẫn ảnh (thay \ bằng /)
df['image'] = df['image'].apply(lambda x: x.replace('\\', '/'))

# Lưu lại file CSV
df.to_csv(csv_path, index=False)
print("Đã cập nhật cột disease và lưu file CSV:")
print(df.head())