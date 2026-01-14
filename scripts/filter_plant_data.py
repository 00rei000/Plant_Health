"""
Script để loại bỏ các plant_type chỉ có 1 class khỏi plant_data.csv
Loại bỏ: Blueberry, Orange, Squash, Soybean, Raspberry
"""
import pandas as pd
import os

# Đường dẫn file CSV gốc (đúng)
csv_path = r"C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Django\Demo\mysite\plant_health_app\data\plant_data.csv"

# Đọc CSV
df = pd.read_csv(csv_path)
print(f"Tổng số ảnh ban đầu: {len(df)}")
print(f"Plant types ban đầu: {df['plant_type'].unique()}")

# Danh sách plant_type cần loại bỏ
REMOVE_PLANTS = ['Blueberry', 'Orange', 'Squash', 'Soybean', 'Raspberry']

# Lọc bỏ các plant_type không mong muốn
df_filtered = df[~df['plant_type'].isin(REMOVE_PLANTS)]

print(f"\n=== SAU KHI LỌC ===")
print(f"Tổng số ảnh còn lại: {len(df_filtered)}")
print(f"Plant types còn lại: {sorted(df_filtered['plant_type'].unique())}")
print(f"Số ảnh bị loại bỏ: {len(df) - len(df_filtered)}")

# Thống kê chi tiết
print(f"\n=== THỐNG KÊ CHI TIẾT ===")
for plant in sorted(df_filtered['plant_type'].unique()):
    count = len(df_filtered[df_filtered['plant_type'] == plant])
    diseases = df_filtered[df_filtered['plant_type'] == plant]['disease'].unique()
    print(f"{plant}: {count} ảnh, {len(diseases)} bệnh - {list(diseases)}")

# Lưu file CSV mới
backup_path = csv_path.replace('.csv', '_backup.csv')
filtered_path = csv_path.replace('.csv', '_filtered.csv')

# Backup file gốc
df.to_csv(backup_path, index=False)
print(f"\n✓ Đã backup file gốc: {backup_path}")

# Lưu file đã lọc
df_filtered.to_csv(filtered_path, index=False)
print(f"✓ Đã lưu file đã lọc: {filtered_path}")

# Hỏi có muốn thay thế file gốc không
print("\n" + "="*50)
print("Để thay thế file gốc, chạy lệnh:")
print(f"import shutil; shutil.copy(r'{filtered_path}', r'{csv_path}')")
