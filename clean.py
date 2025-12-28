import pandas as pd
import os

# 1. Cấu hình đường dẫn dựa trên ảnh thực tế của bạn
DATA_DIR = 'data'
RAW_IMAGES_DIR = os.path.join(DATA_DIR, 'raw_images')
KEYPOINTS_DIR = os.path.join(DATA_DIR, 'keypoints')

print("🧹 ĐANG ĐỒNG BỘ DỮ LIỆU (CSV <-> IMAGE)...")
print("===========================================")

# Lấy danh sách các lớp dựa trên các file CSV trong thư mục keypoints
csv_files = [f for f in os.listdir(KEYPOINTS_DIR) if f.endswith('.csv')]

for csv_file in csv_files:
    # Tên lớp (ví dụ: 'A' từ 'A.csv')
    label_name = os.path.splitext(csv_file)[0]
    csv_path = os.path.join(KEYPOINTS_DIR, csv_file)
    
    # Đường dẫn tới thư mục ảnh tương ứng
    image_folder_path = os.path.join(RAW_IMAGES_DIR, label_name)
    
    # Kiểm tra xem thư mục ảnh có tồn tại không
    if not os.path.exists(image_folder_path):
        print(f"⚠️ Bỏ qua lớp {label_name}: Không tìm thấy thư mục ảnh tại {image_folder_path}")
        continue
        
    print(f"\n📂 Đang xử lý lớp: {label_name}")
    
    try:
        # Đọc file CSV
        df = pd.read_csv(csv_path)
        original_count = len(df)
        
        # Lấy danh sách ảnh thực tế đang có
        existing_images = set([f for f in os.listdir(image_folder_path) if f.endswith('.jpg')])
        
        cleaned_data = []
        
        # Duyệt qua từng dòng trong CSV để đối chiếu với ảnh
        for index, row in df.iterrows():
            # Tên file ảnh kỳ vọng (theo logic code cũ: Label_Index.jpg)
            expected_img_name = f"{label_name}_{index}.jpg"
            
            if expected_img_name in existing_images:
                cleaned_data.append(row)
        
        # Lưu lại file CSV nếu có thay đổi
        if len(cleaned_data) > 0:
            new_df = pd.DataFrame(cleaned_data)
            new_df.to_csv(csv_path, index=False)
            
            deleted_count = original_count - len(new_df)
            print(f"   ✅ Giữ lại: {len(new_df)} dòng")
            if deleted_count > 0:
                print(f"   🗑️ Đã xóa: {deleted_count} dòng rác (không tìm thấy file ảnh tương ứng)")
        else:
            print(f"   ❌ Cảnh báo: Lớp {label_name} không còn dữ liệu nào khớp giữa CSV và Ảnh!")
            
    except Exception as e:
        print(f"   ❌ Lỗi xử lý lớp {label_name}: {e}")

print("\n===========================================")
print("🎉 ĐÃ XONG! Dữ liệu tại thư mục 'keypoints' đã khớp với 'raw_images'.")