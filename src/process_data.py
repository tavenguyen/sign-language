import pandas as pd
import os
import json
import glob
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

# Cấu hình logging: Gọn gàng, dễ theo dõi
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
#                           CẤU HÌNH (CONFIGURATION)
# =========================================================================
@dataclass
class Config:
    # Danh sách nhãn (Thứ tự quyết định ID: 0, 1, 2...)
    LABELS: Tuple[str, ...] = (
        'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y',
        'NOTHING'
    )
    
    # Đường dẫn thư mục
    INPUT_DIR: str = os.path.join('data', 'keypoints')
    OUTPUT_DIR: str = 'dataset'
    
    # Tên file kết quả duy nhất
    FINAL_FILE: str = 'keypoint.csv'
    LABEL_MAP_FILE: str = 'label_map.json'
    
    RANDOM_STATE: int = 42

# =========================================================================
#                           HÀM XỬ LÝ (HELPER FUNCTIONS)
# =========================================================================

def load_and_merge_data(config: Config) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Quét tất cả file CSV của từng thành viên (VD: A_Tuan.csv, A_Vy.csv),
    gộp lại và gán nhãn số (Label Encoding).
    """
    data_frames = []
    label_map = {}
    total_count = 0

    logger.info("Bat dau quet du lieu tu TeamWork...")

    for idx, label in enumerate(config.LABELS):
        label_map[idx] = label
        
        # Tạo mẫu tìm kiếm: data/keypoints/A_*.csv
        search_pattern = os.path.join(config.INPUT_DIR, f"{label}_*.csv")
        files = glob.glob(search_pattern)

        if not files:
            logger.warning("Bo qua: Khong tim thay bat ky file nao cho nhan '%s'", label)
            continue

        label_count = 0
        for file_path in files:
            try:
                # Đọc CSV, header=0 để loại bỏ dòng tiêu đề 'label,v0,v1...'
                df = pd.read_csv(file_path, header=0)

                if df.empty: continue

                # Thay thế cột nhãn (string) bằng ID (int)
                df.iloc[:, 0] = idx
                
                data_frames.append(df)
                label_count += len(df)
                
                # In tên file ngắn gọn để dễ nhìn
                fname = os.path.basename(file_path)
                logger.info(f"   + Load: {fname:<25} | Size: {len(df)}")

            except Exception as e:
                logger.error("Loi doc file %s: %s", file_path, str(e))
        
        total_count += label_count

    if not data_frames:
        raise ValueError("Khong tim thay du lieu hop le de xu ly.")

    # Gộp toàn bộ dữ liệu thành một DataFrame lớn
    full_df = pd.concat(data_frames, ignore_index=True)
    logger.info("-" * 40)
    logger.info("TONG CONG: %d mau du lieu.", total_count)
    
    return full_df, label_map

def save_final_dataset(df: pd.DataFrame, label_map: Dict[int, str], config: Config):
    """
    Xáo trộn dữ liệu và lưu ra một file CSV duy nhất.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Xáo trộn dữ liệu ngẫu nhiên (Shuffle)
    # Giúp model học tốt hơn, không bị bias theo thứ tự
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    try:
        # Đường dẫn file đầu ra
        final_path = os.path.join(config.OUTPUT_DIR, config.FINAL_FILE)
        map_path = os.path.join(config.OUTPUT_DIR, config.LABEL_MAP_FILE)

        # Lưu CSV (header=False để tiện cho việc load vào Model sau này)
        # Format: [Label_ID, x1, y1, ..., x21, y21]
        df.to_csv(final_path, header=False, index=False)

        # Lưu file JSON ánh xạ nhãn (0 -> 'A')
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=4)

        logger.info("Xu ly hoan tat!")
        logger.info("Dataset merged: %s", final_path)
        logger.info("Label mapping:  %s", map_path)

    except Exception as e:
        logger.error("Loi khi luu file: %s", str(e))

# =========================================================================
#                           CHƯƠNG TRÌNH CHÍNH (MAIN)
# =========================================================================
def main():
    config = Config()
    
    try:
        full_df, label_map = load_and_merge_data(config)
        
        # 2. Lưu ra file tổng
        save_final_dataset(full_df, label_map, config)
        
    except Exception as e:
        logger.critical("Chuong trinh dung do loi: %s", str(e))

if __name__ == "__main__":
    main()