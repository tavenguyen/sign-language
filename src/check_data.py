import csv
import cv2 as cv
import numpy as np
import os

LABEL_TO_CHECK = 'B1'  # Thay đổi chữ cái 
DATA_PATH = f'data/keypoints/{LABEL_TO_CHECK}.csv'

def draw_skeleton(image, landmarks):
    # Các cặp điểm nối với nhau tạo thành bàn tay (MediaPipe standard)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Ngón cái
        (0, 5), (5, 6), (6, 7), (7, 8),       # Ngón trỏ
        (5, 9), (9, 10), (10, 11), (11, 12),  # Ngón giữa
        (9, 13), (13, 14), (14, 15), (15, 16),# Ngón áp út
        (13, 17), (17, 18), (18, 19), (19, 20),# Ngón út
        (0, 17) # Cổ tay nối ngón út
    ]
    
    h, w, _ = image.shape
    
    scale = 200 
    offset_x, offset_y = w // 2, h // 2  # Đưa về giữa màn hình

    points = []
    # landmarks là list [x1, y1, x2, y2...]
    for i in range(0, len(landmarks), 2):
        x = float(landmarks[i])
        y = float(landmarks[i+1])
        
        # Denormalize để hiển thị
        px = int(x * scale + offset_x)
        py = int(y * scale + offset_y)
        points.append((px, py))
        
        cv.circle(image, (px, py), 3, (0, 0, 255), -1) # Vẽ khớp đỏ

    # Vẽ xương
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Không tìm thấy file {DATA_PATH}")
        return

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    canvas_size = 600
    
    for row in data:
        landmarks = row[1:] 
        
        black_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        try:
            draw_skeleton(black_img, landmarks)
            
            # Hiển thị số thứ tự frame
            cv.putText(black_img, f"Sample: {LABEL_TO_CHECK}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv.imshow('Data Inspector', black_img)
            
            # Tốc độ tua lại
            if cv.waitKey(30) == ord('q'):
                break
        except Exception as e:
            print(f"{e}")

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()