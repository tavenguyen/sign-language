import cv2
import os
import time
import csv
import glob
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    RAW_IMAGES_DIR: str = 'data/raw_images'
    KEYPOINTS_DIR: str = 'data/keypoints'
    
    OFFSET: int = 20
    IMG_SIZE: int = 224 
    CAPTURE_DELAY: float = 0.4
    MOVEMENT_THRESHOLD: float = 5.0 
    JITTER_THRESHOLD: float = 15.0  
    SMOOTHING_WINDOW: int = 10      
    
    TARGET_HAND: str = 'Left'      
    CAM_ID: int = 1
    
    COLLECTOR_NAME: str = "Unknown" 

    # Danh sách nhãn để kiểm tra thống kê
    EXPECTED_LABELS: Tuple[str, ...] = (
        'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y',
        'NOTHING'
    )

def calc_landmark_px(image: np.ndarray, landmarks) -> List[List[int]]:
    h, w, _ = image.shape
    landmark_point = []
    for lm in landmarks.landmark:
        px = min(int(lm.x * w), w - 1)
        py = min(int(lm.y * h), h - 1)
        landmark_point.append([px, py])
    return landmark_point

def pre_process_landmark(landmark_list: List[List[int]]) -> List[float]:
    temp_lm = np.array(landmark_list)
    base_x, base_y = temp_lm[0]
    temp_lm = temp_lm - [base_x, base_y]
    flattened = temp_lm.flatten()
    max_value = np.max(np.abs(flattened))
    if max_value != 0:
        flattened = flattened / max_value
    return flattened.tolist()

def get_square_bbox(lm_px: List[List[int]], w: int, h: int, offset: int):
    x_list = [pt[0] for pt in lm_px]
    y_list = [pt[1] for pt in lm_px]
    
    min_x, max_x = min(x_list), max(x_list)
    min_y, max_y = min(y_list), max(y_list)
    
    box_w = max_x - min_x
    box_h = max_y - min_y
    
    max_side = max(box_w, box_h) + 2 * offset
    center_x = min_x + box_w // 2
    center_y = min_y + box_h // 2
    
    x1 = max(0, center_x - max_side // 2)
    y1 = max(0, center_y - max_side // 2)
    x2 = min(w, center_x + max_side // 2)
    y2 = min(h, center_y + max_side // 2)
    
    return x1, y1, x2, y2

def count_images(label: str) -> int:
    path = os.path.join(Config.RAW_IMAGES_DIR, label)
    if not os.path.exists(path): return 0
    return len(glob.glob(os.path.join(path, "*.jpg")))

def print_stats():
    print("\n--- STATISTICS ---")
    total = 0
    labels = Config.EXPECTED_LABELS
    for i in range(0, len(labels), 4):
        chunk = labels[i:i+4]
        line = " | ".join([f"{l}: {count_images(l):<4}" for l in chunk])
        print(line)
        total += sum([count_images(l) for l in chunk])
    print(f"Total Images: {total}")
    print("------------------\n")

def setup_dirs(label: str, name: str):
    raw_dir = os.path.join(Config.RAW_IMAGES_DIR, label)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(Config.KEYPOINTS_DIR, exist_ok=True)
    
    csv_path = os.path.join(Config.KEYPOINTS_DIR, f"{label}_{name}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            header = ['label'] + [f'v{i}' for i in range(42)]
            csv.writer(f).writerow(header)
    return raw_dir, csv_path

class HandRecorder:
    def __init__(self, start_count: int = 0):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=Config.SMOOTHING_WINDOW)
        self.prev_wrist = None
        self.prev_lm = None
        
        self.last_save = 0
        self.auto_mode = False       
        self.manual_trigger = False
        self.count = start_count

    def process(self, frame, label, raw_dir, csv_path):
        # [QUAN TRỌNG 1] Luôn lật ảnh để tạo hiệu ứng Gương (Mirror)
        # Nếu bỏ dòng này, tay phải sẽ nằm bên phải màn hình -> Khó điều khiển
        frame = cv2.flip(frame, 1) 
        
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        display = frame.copy()
        skeleton = np.zeros(frame.shape, dtype=np.uint8)
        crop_img = None
        
        status = "AUTO" if self.auto_mode else "MANUAL"
        color = (0, 255, 0) if self.auto_mode else (0, 255, 255)

        if not results.multi_hand_landmarks:
            self._ui(display, label, status, color)
            self._ui(skeleton, label, status, color)
            return display, skeleton, None

        hand_lm = results.multi_hand_landmarks[0]
        
        # [QUAN TRỌNG 2] Kiểm tra tay Trái/Phải
        # Lưu ý: Khi đã flip ảnh (gương), MediaPipe sẽ nhận diện ngược lại.
        # Tay Phải thật -> Lên hình là tay Trái -> MediaPipe báo 'Left'
        detected_hand = results.multi_handedness[0].classification[0].label
        
        # Logic kiểm tra:
        # Nếu bạn dùng tay PHẢI thật để thu thập -> Cần Config.TARGET_HAND = 'Left'
        if detected_hand != Config.TARGET_HAND:
            # Hiển thị cảnh báo to rõ
            msg = f"WRONG HAND! Needed: {Config.TARGET_HAND}, Got: {detected_hand}"
            cv2.putText(display, msg, (20, h//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            return display, skeleton, None

        # Smooth & Calc (Logic giữ nguyên)
        self._smooth(hand_lm)
        lm_px = calc_landmark_px(display, hand_lm)
        
        # Stability Check
        curr_wrist = lm_px[0]
        moving = self._is_moving(curr_wrist)
        jitter = self._is_jittering(lm_px)
        
        self.prev_wrist = curr_wrist
        self.prev_lm = lm_px

        # Draw
        self.mp_drawing.draw_landmarks(display, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(skeleton, hand_lm, self.mp_hands.HAND_CONNECTIONS)

        bbox = get_square_bbox(lm_px, w, h, Config.OFFSET)
        
        box_c = (0, 0, 255) if (moving or jitter) else color
        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_c, 2)
        cv2.rectangle(skeleton, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_c, 2)

        # Save Check
        save = False
        if self.auto_mode:
            if not moving and not jitter and (time.time() - self.last_save > Config.CAPTURE_DELAY):
                save = True
        elif self.manual_trigger:
            save = True
            self.manual_trigger = False

        if save:
            self._save(frame, lm_px, label, raw_dir, csv_path, bbox)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 3)

        # Crop View
        try:
            c = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if c.size != 0: crop_img = cv2.resize(c, (Config.IMG_SIZE, Config.IMG_SIZE))
        except: pass

        self._ui(display, label, status, color)
        self._ui(skeleton, label, status, color)

        return display, skeleton, crop_img

    def _smooth(self, lm):
        curr = [[l.x, l.y, l.z] for l in lm.landmark]
        self.history.append(curr)
        avg = np.mean(np.array(self.history), axis=0)
        for i, l in enumerate(lm.landmark):
            l.x, l.y, l.z = avg[i]

    def _is_moving(self, wrist):
        if self.prev_wrist is None: return False
        return np.linalg.norm(np.array(wrist) - np.array(self.prev_wrist)) > Config.MOVEMENT_THRESHOLD

    def _is_jittering(self, px):
        if self.prev_lm is None: return 0
        diff = np.linalg.norm(np.array(px) - np.array(self.prev_lm), axis=1)
        return np.mean(diff) > Config.JITTER_THRESHOLD

    def _save(self, frame, lm_px, label, raw_dir, csv_path, bbox):
        c = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if c.size == 0: return

        # Save CSV
        norm_lm = pre_process_landmark(lm_px)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([label] + norm_lm)

        # Save Image
        rs = cv2.resize(c, (Config.IMG_SIZE, Config.IMG_SIZE))
        ts = int(time.time() * 1000)
        name = f"{label}_{Config.COLLECTOR_NAME}_{ts}.jpg"
        cv2.imwrite(os.path.join(raw_dir, name), rs)
        
        self.last_save = time.time()
        self.count += 1
        print(f"Saved: {name} (Count: {self.count})")

    def _ui(self, frame, label, status, color):
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(frame, f"Count: {self.count}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        cv2.putText(frame, "[A] Auto | [Space] Manual | [Q] Quit", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

def main():
    print("--- DATA COLLECTOR ---")
    while True:
        name = input("Enter your name (no space): ").strip()
        if name and " " not in name:
            Config.COLLECTOR_NAME = name
            break
        print("Invalid name.")

    while True:
        print_stats()
        try:
            lbl = input(f"({Config.COLLECTOR_NAME}) Enter Label (A, B...) or 'exit': ").strip().upper()
            if lbl in ['EXIT', '']: break
        except EOFError: break

        raw_dir, csv_path = setup_dirs(lbl, Config.COLLECTOR_NAME)
        curr_cnt = count_images(lbl)
        
        print(f"-> Label: {lbl} | Current: {curr_cnt}")
        print(f"-> CSV: {csv_path}")

        cap = cv2.VideoCapture(Config.CAM_ID, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        rec = HandRecorder(curr_cnt)
        
        # Kích thước hiển thị trên màn hình (vẫn giữ nhỏ gọn)
        DISPLAY_WIDTH = 800 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Xử lý trên ảnh GỐC (Độ nét cao nhất có thể)
            disp, skel, crop = rec.process(frame, lbl, raw_dir, csv_path)
            
            # --- HIỂN THỊ NHỎ GỌN ---
            h, w = disp.shape[:2]
            # Tính tỉ lệ để resize chỉ dùng cho việc hiển thị (không ảnh hưởng dữ liệu lưu)
            new_h = int(h * (DISPLAY_WIDTH / w))
            
            disp_show = cv2.resize(disp, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            skel_show = cv2.resize(skel, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow('Camera', disp_show)
            cv2.imshow('Skeleton', skel_show)
            
            if crop is not None: cv2.imshow('Crop', crop)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('a'): 
                rec.auto_mode = not rec.auto_mode
                rec.manual_trigger = False
                print(f"Mode: {'AUTO' if rec.auto_mode else 'MANUAL'}")
            if k == 32 and not rec.auto_mode: # Space
                rec.manual_trigger = True

        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()