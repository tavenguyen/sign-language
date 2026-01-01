import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
import tensorflow as tf
import time
from collections import deque
from dataclasses import dataclass
from typing import List

# Tat log cua TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@dataclass
class Config:
    # Cau hinh duong dan model va du lieu
    MODEL_PATH: str = os.path.join('model', 'keypoint_classifier.h5')
    SCALER_PATH: str = os.path.join('model', 'scaler.p')
    LABEL_MAP_PATH: str = os.path.join('dataset', 'label_map.json')
    CONFIDENCE_THRESHOLD: float = 0.8
    
    # Cau hinh Camera va hien thi
    OFFSET: int = 20
    SMOOTHING_WINDOW: int = 5 
    TARGET_HAND: str = 'Left' 
    CAM_ID: int = 0

    CAM_WIDTH: int = 1280
    CAM_HEIGHT: int = 720

def calc_landmark_px(image: np.ndarray, landmarks) -> List[List[int]]:
    # Chuyen doi toa do chuan hoa sang toa do pixel
    h, w, _ = image.shape
    landmark_point = []
    for lm in landmarks.landmark:
        px = min(int(lm.x * w), w - 1)
        py = min(int(lm.y * h), h - 1)
        landmark_point.append([px, py])
    return landmark_point

def get_square_bbox(lm_px: List[List[int]], w: int, h: int, offset: int):
    # Tinh toan hinh chu nhat bao quanh ban tay
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

def pre_process_landmark(landmark_list):
    # Xu ly du lieu: Chuyen ve toa do tuong doi, lam phang va chuan hoa
    temp_landmark_list = np.array(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    temp_landmark_list = temp_landmark_list - [base_x, base_y]
    temp_landmark_list = temp_landmark_list.flatten()
    max_value = np.max(np.abs(temp_landmark_list))
    if max_value != 0:
        temp_landmark_list = temp_landmark_list / max_value
    return temp_landmark_list.tolist()

class SignLanguageDetector:
    def __init__(self):
        # Khoi tao model va scaler
        self.model, self.scaler, self.labels = self._load_resources()
        
        # Khoi tao MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Hang doi de lam muot chuyen dong
        self.history = deque(maxlen=Config.SMOOTHING_WINDOW)

    def _load_resources(self):
        # Tai file model .h5, scaler .p va label map .json
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Thieu model: {Config.MODEL_PATH}")
        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(f"Thieu scaler: {Config.SCALER_PATH}")
            
        print("Dang tai tai nguyen...")
        model = tf.keras.models.load_model(Config.MODEL_PATH, compile = False)
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            
        labels = {}
        if os.path.exists(Config.LABEL_MAP_PATH):
            with open(Config.LABEL_MAP_PATH, 'r') as f:
                data = json.load(f)
                labels = {int(k): v for k, v in data.items()}
        
        print("Tai thanh cong!")
        return model, scaler, labels

    def process(self, frame):
        # Lat anh
        # frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Doan nay ton CPU nhat
        results = self.hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return frame

        hand_lm = results.multi_hand_landmarks[0]
        detected_hand = results.multi_handedness[0].classification[0].label
        
        if detected_hand != Config.TARGET_HAND:
            msg = f"SAI TAY! Can: {Config.TARGET_HAND}"
            cv2.putText(frame, msg, (20, h//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            return frame

        self._smooth(hand_lm)
        
        lm_px = calc_landmark_px(frame, hand_lm)
        processed_lm = pre_process_landmark(lm_px)
        input_data = np.array([processed_lm])
        input_data = self.scaler.transform(input_data)
        
        prediction_probs = self.model.predict(input_data, verbose=0)[0]
        pred_idx = np.argmax(prediction_probs)
        max_prob = prediction_probs[pred_idx]
        pred_label = self.labels.get(pred_idx, str(pred_idx))

        self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        bbox = get_square_bbox(lm_px, w, h, Config.OFFSET)
        
        if max_prob > Config.CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)
            status_text = f"{pred_label} ({max_prob*100:.0f}%)"
        else:
            color = (0, 165, 255)
            status_text = f"{pred_label}? ({max_prob*100:.0f}%)"
            
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, status_text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

    def _smooth(self, lm):
        # Tinh trung binh vi tri cac khop qua nhieu frame de giam rung
        curr = [[l.x, l.y, l.z] for l in lm.landmark]
        self.history.append(curr)
        avg = np.mean(np.array(self.history), axis=0)
        for i, l in enumerate(lm.landmark):
            l.x, l.y, l.z = avg[i]

def main():
    # Thiet lap Camera do phan giai toi da
    cap = cv2.VideoCapture(Config.CAM_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
    
    detector = SignLanguageDetector()
    
    DISPLAY_WIDTH = 800 
    prev_time = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Xu ly frame
        disp = detector.process(frame)
        
        # Resize de hien thi vua man hinh
        h, w = disp.shape[:2]
        new_h = int(h * (DISPLAY_WIDTH / w))
        disp_show = cv2.resize(disp, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        
        # FPS
        curr_time = time.time()
        fps = 0
        if curr_time - prev_time > 0:
            fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Hien thi FPS goc tren trai
        cv2.putText(disp_show, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Inference Camera', disp_show)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()