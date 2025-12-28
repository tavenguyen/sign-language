import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
import copy
import itertools
import tensorflow as tf

# Tat log rac cua TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================================================================
#                           CAU HINH (CONFIG)
# =========================================================================
MODEL_PATH = os.path.join('model', 'keypoint_classifier.h5')
SCALER_PATH = os.path.join('model', 'scaler.p')
LABEL_MAP_PATH = os.path.join('dataset', 'label_map.json')

CONFIDENCE_THRESHOLD = 0.7

# =========================================================================
#                           HAM XU LY DU LIEU (TU CODE CUA BAN)
# =========================================================================

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 1. Tuong doi
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 2. Flatten
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 3. Chuan hoa
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def load_resources():
    # Load Model TensorFlow (.h5)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load Scaler (.p)
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Khong tim thay scaler: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Load Label Map (.json)
    labels = {}
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, 'r') as f:
            data = json.load(f)
            labels = {int(k): v for k, v in data.items()}
            
    return model, scaler, labels

# =========================================================================
#                           MAIN LOOP
# =========================================================================

def main():
    print("--- STARTING INFERENCE (TF MLP) ---")
    
    # 1. Khoi tao
    try:
        model, scaler, labels = load_resources()
        print("Tai resources thanh cong.")
    except Exception as e:
        print(f"Loi khoi tao: {e}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        debug_image = frame.copy()
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Ve khung xuong
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- XU LY DU LIEU ---
                # 1. Tinh toa do pixel
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # 2. Tien xu ly (Relative -> Flatten -> Normalize)
                processed_lm = pre_process_landmark(landmark_list)
                
                # 3. Chuan hoa bang Scaler (QUAN TRONG)
                input_data = np.array([processed_lm])
                input_data = scaler.transform(input_data)

                # --- DU DOAN ---
                # Model TF tra ve xac suat truc tiep (Softmax)
                prediction_probs = model.predict(input_data, verbose=0)[0]
                pred_idx = np.argmax(prediction_probs)
                max_prob = prediction_probs[pred_idx]
                
                pred_label = labels.get(pred_idx, str(pred_idx))

                # --- HIEN THI (Logic giong code cua ban) ---
                # Tinh hinh chu nhat bao quanh tay
                rect = cv2.boundingRect(np.array(landmark_list))
                x, y, rw, rh = rect
                x1, y1, x2, y2 = x, y, x + rw, y + rh

                if max_prob < CONFIDENCE_THRESHOLD:
                    color = (0, 0, 255) # Do
                    text = f"UNKNOWN ({max_prob*100:.0f}%)"
                else:
                    color = (0, 255, 0) # Xanh
                    text = f"{pred_label} ({max_prob*100:.0f}%)"

                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_image, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Inference TF MLP', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()