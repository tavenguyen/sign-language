import os
import copy
import csv
import itertools
import numpy as np
import cv2 as cv
import mediapipe as mp
from datetime import datetime

STILLNESS_THRESHOLD = 5

TARGET_LABEL = input("Class: ").upper()

# Directory
RAW_IMAGE_DIR = os.path.join("data", "raw_images", TARGET_LABEL)
KEYPOINT_PATH = os.path.join("data", "keypoints", f"{TARGET_LABEL}.csv")

def create_dirs():
    if not os.path.exists(RAW_IMAGE_DIR):
        os.makedirs(RAW_IMAGE_DIR)
    
    keypoint_dir = os.path.dirname(KEYPOINT_PATH)
    if not os.path.exists(keypoint_dir):
        os.makedirs(keypoint_dir)

def is_right_hand(handedness):
    return handedness.classification[0].label == 'Left'  

def is_hand_moving(current_landmarks, previous_landmarks, threshold):
    if previous_landmarks is None:
        return True 
    
    total_movement = 0
    num_landmarks = len(current_landmarks)

    curr_np = np.array(current_landmarks)
    prev_np = np.array(previous_landmarks)
    distances = np.linalg.norm(curr_np - prev_np, axis = 1)
    average_movement = np.mean(distances)

    return average_movement < threshold, average_movement

def calc_landmark_list(image, landmarks):
    """
    Chuyển đổi toạ độ chuẩn hoá (0.0-1.0) sang toạ độ pixel
    """
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z # Tạm thời bỏ qua Z cho mô hình đơn giản
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    """
    1. Chuyển về toạ độ tương đối (Trừ đi toạ độ cổ tay - điểm số 0)
    2. Chuẩn hoá (Normalize) để không phụ thuộc kích thước tay hay khoảng cách camera.
    3. Làm phẳng (Flatten) thành mảng 1 chiều.
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 1. Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 2. Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 3. Normalization (Max absolute scaling)
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0 # Tránh chia cho 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def save_data(image, processed_landmark_list, raw_landmark_list):
    # CSV (Append mode)
    with open(KEYPOINT_PATH, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([TARGET_LABEL, *processed_landmark_list])
    
    # Store Raw
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_filename = f"{TARGET_LABEL}_{timestamp}.jpg"
    cv.imwrite(os.path.join(RAW_IMAGE_DIR, img_filename), image)

def main():
    create_dirs()

    # Camera Setup
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode = False,         # Video mode
        max_num_hands = 1,                 # Only 1 hand.
        min_detection_confidence = 0.7,    # Ngưỡng tin cậy tối thiểu để mô hình nhận diện đó là tay.
        min_tracking_confidence = 0.5,     # Ngưỡng tin cậy tối thiểu để mô hình phát hiện các đốt ngón tay.
    )

    count = 0
    if os.path.exists(KEYPOINT_PATH):
        with open(KEYPOINT_PATH, 'r') as f:
            count = sum(1 for line in f)

    prev_landmark_list = None
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks: 
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if not is_right_hand(hand_handedness):
                    continue

                # Data Processing
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                x, y, w, h = cv.boundingRect(np.array(landmark_list))

                # Confidence Score
                confidence_score = hand_handedness.classification[0].score
                hand_label = hand_handedness.classification[0].label 

                # State Check
                is_stable, movement_val = is_hand_moving(landmark_list, prev_landmark_list, threshold = STILLNESS_THRESHOLD)
                prev_landmark_list = landmark_list

                status_color = (0, 255, 0) if is_stable else (0, 0, 255) 
                status_text = f"Stable ({movement_val:.1f})" if is_stable else f"MOVING! ({movement_val:.1f})"
                conf_text = f"{hand_label}: {confidence_score:.0%}"
                print(status_text)
                print(conf_text)

                # Draw a rectangle covered hand.
                cv.rectangle(debug_image, (x, y), (x + w, y + h), status_color, 2)

                cv.putText(debug_image, status_text, (10, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv.LINE_AA)
                
                cv.putText(debug_image, conf_text, (x, y - 10), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw Skeleton Hand
                mp_drawing.draw_landmarks(
                    debug_image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS)
                
                key = cv.waitKey(10)
                if key == ord('k') or key == ord('K'):
                    if is_stable:
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        save_data(debug_image, pre_processed_landmark_list, landmark_list)
                        count += 1
                        cv.putText(debug_image, "SAVING!", (10, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv.putText(debug_image, "TAY QUA NHANH!", (10, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
        cv.putText(debug_image, f"Count: {count}", (10, 70), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv.imshow('Data Collection', debug_image)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()