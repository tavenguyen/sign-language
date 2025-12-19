import os
import cv2 as cv
import mediapipe as mp

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

def main():
    create_dirs()

    # Camera Setup
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode = False,         # Video mode
        max_num_hands = 1,                 # Only 1 hand.
        min_detection_confidence = 0.7,    # Ngưỡng tin cậy tối thiểu để mô hình nhận diện đó là tay.
        min_tracking_confidence = 0.5,     # Ngưỡng tin cậy tối thiểu để mô hình phát hiện các đốt ngón tay.
    )

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()