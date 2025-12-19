import os
import copy
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

def is_right_hand(handedness):
    return handedness.classification[0].label == 'Left'  

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

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks: 
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if is_right_hand(hand_handedness):
                    confidence_score = hand_handedness.classification[0].score
                    label = hand_handedness.classification[0].label 
                    print(f"Label: {label} | Score: {confidence_score:.4f}")
                    text_display = f"{label}: {confidence_score:.0%}"

                    color = (0, 255, 0)
                    cv.putText(image, text_display, (10, 50), 
                            cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
                    
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS)

        cv.imshow('Data Collection', image)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()