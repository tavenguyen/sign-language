import os
import cv2 as cv

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