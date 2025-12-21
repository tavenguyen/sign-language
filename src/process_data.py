import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# A=0, B=1, C=2, ..., Y=23
LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
    'UNK'
]

INPUT_DIR = 'data/keypoints'
OUTPUT_DIR = 'dataset'
COMBINED_FILE = os.path.join(OUTPUT_DIR, 'combined_data.csv')
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.csv')

def main():
    # Tạo folder dataset nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    master_data = []
    for index, label in enumerate(LABELS):
        file_path = os.path.join(INPUT_DIR, f"{label}.csv")
        
        if not os.path.exists(file_path):
            print(f"NOT FOUND: '{label}'")
            continue
        
        try:
            df = pd.read_csv(file_path, header=None)
            
            if df.empty:
                print(f"[WARNING] File '{label}.csv' is empty.")
                continue

            df.iloc[:, 0] = index
            
            master_data.append(df)
        except Exception as e:
            print(f"[ERROR] Cannot read {label}.csv: {e}")

    if not master_data:
        print("[FATAL] Found no data to process!")
        return

    # Combined all DataFrame
    full_df = pd.concat(master_data, ignore_index=True)
    print("Total:", len(full_df))

    # SHUFFLE 
    shuffled_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_df.to_csv(COMBINED_FILE, header=False, index=False)

    # PARTITIONING
    X = shuffled_df.iloc[:, 1:]
    y = shuffled_df.iloc[:, 0]

    # 80% Train - 20% Test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_df = pd.concat([y_train, X_train], axis=1)
        test_df = pd.concat([y_test, X_test], axis=1)

        train_df.to_csv(TRAIN_FILE, header=False, index=False)
        test_df.to_csv(TEST_FILE, header=False, index=False)
    except ValueError as e:
        print(f"\n[ERROR] Cannot split data {e}")

if __name__ == "__main__":
    main()