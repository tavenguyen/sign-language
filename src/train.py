import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Tat log cua TensorFlow (chi hien loi)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Cau hinh
DATA_PATH = os.path.join('dataset', 'keypoint.csv')
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.p')

def main():
    print("--- TRAIN MLP (TENSORFLOW) ---")

    # 1. Kiem tra va doc du lieu
    if not os.path.exists(DATA_PATH):
        print(f"Error: Khong tim thay file {DATA_PATH}")
        return

    # Header=None vi file keypoint.csv khong co tieu de
    df = pd.read_csv(DATA_PATH, header=None)
    
    # Cot 0 la Nhan (Label), Cot 1->42 la Toa do (Features)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    print(f"Tong so mau: {X.shape[0]}")
    print(f"So luong features: {X.shape[1]}")

    # 2. Xu ly du lieu
    # Chia train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Chuan hoa du lieu (Scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-hot encoding cho nhan
    # Tinh so luong class dua tren du lieu thuc te (cong them 1 vi index bat dau tu 0)
    num_classes = len(np.unique(y))
    # Hoac neu muon an toan hon neu thieu du lieu: num_classes = 23 (theo config)
    
    print(f"So luong class can phan loai: {num_classes}")

    y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_enc = tf.keras.utils.to_categorical(y_test, num_classes)

    # 3. Khoi tao Model
    model = Sequential([
        Input(shape=(42,)),              # Lop dau vao ro rang
        Dense(128, activation='relu'),
        Dropout(0.2),                    # Giam overfit
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax') # Lop dau ra
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Cau hinh Train
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Callback: Dung som neu khong cai thien
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Callback: Luu lai model tot nhat trong qua trinh train
    mc = ModelCheckpoint(MODEL_PATH, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    print("Bat dau train...")
    history = model.fit(
        X_train_scaled, y_train_enc,
        validation_data=(X_test_scaled, y_test_enc),
        epochs=1000,
        batch_size=32,
        callbacks=[es, mc],
        verbose=1
    )

    # 5. Danh gia
    loss, acc = model.evaluate(X_test_scaled, y_test_enc, verbose=0)
    print(f"Ket qua tren tap Test - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # 6. Luu Scaler (Model da duoc luu boi ModelCheckpoint)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Da luu Scaler tai: {SCALER_PATH}")
    print("Hoan tat.")

if __name__ == "__main__":
    main()