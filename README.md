# sign-language

## Introduction
Dự án này xây dựng hệ thống nhận diện bảng chữ cái ngôn ngữ ký hiệu Việt Nam trong thời gian thực. Chúng tôi sẽ sử dụng MediaPipe để trích xuất 21 landmarks trên bàn tay, phương pháp này có tốc độ inference nhanh, độ trễ thấp, và loại bỏ nhiễu tốt hơn so với xử lý ảnh truyền thống.

## Installation
Sử dụng miniconda để tạo môi trường ảo:
```
conda create --name sign-language-env python==3.10 -y
conda activate sign-language-env
pip install mediapipe==0.10.9
```
Sau khi đã cài đặt xong môi trường ảo thì ta chọn môi trường ảo đó để chạy code. Ví dụ (Vistual Studio Code): View -> Command Palette -> Python: Select Interpreter -> Chọn sign-language-env

## Project Structure
File data sẽ không có raw_images bởi vì kích thước của nó quá lớn.
```
sign-language/
│
├── data/                       # Nơi chứa dữ liệu thô và dữ liệu đã qua xử lý MediaPipe
│   ├── raw_images/             # Ảnh gốc 
│   └── keypoints/              # Dữ liệu tọa độ bàn tay
│       ├── A.csv               
│       └── ...
│
├── dataset/                    # Dữ liệu đầu vào cho Model
│   └── combined_data.csv       # -> Đây là file trực tiếp được load vào để train.
│                               
├── model/                      # Các artifacts sau khi huấn luyện
│   ├── keypoint_classifier.hdf5    # Model gốc (Keras/TensorFlow) để đánh giá, tiếp tục train
│
└── src/                        # Source code chính
    ├── collect_data.py         # Script chuyển raw_images -> keypoints
    ├── check_data.py           # Script để kiểm tra chất lượng của data.
    ├── train.py                # Script load combined_data.csv -> train -> lưu vào model/
    └── inference.py            # Script nhận diện realtime
```
