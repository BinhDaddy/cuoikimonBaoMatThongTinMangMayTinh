# Ứng dụng Thủy vân số (Digital Watermarking)

Đây là ứng dụng thử nghiệm các thuật toán thủy vân số nhằm bảo vệ bản quyền ảnh số.

## Các thuật toán đã cài đặt

1. LSB (Least Significant Bit)
   - Nhúng thông điệp vào bit ít quan trọng nhất của ảnh
   - Đơn giản, dễ cài đặt
   - Dễ bị phá hủy khi ảnh bị nén hoặc chỉnh sửa

2. DCT (Discrete Cosine Transform)
   - Nhúng thông điệp vào miền tần số của ảnh
   - Bền vững hơn với các tấn công như nén ảnh
   - Cần ảnh gốc để trích xuất thủy vân

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Các thư viện Python:
  - Pillow
  - numpy
  - opencv-python
  - matplotlib

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Chuẩn bị ảnh test:
   - Đặt ảnh test vào thư mục gốc của dự án
   - Đặt tên file là `test_image.jpg` hoặc thay đổi đường dẫn trong `main.py`

## Sử dụng

1. Chạy chương trình:
```bash
python main.py
```

2. Kết quả:
   - Ảnh đã nhúng thủy vân sẽ được lưu trong thư mục `output`
   - Thông điệp trích xuất sẽ được hiển thị trên màn hình

## Lưu ý

- Đảm bảo ảnh test có kích thước đủ lớn để chứa thông điệp
- Thông điệp càng dài thì càng cần ảnh có kích thước lớn
- Thuật toán DCT yêu cầu ảnh gốc để trích xuất thủy vân 