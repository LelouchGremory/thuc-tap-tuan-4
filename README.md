# AI Camera Core API - Hệ thống nhận diện khuôn mặt

Đây là mã nguồn hệ thống AI Camera Core API (Mốc hoàn thành: **Tuần 4**).
Dự án được xây dựng với các chức năng cốt lõi cho hệ thống AI Camera, hiện đã bao gồm:
1. **Tuần 3**: Phát hiện khuôn mặt (Face Detection YuNet), Đánh giá chất lượng (Quality Gate) & Tiền xử lý ảnh (Preprocessing).
2. **Tuần 4**: Trích xuất đặc trưng khuôn mặt (Face Embedding SFace) & API Đăng ký nhân viên (Enrollment).

## Cấu trúc lưu trữ
`storage/failed/`: Lưu trữ ảnh lỗi (Quality Gate fail, khẩu trang, landmark fail) để phục vụ debug.

## Hướng dẫn cài đặt và khởi chạy

1. **Yêu cầu hệ thống**: Cài đặt sẵn Docker, Docker Compose và Python.
2. **Cấu hình môi trường**: Tạo file `.env` từ file `.env.example` và điền các thông số kết nối cơ sở dữ liệu hoặc biến cấu hình cần thiết.
3. **Tải Model AI**: Mở terminal, chạy lệnh sau để tự động tải các file trọng số `.onnx` (YuNet, SFace) về dự án trước khi build:
   
   python download_models.py

4. **Khởi chạy hệ thống**:
   Mở terminal (PowerShell/Bash) tại thư mục gốc của dự án. Để tránh lỗi xung đột cổng (port) hoặc dính dữ liệu cũ từ các project trước, hãy chạy chuỗi lệnh sau để dọn dẹp sạch sẽ môi trường và khởi tạo mới:
  
   docker stop $(docker ps -q)
   docker rm $(docker ps -aq)
   docker-compose up -d --build

5. **Kiểm tra API (Swagger UI)**:
   Sau khi hệ thống khởi động thành công, truy cập trình duyệt tại địa chỉ:
   [http://localhost:8000/docs](http://localhost:8000/docs)

## Các lệnh tắt hữu ích sau khi khởi chạy

* **Xem log thời gian thực của container API**:
 
  docker-compose logs -f api
  
* **Khởi động lại nhanh container API (khi chỉnh sửa code)**:
  
  docker-compose restart api

* **Truy cập vào Bash shell của container API**:

  docker exec -it ai-camera-api bash

* **Truy cập trực tiếp vào PostgreSQL database**:

  docker exec -it ai-camera-db psql -U postgres -d aicamera_db

* **Dừng toàn bộ hệ thống**:

  docker-compose down

* **Dừng và xoá sạch toàn bộ volume dữ liệu cũ (Reset cứng)**:

  docker-compose down -v


## Danh sách API hiện có
- `GET /`: Kiểm tra trạng thái hoạt động của hệ thống.
- `POST /api/v1/detect_and_preprocess` (Tuần 3): Phát hiện, canh chỉnh và kiểm tra chất lượng khuôn mặt (Quality Gate) từ ảnh chụp.
- `POST /api/v1/enroll` (Tuần 4): Đăng ký nhân viên với nhiều hình ảnh (khuyên dùng 3-5 ảnh), lọc qua Quality Gate, tính toán vector đặc trưng trung bình và lưu vào PostgreSQL pgvector cùng với quản lý phiên bản model.
- `GET /api/v1/employees` (Tuần 4): Lấy danh sách thông tin nhân viên đã được đăng ký trong CSDL.
