# Sử dụng image Python phiên bản slim để tối ưu dung lượng
FROM python:3.10-slim

# Ngăn Python tạo file .pyc và ép log xuất thẳng ra terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và PostgreSQL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt vào trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào thư mục làm việc
COPY . .

# Expose port 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy server bằng Uvicorn có kèm Hot-Reload cho môi trường Dev
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]