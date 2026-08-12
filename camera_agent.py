import os
import cv2
import time
import logging
import threading
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CameraAgent:
    """Agent quản lý luồng Camera với đa luồng, tự động kết nối lại và xử lý hàng đợi frame."""

    def __init__(self):
        # Đọc cấu hình từ biến môi trường
        self.camera_ip = os.getenv("CAMERA_TEST_IP", "0")
        self.camera_user = os.getenv("CAMERA_USER", "")
        self.camera_pass = os.getenv("CAMERA_PASSWORD", "")
        self.target_fps = int(os.getenv("CAMERA_FPS", "5")) # Sampling 5-10 frame/giây
        
        self.frame_queue = Queue(maxsize=30)
        self.is_running = False
        self.reconnect_delay = 5  # Giây
        
        # Cấu hình Vùng quan tâm (ROI) mặc định: (x, y, w, h)
        self.roi = (100, 100, 400, 400) 

    def _get_connection_string(self):
        """Tạo chuỗi kết nối dựa trên cấu hình (hỗ trợ webcam cục bộ, video hoặc RTSP)"""
        if self.camera_ip.isdigit():
            return int(self.camera_ip) # Webcam
        
        # Xử lý chèn user/pass vào RTSP nếu cần
        if self.camera_ip.startswith("rtsp://") and self.camera_user and self.camera_pass:
            if "@" not in self.camera_ip:
                parts = self.camera_ip.split("rtsp://")
                return f"rtsp://{self.camera_user}:{self.camera_pass}@{parts[1]}"
        
        return self.camera_ip

    def _doc_khung_hinh(self):
        """Luồng đọc dữ liệu liên tục từ camera"""
        chuoi_ket_noi = self._get_connection_string()
        
        while self.is_running:
            logger.info(f"Đang thử kết nối tới camera: {chuoi_ket_noi}")
            cap = cv2.VideoCapture(chuoi_ket_noi)
            
            if not cap.isOpened():
                logger.error(f"Lỗi kết nối camera, thử lại sau {self.reconnect_delay} giây...")
                time.sleep(self.reconnect_delay)
                continue

            logger.info("Kết nối camera thành công!")
            frame_delay = 1.0 / self.target_fps
            
            try:
                while self.is_running and cap.isOpened():
                    start_time = time.time()
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning("Mất tín hiệu camera, chuẩn bị kết nối lại...")
                        break

                    frame = self._xu_ly_frame(frame)
                    
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    
                    elapsed = time.time() - start_time
                    sleep_time = frame_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Lỗi trong quá trình đọc camera: {e}")
            finally:
                cap.release()

    def _xu_ly_frame(self, frame):
        """Thêm thời gian và vẽ Vùng quan tâm (ROI) lên khung hình"""
        # 1. Chèn timestamp
        thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, thoi_gian, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 2. Vẽ ROI
        x, y, w, h = self.roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return frame

    def bat_dau(self):
        """Khởi động agent đọc camera đa luồng"""
        if self.is_running: return
        self.is_running = True
        self.luong_doc = threading.Thread(target=self._doc_khung_hinh, daemon=True)
        self.luong_doc.start()
        logger.info("Đã khởi động Camera Agent.")

    def dung_lai(self):
        """Dừng tiến trình"""
        self.is_running = False
        if hasattr(self, 'luong_doc'):
            self.luong_doc.join(timeout=2.0)
        logger.info("Đã dừng Camera Agent.")
        
    def lay_khung_hinh(self):
        """Lấy một khung hình từ hàng đợi"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

if __name__ == "__main__":
    agent = CameraAgent()
    agent.bat_dau()
    try:
        while True:
            frame = agent.lay_khung_hinh()
            if frame is not None:
                # Có thể xử lý frame tại đây
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Dừng hệ thống...")
        agent.dung_lai()
