import os
import cv2
import urllib.request
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector:
    """Module phát hiện khuôn mặt sử dụng thuật toán YuNet của OpenCV."""

    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", 
                 score_threshold=0.9, nms_threshold=0.3, top_k=5000):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        
        self._kiem_tra_va_tai_model()
        
        # Khởi tạo detector, input_size sẽ được cập nhật linh hoạt khi nhận ảnh
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    def _kiem_tra_va_tai_model(self):
        """Tải mô hình YuNet ONNX nếu chưa có sẵn trong hệ thống."""
        if not os.path.exists(self.model_path):
            logger.info(f"Đang tải mô hình YuNet: {self.model_path}...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info("Tải mô hình thành công.")
            except Exception as e:
                logger.error(f"Không thể tải mô hình: {e}")
                raise

    def phat_hien(self, image):
        """
        Phát hiện khuôn mặt trong ảnh.
        Trả về danh sách các khuôn mặt, mỗi khuôn mặt là dictionary chứa:
        - 'bbox': (x, y, w, h)
        - 'landmarks': [(x1, y1), (x2, y2), ...] (mắt trái, mắt phải, mũi, mép trái, mép phải)
        - 'confidence': độ tin cậy
        """
        if image is None:
            return []

        height, width, _ = image.shape
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)
        
        ket_qua = []
        if faces is not None:
            for face in faces:
                box = list(map(int, face[:4]))
                landmarks = [
                    (int(face[4]), int(face[5])),   # Mắt trái
                    (int(face[6]), int(face[7])),   # Mắt phải
                    (int(face[8]), int(face[9])),   # Mũi
                    (int(face[10]), int(face[11])), # Mép miệng trái
                    (int(face[12]), int(face[13]))  # Mép miệng phải
                ]
                confidence = float(face[14])
                
                ket_qua.append({
                    "bbox": box,
                    "landmarks": landmarks,
                    "confidence": confidence
                })
                
        return ket_qua
