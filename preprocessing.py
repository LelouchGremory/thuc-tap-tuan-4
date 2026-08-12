import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FacePreprocessor:
    """Module tiền xử lý và kiểm tra chất lượng khuôn mặt (Face Quality Gate)."""

    def __init__(self, target_size=(112, 112), min_face_size=60, blur_threshold=100.0, brightness_range=(40, 220)):
        self.target_size = target_size
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range

    def _kiem_tra_do_mo(self, face_img):
        """Kiểm tra độ mờ bằng phương sai ảnh Laplacian."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > self.blur_threshold, laplacian_var

    def _kiem_tra_do_sang(self, face_img):
        """Kiểm tra độ sáng trung bình của khuôn mặt."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        min_b, max_b = self.brightness_range
        return min_b <= mean_brightness <= max_b, mean_brightness

    def _kiem_tra_goc_nghieng(self, landmarks):
        """Ước lượng góc nghiêng đơn giản qua tỷ lệ khoảng cách mắt và mũi."""
        # landmarks: [mắt trái, mắt phải, mũi, mép trái, mép phải]
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        
        # Kiểm tra khoảng cách mũi đến hai mắt
        dist_left = np.linalg.norm(nose - left_eye)
        dist_right = np.linalg.norm(nose - right_eye)
        
        # Nếu tỷ lệ quá lệch thì mặt bị nghiêng nhiều
        if dist_left == 0 or dist_right == 0:
            return False
            
        ratio = dist_left / dist_right
        # Ngưỡng tỷ lệ chấp nhận được (gần 1 là trực diện)
        return 0.5 <= ratio <= 2.0

    def kiem_tra_chat_luong(self, image, face_info):
        """
        Face Quality Gate: Kiểm tra tổng thể các tiêu chí chất lượng ảnh khuôn mặt.
        """
        x, y, w, h = face_info['bbox']
        
        # 1. Kiểm tra kích thước tối thiểu
        if w < self.min_face_size or h < self.min_face_size:
            logger.warning(f"Khuôn mặt quá nhỏ: {w}x{h}")
            return False, "Kích thước quá nhỏ"
            
        # Đảm bảo bounding box nằm trong ảnh
        height, width = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            return False, "Lỗi cắt ảnh"

        # 2. Kiểm tra độ mờ
        is_clear, blur_score = self._kiem_tra_do_mo(face_img)
        if not is_clear:
            logger.warning(f"Ảnh quá mờ (điểm: {blur_score:.2f})")
            return False, "Ảnh mờ"

        # 3. Kiểm tra độ sáng
        is_bright, bright_score = self._kiem_tra_do_sang(face_img)
        if not is_bright:
            logger.warning(f"Độ sáng không đạt (điểm: {bright_score:.2f})")
            return False, "Độ sáng kém"

        # 4. Kiểm tra góc nghiêng
        is_frontal = self._kiem_tra_goc_nghieng(face_info['landmarks'])
        if not is_frontal:
            logger.warning("Khuôn mặt bị nghiêng quá nhiều")
            return False, "Góc nghiêng lệch"

        return True, "Đạt yêu cầu"

    def can_chinh_va_cat(self, image, face_info):
        """
        Alignment và cắt khuôn mặt dựa trên tọa độ mắt (landmarks).
        """
        landmarks = face_info['landmarks']
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Tính góc xoay giữa hai mắt
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Tâm xoay (giữa hai mắt)
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # Ma trận xoay
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        
        h, w = image.shape[:2]
        aligned_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Cắt khuôn mặt sau khi xoay (sử dụng bbox gốc làm ước lượng, có thể tính lại chính xác hơn)
        bx, by, bw, bh = face_info['bbox']
        
        # Mở rộng bounding box một chút để lấy trọn vẹn khuôn mặt sau khi xoay
        margin = int(bw * 0.1)
        nx1, ny1 = max(0, bx - margin), max(0, by - margin)
        nx2, ny2 = min(w, bx + bw + margin), min(h, by + bh + margin)
        
        cropped_face = aligned_img[ny1:ny2, nx1:nx2]
        
        # Resize về kích thước chuẩn
        resized_face = cv2.resize(cropped_face, self.target_size)
        
        return resized_face

    def chuan_hoa(self, face_img):
        """
        Normalize ảnh: Đưa giá trị pixel về khoảng [-1, 1] cho mô hình AI.
        """
        face_img = face_img.astype(np.float32)
        # Normalize theo chuẩn thông thường (ví dụ: chia 255.0 hoặc trừ mean)
        normalized = (face_img / 255.0 - 0.5) * 2.0
        return normalized
