import os
import cv2
import urllib.request
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FaceEmbedder:
    """Module Face Embedding using SFace."""
    def __init__(self, model_path="face_recognition_sface_2021dec.onnx"):
        self.model_path = model_path
        self._kiem_tra_va_tai_model()
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
            backend_id=0,
            target_id=0
        )

    def _kiem_tra_va_tai_model(self):
        """Download SFace ONNX model if not exists."""
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading SFace model: {self.model_path}...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info("Downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise

    def trich_xuat_dac_trung(self, aligned_face):
        """
        Extract feature vector from aligned face and apply L2 normalization.
        """
        if aligned_face is None:
            return None
            
        feature = self.recognizer.feature(aligned_face)
        vector = feature.flatten()
        # Ensure L2 normalization as required
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
