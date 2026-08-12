import os
import cv2
import logging
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from detector import FaceDetector
from preprocessing import FacePreprocessor
from embedding import FaceEmbedder
import models

logger = logging.getLogger(__name__)

FAILED_DIR = "storage/failed/enrollment/"
os.makedirs(FAILED_DIR, exist_ok=True)

class EnrollmentService:
    def __init__(self):
        self.detector = FaceDetector()
        self.preprocessor = FacePreprocessor()
        self.embedder = FaceEmbedder()
        self.model_version = "sface_2021dec"
        
    def _save_failed_image(self, ma_nhan_vien: str, img: np.ndarray, reason: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ma_nhan_vien}_{timestamp}.jpg"
        filepath = os.path.join(FAILED_DIR, filename)
        cv2.imwrite(filepath, img)
        logger.error(f"Enrollment failed for {ma_nhan_vien} - Reason: {reason}. Saved to {filepath}")
        return filepath

    def dang_ky_nhan_vien(self, db: Session, ma_nhan_vien: str, ho_ten: str, phong_ban: str, danh_sach_anh: list):
        danh_sach_vector = []
        chi_tiet_loi = []
        
        for i, img in enumerate(danh_sach_anh):
            faces = self.detector.phat_hien(img)
            if not faces:
                filepath = self._save_failed_image(ma_nhan_vien, img, "No face detected")
                chi_tiet_loi.append({
                    "buoc_that_bai": "Face Detection",
                    "ly_do": "No face detected",
                    "duong_dan_file": filepath
                })
                continue
                
            best_face = max(faces, key=lambda x: x['confidence'])
            
            is_valid, msg = self.preprocessor.kiem_tra_chat_luong(img, best_face)
            if not is_valid:
                filepath = self._save_failed_image(ma_nhan_vien, img, f"Quality gate failed: {msg}")
                chi_tiet_loi.append({
                    "buoc_that_bai": "Quality Gate",
                    "ly_do": f"Quality gate failed: {msg}",
                    "duong_dan_file": filepath
                })
                continue
                
            try:
                aligned_face = self.preprocessor.can_chinh_va_cat(img, best_face)
                vector = self.embedder.trich_xuat_dac_trung(aligned_face)
                
                if vector is not None:
                    danh_sach_vector.append(vector)
                else:
                    filepath = self._save_failed_image(ma_nhan_vien, img, "Feature extraction failed")
                    chi_tiet_loi.append({
                        "buoc_that_bai": "Embedding",
                        "ly_do": "Feature extraction failed",
                        "duong_dan_file": filepath
                    })
            except Exception as e:
                filepath = self._save_failed_image(ma_nhan_vien, img, f"Exception during processing: {str(e)}")
                chi_tiet_loi.append({
                    "buoc_that_bai": "Processing",
                    "ly_do": f"Exception during processing: {str(e)}",
                    "duong_dan_file": filepath
                })
                
        if not danh_sach_vector:
            return {
                "thanh_cong": False, 
                "thong_bao": "All provided images failed quality checks.",
                "so_luong_anh_hop_le": 0,
                "so_luong_anh_loi": len(danh_sach_anh),
                "chi_tiet_loi": chi_tiet_loi
            }
            
        avg_vector = np.mean(danh_sach_vector, axis=0)
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm 
        
        try:
            nv = db.query(models.Employee).filter(models.Employee.ma_nhan_vien == ma_nhan_vien).first()
            if not nv:
                nv = models.Employee(
                    ma_nhan_vien=ma_nhan_vien,
                    ho_ten=ho_ten,
                    phong_ban=phong_ban
                )
                db.add(nv)
                db.flush()
            else:
                nv.ho_ten = ho_ten
                nv.phong_ban = phong_ban

            db.query(models.FaceProfile).filter(models.FaceProfile.employee_id == nv.id).delete()
            
            face_profile = models.FaceProfile(
                employee_id=nv.id,
                embedding=avg_vector.tolist(),
                phien_ban_model=self.model_version
            )
            db.add(face_profile)
            db.commit()
            
            return {
                "thanh_cong": True, 
                "thong_bao": f"Successfully enrolled {ho_ten} ({ma_nhan_vien})",
                "so_luong_anh_hop_le": len(danh_sach_vector),
                "so_luong_anh_loi": len(danh_sach_anh) - len(danh_sach_vector),
                "chi_tiet_loi": chi_tiet_loi
            }
        except Exception as e:
            db.rollback()
            logger.error(f"DB Error: {e}")
            return {
                "thanh_cong": False, 
                "thong_bao": f"System error saving to DB: {str(e)}",
                "so_luong_anh_hop_le": len(danh_sach_vector),
                "so_luong_anh_loi": len(danh_sach_anh) - len(danh_sach_vector),
                "chi_tiet_loi": chi_tiet_loi
            }