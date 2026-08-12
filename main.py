import logging
from typing import List, Optional
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import engine, Base, get_db
import models
import cv2
import numpy as np

from detector import FaceDetector
from preprocessing import FacePreprocessor
from enrollment import EnrollmentService
from embedding import FaceEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    logger.info("Database and tables initialized successfully.")
except Exception as e:
    logger.error(f"Database initialization error: {e}")

app = FastAPI(
    title="AI Camera Core API",
    description="Core API for Face Recognition System",
    version="1.0.0"
)

face_detector = FaceDetector()
face_preprocessor = FacePreprocessor()
face_embedder = FaceEmbedder()
enrollment_service = EnrollmentService()

class ErrorDetail(BaseModel):
    buoc_that_bai: str
    ly_do: str
    duong_dan_file: str

class EnrollResponse(BaseModel):
    thanh_cong: bool
    thong_bao: str
    so_luong_anh_hop_le: int
    so_luong_anh_loi: int
    chi_tiet_loi: Optional[List[ErrorDetail]] = []

@app.get("/")
def check_health():
    return {"status": "AI Camera Core API is running normally."}

@app.post("/api/v1/detect_and_preprocess", summary="Detect and preprocess face (Week 3)")
async def detect_and_preprocess(
    anh_camera: UploadFile = File(..., description="Camera captured image")
):
    try:
        contents = await anh_camera.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail="Could not read image file.")

    faces = face_detector.phat_hien(img)
    if not faces:
        return {"so_luong_khuon_mat": 0, "ket_qua": []}

    ket_qua = []
    for i, face_info in enumerate(faces):
        is_valid, msg = face_preprocessor.kiem_tra_chat_luong(img, face_info)
        ket_qua.append({
            "khuon_mat_thu": i + 1,
            "toa_do": face_info["bbox"],
            "do_tin_cay": face_info["confidence"],
            "dat_chat_luong": is_valid,
            "ly_do": msg
        })

    return {
        "so_luong_khuon_mat": len(faces),
        "ket_qua_chi_tiet": ket_qua
    }

@app.post("/api/v1/enroll", response_model=EnrollResponse, summary="Enroll new employee face (Week 4)")
async def dang_ky_nhan_vien(
    ma_nhan_vien: str = Form(..., description="Employee Code (e.g., NV001)"),
    ho_ten: str = Form(..., description="Employee Full Name"),
    phong_ban: str = Form("Chưa phân phòng", description="Department"),
    anh_khuon_mat: List[UploadFile] = File(..., description="List of face images (Recommend 3-5 images)"),
    db: Session = Depends(get_db)
):
    danh_sach_cv2 = []
    for file in anh_khuon_mat:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                danh_sach_cv2.append(img)
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            
    if not danh_sach_cv2:
        raise HTTPException(status_code=400, detail="No valid image data provided.")
        
    ket_qua = enrollment_service.dang_ky_nhan_vien(
        db=db, 
        ma_nhan_vien=ma_nhan_vien, 
        ho_ten=ho_ten, 
        phong_ban=phong_ban, 
        danh_sach_anh=danh_sach_cv2
    )
    
    if not ket_qua["thanh_cong"]:
        raise HTTPException(status_code=400, detail=ket_qua)
        
    return ket_qua

@app.get("/api/v1/employees", summary="Get enrolled employees list")
def danh_sach_nhan_vien(db: Session = Depends(get_db)):
    nv = db.query(models.Employee).all()
    return [{"id": n.id, "ma_nhan_vien": n.ma_nhan_vien, "ho_ten": n.ho_ten} for n in nv]