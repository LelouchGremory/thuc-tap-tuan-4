from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from database import Base

class Employee(Base):
    """Bảng lưu trữ thông tin nhân viên"""
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, index=True)
    ma_nhan_vien = Column(String, unique=True, index=True, nullable=False)
    ho_ten = Column(String, nullable=False)
    phong_ban = Column(String)
    ngay_tao = Column(DateTime(timezone=True), server_default=func.now())

    face_profiles = relationship("FaceProfile", back_populates="employee", cascade="all, delete-orphan")
    recognition_events = relationship("RecognitionEvent", back_populates="employee")
    checkin_events = relationship("CheckinEvent", back_populates="employee")

class FaceProfile(Base):
    """Bảng lưu trữ vector đặc trưng khuôn mặt của nhân viên"""
    __tablename__ = "face_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    embedding = Column(Vector(128))  # Vector 128 chiều
    phien_ban_model = Column(String, default="sface_2021dec")
    ngay_tao = Column(DateTime(timezone=True), server_default=func.now())

    employee = relationship("Employee", back_populates="face_profiles")

class Camera(Base):
    """Bảng lưu trữ cấu hình thiết bị Camera"""
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    ten_camera = Column(String, nullable=False)
    dia_chi_ip = Column(String, nullable=False, unique=True)
    vi_tri = Column(String)
    trang_thai = Column(Boolean, default=True)
    ngay_tao = Column(DateTime(timezone=True), server_default=func.now())

    recognition_events = relationship("RecognitionEvent", back_populates="camera")

class RecognitionEvent(Base):
    """Bảng lưu trữ lịch sử nhận diện khuôn mặt thô từ camera"""
    __tablename__ = "recognition_events"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    do_tin_cay = Column(Float)
    duong_dan_anh = Column(String)
    thoi_gian = Column(DateTime(timezone=True), server_default=func.now())

    camera = relationship("Camera", back_populates="recognition_events")
    employee = relationship("Employee", back_populates="recognition_events")

class CheckinEvent(Base):
    """Bảng lưu trữ sự kiện điểm danh chính thức sau khi phân tích logic"""
    __tablename__ = "checkin_events"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    thoi_gian = Column(DateTime(timezone=True), server_default=func.now())
    trang_thai = Column(String, nullable=False)

    employee = relationship("Employee", back_populates="checkin_events")
