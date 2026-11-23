from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "emotion_posture.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(120), nullable=False)
    email = Column(String(120), nullable=False)
    role = Column(String(20), default="User", nullable=False)  # 'Admin' | 'User'
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    sessions = relationship("MonitoringSession", back_populates="user", cascade="all, delete-orphan")


class MonitoringSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.now, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_sec = Column(Integer, default=0)
    dominant_emotion = Column(String(50), default="Tỉnh táo")
    avg_focus_score = Column(Float, default=0.0)
    drowsiness_alerts_count = Column(Integer, default=0)
    avg_rula_score = Column(Float, default=1.0)
    posture_comment = Column(Text, default="")

    user = relationship("User", back_populates="sessions")
    logs = relationship("SessionLog", back_populates="session", cascade="all, delete-orphan")


class SessionLog(Base):
    __tablename__ = "session_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    emotion_label = Column(String(50), nullable=False)
    emotion_probs_json = Column(Text, default="{}")  # Lưu xác suất 15 trạng thái cảm xúc
    confidence = Column(Float, default=0.0)
    ear = Column(Float, default=0.0)
    blink_rate = Column(Float, default=0.0)
    perclos = Column(Float, default=0.0)
    epi_score = Column(Float, default=0.0)
    rula_score = Column(Float, default=1.0)
    posture_color = Column(String(10), default="green")  # 'green' | 'yellow' | 'red'
    drowsiness_level = Column(Integer, default=0)        # 0 | 1 | 2

    session = relationship("MonitoringSession", back_populates="logs")


class SystemConfig(Base):
    __tablename__ = "system_configs"

    id = Column(Integer, primary_key=True, index=True)
    camera_source = Column(String(255), default="0")     # Chỉ số webcam hoặc URL IP Stream
    resolution_width = Column(Integer, default=1280)
    resolution_height = Column(Integer, default=720)
    ear_thresh_1 = Column(Float, default=0.25)           # Ngưỡng EAR Mức 1
    ear_thresh_2 = Column(Float, default=0.20)           # Ngưỡng EAR Mức 2
    perclos_thresh_1 = Column(Float, default=20.0)       # Ngưỡng PERCLOS (%) Mức 1
    perclos_thresh_2 = Column(Float, default=40.0)       # Ngưỡng PERCLOS (%) Mức 2
    max_neck_angle = Column(Float, default=20.0)
    max_shoulder_tilt = Column(Float, default=10.0)
    max_rula_score = Column(Float, default=4.0)
    active_model_path = Column(String(255), default="models/weights/best_bigru_model.pth")
    active_scaler_path = Column(String(255), default="data/processed_npy/scaler.pkl")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(120), nullable=False, index=True)
    code = Column(String(16), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
def init_db(hash_password_fn):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Tạo cấu hình mặc định nếu chưa có
        if not db.query(SystemConfig).first():
            db.add(SystemConfig())

        # Tạo tài khoản mẫu để kiểm thử phân quyền Admin / User
        if not db.query(User).filter(User.username == "admin").first():
            db.add(
                User(
                    username="admin",
                    password_hash=hash_password_fn("admin123"),
                    full_name="Quản trị viên Hệ thống",
                    email="admin@emotion-ai.local",
                    role="Admin",
                    is_active=True,
                )
            )
        if not db.query(User).filter(User.username == "user1").first():
            db.add(
                User(
                    username="user1",
                    password_hash=hash_password_fn("user123"),
                    full_name="Nguyễn Văn An",
                    email="user1@emotion-ai.local",
                    role="User",
                    is_active=True,
                )
            )
        if not db.query(User).filter(User.username == "user2").first():
            db.add(
                User(
                    username="user2",
                    password_hash=hash_password_fn("user123"),
                    full_name="Trần Thị Bình",
                    email="user2@emotion-ai.local",
                    role="User",
                    is_active=True,
                )
            )
        db.commit()
    finally:
        db.close()
        