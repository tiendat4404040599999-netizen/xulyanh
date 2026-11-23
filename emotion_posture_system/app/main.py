import json
import re
import secrets
import shutil
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
    create_session_token,
    get_current_user_optional,
    hash_password,
    require_admin_api,
    require_admin_page,
    require_login,
    stop_active_camera_stream,
    verify_password,
)
from app.database import (
    MonitoringSession,
    PasswordResetToken,
    SessionLocal,
    SessionLog,
    SystemConfig,
    User,
    get_db,
    init_db,
)
from app.stream_manager import UPLOADS_DIR, stream_manager
from core.dataset_builder import DEFAULT_IMAGES_DIR, DEFAULT_OUTPUT_DIR, build_dataset_from_images
from core.report_exporter import compute_session_statistics, export_session_csv, export_session_pdf
from models.architecture import validate_and_load_checkpoint
from models.train import DEFAULT_WEIGHTS_DIR, STATUS_FILE, train_bigru_model, write_train_status

BASE_DIR = Path(__file__).resolve().parent.parent
EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_training_lock = threading.Lock()
_training_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(hash_password)
    yield


app = FastAPI(
    title="Hệ thống Dự đoán Trạng thái Cảm xúc qua Tư thế và Đặc trưng Khuôn mặt",
    lifespan=lifespan,
)


def validate_password_strength(password: str) -> Optional[str]:
    """Kiểm tra độ mạnh mật khẩu: tối thiểu 8 ký tự, gồm chữ hoa, chữ thường và chữ số."""
    if len(password) < 8:
        return "Mật khẩu phải có ít nhất 8 ký tự."
    if not re.search(r"[A-Z]", password):
        return "Mật khẩu phải chứa ít nhất 1 chữ cái viết hoa (A-Z)."
    if not re.search(r"[a-z]", password):
        return "Mật khẩu phải chứa ít nhất 1 chữ cái viết thường (a-z)."
    if not re.search(r"\d", password):
        return "Mật khẩu phải chứa ít nhất 1 chữ số (0-9)."
    return None


def require_login_api(user: Optional[User] = Depends(get_current_user_optional)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập đã hết hạn hoặc tài khoản đã bị khóa.")
    return user


def _get_settings_file_path() -> Path:
    return BASE_DIR / "data" / "system_settings.json"


def _load_settings_file() -> Dict[str, Any]:
    path = _get_settings_file_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_settings_file(settings_dict: Dict[str, Any]) -> Path:
    path = _get_settings_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, ensure_ascii=False, indent=2)
    return path


def get_system_config_dict(db: Session) -> Dict[str, Any]:
    cfg = db.query(SystemConfig).first()
    if not cfg:
        cfg = SystemConfig()
        db.add(cfg)
        db.commit()
        db.refresh(cfg)

    file_cfg = _load_settings_file()
    confidence_threshold = float(file_cfg.get("confidence_threshold", 40.0))
    inference_stride = max(1, min(60, int(file_cfg.get("inference_stride", 4))))

    return {
        "camera_source": cfg.camera_source,
        "resolution_width": cfg.resolution_width,
        "resolution_height": cfg.resolution_height,
        "ear_thresh_1": cfg.ear_thresh_1,
        "ear_thresh_2": cfg.ear_thresh_2,
        "perclos_thresh_1": cfg.perclos_thresh_1,
        "perclos_thresh_2": cfg.perclos_thresh_2,
        "max_neck_angle": cfg.max_neck_angle,
        "max_shoulder_tilt": cfg.max_shoulder_tilt,
        "max_rula_score": cfg.max_rula_score,
        "confidence_threshold": confidence_threshold,
        "inference_stride": inference_stride,
        "active_model_path": cfg.active_model_path,
        "active_scaler_path": cfg.active_scaler_path,
    }


# --- Schemas ---
class RoleUpdatePayload(BaseModel):
    role: str


class StatusUpdatePayload(BaseModel):
    is_active: bool


class AdminCreateUserPayload(BaseModel):
    full_name: str
    email: str
    password: str
    role: str = "User"


class OverlayTogglePayload(BaseModel):
    show_face_overlay: bool
    show_pose_overlay: bool


class BulkDeleteSessionsPayload(BaseModel):
    session_ids: List[int]


class ConfigUpdatePayload(BaseModel):
    camera_source: str
    ear_thresh_1: float
    ear_thresh_2: float
    perclos_thresh_1: float
    perclos_thresh_2: float
    max_neck_angle: float
    max_shoulder_tilt: float
    max_rula_score: float
    confidence_threshold: Optional[float] = None
    inference_stride: Optional[int] = None


# =========================================================
# NHÓM 1: QUẢN TRỊ TÀI KHOẢN VÀ PHÂN QUYỀN
# =========================================================
@app.get("/", response_class=HTMLResponse)
def root_redirect(user: Optional[User] = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    if user.role == "Admin":
        return RedirectResponse(url="/admin/users", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/monitor", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, user: Optional[User] = Depends(get_current_user_optional)):
    if user:
        target = "/admin/users" if user.role == "Admin" else "/monitor"
        return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None, "success": request.query_params.get("msg"), "username": ""},
    )


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    clean_identifier = username.strip()
    user = (
        db.query(User)
        .filter((User.username == clean_identifier) | (func.lower(User.email) == clean_identifier.lower()))
        .first()
    )

    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Tên đăng nhập/email hoặc mật khẩu không chính xác.", "success": None, "username": clean_identifier},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    if not user.is_active:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Tài khoản của bạn đã bị quản trị viên khóa. Không thể đăng nhập.", "success": None, "username": clean_identifier},
            status_code=status.HTTP_403_FORBIDDEN,
        )

    token = create_session_token(user.id)
    target_url = "/admin/users" if user.role == "Admin" else "/monitor"
    response = RedirectResponse(url=target_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key=SESSION_COOKIE_NAME, value=token, httponly=True, max_age=SESSION_MAX_AGE, samesite="lax")
    return response


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, user: Optional[User] = Depends(get_current_user_optional)):
    if user:
        return RedirectResponse(url="/monitor", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"error": None, "full_name": "", "email": ""},
    )


@app.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    clean_name = full_name.strip()
    clean_email = email.strip().lower()

    def _err(msg: str, code: int = 400):
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": msg, "full_name": clean_name, "email": clean_email},
            status_code=code,
        )

    if len(clean_name) < 2:
        return _err("Họ và tên phải có tối thiểu 2 ký tự.")
    if not EMAIL_REGEX.match(clean_email):
        return _err("Định dạng địa chỉ email không hợp lệ.")
    if password != confirm_password:
        return _err("Mật khẩu xác nhận không khớp.")

    pw_err = validate_password_strength(password)
    if pw_err:
        return _err(pw_err)

    existed = (
        db.query(User)
        .filter((func.lower(User.email) == clean_email) | (func.lower(User.username) == clean_email))
        .first()
    )
    if existed:
        return _err("Địa chỉ email này đã được đăng ký. Vui lòng sử dụng email khác.")

    new_user = User(
        username=clean_email,
        password_hash=hash_password(password),
        full_name=clean_name,
        email=clean_email,
        role="User",
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    return RedirectResponse(
        url="/login?msg=Đăng+ký+tài+khoản+thành+công!+Vui+lòng+đăng+nhập+bằng+email+vừa+tạo.",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="forgot_password.html",
        context={"step": "request", "email": "", "error": None, "message": None, "demo_code": None},
    )


@app.post("/forgot-password/request", response_class=HTMLResponse)
def forgot_password_request(
    request: Request,
    email: str = Form(...),
    db: Session = Depends(get_db),
):
    clean_email = email.strip().lower()
    if not EMAIL_REGEX.match(clean_email):
        return templates.TemplateResponse(
            request=request,
            name="forgot_password.html",
            context={"step": "request", "email": clean_email, "error": "Định dạng email không hợp lệ.", "message": None, "demo_code": None},
            status_code=400,
        )

    target_user = db.query(User).filter(func.lower(User.email) == clean_email).first()
    if not target_user:
        return templates.TemplateResponse(
            request=request,
            name="forgot_password.html",
            context={"step": "request", "email": clean_email, "error": "Không tìm thấy tài khoản nào gắn với địa chỉ email này.", "message": None, "demo_code": None},
            status_code=404,
        )

    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == target_user.id,
        PasswordResetToken.used == False,
    ).update({"used": True})

    code = f"{secrets.randbelow(1000000):06d}"
    expires_at = datetime.now() + timedelta(minutes=10)
    db.add(
        PasswordResetToken(
            user_id=target_user.id,
            email=clean_email,
            code=code,
            expires_at=expires_at,
            used=False,
        )
    )
    db.commit()
    print(f"[EMAIL RESET DEMO] Mã khôi phục mật khẩu cho {clean_email}: {code} (hết hạn sau 10 phút)")

    return templates.TemplateResponse(
        request=request,
        name="forgot_password.html",
        context={
            "step": "verify",
            "email": clean_email,
            "error": None,
            "message": f"Hệ thống đã tạo mã xác nhận có hiệu lực 10 phút và gửi tới {clean_email}.",
            "demo_code": code,
        },
    )


@app.post("/forgot-password/reset", response_class=HTMLResponse)
def forgot_password_reset(
    request: Request,
    email: str = Form(...),
    code: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    clean_email = email.strip().lower()
    clean_code = code.strip()

    def _err(msg: str):
        return templates.TemplateResponse(
            request=request,
            name="forgot_password.html",
            context={"step": "verify", "email": clean_email, "error": msg, "message": None, "demo_code": None},
            status_code=400,
        )

    token_row = (
        db.query(PasswordResetToken)
        .filter(
            func.lower(PasswordResetToken.email) == clean_email,
            PasswordResetToken.code == clean_code,
            PasswordResetToken.used == False,
        )
        .order_by(PasswordResetToken.id.desc())
        .first()
    )
    if not token_row:
        return _err("Mã xác nhận không chính xác hoặc đã được sử dụng.")
    if token_row.expires_at < datetime.now():
        token_row.used = True
        db.commit()
        return _err("Mã xác nhận đã hết hạn (quá 10 phút). Vui lòng yêu cầu gửi lại mã mới.")
    if new_password != confirm_password:
        return _err("Mật khẩu mới và mật khẩu xác nhận không khớp.")

    pw_err = validate_password_strength(new_password)
    if pw_err:
        return _err(pw_err)

    user = db.query(User).filter(User.id == token_row.user_id).first()
    if not user:
        return _err("Không tìm thấy tài khoản tương ứng.")

    user.password_hash = hash_password(new_password)
    token_row.used = True
    db.commit()

    return RedirectResponse(
        url="/login?msg=Đặt+lại+mật+khẩu+thành+công!+Hãy+đăng+nhập+bằng+mật+khẩu+mới.",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/logout")
def logout_action(user: Optional[User] = Depends(get_current_user_optional)):
    if user:
        stop_active_camera_stream(user.id)
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return response


@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, user: User = Depends(require_login)):
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={"user": user, "active_nav": "profile", "message": None, "error": None, "pw_message": None, "pw_error": None},
    )


@app.post("/profile", response_class=HTMLResponse)
def profile_update(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    user: User = Depends(require_login),
    db: Session = Depends(get_db),
):
    clean_name = full_name.strip()
    clean_email = email.strip()

    if len(clean_name) < 2:
        return templates.TemplateResponse(
            request=request,
            name="profile.html",
            context={"user": user, "active_nav": "profile", "message": None, "error": "Họ và tên tối thiểu 2 ký tự.", "pw_message": None, "pw_error": None},
        )
    if not EMAIL_REGEX.match(clean_email):
        return templates.TemplateResponse(
            request=request,
            name="profile.html",
            context={"user": user, "active_nav": "profile", "message": None, "error": "Định dạng địa chỉ email không hợp lệ.", "pw_message": None, "pw_error": None},
        )

    user.full_name = clean_name
    user.email = clean_email
    db.commit()
    db.refresh(user)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={"user": user, "active_nav": "profile", "message": "Cập nhật thông tin cá nhân thành công!", "error": None, "pw_message": None, "pw_error": None},
    )


@app.post("/profile/change-password", response_class=HTMLResponse)
def profile_change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user: User = Depends(require_login),
    db: Session = Depends(get_db),
):
    def _render(pw_msg=None, pw_err=None, code=200):
        return templates.TemplateResponse(
            request=request,
            name="profile.html",
            context={"user": user, "active_nav": "profile", "message": None, "error": None, "pw_message": pw_msg, "pw_error": pw_err},
            status_code=code,
        )

    if not verify_password(current_password, user.password_hash):
        return _render(pw_err="Mật khẩu hiện tại không chính xác.", code=400)
    if new_password != confirm_password:
        return _render(pw_err="Mật khẩu mới nhập lại hai lần không khớp.", code=400)
    if current_password == new_password:
        return _render(pw_err="Mật khẩu mới phải khác mật khẩu hiện tại.", code=400)

    pw_err = validate_password_strength(new_password)
    if pw_err:
        return _render(pw_err=pw_err, code=400)

    user.password_hash = hash_password(new_password)
    db.commit()
    return _render(pw_msg="Đổi mật khẩu thành công! Bản mã hóa mới đã được lưu vào cơ sở dữ liệu.")


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users_page(request: Request, admin: User = Depends(require_admin_page), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.id.asc()).all()
    return templates.TemplateResponse(
        request=request,
        name="admin_users.html",
        context={"user": admin, "users": users, "active_nav": "admin_users"},
    )


@app.post("/api/admin/users")
def admin_create_user(
    payload: AdminCreateUserPayload,
    admin: User = Depends(require_admin_api),
    db: Session = Depends(get_db),
):
    clean_name = payload.full_name.strip()
    clean_email = payload.email.strip().lower()
    role = payload.role.strip()

    if len(clean_name) < 2:
        raise HTTPException(status_code=400, detail="Họ và tên phải có ít nhất 2 ký tự.")
    if not EMAIL_REGEX.match(clean_email):
        raise HTTPException(status_code=400, detail="Định dạng địa chỉ email không hợp lệ.")
    if role not in ("Admin", "User"):
        raise HTTPException(status_code=400, detail="Vai trò chỉ được phép là 'Admin' hoặc 'User'.")

    pw_err = validate_password_strength(payload.password)
    if pw_err:
        raise HTTPException(status_code=400, detail=f"Mật khẩu tạm chưa đạt yêu cầu: {pw_err}")

    existed = (
        db.query(User)
        .filter((func.lower(User.email) == clean_email) | (func.lower(User.username) == clean_email))
        .first()
    )
    if existed:
        raise HTTPException(status_code=400, detail="Email này đã tồn tại trong hệ thống.")

    new_user = User(
        username=clean_email,
        password_hash=hash_password(payload.password),
        full_name=clean_name,
        email=clean_email,
        role=role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return JSONResponse(
        {
            "message": f"Đã thêm tài khoản '{new_user.full_name}' ({new_user.email}) với vai trò {new_user.role}.",
            "user": {
                "id": new_user.id,
                "username": new_user.username,
                "full_name": new_user.full_name,
                "email": new_user.email,
                "role": new_user.role,
                "is_active": new_user.is_active,
            },
        },
        status_code=201,
    )


@app.delete("/api/admin/users/{target_user_id}")
def admin_delete_user(
    target_user_id: int,
    admin: User = Depends(require_admin_api),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == target_user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản cần xóa.")
    if target.id == admin.id:
        raise HTTPException(status_code=400, detail="Không được phép xóa chính tài khoản quản trị viên đang đăng nhập.")

    stream_manager.stop_stream(target.id)

    user_session_ids = [s.id for s in db.query(MonitoringSession.id).filter(MonitoringSession.user_id == target.id).all()]
    if user_session_ids:
        db.query(SessionLog).filter(SessionLog.session_id.in_(user_session_ids)).delete(synchronize_session=False)
        db.query(MonitoringSession).filter(MonitoringSession.user_id == target.id).delete(synchronize_session=False)

    db.query(PasswordResetToken).filter(PasswordResetToken.user_id == target.id).delete(synchronize_session=False)
    deleted_name = target.full_name
    db.delete(target)
    db.commit()
    return JSONResponse({"message": f"Đã xóa tài khoản '{deleted_name}' cùng toàn bộ dữ liệu phiên giám sát liên quan."})


@app.patch("/api/admin/users/{target_user_id}/role")
def update_user_role(
    target_user_id: int,
    payload: RoleUpdatePayload,
    admin: User = Depends(require_admin_api),
    db: Session = Depends(get_db),
):
    if payload.role not in ("Admin", "User"):
        raise HTTPException(status_code=400, detail="Vai trò chỉ được phép là 'Admin' hoặc 'User'.")
    target = db.query(User).filter(User.id == target_user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
    if target.id == admin.id and payload.role != "Admin":
        raise HTTPException(status_code=400, detail="Không thể tự hạ quyền Admin của chính bạn.")
    target.role = payload.role
    db.commit()
    return JSONResponse({"message": f"Đã cập nhật vai trò của '{target.username}' thành {target.role}.", "role": target.role})


@app.patch("/api/admin/users/{target_user_id}/status")
def toggle_user_status(
    target_user_id: int,
    payload: StatusUpdatePayload,
    admin: User = Depends(require_admin_api),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == target_user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
    if target.id == admin.id and not payload.is_active:
        raise HTTPException(status_code=400, detail="Không thể tự khóa tài khoản Admin đang đăng nhập.")
    target.is_active = payload.is_active
    db.commit()
    if not target.is_active:
        stream_manager.stop_stream(target.id)
    label = "Đã mở khóa (Hoạt động)" if target.is_active else "Đã khóa"
    return JSONResponse({"message": f"Tài khoản '{target.username}': {label}.", "is_active": target.is_active})


# =========================================================
# NHÓM 2: GIÁM SÁT VÀ PHÂN TÍCH TRỰC TUYẾN
# =========================================================
@app.get("/monitor", response_class=HTMLResponse)
def monitor_page(request: Request, user: User = Depends(require_login), db: Session = Depends(get_db)):
    cfg = get_system_config_dict(db)
    return templates.TemplateResponse(
        request=request,
        name="monitor.html",
        context={"user": user, "active_nav": "monitor", "config": cfg},
    )


@app.get("/api/monitor/video-feed")
def monitor_video_feed(user: User = Depends(require_login_api)):
    return StreamingResponse(
        stream_manager.generate_mjpeg(user.id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/monitor/start-camera")
def monitor_start_camera(user: User = Depends(require_login_api), db: Session = Depends(get_db)):
    cfg = get_system_config_dict(db)
    session = stream_manager.get_or_create(user.id, cfg)
    session.start_camera(cfg["camera_source"], cfg)
    return JSONResponse({"message": f"Đang khởi động luồng camera ({cfg['camera_source']}) chuẩn 1280x720."})


@app.post("/api/monitor/stop")
def monitor_stop(user: User = Depends(require_login_api)):
    retained_seconds = stream_manager.stop_stream(user.id)
    if retained_seconds > 0:
        msg = (
            f"Đã tắt camera, giải phóng bộ đệm 60 khung hình và giữ lại {retained_seconds} giây dữ liệu. "
            "Bạn có thể bấm 'Lưu phiên' để lưu vào cơ sở dữ liệu."
        )
    else:
        msg = "Đã tắt camera và giải phóng bộ đệm 60 khung hình."
    return JSONResponse({"message": msg, "retained_seconds": retained_seconds})


@app.post("/api/monitor/upload-video")
async def monitor_upload_video(
    file: UploadFile = File(...),
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(status_code=400, detail="Định dạng video không hỗ trợ. Hãy chọn .mp4, .avi, .mov hoặc .mkv.")

    save_path = UPLOADS_DIR / f"upload_user_{user.id}_{int(time.time())}{ext}"
    with open(save_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    cfg = get_system_config_dict(db)
    session = stream_manager.get_or_create(user.id, cfg)
    session.start_video_file(save_path, cfg)
    return JSONResponse({"message": f"Đã tải lên '{file.filename}'. Hệ thống đang trích xuất 22 đặc trưng và suy luận 60 khung hình..."})


@app.post("/api/monitor/overlays")
def monitor_toggle_overlays(
    payload: OverlayTogglePayload,
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    cfg = get_system_config_dict(db)
    session = stream_manager.get_or_create(user.id, cfg)
    session.set_overlays(payload.show_face_overlay, payload.show_pose_overlay)
    return JSONResponse({"message": "Đã cập nhật lớp phủ đồ họa Overlay."})


@app.get("/api/monitor/state")
def monitor_get_state(user: User = Depends(require_login_api)):
    session = stream_manager.get_existing(user.id)
    if not session:
        return JSONResponse(
            {
                "status": "idle",
                "mode": "idle",
                "running": False,
                "emotion_label": "Chưa khởi động",
                "confidence": 0.0,
                "probabilities": [],
                "ear": 0.0,
                "perclos": 0.0,
                "blink_rate": 0.0,
                "epi_score": 100.0,
                "rula_score": 1.0,
                "posture_color": "green",
                "drowsiness_level": 0,
                "alert_message": "Sẵn sàng giám sát",
                "buffer_count": 0,
                "video_progress_pct": 0,
                "logged_seconds": 0,
                "show_face_overlay": True,
                "show_pose_overlay": True,
                "error_message": None,
            }
        )
    return JSONResponse(session.get_state())


@app.post("/api/monitor/save-session")
def monitor_save_session(user: User = Depends(require_login_api), db: Session = Depends(get_db)):
    session = stream_manager.get_existing(user.id)
    if not session:
        raise HTTPException(status_code=400, detail="Chưa có dữ liệu phiên giám sát để lưu.")

    summary = session.build_session_summary()
    if not summary:
        raise HTTPException(status_code=400, detail="Phiên làm việc chưa đủ dữ liệu (cần giám sát ít nhất 1 giây).")

    db_session = MonitoringSession(
        user_id=user.id,
        start_time=summary["start_time"],
        end_time=summary["end_time"],
        duration_sec=summary["duration_sec"],
        dominant_emotion=summary["dominant_emotion"],
        avg_focus_score=summary["avg_focus_score"],
        drowsiness_alerts_count=summary["drowsiness_alerts_count"],
        avg_rula_score=summary["avg_rula_score"],
        posture_comment=summary["posture_comment"],
    )
    db.add(db_session)
    db.flush()

    for row in summary["logs"]:
        db.add(SessionLog(session_id=db_session.id, **row))

    db.commit()
    return JSONResponse(
        {
            "message": f"Đã lưu phiên giám sát #{db_session.id} ({db_session.duration_sec} giây) vào cơ sở dữ liệu!",
            "session_id": db_session.id,
        }
    )


# =========================================================
# NHÓM 3: LƯU TRỮ VÀ THỐNG KÊ PHIÊN LÀM VIỆC
# =========================================================
@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request, user: User = Depends(require_login), db: Session = Depends(get_db)):
    users_list = []
    if user.role == "Admin":
        users_list = db.query(User).order_by(User.full_name.asc()).all()
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "user": user,
            "active_nav": "history",
            "users_list": users_list,
        },
    )


@app.get("/api/history/sessions")
def history_list_sessions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[int] = None,
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    query = db.query(MonitoringSession)
    if user.role != "Admin":
        query = query.filter(MonitoringSession.user_id == user.id)
    elif user_id is not None:
        query = query.filter(MonitoringSession.user_id == user_id)

    if start_date:
        try:
            dt_start = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(MonitoringSession.start_time >= dt_start)
        except ValueError:
            pass
    if end_date:
        try:
            dt_end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(MonitoringSession.start_time < dt_end)
        except ValueError:
            pass

    rows = query.order_by(MonitoringSession.start_time.desc()).all()
    owner_ids = {s.user_id for s in rows}
    owners_map = {}
    if owner_ids:
        owners = db.query(User).filter(User.id.in_(owner_ids)).all()
        owners_map = {u.id: u for u in owners}

    items = []
    for s in rows:
        owner = owners_map.get(s.user_id)
        items.append(
            {
                "id": s.id,
                "user_id": s.user_id,
                "username": owner.username if owner else "N/A",
                "full_name": owner.full_name if owner else "N/A",
                "start_time": s.start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "duration_sec": s.duration_sec,
                "dominant_emotion": s.dominant_emotion,
                "avg_focus_score": s.avg_focus_score,
                "drowsiness_alerts_count": s.drowsiness_alerts_count,
                "avg_rula_score": s.avg_rula_score,
                "posture_comment": s.posture_comment,
            }
        )
    return JSONResponse({"sessions": items})


@app.post("/api/history/sessions/bulk-delete")
def history_bulk_delete_sessions(
    payload: BulkDeleteSessionsPayload,
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    unique_ids = sorted({int(sid) for sid in payload.session_ids if int(sid) > 0})
    if not unique_ids:
        raise HTTPException(status_code=400, detail="Vui lòng chọn ít nhất một phiên làm việc để xóa.")

    sessions = db.query(MonitoringSession).filter(MonitoringSession.id.in_(unique_ids)).all()
    if not sessions:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên làm việc nào khớp với lựa chọn.")

    if user.role != "Admin":
        unauthorized = [s.id for s in sessions if s.user_id != user.id]
        if unauthorized:
            raise HTTPException(
                status_code=403,
                detail="Bạn không có quyền xóa phiên làm việc của người dùng khác.",
            )

    target_ids = [s.id for s in sessions]
    db.query(SessionLog).filter(SessionLog.session_id.in_(target_ids)).delete(synchronize_session=False)
    db.query(MonitoringSession).filter(MonitoringSession.id.in_(target_ids)).delete(synchronize_session=False)
    db.commit()

    return JSONResponse(
        {
            "message": f"Đã xóa thành công {len(target_ids)} phiên làm việc khỏi cơ sở dữ liệu.",
            "deleted_ids": target_ids,
        }
    )


@app.delete("/api/history/sessions/{session_id}")
def history_delete_single_session(
    session_id: int,
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    session_obj = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên làm việc cần xóa.")
    if user.role != "Admin" and session_obj.user_id != user.id:
        raise HTTPException(status_code=403, detail="Bạn chỉ được phép xóa phiên làm việc của chính mình.")

    db.query(SessionLog).filter(SessionLog.session_id == session_id).delete(synchronize_session=False)
    db.delete(session_obj)
    db.commit()
    return JSONResponse({"message": f"Đã xóa phiên làm việc #{session_id} khỏi cơ sở dữ liệu.", "deleted_ids": [session_id]})


@app.get("/api/history/sessions/{session_id}")
def history_session_detail(
    session_id: int,
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    session_obj = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên làm việc.")
    if user.role != "Admin" and session_obj.user_id != user.id:
        raise HTTPException(status_code=403, detail="Bạn không có quyền xem phiên làm việc này.")

    owner = db.query(User).filter(User.id == session_obj.user_id).first()
    logs = db.query(SessionLog).filter(SessionLog.session_id == session_id).order_by(SessionLog.timestamp_sec.asc()).all()
    stats = compute_session_statistics(session_obj, logs)

    timeline = [
        {
            "time_label": f"{int(row.timestamp_sec // 60):02d}:{int(row.timestamp_sec % 60):02d}",
            "timestamp_sec": row.timestamp_sec,
            "emotion_label": row.emotion_label,
            "confidence": row.confidence,
            "ear": row.ear,
            "perclos": row.perclos,
            "epi_score": row.epi_score,
            "rula_score": row.rula_score,
            "drowsiness_level": row.drowsiness_level,
        }
        for row in logs
    ]

    return JSONResponse(
        {
            "session": {
                "id": session_obj.id,
                "user_id": session_obj.user_id,
                "owner_name": owner.full_name if owner else "N/A",
                "owner_username": owner.username if owner else "N/A",
                "start_time": session_obj.start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "duration_sec": session_obj.duration_sec,
                "dominant_emotion": session_obj.dominant_emotion,
                "avg_focus_score": session_obj.avg_focus_score,
                "drowsiness_alerts_count": session_obj.drowsiness_alerts_count,
                "avg_rula_score": session_obj.avg_rula_score,
                "posture_comment": session_obj.posture_comment,
            },
            "stats": stats,
            "timeline": timeline,
        }
    )


@app.get("/api/history/sessions/{session_id}/export")
def history_export_report(
    session_id: int,
    fmt: str = "csv",
    user: User = Depends(require_login_api),
    db: Session = Depends(get_db),
):
    session_obj = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên làm việc.")
    if user.role != "Admin" and session_obj.user_id != user.id:
        raise HTTPException(status_code=403, detail="Bạn không có quyền xuất báo cáo phiên này.")

    owner = db.query(User).filter(User.id == session_obj.user_id).first() or user
    logs = db.query(SessionLog).filter(SessionLog.session_id == session_id).order_by(SessionLog.timestamp_sec.asc()).all()

    if fmt.lower() == "pdf":
        pdf_bytes = export_session_pdf(session_obj, owner, logs)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="bao_cao_phien_{session_id}.pdf"'},
        )

    csv_bytes = export_session_csv(session_obj, owner, logs)
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="bao_cao_phien_{session_id}.csv"'},
    )


# =========================================================
# NHÓM 4: CẤU HÌNH THAM SỐ, MÔ HÌNH & HUẤN LUYỆN AI
# =========================================================
@app.get("/admin/config", response_class=HTMLResponse)
def admin_config_page(request: Request, admin: User = Depends(require_admin_page), db: Session = Depends(get_db)):
    cfg = get_system_config_dict(db)
    img_count = (
        sum(
            1
            for f in DEFAULT_IMAGES_DIR.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if DEFAULT_IMAGES_DIR.exists()
        else 0
    )
    return templates.TemplateResponse(
        request=request,
        name="admin_config.html",
        context={
            "user": admin,
            "active_nav": "config",
            "config": cfg,
            "existing_images_count": img_count,
        },
    )


@app.post("/api/admin/config")
def admin_save_config(
    payload: ConfigUpdatePayload,
    admin: User = Depends(require_admin_api),
    db: Session = Depends(get_db),
):
    cfg = db.query(SystemConfig).first()
    if not cfg:
        cfg = SystemConfig()
        db.add(cfg)

    file_cfg = _load_settings_file()
    conf_thresh = (
        float(payload.confidence_threshold)
        if payload.confidence_threshold is not None
        else float(file_cfg.get("confidence_threshold", 40.0))
    )
    stride = (
        int(payload.inference_stride)
        if payload.inference_stride is not None
        else int(file_cfg.get("inference_stride", 4))
    )

    if not (0.0 <= conf_thresh <= 100.0):
        raise HTTPException(status_code=400, detail="Ngưỡng tin cậy dự đoán phải nằm trong khoảng 0% đến 100%.")
    if not (1 <= stride <= 60):
        raise HTTPException(status_code=400, detail="Bước trượt dự đoán phải là số nguyên từ 1 đến 60 khung hình.")

    cfg.camera_source = payload.camera_source.strip() or "0"
    cfg.resolution_width = 1280
    cfg.resolution_height = 720
    cfg.ear_thresh_1 = payload.ear_thresh_1
    cfg.ear_thresh_2 = payload.ear_thresh_2
    cfg.perclos_thresh_1 = payload.perclos_thresh_1
    cfg.perclos_thresh_2 = payload.perclos_thresh_2
    cfg.max_neck_angle = payload.max_neck_angle
    cfg.max_shoulder_tilt = payload.max_shoulder_tilt
    cfg.max_rula_score = payload.max_rula_score
    db.commit()

    settings_snapshot = {
        "camera_source": cfg.camera_source,
        "resolution_width": cfg.resolution_width,
        "resolution_height": cfg.resolution_height,
        "ear_thresh_1": cfg.ear_thresh_1,
        "ear_thresh_2": cfg.ear_thresh_2,
        "perclos_thresh_1": cfg.perclos_thresh_1,
        "perclos_thresh_2": cfg.perclos_thresh_2,
        "max_neck_angle": cfg.max_neck_angle,
        "max_shoulder_tilt": cfg.max_shoulder_tilt,
        "max_rula_score": cfg.max_rula_score,
        "confidence_threshold": round(conf_thresh, 1),
        "inference_stride": stride,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_settings_file(settings_snapshot)

    updated_cfg = get_system_config_dict(db)
    with stream_manager._lock:
        for sess in stream_manager._sessions.values():
            sess.engine.update_config(updated_cfg)

    return JSONResponse(
        {
            "message": (
                f"Đã lưu cấu hình vào cơ sở dữ liệu và tệp thiết lập data/system_settings.json "
                f"(Ngưỡng tin cậy: {conf_thresh:.0f}% • Bước trượt: {stride} khung hình)!"
            ),
            "confidence_threshold": round(conf_thresh, 1),
            "inference_stride": stride,
        }
    )


@app.post("/api/admin/upload-weights")
async def admin_upload_weights(
    file: UploadFile = File(...),
    admin: User = Depends(require_admin_api),
    db: Session = Depends(get_db),
):
    filename = file.filename or "uploaded_model.pth"
    ext = Path(filename).suffix.lower()
    if ext not in (".pth", ".h5"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận tệp trọng số mô hình định dạng .pth hoặc .h5.")

    DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = DEFAULT_WEIGHTS_DIR / f"temp_validate_{int(time.time())}{ext}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext == ".h5":
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Hệ thống hiện sử dụng lõi suy luận PyTorch (.pth). Vui lòng tải lên tệp trọng số .pth đúng cấu trúc đầu vào 22 chiều và đầu ra 15 lớp.",
        )

    try:
        validate_and_load_checkpoint(temp_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Từ chối tệp trọng số không hợp lệ: {exc}")

    final_path = DEFAULT_WEIGHTS_DIR / f"custom_bigru_{int(time.time())}.pth"
    temp_path.replace(final_path)

    cfg = db.query(SystemConfig).first()
    if cfg:
        cfg.active_model_path = str(final_path.relative_to(BASE_DIR))
        db.commit()

    stream_manager.reload_all_models(final_path, get_system_config_dict(db))
    return JSONResponse(
        {
            "message": f"Kiểm tra cấu trúc 22x15 thành công! Đã nạp nóng trọng số '{final_path.name}' vào bộ nhớ suy luận.",
            "active_model_path": str(final_path.relative_to(BASE_DIR)),
        }
    )


def _background_train_pipeline(extract_from_images: bool, epochs: int) -> None:
    global _training_running
    try:
        if extract_from_images:
            write_train_status(
                {
                    "state": "extracting",
                    "current_epoch": 0,
                    "total_epochs": epochs,
                    "progress_pct": 5,
                    "message": "Đang quét kho ảnh tiền xử lý 1280x720 và trích xuất 22 đặc trưng MediaPipe...",
                }
            )
            build_dataset_from_images(
                images_dir=DEFAULT_IMAGES_DIR,
                output_dir=DEFAULT_OUTPUT_DIR,
                progress_callback=lambda cur, tot, msg: write_train_status(
                    {
                        "state": "extracting",
                        "current_epoch": 0,
                        "total_epochs": epochs,
                        "progress_pct": max(5, int(cur / max(tot, 1) * 40)),
                        "message": f"[Trích xuất ảnh] {msg}",
                    }
                ),
            )

        out_path = DEFAULT_WEIGHTS_DIR / "best_bigru_model.pth"
        train_bigru_model(
            npy_dir=DEFAULT_OUTPUT_DIR,
            output_model_path=out_path,
            epochs=epochs,
            progress_callback=lambda info: write_train_status(
                {
                    **info,
                    "progress_pct": (40 + int(info["progress_pct"] * 0.6)) if extract_from_images else info["progress_pct"],
                }
            ),
        )

        db = SessionLocal()
        try:
            cfg = db.query(SystemConfig).first()
            if cfg:
                cfg.active_model_path = str(out_path.relative_to(BASE_DIR))
                db.commit()
            stream_manager.reload_all_models(out_path, get_system_config_dict(db))
        finally:
            db.close()
    except Exception as exc:
        write_train_status(
            {
                "state": "error",
                "progress_pct": 0,
                "message": f"Lỗi quá trình huấn luyện: {exc}",
            }
        )
    finally:
        with _training_lock:
            _training_running = False


@app.post("/api/admin/train-model")
async def admin_start_training(
    dataset_zip: Optional[UploadFile] = File(default=None),
    epochs: int = Form(default=20),
    skip_extraction: bool = Form(default=False),
    replace_existing: bool = Form(default=False),
    admin: User = Depends(require_admin_api),
):
    global _training_running
    with _training_lock:
        if _training_running:
            raise HTTPException(status_code=409, detail="Một tiến trình huấn luyện AI khác đang chạy. Vui lòng đợi hoàn tất.")
        _training_running = True

    try:
        epochs = max(2, min(int(epochs), 100))
        extract_needed = not skip_extraction

        if dataset_zip and dataset_zip.filename:
            if not dataset_zip.filename.lower().endswith(".zip"):
                raise HTTPException(status_code=400, detail="Gói dữ liệu ảnh tải lên phải có định dạng nén .zip.")

            zip_tmp = UPLOADS_DIR / f"dataset_{int(time.time())}.zip"
            with open(zip_tmp, "wb") as f:
                shutil.copyfileobj(dataset_zip.file, f)

            if replace_existing and DEFAULT_IMAGES_DIR.exists():
                shutil.rmtree(DEFAULT_IMAGES_DIR)
            DEFAULT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_tmp, "r") as zf:
                zf.extractall(DEFAULT_IMAGES_DIR)
            zip_tmp.unlink(missing_ok=True)
            extract_needed = True

        worker = threading.Thread(
            target=_background_train_pipeline,
            args=(extract_needed, epochs),
            daemon=True,
        )
        worker.start()
        return JSONResponse({"message": "Đã kích hoạt tiến trình huấn luyện mô hình BiGRU ở chế độ chạy nền!"})
    except Exception as exc:
        with _training_lock:
            _training_running = False
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/admin/train-status")
def admin_get_train_status(admin: User = Depends(require_admin_api)):
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["is_running"] = _training_running
            return JSONResponse(data)
        except Exception:
            pass
    return JSONResponse({"state": "idle", "is_running": _training_running, "progress_pct": 0, "message": "Chưa có phiên huấn luyện nào."})