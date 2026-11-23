import hashlib
import hmac
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from app.database import User, get_db

SECRET_KEY = "emotion-posture-ai-secret-key-change-in-production"
SESSION_COOKIE_NAME = "eps_session_token"
SESSION_MAX_AGE = 60 * 60 * 8  # 8 giờ

serializer = URLSafeTimedSerializer(SECRET_KEY)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, hex_hash = stored_hash.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return hmac.compare_digest(dk.hex(), hex_hash)


def create_session_token(user_id: int) -> str:
    return serializer.dumps({"user_id": user_id})


def decode_session_token(token: str) -> Optional[int]:
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("user_id")
    except (BadSignature, SignatureExpired):
        return None


def stop_active_camera_stream(user_id: int) -> None:
    """
    Khi người dùng bấm Đăng xuất, hệ thống lập tức ngắt luồng thu nhận hình ảnh webcam
    nếu đang chạy và giải phóng tài nguyên phần cứng.
    """
    from app.stream_manager import stream_manager

    stream_manager.stop_stream(user_id)
    print(f"[INFO] Đã ngắt luồng camera và giải phóng tài nguyên cho user_id={user_id}.")


def get_current_user_optional(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    user_id = decode_session_token(token)
    if not user_id:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    # Nếu tài khoản đã bị khóa, lập tức vô hiệu hóa phiên hiện tại
    if not user or not user.is_active:
        return None
    return user


def require_login(user: Optional[User] = Depends(get_current_user_optional)) -> User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    return user


def require_admin_page(user: User = Depends(require_login)) -> User:
    if user.role != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền Quản trị viên (Admin) để truy cập chức năng này.",
        )
    return user


def require_admin_api(user: Optional[User] = Depends(get_current_user_optional)) -> User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập đã hết hạn hoặc tài khoản đã bị khóa.",
        )
    if user.role != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Quyền truy cập bị từ chối. Yêu cầu vai trò Admin.",
        )
    return user