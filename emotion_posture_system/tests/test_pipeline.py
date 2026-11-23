import io
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.auth as auth_module
import app.database as db_module
import app.main as main_module
import models.train as train_module
from core.feature_extractor import EMOTION_CLASSES_15, EMOTION_DISPLAY_NAMES, FEATURE_NAMES_22
from core.inference_engine import RealTimeInferenceEngine
from models.architecture import validate_and_load_checkpoint
from models.train import train_bigru_model


def _create_synthetic_npy_dataset(npy_dir: Path) -> Path:
    """Tạo bộ dữ liệu giả lập chuẩn (N, 60, 22) + StandardScaler + label_map.json trong thư mục tạm."""
    npy_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=123)

    X_train = rng.normal(0.0, 1.0, size=(30, 60, 22)).astype(np.float32)
    y_train = np.array([i % 3 for i in range(30)], dtype=np.int64)
    X_val = rng.normal(0.0, 1.0, size=(9, 60, 22)).astype(np.float32)
    y_val = np.array([i % 3 for i in range(9)], dtype=np.int64)
    X_test = rng.normal(0.0, 1.0, size=(9, 60, 22)).astype(np.float32)
    y_test = np.array([i % 3 for i in range(9)], dtype=np.int64)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, 22)).reshape(X_train.shape).astype(np.float32)
    X_val_s = scaler.transform(X_val.reshape(-1, 22)).reshape(X_val.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, 22)).reshape(X_test.shape).astype(np.float32)

    np.save(npy_dir / "X_train.npy", X_train_s)
    np.save(npy_dir / "y_train.npy", y_train)
    np.save(npy_dir / "X_val.npy", X_val_s)
    np.save(npy_dir / "y_val.npy", y_val)
    np.save(npy_dir / "X_test.npy", X_test_s)
    np.save(npy_dir / "y_test.npy", y_test)

    with open(npy_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    label_map = {
        "num_classes": 15,
        "window_size": 60,
        "num_features": 22,
        "feature_names": FEATURE_NAMES_22,
        "label_to_id": {name: i for i, name in enumerate(EMOTION_CLASSES_15)},
        "id_to_label": {str(i): name for i, name in enumerate(EMOTION_CLASSES_15)},
        "display_names": EMOTION_DISPLAY_NAMES,
    }
    with open(npy_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    return npy_dir


@pytest.fixture()
def test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Khởi tạo môi trường DB SQLite tạm độc lập và FastAPI TestClient."""
    db_file = tmp_path / "test_emotion_posture.db"
    test_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(main_module, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(auth_module, "SessionLocal", TestSessionLocal, raising=False)

    db_module.Base.metadata.create_all(bind=test_engine)
    db_module.init_db(auth_module.hash_password)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    main_module.app.dependency_overrides[db_module.get_db] = override_get_db

    with TestClient(main_module.app) as client:
        yield client, TestSessionLocal, tmp_path

    main_module.app.dependency_overrides.clear()
    test_engine.dispose()


def _login(client: TestClient, username: str, password: str):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )


# =========================================================================
# CA KIỂM THỬ 1: NHÓM 1 — QUẢN TRỊ TÀI KHOẢN & PHÂN QUYỀN
# =========================================================================
def test_group1_auth_rbac_profile_and_logout(test_env, monkeypatch: pytest.MonkeyPatch):
    client, SessionLocal, _ = test_env

    # 1. Đăng nhập tài khoản User thường -> bị chặn 403 khi gọi API của Admin
    res_user = _login(client, "user1", "user123")
    assert res_user.status_code == 303
    assert res_user.headers["location"] == "/monitor"

    assert client.post(
        "/api/admin/config",
        json={
            "camera_source": "0",
            "ear_thresh_1": 0.25,
            "ear_thresh_2": 0.20,
            "perclos_thresh_1": 20.0,
            "perclos_thresh_2": 40.0,
            "max_neck_angle": 20.0,
            "max_shoulder_tilt": 10.0,
            "max_rula_score": 4.0,
        },
    ).status_code == 403

    # 2. Cập nhật hồ sơ cá nhân (từ chối email sai, chấp nhận email đúng)
    bad_profile = client.post("/profile", data={"full_name": "Nguyễn Văn A", "email": "sai-dinh-dang"})
    assert bad_profile.status_code == 200
    assert "không hợp lệ" in bad_profile.text

    good_profile = client.post("/profile", data={"full_name": "Nguyễn Văn A", "email": "nguyenvana@example.com"})
    assert good_profile.status_code == 200
    assert "thành công" in good_profile.text

    # 3. Kiểm tra đăng xuất: xóa cookie và gọi hàm ngắt luồng camera
    stopped_users = []
    monkeypatch.setattr(main_module, "stop_active_camera_stream", lambda uid: stopped_users.append(uid))
    res_logout = client.post("/logout", follow_redirects=False)
    assert res_logout.status_code == 303
    assert len(stopped_users) == 1
    assert auth_module.SESSION_COOKIE_NAME not in client.cookies

    # 4. Đăng nhập Admin -> đổi quyền và khóa tài khoản user1
    res_admin = _login(client, "admin", "admin123")
    assert res_admin.status_code == 303
    assert res_admin.headers["location"] == "/admin/users"

    db = SessionLocal()
    user1 = db.query(db_module.User).filter(db_module.User.username == "user1").first()
    user1_id = user1.id
    db.close()

    # Đổi quyền user1 -> Admin rồi trả về User
    assert client.patch(f"/api/admin/users/{user1_id}/role", json={"role": "Admin"}).status_code == 200
    assert client.patch(f"/api/admin/users/{user1_id}/role", json={"role": "User"}).status_code == 200

    # Khóa tài khoản user1 -> xác nhận user1 bị chặn đăng nhập (403)
    assert client.patch(f"/api/admin/users/{user1_id}/status", json={"is_active": False}).status_code == 200
    client.post("/logout", follow_redirects=False)

    blocked_login = _login(client, "user1", "user123")
    assert blocked_login.status_code == 403
    assert "khóa" in blocked_login.text.lower()

    # Admin mở khóa lại cho user1 -> đăng nhập bình thường
    _login(client, "admin", "admin123")
    assert client.patch(f"/api/admin/users/{user1_id}/status", json={"is_active": True}).status_code == 200
    client.post("/logout", follow_redirects=False)
    assert _login(client, "user1", "user123").status_code == 303


# =========================================================================
# CA KIỂM THỬ 2: PIPELINE HUẤN LUYỆN AI & XÁC THỰC CHECKPOINT 22x15
# =========================================================================
def test_step2_step3_training_and_checkpoint_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    npy_dir = _create_synthetic_npy_dataset(tmp_path / "processed_npy")
    weights_dir = tmp_path / "weights"
    model_path = weights_dir / "test_bigru.pth"

    monkeypatch.setattr(train_module, "DEFAULT_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(train_module, "STATUS_FILE", weights_dir / "train_status.json")

    result = train_bigru_model(
        npy_dir=npy_dir,
        output_model_path=model_path,
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
    )

    assert result["state"] == "completed"
    assert model_path.exists()

    model, ckpt = validate_and_load_checkpoint(model_path)
    assert ckpt["input_dim"] == 22
    assert ckpt["num_classes"] == 15
    assert ckpt["scaler_bytes"] is not None

    dummy = torch.randn(2, 60, 22)
    out = model(dummy)
    assert out.shape == (2, 15)


# =========================================================================
# CA KIỂM THỬ 3: NHÓM 4 — CẤU HÌNH THAM SỐ & RÀNG BUỘC TRỌNG SỐ 22x15
# =========================================================================
def test_group4_config_and_weights_validation(test_env, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = test_env

    # Cách ly đường dẫn lưu trọng số vào thư mục tạm
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(main_module, "DEFAULT_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(train_module, "DEFAULT_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(train_module, "STATUS_FILE", weights_dir / "train_status.json")

    _login(client, "admin", "admin123")

    # 1. Cập nhật cấu hình camera 1280x720 và ngưỡng sinh trắc
    cfg_res = client.post(
        "/api/admin/config",
        json={
            "camera_source": "http://192.168.1.50:8080/video",
            "ear_thresh_1": 0.24,
            "ear_thresh_2": 0.19,
            "perclos_thresh_1": 22.0,
            "perclos_thresh_2": 42.0,
            "max_neck_angle": 18.0,
            "max_shoulder_tilt": 9.0,
            "max_rula_score": 4.0,
        },
    )
    assert cfg_res.status_code == 200

    # 2. Tải lên trọng số SAI cấu trúc (10 lớp thay vì 15) -> phải bị từ chối 400
    bad_buf = io.BytesIO()
    torch.save({"input_dim": 22, "num_classes": 10, "model_state_dict": {}}, bad_buf)
    bad_buf.seek(0)

    bad_upload = client.post(
        "/api/admin/upload-weights",
        files={"file": ("wrong_model.pth", bad_buf, "application/octet-stream")},
    )
    assert bad_upload.status_code == 400

    # 3. Huấn luyện nhanh trọng số CHUẨN 22x15 và tải lên -> phải chấp nhận 200
    npy_dir = _create_synthetic_npy_dataset(tmp_path / "npy_for_upload")
    valid_model_path = tmp_path / "valid_bigru.pth"
    train_bigru_model(npy_dir=npy_dir, output_model_path=valid_model_path, epochs=2, batch_size=8)

    with open(valid_model_path, "rb") as f:
        good_upload = client.post(
            "/api/admin/upload-weights",
            files={"file": ("valid_bigru.pth", f, "application/octet-stream")},
        )
    assert good_upload.status_code == 200
    assert "active_model_path" in good_upload.json()


# =========================================================================
# CA KIỂM THỬ 4: NHÓM 2 — LÕI SUY LUẬN 60 KHUNG HÌNH, EPI/RULA & BUỒN NGỦ
# =========================================================================
def test_group2_realtime_engine_60_frames_ergonomics_and_drowsiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    npy_dir = _create_synthetic_npy_dataset(tmp_path / "npy_engine")
    weights_dir = tmp_path / "weights_engine"
    model_path = weights_dir / "engine_bigru.pth"
    monkeypatch.setattr(train_module, "DEFAULT_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(train_module, "STATUS_FILE", weights_dir / "train_status.json")

    train_bigru_model(npy_dir=npy_dir, output_model_path=model_path, epochs=2, batch_size=8)

    engine = RealTimeInferenceEngine(model_path=model_path, scaler_path=npy_dir / "scaler.pkl")
    try:
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        telemetry = None
        for _ in range(60):
            annotated, telemetry = engine.process_frame(dummy_frame, show_face_overlay=True, show_pose_overlay=True)

        assert annotated.shape == (720, 1280, 3)
        assert telemetry is not None
        assert telemetry["model_loaded"] is True
        assert telemetry["buffer_count"] == 60
        assert len(telemetry["probabilities"]) == 15

        prob_sum = sum(p["prob"] for p in telemetry["probabilities"])
        assert 99.0 <= prob_sum <= 101.0

        assert 0.0 <= telemetry["epi_score"] <= 100.0
        assert 1.0 <= telemetry["rula_score"] <= 7.0
        assert telemetry["posture_color"] in {"green", "yellow", "red"}

        # Kiểm tra logic cảnh báo buồn ngủ 3 cấp độ (Mức 0, Mức 1, Mức 2)
        lvl0, _ = engine._evaluate_drowsiness(ear=0.31, perclos=5.0, face_detected=True)
        assert lvl0 == 0

        lvl1, _ = engine._evaluate_drowsiness(ear=0.22, perclos=25.0, face_detected=True)
        assert lvl1 == 1

        lvl2, _ = engine._evaluate_drowsiness(ear=0.14, perclos=55.0, face_detected=True)
        assert lvl2 == 2
    finally:
        engine.close()


# =========================================================================
# CA KIỂM THỬ 5: NHÓM 3 — LƯU TRỮ PHIÊN, BIỂU ĐỒ & XUẤT CSV / PDF
# =========================================================================
def test_group3_history_charts_and_csv_pdf_export(test_env):
    client, SessionLocal, _ = test_env

    _login(client, "user1", "user123")

    # Tạo 1 phiên giám sát mẫu có chuỗi nhật ký vào DB tạm
    db = SessionLocal()
    user1 = db.query(db_module.User).filter(db_module.User.username == "user1").first()
    now = datetime.now()
    sample_session = db_module.MonitoringSession(
        user_id=user1.id,
        start_time=now - timedelta(seconds=10),
        end_time=now,
        duration_sec=10,
        dominant_emotion="Tập trung",
        avg_focus_score=85.0,
        drowsiness_alerts_count=1,
        avg_rula_score=2.4,
        posture_comment="Thói quen ngồi rất tốt (80.0% thời gian đạt tư thế an toàn).",
    )
    db.add(sample_session)
    db.flush()

    probs_json = json.dumps({EMOTION_DISPLAY_NAMES[k]: (55.0 if k == "tap_trung" else 3.2) for k in EMOTION_CLASSES_15}, ensure_ascii=False)
    for sec in range(1, 6):
        db.add(
            db_module.SessionLog(
                session_id=sample_session.id,
                timestamp_sec=float(sec),
                emotion_label="Tập trung",
                emotion_probs_json=probs_json,
                confidence=88.5,
                ear=0.28,
                blink_rate=14.0,
                perclos=8.5,
                epi_score=86.0 if sec <= 4 else 62.0,
                rula_score=1.8 if sec <= 4 else 3.6,
                posture_color="green" if sec <= 4 else "yellow",
                drowsiness_level=0 if sec <= 4 else 1,
            )
        )
    db.commit()
    session_id = sample_session.id
    db.close()

    # 1. Kiểm tra danh sách & bộ lọc ngày tháng
    today_str = now.strftime("%Y-%m-%d")
    list_res = client.get(f"/api/history/sessions?start_date={today_str}&end_date={today_str}")
    assert list_res.status_code == 200
    sessions = list_res.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)

    # 2. Kiểm tra dữ liệu chi tiết phục vụ Line Chart, Pie Chart & Bar Chart 15 cảm xúc
    detail_res = client.get(f"/api/history/sessions/{session_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert len(detail["timeline"]) == 5
    assert "green_pct" in detail["stats"]["posture_breakdown"]
    assert len(detail["stats"]["emotion_distribution"]) == 15

    # 3. Kiểm tra xuất báo cáo CSV (UTF-8 BOM + đủ chỉ số)
    csv_res = client.get(f"/api/history/sessions/{session_id}/export?fmt=csv")
    assert csv_res.status_code == 200
    assert csv_res.content.startswith(b"\xef\xbb\xbf")
    csv_text = csv_res.content.decode("utf-8-sig")
    assert "EAR" in csv_text and "PERCLOS" in csv_text and "EPI" in csv_text and "RULA" in csv_text

    # 4. Kiểm tra xuất báo cáo PDF hợp lệ
    pdf_res = client.get(f"/api/history/sessions/{session_id}/export?fmt=pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.content.startswith(b"%PDF")
    assert len(pdf_res.content) > 1024