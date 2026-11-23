import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cv2
import numpy as np

from core.inference_engine import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH, RealTimeInferenceEngine

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _open_capture(source: str) -> cv2.VideoCapture:
    """Mở nguồn webcam vật lý (0, 1...) hoặc URL luồng IP Camera từ điện thoại."""
    src_str = str(source).strip()
    if src_str.isdigit():
        idx = int(src_str)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(src_str)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


class UserStreamSession:
    """Quản lý trạng thái luồng camera/video, overlay và nhật ký 1 giây/lần cho từng người dùng."""

    def __init__(self, user_id: int, config: Dict[str, Any]):
        self.user_id = user_id
        self.lock = threading.Lock()
        self.running = False
        self.mode = "idle"  # 'idle' | 'camera' | 'video_processing' | 'video_replay'
        self.error_message: Optional[str] = None

        self.show_face_overlay = True
        self.show_pose_overlay = True

        self.video_progress_pct = 0
        self.video_total_frames = 0
        self.video_processed_frames = 0

        self.latest_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {
            "status": "idle",
            "mode": "idle",
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
            "buffer_ready": False,
            "video_progress_pct": 0,
            "logged_seconds": 0,
        }

        self.session_start_time: Optional[datetime] = None
        self.session_end_time: Optional[datetime] = None
        self.logs_buffer: List[Dict[str, Any]] = []
        self._last_log_ts: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._processed_video_path: Optional[Path] = None

        model_path = Path(config.get("active_model_path") or DEFAULT_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path
        scaler_path = Path(config.get("active_scaler_path") or DEFAULT_SCALER_PATH)
        if not scaler_path.is_absolute():
            scaler_path = BASE_DIR / scaler_path

        self.engine = RealTimeInferenceEngine(
            model_path=model_path,
            scaler_path=scaler_path,
            config=config,
        )

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return buf.tobytes() if ok else None

    def _record_log_if_due(self, telemetry: Dict[str, Any], force_sec: Optional[float] = None) -> None:
        now = time.time()
        if force_sec is None:
            if now - self._last_log_ts < 1.0:
                return
            self._last_log_ts = now
            elapsed = (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0.0
        else:
            elapsed = force_sec

        probs_dict = {p["label"]: p["prob"] for p in telemetry.get("probabilities", [])}
        self.logs_buffer.append(
            {
                "timestamp_sec": round(elapsed, 1),
                "emotion_label": telemetry.get("emotion_label", "Tỉnh táo"),
                "emotion_probs_json": json.dumps(probs_dict, ensure_ascii=False),
                "confidence": float(telemetry.get("confidence", 0.0)),
                "ear": float(telemetry.get("ear", 0.0)),
                "blink_rate": float(telemetry.get("blink_rate", 0.0)),
                "perclos": float(telemetry.get("perclos", 0.0)),
                "epi_score": float(telemetry.get("epi_score", 100.0)),
                "rula_score": float(telemetry.get("rula_score", 1.0)),
                "posture_color": str(telemetry.get("posture_color", "green")),
                "drowsiness_level": int(telemetry.get("drowsiness_level", 0)),
            }
        )

    def start_camera(self, camera_source: str, config: Dict[str, Any]) -> None:
        self.stop()
        self.engine.update_config(config)
        self.engine.reset()

        with self.lock:
            self.running = True
            self.mode = "camera"
            self.error_message = None
            self.logs_buffer = []
            self.session_start_time = datetime.now()
            self.session_end_time = None
            self._last_log_ts = 0.0

        self._thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_source,),
            daemon=True,
        )
        self._thread.start()

    def _camera_loop(self, camera_source: str) -> None:
        cap = _open_capture(camera_source)
        if not cap.isOpened():
            with self.lock:
                self.running = False
                self.mode = "idle"
                self.error_message = f"Không thể mở nguồn camera '{camera_source}'. Hãy kiểm tra kết nối hoặc cấu hình Admin."
                self.latest_telemetry["status"] = "error"
                self.latest_telemetry["alert_message"] = self.error_message
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                with self.lock:
                    show_face = self.show_face_overlay
                    show_pose = self.show_pose_overlay

                annotated, telemetry = self.engine.process_frame(
                    frame,
                    show_face_overlay=show_face,
                    show_pose_overlay=show_pose,
                )
                jpeg = self._encode_jpeg(annotated)

                with self.lock:
                    self._record_log_if_due(telemetry)
                    telemetry["status"] = "running"
                    telemetry["mode"] = "camera"
                    telemetry["video_progress_pct"] = 0
                    telemetry["logged_seconds"] = len(self.logs_buffer)
                    self.latest_jpeg = jpeg
                    self.latest_telemetry = telemetry
        finally:
            cap.release()
            with self.lock:
                if self.mode == "camera":
                    self.mode = "idle"
                    self.latest_telemetry["status"] = "stopped"

    def start_video_file(self, video_path: Path, config: Dict[str, Any]) -> None:
        self.stop()
        self.engine.update_config(config)
        self.engine.reset()

        with self.lock:
            self.running = True
            self.mode = "video_processing"
            self.error_message = None
            self.video_progress_pct = 0
            self.video_processed_frames = 0
            self.logs_buffer = []
            self.session_start_time = datetime.now()
            self.session_end_time = None

        self._thread = threading.Thread(
            target=self._video_worker,
            args=(Path(video_path),),
            daemon=True,
        )
        self._thread.start()

    def _video_worker(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            with self.lock:
                self.running = False
                self.mode = "idle"
                self.error_message = "Không thể đọc tệp video tải lên."
                self.latest_telemetry["status"] = "error"
                self.latest_telemetry["alert_message"] = self.error_message
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1.0 or fps > 120.0:
            fps = 25.0
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        log_interval_frames = max(int(round(fps)), 1)

        out_avi = UPLOADS_DIR / f"annotated_user_{self.user_id}.avi"
        writer = cv2.VideoWriter(
            str(out_avi),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (1280, 720),
        )
        timeline_telemetry: List[Dict[str, Any]] = []

        try:
            frame_idx = 0
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                with self.lock:
                    show_face = self.show_face_overlay
                    show_pose = self.show_pose_overlay

                annotated, telemetry = self.engine.process_frame(
                    frame,
                    show_face_overlay=show_face,
                    show_pose_overlay=show_pose,
                )
                writer.write(annotated)

                pct = min(100, int(round(frame_idx / total_frames * 100)))
                if frame_idx % log_interval_frames == 0 or frame_idx == total_frames:
                    self._record_log_if_due(telemetry, force_sec=frame_idx / fps)

                telemetry["status"] = "processing"
                telemetry["mode"] = "video_processing"
                telemetry["video_progress_pct"] = pct
                telemetry["logged_seconds"] = len(self.logs_buffer)
                timeline_telemetry.append(telemetry)

                if frame_idx % 3 == 0 or frame_idx == total_frames:
                    jpeg = self._encode_jpeg(annotated)
                    with self.lock:
                        self.video_progress_pct = pct
                        self.video_processed_frames = frame_idx
                        self.video_total_frames = total_frames
                        self.latest_jpeg = jpeg
                        self.latest_telemetry = telemetry
        finally:
            cap.release()
            writer.release()

        if not self.running:
            return

        self._processed_video_path = out_avi
        with self.lock:
            self.mode = "video_replay"
            self.video_progress_pct = 100

        replay_cap = cv2.VideoCapture(str(out_avi))
        frame_delay = 1.0 / min(fps, 30.0)
        try:
            idx = 0
            while self.running and replay_cap.isOpened():
                ret, frame = replay_cap.read()
                if not ret:
                    replay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx = 0
                    continue

                jpeg = self._encode_jpeg(frame)
                tel = timeline_telemetry[min(idx, len(timeline_telemetry) - 1)] if timeline_telemetry else self.latest_telemetry
                tel_copy = dict(tel)
                tel_copy["status"] = "replaying"
                tel_copy["mode"] = "video_replay"
                tel_copy["video_progress_pct"] = 100
                tel_copy["logged_seconds"] = len(self.logs_buffer)

                with self.lock:
                    self.latest_jpeg = jpeg
                    self.latest_telemetry = tel_copy

                idx += 1
                time.sleep(frame_delay)
        finally:
            replay_cap.release()

    def stop(self) -> int:
        """
        Ngắt kết nối webcam/video, giải phóng bộ nhớ đệm cửa sổ trượt 60 khung hình,
        đồng thời giữ nguyên dữ liệu phiên (logs_buffer) đã thu thập để người dùng lưu phiên.
        """
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None

        # Giải phóng bộ nhớ đệm cửa sổ trượt 60 khung hình & bộ đếm tạm của MediaPipe
        self.engine.reset()

        with self.lock:
            self.mode = "idle"
            self.latest_jpeg = None
            if self.logs_buffer and self.session_end_time is None:
                self.session_end_time = datetime.now()
            retained_seconds = len(self.logs_buffer)

            self.latest_telemetry["status"] = "stopped"
            self.latest_telemetry["mode"] = "idle"
            self.latest_telemetry["running"] = False
            self.latest_telemetry["buffer_count"] = 0
            self.latest_telemetry["buffer_ready"] = False
            self.latest_telemetry["drowsiness_level"] = 0
            self.latest_telemetry["logged_seconds"] = retained_seconds
            self.latest_telemetry["alert_message"] = (
                f"Đã dừng camera và giải phóng bộ đệm 60 khung hình. Giữ lại {retained_seconds}s dữ liệu để lưu phiên."
                if retained_seconds > 0
                else "Đã dừng camera và giải phóng bộ đệm 60 khung hình."
            )
            return retained_seconds

    def set_overlays(self, show_face: bool, show_pose: bool) -> None:
        with self.lock:
            self.show_face_overlay = show_face
            self.show_pose_overlay = show_pose

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            state = dict(self.latest_telemetry)
            state["show_face_overlay"] = self.show_face_overlay
            state["show_pose_overlay"] = self.show_pose_overlay
            state["error_message"] = self.error_message
            state["running"] = self.running
            return state

    def build_session_summary(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not self.logs_buffer:
                return None
            logs = list(self.logs_buffer)
            start_dt = self.session_start_time or datetime.now()
            end_dt = self.session_end_time or datetime.now()

        duration_sec = max(int(round(logs[-1]["timestamp_sec"])), len(logs))
        emotions = [row["emotion_label"] for row in logs]
        dominant_emotion = Counter(emotions).most_common(1)[0][0]

        positive_focus = {"Tỉnh táo", "Tập trung", "Hứng thú", "Suy nghĩ", "Vui vẻ"}
        focus_ratio = sum(1 for e in emotions if e in positive_focus) / max(len(emotions), 1) * 100.0
        drowsiness_alerts = sum(1 for row in logs if row["drowsiness_level"] >= 1)
        avg_rula = sum(row["rula_score"] for row in logs) / max(len(logs), 1)
        safe_posture_pct = sum(1 for row in logs if row["posture_color"] == "green") / max(len(logs), 1) * 100.0

        if safe_posture_pct >= 75.0:
            comment = f"Thói quen ngồi rất tốt ({safe_posture_pct:.1f}% thời gian đạt tư thế an toàn)."
        elif safe_posture_pct >= 45.0:
            comment = f"Tư thế ở mức trung bình ({safe_posture_pct:.1f}% an toàn). Cần chú ý giữ thẳng cổ và cân bằng vai."
        else:
            comment = f"Cảnh báo thói quen ngồi sai tư thế nhiều ({100.0 - safe_posture_pct:.1f}% thời gian lệch trục). Nên điều chỉnh lại ghế và màn hình."

        return {
            "start_time": start_dt,
            "end_time": end_dt,
            "duration_sec": duration_sec,
            "dominant_emotion": dominant_emotion,
            "avg_focus_score": round(focus_ratio, 1),
            "drowsiness_alerts_count": drowsiness_alerts,
            "avg_rula_score": round(avg_rula, 2),
            "posture_comment": comment,
            "logs": logs,
        }

    def close(self) -> None:
        self.stop()
        self.engine.close()


class StreamManager:
    """Bộ điều phối trung tâm quản lý toàn bộ luồng giám sát của người dùng."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[int, UserStreamSession] = {}

    def get_or_create(self, user_id: int, config: Dict[str, Any]) -> UserStreamSession:
        with self._lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = UserStreamSession(user_id=user_id, config=config)
            else:
                self._sessions[user_id].engine.update_config(config)
            return self._sessions[user_id]

    def get_existing(self, user_id: int) -> Optional[UserStreamSession]:
        with self._lock:
            return self._sessions.get(user_id)

    def stop_stream(self, user_id: int) -> int:
        session = self.get_existing(user_id)
        if session:
            return session.stop()
        return 0

    def reload_all_models(self, model_path: Path, config: Dict[str, Any]) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            was_running = session.running
            if was_running:
                session.stop()
            session.engine.update_config(config)
            session.engine.load_model(model_path)

    def generate_mjpeg(self, user_id: int) -> Generator[bytes, None, None]:
        """Sinh luồng MJPEG 1280x720 truyền trực tiếp lên thẻ <img> trên giao diện Web."""
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "CAMERA STANDBY (1280 x 720) - CLICK START TO MONITOR",
            (230, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (180, 210, 230),
            2,
            cv2.LINE_AA,
        )
        _, ph_buf = cv2.imencode(".jpg", placeholder)
        ph_bytes = ph_buf.tobytes()

        while True:
            session = self.get_existing(user_id)
            if not session or not session.running:
                frame_bytes = session.latest_jpeg if (session and session.latest_jpeg) else ph_bytes
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                time.sleep(0.2)
                continue

            with session.lock:
                frame_bytes = session.latest_jpeg or ph_bytes

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            time.sleep(0.033)


stream_manager = StreamManager()