import math
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from core.feature_extractor import (
    EMOTION_CLASSES_15,
    EMOTION_DISPLAY_NAMES,
    BiometricFeatureExtractor,
    resize_with_black_padding,
)
from models.architecture import DualStreamBiGRU, validate_and_load_checkpoint

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "weights" / "best_bigru_model.pth"
DEFAULT_SCALER_PATH = BASE_DIR / "data" / "processed_npy" / "scaler.pkl"

LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
OUTER_LIPS_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


class RealTimeInferenceEngine:
    """
    Lõi xử lý thời gian thực cho Nhóm 2 & Nhóm 4:
    - Trích xuất 22 đặc trưng từ khung hình 1280x720
    - Duy trì cửa sổ trượt 60 khung hình & suy luận mạng DualStreamBiGRU theo bước trượt cấu hình
    - Áp dụng ngưỡng tin cậy tối thiểu (trả về 'Không xác định' khi xác suất thấp hơn ngưỡng)
    - Đánh giá công thái học EPI/RULA, cảnh báo buồn ngủ 3 cấp độ & vẽ Overlay tối giản
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        scaler_path: Path = DEFAULT_SCALER_PATH,
        config: Optional[Dict[str, Any]] = None,
        inference_stride: int = 4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._default_config()
        self.config["inference_stride"] = max(1, min(60, int(inference_stride)))
        if config:
            self.config.update(config)
        self.inference_stride = max(1, min(60, int(self.config.get("inference_stride", 4))))

        self.model: Optional[DualStreamBiGRU] = None
        self.scaler = None
        self.class_keys: List[str] = list(EMOTION_CLASSES_15)
        self.display_names: Dict[str, str] = dict(EMOTION_DISPLAY_NAMES)
        self.model_loaded: bool = False
        self.model_path: Path = Path(model_path)

        self.load_model(Path(model_path), Path(scaler_path))

        self.extractor = BiometricFeatureExtractor(
            static_image_mode=False,
            ear_closed_threshold=float(self.config["ear_thresh_1"]),
        )
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.feature_buffer: deque = deque(maxlen=60)
        self.frame_counter: int = 0
        self.consecutive_closed_frames: int = 0

        self.latest_emotion_key: str = "tinh_tao"
        self.latest_emotion_name: str = "Đang thu thập 60 khung hình..."
        self.latest_confidence: float = 0.0
        self.latest_is_uncertain: bool = False
        self.latest_probs: List[Dict[str, Any]] = self._empty_probs()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "ear_thresh_1": 0.25,
            "ear_thresh_2": 0.20,
            "perclos_thresh_1": 20.0,
            "perclos_thresh_2": 40.0,
            "max_neck_angle": 20.0,
            "max_shoulder_tilt": 10.0,
            "max_rula_score": 4.0,
            "confidence_threshold": 40.0,
            "inference_stride": 4,
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)
        self.extractor.ear_closed_threshold = float(self.config["ear_thresh_1"])
        self.inference_stride = max(1, min(60, int(self.config.get("inference_stride", self.inference_stride))))

    def _empty_probs(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": key,
                "label": self.display_names.get(key, key.replace("_", " ").title()),
                "prob": 0.0,
            }
            for key in self.class_keys
        ]

    def load_model(self, model_path: Path, fallback_scaler_path: Path = DEFAULT_SCALER_PATH) -> None:
        self.model_path = Path(model_path)
        self.model_loaded = False
        if not self.model_path.exists():
            print(f"[WARN] Chưa tìm thấy tệp trọng số tại {self.model_path}.")
            return

        model, ckpt = validate_and_load_checkpoint(self.model_path, device=self.device)
        self.model = model
        self.model_loaded = True

        scaler_bytes = ckpt.get("scaler_bytes")
        if scaler_bytes:
            self.scaler = pickle.loads(scaler_bytes)
        elif fallback_scaler_path.exists():
            with open(fallback_scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        if isinstance(ckpt.get("display_names"), dict):
            self.display_names.update(ckpt["display_names"])

        label_map = ckpt.get("label_map") or {}
        id_to_label = label_map.get("id_to_label")
        if isinstance(id_to_label, dict):
            keys = []
            for idx in range(15):
                k = id_to_label.get(str(idx), EMOTION_CLASSES_15[idx])
                keys.append(k)
            self.class_keys = keys

        self.latest_probs = self._empty_probs()

    def reset(self) -> None:
        self.feature_buffer.clear()
        self.extractor.reset_temporal_state()
        self.frame_counter = 0
        self.consecutive_closed_frames = 0
        self.latest_emotion_key = "tinh_tao"
        self.latest_emotion_name = "Đang thu thập 60 khung hình..."
        self.latest_confidence = 0.0
        self.latest_is_uncertain = False
        self.latest_probs = self._empty_probs()

    def _enhance_frame_clarity(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Khử mờ ngược sáng nhẹ bằng CLAHE kênh độ sáng để khuôn mặt rõ và tự nhiên hơn."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self.clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(enhanced, 0.35, frame_bgr, 0.65, 0)

    def _evaluate_drowsiness(self, ear: float, perclos: float, face_detected: bool) -> Tuple[int, str]:
        if not face_detected:
            self.consecutive_closed_frames = 0
            return 0, "Chưa phát hiện khuôn mặt"

        ear_t1 = float(self.config["ear_thresh_1"])
        ear_t2 = float(self.config["ear_thresh_2"])
        per_t1 = float(self.config["perclos_thresh_1"])
        per_t2 = float(self.config["perclos_thresh_2"])

        if ear < ear_t2:
            self.consecutive_closed_frames += 1
        elif ear < ear_t1:
            self.consecutive_closed_frames = max(0, self.consecutive_closed_frames)
        else:
            self.consecutive_closed_frames = max(0, self.consecutive_closed_frames - 2)

        if self.consecutive_closed_frames >= 20 or perclos >= per_t2:
            return 2, "CẢNH BÁO NGUY HIỂM: Phát hiện buồn ngủ / nhắm mắt kéo dài!"

        if ear < ear_t1 or perclos >= per_t1 or self.consecutive_closed_frames >= 8:
            return 1, "Nhắc nhở: Có dấu hiệu mệt mỏi, hãy nghỉ ngơi mắt."

        return 0, "Tỉnh táo"

    def _run_bigru_inference(self) -> None:
        if not self.model_loaded or self.model is None or len(self.feature_buffer) < 60:
            return

        seq = np.array(self.feature_buffer, dtype=np.float32)
        if self.scaler is not None:
            seq = self.scaler.transform(seq).astype(np.float32)

        tensor_x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_prob_pct = round(float(probs[top_idx]) * 100.0, 1)
        conf_threshold = float(self.config.get("confidence_threshold", 40.0))

        self.latest_confidence = top_prob_pct
        if top_prob_pct < conf_threshold:
            self.latest_emotion_key = "khong_xac_dinh"
            self.latest_emotion_name = "Không xác định"
            self.latest_is_uncertain = True
        else:
            self.latest_emotion_key = self.class_keys[top_idx]
            self.latest_emotion_name = self.display_names.get(
                self.latest_emotion_key,
                self.latest_emotion_key.replace("_", " ").title(),
            )
            self.latest_is_uncertain = False

        self.latest_probs = [
            {
                "key": self.class_keys[i],
                "label": self.display_names.get(self.class_keys[i], self.class_keys[i].replace("_", " ").title()),
                "prob": round(float(probs[i]) * 100.0, 2),
            }
            for i in range(15)
        ]

    @staticmethod
    def _draw_face_overlay(frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Vẽ khung nhận diện tối giản kiểu Face ID & đường viền mắt/môi bán trong suốt."""
        pts = meta.get("face_landmarks_px")
        if pts is None:
            return

        h, w = frame.shape[:2]
        min_x = max(8, int(np.min(pts[:, 0])) - 24)
        max_x = min(w - 8, int(np.max(pts[:, 0])) + 24)
        min_y = max(8, int(np.min(pts[:, 1])) - 36)
        max_y = min(h - 8, int(np.max(pts[:, 1])) + 20)

        overlay = frame.copy()
        soft_cyan = (235, 210, 60)
        soft_rose = (200, 160, 240)
        for contour, color in [
            (LEFT_EYE_CONTOUR, soft_cyan),
            (RIGHT_EYE_CONTOUR, soft_cyan),
            (OUTER_LIPS_CONTOUR, soft_rose),
        ]:
            poly = np.array([pts[i] for i in contour], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)

        box_layer = frame.copy()
        bracket_color = (245, 215, 70)
        corner_len = max(18, min(34, (max_x - min_x) // 6))
        t = 2

        cv2.line(box_layer, (min_x, min_y), (min_x + corner_len, min_y), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (min_x, min_y), (min_x, min_y + corner_len), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (max_x, min_y), (max_x - corner_len, min_y), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (max_x, min_y), (max_x, min_y + corner_len), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (min_x, max_y), (min_x + corner_len, max_y), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (min_x, max_y), (min_x, max_y - corner_len), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (max_x, max_y), (max_x - corner_len, max_y), bracket_color, t, cv2.LINE_AA)
        cv2.line(box_layer, (max_x, max_y), (max_x, max_y - corner_len), bracket_color, t, cv2.LINE_AA)

        cx = min(w - 36, max_x + 28)
        cy = max(36, min_y + 24)
        pitch = math.radians(meta.get("pitch", 0.0))
        yaw = math.radians(meta.get("yaw", 0.0))
        axis_len = 22.0
        gx = int(cx + axis_len * math.sin(yaw))
        gy = int(cy - axis_len * math.sin(pitch) * math.cos(yaw))

        cv2.circle(box_layer, (cx, cy), 18, (180, 190, 200), 1, cv2.LINE_AA)
        cv2.line(box_layer, (cx, cy), (gx, gy), (80, 235, 255), 2, cv2.LINE_AA)
        cv2.circle(box_layer, (gx, gy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.addWeighted(box_layer, 0.78, frame, 0.22, 0, frame)

    @staticmethod
    def _draw_pose_overlay(frame: np.ndarray, meta: Dict[str, Any]) -> None:
        """Vẽ thanh căn chỉnh vai & trục cổ ngắn dừng gọn dưới cằm."""
        l_sh = meta.get("left_shoulder_px")
        r_sh = meta.get("right_shoulder_px")
        if l_sh is None or r_sh is None:
            return

        color_map = {
            "green": (110, 235, 145),
            "yellow": (50, 205, 255),
            "red": (105, 105, 255),
        }
        bone_color = color_map.get(meta.get("posture_color", "green"), (110, 235, 145))
        mid_sh = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)

        overlay = frame.copy()
        sh_span = max(abs(r_sh[0] - l_sh[0]) // 2, 60)
        cv2.line(
            overlay,
            (mid_sh[0] - sh_span, mid_sh[1]),
            (mid_sh[0] + sh_span, mid_sh[1]),
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.line(overlay, l_sh, r_sh, bone_color, 2, cv2.LINE_AA)

        nose = meta.get("nose_px")
        if nose is not None:
            neck_top = (
                int(mid_sh[0] + (nose[0] - mid_sh[0]) * 0.35),
                int(mid_sh[1] + (nose[1] - mid_sh[1]) * 0.35),
            )
            cv2.line(overlay, mid_sh, neck_top, bone_color, 2, cv2.LINE_AA)
            cv2.circle(overlay, neck_top, 4, bone_color, -1, cv2.LINE_AA)

        for pt in (l_sh, r_sh):
            cv2.circle(overlay, pt, 5, bone_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, mid_sh, 4, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        show_face_overlay: bool = True,
        show_pose_overlay: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        frame = resize_with_black_padding(frame_bgr, 1280, 720)
        feat_22, meta = self.extractor.extract(frame)

        if meta["pose_detected"]:
            epi_score, rula_score, posture_color = self.extractor.evaluate_ergonomics(
                neck_angle=float(feat_22[18]),
                shoulder_tilt=abs(float(feat_22[14])),
                head_y_offset=float(feat_22[21]),
                max_neck_angle=float(self.config["max_neck_angle"]),
                max_shoulder_tilt=float(self.config["max_shoulder_tilt"]),
                max_rula=float(self.config["max_rula_score"]),
            )
            meta["epi_score"] = epi_score
            meta["rula_score"] = rula_score
            meta["posture_color"] = posture_color

        self.feature_buffer.append(feat_22)
        self.frame_counter += 1

        if len(self.feature_buffer) == 60 and (
            self.frame_counter % self.inference_stride == 0 or self.latest_confidence == 0.0
        ):
            self._run_bigru_inference()

        drowsiness_level, alert_message = self._evaluate_drowsiness(
            ear=float(meta["ear"]),
            perclos=float(meta["perclos"]),
            face_detected=bool(meta["face_detected"]),
        )

        telemetry: Dict[str, Any] = {
            "timestamp": time.time(),
            "model_loaded": self.model_loaded,
            "buffer_count": len(self.feature_buffer),
            "buffer_ready": len(self.feature_buffer) >= 60,
            "face_detected": bool(meta["face_detected"]),
            "pose_detected": bool(meta["pose_detected"]),
            "emotion_key": self.latest_emotion_key,
            "emotion_label": self.latest_emotion_name,
            "confidence": self.latest_confidence,
            "is_uncertain": self.latest_is_uncertain,
            "confidence_threshold": float(self.config.get("confidence_threshold", 40.0)),
            "inference_stride": self.inference_stride,
            "probabilities": self.latest_probs,
            "ear": round(float(meta["ear"]), 3),
            "perclos": round(float(meta["perclos"]), 1),
            "blink_rate": round(float(meta["blink_rate"]), 1),
            "pitch": round(float(meta["pitch"]), 1),
            "yaw": round(float(meta["yaw"]), 1),
            "roll": round(float(meta["roll"]), 1),
            "shoulder_tilt": round(float(feat_22[14]), 1),
            "neck_angle": round(float(feat_22[18]), 1),
            "camera_distance_cm": round(float(feat_22[17]), 1),
            "epi_score": float(meta["epi_score"]),
            "rula_score": float(meta["rula_score"]),
            "posture_color": str(meta["posture_color"]),
            "drowsiness_level": drowsiness_level,
            "alert_message": alert_message,
        }

        annotated = self._enhance_frame_clarity(frame)
        if show_pose_overlay:
            self._draw_pose_overlay(annotated, meta)
        if show_face_overlay:
            self._draw_face_overlay(annotated, meta)

        return annotated, telemetry

    def close(self) -> None:
        self.extractor.close()