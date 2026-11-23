from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Danh sách chuẩn 15 trạng thái cảm xúc theo đặc tả BFD
EMOTION_CLASSES_15: List[str] = [
    "tinh_tao",        # 0: Tỉnh táo
    "tap_trung",       # 1: Tập trung
    "hung_thu",        # 2: Hứng thú
    "vui_ve",          # 3: Vui vẻ
    "binh_thuong",     # 4: Bình thường
    "suy_nghi",        # 5: Suy nghĩ
    "boi_roi",         # 6: Bối rối
    "ngac_nhien",      # 7: Ngạc nhiên
    "mat_tap_trung",   # 8: Mất tập trung
    "buon_chan",       # 9: Buồn chán
    "met_moi",         # 10: Mệt mỏi
    "buon_ngu",        # 11: Buồn ngủ
    "cang_thang",      # 12: Căng thẳng
    "kho_chiu",        # 13: Khó chịu
    "nghieng_lech",    # 14: Nghiêng lệch tư thế
]

EMOTION_DISPLAY_NAMES: Dict[str, str] = {
    "tinh_tao": "Tỉnh táo",
    "tap_trung": "Tập trung",
    "hung_thu": "Hứng thú",
    "vui_ve": "Vui vẻ",
    "binh_thuong": "Bình thường",
    "suy_nghi": "Suy nghĩ",
    "boi_roi": "Bối rối",
    "ngac_nhien": "Ngạc nhiên",
    "mat_tap_trung": "Mất tập trung",
    "buon_chan": "Buồn chán",
    "met_moi": "Mệt mỏi",
    "buon_ngu": "Buồn ngủ",
    "cang_thang": "Căng thẳng",
    "kho_chiu": "Khó chịu",
    "nghieng_lech": "Nghiêng lệch tư thế",
}

FEATURE_NAMES_22: List[str] = [
    "ear_left",
    "ear_right",
    "ear_avg",
    "eye_closed_flag",
    "perclos",
    "blink_rate",
    "pitch",
    "yaw",
    "roll",
    "mar",
    "mouth_corner_asym",
    "brow_eye_dist_left",
    "brow_eye_dist_right",
    "brow_furrow_dist",
    "shoulder_tilt_angle",
    "shoulder_width_norm",
    "head_shoulder_ratio",
    "camera_distance_est",
    "neck_inclination_angle",
    "neck_velocity",
    "head_center_x_offset",
    "head_center_y_offset",
]

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
POSE_3D_IDX = [1, 152, 33, 263, 61, 291]

MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Đỉnh mũi
        (0.0, -330.0, -65.0),     # Cằm
        (-225.0, 170.0, -135.0),  # Khóe mắt trái
        (225.0, 170.0, -135.0),   # Khóe mắt phải
        (-150.0, -150.0, -125.0), # Khóe miệng trái
        (150.0, -150.0, -125.0),  # Khóe miệng phải
    ],
    dtype=np.float64,
)


def resize_with_black_padding(image: np.ndarray, target_w: int = 1280, target_h: int = 720) -> np.ndarray:
    """Đảm bảo khung hình đúng chuẩn 1280x720 với viền đen giữ nguyên tỷ lệ."""
    h, w = image.shape[:2]
    if w == target_w and h == target_h:
        return image

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


class BiometricFeatureExtractor:
    """Trích xuất vector 22 đặc trưng khuôn mặt & tư thế từ khung hình 1280x720."""

    def __init__(self, static_image_mode: bool = False, ear_closed_threshold: float = 0.22):
        self.static_image_mode = static_image_mode
        self.ear_closed_threshold = ear_closed_threshold

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.eye_closure_history: deque = deque(maxlen=60)
        self.blink_transitions: deque = deque(maxlen=60)
        self.prev_eye_closed: bool = False
        self.prev_neck_point: Optional[np.ndarray] = None

    def reset_temporal_state(self) -> None:
        self.eye_closure_history.clear()
        self.blink_transitions.clear()
        self.prev_eye_closed = False
        self.prev_neck_point = None

    @staticmethod
    def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _compute_ear(self, landmarks_px: np.ndarray, indices: List[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [landmarks_px[i] for i in indices]
        v1 = self._euclidean(p2, p6)
        v2 = self._euclidean(p3, p5)
        h = self._euclidean(p1, p4)
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _compute_head_pose(self, landmarks_px: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float]:
        image_points = np.array([landmarks_px[i] for i in POSE_3D_IDX], dtype=np.float64)
        focal_length = float(img_w)
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            MODEL_3D_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
        pitch = float(np.clip(angles[0], -90.0, 90.0))
        yaw = float(np.clip(angles[1], -90.0, 90.0))
        roll = float(np.clip(angles[2], -90.0, 90.0))
        return pitch, yaw, roll

    def extract(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        frame = resize_with_black_padding(frame_bgr, 1280, 720)
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)

        features = np.zeros(22, dtype=np.float32)
        meta: Dict[str, Any] = {
            "face_detected": False,
            "pose_detected": False,
            "face_landmarks_px": None,
            "nose_px": None,
            "left_shoulder_px": None,
            "right_shoulder_px": None,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "ear": 0.0,
            "perclos": 0.0,
            "blink_rate": 0.0,
            "epi_score": 100.0,
            "rula_score": 1.0,
            "posture_color": "green",
        }

        face_width_px = 160.0
        nose_pt = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)

        # 1. Trích xuất 14 đặc trưng khuôn mặt từ MediaPipe Face Mesh
        if face_results.multi_face_landmarks:
            meta["face_detected"] = True
            flm = face_results.multi_face_landmarks[0].landmark
            pts_px = np.array([[lm.x * img_w, lm.y * img_h] for lm in flm], dtype=np.float32)
            meta["face_landmarks_px"] = pts_px

            ear_left = self._compute_ear(pts_px, LEFT_EYE_IDX)
            ear_right = self._compute_ear(pts_px, RIGHT_EYE_IDX)
            ear_avg = (ear_left + ear_right) / 2.0

            eye_closed = ear_avg < self.ear_closed_threshold
            eye_closed_flag = 1.0 if eye_closed else 0.0

            if self.static_image_mode:
                perclos = float(np.clip((0.28 - ear_avg) / 0.14 * 100.0, 0.0, 100.0)) if ear_avg < 0.28 else 0.0
                blink_rate = 18.0 if eye_closed else 12.0
            else:
                self.eye_closure_history.append(1.0 if eye_closed else 0.0)
                is_blink = 1.0 if (not self.prev_eye_closed and eye_closed) else 0.0
                self.blink_transitions.append(is_blink)
                self.prev_eye_closed = eye_closed
                perclos = float(np.mean(self.eye_closure_history) * 100.0)
                blink_rate = float(np.sum(self.blink_transitions) * (30.0 / max(1, len(self.blink_transitions))))

            pitch, yaw, roll = self._compute_head_pose(pts_px, img_w, img_h)

            face_width_px = max(self._euclidean(pts_px[234], pts_px[454]), 1.0)
            mouth_w = max(self._euclidean(pts_px[61], pts_px[291]), 1.0)
            mouth_h = self._euclidean(pts_px[13], pts_px[14])
            mar = mouth_h / mouth_w
            mouth_corner_asym = (pts_px[61][1] - pts_px[291][1]) / face_width_px

            brow_eye_left = self._euclidean(pts_px[105], pts_px[159]) / face_width_px
            brow_eye_right = self._euclidean(pts_px[334], pts_px[386]) / face_width_px
            brow_furrow = self._euclidean(pts_px[107], pts_px[336]) / face_width_px

            nose_pt = pts_px[1]
            meta["nose_px"] = tuple(nose_pt.astype(int))
            meta["pitch"] = pitch
            meta["yaw"] = yaw
            meta["roll"] = roll
            meta["ear"] = ear_avg
            meta["perclos"] = perclos
            meta["blink_rate"] = blink_rate

            features[0:14] = [
                ear_left,
                ear_right,
                ear_avg,
                eye_closed_flag,
                perclos,
                blink_rate,
                pitch,
                yaw,
                roll,
                mar,
                mouth_corner_asym,
                brow_eye_left,
                brow_eye_right,
                brow_furrow,
            ]

        # 2. Trích xuất 8 đặc trưng tư thế thân trên từ MediaPipe Pose
        if pose_results.pose_landmarks:
            meta["pose_detected"] = True
            plm = pose_results.pose_landmarks.landmark
            pose_nose = np.array([plm[0].x * img_w, plm[0].y * img_h], dtype=np.float32)
            left_sh = np.array([plm[11].x * img_w, plm[11].y * img_h], dtype=np.float32)
            right_sh = np.array([plm[12].x * img_w, plm[12].y * img_h], dtype=np.float32)

            if not meta["face_detected"]:
                nose_pt = pose_nose
                meta["nose_px"] = tuple(pose_nose.astype(int))

            meta["left_shoulder_px"] = tuple(left_sh.astype(int))
            meta["right_shoulder_px"] = tuple(right_sh.astype(int))

            dx = right_sh[0] - left_sh[0]
            dy = right_sh[1] - left_sh[1]
            shoulder_tilt = float(np.degrees(np.arctan2(dy, abs(dx) + 1e-6)))
            shoulder_width_px = max(self._euclidean(left_sh, right_sh), 1.0)
            shoulder_width_norm = shoulder_width_px / float(img_w)
            head_shoulder_ratio = face_width_px / shoulder_width_px
            camera_distance_est = float(np.clip((40.0 * img_w) / shoulder_width_px, 30.0, 250.0))

            shoulder_mid = (left_sh + right_sh) / 2.0
            neck_vec = nose_pt - shoulder_mid
            neck_inclination = float(np.degrees(np.arctan2(abs(neck_vec[0]), abs(neck_vec[1]) + 1e-6)))

            if self.static_image_mode or self.prev_neck_point is None:
                neck_velocity = 0.0
            else:
                neck_velocity = self._euclidean(shoulder_mid, self.prev_neck_point) / float(img_w)
            self.prev_neck_point = shoulder_mid.copy()

            head_x_offset = float(neck_vec[0] / shoulder_width_px)
            head_y_offset = float(abs(neck_vec[1]) / shoulder_width_px)

            features[14:22] = [
                shoulder_tilt,
                shoulder_width_norm,
                head_shoulder_ratio,
                camera_distance_est,
                neck_inclination,
                neck_velocity,
                head_x_offset,
                head_y_offset,
            ]

            epi_score, rula_score, posture_color = self.evaluate_ergonomics(
                neck_angle=neck_inclination,
                shoulder_tilt=abs(shoulder_tilt),
                head_y_offset=head_y_offset,
            )
            meta["epi_score"] = epi_score
            meta["rula_score"] = rula_score
            meta["posture_color"] = posture_color

        return features, meta

    @staticmethod
    def evaluate_ergonomics(
        neck_angle: float,
        shoulder_tilt: float,
        head_y_offset: float,
        max_neck_angle: float = 20.0,
        max_shoulder_tilt: float = 10.0,
        max_rula: float = 4.0,
    ) -> Tuple[float, float, str]:
        neck_penalty = max(0.0, (neck_angle - 8.0) / max(max_neck_angle, 1.0)) * 35.0
        tilt_penalty = max(0.0, (shoulder_tilt - 3.0) / max(max_shoulder_tilt, 1.0)) * 30.0
        slouch_penalty = max(0.0, (0.55 - head_y_offset) / 0.25) * 35.0 if head_y_offset > 0 else 0.0

        epi_score = float(np.clip(100.0 - (neck_penalty + tilt_penalty + slouch_penalty), 0.0, 100.0))
        rula_score = float(np.clip(1.0 + (100.0 - epi_score) / 100.0 * 6.0, 1.0, 7.0))

        if rula_score <= max(2.5, max_rula * 0.65):
            color = "green"
        elif rula_score <= max_rula:
            color = "yellow"
        else:
            color = "red"

        return round(epi_score, 1), round(rula_score, 1), color

    def close(self) -> None:
        self.face_mesh.close()
        self.pose.close()