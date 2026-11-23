import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.feature_extractor import (
    EMOTION_CLASSES_15,
    EMOTION_DISPLAY_NAMES,
    FEATURE_NAMES_22,
    BiometricFeatureExtractor,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = BASE_DIR / "data" / "preprocessed_images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed_npy"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class_key(name: str) -> str:
    """Chuẩn hóa tên thư mục: bỏ dấu tiếng Việt, về chữ thường, đổi khoảng trắng/gạch ngang thành gạch dưới."""
    text = unicodedata.normalize("NFD", name.strip().lower().replace("đ", "d"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[\s\-]+", "_", text)
    return text.strip("_")


# Bảng tra tên thư mục -> nhãn chuẩn (chấp nhận cả khóa như "tap_trung" lẫn tên hiển thị như "Tập trung")
CLASS_ALIASES: Dict[str, str] = {normalize_class_key(k): k for k in EMOTION_CLASSES_15}
CLASS_ALIASES.update(
    {normalize_class_key(v): k for k, v in EMOTION_DISPLAY_NAMES.items() if k in EMOTION_CLASSES_15}
)


def imread_unicode(img_path: Path) -> Optional[np.ndarray]:
    """Đọc ảnh an toàn kể cả khi đường dẫn hoặc tên file có dấu tiếng Việt."""
    try:
        raw_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def create_60_frame_sequence(
    base_features: np.ndarray,
    window_size: int = 60,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Chuyển vector 22 đặc trưng của 1 ảnh tĩnh thành chuỗi cửa sổ trượt (60, 22)."""
    if rng is None:
        rng = np.random.default_rng()

    seq = np.repeat(base_features[np.newaxis, :], window_size, axis=0).astype(np.float32)

    noise_scale = np.maximum(np.abs(base_features) * 0.015, 0.002).astype(np.float32)
    noise_scale[3] = 0.0
    noise_scale[19] = 0.0

    noise = rng.normal(loc=0.0, scale=noise_scale, size=(window_size, 22)).astype(np.float32)
    seq = seq + noise

    neck_diff = np.abs(np.diff(seq[:, 18], prepend=seq[0, 18])) * 0.002
    seq[:, 19] = neck_diff.astype(np.float32)
    return seq


def resolve_class_name(img_path: Path, root_dir: Path) -> Optional[str]:
    """Xác định nhãn cảm xúc từ thư mục chứa ảnh.

    Duyệt từ thư mục sát ảnh nhất ra ngoài và lấy thư mục đầu tiên trùng với 1 trong 15 nhãn chuẩn.
    Trả về None nếu ảnh không nằm trong thư mục cảm xúc hợp lệ (ví dụ thư mục nguồn dữ liệu như cnn, coco, mpii)
    để ảnh đó bị bỏ qua thay vì bị gán nhãn sai.
    """
    rel_parts = img_path.relative_to(root_dir).parts[:-1]
    for part in reversed(rel_parts):
        class_name = CLASS_ALIASES.get(normalize_class_key(part))
        if class_name is not None:
            return class_name
    return None


def build_dataset_from_images(
    images_dir: Path = DEFAULT_IMAGES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    window_size: int = 60,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

    # Bảng nhãn cố định 15 lớp, không tự sinh thêm nhãn mới từ tên thư mục lạ
    label_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(EMOTION_CLASSES_15)}

    image_tasks: List[tuple[Path, int, str]] = []
    ignored_counts: Dict[str, int] = {}

    # Quét đệ quy toàn bộ ảnh trong mọi thư mục con bên dưới preprocessed_images
    all_files = sorted(
        f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS
    )

    for img_path in all_files:
        class_name = resolve_class_name(img_path, images_dir)
        if class_name is None:
            rel_parts = img_path.relative_to(images_dir).parts[:-1]
            group = rel_parts[0] if rel_parts else "(thư mục gốc)"
            ignored_counts[group] = ignored_counts.get(group, 0) + 1
            continue
        image_tasks.append((img_path, label_to_id[class_name], class_name))

    if not image_tasks:
        ignored_text = ", ".join(f"{name} ({cnt} ảnh)" for name, cnt in sorted(ignored_counts.items())) or "không có"
        raise ValueError(
            "Không có ảnh nào nằm trong thư mục mang tên trạng thái cảm xúc hợp lệ. "
            f"Hãy chia ảnh vào các thư mục con đặt đúng tên: {', '.join(EMOTION_CLASSES_15)}. "
            f"Các thư mục đã bị bỏ qua: {ignored_text}."
        )

    if ignored_counts:
        print(f"Cảnh báo: bỏ qua ảnh không thuộc thư mục cảm xúc hợp lệ: {ignored_counts}")

    total_images = len(image_tasks)
    if total_images < 4:
        raise ValueError(
            f"Cần tối thiểu 4 ảnh bên trong '{images_dir}' để chia tập Train/Val/Test (hiện có: {total_images} ảnh)."
        )

    scanned_counts: Dict[str, int] = {}
    for _, _, name in image_tasks:
        scanned_counts[name] = scanned_counts.get(name, 0) + 1
    print(f"Tìm thấy {total_images} ảnh thuộc các nhãn: {scanned_counts}")

    extractor = BiometricFeatureExtractor(static_image_mode=True)
    rng = np.random.default_rng(seed=42)

    X_sequences: List[np.ndarray] = []
    y_labels: List[int] = []
    skipped_images = 0

    try:
        for idx, (img_path, class_id, folder_name) in enumerate(image_tasks, start=1):
            img = imread_unicode(img_path)
            if img is None:
                skipped_images += 1
                continue

            feat_22, meta = extractor.extract(img)
            if not meta["face_detected"] and not meta["pose_detected"]:
                skipped_images += 1
                continue

            seq_60x22 = create_60_frame_sequence(feat_22, window_size=window_size, rng=rng)
            X_sequences.append(seq_60x22)
            y_labels.append(class_id)

            if progress_callback and (idx % 20 == 0 or idx == total_images):
                progress_callback(idx, total_images, f"Đã xử lý {idx}/{total_images} ảnh ({folder_name})")
    finally:
        extractor.close()

    valid_count = len(X_sequences)
    if valid_count < 4:
        raise ValueError(f"Chỉ có {valid_count} ảnh nhận diện được khuôn mặt/tư thế (cần tối thiểu 4 ảnh).")

    X = np.stack(X_sequences, axis=0).astype(np.float32)  # (N, 60, 22)
    y = np.array(y_labels, dtype=np.int64)                # (N,)

    # Số mẫu hợp lệ của từng nhãn trong 15 lớp (0 nếu chưa có dữ liệu)
    class_counts: Dict[str, int] = {
        name: int(np.sum(y == idx)) for idx, name in enumerate(EMOTION_CLASSES_15)
    }
    missing_classes = [name for name, cnt in class_counts.items() if cnt == 0]
    if len(EMOTION_CLASSES_15) - len(missing_classes) < 2:
        raise ValueError(
            "Dữ liệu chỉ có 1 nhãn cảm xúc nên không thể huấn luyện bộ phân loại. "
            f"Cần ảnh của ít nhất 2 nhãn trong: {', '.join(EMOTION_CLASSES_15)}."
        )

    _, counts = np.unique(y, return_counts=True)
    can_stratify = bool(np.min(counts) >= 4 and valid_count >= 10)

    # Chia 70% Train / 30% (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y if can_stratify else None
    )

    # Chia đôi 30% thành 15% Val / 15% Test
    _, temp_counts = np.unique(y_temp, return_counts=True)
    can_stratify_temp = bool(np.min(temp_counts) >= 2 and len(y_temp) >= 4)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp if can_stratify_temp else None
    )

    # Fit StandardScaler CHỈ trên tập Train rồi transform cả 3 tập
    scaler = StandardScaler()
    n_train, seq_len, n_feat = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat).astype(np.float32)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape).astype(np.float32)

    np.save(output_dir / "X_train.npy", X_train_scaled)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val_scaled)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test_scaled)
    np.save(output_dir / "y_test.npy", y_test)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    id_to_label = {str(v): k for k, v in label_to_id.items()}
    label_map_data = {
        "num_classes": len(EMOTION_CLASSES_15),
        "window_size": window_size,
        "num_features": n_feat,
        "feature_names": FEATURE_NAMES_22,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "display_names": EMOTION_DISPLAY_NAMES,
        "class_counts": class_counts,
    }
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)

    return {
        "total_scanned": total_images,
        "valid_samples": valid_count,
        "skipped_images": skipped_images,
        "train_shape": list(X_train_scaled.shape),
        "val_shape": list(X_val_scaled.shape),
        "test_shape": list(X_test_scaled.shape),
        "class_counts": class_counts,
        "missing_classes": missing_classes,
        "ignored_folders": ignored_counts,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    print("=== BẮT ĐẦU TRÍCH XUẤT 22 ĐẶC TRƯNG & TẠO CHUỖI 60 KHUNG HÌNH ===")
    summary = build_dataset_from_images(
        progress_callback=lambda cur, tot, msg: print(f"[{cur}/{tot}] {msg}")
    )
    print("\n=== HOÀN TẤT XUẤT DỮ LIỆU HUẤN LUYỆN ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))