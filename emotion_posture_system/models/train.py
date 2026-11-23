import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.feature_extractor import EMOTION_CLASSES_15, EMOTION_DISPLAY_NAMES
from models.architecture import DualStreamBiGRU

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NPY_DIR = BASE_DIR / "data" / "processed_npy"
DEFAULT_WEIGHTS_DIR = BASE_DIR / "models" / "weights"
STATUS_FILE = DEFAULT_WEIGHTS_DIR / "train_status.json"


def write_train_status(status_data: Dict[str, Any]) -> None:
    DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)


def train_bigru_model(
    npy_dir: Path = DEFAULT_NPY_DIR,
    output_model_path: Path = DEFAULT_WEIGHTS_DIR / "best_bigru_model.pth",
    pretrained_weights_path: Optional[Path] = None,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    npy_dir = Path(npy_dir)
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]
    for fname in required_files:
        if not (npy_dir / fname).exists():
            raise FileNotFoundError(f"Thiếu tệp dữ liệu '{fname}' trong {npy_dir}. Hãy chạy Bước 2 trước.")

    X_train = np.load(npy_dir / "X_train.npy")
    y_train = np.load(npy_dir / "y_train.npy")
    X_val = np.load(npy_dir / "X_val.npy")
    y_val = np.load(npy_dir / "y_val.npy")
    X_test = np.load(npy_dir / "X_test.npy")
    y_test = np.load(npy_dir / "y_test.npy")

    scaler_bytes = (npy_dir / "scaler.pkl").read_bytes() if (npy_dir / "scaler.pkl").exists() else None
    label_map: Dict[str, Any] = {}
    if (npy_dir / "label_map.json").exists():
        with open(npy_dir / "label_map.json", "r", encoding="utf-8") as f:
            label_map = json.load(f)

    num_classes = len(EMOTION_CLASSES_15)
    all_labels = np.concatenate([y_train, y_val, y_test])
    if all_labels.min() < 0 or all_labels.max() >= num_classes:
        bad_ids = sorted({int(v) for v in all_labels if v < 0 or v >= num_classes})
        raise ValueError(
            f"Dữ liệu huấn luyện chứa mã nhãn không hợp lệ {bad_ids} (chỉ chấp nhận 0 đến {num_classes - 1}). "
            "Đây là dữ liệu cũ tạo từ các thư mục không phải nhãn cảm xúc (ví dụ cnn, coco, mpii). "
            "Hãy xóa các tệp .npy và label_map.json trong data/processed_npy rồi trích xuất lại từ ảnh "
            "được chia sẵn vào 15 thư mục nhãn cảm xúc."
        )

    present_ids = sorted({int(v) for v in all_labels})
    if len(present_ids) < 2:
        raise ValueError("Dữ liệu huấn luyện chỉ có 1 nhãn cảm xúc, cần ít nhất 2 nhãn để huấn luyện.")
    classes_present = [EMOTION_CLASSES_15[i] for i in present_ids]
    classes_missing = [name for name in EMOTION_CLASSES_15 if name not in classes_present]

    # Luôn dùng bảng nhãn chuẩn 15 lớp để tệp trọng số lưu đúng ánh xạ id -> nhãn
    label_map["num_classes"] = num_classes
    label_map["label_to_id"] = {name: idx for idx, name in enumerate(EMOTION_CLASSES_15)}
    label_map["id_to_label"] = {str(idx): name for idx, name in enumerate(EMOTION_CLASSES_15)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = DualStreamBiGRU(face_dim=14, posture_dim=8, hidden_size=64, num_layers=2, num_classes=15).to(device)

    if pretrained_weights_path and Path(pretrained_weights_path).exists():
        ckpt = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = []

    write_train_status(
        {
            "state": "running",
            "current_epoch": 0,
            "total_epochs": epochs,
            "progress_pct": 0,
            "message": (
                f"Đang khởi tạo huấn luyện trên thiết bị {device} "
                f"({len(classes_present)}/{num_classes} nhãn có dữ liệu)..."
            ),
        }
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = (train_correct / max(train_total, 1)) * 100.0

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_sum += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = (val_correct / max(val_total, 1)) * 100.0
        scheduler.step(val_loss)

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
        }
        history.append(epoch_info)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(
                {
                    "architecture": "DualStreamBiGRU_GatedFusion_TemporalAttention",
                    "input_dim": 22,
                    "window_size": 60,
                    "num_classes": 15,
                    "model_state_dict": model.state_dict(),
                    "scaler_bytes": scaler_bytes,
                    "label_map": label_map,
                    "display_names": EMOTION_DISPLAY_NAMES,
                    "best_val_loss": round(best_val_loss, 4),
                    "best_val_acc": round(best_val_acc, 2),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                output_model_path,
            )

        status_payload = {
            "state": "running",
            "current_epoch": epoch,
            "total_epochs": epochs,
            "progress_pct": int(round(epoch / epochs * 100)),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "message": f"Epoch {epoch}/{epochs} - Val Acc: {val_acc:.2f}% - Val Loss: {val_loss:.4f}",
        }
        write_train_status(status_payload)
        if progress_callback:
            progress_callback(status_payload)

    best_ckpt = torch.load(output_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = torch.argmax(model(batch_x), dim=1)
            test_correct += (preds == batch_y).sum().item()
            test_total += batch_x.size(0)

    test_acc = round((test_correct / max(test_total, 1)) * 100.0, 2)
    best_ckpt["test_acc"] = test_acc
    torch.save(best_ckpt, output_model_path)

    final_result = {
        "state": "completed",
        "current_epoch": epochs,
        "total_epochs": epochs,
        "progress_pct": 100,
        "device": str(device),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 2),
        "test_acc": test_acc,
        "model_path": str(output_model_path),
        "classes_present": classes_present,
        "classes_missing": classes_missing,
        "message": (
            f"Huấn luyện hoàn tất! Độ chính xác tập Test: {test_acc:.2f}%"
            + (f" (Chưa có dữ liệu cho {len(classes_missing)} nhãn: {', '.join(classes_missing)})" if classes_missing else "")
        ),
        "history": history,
    }
    write_train_status(final_result)
    return final_result


if __name__ == "__main__":
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH DUAL-STREAM BiGRU ===")
    result = train_bigru_model(
        epochs=25,
        progress_callback=lambda info: print(
            f"[Epoch {info['current_epoch']:02d}/{info['total_epochs']:02d}] "
            f"Train Loss: {info['train_loss']:.4f} | Train Acc: {info['train_acc']:.2f}% | "
            f"Val Loss: {info['val_loss']:.4f} | Val Acc: {info['val_acc']:.2f}%"
        ),
    )
    print(f"\n=== {result['message']} ===")
    print(f"Tệp trọng số đã lưu tại: {result['model_path']}")