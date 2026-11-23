from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Cơ chế Gated Fusion kết hợp động đặc trưng từ luồng Khuôn mặt và luồng Tư thế."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_face: torch.Tensor, h_posture: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_face, h_posture], dim=-1)
        z = self.gate_layer(combined)
        fused = z * h_face + (1.0 - z) * h_posture
        return self.norm(fused)


class TemporalAttention(nn.Module):
    """Cơ chế Temporal Attention trọng số hóa các khung hình quan trọng trong cửa sổ 60 frames."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn_score(seq_out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(seq_out * weights, dim=1)
        return context, weights.squeeze(-1)


class DualStreamBiGRU(nn.Module):
    """
    Mạng BiGRU hai luồng kết hợp Gated Fusion và Temporal Attention:
    - Đầu vào: (batch_size, 60, 22)
    - Đầu ra: Logits 15 trạng thái cảm xúc (batch_size, 15)
    """

    def __init__(
        self,
        face_dim: int = 14,
        posture_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 15,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.face_dim = face_dim
        self.posture_dim = posture_dim
        self.input_dim = face_dim + posture_dim
        self.num_classes = num_classes
        self.fused_dim = hidden_size * 2

        self.face_proj = nn.Sequential(
            nn.Linear(face_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.face_bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.posture_proj = nn.Sequential(
            nn.Linear(posture_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.posture_bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.gated_fusion = GatedFusion(self.fused_dim)
        self.temporal_attention = TemporalAttention(self.fused_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Kích thước đầu vào không hợp lệ: nhận {tuple(x.shape)}, yêu cầu (batch, 60, {self.input_dim})."
            )

        x_face = x[:, :, : self.face_dim]
        x_posture = x[:, :, self.face_dim :]

        h_face, _ = self.face_bigru(self.face_proj(x_face))
        h_posture, _ = self.posture_bigru(self.posture_proj(x_posture))

        fused_seq = self.gated_fusion(h_face, h_posture)
        context_vec, _ = self.temporal_attention(fused_seq)
        logits = self.classifier(context_vec)
        return logits


def validate_and_load_checkpoint(
    model_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DualStreamBiGRU, Dict[str, Any]]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp trọng số: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Tệp trọng số không đúng định dạng checkpoint của hệ thống.")

    input_dim = int(checkpoint.get("input_dim", -1))
    num_classes = int(checkpoint.get("num_classes", -1))
    if input_dim != 22 or num_classes != 15:
        raise ValueError(
            f"Cấu trúc mô hình không tương thích (nhận input={input_dim}, output={num_classes}; yêu cầu input=22, output=15)."
        )

    state_dict = checkpoint["model_state_dict"]
    out_weight_key = "classifier.4.weight"
    if out_weight_key not in state_dict or state_dict[out_weight_key].shape[0] != 15:
        raise ValueError("Lớp đầu ra của trọng số không khớp 15 trạng thái cảm xúc.")

    model = DualStreamBiGRU(face_dim=14, posture_dim=8, hidden_size=64, num_layers=2, num_classes=15)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint