import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.feature_extractor import EMOTION_CLASSES_15, EMOTION_DISPLAY_NAMES


def _register_vietnamese_font() -> str:
    """Tìm và đăng ký font Unicode tiếng Việt có sẵn trên Windows."""
    font_candidates = [
        ("VNFont", Path("C:/Windows/Fonts/arial.ttf")),
        ("VNFont", Path("C:/Windows/Fonts/times.ttf")),
        ("VNFont", Path("C:/Windows/Fonts/tahoma.ttf")),
        ("VNFont", Path("C:/Windows/Fonts/segoeui.ttf")),
    ]
    for font_name, font_path in font_candidates:
        if font_path.exists():
            try:
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                return font_name
            except Exception:
                continue
    return "Helvetica"


def compute_session_statistics(session_obj: Any, logs: List[Any]) -> Dict[str, Any]:
    """Tổng hợp các thông số EAR, PERCLOS, EPI, RULA và phân bổ 15 trạng thái cảm xúc."""
    total_logs = max(len(logs), 1)
    avg_ear = sum(float(l.ear or 0.0) for l in logs) / total_logs
    avg_perclos = sum(float(l.perclos or 0.0) for l in logs) / total_logs
    avg_epi = sum(float(l.epi_score or 0.0) for l in logs) / total_logs
    avg_rula = sum(float(l.rula_score or 0.0) for l in logs) / total_logs

    green_cnt = sum(1 for l in logs if l.posture_color == "green")
    yellow_cnt = sum(1 for l in logs if l.posture_color == "yellow")
    red_cnt = sum(1 for l in logs if l.posture_color == "red")

    # Tổng hợp phân bổ 15 cảm xúc (ưu tiên lấy trung bình xác suất từ JSON, bổ sung đếm nhãn chủ đạo)
    emotion_prob_sums: Dict[str, float] = {
        EMOTION_DISPLAY_NAMES[k]: 0.0 for k in EMOTION_CLASSES_15
    }
    label_counter = Counter(l.emotion_label for l in logs)

    valid_json_rows = 0
    for l in logs:
        try:
            probs = json.loads(l.emotion_probs_json or "{}")
            if isinstance(probs, dict) and probs:
                valid_json_rows += 1
                for k, v in probs.items():
                    emotion_prob_sums[k] = emotion_prob_sums.get(k, 0.0) + float(v)
        except Exception:
            continue

    emotion_distribution: List[Dict[str, Any]] = []
    if valid_json_rows > 0:
        for label_name, sum_val in emotion_prob_sums.items():
            emotion_distribution.append(
                {
                    "label": label_name,
                    "avg_prob": round(sum_val / valid_json_rows, 2),
                    "seconds": label_counter.get(label_name, 0),
                }
            )
    else:
        for label_name in emotion_prob_sums.keys():
            secs = label_counter.get(label_name, 0)
            emotion_distribution.append(
                {
                    "label": label_name,
                    "avg_prob": round(secs / total_logs * 100.0, 2),
                    "seconds": secs,
                }
            )

    emotion_distribution.sort(key=lambda item: (-item["avg_prob"], -item["seconds"]))

    return {
        "total_records": len(logs),
        "avg_ear": round(avg_ear, 3),
        "avg_perclos": round(avg_perclos, 1),
        "avg_epi": round(avg_epi, 1),
        "avg_rula": round(avg_rula, 2),
        "posture_breakdown": {
            "green_pct": round(green_cnt / total_logs * 100.0, 1),
            "yellow_pct": round(yellow_cnt / total_logs * 100.0, 1),
            "red_pct": round(red_cnt / total_logs * 100.0, 1),
        },
        "emotion_distribution": emotion_distribution,
    }


def export_session_csv(session_obj: Any, user_obj: Any, logs: List[Any]) -> bytes:
    """Xuất báo cáo tổng hợp CSV (UTF-8 BOM hiển thị chuẩn tiếng Việt trên Excel)."""
    stats = compute_session_statistics(session_obj, logs)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["BAO CAO TONG HOP PHIEN GIAM SAT CAM XUC & TU THE"])
    writer.writerow(["Ma phien", f"#{session_obj.id}"])
    writer.writerow(["Nguoi dung", f"{user_obj.full_name} ({user_obj.username})"])
    writer.writerow(["Thoi gian bat dau", session_obj.start_time.strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Thoi luong (giay)", session_obj.duration_sec])
    writer.writerow(["Cam xuc chu dao", session_obj.dominant_emotion])
    writer.writerow(["Muc tap trung trung binh (%)", session_obj.avg_focus_score])
    writer.writerow(["So lan canh bao buon ngu", session_obj.drowsiness_alerts_count])
    writer.writerow(["EAR trung binh", stats["avg_ear"]])
    writer.writerow(["PERCLOS trung binh (%)", stats["avg_perclos"]])
    writer.writerow(["EPI trung binh", stats["avg_epi"]])
    writer.writerow(["RULA trung binh", stats["avg_rula"]])
    writer.writerow(["Nhan xet tu the", session_obj.posture_comment])
    writer.writerow([])

    writer.writerow(["=== BANG PHAN BO 15 TRANG THAI CAM XUC ==="])
    writer.writerow(["Trang thai cam xuc", "Xac suat trung binh (%)", "So giay chu dao"])
    for item in stats["emotion_distribution"]:
        writer.writerow([item["label"], item["avg_prob"], item["seconds"]])
    writer.writerow([])

    writer.writerow(["=== CHI TIET CHUOI THOI GIAN TUNG GIAY ==="])
    writer.writerow(
        [
            "Thoi diem (giay)",
            "Cam xuc du doan",
            "Do tin cay (%)",
            "EAR",
            "Tan suat chop mat",
            "PERCLOS (%)",
            "EPI",
            "RULA",
            "Trang thai tu the",
            "Muc canh bao buon ngu",
        ]
    )
    for row in logs:
        writer.writerow(
            [
                row.timestamp_sec,
                row.emotion_label,
                row.confidence,
                row.ear,
                row.blink_rate,
                row.perclos,
                row.epi_score,
                row.rula_score,
                row.posture_color,
                row.drowsiness_level,
            ]
        )

    return output.getvalue().encode("utf-8-sig")


def export_session_pdf(session_obj: Any, user_obj: Any, logs: List[Any]) -> bytes:
    """Xuất báo cáo tổng hợp PDF định dạng đẹp, hỗ trợ tiếng Việt."""
    stats = compute_session_statistics(session_obj, logs)
    font_name = _register_vietnamese_font()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VNTitle",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "VNSection",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "VNBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
    )

    elements = []
    elements.append(Paragraph("BÁO CÁO TỔNG HỢP PHIÊN GIÁM SÁT CẢM XÚC & TƯ THẾ", title_style))
    elements.append(
        Paragraph(
            f"Mã phiên: #{session_obj.id} | Người dùng: {user_obj.full_name} ({user_obj.username}) | "
            f"Ngày ghi nhận: {session_obj.start_time.strftime('%d/%m/%Y %H:%M:%S')}",
            body_style,
        )
    )
    elements.append(Spacer(1, 10))

    # 1. Bảng tổng hợp chỉ số sinh trắc & công thái học
    elements.append(Paragraph("1. Thông số tổng quan phiên làm việc", section_style))
    overview_data = [
        ["Chỉ tiêu đánh giá", "Giá trị ghi nhận", "Chỉ tiêu đánh giá", "Giá trị ghi nhận"],
        ["Thời lượng phiên", f"{session_obj.duration_sec} giây", "EAR trung bình", f"{stats['avg_ear']}"],
        ["Cảm xúc chủ đạo", session_obj.dominant_emotion, "PERCLOS trung bình", f"{stats['avg_perclos']}%"],
        ["Mức tập trung TB", f"{session_obj.avg_focus_score}%", "Chỉ số EPI trung bình", f"{stats['avg_epi']} / 100"],
        ["Cảnh báo buồn ngủ", f"{session_obj.drowsiness_alerts_count} lần", "Điểm RULA trung bình", f"{stats['avg_rula']} / 7"],
    ]
    overview_table = Table(overview_data, colWidths=[4.0 * cm, 4.6 * cm, 4.2 * cm, 4.4 * cm])
    overview_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(overview_table)
    elements.append(Spacer(1, 8))

    # 2. Đánh giá công thái học tư thế
    pb = stats["posture_breakdown"]
    elements.append(Paragraph("2. Thống kê đánh giá tư thế (RULA / EPI)", section_style))
    elements.append(
        Paragraph(
            f"• Tỷ lệ tư thế An toàn (Xanh): <b>{pb['green_pct']}%</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Hơi lệch (Vàng): <b>{pb['yellow_pct']}%</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Sai nghiêm trọng (Đỏ): <b>{pb['red_pct']}%</b>",
            body_style,
        )
    )
    elements.append(Paragraph(f"• Nhận xét tổng quan: {session_obj.posture_comment}", body_style))
    elements.append(Spacer(1, 8))

    # 3. Bảng phân bổ 15 trạng thái cảm xúc
    elements.append(Paragraph("3. Bảng phân bổ 15 trạng thái cảm xúc", section_style))
    emo_rows = [["STT", "Trạng thái cảm xúc", "Xác suất trung bình (%)", "Thời lượng chủ đạo (giây)"]]
    for idx, item in enumerate(stats["emotion_distribution"][:15], start=1):
        emo_rows.append([str(idx), item["label"], f"{item['avg_prob']}%", str(item["seconds"])])

    emo_table = Table(emo_rows, colWidths=[1.5 * cm, 7.2 * cm, 4.3 * cm, 4.2 * cm])
    emo_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(emo_table)

    doc.build(elements)
    return buffer.getvalue()