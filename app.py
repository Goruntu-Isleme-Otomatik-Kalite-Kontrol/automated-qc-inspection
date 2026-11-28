from datetime import datetime, timezone
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.inspection import detect_defects
from src.nn_detector import ResNetPatchFeatureExtractor, detect_nn_defects, load_memory_bank
from src.preprocessing import preprocess_image

st.set_page_config(
    page_title="Otomatik Kalite Kontrol (v2.0.0)",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #151C2C;
        border: 1px solid #263248;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-size: 0.85rem;
    }
    div[data-testid="stImage"] img {
        border: 1px solid #2D3748;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

OBJECTS = {
    "Şişe (bottle)": "bottle",
    "Metal Somun (metal_nut)": "metal_nut",
    "Fındık (hazelnut)": "hazelnut",
}

METHODS = {
    "Hibrit Yapay Zeka (OpenCV + ResNet-18)": "nn",
    "Klasik Görüntü İşleme (CIE LAB İstatistiksel)": "cv",
}


@st.cache_resource
def get_feature_extractor() -> ResNetPatchFeatureExtractor:
    """ResNet-18 modelini bellekte bir kez yükler ve tekrar kullanır."""
    return ResNetPatchFeatureExtractor()


@st.cache_data
def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_centering_shift(bgr_img: np.ndarray, blur_ksize: int = 9) -> tuple[int, int]:
    """Ön işlemede uygulanan merkezleme kaymasını (dx, dy) piksel cinsinden hesaplar."""
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    corners = [mask[0, 0], mask[0, -1], mask[-1, 0], mask[-1, -1]]
    if np.mean(corners) > 127:
        mask = cv2.bitwise_not(mask)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0, 0

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (w // 2) - cx, (h // 2) - cy


def overlay_heatmap(bgr_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Anomali ısı haritasını orijinal görüntü üzerine renkli olarak bindirir."""
    h, w = bgr_img.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_norm = np.clip(hm_resized / 1.2, 0.0, 1.0)
    hm_uint8 = (hm_norm * 255).astype(np.uint8)
    colored_hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_img, 1.0 - alpha, colored_hm, alpha, 0)


def draw_inspection_boxes(
    bgr_img: np.ndarray, boxes: list[tuple[int, int, int, int]], status: str
) -> np.ndarray:
    """Tespit edilen kusur kutularını ve görüntü boyutuna ölçekli durum etiketini çizer."""
    annotated = bgr_img.copy()
    h, w = annotated.shape[:2]

    if status == "FAIL":
        color = (0, 0, 255)
    elif status == "INVALID":
        color = (0, 165, 255)
    else:
        color = (0, 200, 0)

    box_thick = max(2, int(round(w / 320.0)))
    for x, y, bw, bh in boxes:
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, box_thick)

    label = f"STATUS: {status} ({len(boxes)})"
    font_scale = max(0.45, min(1.05, w / 820.0))
    text_thick = max(1, int(round(font_scale * 2.2)))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)

    pad = max(6, int(round(w * 0.015)))
    cv2.rectangle(annotated, (pad, pad), (pad + tw + 12, pad + th + baseline + 10), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        label,
        (pad + 6, pad + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        text_thick,
        cv2.LINE_AA,
    )
    return annotated


def load_ground_truth_vis(
    bgr_img: np.ndarray, image_path: Path | None, category: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """MVTec test görüntüsü için gerçek kusur maskesini ve yeşil konturlu görselini döndürür."""
    if image_path is None:
        return None, None

    h, w = bgr_img.shape[:2]
    if category == "good":
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        gt_vis = bgr_img.copy()
        return gt_mask, gt_vis

    gt_path = image_path.parent.parent.parent / "ground_truth" / category / f"{image_path.stem}_mask.png"
    if not gt_path.exists():
        return None, None

    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return None, None

    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_bin = (gt_mask > 127).astype(np.uint8) * 255
    gt_vis = bgr_img.copy()
    green_overlay = np.zeros_like(gt_vis)
    green_overlay[:, :] = (0, 220, 0)
    mask_bool = gt_bin > 0
    gt_vis[mask_bool] = cv2.addWeighted(gt_vis, 0.55, green_overlay, 0.45, 0)[mask_bool]

    contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gt_vis, contours, -1, (0, 255, 0), max(2, int(round(w / 350.0))))
    return gt_bin, gt_vis


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Tahmin maskesi ile gerçek maske arasındaki piksel IoU (%) ve Dice (%) değerlerini hesaplar."""
    pred_b = pred_mask > 0
    gt_b = gt_mask > 0

    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    denom_dice = pred_b.sum() + gt_b.sum()

    if union == 0:
        return 100.0, 100.0

    iou = (intersection / union) * 100.0
    dice = (2.0 * intersection / denom_dice) * 100.0 if denom_dice > 0 else 0.0
    return float(iou), float(dice)


def run_single_image_inspection(
    bgr_img: np.ndarray,
    cfg: dict,
    threshold_override: float,
    heatmap_alpha: float = 0.45,
) -> dict:
    """Tek bir görüntü üzerinde seçilen yönteme göre denetim yapar."""
    start_time = time.perf_counter()
    orig_h, orig_w = bgr_img.shape[:2]
    model_type = cfg.get("model", {}).get("type", "statistical_cv")
    prep_cfg = cfg["preprocessing"]
    insp_cfg = cfg["inspection"]
    cache_path = Path(cfg["paths"]["reference_cache"])

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Referans önbellek dosyası bulunamadı: {cache_path}. "
            "Önce 'python -m src.evaluate --all' komutunu çalıştırın."
        )

    blur_k = prep_cfg.get("blur_kernel_size", 9)
    dx, dy = compute_centering_shift(bgr_img, blur_k)
    inv_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])

    if model_type == "nn_resnet":
        target_size = tuple(prep_cfg.get("target_size", [224, 224]))
        proc_img = preprocess_image(
            bgr_img,
            kernel_size=blur_k,
            color_space="rgb",
            bg_mode=prep_cfg.get("bg_mode", "auto"),
            use_clahe=prep_cfg.get("use_clahe", False),
            target_size=target_size,
        )
        memory_bank = load_memory_bank(str(cache_path))
        extractor = get_feature_extractor()
        res = detect_nn_defects(
            proc_img,
            memory_bank=memory_bank,
            extractor=extractor,
            anomaly_threshold=threshold_override,
            min_defect_area_px=insp_cfg.get("min_defect_area_px", 120),
            morph_kernel_size=insp_cfg.get("morph_kernel_size", 5),
        )

        scale_x = orig_w / target_size[0]
        scale_y = orig_h / target_size[1]
        scaled_boxes = [
            (
                max(0, int(x * scale_x) - dx),
                max(0, int(y * scale_y) - dy),
                int(w * scale_x),
                int(h * scale_y),
            )
            for (x, y, w, h) in res["boxes"]
        ]

        base_thresh = float(insp_cfg.get("anomaly_threshold", 0.50))
        heatmap = res["heatmap"]
        mean_score = float(np.mean(heatmap))
        p90_score = float(np.percentile(heatmap, 90))
        max_score = float(res["max_score"])
        ood_area_ratio = float(np.mean(heatmap > (base_thresh * 0.85)))

        gray_proc = cv2.cvtColor(proc_img, cv2.COLOR_RGB2GRAY)
        _, fg_mask = cv2.threshold(gray_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners = [fg_mask[0, 0], fg_mask[0, -1], fg_mask[-1, 0], fg_mask[-1, -1]]
        if np.mean(corners) > 127:
            fg_mask = cv2.bitwise_not(fg_mask)
        fg_ratio = float(np.mean(fg_mask > 0))

        is_ood = (
            ood_area_ratio > 0.45
            or mean_score > 0.52
            or p90_score > 0.78
            or max_score > 0.98
            or fg_ratio < 0.10
            or fg_ratio > 0.80
        )

        if is_ood:
            status = "INVALID"
            scaled_boxes = []
            aligned_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            status = "FAIL" if len(scaled_boxes) > 0 else "PASS"
            mask_orig = cv2.resize(res["binary_mask"], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            aligned_mask = cv2.warpAffine(mask_orig, inv_matrix, (orig_w, orig_h))

        heatmap_orig = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        heatmap_aligned = cv2.warpAffine(heatmap_orig, inv_matrix, (orig_w, orig_h))
        heatmap_vis = overlay_heatmap(bgr_img, heatmap_aligned, alpha=heatmap_alpha)
        result_vis = draw_inspection_boxes(bgr_img, scaled_boxes, status)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": status,
            "boxes": scaled_boxes,
            "pred_mask": aligned_mask,
            "max_score": max_score,
            "defect_ratio": ood_area_ratio,
            "heatmap_vis": heatmap_vis,
            "result_vis": result_vis,
            "latency_ms": latency_ms,
        }

    # Classical Statistical CV Track
    with np.load(cache_path) as ref_data:
        mean_img = ref_data["mean_img"]
        std_img = ref_data["std_img"]

    ref_h, ref_w = mean_img.shape[:2]
    proc_img = preprocess_image(
        bgr_img,
        kernel_size=blur_k,
        bg_threshold=prep_cfg.get("bg_threshold", 210),
        color_space=prep_cfg.get("color_space", "lab"),
        bg_mode=prep_cfg.get("bg_mode", "auto"),
        target_size=(ref_w, ref_h),
    )

    diff_map = np.mean(np.abs(proc_img.astype(np.float32) - mean_img.astype(np.float32)), axis=-1)
    mean_diff = float(np.mean(diff_map))
    ood_area_ratio = float(np.mean(diff_map > 35.0))
    aspect_diff = abs((orig_w / orig_h) - (ref_w / ref_h))

    if ood_area_ratio > 0.35 or mean_diff > 28.0 or aspect_diff > 0.25:
        status = "INVALID"
        aligned_boxes = []
    else:
        is_defective, raw_boxes = detect_defects(
            proc_img,
            mean_img=mean_img,
            std_img=std_img,
            std_multiplier=threshold_override,
            min_abs_diff=insp_cfg.get("min_abs_diff", 22),
            min_defect_area_px=insp_cfg.get("min_defect_area_px", 160),
        )
        status = "FAIL" if is_defective else "PASS"

        scale_x = orig_w / ref_w
        scale_y = orig_h / ref_h
        aligned_boxes = [
            (
                max(0, int(x * scale_x) - dx),
                max(0, int(y * scale_y) - dy),
                int(w * scale_x),
                int(h * scale_y),
            )
            for (x, y, w, h) in raw_boxes
        ]

    aligned_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for x, y, bw, bh in aligned_boxes:
        aligned_mask[y : y + bh, x : x + bw] = 255

    diff_norm = np.clip(diff_map / 60.0, 0.0, 1.0)
    diff_orig = cv2.resize(diff_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    diff_aligned = cv2.warpAffine(diff_orig, inv_matrix, (orig_w, orig_h))

    heatmap_vis = overlay_heatmap(bgr_img, diff_aligned, alpha=heatmap_alpha)
    result_vis = draw_inspection_boxes(bgr_img, aligned_boxes, status)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "status": status,
        "boxes": aligned_boxes,
        "pred_mask": aligned_mask,
        "max_score": float(np.max(diff_map)),
        "defect_ratio": ood_area_ratio,
        "heatmap_vis": heatmap_vis,
        "result_vis": result_vis,
        "latency_ms": latency_ms,
    }


def render_inspection_block(
    title: str,
    method_tag: str,
    bgr_img: np.ndarray,
    output: dict,
    obj_key: str,
    image_label: str,
    gt_mask: np.ndarray | None,
    gt_vis: np.ndarray | None,
    selected_cat: str | None,
) -> None:
    """Tek bir denetim yönteminin durum şeridini, metriklerini, görsellerini ve kanıt indirme butonlarını çizer."""
    st.markdown(f"### {title}")

    status = output["status"]
    if status == "INVALID":
        st.warning(
            f"⚠️ **GEÇERSİZ GÖRÜNTÜ (INVALID):** Yüklenen fotoğraf `{obj_key}` kategorisine ait değil "
            f"(Yabancı alan oranı: %{output.get('defect_ratio', 0.0) * 100:.0f}). Lütfen uygun bir ürün fotoğrafı seçin."
        )
    elif status == "FAIL":
        st.error(
            f"❌ **HATALI ÜRÜN TESPİT EDİLDİ (FAIL):** Ürün yüzeyinde **{len(output['boxes'])} adet** kusurlu bölge bulundu."
        )
    else:
        st.success("✅ **KALİTE KONTROLDEN GEÇTİ (PASS):** Ürün yüzeyinde kusur tespit edilmedi.")

    has_gt = gt_mask is not None and gt_vis is not None and status != "INVALID"
    cols_metric = st.columns(5 if has_gt else 4)
    cols_metric[0].metric("Karar (Status)", status)
    cols_metric[1].metric("Tespit Edilen Kusur", f"{len(output['boxes'])} adet")
    cols_metric[2].metric("En Yüksek Skor", f"{output['max_score']:.3f}")
    cols_metric[3].metric("İşlem Süresi", f"{output['latency_ms']:.1f} ms")

    iou_val, dice_val = None, None
    if has_gt:
        iou_val, dice_val = compute_mask_metrics(output["pred_mask"], gt_mask)
        if selected_cat == "good":
            cols_metric[4].metric("Piksel Doğruluğu", "Sağlam Örnek" if status == "PASS" else "Yanlış Alarm")
        else:
            cols_metric[4].metric("Anlık IoU / Dice", f"%{iou_val:.1f} / %{dice_val:.1f}")

    img_cols = st.columns(4 if has_gt else 3)
    with img_cols[0]:
        st.image(
            cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB),
            caption="1. Orijinal Giriş Görüntüsü",
            use_container_width=True,
        )
    with img_cols[1]:
        st.image(
            cv2.cvtColor(output["heatmap_vis"], cv2.COLOR_BGR2RGB),
            caption="2. Anomali Isı Haritası (Heatmap)",
            use_container_width=True,
        )
        st.caption("🔵 Normal Doku  |  🟡 Sınır Bölge  |  🔴 Yüksek Anomali")
    with img_cols[2]:
        st.image(
            cv2.cvtColor(output["result_vis"], cv2.COLOR_BGR2RGB),
            caption=f"3. Nihai Karar & Kusur Konumu ({status})",
            use_container_width=True,
        )
    if has_gt:
        with img_cols[3]:
            gt_caption = (
                "4. Gerçek Maske (Sağlam Ürün - Kusur Yok)"
                if selected_cat == "good"
                else "4. Gerçek Kusur Maskesi (Ground Truth)"
            )
            st.image(
                cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB),
                caption=gt_caption,
                use_container_width=True,
            )

    # İzlenebilirlik & Kanıt Paketi İndirme
    ok, png_buf = cv2.imencode(".png", output["result_vis"])
    report_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "product_category": obj_key,
        "sample_label": image_label,
        "method": method_tag,
        "status": status,
        "defect_count": len(output["boxes"]),
        "bounding_boxes_xywh": [list(b) for b in output["boxes"]],
        "max_anomaly_score": round(output["max_score"], 4),
        "latency_ms": round(output["latency_ms"], 2),
        "pixel_iou_pct": round(iou_val, 2) if (iou_val is not None and selected_cat != "good") else None,
        "pixel_dice_pct": round(dice_val, 2) if (dice_val is not None and selected_cat != "good") else None,
    }
    json_bytes = json.dumps(report_payload, indent=2, ensure_ascii=False).encode("utf-8")

    dl_col1, dl_col2, _ = st.columns([1.2, 1.2, 3.6])
    if ok:
        dl_col1.download_button(
            label="📥 İşaretli Görseli İndir (.png)",
            data=png_buf.tobytes(),
            file_name=f"qc_{obj_key}_{method_tag}_{status.lower()}.png",
            mime="image/png",
            key=f"dl_img_{method_tag}_{image_label}",
        )
    dl_col2.download_button(
        label="📄 Denetim Kaydını İndir (.json)",
        data=json_bytes,
        file_name=f"qc_{obj_key}_{method_tag}_{status.lower()}.json",
        mime="application/json",
        key=f"dl_json_{method_tag}_{image_label}",
    )


# ==================== UI LAYOUT ====================
st.title("🔍 Endüstriyel Otomatik Kalite Kontrol Sistemi (v2.0.0)")
st.caption(
    "Klasik İstatistiksel Görüntü İşleme (CIE LAB) ile Hibrit Derin Öğrenme (OpenCV + ResNet-18 Patch Memory Bank) Karşılaştırmalı Denetim Platformu"
)

tab_inspect, tab_benchmark = st.tabs(
    ["🧪 Canlı Denetim & Görsel Teşhis", "📊 6 Yönlü Benchmark Raporu"]
)

with tab_inspect:
    st.sidebar.header("⚙️ Denetim Ayarları")
    selected_obj_label = st.sidebar.selectbox("1. Ürün Kategorisi", list(OBJECTS.keys()))
    obj_key = OBJECTS[selected_obj_label]

    compare_mode = st.sidebar.checkbox("⚖️ İki Yöntemi Yan Yana Karşılaştır", value=False)

    cfg_nn = load_yaml_config(f"configs/{obj_key}_nn.yaml")
    cfg_cv = load_yaml_config(f"configs/{obj_key}_cv.yaml")

    if compare_mode:
        nn_thresh = st.sidebar.slider(
            "Hibrit AI Eşiği (anomaly_threshold)",
            min_value=0.30,
            max_value=1.00,
            value=float(cfg_nn["inspection"].get("anomaly_threshold", 0.50)),
            step=0.01,
        )
        cv_thresh = st.sidebar.slider(
            "Klasik CV Eşiği (std_multiplier)",
            min_value=1.5,
            max_value=6.0,
            value=float(cfg_cv["inspection"].get("std_multiplier", 3.1)),
            step=0.1,
        )
    else:
        selected_method_label = st.sidebar.selectbox("2. Denetim Yöntemi", list(METHODS.keys()))
        method_key = METHODS[selected_method_label]
        if method_key == "nn":
            threshold_val = st.sidebar.slider(
                "Anomali Eşik Değeri (anomaly_threshold)",
                min_value=0.30,
                max_value=1.00,
                value=float(cfg_nn["inspection"].get("anomaly_threshold", 0.50)),
                step=0.01,
            )
        else:
            threshold_val = st.sidebar.slider(
                "Standart Sapma Çarpanı (std_multiplier)",
                min_value=1.5,
                max_value=6.0,
                value=float(cfg_cv["inspection"].get("std_multiplier", 3.1)),
                step=0.1,
            )

    heatmap_alpha = st.sidebar.slider(
        "Isı Haritası Saydamlığı (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    st.sidebar.divider()
    source_mode = st.sidebar.radio(
        "3. Görüntü Kaynağı", ["MVTec Test Veri Setinden Seç", "Bilgisayardan Fotoğraf Yükle"]
    )

    bgr_image = None
    image_label = ""
    selected_file_path: Path | None = None
    selected_cat_name: str | None = None

    if source_mode == "MVTec Test Veri Setinden Seç":
        test_root = Path(cfg_nn["paths"]["test_root_dir"])
        categories = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
        selected_cat_name = st.sidebar.selectbox("Test Alt Klasörü (Kusur Tipi)", categories)
        cat_dir = test_root / selected_cat_name
        image_files = sorted(list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")))
        selected_file_path = st.sidebar.selectbox(
            "Test Fotoğrafı", image_files, format_func=lambda p: p.name
        )
        if selected_file_path:
            bgr_image = cv2.imread(str(selected_file_path))
            image_label = f"{obj_key} / {selected_cat_name} / {selected_file_path.name}"
    else:
        uploaded_file = st.sidebar.file_uploader(
            "PNG veya JPG dosyası yükleyin", type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_label = uploaded_file.name

    if bgr_image is not None:
        st.subheader(f"İncelenen Örnek: `{image_label}`")
        gt_mask, gt_vis = load_ground_truth_vis(bgr_image, selected_file_path, selected_cat_name or "")

        if compare_mode:
            out_nn = run_single_image_inspection(bgr_image, cfg_nn, nn_thresh, heatmap_alpha)
            render_inspection_block(
                "🧠 Hibrit Yapay Zeka (OpenCV + ResNet-18 Patch Memory Bank)",
                "nn_resnet",
                bgr_image,
                out_nn,
                obj_key,
                image_label,
                gt_mask,
                gt_vis,
                selected_cat_name,
            )
            st.divider()
            out_cv = run_single_image_inspection(bgr_image, cfg_cv, cv_thresh, heatmap_alpha)
            render_inspection_block(
                "📐 Klasik Görüntü İşleme (CIE LAB İstatistiksel Referans)",
                "statistical_cv",
                bgr_image,
                out_cv,
                obj_key,
                image_label,
                gt_mask,
                gt_vis,
                selected_cat_name,
            )
        else:
            active_cfg = cfg_nn if method_key == "nn" else cfg_cv
            active_tag = "nn_resnet" if method_key == "nn" else "statistical_cv"
            active_title = (
                "🧠 Hibrit Yapay Zeka (OpenCV + ResNet-18)"
                if method_key == "nn"
                else "📐 Klasik Görüntü İşleme (CIE LAB İstatistiksel)"
            )
            output = run_single_image_inspection(bgr_image, active_cfg, threshold_val, heatmap_alpha)
            render_inspection_block(
                active_title,
                active_tag,
                bgr_image,
                output,
                obj_key,
                image_label,
                gt_mask,
                gt_vis,
                selected_cat_name,
            )

with tab_benchmark:
    st.subheader("Klasik CV vs. Hibrit ResNet-18 Karşılaştırma Sonuçları")
    summary_path = Path("results/benchmark_summary.csv")

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        nn_df = df[df["config"].str.endswith("_nn")]
        cv_df = df[df["config"].str.endswith("_cv")]

        avg_nn_f1 = float(nn_df["f1_pct"].mean()) if not nn_df.empty else 0.0
        avg_cv_f1 = float(cv_df["f1_pct"].mean()) if not cv_df.empty else 0.0
        f1_gain = avg_nn_f1 - avg_cv_f1
        avg_nn_fps = float(nn_df["fps"].mean()) if not nn_df.empty else 0.0
        avg_nn_iou = float(nn_df["mean_iou_pct"].mean()) if not nn_df.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ortalama Hibrit F1", f"%{avg_nn_f1:.2f}", delta=f"+{f1_gain:.2f} puan (Klasik CV'ye göre)")
        k2.metric("Ortalama Klasik CV F1", f"%{avg_cv_f1:.2f}")
        k3.metric("Ortalama Piksel IoU (Hibrit)", f"%{avg_nn_iou:.2f}")
        k4.metric("Ortalama Hibrit Hız", f"{avg_nn_fps:.1f} FPS")

        st.markdown("---")
        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**F1-Score (%) ve Piksel IoU (%) Karşılaştırması**")
            chart_df = df.set_index("config")[["f1_pct", "mean_iou_pct"]]
            chart_df.columns = ["F1-Score (%)", "Mean IoU (%)"]
            st.bar_chart(chart_df, stack=False)
        with c2:
            st.markdown("**İşlem Hızı Karşılaştırması (FPS)**")
            fps_df = df.set_index("config")[["fps"]]
            fps_df.columns = ["Throughput (FPS)"]
            st.bar_chart(fps_df)
    else:
        st.warning(
            "Henüz `results/benchmark_summary.csv` dosyası oluşturulmamış. "
            "Terminalde `python -m src.evaluate --all` komutunu çalıştırın."
        )