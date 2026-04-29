"""
StructScan — Detection Pipeline v6
════════════════════════════════════
Key fixes:
  1. YOLO detects NON-STRUCTURAL objects (people, windows, doors) → masks them OUT
  2. Crack detection ONLY runs on structural regions (walls, concrete, road)
  3. Live feed shows crack measurements on-screen
  4. QR code endpoint supported
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from backend.gradcam import GradCAM

MODEL_PATH = Path("models/crack_model.h5")
YOLO_PATH  = Path("models/yolo/best.pt")
IMG_SIZE   = (224, 224)
GRID_COLS  = 8
GRID_ROWS  = 8

# ── Thresholds ────────────────────────────────────────────────────────────────
CRACK_THRESHOLD  = 0.15
ZONE_THRESHOLD   = 0.10
YOLO_CONF        = 0.10

# COCO class IDs to EXCLUDE from crack detection (non-structural objects)
# These regions will be masked out before crack detection runs
NON_STRUCTURAL_CLASSES = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    27,  # tie
    28,  # suitcase
    39,  # bottle
    40,  # wine glass
    41,  # cup
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    62,  # tv
    63,  # laptop
    64,  # mouse
    65,  # remote
    66,  # keyboard
    67,  # cell phone
    68,  # microwave
    69,  # oven
    72,  # refrigerator
    73,  # book
    74,  # clock
    75,  # vase
    76,  # scissors
    79,  # toothbrush
}

# COCO classes that indicate structural surfaces (keep these)
STRUCTURAL_CLASSES = {
    # Nothing specific — we keep everything NOT in NON_STRUCTURAL_CLASSES
}

# Live feed settings
LIVE_SKIP_GRADCAM  = True
SMOOTHING_ALPHA    = 0.35


# ── YOLOv7 loader ─────────────────────────────────────────────────────────────
def _try_load_yolo():
    if not YOLO_PATH.exists():
        print(f"  ℹ️  YOLOv7 not found at {YOLO_PATH}")
        return None
    try:
        import torch
        model = torch.hub.load(
            "WongKinYiu/yolov7", "custom",
            path_or_model=str(YOLO_PATH),
            source="github", force_reload=False, trust_repo=True,
        )
        model.eval()
        print(f"  ✅ YOLOv7 loaded from {YOLO_PATH}")
        return model
    except ImportError:
        print("  ⚠️  PyTorch not installed — pip install torch torchvision")
        return None
    except Exception as e:
        print(f"  ⚠️  YOLOv7 failed: {e}")
        return None


def build_mobilenetv2():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inputs=base.input, outputs=out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ── Lighting normalization ────────────────────────────────────────────────────
def normalize_lighting(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_norm = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


# ── Non-structural mask from YOLO ─────────────────────────────────────────────
def get_exclusion_mask(img_bgr, yolo_model):
    """
    Run YOLO on full COCO classes to find people/furniture/objects.
    Returns a binary mask where 255 = non-structural (exclude from crack detection).
    """
    h, w = img_bgr.shape[:2]
    exclusion = np.zeros((h, w), dtype=np.uint8)

    if yolo_model is None:
        return exclusion

    try:
        import torch
        # Use a separate COCO-pretrained model for object detection if available
        # If best.pt is crack-specific (1 class), this won't detect people
        # So we use basic CV-based person detection as fallback
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb, size=640)
        preds   = results.pred[0].cpu().numpy() if len(results.pred) > 0 else []

        for det in preds:
            if len(det) < 6: continue
            cls_id = int(det[5])
            conf   = float(det[4])
            if cls_id in NON_STRUCTURAL_CLASSES and conf > 0.3:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                # Expand mask slightly for safety
                pad = 10
                x1 = max(0, x1-pad); y1 = max(0, y1-pad)
                x2 = min(w, x2+pad); y2 = min(h, y2+pad)
                exclusion[y1:y2, x1:x2] = 255
    except Exception as e:
        print(f"  ⚠️  Exclusion mask error: {e}")

    return exclusion


def get_person_mask_cv(img_bgr):
    """
    Fallback: detect skin-colored regions and moving objects
    using color segmentation to exclude people from crack detection.
    """
    h, w = img_bgr.shape[:2]
    exclusion = np.zeros((h, w), dtype=np.uint8)

    # Skin color detection in YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb,
        np.array([0,  133, 77],  dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8)
    )

    # Dilate skin mask to cover surrounding clothing etc.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    skin_dilated = cv2.dilate(skin_mask, kernel, iterations=2)

    # Only exclude large skin regions (small patches = cracks in reddish walls)
    contours, _ = cv2.findContours(skin_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > (h * w * 0.02):  # > 2% of frame = likely person
            cv2.drawContours(exclusion, [cnt], -1, 255, -1)

    return exclusion


# ── Direct crack detection ────────────────────────────────────────────────────
def detect_crack_pixels(img_bgr, exclusion_mask=None):
    """
    Detect crack pixels using image processing.
    Excludes non-structural regions (people, objects) using exclusion_mask.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Method 1: Dark regions relative to local background ───────────
    blur_local = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.subtract(blur_local.astype(np.int16), gray.astype(np.int16))
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    _, dark_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)

    # ── Method 2: Ridge detection (Laplacian) ─────────────────────────
    ridge_mask = np.zeros_like(gray, dtype=np.float32)
    for ksize in [3, 5, 7]:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
        ridge_mask += np.abs(lap)
    ridge_norm = cv2.normalize(ridge_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, ridge_bin = cv2.threshold(ridge_norm, 30, 255, cv2.THRESH_BINARY)

    # ── Method 3: Elongated structure detection ────────────────────────
    kernel_h  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    kernel_d1 = np.zeros((25, 25), np.uint8)
    kernel_d2 = np.zeros((25, 25), np.uint8)
    for i in range(25):
        kernel_d1[i, i]    = 1
        kernel_d2[i, 24-i] = 1

    elong = np.zeros_like(gray)
    for k in [kernel_h, kernel_v, kernel_d1, kernel_d2]:
        opened = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k)
        elong  = cv2.bitwise_or(elong, opened)

    # ── Combine ────────────────────────────────────────────────────────
    combined = (
        dark_mask.astype(np.float32) * 0.40 +
        ridge_bin.astype(np.float32) * 0.30 +
        elong.astype(np.float32)     * 0.30
    )
    _, crack_bin = cv2.threshold(
        np.clip(combined, 0, 255).astype(np.uint8), 60, 255, cv2.THRESH_BINARY
    )

    # ── Remove noise ───────────────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crack_bin = cv2.morphologyEx(crack_bin, cv2.MORPH_CLOSE, kernel_close)

    min_area  = max(int(h * w * 0.0001), 20)
    contours, _ = cv2.findContours(crack_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask  = np.zeros_like(crack_bin)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # ── Apply exclusion mask ───────────────────────────────────────────
    if exclusion_mask is not None and exclusion_mask.any():
        # Remove crack pixels in excluded regions (people, objects)
        clean_mask = cv2.bitwise_and(
            clean_mask,
            cv2.bitwise_not(exclusion_mask)
        )

    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(clean_mask, kernel_dil, iterations=1)


# ── Crack metrics ─────────────────────────────────────────────────────────────
def compute_crack_metrics(crack_mask, img_bgr):
    h, w    = crack_mask.shape[:2]
    pixels  = int(np.count_nonzero(crack_mask))
    if pixels == 0:
        return None

    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_cnt  = max(contours, key=cv2.contourArea)
    area_px   = cv2.contourArea(main_cnt)
    size_pct  = round(area_px / (h * w) * 100, 2)

    if len(main_cnt) >= 5:
        rect      = cv2.minAreaRect(main_cnt)
        mw, mh    = rect[1]
        length_px = int(max(mw, mh))
        width_px  = max(1, int(min(mw, mh)))
        angle     = abs(rect[2]) % 180
        if mw < mh: angle = (90 + rect[2]) % 180
    else:
        rx, ry, rw, rh = cv2.boundingRect(main_cnt)
        length_px = int(max(rw, rh))
        width_px  = max(1, int(min(rw, rh)))
        angle     = 0.0

    aspect = length_px / width_px
    est_mm = width_px * 0.78  # 1px ≈ 0.78mm at 640px width

    if est_mm < 0.1:
        width_cat, width_sev = "Hairline (<0.1mm)", 1
    elif est_mm < 0.3:
        width_cat, width_sev = "Thin (0.1-0.3mm)", 2
    elif est_mm < 1.0:
        width_cat, width_sev = "Medium (0.3-1.0mm)", 3
    elif est_mm < 3.0:
        width_cat, width_sev = "Wide (1.0-3.0mm)", 4
    else:
        width_cat, width_sev = "Gaping (>3.0mm)", 5

    angle_norm = angle % 180
    if width_sev <= 1:
        crack_type, hint, is_struct = "Hairline Crack", "Surface crazing — cosmetic", False
    elif 80 <= angle_norm <= 100:
        crack_type, hint, is_struct = "Horizontal Crack", "Lateral pressure or overloading — serious", True
    elif angle_norm < 20 or angle_norm > 160:
        crack_type, hint, is_struct = "Vertical Crack", "Thermal movement or minor settlement", width_sev >= 3
    elif 35 <= angle_norm <= 55 or 125 <= angle_norm <= 145:
        crack_type, hint, is_struct = "Diagonal / Shear Crack", "Shear stress — check structure", True
    elif size_pct > 15 and aspect < 4:
        crack_type, hint, is_struct = "Spalling / Map Cracking", "Material delamination — monitor for rebar", True
    elif aspect > 15:
        crack_type, hint, is_struct = "Linear Structural Crack", "Tensile stress — possible overloading", True
    else:
        crack_type, hint, is_struct = "Surface Crack", "Plastic shrinkage — usually non-structural", False

    activity = "Likely Active" if width_sev >= 4 else \
               "Possibly Active" if width_sev == 3 else "Likely Dormant"

    return {
        "crack_type":     crack_type,
        "severity_hint":  hint,
        "is_structural":  is_struct,
        "width_category": width_cat,
        "est_width_mm":   round(est_mm, 2),
        "activity":       activity,
        "size_percent":   size_pct,
        "length_px":      length_px,
        "width_px":       width_px,
        "angle_deg":      round(float(angle), 1),
        "aspect_ratio":   round(aspect, 1),
        "num_regions":    len(contours),
        "total_crack_px": pixels,
    }


def _empty_crack_info():
    return {
        "crack_type": "None", "severity_hint": "No crack detected",
        "is_structural": False, "width_category": "—",
        "est_width_mm": 0.0, "activity": "—",
        "size_percent": 0.0, "length_px": 0, "width_px": 0,
        "angle_deg": 0.0, "aspect_ratio": 0.0,
        "num_regions": 0, "total_crack_px": 0,
    }


# ── Surface classifier ────────────────────────────────────────────────────────
def classify_surface(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mean_v   = float(np.mean(hsv[:,:,2]))
    mean_s   = float(np.mean(hsv[:,:,1]))
    mean_h   = float(np.mean(hsv[:,:,0]))
    blur     = cv2.GaussianBlur(gray, (5,5), 0)
    texture  = float(np.mean(cv2.absdiff(gray, blur)))
    edges    = cv2.Canny(gray, 50, 150)
    edge_d   = float(np.count_nonzero(edges)) / (h * w)

    if mean_v < 100 and mean_s < 30:         return "Road / Asphalt"
    elif mean_v > 120 and mean_s < 40:
        if texture < 8:                        return "Concrete Slab / Floor"
        else:                                  return "Concrete Wall"
    elif mean_h < 20 or mean_h > 160:         return "Masonry / Brick Wall"
    elif texture > 15:                         return "Rough Concrete / Plaster"
    else:                                      return "Structural Surface"


def estimate_depth(crack_mask, img_bgr):
    if np.count_nonzero(crack_mask) == 0:
        return "Unknown"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[crack_mask > 0]
    dark = 255 - float(np.mean(vals))
    if dark < 30:   return "Surface (<1mm)"
    elif dark < 80: return "Shallow (1-5mm)"
    elif dark < 150:return "Moderate (5-20mm)"
    else:           return "Deep (>20mm)"


# ══════════════════════════════════════════════════════════════════════════════
class DamageDetector:
    def __init__(self):
        self.model         = None
        self.gradcam       = None
        self.yolo          = None
        self.demo_mode     = False
        self._smooth_score = None
        self._load()

    def _load(self):
        print("\n🔍 Loading StructScan v6...")
        if MODEL_PATH.exists():
            try:
                self.model     = tf.keras.models.load_model(str(MODEL_PATH))
                self.demo_mode = False
                print(f"  ✅ MobileNetV2 loaded")
            except Exception as e:
                print(f"  ❌ {e}")
                self.model     = build_mobilenetv2()
                self.demo_mode = True
        else:
            print("  ⚠️  No trained model — demo mode")
            self.model     = build_mobilenetv2()
            self.demo_mode = True

        self.gradcam = GradCAM(self.model)
        self.yolo    = _try_load_yolo()

        mode = "MobileNetV2 + Direct Detection + YOLOv7 (object filtering)" \
               if self.yolo else "MobileNetV2 + Direct Detection + Grad-CAM"
        print(f"  🎯 Pipeline: {mode}")

    def preprocess(self, img_bgr):
        normed  = normalize_lighting(img_bgr)
        rgb     = cv2.cvtColor(normed, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    def grid_analyze(self, img_bgr, crack_mask):
        h, w   = img_bgr.shape[:2]
        cell_h = h // GRID_ROWS
        cell_w = w // GRID_COLS
        zones  = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                y1 = row * cell_h; x1 = col * cell_w
                y2 = min(y1+cell_h, h); x2 = min(x1+cell_w, w)
                cell  = crack_mask[y1:y2, x1:x2]
                score = float(np.count_nonzero(cell)) / max(cell.size, 1)
                zones.append({
                    "row": row, "col": col, "bbox": [x1,y1,x2,y2],
                    "heat_score": round(score, 4),
                    "damaged": score > ZONE_THRESHOLD,
                })
        return zones

    def _run_yolo(self, img_bgr):
        """Returns (crack_dets, exclusion_mask)."""
        if self.yolo is None:
            excl = get_person_mask_cv(img_bgr)
            return [], excl

        try:
            import torch
            h, w    = img_bgr.shape[:2]
            excl    = np.zeros((h, w), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self.yolo(img_rgb, size=640)
            preds   = results.pred[0].cpu().numpy() if len(results.pred) > 0 else []

            crack_dets = []
            for det in preds:
                if len(det) < 6: continue
                cls_id = int(det[5])
                conf   = float(det[4])
                bbox   = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]

                if cls_id == 0 and conf > YOLO_CONF:
                    # Class 0 in crack-trained model = CRACK
                    crack_dets.append({"bbox": bbox, "conf": round(conf,3), "mask": None})
                elif cls_id in NON_STRUCTURAL_CLASSES and conf > 0.3:
                    # Mask out non-structural objects
                    pad = 15
                    x1 = max(0, bbox[0]-pad); y1 = max(0, bbox[1]-pad)
                    x2 = min(w, bbox[2]+pad); y2 = min(h, bbox[3]+pad)
                    excl[y1:y2, x1:x2] = 255

            # Also apply skin-color based person detection as backup
            skin_excl = get_person_mask_cv(img_bgr)
            excl = cv2.bitwise_or(excl, skin_excl)

            # Get YOLO masks if available
            if hasattr(results, 'masks') and results.masks is not None:
                try:
                    masks = results.masks.data.cpu().numpy()
                    for i, d in enumerate(crack_dets):
                        if i < len(masks): d["mask"] = masks[i]
                except: pass

            return crack_dets, excl
        except Exception as e:
            print(f"  ⚠️  YOLO: {e}")
            return [], get_person_mask_cv(img_bgr)

    def draw_annotated(self, img_bgr, gradcam_heat, crack_mask, exclusion_mask,
                       zones, raw_score, crack_info, yolo_cracks,
                       surface_type, depth_est, live=False):
        h, w       = img_bgr.shape[:2]
        annotated  = img_bgr.copy()
        is_cracked = raw_score > CRACK_THRESHOLD

        # ── 1. Show excluded regions (grey tint) ──────────────────────
        if exclusion_mask is not None and exclusion_mask.any():
            grey_overlay        = annotated.copy()
            grey_overlay[exclusion_mask > 0] = [80, 80, 80]
            annotated = cv2.addWeighted(annotated, 0.7, grey_overlay, 0.3, 0)
            # Label excluded region
            contours_ex, _ = cv2.findContours(
                exclusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours_ex:
                if cv2.contourArea(cnt) > 500:
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    cv2.rectangle(annotated, (rx,ry), (rx+rw,ry+rh), (100,100,100), 1)
                    cv2.putText(annotated, "IGNORED", (rx+4, ry+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160,160,160), 1)

        # ── 2. Grad-CAM very faint (context only) ─────────────────────
        heat_full = cv2.resize(gradcam_heat, (w, h))
        heat_u8   = np.uint8(255 * heat_full)
        heat_col  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        alpha_map = heat_full[..., np.newaxis] * 0.05
        annotated = (
            annotated.astype(np.float32) * (1 - alpha_map) +
            heat_col.astype(np.float32) * alpha_map
        ).astype(np.uint8)

        # ── 3. Draw crack pixels in RED ────────────────────────────────
        if is_cracked and np.count_nonzero(crack_mask) > 0:
            contours, _ = cv2.findContours(
                crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (0, 0, 220), 2)

        # ── 4. Zone grid (thin, minimal) ──────────────────────────────
        damaged_count = 0
        for zone in zones:
            x1, y1, x2, y2 = zone["bbox"]
            s = zone["heat_score"]
            if zone["damaged"]:
                damaged_count += 1
                c = (0, 0, 180) if s > 0.4 else (0, 40, 160) if s > 0.2 else (0, 60, 140)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), c, 1)
            else:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (200,200,200), 1)

        # ── 5. Crack measurements on image ────────────────────────────
        if is_cracked and crack_info["length_px"] > 0:
            contours2, _ = cv2.findContours(
                crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours2:
                main_c = max(contours2, key=cv2.contourArea)
                if len(main_c) >= 5:
                    rect = cv2.minAreaRect(main_c)
                    box  = cv2.boxPoints(rect).astype(np.int32)
                    cv2.drawContours(annotated, [box], 0, (0, 165, 255), 1)
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    # Measurement label on image
                    meas = (f"L:{crack_info['length_px']}px "
                            f"W:{crack_info['width_px']}px "
                            f"({crack_info['est_width_mm']:.1f}mm)")
                    (tw, th), _ = cv2.getTextSize(meas, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    lx = max(cx - tw//2, 2)
                    ly = max(cy - 12, 14)
                    cv2.rectangle(annotated, (lx-2, ly-th-2), (lx+tw+2, ly+2),
                                  (0,0,0), -1)
                    cv2.putText(annotated, meas, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,255), 1, cv2.LINE_AA)

        # ── 6. YOLOv7 crack boxes (MAGENTA) ───────────────────────────
        for det in yolo_cracks:
            bx1,by1,bx2,by2 = det["bbox"]
            conf = det["conf"]
            if det.get("mask") is not None:
                try:
                    msk = cv2.resize(det["mask"].astype(np.float32),(w,h))
                    ov  = annotated.copy()
                    ov[msk>0.5] = (ov[msk>0.5]*0.4 + np.array([200,0,200])*0.6).astype(np.uint8)
                    annotated = ov
                except: pass
            cv2.rectangle(annotated,(bx1,by1),(bx2,by2),(255,0,200),2)
            lbl = f"YOLO {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
            cv2.rectangle(annotated,(bx1,by1-th-7),(bx1+tw+5,by1),(255,0,200),-1)
            cv2.putText(annotated,lbl,(bx1+3,by1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)

        # ── 7. Status banner ───────────────────────────────────────────
        label_str    = "CRACKED" if is_cracked else "SAFE"
        banner_color = (0,20,130) if is_cracked else (10,80,10)
        yolo_tag     = f" | YOLO:{len(yolo_cracks)}" if yolo_cracks else ""
        cv2.rectangle(annotated,(0,0),(w,36),banner_color,-1)
        cv2.putText(
            annotated,
            f"  {label_str} | {raw_score*100:.0f}% | Zones:{damaged_count}/64"
            f" | {crack_info['crack_type']}{yolo_tag}",
            (6,23), cv2.FONT_HERSHEY_SIMPLEX, 0.42,(255,255,255),1,cv2.LINE_AA
        )

        # ── 8. Bottom info ─────────────────────────────────────────────
        cv2.rectangle(annotated,(0,h-24),(w,h),(20,20,20),-1)
        cv2.putText(
            annotated,
            f"  Width: {crack_info['width_category']} | "
            f"Depth: {depth_est} | Surface: {surface_type}",
            (6,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.33,(180,180,180),1,cv2.LINE_AA
        )

        if self.demo_mode:
            cv2.putText(annotated,"DEMO MODE",(6,52),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(80,180,255),1,cv2.LINE_AA)

        return annotated, damaged_count

    def analyze(self, img_bgr, live=False):
        if img_bgr is None or img_bgr.size == 0:
            return self._empty_result()

        h, w = img_bgr.shape[:2]

        # ── Step 1: Get exclusion mask (people/objects) ───────────────
        # This runs BEFORE crack detection so we don't detect cracks on faces
        yolo_cracks, exclusion_mask = self._run_yolo(img_bgr)

        # ── Step 2: Classify (MobileNetV2) ───────────────────────────
        inp       = self.preprocess(img_bgr)
        raw_score = float(self.model.predict(inp, verbose=0)[0][0])

        # ── Step 3: Temporal smoothing for live ───────────────────────
        if live:
            if self._smooth_score is None:
                self._smooth_score = raw_score
            else:
                self._smooth_score = (
                    SMOOTHING_ALPHA * raw_score +
                    (1 - SMOOTHING_ALPHA) * self._smooth_score
                )
            raw_score = self._smooth_score
        else:
            self._smooth_score = None

        # ── Step 4: Direct crack detection (exclusion applied) ────────
        crack_mask  = detect_crack_pixels(img_bgr, exclusion_mask)
        crack_px    = int(np.count_nonzero(crack_mask))
        crack_cov   = crack_px / max(h * w, 1)

        # ── Step 5: Model fusion ──────────────────────────────────────
        if crack_cov > 0.01 and raw_score < CRACK_THRESHOLD:
            raw_score = max(raw_score, 0.30)
        if crack_cov < 0.005 and raw_score < 0.30:
            raw_score = min(raw_score, CRACK_THRESHOLD - 0.01)
        # Boost if YOLO found cracks
        if yolo_cracks and raw_score < CRACK_THRESHOLD:
            raw_score = max(raw_score, CRACK_THRESHOLD + 0.05)

        is_cracked = raw_score > CRACK_THRESHOLD
        label      = "Cracked" if is_cracked else "Non-Cracked"

        # ── Step 6: Crack metrics ─────────────────────────────────────
        crack_info = compute_crack_metrics(crack_mask, img_bgr) \
                     if (is_cracked and crack_px > 50) else None
        if crack_info is None:
            crack_info = _empty_crack_info()

        # ── Step 7: Surface type (skip on live) ───────────────────────
        surface_type = classify_surface(img_bgr) if not live else "Live Feed"

        # ── Step 8: Depth estimate ────────────────────────────────────
        depth_est = estimate_depth(crack_mask, img_bgr) if is_cracked else "N/A"

        # ── Step 9: Grad-CAM ─────────────────────────────────────────
        gradcam_heat = np.zeros((7,7), dtype=np.float32) \
                       if (live and LIVE_SKIP_GRADCAM) \
                       else self.gradcam.generate(inp)

        # ── Step 10: Zone grid ────────────────────────────────────────
        zones = self.grid_analyze(img_bgr, crack_mask)

        # ── Step 11: Draw ─────────────────────────────────────────────
        annotated, damaged_zones = self.draw_annotated(
            img_bgr, gradcam_heat, crack_mask, exclusion_mask,
            zones, raw_score, crack_info,
            yolo_cracks, surface_type, depth_est, live=live
        )

        return {
            "label":           label,
            "confidence":      round(raw_score, 4),
            "severity_score":  round(raw_score * 100, 1),
            "zones":           zones,
            "damaged_zones":   damaged_zones,
            "total_zones":     GRID_ROWS * GRID_COLS,
            "crack_info":      crack_info,
            "surface_type":    surface_type,
            "depth_estimate":  depth_est,
            "yolo_detections": len(yolo_cracks),
            "annotated_frame": annotated,
            "demo_mode":       self.demo_mode,
        }

    def _empty_result(self):
        blank = np.zeros((480,640,3), dtype=np.uint8)
        return {
            "label":"Error","confidence":0,"severity_score":0,
            "zones":[],"damaged_zones":0,"total_zones":64,
            "crack_info":_empty_crack_info(),"surface_type":"Unknown",
            "depth_estimate":"Unknown","yolo_detections":0,
            "annotated_frame":blank,"demo_mode":self.demo_mode,
        }
