"""
SafeSight Helmet Detection System
Classical Computer Vision Pipeline (No YOLO)
Pipeline: Video → Frame Processing → Person Detection (HOG+SVM) → Head ROI → HOG Features → SVM Classification → Alert
"""

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
HEAD_ROI_FRACTION   = 0.30   # top 30% of person bbox = head region
HELMET_OFFSET_RATIO = 0.25   # expand head bbox by 25% each side for helmet coverage
HOG_WIN_SIZE        = (64, 64)
HOG_ORIENTATIONS    = 12          # increased for richer gradient detail
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "helmet_svm_model.pkl")


# ─────────────────────────────────────────────
#  HOG FEATURE EXTRACTION
# ─────────────────────────────────────────────
def _preprocess(image: np.ndarray) -> tuple:
    """Preprocess: resize with INTER_CUBIC (better for upscaling tiny images),
       apply CLAHE contrast enhancement."""
    resized = cv2.resize(image, HOG_WIN_SIZE, interpolation=cv2.INTER_CUBIC)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
        resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # CLAHE for contrast enhancement — critical for tiny noisy crops
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    return resized, gray


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """Resize to HOG window size, convert to grayscale, extract HOG features."""
    _, gray = _preprocess(image)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features



from skimage.feature import local_binary_pattern


def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    _, gray = _preprocess(image)
    # Multi-scale LBP: 3 scales for richer texture representation
    feats = []
    for (P, R) in [(8, 1), (16, 2), (24, 3)]:
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-7)  # L1-normalize
        feats.append(hist)
    return np.hstack(feats)


def extract_color_features(image: np.ndarray) -> np.ndarray:
    """Extract color histogram in HSV — helmets have distinctive colors
       (yellow, white, orange, red) that grayscale features miss entirely."""
    resized = cv2.resize(image, HOG_WIN_SIZE, interpolation=cv2.INTER_CUBIC)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    feats = []
    # H: 18 bins, S: 16 bins, V: 16 bins — matches improved trainer
    for ch, bins, rng in [(0, 18, (0, 180)), (1, 16, (0, 256)), (2, 16, (0, 256))]:
        hist = cv2.calcHist([hsv], [ch], None, [bins], list(rng)).flatten()
        hist /= (hist.sum() + 1e-7)
        feats.append(hist)
    return np.hstack(feats)


def extract_edge_shape_features(image: np.ndarray) -> np.ndarray:
    """Edge density, gradient statistics, and Hu moments —
       helmets have smooth curved edges vs hair/bare heads."""
    _, gray = _preprocess(image)
    # Canny edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    # Sobel gradient magnitudes
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_stats = [np.mean(mag), np.std(mag), np.median(mag), np.max(mag)]   # added max
    # Hu moments (log-transformed, first 5)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()[:5]
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return np.hstack([[edge_density], grad_stats, hu])


def extract_features(image: np.ndarray) -> np.ndarray:
    """Combined feature vector: HOG + multi-scale LBP + color histograms +
       edge/shape features. Total ~1900+ dimensions for rich SVM input."""
    hog_feat   = extract_hog_features(image)
    lbp_feat   = extract_lbp_features(image)
    color_feat = extract_color_features(image)
    edge_feat  = extract_edge_shape_features(image)
    return np.hstack([hog_feat, lbp_feat, color_feat, edge_feat])

# ─────────────────────────────────────────────
#  SYNTHETIC DATASET TRAINER
#  (used when no real dataset is supplied)
# ─────────────────────────────────────────────
def _make_synthetic_head(has_helmet: bool, size: int = 64) -> np.ndarray:
    """
    Generates a simple synthetic head image.
    Helmet = round blob on upper half (hard hat shape).
    No helmet = bare oval head.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 220
    cx, cy = size // 2, int(size * 0.6)
    # draw face oval
    cv2.ellipse(img, (cx, cy), (size//4, size//3), 0, 0, 360, (200, 170, 130), -1)
    if has_helmet:
        hat_y = int(size * 0.35)
        cv2.ellipse(img, (cx, hat_y), (size//3, size//6), 0, 180, 360, (30, 30, 200), -1)
        cv2.rectangle(img, (cx - size//3, hat_y), (cx + size//3, hat_y + 6), (20, 20, 180), -1)
    # add noise
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def train_synthetic_model(n_samples: int = 800) -> Pipeline:
    """Train SVM on synthetic helmet/no-helmet head images."""
    logger.info("Training SVM on synthetic dataset (%d samples)…", n_samples)
    X, y = [], []
    for _ in range(n_samples // 2):
        X.append(extract_features(_make_synthetic_head(True)))
        y.append(1)   # helmet
        X.append(extract_features(_make_synthetic_head(False)))
        y.append(0)   # no helmet

    X = np.array(X)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
    ])
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", MODEL_PATH)
    return model


def load_or_train_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            logger.info("Loaded existing model from %s", MODEL_PATH)
            return pickle.load(f)
    return train_synthetic_model()


# ─────────────────────────────────────────────
#  HEAD ROI EXTRACTION WITH OFFSET
# ─────────────────────────────────────────────
def get_head_roi_with_offset(frame: np.ndarray, px: int, py: int, pw: int, ph: int):
    """
    Given a person bounding box, extract the head ROI (top HEAD_ROI_FRACTION)
    and expand it by HELMET_OFFSET_RATIO to include space for the helmet.

    Returns:
        roi        - cropped head image (BGR)
        bbox_draw  - (x1, y1, x2, y2) expanded bbox for drawing
    """
    h, w = frame.shape[:2]

    # Base head region: top fraction of the person box
    head_h = int(ph * HEAD_ROI_FRACTION)
    hx1, hy1 = px, py
    hx2, hy2 = px + pw, py + head_h

    # Expand by offset (to capture helmet above/around head)
    off_x = int(pw * HELMET_OFFSET_RATIO)
    off_y = int(head_h * HELMET_OFFSET_RATIO)

    ex1 = max(0, hx1 - off_x)
    ey1 = max(0, hy1 - off_y)
    ex2 = min(w, hx2 + off_x)
    ey2 = min(h, hy2 + off_y)

    roi = frame[ey1:ey2, ex1:ex2]
    return roi, (ex1, ey1, ex2, ey2)


# ─────────────────────────────────────────────
#  ALERT SYSTEM
# ─────────────────────────────────────────────
class AlertSystem:
    def __init__(self, cooldown_sec: float = 3.0, log_dir: str = "alerts"):
        self.cooldown      = cooldown_sec
        self._last_alert   = 0.0
        self.violation_count = 0
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, "violations.log")

    def trigger(self, frame: np.ndarray, frame_no: int):
        now = time.time()
        if now - self._last_alert < self.cooldown:
            return
        self._last_alert = now
        self.violation_count += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] ⚠ ALERT #{self.violation_count}: Helmet violation at frame {frame_no}"
        logger.warning(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")
        # Save snapshot
        snap_path = os.path.join(self.log_dir, f"violation_{self.violation_count:04d}_f{frame_no}.jpg")
        cv2.imwrite(snap_path, frame)


# ─────────────────────────────────────────────
#  MAIN DETECTOR CLASS
# ─────────────────────────────────────────────
class HelmetDetector:
    def __init__(self, skip_frames: int = 2, resize_width: int = 640):
        self.hog_person   = cv2.HOGDescriptor()
        self.hog_person.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.model        = load_or_train_model()
        self.alert        = AlertSystem()
        self.skip_frames  = skip_frames
        self.resize_width = resize_width
        self._frame_no    = 0

    # ── per-frame inference ──
    def process_frame(self, frame: np.ndarray):
        """
        Returns annotated frame with bounding boxes and labels.
        Green box + 'HELMET'    → safe
        Red box   + 'NO HELMET' → violation + alert triggered
        """
        h, w = frame.shape[:2]
        scale = self.resize_width / w
        small = cv2.resize(frame, (self.resize_width, int(h * scale)))

        self._frame_no += 1

        # ── step 1: Person detection via HOG + pre-trained SVM ──
        persons, _ = self.hog_person.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        annotated = frame.copy()

        if len(persons) == 0:
            self._overlay_status(annotated, "No persons detected", (200, 200, 200))
            return annotated

        violations = 0

        for (sx, sy, sw, sh) in persons:
            # Scale coords back to original frame
            px = int(sx / scale);  py = int(sy / scale)
            pw = int(sw / scale);  ph = int(sh / scale)

            # Draw full person box (thin, gray)
            cv2.rectangle(annotated, (px, py), (px+pw, py+ph), (180, 180, 180), 1)

            # ── step 2: Head ROI with helmet offset ──
            roi, (ex1, ey1, ex2, ey2) = get_head_roi_with_offset(annotated, px, py, pw, ph)

            if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
                continue

            # ── step 3: HOG feature extraction ──
            feats = extract_features(roi).reshape(1, -1)

            # ── step 4: SVM classification ──
            pred  = self.model.predict(feats)[0]
            proba = self.model.predict_proba(feats)[0]
            conf  = proba[pred] * 100

            has_helmet = (pred == 1)

            if has_helmet:
                color = (0, 200, 0)          # green
                label = f"HELMET  {conf:.0f}%"
            else:
                color = (0, 0, 220)          # red (BGR)
                label = f"NO HELMET  {conf:.0f}%"
                violations += 1

            # Draw expanded head bbox
            cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), color, 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (ex1, ey1 - lh - 8), (ex1 + lw + 4, ey1), color, -1)
            cv2.putText(annotated, label, (ex1 + 2, ey1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # ── step 5: Alert ──
        if violations > 0:
            self.alert.trigger(annotated, self._frame_no)
            self._overlay_status(annotated, f"⚠ VIOLATION: {violations} worker(s) without helmet!", (0, 0, 220))
        else:
            self._overlay_status(annotated, "✓ All workers wearing helmets", (0, 180, 0))

        # Frame counter
        cv2.putText(annotated, f"Frame: {self._frame_no}", (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return annotated

    # ── helpers ──
    @staticmethod
    def _overlay_status(frame: np.ndarray, text: str, color):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 38), (30, 30, 30), -1)
        cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # ── video processing loop ──
    def run_video(self, source, output_path: str = None, show: bool = False):
        """
        source: path to video file, or 0 for webcam.
        output_path: optional path to save annotated video.
        show: display live window (requires display).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video: %dx%d @ %.1f FPS, %d frames", width, height, fps, total)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        t_start = time.time()
        processed = 0
        raw_frame_no = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frame_no += 1

                # Frame skipping for performance
                if raw_frame_no % (self.skip_frames + 1) != 0:
                    if writer:
                        writer.write(frame)   # write original for skipped frames
                    continue

                annotated = self.process_frame(frame)
                processed += 1

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow("SafeSight - Helmet Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if processed % 30 == 0:
                    elapsed = time.time() - t_start
                    eff_fps = processed / elapsed if elapsed > 0 else 0
                    logger.info("Processed %d/%d frames | %.1f FPS | Violations: %d",
                                processed, total // (self.skip_frames + 1),
                                eff_fps, self.alert.violation_count)
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        logger.info("Done. %d frames processed in %.1fs | Total violations: %d | Log: %s",
                    processed, elapsed, self.alert.violation_count, self.alert.log_path)
        return self.alert.violation_count
