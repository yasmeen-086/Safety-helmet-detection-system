"""
SafeSight — Streamlit Web Interface
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from helmet_detector import HelmetDetector, load_or_train_model, extract_features


class ScaledPipeline:
    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf = clf

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))


class ThresholdedPipeline:
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold

    def predict(self, X):
        p = self.base.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base.predict_proba(X)
        
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SafeSight — Helmet Detection",
    page_icon="🪖",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; color: #e0e0e0; }
.stMetric { background: #1e2130; border-radius: 8px; padding: 12px; }
.violation-box { background: #3d0000; border: 2px solid #ff4444;
                 border-radius: 8px; padding: 12px; margin: 8px 0; }
.safe-box { background: #003d00; border: 2px solid #44ff44;
            border-radius: 8px; padding: 12px; margin: 8px 0; }
.header-title { font-size: 2rem; font-weight: 800; color: #00cfff; margin-bottom: 0; }
.subheader    { color: #888; font-size: 0.9rem; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────
st.markdown('<p class="header-title">🪖 SafeSight</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Classical CV Helmet Detection · HOG + SVM · No YOLO</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar settings ─────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    skip_frames   = st.slider("Frame skip (performance)", 0, 5, 2,
                               help="Skip N frames between inferences")
    resize_width  = st.slider("Resize width (px)", 320, 960, 640, step=64,
                               help="Smaller = faster, less accurate")
    alert_cooldown = st.slider("Alert cooldown (s)", 1.0, 10.0, 3.0)
    show_hog_viz   = st.checkbox("Show HOG visualization", value=False)
    st.divider()
    st.markdown("**Pipeline**")
    st.code("""Video
→ Frame resize & skip
→ HOG person detection
→ Head ROI + offset
→ HOG feature extract
→ SVM classify
→ Alert if violation""", language="text")

# ─── Model loading ────────────────────────────
@st.cache_resource
def get_model():
    return load_or_train_model()

model = get_model()

# ─── Tabs ─────────────────────────────────────
tab_video, tab_webcam, tab_demo = st.tabs(["📹 Video File", "📷 Webcam Demo", "🔬 Frame Analysis"])

# ══════════════════════════════════════════════
#  TAB 1 — Video File
# ══════════════════════════════════════════════
with tab_video:
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        # Save to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        src_path = tfile.name

        col1, col2, col3 = st.columns(3)
        cap_tmp = cv2.VideoCapture(src_path)
        total_f  = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps  = cap_tmp.get(cv2.CAP_PROP_FPS)
        width_v  = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_v = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        with col1: st.metric("Total Frames", f"{total_f:,}")
        with col2: st.metric("FPS",          f"{vid_fps:.1f}")
        with col3: st.metric("Resolution",   f"{width_v}×{height_v}")

        if st.button("▶ Run Helmet Detection", type="primary", use_container_width=True):
            detector = HelmetDetector(skip_frames=skip_frames, resize_width=resize_width)
            detector.alert.cooldown = alert_cooldown

            out_path = src_path.replace(".mp4", "_annotated.mp4")

            progress_bar = st.progress(0, "Processing video…")
            status_box   = st.empty()
            metrics_row  = st.columns(3)
            frame_display = st.empty()

            # Open video and process frame by frame with live preview
            cap = cv2.VideoCapture(src_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_out = cv2.VideoWriter(out_path, fourcc, vid_fps, (width_v, height_v))

            frame_idx = 0
            t0 = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % (skip_frames + 1) == 0:
                    annotated = detector.process_frame(frame)
                    writer_out.write(annotated)

                    # Show every 10th processed frame in UI
                    if frame_idx % (10 * (skip_frames + 1)) == 0:
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_display.image(rgb, caption=f"Frame {frame_idx}", use_container_width=True)

                    pct = frame_idx / max(total_f, 1)
                    progress_bar.progress(min(pct, 1.0), f"Frame {frame_idx}/{total_f}")
                    eff_fps = frame_idx / max(time.time() - t0, 0.001)
                    status_box.markdown(f"**{eff_fps:.1f} FPS** | Violations: **{detector.alert.violation_count}**")
                else:
                    writer_out.write(frame)

            cap.release()
            writer_out.release()
            progress_bar.progress(1.0, "✅ Done!")

            v = detector.alert.violation_count
            if v > 0:
                st.markdown(f'<div class="violation-box">⚠️ <b>{v} violation alert(s)</b> were triggered. Snapshots saved to <code>alerts/</code>.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-box">✅ No violations detected — all workers wearing helmets.</div>', unsafe_allow_html=True)

            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="safesight_output.mp4", mime="video/mp4", use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 2 — Webcam Demo
# ══════════════════════════════════════════════
with tab_webcam:
    st.info("💡 Click **Start Webcam** to run real-time detection from your camera. Press **Stop** to end.")
    run_webcam = st.toggle("Start Webcam")

    if run_webcam:
        detector = HelmetDetector(skip_frames=skip_frames, resize_width=resize_width)
        detector.alert.cooldown = alert_cooldown
        cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()
        stats_placeholder  = st.empty()

        if not cap.isOpened():
            st.error("No webcam found. Please connect a camera.")
        else:
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read from webcam.")
                    break
                annotated = detector.process_frame(frame)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                stats_placeholder.markdown(
                    f"**Frame:** {detector._frame_no} | **Violations:** {detector.alert.violation_count}"
                )
            cap.release()

# ══════════════════════════════════════════════
#  TAB 3 — Frame Analysis / HOG Visualization
# ══════════════════════════════════════════════
with tab_demo:
    st.subheader("🔬 Single Image Analysis")
    st.caption("Upload an image to inspect HOG features and helmet classification.")

    img_upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img_up")

    if img_upload:
        file_bytes = np.frombuffer(img_upload.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img_rgb, caption="Input Image", use_container_width=True)

        # HOG on the whole image (for visualization)
        from skimage.feature import hog as sk_hog
        resized   = cv2.resize(img_bgr, (128, 128))
        gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        feats, hog_img = sk_hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
            feature_vector=True
        )
        # Normalize HOG viz
        hog_norm = ((hog_img - hog_img.min()) / (hog_img.max() - hog_img.min() + 1e-8) * 255).astype(np.uint8)
        hog_color = cv2.applyColorMap(hog_norm, cv2.COLORMAP_INFERNO)

        with col_b:
            st.image(cv2.cvtColor(hog_color, cv2.COLOR_BGR2RGB), caption="HOG Visualization", use_container_width=True)

        # Classify
        feat_vec = extract_features(img_bgr).reshape(1, -1)
        pred     = model.predict(feat_vec)[0]
        proba    = model.predict_proba(feat_vec)[0]

        if pred == 1:
            st.markdown(f'<div class="safe-box">✅ <b>HELMET DETECTED</b> — Confidence: {proba[1]*100:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="violation-box">⚠️ <b>NO HELMET DETECTED</b> — Confidence: {proba[0]*100:.1f}%</div>', unsafe_allow_html=True)

        with st.expander("HOG feature vector (first 50 values)"):
            st.line_chart(feats[:50])

        st.caption(f"Feature vector length: {len(feats)}")
