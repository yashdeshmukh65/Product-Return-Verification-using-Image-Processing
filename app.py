"""
app.py
Fake Return Product Detection — Streamlit UI
Runs locally and on Streamlit Cloud.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

from preprocessing import preprocess
from features import get_features
from comparison import (
    compute_ssim, pixel_difference, compare_histograms,
    detect_damage, highlight_damage, decide_enhanced, compare_texture, compare_shape_features
)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Return Product Detector",
    page_icon="🔍",
    layout="wide"
)

# ─── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Fake Return Product Detector")
st.caption("Upload an original and a returned product image to detect fraud using Image Processing.")
st.divider()


# ─── Helper: save uploaded file to temp path ───────────────────────────────────
def save_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name


# ─── Helper: convert BGR numpy array → PIL Image ──────────────────────────────
def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def gray_to_pil(img):
    return Image.fromarray(img)


# ─── Upload Section ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Original Product Image")
    orig_file = st.file_uploader("Upload original image", type=["jpg", "jpeg", "png", "bmp"],
                                  key="original")
    if orig_file:
        st.image(orig_file, caption="Original", use_container_width=True)

with col2:
    st.subheader("📬 Returned Product Image")
    ret_file = st.file_uploader("Upload return image", type=["jpg", "jpeg", "png", "bmp"],
                                 key="return")
    if ret_file:
        st.image(ret_file, caption="Returned", use_container_width=True)

st.divider()

# ─── Detect Button ─────────────────────────────────────────────────────────────
detect_clicked = st.button("🔎 DETECT", type="primary", use_container_width=True)

# ─── Detection Pipeline ────────────────────────────────────────────────────────
if detect_clicked:
    if not orig_file or not ret_file:
        st.warning("⚠️ Please upload both images before detecting.")
    else:
        with st.spinner("Running image processing pipeline..."):

            # Save uploads to temp files (OpenCV needs file paths)
            orig_path = save_temp(orig_file)
            ret_path  = save_temp(ret_file)

            # ── Preprocessing (UNIT 3) ────────────────────────────────────────
            orig_gray, orig_color = preprocess(orig_path)
            ret_gray,  ret_color  = preprocess(ret_path)

            # ── Feature Extraction (UNIT 4) ───────────────────────────────────
            orig_features = get_features(orig_gray)
            ret_features  = get_features(ret_gray)

            # ── Comparison ────────────────────────────────────────────────────
            ssim_score = compute_ssim(orig_gray, ret_gray, orig_features['mask'], ret_features['mask'])
            diff_img   = pixel_difference(orig_gray, ret_gray, orig_features['mask'], ret_features['mask'])
            hist_score = compare_histograms(orig_features['histogram'], ret_features['histogram'])
            texture_score = compare_texture(orig_features['texture'], ret_features['texture'])
            shape_score = compare_shape_features(orig_features['shape'], ret_features['shape'])

            # ── Damage Detection — morphological operations ───────────────────
            damage_mask, damaged_pixels = detect_damage(diff_img, orig_features['mask'], ret_features['mask'])
            damage_overlay = highlight_damage(ret_color, damage_mask)

            # ── Enhanced Decision ──────────────────────────────────────────────
            total_pixels = orig_gray.shape[0] * orig_gray.shape[1]
            result, composite_score = decide_enhanced(
                ssim_score, hist_score, texture_score, shape_score, damaged_pixels, total_pixels
            )

            # Cleanup temp files
            os.unlink(orig_path)
            os.unlink(ret_path)

        st.divider()

        # ── Result Banner ─────────────────────────────────────────────────────
        result_config = {
            "SAME":      ("✅ SAME PRODUCT",      "success"),
            "DAMAGED":   ("⚠️  DAMAGED PRODUCT",  "warning"),
            "DIFFERENT": ("❌ DIFFERENT PRODUCT", "error"),
        }
        label, msg_type = result_config[result]
        getattr(st, msg_type)(f"**Result: {label}**")

        # ── Score Metrics ─────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("SSIM Score",              f"{ssim_score:.3f}",      help="Structural Similarity (0–1). Higher = more similar.")
        m2.metric("Histogram Correlation",   f"{hist_score:.3f}",      help="Color/intensity distribution match (0–1).")
        m3.metric("Texture Correlation",     f"{texture_score:.3f}",   help="Surface texture similarity (0–1).")
        m4.metric("Shape Similarity",        f"{shape_score:.3f}",     help="Product shape/outline match (0–1).")
        m5.metric("Composite Score",         f"{composite_score:.3f}",  help="Overall weighted similarity score.")
        
        st.metric("Damaged Pixels", f"{damaged_pixels}", help="Number of pixels flagged as damaged/changed.")

        st.divider()

        # ── Decision Threshold Info ───────────────────────────────────────────
        with st.expander("📊 How the decision was made"):
            st.markdown("""
| SSIM Range | Decision |
|---|---|
| > 0.80 | ✅ SAME |
| 0.50 – 0.80 | ⚠️ DAMAGED |
| < 0.50 | ❌ DIFFERENT |
            """)
            st.progress(float(np.clip(ssim_score, 0, 1)), text=f"SSIM: {ssim_score:.4f}")

        # ── 4-Panel Image Analysis ────────────────────────────────────────────
        st.subheader("🖼️ Image Analysis")
        c1, c2, c3, c4 = st.columns(4)

        c1.image(bgr_to_pil(orig_color),    caption="Original",        use_container_width=True)
        c2.image(bgr_to_pil(ret_color),     caption="Returned",        use_container_width=True)
        c3.image(gray_to_pil(diff_img),     caption="Difference Map",  use_container_width=True)
        c4.image(bgr_to_pil(damage_overlay),caption="Damage Highlight (Red)", use_container_width=True)

        # ── Histogram Plot ────────────────────────────────────────────────────
        st.subheader("📈 Intensity Histogram Comparison")
        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#2a2a3e")
        ax.plot(orig_features['histogram'], color="#7c3aed", label="Original", linewidth=1.5)
        ax.plot(ret_features['histogram'],  color="#f59e0b", label="Returned",  linewidth=1.5)
        ax.legend(facecolor="#2a2a3e", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a5c")
        st.pyplot(fig)
        plt.close(fig)
