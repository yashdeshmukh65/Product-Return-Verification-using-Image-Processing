"""
comparison.py
Handles: SSIM, pixel-wise difference, histogram comparison,
         damage detection using morphological operations, and decision logic.
"""

import cv2
import numpy as np


# ─── SSIM (Structural Similarity Index) ────────────────────────────────────────

def compute_ssim(img1, img2):
    """
    Structural Similarity Index Measure (SSIM).
    Compares luminance, contrast, and structure between two grayscale images.
    Returns a score in [-1, 1]; closer to 1 means more similar.
    Pure image processing — no ML involved.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    def conv(img):
        return cv2.filter2D(img, -1, window)

    mu1, mu2 = conv(img1), conv(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = conv(img1 ** 2) - mu1_sq
    sigma2_sq = conv(img2 ** 2) - mu2_sq
    sigma12   = conv(img1 * img2) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


# ─── Pixel-wise Difference ──────────────────────────────────────────────────────

def pixel_difference(img1, img2):
    """
    Pixel-wise absolute difference (image subtraction).
    Highlights regions where the two images differ.
    """
    return cv2.absdiff(img1, img2)


# ─── Histogram Comparison ──────────────────────────────────────────────────────

def compare_histograms(hist1, hist2):
    """
    Compare two normalized histograms using correlation method.
    Returns a score in [-1, 1]; 1 = identical distribution.
    """
    return cv2.compareHist(
        hist1.reshape(-1, 1).astype(np.float32),
        hist2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL
    )


# ─── Damage Detection ──────────────────────────────────────────────────────────

def detect_damage(diff_img):
    """
    Damage detection pipeline:
    1. Thresholding   — isolate significant difference regions
    2. Morphological dilation  — expand detected regions (fills small gaps)
    3. Morphological erosion   — shrink back to remove noise
    Returns the cleaned binary mask and a count of damaged pixels.
    """
    # Step 1: Thresholding — pixels with diff > 30 are marked as changed
    _, thresh = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    # Step 2: Dilation — morphological operation to expand damage regions
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Step 3: Erosion — morphological operation to remove small noise blobs
    eroded = cv2.erode(dilated, kernel, iterations=1)

    damaged_pixels = int(np.sum(eroded > 0))
    return eroded, damaged_pixels


def highlight_damage(color_img, damage_mask):
    """Overlay damage mask on the color image in red for visualization."""
    result = color_img.copy()
    result[damage_mask > 0] = [0, 0, 255]  # Red highlight
    return result


# ─── Decision Logic ────────────────────────────────────────────────────────────

def decide(ssim_score):
    """
    Decision based on SSIM similarity score:
      > 0.80  → SAME product
      0.50–0.80 → DAMAGED product
      < 0.50  → DIFFERENT product
    """
    if ssim_score > 0.80:
        return "SAME"
    elif ssim_score >= 0.50:
        return "DAMAGED"
    else:
        return "DIFFERENT"
