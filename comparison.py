"""
comparison.py
Handles: ECC alignment, SSIM, pixel difference, histogram comparison,
         texture comparison, shape comparison, damage detection,
         damage highlight, and decision logic.
All operations strictly on product region only (background = black = excluded).
"""

import cv2
import numpy as np


# ─── ECC Image Alignment ──────────────────────────────────────────────────────
def align_images_ecc(img_ref, img_target):
    """
    ECC (Enhanced Correlation Coefficient) alignment:
    Aligns img_target to match the spatial position of img_ref.

    Why ECC?
    - Handles translation, rotation, and scale differences between shots
    - Purely mathematical — no feature matching, no deep learning
    - Robust to illumination changes (works on normalized images)
    - Prevents false SSIM differences caused by positional shifts

    Motion model: MOTION_EUCLIDEAN (translation + rotation)
    Falls back to unaligned target if ECC fails.
    """
    # ECC works on grayscale
    ref_gray    = img_ref    if img_ref.ndim == 2 else cv2.cvtColor(img_ref,    cv2.COLOR_BGR2GRAY)
    target_gray = img_target if img_target.ndim == 2 else cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, target_gray,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None, 5
        )
        h, w = img_ref.shape[:2]

        if img_target.ndim == 2:
            aligned = cv2.warpAffine(
                img_target, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        else:
            aligned = cv2.warpAffine(
                img_target, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
        return aligned

    except Exception:
        # ECC failed (too different or no overlap) — return original
        return img_target


# ─── Product Mask from Black Background ───────────────────────────────────────
def get_product_mask(img1, img2, mask1=None, mask2=None):
    """
    Build product-only mask using non-black pixel intersection.
    Background = pure black (0) from preprocessing.
    Intersection ensures only pixels that are product in BOTH images are used.
    """
    nb1 = (img1 > 0).astype(np.uint8) * 255
    nb2 = (img2 > 0).astype(np.uint8) * 255
    auto = cv2.bitwise_and(nb1, nb2)

    if mask1 is not None and mask2 is not None:
        explicit = cv2.bitwise_and(mask1, mask2)
        return cv2.bitwise_and(auto, explicit)

    return auto


# ─── SSIM ─────────────────────────────────────────────────────────────────────
def compute_ssim(img1, img2, mask1=None, mask2=None):
    """
    SSIM computed strictly on product region.
    img2 is ECC-aligned to img1 before comparison to eliminate
    false differences caused by positional shifts between shots.
    Black background pixels are excluded via product mask.
    """
    # Align img2 to img1 using ECC
    img2_aligned = align_images_ecc(img1, img2)

    product_mask = get_product_mask(img1, img2_aligned, mask1, mask2)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    i1 = img1.astype(np.float64)
    i2 = img2_aligned.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    def conv(x):
        return cv2.filter2D(x, -1, window)

    mu1, mu2 = conv(i1), conv(i2)
    mu1_sq   = mu1 ** 2
    mu2_sq   = mu2 ** 2
    mu1_mu2  = mu1 * mu2
    s1_sq    = conv(i1 ** 2) - mu1_sq
    s2_sq    = conv(i2 ** 2) - mu2_sq
    s12      = conv(i1 * i2) - mu1_mu2

    num      = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den      = (mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2)
    ssim_map = num / den

    if np.sum(product_mask) > 0:
        return float(np.mean(ssim_map[product_mask > 0]))

    return float(np.mean(ssim_map))


# ─── Pixel Difference ─────────────────────────────────────────────────────────
def pixel_difference(img1, img2, mask1=None, mask2=None):
    """
    Pixel-wise absolute difference after ECC alignment.
    Strictly on product region — background (black) pixels excluded.
    """
    img2_aligned = align_images_ecc(img1, img2)
    diff         = cv2.absdiff(img1, img2_aligned)
    product_mask = get_product_mask(img1, img2_aligned, mask1, mask2)
    diff         = cv2.bitwise_and(diff, diff, mask=product_mask)
    return diff


# ─── Histogram Comparison ─────────────────────────────────────────────────────
def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    return cv2.compareHist(
        hist1.reshape(-1, 1).astype(np.float32),
        hist2.reshape(-1, 1).astype(np.float32),
        method
    )


# ─── Texture Comparison ───────────────────────────────────────────────────────
def compare_texture(texture1, texture2):
    t1 = texture1.flatten().astype(np.float32)
    t2 = texture2.flatten().astype(np.float32)
    corr = np.corrcoef(t1, t2)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


# ─── Shape Comparison ─────────────────────────────────────────────────────────
def compare_shape_features(shape1, shape2):
    if shape1['area'] == 0 or shape2['area'] == 0:
        return 0.0

    area_ratio = min(shape1['area'], shape2['area']) / max(shape1['area'], shape2['area'])

    if shape1['perimeter'] == 0 or shape2['perimeter'] == 0:
        perim_ratio = 0.0
    else:
        perim_ratio = min(shape1['perimeter'], shape2['perimeter']) / \
                      max(shape1['perimeter'], shape2['perimeter'])

    aspect_sim = max(0.0, 1.0 - abs(shape1['aspect_ratio'] - shape2['aspect_ratio']))

    return area_ratio * 0.4 + perim_ratio * 0.3 + aspect_sim * 0.3


# ─── Damage Detection ─────────────────────────────────────────────────────────
def detect_damage(diff_img, mask1=None, mask2=None):
    """
    Damage detection strictly on product region:
    1. Product mask from non-black pixels
    2. Adaptive threshold on product pixels only
       (mean + 1.5*std — sensitive to real damage)
    3. Morphological dilation — expand damage regions, fill gaps
    4. Morphological erosion  — remove small noise blobs
    5. Final AND with product mask — zero background contamination
    Returns clean damage mask + damaged pixel count.
    """
    product_mask = (diff_img > 0).astype(np.uint8) * 255

    if mask1 is not None and mask2 is not None:
        explicit     = cv2.bitwise_and(mask1, mask2)
        product_mask = cv2.bitwise_and(product_mask, explicit)

    px = diff_img[product_mask > 0]
    if len(px) == 0:
        return np.zeros_like(diff_img), 0

    thresh_val = max(20, float(np.mean(px)) + 1.5 * float(np.std(px)))

    _, thresh = cv2.threshold(diff_img, thresh_val, 255, cv2.THRESH_BINARY)
    thresh    = cv2.bitwise_and(thresh, thresh, mask=product_mask)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(thresh,  kernel, iterations=2)
    eroded  = cv2.erode(dilated,  kernel, iterations=1)
    eroded  = cv2.bitwise_and(eroded, eroded, mask=product_mask)

    return eroded, int(np.sum(eroded > 0))


# ─── Damage Highlight ─────────────────────────────────────────────────────────
def highlight_damage(color_img, damage_mask):
    """
    Overlay damage in red ONLY on product pixels.
    Background pixels (pure black) are never highlighted.
    """
    result         = color_img.copy()
    is_product     = np.any(color_img > 0, axis=2)
    damage_region  = (damage_mask > 0) & is_product
    result[damage_region] = [0, 0, 255]
    return result


# ─── Composite Score ──────────────────────────────────────────────────────────
def compute_composite_score(ssim, hist, texture, shape, damaged_px, total_px):
    damage_ratio = damaged_px / max(total_px, 1)
    damage_score = max(0.0, 1.0 - damage_ratio * 10)

    return (
        0.35 * max(0.0, ssim)    +
        0.25 * max(0.0, hist)    +
        0.15 * max(0.0, texture) +
        0.15 * shape             +
        0.10 * damage_score
    )


# ─── Decision ─────────────────────────────────────────────────────────────────
def decide_enhanced(ssim, hist, texture, shape, damaged_px, total_px):
    """
    Multi-criteria decision — returns strictly SAME / DAMAGED / DIFFERENT.
    Uses composite score + damage ratio for robust classification.
    """
    composite    = compute_composite_score(ssim, hist, texture, shape, damaged_px, total_px)
    damage_ratio = damaged_px / max(total_px, 1)

    # Strong histogram match = same product type
    if hist > 0.85:
        if damage_ratio < 0.10:
            return "SAME", composite
        else:
            return "DAMAGED", composite

    # Composite-based logic
    if composite >= 0.40:
        if damage_ratio <= 0.08:
            return "SAME", composite
        else:
            return "DAMAGED", composite

    elif composite >= 0.30:
        if shape < 0.3 or hist < 0.4:
            return "DIFFERENT", composite
        return "DAMAGED", composite

    else:
        return "DIFFERENT", composite


# ─── Backward compatibility ───────────────────────────────────────────────────
def decide(ssim_score):
    if ssim_score > 0.80:
        return "SAME"
    elif ssim_score >= 0.50:
        return "DAMAGED"
    else:
        return "DIFFERENT"