"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering

Pipeline:
  Load -> Resize
    -> LAB + CLAHE (illumination normalization)
    -> GrabCut foreground extraction
    -> Morphological Closing
    -> Canny Edge Detection
    -> Largest Centered Contour
    -> Clean Binary Mask
    -> Background = Pure Black
    -> Tight Crop -> Resize to 256x256
    -> Grayscale -> Gaussian Blur -> CLAHE
"""

import cv2
import numpy as np

WORK_SIZE   = (512, 512)
OUTPUT_SIZE = (256, 256)


# ──────────────────────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


# ─── STEP 1: LAB + CLAHE illumination normalization ───────────────────────────
def normalize_lab_clahe(img):
    """
    Convert to LAB and apply CLAHE only on the L (lightness) channel.
    This normalizes illumination differences (shadows, reflections, lighting)
    before segmentation so GrabCut clusters on color/texture, not brightness.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─── STEP 2: GrabCut foreground extraction ────────────────────────────────────
def grabcut_segment(img):
    """
    GrabCut initialized with a center rectangle (product assumed in center).
    Uses iterative graph-cut optimization with Gaussian Mixture Models
    to separate foreground (product) from background at pixel level.
    More robust than K-Means when product and background share similar colors.

    Rect margin: 8% from each edge — assumes product occupies center region.
    5 iterations for stable convergence.
    """
    h, w = img.shape[:2]
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    rect = (margin_x, margin_y,
            w - 2 * margin_x,
            h - 2 * margin_y)

    mask    = np.zeros((h, w), np.uint8)
    bgd     = np.zeros((1, 65), np.float64)
    fgd     = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    # Definite + probable foreground = product
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return fg_mask


# ─── STEP 3: Morphological Closing ────────────────────────────────────────────
def morph_close(mask):
    """
    Morphological Closing (dilate then erode):
    Fills small holes and gaps left by GrabCut inside the product region.
    Connects nearby foreground fragments into one solid region.
    Elliptical kernel avoids sharp rectangular artifacts.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)


# ─── STEP 4: Canny Edge Detection ─────────────────────────────────────────────
def canny_edges(img, mask):
    """
    Canny edge detection on the GrabCut-masked region only.
    Detects sharp intensity transitions at product boundaries and surface features.
    Combined with the GrabCut mask to reinforce product boundary detection.
    Thresholds 40/120 tuned for product images with varied textures.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply mask before edge detection — ignore background edges
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    blurred     = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    edges       = cv2.Canny(blurred, 40, 120)
    # Combine edges with GrabCut mask
    combined    = cv2.bitwise_or(mask, edges)
    return combined


# ─── STEP 5: Largest Centered Contour ─────────────────────────────────────────
def get_largest_centered_contour(combined_mask, img_shape):
    """
    Detect all external contours from the combined GrabCut+Canny mask.
    Filter by minimum area (3% of image) to remove noise fragments.
    Score each valid contour by:
      - Area score (70%): larger contour = more likely the product
      - Center score (30%): closer to image center = more likely product
    Select the highest-scoring contour as the main product.
    Draw a FRESH filled mask from ONLY this contour.
    All other contours (shadows, reflections, background objects) are discarded.
    """
    h, w = img_shape[:2]
    cx, cy   = w // 2, h // 2
    min_area = h * w * 0.03
    max_dist = np.sqrt(cx**2 + cy**2)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid:
        # Fallback: center 70% rectangle
        fb = np.zeros((h, w), dtype=np.uint8)
        m  = 0.15
        cv2.rectangle(fb,
                      (int(w*m), int(h*m)),
                      (int(w*(1-m)), int(h*(1-m))),
                      255, -1)
        return fb, None

    max_area = max(cv2.contourArea(c) for c in valid)

    def score(c):
        area = cv2.contourArea(c)
        M    = cv2.moments(c)
        if M['m00'] == 0:
            return 0.0
        ccx  = M['m10'] / M['m00']
        ccy  = M['m01'] / M['m00']
        dist = np.sqrt((ccx - cx)**2 + (ccy - cy)**2)
        return 0.70 * (area / max_area) + 0.30 * (1.0 - dist / max_dist)

    best  = max(valid, key=score)

    clean = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean, [best], -1, 255, thickness=cv2.FILLED)

    return clean, best


# ─── STEP 6: GrabCut refinement pass 2 (optional) ────────────────────────────
def grabcut_refine_with_mask(img, contour_mask, best_contour):
    """
    Second GrabCut pass initialized from the largest contour mask.
    Provides pixel-precise boundary refinement after contour selection.
    Three initialization zones:
      - Definite BG: outside dilated contour
      - Probable FG: inside contour mask
      - Definite FG: inside heavily eroded contour (core product)
    Falls back to contour_mask on failure.
    """
    h, w = img.shape[:2]
    if best_contour is None:
        return contour_mask

    try:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

        eroded  = cv2.erode(contour_mask,  ke, iterations=3)
        dilated = cv2.dilate(contour_mask, kd, iterations=3)

        gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc[dilated == 0]          = cv2.GC_BGD
        gc[contour_mask == 255]   = cv2.GC_PR_FGD
        gc[eroded == 255]         = cv2.GC_FGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

        refined = np.where(
            (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        if np.sum(refined == 255) < (h * w * 0.02):
            return contour_mask

        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kc)

        return refined

    except Exception:
        return contour_mask


# ─── STEP 7: Extract clean product ────────────────────────────────────────────
def extract_clean_product(img, final_mask):
    """
    Set ALL background pixels to pure black (0, 0, 0).
    Tight bounding box crop around non-zero mask pixels.
    Pure black background ensures pixel_difference() and SSIM
    never include background in their calculations.
    """
    product = img.copy()
    product[final_mask == 0] = 0

    coords = cv2.findNonZero(final_mask)
    if coords is None:
        return product

    x, y, bw, bh = cv2.boundingRect(coords)
    x  = max(0, x);  y  = max(0, y)
    bw = min(bw, img.shape[1] - x)
    bh = min(bh, img.shape[0] - y)

    cropped = product[y:y+bh, x:x+bw]
    return cropped if cropped.shape[0] >= 20 and cropped.shape[1] >= 20 else product


# ─── STEP 8: Illumination normalization ───────────────────────────────────────
def remove_illumination(gray):
    """
    Subtract large-scale illumination field estimated by heavy Gaussian blur.
    Removes gradients caused by shadows, uneven lighting, and reflections.
    Result is normalized to 0-255 range.
    """
    illum     = cv2.GaussianBlur(gray.astype(np.float32), (61, 61), 0)
    corrected = np.clip(gray.astype(np.float32) - illum + 128.0, 0, 255)
    return corrected.astype(np.uint8)


# ─── MAIN PREPROCESS ──────────────────────────────────────────────────────────
def preprocess(path):
    """
    Full pipeline:
    Load -> Resize(512)
      -> LAB+CLAHE normalization
      -> GrabCut (rect init)
      -> Morphological Closing
      -> Canny edges (on masked region)
      -> Largest Centered Contour
      -> GrabCut refinement (mask init)
      -> Pure black background
      -> Tight crop -> Resize(256)
      -> Grayscale -> Gaussian Blur
      -> Illumination removal
      -> CLAHE

    Returns:
        enhanced  (256x256 grayscale) — for SSIM / features / damage
        segmented (256x256 BGR)       — for UI display (black background)
    """
    img = load_image(path)
    img = cv2.resize(img, WORK_SIZE, interpolation=cv2.INTER_AREA)

    # Illumination normalization before segmentation
    norm_img = normalize_lab_clahe(img)

    # GrabCut pass 1 (rect init)
    gc_mask = grabcut_segment(norm_img)

    # Morphological closing
    closed = morph_close(gc_mask)

    # Canny + combine
    combined = canny_edges(norm_img, closed)

    # Largest centered contour
    contour_mask, best_contour = get_largest_centered_contour(combined, img.shape)

    # GrabCut pass 2 (mask init refinement)
    final_mask = grabcut_refine_with_mask(norm_img, contour_mask, best_contour)

    # Extract product (pure black BG, tight crop)
    product = extract_clean_product(img, final_mask)

    # Resize to output
    resized = cv2.resize(product, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Remove illumination gradients
    normalized = remove_illumination(blurred)

    # CLAHE
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    return enhanced, resized
