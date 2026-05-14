"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering

Pipeline:
  Input
    -> LAB Color Conversion
    -> K-Means Clustering (k=2) on LAB
    -> Morphological Opening + Closing
    -> Largest Contour Detection
    -> Clean Binary Mask (product only)
    -> GrabCut Refinement
    -> Background = Black (complete removal)
    -> Tight Bounding Box Crop
    -> Resize -> Grayscale -> Gaussian Blur -> CLAHE
"""

import cv2
import numpy as np

TARGET_SIZE = (256, 256)


def load_image(path):
    """Load image from disk in BGR format."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def kmeans_lab(img, k=2):
    """
    Step 1+2 — LAB Color Conversion + K-Means Clustering (k=2):

    Why LAB?
    - L channel = lightness (separates brightness from color)
    - A channel = green-red axis
    - B channel = blue-yellow axis
    - LAB is perceptually uniform — similar colors cluster better than BGR
    - Separates product from background more cleanly under varied lighting

    Why k=2?
    - Only 2 clusters needed: foreground (product) vs background
    - Simpler, faster, and more reliable than k=3 for binary separation
    """
    h, w = img.shape[:2]

    # Convert BGR -> LAB for perceptually uniform clustering
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Reshape to flat pixel list (N, 3)
    pixel_data = lab.reshape((-1, 3)).astype(np.float32)

    # K-Means with k=2: foreground vs background
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        pixel_data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape((h, w)).astype(np.uint8)

    # Identify background cluster: dominates image border pixels
    border = np.concatenate([
        labels[0, :], labels[-1, :],
        labels[:, 0], labels[:, -1]
    ])
    border_counts = np.bincount(border.astype(np.int32), minlength=k)
    bg_label = int(np.argmax(border_counts))
    fg_label = 1 - bg_label  # Since k=2, foreground is the other label

    # Raw binary mask: product=255, background=0
    raw_mask = np.where(labels == fg_label, 255, 0).astype(np.uint8)

    return raw_mask


def morphological_cleanup(mask):
    """
    Step 3 — Morphological Opening + Closing:

    OPENING  (erode then dilate):
    - Removes small isolated noise blobs outside the product
    - Breaks thin connections between product and background

    CLOSING  (dilate then erode):
    - Fills small holes and gaps inside the product region
    - Connects nearby product fragments into one solid region

    Two passes of each for thorough cleanup on phone camera images.
    """
    kernel_open  = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((11, 11), np.uint8)

    # Opening first: remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
    # Closing second: fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def get_largest_contour_mask(mask, img_shape):
    """
    Step 4+5 — Largest Contour Detection + Clean Binary Mask:

    - Detect ALL external contours from the cleaned mask
    - Select ONLY the single largest contour = main product
    - Discard every other contour (shadows, reflections, noise fragments)
    - Draw a fresh filled mask using ONLY this largest contour
    - Result: perfectly clean binary mask with zero background fragments
    """
    h, w = img_shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: center 70% region
        clean = np.zeros((h, w), dtype=np.uint8)
        m = 0.15
        cv2.rectangle(clean,
                      (int(w * m), int(h * m)),
                      (int(w * (1-m)), int(h * (1-m))),
                      255, -1)
        return clean, None

    # Pick the single largest contour
    largest = max(contours, key=cv2.contourArea)

    # Minimum area check: must cover at least 3% of image
    if cv2.contourArea(largest) < (h * w * 0.03):
        clean = np.zeros((h, w), dtype=np.uint8)
        m = 0.15
        cv2.rectangle(clean,
                      (int(w * m), int(h * m)),
                      (int(w * (1-m)), int(h * (1-m))),
                      255, -1)
        return clean, None

    # Draw ONLY the largest contour — completely fresh mask
    # All other background fragments are gone
    clean = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)

    return clean, largest


def grabcut_refinement(img, contour_mask, largest_contour):
    """
    Step 6 — GrabCut Refinement:

    Uses the largest contour mask as initialization for GrabCut.
    GrabCut builds Gaussian Mixture Models for foreground and background
    and iteratively refines the boundary at pixel level.

    Initialization:
    - GC_FGD (definite foreground): pixels inside eroded contour mask
    - GC_PR_FGD (probable foreground): pixels inside contour mask
    - GC_BGD (definite background): pixels outside dilated contour mask
    - GC_PR_BGD (probable background): remaining pixels

    Falls back to contour_mask if GrabCut fails or produces poor result.
    """
    h, w = img.shape[:2]

    if largest_contour is None:
        return contour_mask

    # Get bounding rect from largest contour
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Add padding
    pad = 15
    x  = max(0, x - pad)
    y  = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    if bw < 30 or bh < 30:
        return contour_mask

    try:
        # Build GrabCut initialization mask from contour mask
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Definite background: outside dilated contour
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(contour_mask, kernel, iterations=2)
        gc_mask[dilated == 0] = cv2.GC_BGD

        # Probable foreground: inside contour mask
        gc_mask[contour_mask == 255] = cv2.GC_PR_FGD

        # Definite foreground: inside eroded contour (core product)
        eroded = cv2.erode(contour_mask, kernel, iterations=2)
        gc_mask[eroded == 255] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Run GrabCut using mask initialization (5 iterations)
        cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_MASK)

        # Extract foreground: definite + probable foreground
        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        # Validate: must have reasonable coverage
        if np.sum(refined == 255) < (h * w * 0.03):
            return contour_mask

        # Final closing to fill any holes GrabCut left
        k2 = np.ones((7, 7), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k2)

        return refined

    except Exception:
        return contour_mask


def extract_product(img, final_mask):
    """
    Step 7 — Product Extraction:

    - Set ALL background pixels to pure black (0,0,0)
    - Crop tightly to the product bounding box
    - Result: only the product object, zero background contamination
    - This ensures SSIM and damage detection operate on product only
    """
    h, w = img.shape[:2]

    # Set background pixels to black — complete removal
    product_only = img.copy()
    product_only[final_mask == 0] = 0

    # Tight crop to bounding box of non-zero mask pixels
    coords = cv2.findNonZero(final_mask)
    if coords is None:
        return product_only

    x, y, bw, bh = cv2.boundingRect(coords)
    x  = max(0, x)
    y  = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    cropped = product_only[y:y+bh, x:x+bw]

    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        return product_only

    return cropped


def apply_gaussian_blur(gray_img, kernel_size=(3, 3)):
    """Spatial Filtering (UNIT 3): Gaussian blur for noise removal."""
    return cv2.GaussianBlur(gray_img, kernel_size, 0)


def clahe_enhancement(gray_img, clip_limit=3.0, tile_size=(8, 8)):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization (UNIT 3).
    Divides image into 8x8 tiles, applies independent contrast per tile.
    clip_limit=3.0 prevents noise over-amplification.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def preprocess(path, use_grabcut=True):
    """
    Complete preprocessing pipeline:

    Input
      -> LAB Color Conversion
      -> K-Means (k=2)                  [foreground vs background]
      -> Morphological Open + Close     [noise removal + gap filling]
      -> Largest Contour Detection      [select main product only]
      -> Clean Binary Mask              [all background fragments removed]
      -> GrabCut Refinement             [pixel-precise boundary]
      -> Background = Black             [complete background removal]
      -> Tight Bounding Box Crop        [product-only region]
      -> Resize to 256x256
      -> Grayscale
      -> Gaussian Blur
      -> CLAHE

    Returns:
        enhanced  (256x256 grayscale) — for SSIM, features, damage detection
        segmented (256x256 BGR)       — for UI display (product only, black bg)
    """
    # Load
    img = load_image(path)

    # LAB K-Means (k=2)
    raw_mask = kmeans_lab(img, k=2)

    # Morphological cleanup
    clean_mask = morphological_cleanup(raw_mask)

    # Largest contour → clean product-only mask
    contour_mask, largest_contour = get_largest_contour_mask(clean_mask, img.shape)

    # GrabCut refinement
    if use_grabcut:
        final_mask = grabcut_refinement(img, contour_mask, largest_contour)
    else:
        final_mask = contour_mask

    # Extract product (background = black, tight crop)
    product_img = extract_product(img, final_mask)

    # Resize
    resized = cv2.resize(product_img, TARGET_SIZE)

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = apply_gaussian_blur(gray)

    # CLAHE
    enhanced = clahe_enhancement(blurred)

    return enhanced, resized
