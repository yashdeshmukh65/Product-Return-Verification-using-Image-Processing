"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering

Pipeline:
  Input
    -> K-Means Segmentation (k=3)
    -> Morphological Cleanup (close + open)
    -> Largest Contour Selection
    -> Clean Binary Mask (product only, all background removed)
    -> GrabCut Refinement (optional, improves foreground accuracy)
    -> ROI Extraction (product-only crop)
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


def kmeans_segmentation(img, k=3):
    """
    Step 1 — K-Means Segmentation:
    Cluster all pixels into k color groups.
    Identify background cluster by border pixel dominance.
    Return raw binary mask: product=255, background=0.
    """
    h, w = img.shape[:2]

    # Reshape image to flat pixel list (N, 3)
    pixel_data = img.reshape((-1, 3)).astype(np.float32)

    # K-Means: 10 attempts, max 20 iterations, epsilon=1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.reshape((h, w))

    # Identify background: cluster dominating image border pixels
    border = np.concatenate([
        labels[0, :], labels[-1, :],
        labels[:, 0], labels[:, -1]
    ])
    border_counts = np.bincount(border.astype(np.int32), minlength=k)
    bg_label = np.argmax(border_counts)

    # Binary mask: non-background = product
    raw_mask = np.where(labels != bg_label, 255, 0).astype(np.uint8)

    return raw_mask


def morphological_cleanup(mask):
    """
    Step 2 — Morphological Operations:
    CLOSE: fills small holes and gaps inside the product region.
    OPEN:  removes small isolated noise blobs outside the product.
    Uses a large 9x9 kernel for thorough cleanup on phone camera images.
    """
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def get_largest_contour_mask(mask):
    """
    Step 3 — Largest Contour Selection:
    Detect all external contours from the cleaned K-Means mask.
    Select ONLY the largest contour = main product region.
    Discard all smaller contours (noise, reflections, background fragments).
    Create a fresh clean binary mask from this single largest contour.
    This ensures ONLY the product object remains — no background regions.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: use center 70% of image as product region
        h, w = mask.shape
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        m = 0.15
        cv2.rectangle(
            clean_mask,
            (int(w * m), int(h * m)),
            (int(w * (1 - m)), int(h * (1 - m))),
            255, -1
        )
        return clean_mask

    # Select the single largest contour (main product)
    largest_contour = max(contours, key=cv2.contourArea)

    # Verify it covers at least 5% of image area (not just noise)
    h, w = mask.shape
    min_area = h * w * 0.05
    if cv2.contourArea(largest_contour) < min_area:
        # Fallback to center crop
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        m = 0.15
        cv2.rectangle(
            clean_mask,
            (int(w * m), int(h * m)),
            (int(w * (1 - m)), int(h * (1 - m))),
            255, -1
        )
        return clean_mask

    # Draw ONLY the largest contour on a fresh black mask
    # This completely removes all other background fragments
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return clean_mask


def grabcut_refinement(img, contour_mask):
    """
    Step 4 — GrabCut Refinement (optional):
    Uses the largest-contour mask as initialization for GrabCut.
    GrabCut iteratively refines the foreground/background boundary
    using Gaussian Mixture Models on color information.
    This gives a more accurate product boundary than contour alone.
    Falls back to contour_mask if GrabCut fails.
    """
    h, w = img.shape[:2]

    # Get bounding rect from contour mask for GrabCut rect initialization
    coords = cv2.findNonZero(contour_mask)
    if coords is None:
        return contour_mask

    x, y, bw, bh = cv2.boundingRect(coords)

    # Add small padding to bounding rect
    pad = 10
    x  = max(0, x - pad)
    y  = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    # GrabCut needs minimum rect size
    if bw < 20 or bh < 20:
        return contour_mask

    try:
        gc_mask  = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (x, y, bw, bh)

        # Run GrabCut with rect initialization (5 iterations)
        cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Pixels marked as definite/probable foreground = product
        refined_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Validate: refined mask must have reasonable coverage
        if np.sum(refined_mask == 255) < (h * w * 0.03):
            return contour_mask

        # Final morphological cleanup on GrabCut result
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

        return refined_mask

    except Exception:
        # GrabCut failed — return contour mask as fallback
        return contour_mask


def extract_product(img, final_mask):
    """
    Step 5 — Product Extraction:
    Apply the final clean mask to the original image.
    Crop tightly to the product bounding rectangle.
    Fill any remaining non-product pixels with the product mean color
    to avoid black artifacts in CLAHE and SSIM comparison.
    """
    h, w = img.shape[:2]

    # Apply mask — zero out all background pixels
    product_only = cv2.bitwise_and(img, img, mask=final_mask)

    # Tight crop to product bounding rectangle
    coords = cv2.findNonZero(final_mask)
    if coords is None:
        return img  # Fallback: return original

    x, y, bw, bh = cv2.boundingRect(coords)
    x  = max(0, x)
    y  = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    cropped       = product_only[y:y+bh, x:x+bw].copy()
    cropped_mask  = final_mask[y:y+bh, x:x+bw]

    # Fill background pixels (inside bounding box but outside mask) with mean color
    if np.sum(cropped_mask) > 0:
        mean_color = cv2.mean(cropped, mask=cropped_mask)[:3]
        cropped[cropped_mask == 0] = mean_color

    # Safety: ensure valid size
    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        return img

    return cropped


def apply_gaussian_blur(gray_img, kernel_size=(3, 3)):
    """Spatial Filtering (UNIT 3): Gaussian blur for noise removal."""
    return cv2.GaussianBlur(gray_img, kernel_size, 0)


def clahe_enhancement(gray_img, clip_limit=3.0, tile_size=(8, 8)):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization (UNIT 3).
    Divides image into 8x8 tiles, applies independent contrast enhancement per tile.
    clip_limit=3.0 prevents noise over-amplification.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def preprocess(path, use_grabcut=True):
    """
    Full preprocessing pipeline:

    Input Image
        -> K-Means Segmentation (k=3)          [color clustering]
        -> Morphological Cleanup               [close + open]
        -> Largest Contour Selection           [remove all background fragments]
        -> Clean Binary Mask                   [product only]
        -> GrabCut Refinement (optional)       [precise boundary]
        -> Product Extraction                  [tight crop, no background]
        -> Resize to 256x256
        -> Grayscale Conversion
        -> Gaussian Blur
        -> CLAHE Enhancement

    Returns:
        enhanced  (256x256 grayscale) — for SSIM, feature extraction, damage detection
        segmented (256x256 BGR color) — for display in UI (product only, no background)
    """
    # Step 1: Load
    img = load_image(path)

    # Step 2: K-Means Segmentation
    raw_mask = kmeans_segmentation(img, k=3)

    # Step 3: Morphological Cleanup
    clean_mask = morphological_cleanup(raw_mask)

    # Step 4: Largest Contour — clean product-only mask
    contour_mask = get_largest_contour_mask(clean_mask)

    # Step 5: GrabCut Refinement (refines boundary using color model)
    if use_grabcut:
        final_mask = grabcut_refinement(img, contour_mask)
    else:
        final_mask = contour_mask

    # Step 6: Extract product-only region (tight crop, no background)
    product_img = extract_product(img, final_mask)

    # Step 7: Resize
    resized = cv2.resize(product_img, TARGET_SIZE)

    # Step 8: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 9: Gaussian Blur
    blurred = apply_gaussian_blur(gray)

    # Step 10: CLAHE
    enhanced = clahe_enhancement(blurred)

    return enhanced, resized
