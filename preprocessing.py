"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering
Pipeline: Input -> K-Means Segmentation -> Polygon ROI Extraction -> CLAHE -> Output
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
    K-Means Segmentation (UNIT 3):
    Clusters image pixels into K color groups.
    The largest non-background cluster = product region.
    
    Steps:
    1. Reshape image to list of pixels
    2. Apply K-Means clustering (k=3: background, product, shadow/edge)
    3. Identify background cluster (largest cluster touching image border)
    4. Create binary mask: product=255, background=0
    5. Morphological cleanup
    """
    h, w = img.shape[:2]

    # Step 1: Reshape to (N, 3) pixel array for K-Means
    pixel_data = img.reshape((-1, 3)).astype(np.float32)

    # Step 2: K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Reshape labels back to image shape
    labels = labels.reshape((h, w))

    # Step 3: Identify background cluster
    # Background = cluster that has most pixels touching the image border
    border_pixels = np.concatenate([
        labels[0, :],        # top row
        labels[-1, :],       # bottom row
        labels[:, 0],        # left column
        labels[:, -1]        # right column
    ])

    # Count which cluster dominates the border
    border_counts = np.bincount(border_pixels.astype(np.int32), minlength=k)
    bg_label = np.argmax(border_counts)

    # Step 4: Create binary mask — product = 255, background = 0
    product_mask = np.where(labels != bg_label, 255, 0).astype(np.uint8)

    # Step 5: Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    return product_mask


def get_polygon_roi(product_mask):
    """
    Extract polygon ROI from K-Means product mask.
    Finds the largest contour and approximates it to a polygon.
    """
    contours, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: center region polygon
        h, w = product_mask.shape
        m = 0.15
        return np.array([
            [int(w*m), int(h*m)], [int(w*(1-m)), int(h*m)],
            [int(w*(1-m)), int(h*(1-m))], [int(w*m), int(h*(1-m))]
        ])

    # Largest contour = main product
    largest = max(contours, key=cv2.contourArea)

    # Douglas-Peucker polygon approximation
    epsilon = 0.015 * cv2.arcLength(largest, True)
    polygon = cv2.approxPolyDP(largest, epsilon, True)

    return polygon.reshape(-1, 2)


def extract_roi(img, polygon):
    """
    Extract product ROI using polygon mask.
    Returns segmented color image (background filled with mean color).
    """
    h, w = img.shape[:2]

    # Create polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

    # Apply mask
    masked = cv2.bitwise_and(img, img, mask=mask)

    # Crop to bounding rectangle
    x, y, bw, bh = cv2.boundingRect(polygon.astype(np.int32))
    x, y = max(0, x), max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    cropped = masked[y:y+bh, x:x+bw]
    cropped_mask = mask[y:y+bh, x:x+bw]

    # Fill non-product pixels with mean product color (no black artifacts)
    if np.sum(cropped_mask) > 0:
        mean_color = cv2.mean(cropped, mask=cropped_mask)[:3]
        cropped[cropped_mask == 0] = mean_color

    # Safety check: ensure valid crop
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return img

    return cropped


def apply_gaussian_blur(gray_img, kernel_size=(3, 3)):
    """Spatial Filtering (UNIT 3): Gaussian blur for noise removal."""
    return cv2.GaussianBlur(gray_img, kernel_size, 0)


def clahe_enhancement(gray_img, clip_limit=3.0, tile_size=(8, 8)):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization (UNIT 3).
    Applies adaptive contrast enhancement tile-by-tile.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def preprocess(path):
    """
    Full pipeline:
    Input -> K-Means Segmentation -> Polygon ROI Extraction -> Resize -> Grayscale -> Blur -> CLAHE

    Returns:
        enhanced  : preprocessed grayscale image (for comparison)
        segmented : segmented color image (for display in UI)
    """
    # Step 1: Load
    img = load_image(path)

    # Step 2: K-Means Segmentation
    product_mask = kmeans_segmentation(img, k=3)

    # Step 3: Polygon ROI from mask
    polygon = get_polygon_roi(product_mask)

    # Step 4: Extract ROI (product only, background removed)
    segmented = extract_roi(img, polygon)

    # Step 5: Resize to standard size
    resized = cv2.resize(segmented, TARGET_SIZE)

    # Step 6: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 7: Gaussian Blur
    blurred = apply_gaussian_blur(gray)

    # Step 8: CLAHE
    enhanced = clahe_enhancement(blurred)

    return enhanced, resized  # (grayscale for processing, color for display)
