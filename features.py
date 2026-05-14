"""
features.py
UNIT 4 - Feature Extraction
Handles: edge detection (Canny), contour detection, histogram extraction
Enhanced with product-focused feature extraction
"""

import cv2
import numpy as np


def extract_edges(gray_img, low_threshold=30, high_threshold=100):
    """
    Feature Extraction (UNIT 4): Canny edge detection.
    Detects boundaries by finding rapid intensity changes using gradient computation.
    Adjusted thresholds for better product edge detection.
    """
    return cv2.Canny(gray_img, threshold1=low_threshold, threshold2=high_threshold)


def extract_contours(edge_img):
    """
    Feature Extraction (UNIT 4): Contour detection.
    Finds connected curves of edge pixels — represents object boundaries.
    """
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def extract_histogram(gray_img, bins=256, mask=None):
    """
    Feature Extraction (UNIT 4): Intensity histogram.
    Counts pixel frequency at each intensity level (0–255).
    Normalized so images of different sizes can be compared.
    Enhanced with optional mask to focus on product regions.
    """
    hist = cv2.calcHist([gray_img], [0], mask, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def create_product_mask(gray_img):
    """
    Create a mask to focus histogram calculation on product regions.
    Uses Otsu's thresholding to separate product from remaining background.
    """
    # Apply Otsu's thresholding to separate foreground (product) from background
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def extract_texture_features(gray_img):
    """
    Extract texture features using Local Binary Pattern (LBP) concept.
    Simplified version for product texture analysis.
    """
    # Calculate local standard deviation (texture measure)
    kernel = np.ones((9, 9), np.float32) / 81
    mean = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
    sqr_mean = cv2.filter2D((gray_img.astype(np.float32))**2, -1, kernel)
    variance = sqr_mean - mean**2
    variance = np.clip(variance, 0, None)  # Prevent negative values from float precision
    texture = np.sqrt(variance)
    
    # Normalize texture values
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return texture


def extract_shape_features(contours):
    """
    Extract shape-based features from contours.
    Returns area, perimeter, and aspect ratio of the largest contour.
    """
    if not contours:
        return {"area": 0, "perimeter": 0, "aspect_ratio": 1.0}
    
    # Find largest contour (main product)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Bounding rectangle for aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0
    
    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio
    }


def get_features(gray_img):
    """
    Enhanced feature extraction for preprocessed grayscale image.
    Returns edges, contours, histogram (with mask), texture, and shape features.
    """
    # Extract edges with optimized parameters
    edges = extract_edges(gray_img)
    
    # Extract contours
    contours = extract_contours(edges)
    
    # Create product mask for focused histogram
    product_mask = create_product_mask(gray_img)
    
    # Extract histogram using product mask
    histogram = extract_histogram(gray_img, mask=product_mask)
    
    # Extract texture features
    texture = extract_texture_features(gray_img)
    
    # Extract shape features
    shape_features = extract_shape_features(contours)
    
    return {
        "edges": edges,
        "contours": contours,
        "histogram": histogram,
        "texture": texture,
        "shape": shape_features,
        "mask": product_mask
    }
