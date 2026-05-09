"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering
Pipeline: Input → Segmentation → ROI Extraction → CLAHE → Output
Handles: polygon ROI, product segmentation, CLAHE enhancement
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


def create_polygon_roi(img, method='auto'):
    """
    Create polygon ROI for precise product region extraction.
    Methods: 'auto' (automatic detection) or 'manual' (interactive selection)
    """
    h, w = img.shape[:2]
    
    if method == 'auto':
        # Automatic polygon ROI using edge detection and contour approximation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with adaptive thresholds
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the product)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            poly_roi = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Ensure minimum area
            if cv2.contourArea(poly_roi) > (w * h * 0.1):  # At least 10% of image
                return poly_roi.reshape(-1, 2)
    
    # Fallback: Create rectangular ROI around center
    margin = 0.15  # 15% margin from edges
    x1, y1 = int(w * margin), int(h * margin)
    x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
    
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def extract_roi_with_polygon(img, poly_points):
    """
    Extract Region of Interest using polygon mask.
    Returns the cropped image containing only the product region.
    """
    # Create mask from polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_points.astype(np.int32)], 255)
    
    # Apply mask to image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Find bounding rectangle of the polygon
    x, y, w, h = cv2.boundingRect(poly_points.astype(np.int32))
    
    # Crop to bounding rectangle
    cropped_img = masked_img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Replace masked areas with mean color to avoid black regions
    if np.sum(cropped_mask) > 0:
        mean_color = cv2.mean(cropped_img, mask=cropped_mask)[:3]
        cropped_img[cropped_mask == 0] = mean_color
    
    return cropped_img, cropped_mask


def advanced_segmentation(img):
    """
    Advanced product segmentation using multiple techniques.
    Combines edge detection, color segmentation, and morphological operations.
    """
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Method 1: Edge-based segmentation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    # Method 2: Color-based segmentation (remove background colors)
    # Assume background is often white/light colored
    lower_bg = np.array([0, 0, 200])  # Light colors in HSV
    upper_bg = np.array([180, 30, 255])
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    fg_mask = cv2.bitwise_not(bg_mask)
    
    # Combine edge and color information
    combined = cv2.bitwise_or(edges, fg_mask)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours and select the largest one
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create polygon ROI from contour
        epsilon = 0.015 * cv2.arcLength(largest_contour, True)
        poly_roi = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return poly_roi.reshape(-1, 2)
    
    # Fallback to center region
    h, w = img.shape[:2]
    margin = 0.2
    x1, y1 = int(w * margin), int(h * margin)
    x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
    
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def resize_image(img):
    """Spatial operation: resize to fixed dimensions for consistent comparison."""
    return cv2.resize(img, TARGET_SIZE)


def to_grayscale(img):
    """Convert BGR to grayscale — reduces 3-channel to single intensity channel."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_img, kernel_size=(3, 3)):
    """
    Spatial Filtering (UNIT 3): Gaussian blur for noise removal.
    Uses a weighted average kernel — pixels closer to center have higher weight.
    """
    return cv2.GaussianBlur(gray_img, kernel_size, 0)


def clahe_enhancement(gray_img, clip_limit=3.0, tile_size=(8, 8)):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization (UNIT 3).
    Final step: applies adaptive contrast enhancement to the ROI.
    Enhanced parameters for better product visibility.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def preprocess(path):
    """
    Enhanced preprocessing pipeline following your specified flow:
    
    Input Images → Segmentation → ROI Extraction → CLAHE → Output
    
    Steps:
    1. Load input image
    2. Segmentation: Advanced product segmentation with polygon ROI
    3. ROI Extraction: Extract product region using polygon mask
    4. Resize to standard dimensions
    5. Convert to grayscale
    6. Apply Gaussian blur
    7. CLAHE: Adaptive contrast enhancement
    
    Returns both the processed grayscale image and the processed color image.
    """
    # Step 1: Input Images
    img = load_image(path)
    
    # Step 2: Segmentation - Create polygon ROI for product
    poly_roi = advanced_segmentation(img)
    
    # Step 3: ROI Extraction - Extract product region using polygon
    roi_img, roi_mask = extract_roi_with_polygon(img, poly_roi)
    
    # Resize extracted ROI to standard size
    resized_img = resize_image(roi_img)
    
    # Convert to grayscale
    gray = to_grayscale(resized_img)
    
    # Apply light Gaussian blur for noise reduction
    blurred = apply_gaussian_blur(gray)
    
    # Step 4: CLAHE - Final adaptive contrast enhancement
    enhanced = clahe_enhancement(blurred)
    
    return enhanced, resized_img  # (processed grayscale, processed color for display)
