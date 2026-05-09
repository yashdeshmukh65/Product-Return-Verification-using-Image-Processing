"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering
Handles: resizing, grayscale conversion, Gaussian blur, CLAHE, product segmentation
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


def segment_product(img):
    """
    Product Segmentation: Extract only the product region, remove background.
    Uses multiple techniques for robust segmentation.
    """
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Method 1: Edge-based segmentation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: return center crop if no contours found
        h, w = img.shape[:2]
        crop_size = min(h, w) // 2
        center_y, center_x = h // 2, w // 2
        y1 = max(0, center_y - crop_size // 2)
        y2 = min(h, center_y + crop_size // 2)
        x1 = max(0, center_x - crop_size // 2)
        x2 = min(w, center_x + crop_size // 2)
        return img[y1:y2, x1:x2]
    
    # Find the largest contour (likely the main product)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding around the product (10% on each side)
    padding = 0.1
    img_h, img_w = img.shape[:2]
    
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)
    
    # Extract product region
    product_region = img[y1:y2, x1:x2]
    
    # If extracted region is too small, use center crop as fallback
    if product_region.shape[0] < 50 or product_region.shape[1] < 50:
        crop_size = min(img_h, img_w) // 2
        center_y, center_x = img_h // 2, img_w // 2
        y1 = max(0, center_y - crop_size // 2)
        y2 = min(img_h, center_y + crop_size // 2)
        x1 = max(0, center_x - crop_size // 2)
        x2 = min(img_w, center_x + crop_size // 2)
        return img[y1:y2, x1:x2]
    
    return product_region


def remove_background(img):
    """
    Background removal using GrabCut algorithm.
    Creates a mask to separate product from background.
    """
    # Create mask for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Define rectangle around the center (likely product location)
    h, w = img.shape[:2]
    rect = (w//6, h//6, w*2//3, h*2//3)
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        result = img * mask2[:, :, np.newaxis]
        
        # Find bounding box of non-zero pixels
        coords = cv2.findNonZero(mask2)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return result[y:y+h, x:x+w]
        else:
            return img
    except:
        # If GrabCut fails, return original image
        return img


def resize_image(img):
    """Spatial operation: resize to fixed dimensions for consistent comparison."""
    return cv2.resize(img, TARGET_SIZE)


def to_grayscale(img):
    """Convert BGR to grayscale — reduces 3-channel to single intensity channel."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_img, kernel_size=(5, 5)):
    """
    Spatial Filtering (UNIT 3): Gaussian blur for noise removal.
    Uses a weighted average kernel — pixels closer to center have higher weight.
    """
    return cv2.GaussianBlur(gray_img, kernel_size, 0)


def clahe_enhancement(gray_img, clip_limit=2.0, tile_size=(8, 8)):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization (UNIT 3).
    Better than global histogram equalization for mixed lighting conditions.
    Applies different contrast enhancement to different regions of the image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def preprocess(path):
    """
    Enhanced preprocessing pipeline:
    load → segment product → remove background → resize → grayscale → blur → CLAHE
    Returns both the preprocessed grayscale image and the processed color image.
    """
    # Load original image
    img = load_image(path)
    
    # Step 1: Segment product region (remove most background)
    product_img = segment_product(img)
    
    # Step 2: Further background removal using GrabCut
    clean_img = remove_background(product_img)
    
    # Step 3: Resize to standard size
    resized_img = resize_image(clean_img)
    
    # Step 4: Convert to grayscale
    gray = to_grayscale(resized_img)
    
    # Step 5: Apply Gaussian blur for noise reduction
    blurred = apply_gaussian_blur(gray)
    
    # Step 6: Apply CLAHE for adaptive contrast enhancement
    enhanced = clahe_enhancement(blurred)
    
    return enhanced, resized_img  # (processed grayscale, processed color for display)
