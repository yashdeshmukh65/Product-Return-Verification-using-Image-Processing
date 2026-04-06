"""
preprocessing.py
UNIT 3 - Image Enhancement & Spatial Filtering
Handles: resizing, grayscale conversion, Gaussian blur, histogram equalization
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


def histogram_equalization(gray_img):
    """
    Image Enhancement (UNIT 3): Histogram equalization.
    Redistributes pixel intensities to improve contrast.
    """
    return cv2.equalizeHist(gray_img)


def preprocess(path):
    """
    Full preprocessing pipeline:
    load → resize → grayscale → Gaussian blur → histogram equalization
    Returns both the preprocessed grayscale image and the resized color image.
    """
    img = load_image(path)
    img = resize_image(img)
    gray = to_grayscale(img)
    blurred = apply_gaussian_blur(gray)
    enhanced = histogram_equalization(blurred)
    return enhanced, img  # (processed grayscale, original color for display)
