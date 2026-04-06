"""
features.py
UNIT 4 - Feature Extraction
Handles: edge detection (Canny), contour detection, histogram extraction
"""

import cv2
import numpy as np


def extract_edges(gray_img):
    """
    Feature Extraction (UNIT 4): Canny edge detection.
    Detects boundaries by finding rapid intensity changes using gradient computation.
    """
    return cv2.Canny(gray_img, threshold1=50, threshold2=150)


def extract_contours(edge_img):
    """
    Feature Extraction (UNIT 4): Contour detection.
    Finds connected curves of edge pixels — represents object boundaries.
    """
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def extract_histogram(gray_img, bins=256):
    """
    Feature Extraction (UNIT 4): Intensity histogram.
    Counts pixel frequency at each intensity level (0–255).
    Normalized so images of different sizes can be compared.
    """
    hist = cv2.calcHist([gray_img], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def get_features(gray_img):
    """Returns edges, contours, and histogram for a preprocessed grayscale image."""
    edges = extract_edges(gray_img)
    contours = extract_contours(edges)
    histogram = extract_histogram(gray_img)
    return edges, contours, histogram
