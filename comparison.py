"""
comparison.py
Handles: Enhanced SSIM, pixel-wise difference, histogram comparison,
         damage detection using morphological operations, texture comparison,
         and improved decision logic.
"""

import cv2
import numpy as np


# ─── Enhanced SSIM (Structural Similarity Index) ────────────────────────────────

def compute_ssim(img1, img2, mask1=None, mask2=None):
    """
    Enhanced Structural Similarity Index Measure (SSIM).
    Compares luminance, contrast, and structure between two grayscale images.
    Now supports masks to focus on product regions only.
    Returns a score in [-1, 1]; closer to 1 means more similar.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    def conv(img):
        return cv2.filter2D(img, -1, window)

    mu1, mu2 = conv(img1), conv(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = conv(img1 ** 2) - mu1_sq
    sigma2_sq = conv(img2 ** 2) - mu2_sq
    sigma12   = conv(img1 * img2) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    
    # If masks are provided, compute SSIM only on masked regions
    if mask1 is not None and mask2 is not None:
        # Combine masks
        combined_mask = cv2.bitwise_and(mask1, mask2)
        if np.sum(combined_mask) > 0:
            masked_ssim = ssim_map[combined_mask > 0]
            return float(np.mean(masked_ssim))
    
    return float(np.mean(ssim_map))


# ─── Enhanced Pixel-wise Difference ─────────────────────────────────────────────

def pixel_difference(img1, img2, mask1=None, mask2=None):
    """
    Pixel-wise absolute difference strictly on product region.
    Background pixels (black = 0) are excluded from difference.
    Combined product mask ensures diff map shows product changes only.
    """
    diff = cv2.absdiff(img1, img2)

    # Build product-only mask from non-black pixels in both images
    # Since background = black (0), non-zero pixels = product
    auto_mask1 = (img1 > 0).astype(np.uint8) * 255
    auto_mask2 = (img2 > 0).astype(np.uint8) * 255
    auto_combined = cv2.bitwise_and(auto_mask1, auto_mask2)

    # If explicit masks provided, intersect with auto mask
    if mask1 is not None and mask2 is not None:
        explicit_combined = cv2.bitwise_and(mask1, mask2)
        final_mask = cv2.bitwise_and(auto_combined, explicit_combined)
    else:
        final_mask = auto_combined

    # Zero out all background pixels in diff map
    diff = cv2.bitwise_and(diff, diff, mask=final_mask)

    return diff


# ─── Enhanced Histogram Comparison ─────────────────────────────────────────────

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """
    Compare two normalized histograms using specified method.
    Default: correlation method returns score in [-1, 1]; 1 = identical distribution.
    Enhanced with multiple comparison methods.
    """
    return cv2.compareHist(
        hist1.reshape(-1, 1).astype(np.float32),
        hist2.reshape(-1, 1).astype(np.float32),
        method
    )


def compare_multiple_histograms(hist1, hist2):
    """
    Compare histograms using multiple methods for robust comparison.
    Returns a dictionary of different similarity scores.
    """
    methods = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi_square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
    }
    
    results = {}
    for name, method in methods.items():
        score = compare_histograms(hist1, hist2, method)
        results[name] = score
    
    return results


# ─── Texture Comparison ────────────────────────────────────────────────────────

def compare_texture(texture1, texture2):
    """
    Compare texture features between two images.
    Returns normalized correlation coefficient.
    """
    # Flatten texture images
    tex1_flat = texture1.flatten().astype(np.float32)
    tex2_flat = texture2.flatten().astype(np.float32)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(tex1_flat, tex2_flat)[0, 1]
    
    # Handle NaN case (when one texture is constant)
    if np.isnan(correlation):
        correlation = 0.0
    
    return correlation


# ─── Shape Comparison ──────────────────────────────────────────────────────────

def compare_shape_features(shape1, shape2):
    """
    Compare shape features between two products.
    Returns a similarity score based on area, perimeter, and aspect ratio.
    """
    # Handle case where no contours were found
    if shape1['area'] == 0 or shape2['area'] == 0:
        return 0.0
    
    # Compare areas (normalized)
    area_ratio = min(shape1['area'], shape2['area']) / max(shape1['area'], shape2['area'])
    
    # Compare perimeters (normalized)
    if shape1['perimeter'] == 0 or shape2['perimeter'] == 0:
        perimeter_ratio = 0.0
    else:
        perimeter_ratio = min(shape1['perimeter'], shape2['perimeter']) / max(shape1['perimeter'], shape2['perimeter'])
    
    # Compare aspect ratios
    aspect_diff = abs(shape1['aspect_ratio'] - shape2['aspect_ratio'])
    aspect_similarity = max(0, 1 - aspect_diff)  # Convert difference to similarity
    
    # Weighted combination
    shape_similarity = (area_ratio * 0.4 + perimeter_ratio * 0.3 + aspect_similarity * 0.3)
    
    return shape_similarity


# ─── Enhanced Damage Detection ─────────────────────────────────────────────────

def detect_damage(diff_img, mask1=None, mask2=None):
    """
    Damage detection strictly on product region:
    1. Build product mask from non-black pixels (background=black from preprocessing)
    2. Adaptive threshold on product pixels only
    3. Morphological dilation — expand damage regions
    4. Morphological erosion  — remove noise
    5. Strict mask: zero out any remaining background pixels
    Returns clean damage mask (product region only) + damaged pixel count.
    """
    # Product mask: non-black pixels = product (background was set to 0)
    auto_mask = (diff_img > 0).astype(np.uint8) * 255

    if mask1 is not None and mask2 is not None:
        explicit = cv2.bitwise_and(mask1, mask2)
        product_mask = cv2.bitwise_and(auto_mask, explicit)
    else:
        product_mask = auto_mask

    # Adaptive threshold only on product pixels
    product_pixels = diff_img[product_mask > 0]
    if len(product_pixels) == 0:
        return np.zeros_like(diff_img), 0

    mean_diff = float(np.mean(product_pixels))
    std_diff  = float(np.std(product_pixels))
    threshold_value = max(25, mean_diff + 1.5 * std_diff)

    _, thresh = cv2.threshold(diff_img, threshold_value, 255, cv2.THRESH_BINARY)

    # Keep only product region in threshold result
    thresh = cv2.bitwise_and(thresh, thresh, mask=product_mask)

    kernel = np.ones((5, 5), np.uint8)

    # Dilation — expand damage regions, fill small gaps
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Erosion — remove small noise blobs
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Final strict mask — ensure zero background contamination
    eroded = cv2.bitwise_and(eroded, eroded, mask=product_mask)

    damaged_pixels = int(np.sum(eroded > 0))
    return eroded, damaged_pixels


def highlight_damage(color_img, damage_mask):
    """
    Overlay damage mask on the segmented product image in red.
    Background pixels (black) are never highlighted — only product pixels.
    """
    result = color_img.copy()
    # Only highlight where damage AND product pixel (non-black)
    product_pixels = np.any(color_img > 0, axis=2)  # True where product exists
    damage_region  = (damage_mask > 0) & product_pixels
    result[damage_region] = [0, 0, 255]  # Red
    return result


# ─── Enhanced Decision Logic ───────────────────────────────────────────────────

def compute_composite_score(ssim_score, hist_corr, texture_corr, shape_sim, damaged_pixels, total_pixels):
    """
    Compute a composite similarity score using multiple features.
    Weights different aspects of similarity for robust decision making.
    """
    # Normalize damage ratio
    damage_ratio = damaged_pixels / max(total_pixels, 1)
    damage_score = max(0, 1 - damage_ratio * 10)  # Heavy penalty for damage
    
    # Weighted combination of all scores
    weights = {
        'ssim': 0.35,
        'histogram': 0.25,
        'texture': 0.15,
        'shape': 0.15,
        'damage': 0.10
    }
    
    composite = (
        weights['ssim'] * max(0, ssim_score) +
        weights['histogram'] * max(0, hist_corr) +
        weights['texture'] * max(0, texture_corr) +
        weights['shape'] * shape_sim +
        weights['damage'] * damage_score
    )
    
    return composite


def decide_enhanced(ssim_score, hist_corr, texture_corr, shape_sim, damaged_pixels, total_pixels):
    """
    Enhanced decision logic using multiple features and adaptive thresholds.
    """
    # Compute composite score
    composite_score = compute_composite_score(
        ssim_score, hist_corr, texture_corr, shape_sim, damaged_pixels, total_pixels
    )
    
    # Damage ratio for additional checks
    damage_ratio = damaged_pixels / max(total_pixels, 1)
    
    # Decision logic with multiple criteria
    if composite_score > 0.75 and damage_ratio < 0.05:
        return "SAME", composite_score
    elif composite_score > 0.45 and ssim_score > 0.40:
        if damage_ratio > 0.15:  # Significant damage detected
            return "DAMAGED", composite_score
        elif shape_sim < 0.6:  # Shape is very different
            return "DIFFERENT", composite_score
        else:
            return "DAMAGED", composite_score
    else:
        return "DIFFERENT", composite_score


def decide(ssim_score):
    """
    Simple decision logic (kept for backward compatibility).
    Based on SSIM similarity score:
      > 0.80  → SAME product
      0.50–0.80 → DAMAGED product
      < 0.50  → DIFFERENT product
    """
    if ssim_score > 0.80:
        return "SAME"
    elif ssim_score >= 0.50:
        return "DAMAGED"
    else:
        return "DIFFERENT"
