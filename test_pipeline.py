"""
test_pipeline.py
Test the enhanced pipeline: Input → Segmentation → ROI Extraction → CLAHE → SSIM
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess, preprocess_pair
from features import get_features
from comparison import compute_ssim, decide_enhanced

def test_pipeline():
    """Test the enhanced pipeline on sample images"""
    
    # Test with first two images from dataset
    try:
        orig_path = "dataset/Product 1/Original/IMG20260406231602.jpg.jpeg"
        ret_path = "dataset/Product 1/return/IMG20260406231957.jpg.jpeg"
        
        print("Testing Enhanced Pipeline")
        print("=" * 50)
        
        # Step 1: Input Images → Segmentation → ROI Extraction → CLAHE
        print("Step 1: Processing Original Image and Return Image Pair...")
        (orig_gray, orig_color), (ret_gray, ret_color) = preprocess_pair(orig_path, ret_path)
        print(f"  Original processed: {orig_gray.shape}")
        print(f"  Return processed: {ret_gray.shape}")
        
        # Step 3: Feature Extraction
        print("Step 3: Feature Extraction...")
        orig_features = get_features(orig_gray)
        ret_features = get_features(ret_gray)
        print(f"  Features extracted")
        
        # Step 4: SSIM Comparison
        print("Step 4: SSIM Comparison...")
        ssim_score = compute_ssim(orig_gray, ret_gray, 
                                 orig_features['mask'], ret_features['mask'])
        print(f"  SSIM Score: {ssim_score:.4f}")
        
        # Step 5: Enhanced Decision
        print("Step 5: Enhanced Decision...")
        from comparison import compare_histograms, compare_texture, compare_shape_features, detect_damage
        
        hist_score = compare_histograms(orig_features['histogram'], ret_features['histogram'])
        texture_score = compare_texture(orig_features['texture'], ret_features['texture'])
        shape_score = compare_shape_features(orig_features['shape'], ret_features['shape'])
        
        diff_img = cv2.absdiff(orig_gray, ret_gray)
        damage_mask, damaged_pixels = detect_damage(diff_img, orig_features['mask'], ret_features['mask'])
        
        total_pixels = orig_gray.shape[0] * orig_gray.shape[1]
        result, composite_score = decide_enhanced(
            ssim_score, hist_score, texture_score, shape_score, damaged_pixels, total_pixels
        )
        
        print("=" * 50)
        print("RESULTS:")
        print(f"  SSIM Score:        {ssim_score:.4f}")
        print(f"  Histogram Score:   {hist_score:.4f}")
        print(f"  Texture Score:     {texture_score:.4f}")
        print(f"  Shape Score:       {shape_score:.4f}")
        print(f"  Composite Score:   {composite_score:.4f}")
        print(f"  Damaged Pixels:    {damaged_pixels}")
        print(f"  Final Decision:    {result}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print("Error:", e)
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("Enhanced pipeline test completed successfully!")
    else:
        print("Pipeline test failed!")