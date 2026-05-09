"""
main.py
Fake Return Product Detection using Image Processing
Entry point — processes all products and image pairs automatically.
"""

import os
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess
from features import get_features
from comparison import (
    compute_ssim, pixel_difference, compare_histograms,
    detect_damage, highlight_damage, decide
)

DATASET_DIR = "dataset"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def get_images(folder):
    """Return sorted list of image file paths from a folder."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ])


def display_result(orig_color, ret_color, diff_img, damage_overlay, result, ssim_score, title):
    """Display original, returned, difference, and damage-highlighted images."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{title}  |  Result: {result}  |  SSIM: {ssim_score:.4f}", fontsize=13)

    for ax, img, label in zip(
        axes,
        [orig_color, ret_color, diff_img, damage_overlay],
        ["Original", "Returned", "Difference", "Damage Highlight"]
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def compare_pair(orig_path, ret_path, show_plot=True):
    """
    Full pipeline for one original–return image pair:
    preprocess → feature extraction → comparison → damage detection → decision
    """
    # ── Preprocessing (UNIT 3) ──────────────────────────────────────────────
    orig_gray, orig_color = preprocess(orig_path)
    ret_gray,  ret_color  = preprocess(ret_path)

    # ── Feature Extraction (UNIT 4) ─────────────────────────────────────────
    orig_features = get_features(orig_gray)
    ret_features  = get_features(ret_gray)

    # ── Comparison ──────────────────────────────────────────────────────────
    ssim_score  = compute_ssim(orig_gray, ret_gray, orig_features['mask'], ret_features['mask'])
    diff_img    = pixel_difference(orig_gray, ret_gray, orig_features['mask'], ret_features['mask'])
    hist_score  = compare_histograms(orig_features['histogram'], ret_features['histogram'])
    texture_score = compare_texture(orig_features['texture'], ret_features['texture'])
    shape_score = compare_shape_features(orig_features['shape'], ret_features['shape'])

    # ── Damage Detection (morphological operations) ──────────────────────────
    damage_mask, damaged_pixels = detect_damage(diff_img, orig_features['mask'], ret_features['mask'])
    damage_overlay = highlight_damage(ret_color, damage_mask)

    # ── Enhanced Decision ────────────────────────────────────────────────────
    total_pixels = orig_gray.shape[0] * orig_gray.shape[1]
    result, composite_score = decide_enhanced(
        ssim_score, hist_score, texture_score, shape_score, damaged_pixels, total_pixels
    )

    # Convert diff to 3-channel for display
    diff_display = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)

    if show_plot:
        title = f"{os.path.basename(orig_path)}  vs  {os.path.basename(ret_path)}"
        display_result(orig_color, ret_color, diff_display, damage_overlay, result, ssim_score, title)

    return result, ssim_score, hist_score, damaged_pixels, composite_score, texture_score, shape_score


def process_all(show_plots=True):
    """
    Automatically iterate over all products and compare
    every original image against every return image.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset folder '{DATASET_DIR}' not found.")
        return

    products = sorted([
        p for p in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, p))
    ])

    if not products:
        print("[ERROR] No product folders found inside dataset/")
        return

    for product in products:
        # Handle mixed case folder names (Original/original, Return/return)
        product_path = os.path.join(DATASET_DIR, product)
        subfolders = {f.lower(): f for f in os.listdir(product_path)
                      if os.path.isdir(os.path.join(product_path, f))}

        if "original" not in subfolders or "return" not in subfolders:
            print(f"[SKIP] {product}: missing 'original' or 'return' folder.")
            continue

        orig_dir = os.path.join(product_path, subfolders["original"])
        ret_dir  = os.path.join(product_path, subfolders["return"])

        orig_images = get_images(orig_dir)
        ret_images  = get_images(ret_dir)

        if not orig_images or not ret_images:
            print(f"[SKIP] {product}: no images found.")
            continue

        print(f"\n{'='*60}")
        print(f"  Product: {product}")
        print(f"  Originals: {len(orig_images)}  |  Returns: {len(ret_images)}")
        print(f"{'='*60}")

        for orig_path in orig_images:
            for ret_path in ret_images:
                result, ssim, hist_corr, dmg_px, composite, texture, shape = compare_pair(
                    orig_path, ret_path, show_plot=show_plots
                )
                print(
                    f"  {os.path.basename(orig_path):20s} vs {os.path.basename(ret_path):20s}"
                    f"  →  {result:10s}  SSIM={ssim:.3f}  Hist={hist_corr:.3f}  Tex={texture:.3f}  Shape={shape:.3f}  Composite={composite:.3f}  DmgPx={dmg_px}"
                )


if __name__ == "__main__":
    process_all(show_plots=True)
