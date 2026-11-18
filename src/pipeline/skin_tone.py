"""Skin-tone estimation from aligned facial cheek pixels."""

import cv2
import numpy as np


def estimate_skin_tone(
    image_bgr: np.ndarray,
    cheek_mask: np.ndarray,
) -> dict:
    """
    Estimate skin tone using median CIE LAB values
    from the cheek region.

    Args:
        image_bgr: Aligned three-channel BGR image.
        cheek_mask: Single-channel mask. Nonzero pixels
                    indicate valid cheek skin.

    Returns:
        Dictionary containing LAB values, undertone,
        and representative shade hex.
    """

    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("image_bgr must be a NumPy array.")

    if not isinstance(cheek_mask, np.ndarray):
        raise ValueError("cheek_mask must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have three BGR channels.")

    if image_bgr.size == 0:
        raise ValueError("image_bgr is empty.")

    if cheek_mask.ndim == 3:
        if cheek_mask.shape[2] != 1:
            raise ValueError("cheek_mask must be single-channel.")
        cheek_mask = cheek_mask[:, :, 0]

    if cheek_mask.ndim != 2:
        raise ValueError("cheek_mask must be a 2D mask.")

    if cheek_mask.shape != image_bgr.shape[:2]:
        raise ValueError("cheek_mask dimensions must match image_bgr dimensions.")

    if not np.isfinite(image_bgr).all():
        raise ValueError("image_bgr contains invalid values.")

    # Convert image to OpenCV LAB.
    lab = cv2.cvtColor(
        image_bgr.astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    # Select cheek pixels.
    valid_pixels = lab[cheek_mask > 0]

    if len(valid_pixels) < 20:
        raise ValueError("Insufficient cheek pixels for reliable skin-tone estimation.")

    # Median LAB values.
    median_lab = np.median(valid_pixels, axis=0)

    L = float(median_lab[0] * (100.0 / 255.0))
    a = float(median_lab[1] - 128.0)
    b = float(median_lab[2] - 128.0)

    # Basic undertone heuristic.
    if b - a > 6.0:
        undertone = "warm"
    elif a - b > 2.0 or b < 10.0:
        undertone = "cool"
    else:
        undertone = "neutral"

    # Convert median LAB color to RGB.
    median_lab_uint8 = np.uint8([[median_lab]])

    median_rgb = cv2.cvtColor(
        median_lab_uint8,
        cv2.COLOR_LAB2RGB,
    )[0, 0]

    shade_hex = "#{:02x}{:02x}{:02x}".format(
        int(median_rgb[0]),
        int(median_rgb[1]),
        int(median_rgb[2]),
    )

    return {
        "L": round(L, 2),
        "a": round(a, 2),
        "b": round(b, 2),
        "undertone": undertone,
        "shade_hex": shade_hex,
    }
