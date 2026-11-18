"""Pipeline composition - orchestrates the full skin scan analysis."""

import numpy as np
from typing import Dict
import cv2
from .preprocess import preprocess
from .face_mesh import FaceMeshDetector, make_region_masks
from .maps.redness import redness_map
from .maps.oiliness import oiliness_map
from .maps.texture import texture_map
from .maps.pores import pores_map
from .maps.blemishes import blemish_map
from .maps.hydration import hydration_map
from .maps.pigment import pigment_map
from .visualize import generate_all_overlays
from ..app.utils_io import encode_png_base64


def score_from_map(map_data: np.ndarray, masks: dict[str, np.ndarray]) -> float:
    """
    Compute single score from map using region-weighted average.

    Args:
        map_data: Normalized map [0, 1]
        masks: Region masks

    Returns:
        Score [0, 1]
    """
    # Create combined face mask
    face_mask = np.zeros(map_data.shape, dtype=bool)
    for region_mask in masks.values():
        face_mask |= region_mask > 0

    if not face_mask.any():
        return 0.0

    # Compute weighted average
    face_vals = map_data[face_mask]
    return float(np.mean(face_vals))


def run_scan(img: np.ndarray) -> Dict:
    """
    Run complete skin scan pipeline.

    Returns:
        scores: Skin analysis scores.
        overlays: Base64 PNG overlays.
        regions: Detected facial regions.
        aligned_bgr: Preprocessed BGR image.
        cheek_mask: Binary mask containing cheek regions.
    """

    # 1. Preprocess image
    img_processed = preprocess(img, max_size=1024)

    if img_processed is None or img_processed.size == 0:
        raise ValueError("Image preprocessing failed.")

    # 2. Detect face landmarks
    detector = FaceMeshDetector()
    landmarks = detector.detect(img_processed)

    if landmarks is None:
        raise ValueError("No face detected in image.")

    # 3. Create facial region masks
    masks = make_region_masks(
        landmarks,
        img_processed.shape,
    )

    if not masks:
        raise ValueError("Could not create region masks from landmarks.")

    # 4. Extract cheek masks
    cheek_mask = np.zeros(
        img_processed.shape[:2],
        dtype=np.uint8,
    )

    # Support common cheek-region naming conventions.
    cheek_names = {
        "left_cheek",
        "right_cheek",
        "cheek_left",
        "cheek_right",
        "left_cheekbone",
        "right_cheekbone",
    }

    for name, region_mask in masks.items():
        if name.lower() in cheek_names:
            if region_mask is None:
                continue

            region_mask = np.asarray(region_mask)

            if region_mask.shape[:2] != cheek_mask.shape:
                raise ValueError(f"Region mask '{name}' has incorrect dimensions.")

            cheek_mask = cv2.bitwise_or(
                cheek_mask,
                (region_mask > 0).astype(np.uint8) * 255,
            )

    if not np.any(cheek_mask):
        raise ValueError(
            "No cheek regions found. Check make_region_masks() "
            "for left_cheek and right_cheek masks."
        )

    # 5. Run skin analysis maps
    maps = {
        "redness": redness_map(img_processed, masks),
        "oiliness": oiliness_map(img_processed, masks),
        "texture": texture_map(img_processed, masks),
        "pores": pores_map(img_processed, masks),
        "blemishes": blemish_map(img_processed, masks),
        "hydration": hydration_map(img_processed, masks),
        "pigment": pigment_map(img_processed, masks),
    }

    # 6. Calculate scores
    scores = {name: score_from_map(map_data, masks) for name, map_data in maps.items()}

    # 7. Generate heatmap overlays
    overlay_images = generate_all_overlays(
        maps,
        alpha=0.6,
    )

    overlays = {
        name: encode_png_base64(overlay_rgba)
        for name, overlay_rgba in overlay_images.items()
    }

    # 8. Region names
    regions = list(masks.keys())

    # 9. Return public and internal pipeline outputs
    return {
        "scores": scores,
        "overlays": overlays,
        "regions": regions,
        # Internal skin-tone processing data
        "aligned_bgr": img_processed,
        "cheek_mask": cheek_mask,
    }
