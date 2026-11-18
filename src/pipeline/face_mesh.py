"""Face mesh detection using MediaPipe and region mask generation."""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional


class FaceMeshDetector:
    """MediaPipe face mesh detector."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks.

        Args:
            img_bgr: Input BGR image

        Returns:
            Landmarks as (N, 2) array of (x, y) pixel coordinates, or None
        """
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        # Get first face
        face_landmarks = results.multi_face_landmarks[0]

        # Convert normalized coordinates to pixels
        h, w = img_bgr.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])

        return np.array(landmarks, dtype=np.int32)

    def __del__(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()


def make_region_masks(
    landmarks: np.ndarray,
    img_shape: tuple,
) -> dict[str, np.ndarray]:
    """
    Generate facial region masks using MediaPipe Face Mesh landmarks.

    Returns:
        forehead
        left_cheek
        right_cheek
        cheeks
        nose
        chin
    """

    h, w = img_shape[:2]
    masks = {}

    if landmarks is None or len(landmarks) < 468:
        raise ValueError("Invalid or incomplete face landmarks.")

    landmarks = np.asarray(landmarks, dtype=np.int32)

    # Keep landmark coordinates inside image boundaries.
    points = landmarks.copy()
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)

    def create_mask(indices):
        mask = np.zeros((h, w), dtype=np.uint8)

        valid_indices = [i for i in indices if 0 <= i < len(points)]

        if len(valid_indices) < 3:
            return mask

        region_points = points[valid_indices]

        hull = cv2.convexHull(region_points)

        cv2.fillConvexPoly(mask, hull, 255)

        return mask

    # Forehead approximation.
    forehead_indices = [
        10,
        109,
        67,
        103,
        54,
        21,
        162,
        127,
        234,
        93,
        132,
        58,
        172,
        136,
        150,
        149,
        176,
        148,
        152,
        377,
        400,
        378,
        379,
        365,
        397,
        288,
        361,
        323,
        454,
        356,
        389,
        251,
        284,
        332,
        297,
        338,
    ]

    # Image-left cheek landmarks.
    left_cheek_indices = [
        116,
        117,
        118,
        119,
        100,
        126,
        142,
        36,
        205,
        206,
        207,
        213,
        192,
        147,
        123,
    ]

    # Image-right cheek landmarks.
    right_cheek_indices = [
        345,
        346,
        347,
        348,
        329,
        355,
        371,
        266,
        425,
        426,
        427,
        436,
        416,
        376,
        352,
    ]

    # Nose.
    nose_indices = [
        168,
        6,
        197,
        195,
        5,
        4,
        1,
        19,
        94,
        2,
        164,
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        200,
        199,
        175,
    ]

    # Chin.
    chin_indices = [
        152,
        377,
        400,
        378,
        379,
        365,
        397,
        288,
        361,
        323,
        454,
        356,
        389,
        251,
        284,
        332,
        297,
        338,
        10,
        109,
        67,
        103,
        54,
        21,
        162,
        127,
        234,
        93,
        132,
        58,
        172,
        136,
        150,
        176,
    ]

    # Generate individual regions.
    masks["forehead"] = create_mask(forehead_indices)
    masks["left_cheek"] = create_mask(left_cheek_indices)
    masks["right_cheek"] = create_mask(right_cheek_indices)
    masks["nose"] = create_mask(nose_indices)
    masks["chin"] = create_mask(chin_indices)

    # Combine cheeks without deleting the individual masks.
    masks["cheeks"] = cv2.bitwise_or(
        masks["left_cheek"],
        masks["right_cheek"],
    )

    # Validate cheek regions.
    if not np.any(masks["left_cheek"]):
        raise ValueError("Left cheek mask is empty.")

    if not np.any(masks["right_cheek"]):
        raise ValueError("Right cheek mask is empty.")

    return masks
