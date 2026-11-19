"""Capture an image with OpenCV, or create a mock image for testing."""

import warnings
from typing import Optional

import numpy as np  # type: ignore[import-not-found]
try:
    import cv2  # type: ignore[import-not-found]
except ImportError:
    cv2 = None


def create_mock_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a simple BGR image resembling a leaf, without camera hardware."""
    image = np.full((height, width, 3), (190, 220, 235), dtype=np.uint8)

    y, x = np.mgrid[0:height, 0:width]
    center_x, center_y = width // 2, height // 2
    leaf_mask = (
        ((x - center_x) / (width * 0.32)) ** 2
        + ((y - center_y) / (height * 0.19)) ** 2
        <= 1
    )

    # OpenCV images use BGR channel order.
    image[leaf_mask] = (45, 135, 55)

    # Add a simple central vein.
    for offset in range(-2, 3):
        vein_x = center_x + offset
        image[
            max(0, center_y - height // 6): min(height, center_y + height // 6),
            max(0, vein_x): min(width, vein_x + 1),
        ] = (100, 175, 105)

    return image


def capture_image(
    camera_index: int = 0,
    allow_mock: bool = True,
) -> np.ndarray:
    """
    Capture one BGR image from a camera.

    If OpenCV or camera hardware is unavailable, return a mock image when
    allow_mock is True; otherwise raise an error.
    """
    if cv2 is None:
        if allow_mock:
            warnings.warn("OpenCV is unavailable; using a mock image.")
            return create_mock_image()
        raise RuntimeError("OpenCV is not installed.")

    camera: Optional[object] = None
    try:
        camera = cv2.VideoCapture(camera_index)
        if camera.isOpened():
            success, image = camera.read()
            if success and image is not None:
                return image
    except Exception as error:
        warnings.warn(f"Camera capture failed: {error}")
    finally:
        if camera is not None:
            camera.release()

    if allow_mock:
        warnings.warn("Camera unavailable; using a mock image.")
        return create_mock_image()

    raise RuntimeError("Could not capture an image from the camera.")