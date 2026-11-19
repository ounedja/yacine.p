"""Resize and normalize images for model input."""

from __future__ import annotations

import importlib
from typing import Any

try:
    np = importlib.import_module("numpy")
except ImportError as exc:
    raise ImportError(
        "NumPy is required; install it with `python -m pip install numpy`."
    ) from exc

try:
    cv2 = importlib.import_module("cv2")
except ImportError:
    cv2 = None


def preprocess_image(
    image: Any,
    target_size: tuple = (224, 224),
) -> Any:
    """
    Resize an image, scale its pixels to [0, 1], and add a batch dimension.

    The returned array has shape (1, height, width, 3) and dtype float32.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have 1 or 3 color channels")

    height, width = target_size

    if cv2 is not None:
        resized = cv2.resize(image, (width, height))
    else:
        # NumPy nearest-neighbor resize keeps preprocessing usable without cv2.
        y_indices = np.linspace(0, image.shape[0] - 1, height).astype(int)
        x_indices = np.linspace(0, image.shape[1] - 1, width).astype(int)
        resized = image[y_indices][:, x_indices]

    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)