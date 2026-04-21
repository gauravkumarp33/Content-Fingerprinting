from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_media(path: str) -> list[np.ndarray]:
    suffix = Path(path).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Unable to read image.")
        return [image]

    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported media type.")

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("Unable to read video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    step = max(int(round(fps)), 1)
    frames: list[np.ndarray] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            frames.append(frame)
        frame_index += 1

    capture.release()

    if not frames:
        raise ValueError("No frames extracted from video.")

    return frames
