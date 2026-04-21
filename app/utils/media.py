import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import UploadFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def save_upload_to_temp(upload: UploadFile) -> Tuple[str, str]:
    suffix = Path(upload.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = upload.file.read()
        temp_file.write(content)
        return temp_file.name, suffix


def extract_frames(path: str, suffix: str) -> List[np.ndarray]:
    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Unable to read uploaded image.")
        return [image]

    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported media type.")

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("Unable to read uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    step = max(int(round(fps)), 1)
    frames: List[np.ndarray] = []
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
        raise ValueError("No frames extracted from uploaded video.")

    return frames
