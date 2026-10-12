"""Image file I/O that works with non-ASCII paths on Windows (cv2.imread/imwrite do not)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def decode(data: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("cannot decode image data")
    return image


def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def encode(image: np.ndarray, ext: str, params: Sequence[int] = ()) -> bytes:
    ok, buf = cv2.imencode(ext, image, list(params))
    if not ok:
        raise ValueError(f"cannot encode image as {ext}")
    return buf.tobytes()


def imwrite(path: Path, image: np.ndarray, jpeg_quality: int = 90) -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    Path(path).write_bytes(encode(image, path.suffix, params))
