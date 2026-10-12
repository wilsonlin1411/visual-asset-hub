"""Image I/O and debug visualizations shared by the CLIs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..imageio import imread, imwrite  # noqa: F401  (re-exported for the CLIs)
from ..inference import RegionMask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(item)
    return paths


def fit(image: np.ndarray, max_side: int) -> np.ndarray:
    scale = max_side / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def color_for(i: int) -> Tuple[int, int, int]:
    hue = int((i * 0.618034 % 1.0) * 180)  # golden-ratio hue spacing keeps any index pair apart
    hsv = np.uint8([[[hue, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_overlay(image: np.ndarray, items: Sequence[Tuple[RegionMask, Sequence[float], str]],
                 alpha: float = 0.45) -> np.ndarray:
    """Draw translucent masks, outlines, boxes and labels. items = (mask, bbox_xyxy, label)."""
    out = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 600))
    font_scale = max(0.5, max(image.shape[:2]) / 1500)
    # Largest first so small objects stay visible on top.
    order = sorted(range(len(items)), key=lambda i: -items[i][0].area)
    for i in order:
        mask = items[i][0]
        if mask.data.size == 0:
            continue
        color = color_for(i)
        view = out[mask.y0:mask.y1, mask.x0:mask.x1]
        view[mask.data] = (view[mask.data].astype(np.float32) * (1 - alpha)
                           + np.float32(color) * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask.data.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE, offset=(mask.x0, mask.y0))
        cv2.drawContours(out, contours, -1, color, thickness)

    for i, (_, bbox, label) in enumerate(items):
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        color = color_for(i)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, thickness // 2))
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(y1, th + base)
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty), color, -1)
        cv2.putText(out, label, (x1, ty - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), max(1, thickness // 2), cv2.LINE_AA)
    return out


def checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    tile = ((yy // cell + xx // cell) % 2).astype(np.uint8)
    board = np.where(tile[..., None] == 1, np.uint8(205), np.uint8(255))
    return np.repeat(board, 3, axis=2)


def composite(bgra: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = bgra[..., 3:4].astype(np.float32) / 255
    return (bgra[..., :3].astype(np.float32) * a + background.astype(np.float32) * (1 - a)).astype(np.uint8)


def contact_sheet(cutouts: Sequence[np.ndarray], labels: Sequence[str], cell: int = 320,
                  columns: int = 5) -> np.ndarray:
    """Grid of cut-outs on a checkerboard, to eyeball edge quality."""
    if not cutouts:
        return np.full((cell, cell, 3), 255, np.uint8)
    rows = (len(cutouts) + columns - 1) // columns
    label_h = 28
    sheet = np.full((rows * (cell + label_h), min(columns, len(cutouts)) * cell, 3), 255, np.uint8)
    for k, (rgba, label) in enumerate(zip(cutouts, labels)):
        scale = min((cell - 8) / rgba.shape[1], (cell - 8) / rgba.shape[0], 1.0)
        small = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgba
        tile = checkerboard(cell, cell)
        oy, ox = (cell - small.shape[0]) // 2, (cell - small.shape[1]) // 2
        region = tile[oy:oy + small.shape[0], ox:ox + small.shape[1]]
        tile[oy:oy + small.shape[0], ox:ox + small.shape[1]] = composite(small, region)
        r, c = divmod(k, columns)
        y, x = r * (cell + label_h), c * cell
        sheet[y:y + cell, x:x + cell] = tile
        cv2.putText(sheet, label, (x + 6, y + cell + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return sheet
