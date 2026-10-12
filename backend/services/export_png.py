"""Compose a transparent cut-out (BGRA) from the photo and a refined alpha matte."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .refine import AlphaMatte

OPAQUE = 0.95


@dataclass
class Cutout:
    rgba: np.ndarray                 # (h, w, 4) uint8, BGRA, straight (non-premultiplied) alpha
    bbox: Tuple[int, int, int, int]  # tight object box in the original image, xyxy
    origin: Tuple[int, int]          # PNG top-left in original coords (negative when padding sticks out)


def estimate_foreground(roi_bgr: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Replace colours of semi-transparent edge pixels with nearby solid-foreground colour.

    Edge pixels in the photo are a mix of object and background; keeping them as-is leaves
    a halo of the old background when the cut-out is placed on a new one. The estimate is
    an alpha-weighted Gaussian blur over confidently opaque pixels. The blur result is
    low-frequency, so it runs at 1/(sigma/4) resolution and is sampled only at edge pixels.
    """
    edge = (alpha > 0) & (alpha < OPAQUE)
    ys, xs = np.nonzero(edge)
    if ys.size == 0:
        return roi_bgr
    solid = alpha >= OPAQUE
    weighted = np.dstack([np.where(solid[..., None], roi_bgr, 0), solid.astype(np.uint8) * 255])

    h, w = alpha.shape
    factor = max(1, int(sigma // 4))
    sw, sh = max(1, w // factor), max(1, h // factor)
    small = weighted if factor == 1 else cv2.resize(weighted, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), sigma / factor)

    # Bilinear sample of the low-res blur at each edge pixel (same mapping as cv2.resize).
    # remap needs maps under 32767 rows/cols, so the point list is laid out as a 2-D grid.
    n = ys.size
    cols = 4096
    rows = -(-n // cols)
    map_x = np.zeros(rows * cols, np.float32)
    map_y = np.zeros(rows * cols, np.float32)
    map_x[:n] = (xs + 0.5) * (sw / w) - 0.5
    map_y[:n] = (ys + 0.5) * (sh / h) - 0.5
    vals = cv2.remap(small, map_x.reshape(rows, cols), map_y.reshape(rows, cols), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).reshape(-1, 4)[:n]
    weight = vals[:, 3] / 255.0
    ok = weight > 1e-3
    out = roi_bgr.copy()
    out[ys[ok], xs[ok]] = np.clip(vals[ok, :3] / weight[ok, None] + 0.5, 0, 255).astype(np.uint8)
    return out


def make_cutout(image_bgr: np.ndarray, matte: AlphaMatte, padding: int = 8,
                decontaminate_sigma: float = 4.0) -> Cutout:
    rx0, ry0, _, _ = matte.box
    ys = np.flatnonzero((matte.alpha > 0).any(axis=1))
    xs = np.flatnonzero((matte.alpha > 0).any(axis=0))
    if ys.size == 0:
        raise ValueError("matte is empty")
    ty0, ty1, tx0, tx1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1

    # Every opaque pixel lies inside the tight box, so the colour estimate needs nothing outside it.
    alpha = matte.alpha[ty0:ty1, tx0:tx1]
    roi = image_bgr[ry0 + ty0:ry0 + ty1, rx0 + tx0:rx0 + tx1]
    color = estimate_foreground(roi, alpha, decontaminate_sigma) if decontaminate_sigma > 0 else roi

    a8 = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([color, a8])
    rgba[a8 == 0, :3] = 0  # fully transparent pixels carry no colour (smaller PNG, no stray data)
    if padding:
        rgba = cv2.copyMakeBorder(rgba, padding, padding, padding, padding,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    bbox = (int(rx0 + tx0), int(ry0 + ty0), int(rx0 + tx1), int(ry0 + ty1))
    return Cutout(rgba, bbox, (bbox[0] - padding, bbox[1] - padding))
