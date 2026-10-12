"""Turn a coarse binary instance mask into a soft alpha matte that follows image edges.

Seg masks come from a 160x160 prototype grid, so at 12 MP a mask edge can be off by
~25 px. GuidedFilterRefiner:
  1. works in a region around the mask (box + margin),
  2. smooths the binary mask with a small open/close,
  3. marks a band around the coarse edge as uncertain (deep inside = object, far outside =
     background) and lets GrabCut relabel the band by colour, which moves the edge onto
     the real one (a guided filter alone only smooths: it cannot pull a mask back out of
     a flat background),
  4. runs a colour guided filter on that hard mask for soft, edge-aware alpha,
  5. feathers slightly and clamps near-0 / near-1 alpha.
A matting model can replace it later by implementing `Refiner`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..config import RefineConfig
from ..inference import RegionMask


@dataclass
class AlphaMatte:
    alpha: np.ndarray  # (h, w) float32 in [0, 1]
    x0: int
    y0: int

    @property
    def box(self) -> Tuple[int, int, int, int]:
        h, w = self.alpha.shape
        return self.x0, self.y0, self.x0 + w, self.y0 + h


class Refiner(ABC):
    @abstractmethod
    def refine(self, image_bgr: np.ndarray, mask: RegionMask) -> AlphaMatte: ...


def edge_band_px(image_shape: Tuple[int, int], cfg: RefineConfig) -> int:
    """Width of the uncertain zone around a coarse mask edge, scaled to the photo size."""
    return max(cfg.min_edge_band_px, round(cfg.edge_band_ratio * max(image_shape[:2])))


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def grabcut_band(roi_bgr: np.ndarray, coarse: np.ndarray, sure_fg: np.ndarray, maybe_fg: np.ndarray,
                 iters: int, max_side: int) -> np.ndarray:
    """Relabel the uncertain band (maybe_fg & ~sure_fg) with GrabCut. Returns a bool mask."""
    gc = np.full(coarse.shape, cv2.GC_BGD, np.uint8)
    gc[maybe_fg & ~coarse] = cv2.GC_PR_BGD
    gc[maybe_fg & coarse] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD

    scale = min(1.0, max_side / max(coarse.shape))
    img, labels = roi_bgr, gc
    if scale < 1.0:
        size = (max(1, round(coarse.shape[1] * scale)), max(1, round(coarse.shape[0] * scale)))
        img = cv2.resize(roi_bgr, size, interpolation=cv2.INTER_AREA)
        labels = cv2.resize(gc, size, interpolation=cv2.INTER_NEAREST)
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, labels, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse  # e.g. no background samples in the region: keep the model mask
    fg = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.float32)
    if scale < 1.0:
        fg = cv2.resize(fg, coarse.shape[::-1], interpolation=cv2.INTER_LINEAR)
    # Certain regions stay certain at full resolution.
    return ((fg > 0.5) | sure_fg) & maybe_fg


class GuidedFilterRefiner(Refiner):
    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg

    def refine(self, image_bgr: np.ndarray, mask: RegionMask) -> AlphaMatte:
        cfg = self.cfg
        h, w = image_bgr.shape[:2]
        band = edge_band_px((h, w), cfg)
        mh, mw = mask.data.shape
        margin = max(2 * band, round(cfg.margin_ratio * max(mh, mw)))
        rx0, ry0 = max(mask.x0 - margin, 0), max(mask.y0 - margin, 0)
        rx1, ry1 = min(mask.x1 + margin, w), min(mask.y1 + margin, h)
        roi = image_bgr[ry0:ry1, rx0:rx1]

        m = np.zeros(roi.shape[:2], dtype=np.uint8)
        m[mask.y0 - ry0:mask.y1 - ry0, mask.x0 - rx0:mask.x1 - rx0] = mask.data

        scale = min(1.0, cfg.max_work_side / max(roi.shape[:2]))
        if scale < 1.0:
            size = (max(1, round(roi.shape[1] * scale)), max(1, round(roi.shape[0] * scale)))
            work_roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
            work_m = (cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
            alpha = self._alpha(work_roi, work_m, max(cfg.min_edge_band_px, round(band * scale)))
            alpha = cv2.resize(alpha, roi.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        else:
            alpha = self._alpha(roi, m, band)

        if cfg.feather_px > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), cfg.feather_px)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha[alpha < cfg.alpha_floor] = 0.0
        alpha[alpha > cfg.alpha_ceil] = 1.0
        return AlphaMatte(alpha, rx0, ry0)

    def _alpha(self, roi: np.ndarray, m: np.ndarray, band: int) -> np.ndarray:
        """Hard-edge correction (GrabCut in the band) + soft alpha (guided filter) on one region."""
        cfg = self.cfg
        smooth = _ellipse(max(1, band // 4))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, smooth)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, smooth)

        # Distance transforms instead of erode/dilate with a (2*band+1)^2 kernel: O(N) for any band.
        sure_fg = cv2.distanceTransform(m, cv2.DIST_L2, 3) > band
        maybe_fg = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3) <= band

        hard = m.astype(bool)
        if cfg.grabcut_iters > 0:
            hard = grabcut_band(roi, hard, sure_fg, maybe_fg, cfg.grabcut_iters, cfg.grabcut_max_side)

        # uint8 guide with eps scaled to 0-255: a float [0, 1] guide with tiny eps makes the 3x3
        # colour covariance near-singular (correlated channels) and the output degrades to a blur.
        radius = max(2, round(band * cfg.guided_radius_ratio))
        alpha = cv2.ximgproc.guidedFilter(roi, hard.astype(np.float32), radius, cfg.guided_eps * 255 ** 2)
        return np.where(sure_fg, 1.0, np.where(maybe_fg, alpha, 0.0)).astype(np.float32)
