"""Display-size derivatives so the UI never loads 12 MP photos or multi-MB PNGs.

All paths are derived from ids (no DB columns); files are created on demand and cached.
    thumbs/assets/{asset_id}.webp                 asset cut-out, longer side 512, keeps alpha
    thumbs/medium/{asset_id}.png                  asset cut-out, longer side 2048 (canvas editor)
    thumbs/photos/{photo_id}.jpg                  photo, longer side 1600
    thumbs/candidates/{photo_id}/{idx}.webp       candidate preview (mask applied), longer side 256
    thumbs/candidates/{photo_id}/{idx}.ov.png     candidate mask at photo-display scale (alpha = mask)
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from ..imageio import encode, imread

ASSET_SIDE = 512
MEDIUM_SIDE = 2048  # canvas editing / export: covers the widest preset (Banner 1640 px)
PHOTO_SIDE = 1600
CANDIDATE_SIDE = 256


def fit_scale(shape: Tuple[int, int], max_side: int) -> float:
    return min(1.0, max_side / max(shape[:2]))


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


class Thumbs:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "thumbs"

    # -- paths
    def asset_path(self, asset_id: int) -> Path:
        return self.root / "assets" / f"{asset_id}.webp"

    def medium_path(self, asset_id: int) -> Path:
        return self.root / "medium" / f"{asset_id}.png"

    def photo_path(self, photo_id: int) -> Path:
        return self.root / "photos" / f"{photo_id}.jpg"

    def candidate_path(self, photo_id: int, idx: int) -> Path:
        return self.root / "candidates" / str(photo_id) / f"{idx}.webp"

    def overlay_path(self, photo_id: int, idx: int) -> Path:
        return self.root / "candidates" / str(photo_id) / f"{idx}.ov.png"

    @staticmethod
    def _save(path: Path, data: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: concurrent readers never see a half-written file
        return path

    # -- builders (return the cached file, creating it when missing)
    def asset(self, asset_id: int, png_file: Path, rgba: np.ndarray = None) -> Path:
        path = self.asset_path(asset_id)
        if not path.is_file():
            img = rgba if rgba is not None else imread(png_file, cv2.IMREAD_UNCHANGED)
            small = resize(img, fit_scale(img.shape, ASSET_SIDE))
            self._save(path, encode(small, ".webp", [cv2.IMWRITE_WEBP_QUALITY, 85]))
        return path

    def medium(self, asset_id: int, png_file: Path) -> Path:
        """Transparent PNG with the longer side capped at 2048 (the original when smaller)."""
        path = self.medium_path(asset_id)
        if not path.is_file():
            img = imread(png_file, cv2.IMREAD_UNCHANGED)
            s = fit_scale(img.shape, MEDIUM_SIDE)
            if s >= 1.0:
                return png_file
            self._save(path, encode(resize(img, s), ".png"))
        return path

    def photo(self, photo_id: int, photo_file: Path) -> Path:
        path = self.photo_path(photo_id)
        if not path.is_file():
            img = imread(photo_file)
            small = resize(img, fit_scale(img.shape, PHOTO_SIDE))
            self._save(path, encode(small, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]))
        return path

    def candidate(self, photo_id: int, idx: int, photo_file: Path, mask) -> Path:
        """Unrefined preview: the candidate's box with off-mask pixels transparent."""
        path = self.candidate_path(photo_id, idx)
        if not path.is_file():
            img = imread(photo_file)
            crop = img[mask.y0:mask.y1, mask.x0:mask.x1]
            bgra = np.dstack([crop, mask.data.astype(np.uint8) * 255])
            small = resize(bgra, fit_scale(bgra.shape, CANDIDATE_SIDE))
            self._save(path, encode(small, ".webp", [cv2.IMWRITE_WEBP_QUALITY, 80]))
        return path

    def overlay(self, photo_id: int, idx: int, photo_shape: Tuple[int, int], mask) -> Path:
        """Mask as a white PNG with alpha, scaled like the display photo; the UI tints it with CSS."""
        path = self.overlay_path(photo_id, idx)
        if not path.is_file():
            s = fit_scale(photo_shape, PHOTO_SIDE)
            alpha = resize(mask.data.astype(np.uint8) * 255, s) if s < 1 else mask.data.astype(np.uint8) * 255
            rgba = np.dstack([np.full(alpha.shape, 255, np.uint8)] * 3 + [alpha])
            self._save(path, encode(rgba, ".png"))
        return path

    # -- invalidation
    def drop_asset(self, asset_id: int) -> None:
        self.asset_path(asset_id).unlink(missing_ok=True)
        self.medium_path(asset_id).unlink(missing_ok=True)

    def drop_photo(self, photo_id: int) -> None:
        self.photo_path(photo_id).unlink(missing_ok=True)
        shutil.rmtree(self.root / "candidates" / str(photo_id), ignore_errors=True)
