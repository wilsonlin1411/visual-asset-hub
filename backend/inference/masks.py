"""Instance masks stored as a box-sized crop plus its offset in the original image.

Phone photos are ~12 MP and FastSAM yields up to 100 masks per photo, so full-frame
bool masks (12 MB each) are not an option. All mask math works on the crops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RegionMask:
    data: np.ndarray  # (h, w) bool
    x0: int
    y0: int

    @property
    def x1(self) -> int:
        return self.x0 + self.data.shape[1]

    @property
    def y1(self) -> int:
        return self.y0 + self.data.shape[0]

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def area(self) -> int:
        return int(self.data.sum())

    def to_full(self, shape: Tuple[int, int]) -> np.ndarray:
        full = np.zeros(shape[:2], dtype=bool)
        full[self.y0:self.y1, self.x0:self.x1] = self.data
        return full

    def tight(self) -> "RegionMask":
        """Shrink the crop to the bounding box of its set pixels."""
        ys = np.flatnonzero(self.data.any(axis=1))
        xs = np.flatnonzero(self.data.any(axis=0))
        if ys.size == 0:
            return RegionMask(np.zeros((0, 0), dtype=bool), self.x0, self.y0)
        return RegionMask(self.data[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1].copy(),
                          self.x0 + int(xs[0]), self.y0 + int(ys[0]))

    @classmethod
    def from_full(cls, full: np.ndarray) -> "RegionMask":
        return cls(full.astype(bool), 0, 0).tight()


def intersection(a: RegionMask, b: RegionMask) -> int:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if x0 >= x1 or y0 >= y1:
        return 0
    sa = a.data[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
    sb = b.data[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]
    return int(np.count_nonzero(sa & sb))


def mask_iou(a: RegionMask, b: RegionMask, area_a: int | None = None, area_b: int | None = None) -> float:
    """Exact IoU, computed only on the overlap of the two crops."""
    area_a = a.area if area_a is None else area_a
    area_b = b.area if area_b is None else area_b
    if area_a == 0 or area_b == 0:
        return 0.0
    inter = intersection(a, b)
    return inter / (area_a + area_b - inter)
