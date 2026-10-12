"""Backend-agnostic inference contract.

Every model call in the app goes through `InferenceBackend`. A backend only runs the
network and returns raw output tensors; decoding, NMS and mask assembly happen on the
host in `postprocess.py`, because HailoRT also returns raw tensors only.

Input contract: callers pass an RGB uint8 image (H, W, 3) already letterboxed/resized to
the model input size. Each backend converts it to its native format:
  - ONNX: float32 NCHW, normalized with the model's mean/std.
  - Hailo: uint8 NHWC as-is (normalization is folded into the HEF at compile time).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .masks import RegionMask


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    path: Path
    input_size: Optional[Tuple[int, int]] = None  # (height, width); None for non-image models
    # Per-channel normalization applied to 0-255 RGB input: (x - mean) / std.
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (255.0, 255.0, 255.0)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[Optional[int], ...]  # None for dynamic dims
    dtype: str


@dataclass
class Detection:
    bbox: np.ndarray  # (4,) float32 xyxy, original image pixels
    class_id: Optional[int]  # None for class-agnostic models (FastSAM)
    class_name: Optional[str]
    score: float
    mask: RegionMask  # bool crop covering the bbox, original image resolution
    source_model: str

    @property
    def mask_area(self) -> int:
        return self.mask.area


class InferenceBackend(ABC):
    """Runs models and returns raw tensors. Implementations: OnnxBackend, HailoBackend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier stored alongside results, e.g. 'onnx-cpu', 'onnx-cuda', 'hailo'."""

    @abstractmethod
    def load_model(self, spec: ModelSpec) -> None: ...

    @abstractmethod
    def input_specs(self, model_id: str) -> List[TensorSpec]: ...

    @abstractmethod
    def output_specs(self, model_id: str) -> List[TensorSpec]: ...

    @abstractmethod
    def infer(self, model_id: str, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run one RGB uint8 (H, W, 3) image; return raw outputs keyed by tensor name."""

    @abstractmethod
    def infer_tensors(self, model_id: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run a model whose inputs are not images (e.g. text encoders: token ids + mask)."""

    def close(self) -> None:
        pass

    def __enter__(self) -> "InferenceBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
