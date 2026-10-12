"""HailoRT backend for ASUS UGen300 (Hailo-10H). Stub until hardware is available.

Notes for the real implementation:
  - Input: feed the RGB uint8 HWC image as-is; normalization is compiled into the HEF.
  - YOLOv8-seg HEF outputs are raw per-scale tensors (DFL box 64ch, class 80ch,
    mask coeff 32ch for strides 8/16/32) plus the 32ch proto, not the decoded
    [1, 116, 8400] tensor the ONNX export produces. Add a decoder in postprocess.py
    that turns them into the same `HeadOutput` (DFL softmax + anchor/stride decode,
    class sigmoid); NMS and mask assembly are then shared with the ONNX path.
  - INT8 quantization changes CLIP vectors, so embeddings must be recomputed
    when switching to this backend (see `embedding_model` / `backend` columns).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import InferenceBackend, ModelSpec, TensorSpec


class HailoBackend(InferenceBackend):
    def __init__(self, device_id: str | None = None):
        self._device_id = device_id

    @property
    def name(self) -> str:
        return "hailo"

    def load_model(self, spec: ModelSpec) -> None:
        raise NotImplementedError("HailoBackend is a stub; no Hailo-10H device yet")

    def input_specs(self, model_id: str) -> List[TensorSpec]:
        raise NotImplementedError

    def output_specs(self, model_id: str) -> List[TensorSpec]:
        raise NotImplementedError

    def infer(self, model_id: str, image: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def infer_tensors(self, model_id: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Text encoders always run on the host CPU (OnnxBackend), never on the Hailo device.
        raise NotImplementedError("text encoders run on the host via OnnxBackend")
