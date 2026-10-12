"""ONNX Runtime backend (demo). CUDA when usable, otherwise CPU."""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from .base import InferenceBackend, ModelSpec, TensorSpec

log = logging.getLogger(__name__)

_CUDA = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"
_NP_TYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32}


class OnnxBackend(InferenceBackend):
    def __init__(self, device: str = "auto", intra_op_threads: int = 0):
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"device must be auto|cuda|cpu, got {device!r}")
        self._device = device
        self._intra_op_threads = intra_op_threads
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._specs: Dict[str, ModelSpec] = {}
        self._active_provider = _CPU

    @property
    def name(self) -> str:
        return "onnx-cuda" if self._active_provider == _CUDA else "onnx-cpu"

    def _providers(self) -> List[str]:
        available = ort.get_available_providers()
        if self._device == "cpu":
            return [_CPU]
        if _CUDA in available:
            return [_CUDA, _CPU]
        if self._device == "cuda":
            log.warning("CUDA requested but onnxruntime has no CUDAExecutionProvider "
                        "(installed package: %s); falling back to CPU", ort.__version__)
        return [_CPU]

    def load_model(self, spec: ModelSpec) -> None:
        if not spec.path.is_file():
            raise FileNotFoundError(f"model file not found: {spec.path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._intra_op_threads:
            opts.intra_op_num_threads = self._intra_op_threads
        session = ort.InferenceSession(str(spec.path), sess_options=opts, providers=self._providers())
        # ORT silently drops CUDA when its DLLs fail to load, so read back what is actually active.
        self._active_provider = session.get_providers()[0]
        if self._device == "cuda" and self._active_provider != _CUDA:
            log.warning("CUDA provider failed to initialize for %s; running on CPU", spec.model_id)
        self._sessions[spec.model_id] = session
        self._specs[spec.model_id] = spec

    def _session(self, model_id: str) -> ort.InferenceSession:
        try:
            return self._sessions[model_id]
        except KeyError:
            raise KeyError(f"model {model_id!r} not loaded") from None

    def input_specs(self, model_id: str) -> List[TensorSpec]:
        return [_to_spec(t) for t in self._session(model_id).get_inputs()]

    def output_specs(self, model_id: str) -> List[TensorSpec]:
        return [_to_spec(t) for t in self._session(model_id).get_outputs()]

    def infer(self, model_id: str, image: np.ndarray) -> Dict[str, np.ndarray]:
        session = self._session(model_id)
        spec = self._specs[model_id]
        if spec.input_size is None:
            raise ValueError(f"{model_id} is not an image model; use infer_tensors()")
        expected = (*spec.input_size, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image of shape {expected}, got {image.dtype} {image.shape}")

        mean = np.asarray(spec.mean, dtype=np.float32)
        std = np.asarray(spec.std, dtype=np.float32)
        x = (image.astype(np.float32) - mean) / std
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # HWC -> NCHW

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, {input_name: x})
        return dict(zip(output_names, outputs))

    def infer_tensors(self, model_id: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        session = self._session(model_id)
        feeds = {}
        for spec in session.get_inputs():
            if spec.name not in inputs:
                raise KeyError(f"{model_id}: missing input {spec.name!r}")
            dtype = _NP_TYPES.get(spec.type)
            feeds[spec.name] = np.ascontiguousarray(inputs[spec.name], dtype=dtype)
        output_names = [o.name for o in session.get_outputs()]
        return dict(zip(output_names, session.run(output_names, feeds)))

    def close(self) -> None:
        self._sessions.clear()
        self._specs.clear()


def _to_spec(t: "ort.NodeArg") -> TensorSpec:
    shape = tuple(d if isinstance(d, int) else None for d in t.shape)
    return TensorSpec(name=t.name, shape=shape, dtype=t.type)
