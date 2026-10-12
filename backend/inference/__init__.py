from .base import Detection, InferenceBackend, ModelSpec, TensorSpec
from .masks import RegionMask


def create_backend(kind: str = "onnx", device: str = "auto") -> InferenceBackend:
    """Factory so the rest of the app never imports a concrete backend directly."""
    if kind == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend(device=device)
    if kind == "hailo":
        from .hailo_backend import HailoBackend
        return HailoBackend()
    raise ValueError(f"unknown backend {kind!r}")


__all__ = ["Detection", "InferenceBackend", "ModelSpec", "RegionMask", "TensorSpec", "create_backend"]
