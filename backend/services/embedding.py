"""CLIP encoders on top of InferenceBackend.

  ClipImageEncoder  image backend (ONNX now, Hailo later): cut-outs, candidate regions, photos
  TextEncoder       always a CPU OnnxBackend: English CLIP text (tagging prompts) and the
                    multilingual text encoder aligned to the same space (Chinese queries)
All vectors are L2-normalized float32 [512].
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from tokenizers import Tokenizer

from .. import config
from ..db.vector_index import EmbeddingSpace
from ..inference import InferenceBackend, ModelSpec, RegionMask


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def pad_to_square(rgb: np.ndarray, size: int, gray: int) -> np.ndarray:
    """Fit the whole image into a size x size square (no cropping), padding with gray."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    small = cv2.resize(rgb, (nw, nh), interpolation=interp)
    out = np.full((size, size, 3), gray, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    """Standard CLIP preprocessing for whole photos: short side -> size, center crop."""
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    resized = cv2.resize(rgb, (max(size, round(w * scale)), max(size, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    y0, x0 = (resized.shape[0] - size) // 2, (resized.shape[1] - size) // 2
    return np.ascontiguousarray(resized[y0:y0 + size, x0:x0 + size])


class ClipImageEncoder:
    model_id = "clip-image"

    def __init__(self, backend: InferenceBackend, cfg: config.ClipConfig = config.CLIP,
                 backend_family: str = config.BACKEND):
        self.backend = backend
        self.cfg = cfg
        self.space = EmbeddingSpace(cfg.embedding_model, backend_family)
        backend.load_model(ModelSpec(self.model_id, cfg.image_model_path, (cfg.image_size, cfg.image_size),
                                     cfg.mean, cfg.std))

    def encode_rgb(self, rgb224: np.ndarray) -> np.ndarray:
        out = self.backend.infer(self.model_id, rgb224)
        return l2_normalize(next(iter(out.values())).reshape(-1))

    def encode_cutout(self, bgra: np.ndarray) -> np.ndarray:
        """Transparent cut-out composited on gray, padded to a square."""
        a = bgra[..., 3:4].astype(np.float32) / 255
        rgb = cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        comp = (rgb * a + self.cfg.pad_gray * (1 - a) + 0.5).astype(np.uint8)
        return self.encode_rgb(pad_to_square(comp, self.cfg.image_size, self.cfg.pad_gray))

    def encode_region(self, image_bgr: np.ndarray, mask: RegionMask) -> np.ndarray:
        """Candidate as CLIP should judge it: its box (+margin), everything off-mask painted gray."""
        h, w = image_bgr.shape[:2]
        mh, mw = mask.data.shape
        m = round(self.cfg.candidate_margin_ratio * max(mh, mw))
        x0, y0 = max(mask.x0 - m, 0), max(mask.y0 - m, 0)
        x1, y1 = min(mask.x1 + m, w), min(mask.y1 + m, h)
        crop = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        keep = np.zeros(crop.shape[:2], bool)
        keep[mask.y0 - y0:mask.y1 - y0, mask.x0 - x0:mask.x1 - x0] = mask.data
        crop[~keep] = self.cfg.pad_gray
        return self.encode_rgb(pad_to_square(crop, self.cfg.image_size, self.cfg.pad_gray))

    def encode_photo(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.encode_rgb(center_crop(rgb, self.cfg.image_size))


class TextEncoder:
    """Tokenizer (`tokenizers`, no torch) + ONNX text model run via InferenceBackend.infer_tensors."""

    def __init__(self, backend: InferenceBackend, model_id: str, model_path: Path, tokenizer_dir: Path,
                 max_len: int, pad_to_max: bool, batch_size: int = 64):
        self.backend = backend
        self.model_id = model_id
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
        pad_token = next((t for t in ("<|endoftext|>", "[PAD]", "<pad>") if self.tokenizer.token_to_id(t) is not None),
                         None)
        if pad_token is None:
            raise ValueError(f"no pad token found in {tokenizer_dir}")
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad_token), pad_token=pad_token,
                                      length=max_len if pad_to_max else None)
        backend.load_model(ModelSpec(model_id, Path(model_path)))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch(list(texts[i:i + self.batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            res = self.backend.infer_tensors(self.model_id, {"input_ids": ids, "attention_mask": mask})
            out.append(next(iter(res.values())))
        if not out:
            return np.zeros((0, config.EMBEDDING_DIM), np.float32)
        return l2_normalize(np.concatenate(out))


def clip_text_encoder(cpu_backend: InferenceBackend, cfg: config.ClipConfig = config.CLIP) -> TextEncoder:
    """English CLIP text encoder (used for tagging prompts)."""
    return TextEncoder(cpu_backend, "clip-text", cfg.text_model_path, cfg.text_tokenizer_dir,
                       cfg.text_max_len, pad_to_max=True)


def query_encoder(cpu_backend: InferenceBackend, cfg: config.ClipConfig = config.CLIP) -> TextEncoder:
    """Multilingual text encoder aligned to CLIP ViT-B/32 (used for Chinese search queries)."""
    return TextEncoder(cpu_backend, "clip-query", cfg.query_model_path, cfg.query_tokenizer_dir,
                       cfg.query_max_len, pad_to_max=False)
