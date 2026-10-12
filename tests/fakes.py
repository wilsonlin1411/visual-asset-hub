"""Test doubles for the model-backed services (no ONNX files needed)."""
import hashlib

import numpy as np

from backend.config import EMBEDDING_DIM
from backend.db.vector_index import EmbeddingSpace
from backend.services.tagging import AssetAnalysis, CandidateAnalysis, CategoryResult

SPACE = EmbeddingSpace("fake-clip", "test")


def text_vector(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding per string."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).normal(size=EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeTextEncoder:
    model_id = "fake-text"

    def encode(self, texts):
        return np.stack([text_vector(t) for t in texts]) if texts else np.zeros((0, EMBEDDING_DIM), np.float32)


class FakeImageEncoder:
    space = SPACE

    def encode_cutout(self, bgra):
        mean = bgra[..., :3][bgra[..., 3] > 0].mean(axis=0) if (bgra[..., 3] > 0).any() else np.zeros(3)
        return text_vector(f"color-{int(mean[2]) // 32}-{int(mean[1]) // 32}-{int(mean[0]) // 32}")


class FakeAnalyzer:
    """YOLO-labelled candidates look like objects ('咖啡杯'); unlabelled ones like background."""

    objectness_threshold = 0.35
    space = SPACE
    image_encoder = FakeImageEncoder()

    def analyze_candidates(self, image, masks, labels, scores):
        return [CandidateAnalysis(0.9 if label else 0.2, "咖啡杯" if label else "其他", 0.8, "clip")
                for label in labels]

    def scene_contexts(self, image):
        return []

    def rule_contexts(self, candidates, assets):
        evidence = any(c.class_name for c in candidates) or any(a.category == "咖啡杯" for a in assets)
        return [("context", "咖啡廳", 1.0, "rule:咖啡杯 + 甜點")] if evidence else []

    def analyze_asset(self, bgra, label, score):
        cat = CategoryResult("咖啡杯", 0.8, "yolo") if label else CategoryResult("其他", 0.1, "fallback")
        return AssetAnalysis(self.image_encoder.encode_cutout(bgra), cat, [("attribute", "紅", 0.6, "color")])
