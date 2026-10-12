"""Central configuration. Thresholds live here so later stages can tune them in one place."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DB_FILENAME = "vah.db"

# CLIP ViT-B/32 image/text embedding size; fixed by the vec0 table definition.
EMBEDDING_DIM = 512
# Photos with a longer side above this are downscaled on import (a 36 MP stock photo would
# otherwise blow the per-photo time budget); the library keeps the downscaled copy.
MAX_PHOTO_SIDE = 4096
VOCABULARY_PATH = ROOT_DIR / "backend" / "vocab" / "vocabulary.toml"


@dataclass(frozen=True)
class ClipConfig:
    # Stored with every vector: (embedding_model, backend family). ONNX on CPU or CUDA gives
    # the same FP32 vectors, so the space is keyed by backend family, not device.
    embedding_model: str = "clip-vit-b32"
    image_model_path: Path = MODELS_DIR / "clip-vit-b32-image.onnx"
    image_size: int = 224
    # OpenAI CLIP normalization, in 0-255 units.
    mean: tuple = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
    std: tuple = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
    # Cut-outs and candidate crops are padded to a square of this gray (CLIP-neutral).
    pad_gray: int = 128
    candidate_margin_ratio: float = 0.1
    # English text encoder (tagging prompts) and multilingual text encoder (search queries);
    # both always run on the host CPU.
    text_model_path: Path = MODELS_DIR / "clip-vit-b32-text.onnx"
    text_tokenizer_dir: Path = MODELS_DIR / "clip-vit-b32-tokenizer"
    text_max_len: int = 77
    query_model_path: Path = MODELS_DIR / "clip-multilingual-text.onnx"
    query_tokenizer_dir: Path = MODELS_DIR / "clip-multilingual-tokenizer"
    query_max_len: int = 128


CLIP = ClipConfig()


@dataclass(frozen=True)
class SearchConfig:
    # Vector neighbours fetched before re-ranking.
    candidates: int = 200
    # Score = cosine + bonus per matched query term (category / context / attribute tag).
    category_bonus: float = 0.10
    context_bonus: float = 0.08
    attribute_bonus: float = 0.05
    # Query mentions a context and the asset's category is one of that context's rule
    # categories (小酌 -> 酒瓶 / 罐裝飲料): context tags alone hit every asset of the photo.
    context_object_bonus: float = 0.08
    default_k: int = 20


SEARCH = SearchConfig()

# Canvas output sizes (docs/SPEC.md section 2): key -> (display name, width, height).
CANVAS_PRESETS = {
    "product": ("商品主圖", 800, 800),
    "ig": ("IG 貼文", 1080, 1350),
    "banner": ("Banner", 1640, 856),
}

# Which inference backend to use: "onnx" (demo) or "hailo" (target hardware, not yet implemented).
BACKEND = os.environ.get("VAH_BACKEND", "onnx")
# Execution device for the ONNX backend: "auto" (CUDA if usable, else CPU), "cuda", or "cpu".
DEVICE = os.environ.get("VAH_DEVICE", "auto")


@dataclass(frozen=True)
class SegmentationConfig:
    model_id: str
    model_path: Path
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_det: int = 100
    mask_threshold: float = 0.5


YOLO_SEG = SegmentationConfig(
    model_id="yolov8s-seg",
    model_path=MODELS_DIR / "yolov8s-seg.onnx",
)

# FastSAM "segment everything": class-agnostic, 640 to match Hailo Model Zoo fast_sam_s.
# conf/iou follow the FastSAM reference defaults.
FASTSAM = SegmentationConfig(
    model_id="fastsam-s",
    model_path=MODELS_DIR / "fastsam-s.onnx",
    conf_threshold=0.4,
    iou_threshold=0.9,
    max_det=100,
)


@dataclass(frozen=True)
class MergeConfig:
    # A YOLO detection and a FastSAM mask with mask IoU >= fuse_iou are the same object:
    # keep the YOLO label, take the FastSAM mask (SA-1B trained, better edges).
    fuse_iou: float = 0.7
    # Remaining candidates with mask IoU >= dedup_iou against a kept one are dropped.
    dedup_iou: float = 0.7
    # Masks outside [min, max] fraction of the image area are dropped.
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.9
    # Connected components smaller than this fraction of their mask are removed.
    min_component_ratio: float = 0.02


@dataclass(frozen=True)
class RefineConfig:
    # Work region = mask box grown by this fraction of its longer side.
    margin_ratio: float = 0.1
    # Larger work regions are refined at this longer side and the alpha upsampled
    # (keeps full-frame objects in 12 MP photos within the per-asset time budget).
    max_work_side: int = 2048
    # Uncertain band around the coarse mask edge, as a fraction of the image's longer side.
    # Masks come from a 160x160 proto, so edges are off by up to ~1 proto cell (long_side/160).
    edge_band_ratio: float = 0.006
    min_edge_band_px: int = 4
    # GrabCut relabels pixels inside the band by colour (moves the edge onto the real one).
    # Runs on the work region downscaled to this longer side; 0 iterations disables it.
    grabcut_iters: int = 3
    grabcut_max_side: int = 768
    # Guided filter turns the hard mask into soft alpha along image edges.
    # Radius as a fraction of the edge band; eps with guide and mask in [0, 1].
    guided_radius_ratio: float = 0.5
    guided_eps: float = 1e-4
    feather_px: float = 1.5
    # Alpha below floor -> 0, above ceil -> 1, removes faint haze and pinholes.
    alpha_floor: float = 0.05
    alpha_ceil: float = 0.95
    # Transparent border kept around the object in the exported PNG.
    png_padding: int = 8


MERGE = MergeConfig()
REFINE = RefineConfig()

# Acceptance targets on CPU (docs/SPEC.md section 6).
PROPOSE_BUDGET_MS = 5000          # both seg models + merge, per photo
EXTRACT_PER_ASSET_BUDGET_MS = 3000  # refine + cut-out, per selected object
