"""Semantic tagging with CLIP zero-shot + pixel statistics + co-occurrence rules.

  Tagger    pure functions over CLIP vectors (prompt embeddings built once per vocabulary)
  Analyzer  Tagger + ClipImageEncoder: what the library calls for candidates, photos, assets

Objectness: CLIP looks at the candidate region (off-mask pixels painted gray) and we take
the softmax probability mass on "object" prompts versus "background" prompts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from ..inference import RegionMask
from .embedding import ClipImageEncoder, TextEncoder, l2_normalize
from .vocab import Entry, Vocabulary

log = logging.getLogger(__name__)

TagTuple = Tuple[str, str, float, str]  # (kind, value, score, source)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class LabelSet:
    names: List[str]
    matrix: np.ndarray  # (L, D) one normalized embedding per label (prompt ensemble)

    def probs(self, vec: np.ndarray, scale: float) -> np.ndarray:
        return softmax(scale * (self.matrix @ vec))


class PromptBank:
    """Text embeddings of prompts, cached on disk per text model (prompts rarely change)."""

    def __init__(self, encoder: TextEncoder, cache_file: Optional[Path] = None):
        self.encoder = encoder
        self.cache_file = cache_file
        self._cache: Dict[str, np.ndarray] = {}
        if cache_file and cache_file.is_file():
            data = np.load(cache_file)
            texts = json.loads(str(data["texts"]))
            self._cache = dict(zip(texts, data["vectors"]))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.encoder.encode(missing)):
                self._cache[t] = v
            if self.cache_file:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                texts_all = list(self._cache)
                np.savez(self.cache_file, texts=json.dumps(texts_all, ensure_ascii=False),
                         vectors=np.stack([self._cache[t] for t in texts_all]))
        return np.stack([self._cache[t] for t in texts])

    def label_set(self, entries: Sequence[Entry], templates: Sequence[str]) -> LabelSet:
        rows = []
        for e in entries:
            texts = [tpl.format(p) for p in e.prompts for tpl in templates]
            rows.append(l2_normalize(self.embed(texts).mean(axis=0)))
        return LabelSet([e.name for e in entries], np.stack(rows))


# ------------------------------------------------------------------------------- colours

def classify_colors(hsv: np.ndarray) -> np.ndarray:
    """Per-pixel colour id (index into COLOR_KEYS) from OpenCV HSV (H 0-180, S/V 0-255)."""
    h, s, v = (hsv[:, i].astype(np.int32) for i in range(3))
    out = np.full(len(hsv), COLOR_KEYS.index("gray"), np.int32)

    hue = np.select(
        [(h < 8) | (h >= 170), h < 20, h < 35, h < 85, h < 130, h < 150],
        [COLOR_KEYS.index(k) for k in ("red", "orange", "yellow", "green", "blue", "purple")],
        default=COLOR_KEYS.index("pink"))
    chromatic = s >= 40
    out[chromatic] = hue[chromatic]
    warm = (h >= 5) & (h < 30)
    out[chromatic & warm & (v < 170)] = COLOR_KEYS.index("brown")
    out[chromatic & warm & (s < 100) & (v >= 170)] = COLOR_KEYS.index("beige")
    out[chromatic & ((h < 8) | (h >= 160)) & (s < 120) & (v >= 180)] = COLOR_KEYS.index("pink")
    out[(~chromatic) & (v >= 200)] = COLOR_KEYS.index("white")
    out[v < 50] = COLOR_KEYS.index("black")
    return out


COLOR_KEYS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink",
              "brown", "beige"]


def color_ratios(bgra: np.ndarray, max_pixels: int = 50_000) -> Dict[str, float]:
    opaque = bgra[..., 3] >= 128
    px = bgra[..., :3][opaque]
    if px.size == 0:
        return {}
    if len(px) > max_pixels:
        px = px[np.random.default_rng(0).choice(len(px), max_pixels, replace=False)]
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    counts = np.bincount(classify_colors(hsv), minlength=len(COLOR_KEYS))
    return {k: counts[i] / counts.sum() for i, k in enumerate(COLOR_KEYS) if counts[i]}


# -------------------------------------------------------------------------------- tagger

@dataclass
class CategoryResult:
    name: str
    score: float
    source: str  # yolo | clip | fallback


class Tagger:
    def __init__(self, vocab: Vocabulary, text_encoder: TextEncoder, cache_dir: Optional[Path] = None):
        self.vocab = vocab
        cache = cache_dir / f"prompts-{vocab.digest}-{text_encoder.model_id}.npz" if cache_dir else None
        bank = PromptBank(text_encoder, cache)
        t = vocab.templates
        self.categories = bank.label_set(vocab.categories, t["category"])
        self.materials = bank.label_set(vocab.materials, t["material"])
        self.styles = bank.label_set(vocab.styles, t["style"])
        self.contexts = bank.label_set(vocab.contexts, t["context"])
        obj, bg = vocab.objectness["object"], vocab.objectness["background"]
        self._obj_matrix = bank.embed([*obj, *bg])
        self._n_obj = len(obj)
        # Scene matching: contexts with scene=true plus a "none of these" row (generic prompts).
        scene_entries = [c for c in vocab.contexts if c.scene]
        self._scene_names = [c.name for c in scene_entries]
        rows = [self.contexts.matrix[self.contexts.names.index(n)] for n in self._scene_names]
        if vocab.scene_none:
            rows.append(l2_normalize(bank.embed(vocab.scene_none).mean(axis=0)))
        self._scene_matrix = np.stack(rows)
        self.params = vocab.tagging
        self.scale = float(self.params["logit_scale"])
        self.objectness_threshold = float(vocab.objectness["threshold"])
        self._color_names = {c.key: c.name for c in vocab.colors}

    def objectness(self, vec: np.ndarray) -> float:
        p = softmax(self.scale * (self._obj_matrix @ vec))
        return float(p[:self._n_obj].sum())

    def category(self, vec: np.ndarray, coco_label: Optional[str] = None,
                 coco_score: Optional[float] = None) -> CategoryResult:
        probs = self.categories.probs(vec, self.scale)
        order = np.argsort(-probs)
        names = self.categories.names
        mapped = self.vocab.coco_to_category(coco_label)
        topk = {names[i] for i in order[:int(self.params["yolo_trust_topk"])]}
        if (mapped and (coco_score or 0) >= self.params["yolo_trust_score"] and mapped in topk
                and probs[names.index(mapped)] >= self.params["yolo_min_prob"]):
            return CategoryResult(mapped, float(probs[names.index(mapped)]), "yolo")
        best = names[order[0]]
        fallback = self.vocab.fallback_category.name
        if probs[order[0]] < self.params["category_min_prob"] or best == fallback:
            return CategoryResult(fallback, float(probs[order[0]]), "fallback")
        return CategoryResult(best, float(probs[order[0]]), "clip")

    def attributes(self, vec: np.ndarray) -> List[TagTuple]:
        tags: List[TagTuple] = []
        for label_set, source in ((self.materials, "clip-material"), (self.styles, "clip-style")):
            probs = label_set.probs(vec, self.scale)
            for i in np.argsort(-probs)[:2]:
                if probs[i] >= self.params["attribute_min_prob"]:
                    tags.append(("attribute", label_set.names[i], float(probs[i]), source))
        return tags

    def colors(self, bgra: np.ndarray) -> List[TagTuple]:
        ratios = sorted(color_ratios(bgra).items(), key=lambda kv: -kv[1])[:2]
        return [("attribute", self._color_names[k], float(r), "color") for k, r in ratios
                if r >= self.params["color_min_ratio"] and k in self._color_names]

    def scene_contexts(self, photo_vec: np.ndarray) -> List[TagTuple]:
        probs = softmax(self.scale * (self._scene_matrix @ photo_vec))
        n = len(self._scene_names)
        if int(np.argmax(probs)) >= n:  # the generic "none" prompts win: no scene context
            return []
        return [("context", self._scene_names[i], float(probs[i]), "clip-scene")
                for i in np.argsort(-probs[:n])[:2] if probs[i] >= self.params["scene_min_prob"]]

    def rule_categories(self, candidates: Sequence, assets: Sequence = ()) -> Set[str]:
        """Categories present in a photo, as evidence for the co-occurrence rules.

        Confirming an asset validates that the object exists, not its CLIP-guessed category,
        so assets count when their category came from YOLO+CLIP agreement, from the user, or
        from a confident CLIP guess (>= rule_min_prob). A candidate counts when it is visible
        (YOLO-labelled or objectness >= threshold) and either YOLO and CLIP agree on its
        category or CLIP alone is confident. `candidates` are rows with class_name, score,
        category, category_score, objectness; `assets` rows with category, category_score,
        category_source.
        """
        vocab = self.vocab
        ignore = {vocab.fallback_category.name, *vocab.excluded_categories}
        min_prob = self.params["rule_min_prob"]
        cats = {a.category for a in assets if a.category and a.category not in ignore
                and (a.category_source in ("yolo", "user") or (a.category_score or 0) >= min_prob)}
        for c in candidates:
            if not c.category or c.category in ignore:
                continue
            labeled = c.class_name is not None
            if not labeled and (c.objectness or 0) < self.objectness_threshold:
                continue
            yolo_agrees = (labeled and vocab.coco_to_category(c.class_name) == c.category
                           and c.score >= self.params["yolo_trust_score"]
                           and (c.category_score or 0) >= self.params["yolo_min_prob"])
            if yolo_agrees or (c.category_score or 0) >= min_prob:
                cats.add(c.category)
        return cats

    def rule_contexts(self, categories: Set[str]) -> List[TagTuple]:
        out: Dict[str, TagTuple] = {}
        for rule in self.vocab.rules:
            if rule.matches(categories):
                for ctx in rule.contexts:
                    out.setdefault(ctx, ("context", ctx, 1.0, f"rule:{rule.describe()}"))
        return list(out.values())


# ------------------------------------------------------------------------------ analyzer

@dataclass
class CandidateAnalysis:
    objectness: float
    category: str
    category_score: float
    category_source: str


@dataclass
class AssetAnalysis:
    vector: np.ndarray
    category: CategoryResult
    attributes: List[TagTuple] = field(default_factory=list)


class Analyzer:
    """CLIP image encoder + Tagger: the semantic step of the capture pipeline."""

    def __init__(self, image_encoder: ClipImageEncoder, tagger: Tagger):
        self.image_encoder = image_encoder
        self.tagger = tagger

    @property
    def space(self):
        return self.image_encoder.space

    @property
    def objectness_threshold(self) -> float:
        return self.tagger.objectness_threshold

    def analyze_candidates(self, image_bgr: np.ndarray, masks: Sequence[RegionMask],
                           coco_labels: Sequence[Optional[str]], coco_scores: Sequence[Optional[float]]
                           ) -> List[CandidateAnalysis]:
        out = []
        for mask, label, score in zip(masks, coco_labels, coco_scores):
            vec = self.image_encoder.encode_region(image_bgr, mask)
            cat = self.tagger.category(vec, label, score)
            out.append(CandidateAnalysis(self.tagger.objectness(vec), cat.name, cat.score, cat.source))
        return out

    def scene_contexts(self, image_bgr: np.ndarray) -> List[TagTuple]:
        return self.tagger.scene_contexts(self.image_encoder.encode_photo(image_bgr))

    def rule_contexts(self, candidates: Sequence, assets: Sequence) -> List[TagTuple]:
        """Contexts from category co-occurrence; needs no model, only stored analysis rows."""
        return self.tagger.rule_contexts(self.tagger.rule_categories(candidates, assets))

    def analyze_asset(self, bgra: np.ndarray, coco_label: Optional[str], coco_score: Optional[float]
                      ) -> AssetAnalysis:
        vec = self.image_encoder.encode_cutout(bgra)
        return AssetAnalysis(vec, self.tagger.category(vec, coco_label, coco_score),
                             self.tagger.attributes(vec) + self.tagger.colors(bgra))
