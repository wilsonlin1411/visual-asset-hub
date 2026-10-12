"""Asset library: photos -> stored candidates -> confirmed, tagged, embedded assets.

Persists everything the capture pipeline produces so the user steps (look at candidates,
confirm some) can happen in different requests or processes:

    register_photo()  store the upload (status 'processing'); returns at once
    process_photo()   segmentation + CLIP per candidate (objectness, category) + photo
                      contexts; status 'ready' (run by the background queue)
    add_photo()       register + process, synchronously (CLI)
    candidates()      stored candidates of a photo
    confirm()         refine + cut out selected candidates, then CLIP vector + tags
    discard()         mark candidates the user rejected
    retag() / reembed()  recompute tags after a vocabulary change / vectors for a new backend

Files live under data/ (paths in the DB are relative to it):
    photos/{photo_id}.jpg   candidates/{photo_id}/{idx:03d}.png (1-bit mask crop)   assets/{asset_id}.png
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .. import config
from ..db import Asset, CandidateRecord, Database, Photo, Tag, repo
from ..db.vector_index import EmbeddingSpace, NumpyIndex, VectorIndex, create_index
from ..imageio import decode, encode, imread
from ..inference import RegionMask, create_backend
from .capture import CapturePipeline, Proposal, extract_assets
from .merge import Candidate
from .refine import GuidedFilterRefiner
from .tagging import Analyzer
from .thumbs import Thumbs

log = logging.getLogger(__name__)

PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class AddPhotoResult:
    photo: Photo
    created: bool                  # False when the same file was already in the library
    proposal: Optional[Proposal]   # None when not processed now


def default_pipeline() -> CapturePipeline:
    return CapturePipeline(create_backend(config.BACKEND, config.DEVICE))


def default_analyzer() -> Analyzer:
    from .embedding import ClipImageEncoder, clip_text_encoder
    from .tagging import Tagger
    from .vocab import load_vocabulary

    image_encoder = ClipImageEncoder(create_backend(config.BACKEND, config.DEVICE))
    # Text encoders always run on the host CPU.
    tagger = Tagger(load_vocabulary(), clip_text_encoder(create_backend("onnx", "cpu")),
                    cache_dir=config.DATA_DIR / "cache")
    return Analyzer(image_encoder, tagger)


def embedding_space() -> EmbeddingSpace:
    """Space of the image vectors in use, known without loading any model."""
    return EmbeddingSpace(config.CLIP.embedding_model, config.BACKEND)


class AssetLibrary:
    def __init__(self, data_dir: Path = config.DATA_DIR, db: Optional[Database] = None,
                 pipeline_factory: Callable[[], CapturePipeline] = default_pipeline,
                 analyzer_factory: Callable[[], Optional[Analyzer]] = default_analyzer,
                 refine_cfg: config.RefineConfig = config.REFINE):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or Database(self.data_dir / config.DB_FILENAME)
        self.index: VectorIndex = create_index(self.db, config.EMBEDDING_DIM)
        self.refine_cfg = refine_cfg
        self.refiner = GuidedFilterRefiner(refine_cfg)
        self._pipeline_factory = pipeline_factory
        self._analyzer_factory = analyzer_factory
        self._pipeline: Optional[CapturePipeline] = None
        self._analyzer: Optional[Analyzer] = None
        self._analyzer_loaded = False
        self._init_lock = threading.Lock()
        self.thumbs = Thumbs(self.data_dir)

    @property
    def pipeline(self) -> CapturePipeline:
        """Segmentation models load on first use; confirm/list/delete never need them."""
        with self._init_lock:
            if self._pipeline is None:
                self._pipeline = self._pipeline_factory()
            return self._pipeline

    @property
    def analyzer(self) -> Optional[Analyzer]:
        """CLIP models load on first use; None disables semantic tagging (tests, tools)."""
        with self._init_lock:
            if not self._analyzer_loaded:
                self._analyzer = self._analyzer_factory()
                self._analyzer_loaded = True
            return self._analyzer

    @property
    def objectness_threshold(self) -> float:
        a = self.analyzer
        return a.objectness_threshold if a is not None else 0.0

    def path(self, relative: str) -> Path:
        return self.data_dir / relative

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------------ photos

    def register_photo(self, data: bytes, original_name: str) -> Tuple[Photo, bool]:
        """Store an upload and return immediately (status 'processing'). Dedupes by content."""
        sha256 = hashlib.sha256(data).hexdigest()
        existing = repo.get_photo_by_sha256(self.db, sha256)
        if existing is not None:
            return existing, False

        image = decode(data)
        h, w = image.shape[:2]
        ext = Path(original_name).suffix.lower()
        ext = ext if ext in PHOTO_EXTS else ".jpg"
        if max(h, w) > config.MAX_PHOTO_SIDE:
            scale = config.MAX_PHOTO_SIDE / max(h, w)
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
            data, ext = encode(image, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]), ".jpg"
            log.info("%s downscaled to %dx%d on import", original_name, w, h)

        written: List[Path] = []
        try:
            with self.db.transaction():
                photo = repo.insert_photo(self.db, "", original_name, sha256, w, h, status="processing")
                photo.path = f"photos/{photo.id}{ext}"
                self._write(photo.path, data, written)
                repo.set_photo_path(self.db, photo.id, photo.path)
        except BaseException:
            self._remove(written)
            raise
        return photo, True

    def process_photo(self, photo_id: int) -> Proposal:
        """Segment, analyse and store candidates; marks the photo 'ready' (or 'failed')."""
        photo = self.get_photo(photo_id)
        try:
            image = imread(self.path(photo.path))
            proposal = self.pipeline.propose(image)
            cands = proposal.candidates
            analyzer = self.analyzer
            analyses, scene = None, []
            if analyzer is not None:
                if cands:
                    analyses = analyzer.analyze_candidates(image, [c.mask for c in cands],
                                                           [c.class_name for c in cands], [c.score for c in cands])
                scene = analyzer.scene_contexts(image)
        except Exception as exc:
            with self.db.transaction():
                repo.set_photo_status(self.db, photo_id, "failed", f"{type(exc).__name__}: {exc}")
            raise

        written: List[Path] = []
        try:
            with self.db.transaction():
                # Re-processing (e.g. after a crash) replaces earlier candidates.
                self.db.conn.execute("DELETE FROM candidates WHERE photo_id = ?", (photo_id,))
                for i, c in enumerate(cands):
                    mask_rel = f"candidates/{photo_id}/{c.id:03d}.png"
                    self._write(mask_rel, encode_mask(c.mask), written)
                    cid = repo.insert_candidate(self.db, photo_id, c.id, c.class_id, c.class_name, c.score,
                                                c.label_model, c.mask_model, c.mask.box, c.area, mask_rel)
                    if analyses is not None:
                        a = analyses[i]
                        repo.set_candidate_analysis(self.db, cid, a.objectness, a.category, a.category_score)
                repo.set_photo_tags(self.db, photo_id, scene)
                self._refresh_contexts(photo_id)
                repo.set_photo_status(self.db, photo_id, "ready")
        except BaseException:
            self._remove(written)
            raise
        return proposal

    def add_photo(self, src: Path, original_name: Optional[str] = None) -> AddPhotoResult:
        src = Path(src)
        photo, created = self.register_photo(src.read_bytes(), original_name or src.name)
        if not created and photo.status == "ready":
            return AddPhotoResult(photo, False, None)
        proposal = self.process_photo(photo.id)
        return AddPhotoResult(self.get_photo(photo.id), created, proposal)

    def get_photo(self, photo_id: int) -> Photo:
        photo = repo.get_photo(self.db, photo_id)
        if photo is None:
            raise KeyError(f"photo {photo_id} not found")
        return photo

    def list_photos(self, limit: int = 100, offset: int = 0) -> List[Photo]:
        return repo.list_photos(self.db, limit, offset)

    def photo_contexts(self, photo_id: int) -> List[tuple]:
        return repo.get_photo_tags(self.db, photo_id)

    def delete_photo(self, photo_id: int) -> bool:
        photo = repo.get_photo(self.db, photo_id)
        if photo is None:
            return False
        asset_files = [a.png_path for a in repo.list_assets(self.db, photo_id=photo_id, limit=1_000_000)]
        with self.db.transaction():
            repo.delete_photo(self.db, photo_id)  # cascades rows (and vec rows via trigger)
        # Rows first, files second: a failed unlink leaves an orphan file, never a dangling row.
        self._remove([self.path(p) for p in (photo.path, *asset_files)])
        shutil.rmtree(self.path(f"candidates/{photo_id}"), ignore_errors=True)
        self.thumbs.drop_photo(photo_id)
        for p in asset_files:
            self.thumbs.drop_asset(int(Path(p).stem))
        self._invalidate_index()
        return True

    def photo_counts(self, photo_ids: List[int]) -> dict:
        return repo.photo_counts(self.db, photo_ids)

    # -------------------------------------------------------------------- candidates

    def candidates(self, photo_id: int, statuses: Optional[List[str]] = None,
                   include_hidden: bool = True) -> List[CandidateRecord]:
        self.get_photo(photo_id)
        records = repo.list_candidates(self.db, photo_id, statuses=statuses)
        if not include_hidden:
            threshold = self.objectness_threshold
            records = [r for r in records if not r.hidden(threshold)]
        return records

    def load_candidate_mask(self, record: CandidateRecord) -> RegionMask:
        return decode_mask(self.path(record.mask_path).read_bytes(), record.bbox)

    def confirm(self, photo_id: int, idxs: Optional[Iterable[int]] = None) -> List[Asset]:
        """Create assets for the given candidates (default: all pending). Idempotent per candidate."""
        photo = self.get_photo(photo_id)
        if photo.status != "ready":
            raise ValueError(f"photo {photo_id} is {photo.status}, not ready")
        if idxs is None:
            records = repo.list_candidates(self.db, photo_id, statuses=["pending"])
        else:
            idxs = sorted(set(idxs))
            records = repo.list_candidates(self.db, photo_id, idxs)
            unknown = set(idxs) - {r.idx for r in records}
            if unknown:
                raise KeyError(f"photo {photo_id} has no candidates {sorted(unknown)}")

        done = {r.idx: repo.get_asset_by_candidate(self.db, r.id) for r in records if r.status == "confirmed"}
        todo = [r for r in records if r.status != "confirmed"]
        if not todo:
            return [done[i] for i in sorted(done)]

        image = imread(self.path(photo.path))
        cands = [Candidate(id=r.idx, mask=self.load_candidate_mask(r), score=r.score, class_id=r.class_id,
                           class_name=r.class_name, label_model=r.label_model, mask_model=r.mask_model,
                           area=r.area) for r in todo]
        extracted = extract_assets(image, cands, None, self.refiner, self.refine_cfg)
        by_idx = {r.idx: r for r in todo}
        analyzer = self.analyzer
        analyses = [analyzer.analyze_asset(ex.rgba, by_idx[ex.candidate_id].class_name,
                                           by_idx[ex.candidate_id].score) if analyzer else None
                    for ex in extracted]

        written: List[Path] = []
        try:
            with self.db.transaction():
                for ex, an in zip(extracted, analyses):
                    rec = by_idx[ex.candidate_id]
                    category, cat_score, cat_source = (
                        (an.category.name, an.category.score, an.category.source) if an else (ex.class_name, None, None))
                    asset = repo.insert_asset(
                        self.db, photo_id, rec.id, "", ex.bbox, ex.origin, ex.rgba.shape[1], ex.rgba.shape[0],
                        category=category, source_model=ex.mask_model, label_model=ex.label_model,
                        mask_model=ex.mask_model, score=ex.score, category_score=cat_score,
                        category_source=cat_source)
                    asset.png_path = f"assets/{asset.id}.png"
                    self._write(asset.png_path, encode(ex.rgba, ".png"), written)
                    repo.set_asset_png_path(self.db, asset.id, asset.png_path)
                    repo.set_candidate_status(self.db, [rec.id], "confirmed")
                    if an is not None:
                        self._store_semantics(asset.id, an)
                    done[rec.idx] = asset
                # Confirmed objects are the strongest evidence for the photo's contexts.
                self._refresh_contexts(photo_id)
        except BaseException:
            self._remove(written)
            raise
        for ex in extracted:  # display thumbnails right away (the library grid loads only these)
            self.thumbs.asset(done[ex.candidate_id].id, self.path(done[ex.candidate_id].png_path), ex.rgba)
        return [done[i] for i in sorted(done)]

    def discard(self, photo_id: int, idxs: Iterable[int]) -> int:
        records = repo.list_candidates(self.db, photo_id, list(idxs), statuses=["pending"])
        with self.db.transaction():
            repo.set_candidate_status(self.db, [r.id for r in records], "discarded")
        return len(records)

    # ------------------------------------------------------------------------ assets

    def get_asset(self, asset_id: int) -> Asset:
        asset = repo.get_asset(self.db, asset_id)
        if asset is None:
            raise KeyError(f"asset {asset_id} not found")
        return asset

    def list_assets(self, **filters) -> List[Asset]:
        return repo.list_assets(self.db, **filters)

    def tags(self, asset_id: int) -> List[Tag]:
        return repo.get_tags(self.db, asset_id)

    def delete_asset(self, asset_id: int) -> bool:
        """Delete an asset; its candidate goes back to 'pending' so it can be confirmed again."""
        asset = repo.get_asset(self.db, asset_id)
        if asset is None:
            return False
        with self.db.transaction():
            repo.delete_asset(self.db, asset_id)  # cascades tags + embeddings (+ vec rows)
            if asset.candidate_id is not None:
                repo.set_candidate_status(self.db, [asset.candidate_id], "pending")
            self._refresh_contexts(asset.photo_id)
        self._remove([self.path(asset.png_path)])
        self.thumbs.drop_asset(asset_id)
        self._invalidate_index()
        return True

    # ------------------------------------------------------------------- user edits

    def set_category(self, asset_id: int, category: Optional[str]) -> Asset:
        """User-chosen category (source 'user'); the photo's rule contexts are recomputed with it."""
        asset = self.get_asset(asset_id)
        with self.db.transaction():
            repo.set_asset_category(self.db, asset_id, category, None, "user")
            self._refresh_contexts(asset.photo_id)
        return self.get_asset(asset_id)

    def add_tag(self, asset_id: int, kind: str, value: str) -> List[Tag]:
        self.get_asset(asset_id)
        with self.db.transaction():
            repo.set_tag_override(self.db, asset_id, kind, value, "add")
            if not any(t.kind == kind and t.value == value for t in repo.get_tags(self.db, asset_id)):
                repo.set_tags(self.db, asset_id, [*self._tags_of(asset_id, kind),
                                                  Tag(asset_id, kind, value, None, "user")], kind=kind)
        return repo.get_tags(self.db, asset_id)

    def remove_tag(self, asset_id: int, kind: str, value: str) -> List[Tag]:
        self.get_asset(asset_id)
        with self.db.transaction():
            repo.set_tag_override(self.db, asset_id, kind, value, "remove")
            repo.set_tags(self.db, asset_id, [t for t in self._tags_of(asset_id, kind) if t.value != value],
                          kind=kind)
        return repo.get_tags(self.db, asset_id)

    def _tags_of(self, asset_id: int, kind: str) -> List[Tag]:
        return [t for t in repo.get_tags(self.db, asset_id) if t.kind == kind]

    def _apply_tags(self, asset_id: int, kind: str, computed: List[Tag]) -> None:
        """Write automatically computed tags of one kind, honouring the user's add/remove edits."""
        overrides = repo.get_tag_overrides(self.db, asset_id, kind).get(kind, {})
        tags = [t for t in computed if overrides.get(t.value) != "remove"]
        present = {t.value for t in tags}
        tags += [Tag(asset_id, kind, v, None, "user") for v, action in overrides.items()
                 if action == "add" and v not in present]
        repo.set_tags(self.db, asset_id, tags, kind=kind)

    # ------------------------------------------------------------------ re-analysis

    def retag(self, photo_ids: Optional[Iterable[int]] = None,
              progress: Optional[Callable[[str], None]] = None) -> dict:
        """Recompute candidate analysis, photo contexts, asset categories/tags and vectors
        (after a vocabulary or model change). Masks and PNGs are reused, not recomputed."""
        analyzer = self.analyzer
        if analyzer is None:
            raise RuntimeError("retag needs the CLIP analyzer")
        ids = list(photo_ids) if photo_ids is not None else repo.photo_ids_with_status(self.db, "ready")
        n_assets = 0
        for pid in ids:
            photo = self.get_photo(pid)
            image = imread(self.path(photo.path))
            records = repo.list_candidates(self.db, pid)
            analyses = analyzer.analyze_candidates(image, [self.load_candidate_mask(r) for r in records],
                                                   [r.class_name for r in records], [r.score for r in records])
            scene = analyzer.scene_contexts(image)
            by_cid = {r.id: r for r in records}
            assets = repo.list_assets(self.db, photo_id=pid, limit=1_000_000)
            asset_an = []
            for a in assets:
                rec = by_cid.get(a.candidate_id)
                rgba = imread(self.path(a.png_path), cv2.IMREAD_UNCHANGED)
                asset_an.append(analyzer.analyze_asset(rgba, rec.class_name if rec else None,
                                                       rec.score if rec else None))
            with self.db.transaction():
                for r, an in zip(records, analyses):
                    repo.set_candidate_analysis(self.db, r.id, an.objectness, an.category, an.category_score)
                repo.set_photo_tags(self.db, pid, scene)
                for a, an in zip(assets, asset_an):
                    if a.category_source != "user":  # never overwrite a category the user chose
                        repo.set_asset_category(self.db, a.id, an.category.name, an.category.score,
                                                an.category.source)
                    self._store_semantics(a.id, an)
                contexts = self._refresh_contexts(pid)
            n_assets += len(assets)
            if progress:
                progress(f"photo {pid}: {len(records)} candidates, {len(assets)} assets, "
                         f"contexts={[f'{c[1]}({c[3]})' for c in contexts]}")
        return {"photos": len(ids), "assets": n_assets}

    def reembed(self, progress: Optional[Callable[[str], None]] = None) -> int:
        """Compute vectors for assets that have none in the current embedding space."""
        analyzer = self.analyzer
        if analyzer is None:
            raise RuntimeError("reembed needs the CLIP analyzer")
        missing = self.index.missing(analyzer.space)
        for aid in missing:
            rgba = imread(self.path(self.get_asset(aid).png_path), cv2.IMREAD_UNCHANGED)
            self.index.upsert(aid, analyzer.space, analyzer.image_encoder.encode_cutout(rgba))
            if progress:
                progress(f"asset {aid}")
        return len(missing)

    def stats(self) -> dict:
        s = repo.counts(self.db)
        s["photos_by_status"] = {st: len(repo.photo_ids_with_status(self.db, st))
                                 for st in ("processing", "ready", "failed")}
        s["vector_index"] = type(self.index).__name__
        s["embedding_spaces"] = {sp.key: n for sp, n in self.index.spaces()}
        return s

    # ----------------------------------------------------------------------- helpers

    def _store_semantics(self, asset_id: int, an) -> None:
        """Vector + attribute tags of one asset (contexts come from _refresh_contexts)."""
        self.index.upsert(asset_id, self.analyzer.space, an.vector)
        self._apply_tags(asset_id, "attribute",
                         [Tag(asset_id, kind, value, score, source) for kind, value, score, source in an.attributes])

    def _refresh_contexts(self, photo_id: int) -> List[tuple]:
        """Recompute the photo's rule contexts from stored candidate analysis + confirmed assets,
        keep its scene contexts, and copy the result onto every asset of the photo."""
        analyzer = self.analyzer
        if analyzer is None:
            return []
        records = repo.list_candidates(self.db, photo_id)
        assets = repo.list_assets(self.db, photo_id=photo_id, limit=1_000_000)
        scene = [t for t in repo.get_photo_tags(self.db, photo_id) if t[3] == "clip-scene"]
        tags = {t[1]: t for t in analyzer.rule_contexts(records, assets)}
        for t in scene:
            tags.setdefault(t[1], t)
        contexts = list(tags.values())
        repo.set_photo_tags(self.db, photo_id, contexts)
        for a in assets:
            self._apply_tags(a.id, "context", [Tag(a.id, *t) for t in contexts])
        return contexts

    def _write(self, relative: str, data: bytes, written: List[Path]) -> None:
        p = self.path(relative)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written.append(p)

    @staticmethod
    def _remove(paths: Iterable[Path]) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not remove %s: %s", p, exc)

    def _invalidate_index(self) -> None:
        if isinstance(self.index, NumpyIndex):
            self.index.invalidate()


def encode_mask(mask: RegionMask) -> bytes:
    """1-bit PNG of the mask crop; its offset is stored in the candidate row (bbox)."""
    return encode(mask.data.astype(np.uint8) * 255, ".png", [cv2.IMWRITE_PNG_BILEVEL, 1])


def decode_mask(data: bytes, bbox: Tuple[int, int, int, int]) -> RegionMask:
    img = decode(data, cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = bbox
    if img.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"mask size {img.shape} does not match bbox {bbox}")
    return RegionMask(img > 127, x0, y0)
