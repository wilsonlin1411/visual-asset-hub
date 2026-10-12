"""HTTP API (FastAPI). Run:  python -m backend.api   (http://127.0.0.1:8000/api/docs)

All endpoints live under /api; when frontend/dist exists it is served at / (single-port demo).

    POST   /api/photos                          upload; processed in the background
    GET    /api/photos, /api/photos/{id}        status, counts, photo-level contexts
    DELETE /api/photos/{id}
    GET    /api/photos/{id}/image | /display    original | display size (long side 1600)
    GET    /api/photos/{id}/candidates          low-objectness hidden unless include_hidden
    GET    /api/photos/{id}/candidates/{idx}/mask | /preview | /overlay
    POST   /api/assets/confirm                  {photo_id, idxs?}
    POST   /api/candidates/discard              {photo_id, idxs}
    GET    /api/assets, /api/assets/{id}        filter by category / tag / photo
    PATCH  /api/assets/{id}                     {category}   (user-set category)
    POST   /api/assets/{id}/tags                {kind, value} (user-added tag)
    DELETE /api/assets/{id}/tags?kind=&value=   (user-removed tag)
    GET    /api/assets/{id}/png | /thumb
    DELETE /api/assets/{id}
    GET    /api/search?q=
    GET    /api/vocabulary, /api/stats
    GET    /api/assets/{id}/medium              cut-out, longer side 2048 (canvas)
    GET    /api/presets                         canvas sizes
    GET/POST /api/designs, GET/PUT/DELETE /api/designs/{id}, GET /api/designs/{id}/thumb
    POST   /api/designs/{id}/export             browser-rendered image (must match the design size)
    GET    /api/exports?design_id=, /api/exports/{id}/file
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from ..db import Asset, CandidateRecord, Photo, Tag, repo
from ..services.designs import Design, DesignStore, Export, Scene, scene_json
from ..services.library import AssetLibrary, embedding_space
from ..services.queue import PhotoQueue
from ..services.search import SearchService
from ..services.vocab import Vocabulary, load_vocabulary

log = logging.getLogger(__name__)

FRONTEND_DIST = config.ROOT_DIR / "frontend" / "dist"
IMMUTABLE = {"Cache-Control": "public, max-age=86400"}


@lru_cache(maxsize=1)
def vocabulary() -> Vocabulary:
    return load_vocabulary()


# ---------------------------------------------------------------------------- schemas

class TagOut(BaseModel):
    kind: str
    value: str
    score: Optional[float]
    source: str


class PhotoOut(BaseModel):
    id: int
    original_name: str
    width: int
    height: int
    status: str
    error: Optional[str]
    created_at: str
    image_url: str
    display_url: str
    contexts: List[TagOut] = []
    counts: Dict[str, int] = {}
    queue_position: Optional[int] = None


class CandidateOut(BaseModel):
    idx: int
    yolo_label: Optional[str]
    category: Optional[str]
    category_score: Optional[float]
    score: float
    objectness: Optional[float]
    hidden: bool
    status: str
    bbox: List[int]
    area: int
    label_model: Optional[str]
    mask_model: str
    mask_url: str
    preview_url: str
    overlay_url: str


class CandidatesOut(BaseModel):
    photo_id: int
    photo_status: str
    objectness_threshold: float
    hidden_count: int
    candidates: List[CandidateOut]


class AssetOut(BaseModel):
    id: int
    photo_id: int
    category: Optional[str]
    category_score: Optional[float]
    category_source: Optional[str]
    bbox: List[int]
    origin: List[int]
    width: int
    height: int
    label_model: Optional[str]
    mask_model: str
    created_at: str
    png_url: str
    thumb_url: str
    tags: List[TagOut] = []


class ConfirmIn(BaseModel):
    photo_id: int
    idxs: Optional[List[int]] = None


class DiscardIn(BaseModel):
    photo_id: int
    idxs: List[int]


class CategoryIn(BaseModel):
    category: Optional[str]


class TagIn(BaseModel):
    kind: str
    value: str


class UploadOut(BaseModel):
    photo: PhotoOut
    created: bool


class SearchHitOut(BaseModel):
    asset: AssetOut
    score: float
    cosine: float
    reasons: List[str]


class SearchOut(BaseModel):
    query: str
    categories: List[str]
    attributes: List[str]
    contexts: List[str]
    excluded_categories: List[str]
    hits: List[SearchHitOut]


class PresetOut(BaseModel):
    key: str
    name: str
    width: int
    height: int


class DesignSummary(BaseModel):
    id: int
    name: str
    preset: str
    width: int
    height: int
    created_at: str
    updated_at: str
    layer_count: int
    thumb_url: Optional[str]


class DesignAssetOut(BaseModel):
    width: int
    height: int
    category: Optional[str]
    medium_url: str
    thumb_url: str


class DesignOut(DesignSummary):
    scene: dict
    assets: Dict[int, DesignAssetOut]   # assets referenced by the scene
    missing_assets: List[int]          # referenced but deleted -> shown as placeholders


class DesignCreateIn(BaseModel):
    preset: str
    name: Optional[str] = None
    scene: Optional[Scene] = None


class DesignUpdateIn(BaseModel):
    scene: Scene
    name: Optional[str] = None
    preset: Optional[str] = None


class ExportOut(BaseModel):
    id: int
    design_id: int
    format: str
    width: int
    height: int
    bytes: int
    created_at: str
    url: str


class VocabularyOut(BaseModel):
    categories: List[Dict[str, str]]          # {name, group}
    contexts: List[str]
    attributes: Dict[str, List[str]]          # color / material / style -> names
    excluded_categories: List[str]


def tag_out(t) -> TagOut:
    if isinstance(t, Tag):
        return TagOut(kind=t.kind, value=t.value, score=t.score, source=t.source)
    kind, value, score, source = t
    return TagOut(kind=kind, value=value, score=score, source=source)


def asset_out(a: Asset, tags=()) -> AssetOut:
    return AssetOut(id=a.id, photo_id=a.photo_id, category=a.category, category_score=a.category_score,
                    category_source=a.category_source, bbox=list(a.bbox), origin=list(a.origin), width=a.width,
                    height=a.height, label_model=a.label_model, mask_model=a.mask_model, created_at=a.created_at,
                    png_url=f"/api/assets/{a.id}/png", thumb_url=f"/api/assets/{a.id}/thumb",
                    tags=[tag_out(t) for t in tags])


def candidate_out(c: CandidateRecord, threshold: float) -> CandidateOut:
    base = f"/api/photos/{c.photo_id}/candidates/{c.idx}"
    return CandidateOut(idx=c.idx, yolo_label=c.class_name, category=c.category, category_score=c.category_score,
                        score=c.score, objectness=c.objectness, hidden=c.hidden(threshold), status=c.status,
                        bbox=list(c.bbox), area=c.area, label_model=c.label_model, mask_model=c.mask_model,
                        mask_url=f"{base}/mask", preview_url=f"{base}/preview", overlay_url=f"{base}/overlay")


def candidate_order(c: CandidateRecord):
    """YOLO-labelled first (by score), then the rest by objectness, best first."""
    if c.class_name is not None:
        return (0, -c.score)
    return (1, -(c.objectness if c.objectness is not None else c.score))


# -------------------------------------------------------------------------------- app

def create_app(library: Optional[AssetLibrary] = None, search_factory=None, start_worker: bool = True,
               serve_frontend: bool = True) -> FastAPI:
    lib = library or AssetLibrary()
    worker = PhotoQueue(lib)
    search_holder: dict = {}
    search_lock = threading.Lock()

    def get_search() -> SearchService:
        with search_lock:
            if "svc" not in search_holder:
                if search_factory is not None:
                    search_holder["svc"] = search_factory(lib)
                else:
                    from ..inference import create_backend
                    from ..services.embedding import query_encoder
                    search_holder["svc"] = SearchService(lib, query_encoder(create_backend("onnx", "cpu")),
                                                         vocabulary(), embedding_space())
            return search_holder["svc"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            worker.start()
        yield
        worker.stop()
        if library is None:
            lib.close()

    app = FastAPI(title="Visual Asset Hub", version="0.5", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.library = lib
    app.state.worker = worker
    api = APIRouter(prefix="/api")

    def photo_or_404(photo_id: int) -> Photo:
        try:
            return lib.get_photo(photo_id)
        except KeyError:
            raise HTTPException(404, f"photo {photo_id} not found")

    def asset_or_404(asset_id: int) -> Asset:
        try:
            return lib.get_asset(asset_id)
        except KeyError:
            raise HTTPException(404, f"asset {asset_id} not found")

    def candidate_or_404(photo_id: int, idx: int) -> CandidateRecord:
        photo_or_404(photo_id)
        recs = repo.list_candidates(lib.db, photo_id, [idx])
        if not recs:
            raise HTTPException(404, f"photo {photo_id} has no candidate {idx}")
        return recs[0]

    def photo_out(p: Photo, contexts=(), counts=None) -> PhotoOut:
        return PhotoOut(id=p.id, original_name=p.original_name, width=p.width, height=p.height, status=p.status,
                        error=p.error, created_at=p.created_at, image_url=f"/api/photos/{p.id}/image",
                        display_url=f"/api/photos/{p.id}/display", contexts=[tag_out(t) for t in contexts],
                        counts=counts or {}, queue_position=worker.position(p.id))

    def check_tag(kind: str, value: str) -> None:
        v = vocabulary()
        allowed = ({c.name for c in v.contexts} if kind == "context" else
                   {e.name for e in (*v.colors, *v.materials, *v.styles)} if kind == "attribute" else None)
        if allowed is None:
            raise HTTPException(400, "kind must be 'context' or 'attribute'")
        if value not in allowed:
            raise HTTPException(400, f"{value!r} is not a {kind} in the vocabulary")

    # ---- photos

    @api.post("/photos", response_model=UploadOut, status_code=202)
    def upload_photo(file: UploadFile = File(...)):
        data = file.file.read()
        try:
            photo, created = lib.register_photo(data, file.filename or "upload.jpg")
        except ValueError:
            raise HTTPException(400, "file is not a readable image")
        if photo.status == "processing":
            worker.submit(photo.id)
        return UploadOut(photo=photo_out(photo), created=created)

    @api.get("/photos", response_model=List[PhotoOut])
    def list_photos(limit: int = 100, offset: int = 0):
        photos = lib.list_photos(limit, offset)
        counts = lib.photo_counts([p.id for p in photos])
        return [photo_out(p, counts=counts[p.id]) for p in photos]

    @api.get("/photos/{photo_id}", response_model=PhotoOut)
    def get_photo(photo_id: int):
        p = photo_or_404(photo_id)
        return photo_out(p, lib.photo_contexts(photo_id), lib.photo_counts([photo_id])[photo_id])

    @api.delete("/photos/{photo_id}")
    def delete_photo(photo_id: int):
        photo = photo_or_404(photo_id)
        if photo.status == "processing":
            raise HTTPException(409, "photo is still being processed")
        lib.delete_photo(photo_id)
        return {"deleted": photo_id}

    @api.get("/photos/{photo_id}/image")
    def photo_image(photo_id: int):
        return FileResponse(lib.path(photo_or_404(photo_id).path), headers=IMMUTABLE)

    @api.get("/photos/{photo_id}/display")
    def photo_display(photo_id: int):
        p = photo_or_404(photo_id)
        return FileResponse(lib.thumbs.photo(photo_id, lib.path(p.path)), media_type="image/jpeg", headers=IMMUTABLE)

    @api.get("/photos/{photo_id}/candidates", response_model=CandidatesOut)
    def list_candidates(photo_id: int, include_hidden: bool = False):
        photo = photo_or_404(photo_id)
        threshold = lib.objectness_threshold if photo.status == "ready" else 0.0
        records = lib.candidates(photo_id)
        visible = [c for c in records if include_hidden or not c.hidden(threshold)]
        visible.sort(key=candidate_order)
        return CandidatesOut(photo_id=photo_id, photo_status=photo.status, objectness_threshold=threshold,
                             hidden_count=sum(c.hidden(threshold) for c in records),
                             candidates=[candidate_out(c, threshold) for c in visible])

    @api.get("/photos/{photo_id}/candidates/{idx}/mask")
    def candidate_mask(photo_id: int, idx: int):
        return FileResponse(lib.path(candidate_or_404(photo_id, idx).mask_path), media_type="image/png")

    @api.get("/photos/{photo_id}/candidates/{idx}/preview")
    def candidate_preview(photo_id: int, idx: int):
        rec = candidate_or_404(photo_id, idx)
        path = lib.thumbs.candidate(photo_id, idx, lib.path(lib.get_photo(photo_id).path), lib.load_candidate_mask(rec))
        return FileResponse(path, media_type="image/webp", headers=IMMUTABLE)

    @api.get("/photos/{photo_id}/candidates/{idx}/overlay")
    def candidate_overlay(photo_id: int, idx: int):
        rec = candidate_or_404(photo_id, idx)
        p = lib.get_photo(photo_id)
        path = lib.thumbs.overlay(photo_id, idx, (p.height, p.width), lib.load_candidate_mask(rec))
        return FileResponse(path, media_type="image/png", headers=IMMUTABLE)

    # ---- candidates -> assets

    @api.post("/assets/confirm", response_model=List[AssetOut])
    def confirm(body: ConfirmIn):
        photo_or_404(body.photo_id)
        try:
            assets = lib.confirm(body.photo_id, body.idxs)
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0]))
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        tags = repo.tags_for_assets(lib.db, [a.id for a in assets])
        return [asset_out(a, tags[a.id]) for a in assets]

    @api.post("/candidates/discard")
    def discard(body: DiscardIn):
        photo_or_404(body.photo_id)
        return {"discarded": lib.discard(body.photo_id, body.idxs)}

    # ---- assets

    @api.get("/assets", response_model=List[AssetOut])
    def list_assets(category: Optional[str] = None, tag: Optional[str] = None, photo_id: Optional[int] = None,
                    limit: int = Query(100, le=1000), offset: int = 0):
        assets = lib.list_assets(category=category, tag=tag, photo_id=photo_id, limit=limit, offset=offset)
        tags = repo.tags_for_assets(lib.db, [a.id for a in assets])
        return [asset_out(a, tags[a.id]) for a in assets]

    @api.get("/assets/{asset_id}", response_model=AssetOut)
    def get_asset(asset_id: int):
        return asset_out(asset_or_404(asset_id), lib.tags(asset_id))

    @api.patch("/assets/{asset_id}", response_model=AssetOut)
    def update_asset(asset_id: int, body: CategoryIn):
        asset_or_404(asset_id)
        if body.category is not None and vocabulary().category(body.category) is None:
            raise HTTPException(400, f"{body.category!r} is not a category in the vocabulary")
        return asset_out(lib.set_category(asset_id, body.category), lib.tags(asset_id))

    @api.post("/assets/{asset_id}/tags", response_model=AssetOut)
    def add_tag(asset_id: int, body: TagIn):
        asset_or_404(asset_id)
        check_tag(body.kind, body.value)
        tags = lib.add_tag(asset_id, body.kind, body.value)
        return asset_out(lib.get_asset(asset_id), tags)

    @api.delete("/assets/{asset_id}/tags", response_model=AssetOut)
    def remove_tag(asset_id: int, kind: str, value: str):
        asset_or_404(asset_id)
        check_tag(kind, value)
        tags = lib.remove_tag(asset_id, kind, value)
        return asset_out(lib.get_asset(asset_id), tags)

    @api.get("/assets/{asset_id}/png")
    def asset_png(asset_id: int):
        a = asset_or_404(asset_id)
        return FileResponse(lib.path(a.png_path), media_type="image/png",
                            filename=f"asset-{a.id}-{a.category or 'object'}.png")

    @api.get("/assets/{asset_id}/thumb")
    def asset_thumb(asset_id: int):
        a = asset_or_404(asset_id)
        return FileResponse(lib.thumbs.asset(a.id, lib.path(a.png_path)), media_type="image/webp")

    @api.delete("/assets/{asset_id}")
    def delete_asset(asset_id: int):
        asset_or_404(asset_id)
        lib.delete_asset(asset_id)
        return {"deleted": asset_id}

    # ---- search, vocabulary, stats

    @api.get("/search", response_model=SearchOut)
    def search(q: str = Query(..., min_length=1), k: int = Query(20, ge=1, le=200),
               category: Optional[str] = None, tag: Optional[str] = None, include_people: bool = False):
        res = get_search().search(q, k=k, category=category, tag=tag, include_excluded=include_people)
        return SearchOut(query=q, categories=res.query.categories, attributes=res.query.attributes,
                         contexts=res.query.contexts, excluded_categories=res.excluded_categories,
                         hits=[SearchHitOut(asset=asset_out(h.asset, h.tags), score=h.score, cosine=h.cosine,
                                            reasons=h.reasons) for h in res.hits])

    @api.get("/vocabulary", response_model=VocabularyOut)
    def get_vocabulary():
        v = vocabulary()
        return VocabularyOut(
            categories=[{"name": c.name, "group": c.group} for c in v.categories],
            contexts=[c.name for c in v.contexts],
            attributes={"color": [c.name for c in v.colors], "material": [m.name for m in v.materials],
                        "style": [s.name for s in v.styles]},
            excluded_categories=v.excluded_categories)

    @api.get("/stats")
    def stats():
        s = lib.stats()
        s["queue_current"] = worker.current
        return s

    # ---- canvas designs and exports (stage 6)

    designs = DesignStore(lib.db, lib.data_dir)
    app.state.designs = designs

    def design_or_404(design_id: int) -> Design:
        try:
            return designs.get(design_id)
        except KeyError:
            raise HTTPException(404, f"design {design_id} not found")

    def design_summary(d: Design) -> DesignSummary:
        return DesignSummary(id=d.id, name=d.name, preset=d.preset, width=d.width, height=d.height,
                             created_at=d.created_at, updated_at=d.updated_at, layer_count=len(d.scene.layers),
                             thumb_url=f"/api/designs/{d.id}/thumb" if designs.thumb_path(d.id).is_file() else None)

    def design_out(d: Design) -> DesignOut:
        ids = designs.asset_ids(d)
        found = repo.get_assets(lib.db, ids)
        return DesignOut(**design_summary(d).model_dump(), scene=scene_json(d.scene),
                         assets={a.id: DesignAssetOut(width=a.width, height=a.height, category=a.category,
                                                      medium_url=f"/api/assets/{a.id}/medium",
                                                      thumb_url=f"/api/assets/{a.id}/thumb")
                                 for a in found.values()},
                         missing_assets=[i for i in ids if i not in found])

    def export_out(e: Export) -> ExportOut:
        return ExportOut(id=e.id, design_id=e.design_id, format=e.format, width=e.width, height=e.height,
                         bytes=e.bytes, created_at=e.created_at, url=f"/api/exports/{e.id}/file")

    @api.get("/presets", response_model=List[PresetOut])
    def presets():
        return [PresetOut(key=k, name=n, width=w, height=h) for k, (n, w, h) in config.CANVAS_PRESETS.items()]

    @api.get("/designs", response_model=List[DesignSummary])
    def list_designs():
        return [design_summary(d) for d in designs.list()]

    @api.post("/designs", response_model=DesignOut, status_code=201)
    def create_design(body: DesignCreateIn):
        try:
            return design_out(designs.create(body.preset, body.name, body.scene))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @api.get("/designs/{design_id}", response_model=DesignOut)
    def get_design(design_id: int):
        return design_out(design_or_404(design_id))

    @api.put("/designs/{design_id}", response_model=DesignOut)
    def update_design(design_id: int, body: DesignUpdateIn):
        design_or_404(design_id)
        try:
            return design_out(designs.update(design_id, body.scene, body.name, body.preset))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @api.delete("/designs/{design_id}")
    def delete_design(design_id: int):
        design_or_404(design_id)
        designs.delete(design_id)
        return {"deleted": design_id}

    @api.get("/designs/{design_id}/thumb")
    def design_thumb(design_id: int):
        design_or_404(design_id)
        path = designs.thumb_path(design_id)
        if not path.is_file():
            raise HTTPException(404, "design has not been exported yet")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @api.post("/designs/{design_id}/export", response_model=ExportOut, status_code=201)
    def export_design(design_id: int, file: UploadFile = File(...), format: str = Form("png")):
        design_or_404(design_id)
        try:
            return export_out(designs.save_export(design_id, file.file.read(), format))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @api.get("/exports", response_model=List[ExportOut])
    def list_exports(design_id: Optional[int] = None):
        return [export_out(e) for e in designs.exports(design_id)]

    @api.get("/exports/{export_id}/file")
    def export_file(export_id: int):
        try:
            e = designs.get_export(export_id)
        except KeyError:
            raise HTTPException(404, f"export {export_id} not found")
        d = designs.get(e.design_id)
        ext = "png" if e.format == "png" else "jpg"
        return FileResponse(designs.export_file(e), media_type=f"image/{e.format}",
                            filename=f"{d.preset}-{e.width}x{e.height}-{e.id}.{ext}")

    @api.get("/assets/{asset_id}/medium")
    def asset_medium(asset_id: int):
        a = asset_or_404(asset_id)
        return FileResponse(lib.thumbs.medium(a.id, lib.path(a.png_path)), media_type="image/png", headers=IMMUTABLE)

    app.include_router(api)

    # ---- built frontend (single-port demo); unknown non-API paths fall back to index.html (SPA routes)
    if serve_frontend and (FRONTEND_DIST / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            file = (FRONTEND_DIST / path).resolve()
            if path and file.is_file() and FRONTEND_DIST.resolve() in file.parents:
                return FileResponse(file)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app
