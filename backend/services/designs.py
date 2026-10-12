"""Canvas designs and their exports.

A design is a preset size plus a scene (background + ordered layers) in design pixels. The
browser renders the scene with Konva and uploads the exported image; the backend stores
it after checking that its pixel size is exactly the design size (WYSIWYG: the browser's
rendering, fonts included, is the product).

Files: data/exports/{export_id}.png|jpg, data/thumbs/designs/{design_id}.jpg (list preview)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .. import config
from ..db import Database
from ..db.models import utcnow
from ..imageio import decode, encode
from .thumbs import fit_scale, resize

DESIGN_THUMB_SIDE = 480
COLOR = r"^#[0-9a-fA-F]{6}$"


# ------------------------------------------------------------------------- scene schema

class Background(BaseModel):
    type: Literal["color", "transparent", "gradient"] = "color"
    color: str = Field("#ffffff", pattern=COLOR)
    color2: str = Field("#e8eefc", pattern=COLOR)
    angle: float = 90.0  # gradient direction in degrees (0 = left -> right, 90 = top -> bottom)


class AssetLayer(BaseModel):
    """x, y = centre in design pixels; width in design pixels (height follows the asset's aspect)."""
    id: str = Field(min_length=1, max_length=64)
    type: Literal["asset"]
    asset_id: int
    x: float
    y: float
    width: float = Field(gt=0)
    rotation: float = 0.0
    flip_x: bool = False
    opacity: float = Field(1.0, ge=0, le=1)
    shadow: bool = False


class TextLayer(BaseModel):
    """x, y = centre in design pixels; width = wrapping box width."""
    id: str = Field(min_length=1, max_length=64)
    type: Literal["text"]
    text: str = Field(max_length=500)
    x: float
    y: float
    width: float = Field(gt=0)
    font_size: float = Field(gt=0, le=1000)
    font_family: str = Field("Microsoft JhengHei", max_length=100)
    fill: str = Field("#1d1d1b", pattern=COLOR)
    font_weight: Literal["normal", "bold"] = "bold"
    align: Literal["left", "center", "right"] = "center"
    line_height: float = Field(1.2, gt=0, le=5)
    rotation: float = 0.0
    opacity: float = Field(1.0, ge=0, le=1)


Layer = Annotated[Union[AssetLayer, TextLayer], Field(discriminator="type")]


class Scene(BaseModel):
    version: Literal[1] = 1
    background: Background = Background()
    layers: List[Layer] = Field(default_factory=list, max_length=100)


# ------------------------------------------------------------------------------ records

@dataclass
class Design:
    id: int
    name: str
    preset: str
    width: int
    height: int
    scene: Scene
    created_at: str
    updated_at: str


@dataclass
class Export:
    id: int
    design_id: int
    path: str
    format: str
    width: int
    height: int
    bytes: int
    created_at: str


def preset_size(preset: str):
    if preset not in config.CANVAS_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(config.CANVAS_PRESETS)}")
    _, w, h = config.CANVAS_PRESETS[preset]
    return w, h


class DesignStore:
    def __init__(self, db: Database, data_dir: Path):
        self.db = db
        self.data_dir = Path(data_dir)

    # -- paths
    def export_file(self, export: Export) -> Path:
        return self.data_dir / export.path

    def thumb_path(self, design_id: int) -> Path:
        return self.data_dir / "thumbs" / "designs" / f"{design_id}.jpg"

    # -- designs
    @staticmethod
    def _design(r) -> Design:
        return Design(r["id"], r["name"], r["preset"], r["width"], r["height"],
                      Scene.model_validate_json(r["scene"]), r["created_at"], r["updated_at"])

    def create(self, preset: str, name: Optional[str] = None, scene: Optional[Scene] = None) -> Design:
        w, h = preset_size(preset)
        now = utcnow()
        scene = scene or Scene()
        # Default name in local time (timestamps in the DB stay UTC).
        name = name or f"{config.CANVAS_PRESETS[preset][0]} {datetime.now().strftime('%m-%d %H:%M')}"
        with self.db.transaction():
            cur = self.db.conn.execute(
                "INSERT INTO designs (name, preset, width, height, scene, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)", (name, preset, w, h, scene.model_dump_json(), now, now))
        return self.get(cur.lastrowid)

    def get(self, design_id: int) -> Design:
        r = self.db.conn.execute("SELECT * FROM designs WHERE id = ?", (design_id,)).fetchone()
        if r is None:
            raise KeyError(f"design {design_id} not found")
        return self._design(r)

    def list(self, limit: int = 200) -> List[Design]:
        rows = self.db.conn.execute("SELECT * FROM designs ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,))
        return [self._design(r) for r in rows]

    def update(self, design_id: int, scene: Scene, name: Optional[str] = None,
               preset: Optional[str] = None) -> Design:
        current = self.get(design_id)
        preset = preset or current.preset
        w, h = preset_size(preset)
        with self.db.transaction():
            self.db.conn.execute(
                "UPDATE designs SET name = ?, preset = ?, width = ?, height = ?, scene = ?, updated_at = ?"
                " WHERE id = ?", (name or current.name, preset, w, h, scene.model_dump_json(), utcnow(), design_id))
        return self.get(design_id)

    def delete(self, design_id: int) -> bool:
        exports = self.exports(design_id)
        with self.db.transaction():
            n = self.db.conn.execute("DELETE FROM designs WHERE id = ?", (design_id,)).rowcount
        for e in exports:
            self.export_file(e).unlink(missing_ok=True)
        self.thumb_path(design_id).unlink(missing_ok=True)
        return n > 0

    # -- exports
    @staticmethod
    def _export(r) -> Export:
        return Export(r["id"], r["design_id"], r["path"], r["format"], r["width"], r["height"], r["bytes"],
                      r["created_at"])

    def exports(self, design_id: Optional[int] = None) -> List[Export]:
        sql, params = "SELECT * FROM exports", []
        if design_id is not None:
            sql += " WHERE design_id = ?"
            params.append(design_id)
        return [self._export(r) for r in self.db.conn.execute(sql + " ORDER BY id DESC", params)]

    def get_export(self, export_id: int) -> Export:
        r = self.db.conn.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
        if r is None:
            raise KeyError(f"export {export_id} not found")
        return self._export(r)

    def save_export(self, design_id: int, data: bytes, fmt: str) -> Export:
        """Store a browser-rendered export after checking it is exactly the design's size."""
        design = self.get(design_id)
        if fmt not in ("png", "jpeg"):
            raise ValueError("format must be png or jpeg")
        img = decode(data, cv2.IMREAD_UNCHANGED)  # ValueError if not an image
        h, w = img.shape[:2]
        if (w, h) != (design.width, design.height):
            raise ValueError(f"export is {w}x{h}, design {design.id} is {design.width}x{design.height}")
        with self.db.transaction():
            cur = self.db.conn.execute(
                "INSERT INTO exports (design_id, path, format, width, height, bytes, created_at)"
                " VALUES (?, '', ?, ?, ?, ?, ?)", (design_id, fmt, w, h, len(data), utcnow()))
            eid = cur.lastrowid
            rel = f"exports/{eid}.{'png' if fmt == 'png' else 'jpg'}"
            path = self.data_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.db.conn.execute("UPDATE exports SET path = ? WHERE id = ?", (rel, eid))
        self._write_thumb(design_id, img)
        return self.get_export(eid)

    def _write_thumb(self, design_id: int, img: np.ndarray) -> None:
        if img.ndim == 3 and img.shape[2] == 4:  # flatten transparency onto white for the preview
            a = img[..., 3:4].astype(np.float32) / 255
            img = (img[..., :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        small = resize(img, fit_scale(img.shape, DESIGN_THUMB_SIDE))
        path = self.thumb_path(design_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode(small, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]))

    def asset_ids(self, design: Design) -> List[int]:
        return sorted({layer.asset_id for layer in design.scene.layers if layer.type == "asset"})


def scene_json(scene: Scene) -> dict:
    return json.loads(scene.model_dump_json())
