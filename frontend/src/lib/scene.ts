// Canvas scene model (mirrors backend/services/designs.py) and pure editing operations.
// Coordinates are design pixels; x, y of a layer is its centre. Layer order = paint order.
import type { PresetKey } from "../api";

export interface Background {
  type: "color" | "transparent" | "gradient";
  color: string;
  color2: string;
  angle: number;
}

export interface AssetLayer {
  id: string;
  type: "asset";
  asset_id: number;
  x: number;
  y: number;
  width: number; // height follows the asset's aspect ratio
  rotation: number;
  flip_x: boolean;
  opacity: number;
  shadow: boolean;
}

export interface TextLayer {
  id: string;
  type: "text";
  text: string;
  x: number;
  y: number;
  width: number; // wrapping box
  font_size: number;
  font_family: string;
  fill: string;
  font_weight: "normal" | "bold";
  align: "left" | "center" | "right";
  line_height: number;
  rotation: number;
  opacity: number;
}

export type Layer = AssetLayer | TextLayer;

export interface Scene {
  version: 1;
  background: Background;
  layers: Layer[];
}

export interface Size { w: number; h: number }
export type SizeOf = (assetId: number) => Size | undefined;

export const FONTS: { family: string; label: string }[] = [
  { family: "Microsoft JhengHei", label: "微軟正黑體" },
  { family: "PMingLiU", label: "新細明體" },
  { family: "Arial", label: "Arial" },
];

export const emptyScene = (): Scene => ({
  version: 1,
  background: { type: "color", color: "#ffffff", color2: "#e8eefc", angle: 90 },
  layers: [],
});

let counter = 0;
export const newId = (prefix = "l") => `${prefix}${Date.now().toString(36)}${(counter++).toString(36)}`;

/** Width that makes an item of `size` fit inside a bw x bh box (object-fit: contain). */
export function containWidth(size: Size, bw: number, bh: number): number {
  return Math.min(bw, bh * (size.w / size.h));
}

export function assetHeight(layer: AssetLayer, size: Size): number {
  return layer.width * (size.h / size.w);
}

export function newAssetLayer(assetId: number, size: Size, canvas: Size, at?: { x: number; y: number }): AssetLayer {
  return {
    id: newId("a"), type: "asset", asset_id: assetId,
    x: at?.x ?? canvas.w / 2, y: at?.y ?? canvas.h / 2,
    width: containWidth(size, canvas.w * 0.6, canvas.h * 0.6),
    rotation: 0, flip_x: false, opacity: 1, shadow: false,
  };
}

export function newTextLayer(canvas: Size, text = "輸入文字", over: Partial<TextLayer> = {}): TextLayer {
  return {
    id: newId("t"), type: "text", text, x: canvas.w / 2, y: canvas.h / 2, width: canvas.w * 0.8,
    font_size: Math.round(canvas.h * 0.06), font_family: FONTS[0].family, fill: "#1d1d1b",
    font_weight: "bold", align: "center", line_height: 1.2, rotation: 0, opacity: 1, ...over,
  };
}

/** Centre an asset layer in a box and make it fit (contain) with the given margin ratio. */
export function fitAsset(layer: AssetLayer, size: Size, box: { cx: number; cy: number; w: number; h: number }): AssetLayer {
  return { ...layer, x: box.cx, y: box.cy, width: containWidth(size, box.w, box.h), rotation: 0 };
}

/** Switching preset: positions follow the canvas proportionally, sizes scale uniformly
 *  (min ratio) so nothing gets stretched. */
export function rescaleScene(scene: Scene, from: Size, to: Size): Scene {
  const sx = to.w / from.w;
  const sy = to.h / from.h;
  const s = Math.min(sx, sy);
  return {
    ...scene,
    layers: scene.layers.map((l) =>
      l.type === "asset"
        ? { ...l, x: l.x * sx, y: l.y * sy, width: l.width * s }
        : { ...l, x: l.x * sx, y: l.y * sy, width: l.width * s, font_size: l.font_size * s }),
  };
}

export function updateLayer(scene: Scene, id: string, patch: Partial<Layer>): Scene {
  return { ...scene, layers: scene.layers.map((l) => (l.id === id ? ({ ...l, ...patch } as Layer) : l)) };
}

export function removeLayer(scene: Scene, id: string): Scene {
  return { ...scene, layers: scene.layers.filter((l) => l.id !== id) };
}

export function duplicateLayer(scene: Scene, id: string): { scene: Scene; id: string | null } {
  const i = scene.layers.findIndex((l) => l.id === id);
  if (i < 0) return { scene, id: null };
  const copy = { ...scene.layers[i], id: newId(scene.layers[i].type === "asset" ? "a" : "t"),
                 x: scene.layers[i].x + 24, y: scene.layers[i].y + 24 } as Layer;
  const layers = [...scene.layers];
  layers.splice(i + 1, 0, copy);
  return { scene: { ...scene, layers }, id: copy.id };
}

export type Order = "up" | "down" | "top" | "bottom";

export function reorder(scene: Scene, id: string, op: Order): Scene {
  const layers = [...scene.layers];
  const i = layers.findIndex((l) => l.id === id);
  if (i < 0) return scene;
  const [l] = layers.splice(i, 1);
  const j = op === "top" ? layers.length : op === "bottom" ? 0 : op === "up" ? Math.min(i + 1, layers.length) : Math.max(i - 1, 0);
  layers.splice(j, 0, l);
  return { ...scene, layers };
}

/** Linear-gradient endpoints for an angle (0 = left -> right, 90 = top -> bottom) covering the canvas. */
export function gradientPoints(angle: number, w: number, h: number) {
  const r = (angle * Math.PI) / 180;
  const dx = Math.cos(r);
  const dy = Math.sin(r);
  const half = (Math.abs(w * dx) + Math.abs(h * dy)) / 2;
  return { start: { x: w / 2 - dx * half, y: h / 2 - dy * half }, end: { x: w / 2 + dx * half, y: h / 2 + dy * half } };
}

// ------------------------------------------------------------------------------ history

export interface History { past: Scene[]; present: Scene; future: Scene[] }
export type HistoryAction = { type: "commit"; scene: Scene } | { type: "undo" } | { type: "redo" } | { type: "reset"; scene: Scene };
export const HISTORY_LIMIT = 50;

export function historyReducer(h: History, a: HistoryAction): History {
  switch (a.type) {
    case "commit":
      if (a.scene === h.present) return h;
      return { past: [...h.past, h.present].slice(-HISTORY_LIMIT), present: a.scene, future: [] };
    case "undo":
      if (!h.past.length) return h;
      return { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] };
    case "redo":
      if (!h.future.length) return h;
      return { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) };
    case "reset":
      return { past: [], present: a.scene, future: [] };
  }
}

// ---------------------------------------------------------------------------- templates

export interface Template {
  key: string;
  name: string;
  preset: PresetKey;
  hint: string;
  apply: (scene: Scene, canvas: Size, sizeOf: SizeOf) => Scene;
}

interface TextSpec { text: string; cx: number; cy: number; w: number; size: number; weight?: "normal" | "bold"; align?: TextLayer["align"]; fill?: string }

/** Re-lay out the scene: the first N asset layers go into `boxes`, the first M text layers get
 *  `texts` (missing text layers are created with placeholder text; existing text is kept). */
function layout(scene: Scene, sizeOf: SizeOf, canvas: Size, boxes: { cx: number; cy: number; w: number; h: number }[],
                texts: TextSpec[], background?: Partial<Background>): Scene {
  let a = 0;
  let t = 0;
  const layers: Layer[] = scene.layers.map((l) => {
    if (l.type === "asset" && a < boxes.length) {
      const size = sizeOf(l.asset_id);
      const box = boxes[a++];
      return size ? fitAsset(l, size, box) : { ...l, x: box.cx, y: box.cy };
    }
    if (l.type === "text" && t < texts.length) {
      const s = texts[t++];
      return { ...l, x: s.cx, y: s.cy, width: s.w, font_size: s.size, font_weight: s.weight ?? l.font_weight,
               align: s.align ?? "center", fill: s.fill ?? l.fill, rotation: 0 };
    }
    return l;
  });
  for (; t < texts.length; t++) {
    const s = texts[t];
    layers.push(newTextLayer(canvas, s.text, { x: s.cx, y: s.cy, width: s.w, font_size: s.size,
      font_weight: s.weight ?? "bold", align: s.align ?? "center", fill: s.fill ?? "#1d1d1b" }));
  }
  // Text layers the template has no slot for would stay where the previous layout left them
  // (e.g. a banner subtitle floating over the product): stack them under the template's last text.
  const last = texts[texts.length - 1];
  if (last) {
    let y = last.cy;
    let prevSize = last.size;
    let seen = 0;
    for (let i = 0; i < layers.length; i++) {
      const l = layers[i];
      if (l.type !== "text" || seen++ < texts.length) continue;
      const size = Math.min(l.font_size, last.size * 0.6);
      y = Math.min(y + prevSize * 0.75 + size * 0.9, canvas.h - size);
      layers[i] = { ...l, x: last.cx, y, width: last.w, font_size: size, align: last.align ?? "center", rotation: 0 };
      prevSize = size;
    }
  }
  return { ...scene, background: { ...scene.background, ...background }, layers };
}

export const TEMPLATES: Template[] = [
  {
    key: "product-center", name: "置中商品", preset: "product", hint: "第 1 個素材置中，四周留白 10%",
    apply: (s, c, size) => layout(s, size, c, [{ cx: c.w / 2, cy: c.h / 2, w: c.w * 0.8, h: c.h * 0.8 }], [],
      { type: "color", color: "#ffffff" }),
  },
  {
    key: "product-caption", name: "商品＋底部文字", preset: "product", hint: "素材在上，商品名稱在下",
    apply: (s, c, size) => layout(s, size, c, [{ cx: c.w / 2, cy: c.h * 0.42, w: c.w * 0.8, h: c.h * 0.62 }],
      [{ text: "商品名稱", cx: c.w / 2, cy: c.h * 0.87, w: c.w * 0.86, size: Math.round(c.h * 0.07) }],
      { type: "color", color: "#ffffff" }),
  },
  {
    key: "ig-hero", name: "大圖＋標題", preset: "ig", hint: "素材在上，標題與副標在下，暖色漸層底",
    apply: (s, c, size) => layout(s, size, c, [{ cx: c.w / 2, cy: c.h * 0.4, w: c.w * 0.8, h: c.h * 0.56 }], [
      { text: "標題文字", cx: c.w / 2, cy: c.h * 0.77, w: c.w * 0.86, size: Math.round(c.h * 0.055) },
      { text: "副標或說明文字", cx: c.w / 2, cy: c.h * 0.85, w: c.w * 0.86, size: Math.round(c.h * 0.03), weight: "normal", fill: "#555555" },
    ], { type: "gradient", color: "#f7f1ea", color2: "#eadfce", angle: 90 }),
  },
  {
    key: "ig-pair", name: "雙素材", preset: "ig", hint: "標題在上，兩個素材左右並排",
    apply: (s, c, size) => layout(s, size, c, [
      { cx: c.w * 0.28, cy: c.h * 0.54, w: c.w * 0.44, h: c.h * 0.56 },
      { cx: c.w * 0.72, cy: c.h * 0.54, w: c.w * 0.44, h: c.h * 0.56 },
    ], [{ text: "標題文字", cx: c.w / 2, cy: c.h * 0.13, w: c.w * 0.86, size: Math.round(c.h * 0.055) }],
    { type: "color", color: "#ffffff" }),
  },
  {
    key: "banner-text-left", name: "左文右圖", preset: "banner", hint: "標題與副標靠左，素材在右",
    apply: (s, c, size) => layout(s, size, c, [{ cx: c.w * 0.72, cy: c.h / 2, w: c.w * 0.44, h: c.h * 0.82 }], [
      { text: "標題文字", cx: c.w * 0.27, cy: c.h * 0.42, w: c.w * 0.42, size: Math.round(c.h * 0.11), align: "left" },
      { text: "副標或說明文字", cx: c.w * 0.27, cy: c.h * 0.6, w: c.w * 0.42, size: Math.round(c.h * 0.05), weight: "normal", align: "left", fill: "#555555" },
    ], { type: "gradient", color: "#ffffff", color2: "#e8eefc", angle: 0 }),
  },
  {
    key: "banner-text-right", name: "左圖右文", preset: "banner", hint: "素材在左，標題與副標靠右",
    apply: (s, c, size) => layout(s, size, c, [{ cx: c.w * 0.28, cy: c.h / 2, w: c.w * 0.44, h: c.h * 0.82 }], [
      { text: "標題文字", cx: c.w * 0.73, cy: c.h * 0.42, w: c.w * 0.42, size: Math.round(c.h * 0.11), align: "left" },
      { text: "副標或說明文字", cx: c.w * 0.73, cy: c.h * 0.6, w: c.w * 0.42, size: Math.round(c.h * 0.05), weight: "normal", align: "left", fill: "#555555" },
    ], { type: "gradient", color: "#e8eefc", color2: "#ffffff", angle: 0 }),
  },
];
