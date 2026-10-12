import { describe, expect, it } from "vitest";
import {
  TEMPLATES, assetHeight, containWidth, duplicateLayer, emptyScene, gradientPoints, historyReducer,
  newAssetLayer, newTextLayer, rescaleScene, reorder, type AssetLayer, type Scene, type Size,
} from "./scene";

const PRODUCT: Size = { w: 800, h: 800 };
const BANNER: Size = { w: 1640, h: 856 };
const IG: Size = { w: 1080, h: 1350 };
const sizes: Record<number, Size> = { 1: { w: 400, h: 200 }, 2: { w: 100, h: 300 } };
const sizeOf = (id: number) => sizes[id];

const withLayers = (...layers: Scene["layers"]): Scene => ({ ...emptyScene(), layers });

describe("placement", () => {
  it("contain-fits an asset", () => {
    expect(containWidth({ w: 400, h: 200 }, 100, 100)).toBe(100);   // limited by width
    expect(containWidth({ w: 100, h: 300 }, 100, 90)).toBe(30);     // limited by height
  });

  it("new assets are centred at 60% of the canvas", () => {
    const l = newAssetLayer(2, sizes[2], PRODUCT);
    expect([l.x, l.y]).toEqual([400, 400]);
    expect(assetHeight(l, sizes[2])).toBeCloseTo(480);
    expect(newAssetLayer(1, sizes[1], PRODUCT, { x: 10, y: 20 })).toMatchObject({ x: 10, y: 20, width: 480 });
  });
});

describe("rescale on preset change", () => {
  it("moves proportionally and scales sizes uniformly", () => {
    const a = { ...newAssetLayer(1, sizes[1], PRODUCT), x: 400, y: 400, width: 400 };
    const t = newTextLayer(PRODUCT, "hi", { x: 200, y: 600, width: 400, font_size: 40 });
    const out = rescaleScene(withLayers(a, t), PRODUCT, BANNER);
    const [ra, rt] = out.layers;
    expect(ra.x).toBeCloseTo(820);                     // still centred
    expect(ra.y).toBeCloseTo(428);
    expect(ra.width).toBeCloseTo(400 * (856 / 800));   // min(1640/800, 856/800)
    expect(rt.type === "text" && rt.font_size).toBeCloseTo(40 * 1.07);
  });
});

describe("layers", () => {
  const s = withLayers(newTextLayer(PRODUCT, "a", { id: "a" }), newTextLayer(PRODUCT, "b", { id: "b" }),
                       newTextLayer(PRODUCT, "c", { id: "c" }));
  const ids = (x: Scene) => x.layers.map((l) => l.id);

  it("reorders", () => {
    expect(ids(reorder(s, "a", "up"))).toEqual(["b", "a", "c"]);
    expect(ids(reorder(s, "a", "top"))).toEqual(["b", "c", "a"]);
    expect(ids(reorder(s, "c", "bottom"))).toEqual(["c", "a", "b"]);
    expect(ids(reorder(s, "a", "down"))).toEqual(["a", "b", "c"]);
  });

  it("duplicates right above the original, offset", () => {
    const { scene, id } = duplicateLayer(s, "a");
    expect(ids(scene)[0]).toBe("a");
    expect(ids(scene)[1]).toBe(id);
    expect(scene.layers[1].x).toBe(s.layers[0].x + 24);
  });
});

describe("history", () => {
  it("undo / redo, capped and cleared on new commit", () => {
    const s0 = emptyScene();
    const s1 = withLayers(newTextLayer(PRODUCT));
    const s2 = withLayers(newTextLayer(PRODUCT), newTextLayer(PRODUCT));
    let h = historyReducer({ past: [], present: s0, future: [] }, { type: "commit", scene: s1 });
    h = historyReducer(h, { type: "commit", scene: s2 });
    h = historyReducer(h, { type: "undo" });
    expect(h.present).toBe(s1);
    h = historyReducer(h, { type: "redo" });
    expect(h.present).toBe(s2);
    h = historyReducer(h, { type: "undo" });
    h = historyReducer(h, { type: "commit", scene: s0 });
    expect(h.future).toEqual([]);
    for (let i = 0; i < 80; i++) h = historyReducer(h, { type: "commit", scene: emptyScene() });
    expect(h.past.length).toBe(50);
  });
});

describe("gradient", () => {
  it("spans the canvas along the angle", () => {
    expect(gradientPoints(0, 100, 50)).toEqual({ start: { x: 0, y: 25 }, end: { x: 100, y: 25 } });
    const v = gradientPoints(90, 100, 50);
    expect(v.start.x).toBeCloseTo(50);
    expect(v.start.y).toBeCloseTo(0);
    expect(v.end.y).toBeCloseTo(50);
  });
});

describe("templates", () => {
  const canvasOf = { product: PRODUCT, ig: IG, banner: BANNER };

  it("each preset has templates and every template keeps layers inside the canvas", () => {
    for (const preset of ["product", "ig", "banner"] as const) {
      const list = TEMPLATES.filter((t) => t.preset === preset);
      expect(list.length).toBeGreaterThanOrEqual(2);
      for (const t of list) {
        const c = canvasOf[preset];
        const out = t.apply(withLayers(newAssetLayer(1, sizes[1], c), newAssetLayer(2, sizes[2], c)), c, sizeOf);
        for (const l of out.layers) {
          const h = l.type === "asset" ? assetHeight(l, sizes[l.asset_id]) : l.font_size;
          expect(l.x - l.width / 2).toBeGreaterThanOrEqual(-0.5);
          expect(l.x + l.width / 2).toBeLessThanOrEqual(c.w + 0.5);
          expect(l.y - h / 2).toBeGreaterThanOrEqual(-0.5);
          expect(l.y + h / 2).toBeLessThanOrEqual(c.h + 0.5);
        }
      }
    }
  });

  it("product caption: asset above, text added below, existing text reused", () => {
    const t = TEMPLATES.find((x) => x.key === "product-caption")!;
    const out = t.apply(withLayers(newAssetLayer(1, sizes[1], PRODUCT)), PRODUCT, sizeOf);
    const [asset, text] = out.layers as [AssetLayer, Scene["layers"][number]];
    expect(asset.y).toBeLessThan(text.y);
    expect(text.type === "text" && text.text).toBe("商品名稱");
    const again = t.apply(out, PRODUCT, sizeOf);
    expect(again.layers.filter((l) => l.type === "text")).toHaveLength(1);   // reused, not duplicated
  });

  it("text layers without a template slot are stacked under the template text, inside the canvas", () => {
    const banner = TEMPLATES.find((x) => x.key === "banner-text-left")!;
    const product = TEMPLATES.find((x) => x.key === "product-caption")!;
    const fromBanner = rescaleScene(banner.apply(withLayers(newAssetLayer(1, sizes[1], BANNER)), BANNER, sizeOf), BANNER, PRODUCT);
    const out = product.apply(fromBanner, PRODUCT, sizeOf);
    const texts = out.layers.filter((l) => l.type === "text");
    expect(texts).toHaveLength(2);
    const [title, sub] = texts;
    expect(sub.y).toBeGreaterThan(title.y);                       // below the caption, not over the product
    expect(sub.x).toBeCloseTo(title.x);
    expect(sub.type === "text" && sub.y + sub.font_size / 2).toBeLessThanOrEqual(PRODUCT.h);
  });

  it("banner text-left puts text on the left and the asset on the right", () => {
    const t = TEMPLATES.find((x) => x.key === "banner-text-left")!;
    const out = t.apply(withLayers(newAssetLayer(1, sizes[1], BANNER)), BANNER, sizeOf);
    const asset = out.layers.find((l) => l.type === "asset")!;
    const texts = out.layers.filter((l) => l.type === "text");
    expect(asset.x).toBeGreaterThan(BANNER.w / 2);
    expect(texts).toHaveLength(2);
    expect(texts.every((l) => l.x < BANNER.w / 2 && l.type === "text" && l.align === "left")).toBe(true);
  });
});
