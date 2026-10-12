import type { BBox } from "../api";

/** A box in original-photo pixels as CSS percentages of the photo (for absolutely positioned overlays). */
export function boxToPercent(bbox: BBox, width: number, height: number) {
  const [x0, y0, x1, y1] = bbox;
  return {
    left: `${(x0 / width) * 100}%`,
    top: `${(y0 / height) * 100}%`,
    width: `${((x1 - x0) / width) * 100}%`,
    height: `${((y1 - y0) / height) * 100}%`,
  };
}

/** Pointer position inside a rendered element -> original-photo pixel coordinates. */
export function clientToPhoto(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number,
): [number, number] {
  return [((clientX - rect.left) / rect.width) * width, ((clientY - rect.top) / rect.height) * height];
}

export interface AlphaMask {
  width: number;
  height: number;
  data: Uint8ClampedArray; // RGBA, as from getImageData
}

/** Is photo point (x, y) inside the candidate's mask? The overlay PNG covers the bbox at reduced scale. */
export function maskContains(bbox: BBox, mask: AlphaMask, x: number, y: number): boolean {
  const [x0, y0, x1, y1] = bbox;
  if (x < x0 || x >= x1 || y < y0 || y >= y1) return false;
  const mx = Math.min(mask.width - 1, Math.floor(((x - x0) / (x1 - x0)) * mask.width));
  const my = Math.min(mask.height - 1, Math.floor(((y - y0) / (y1 - y0)) * mask.height));
  return mask.data[(my * mask.width + mx) * 4 + 3] > 127;
}

/** The candidate under a point: among those whose mask contains it, the smallest (most specific). */
export function pickAt<T extends { idx: number; bbox: BBox; area: number }>(
  items: T[],
  masks: ReadonlyMap<number, AlphaMask>,
  x: number,
  y: number,
): T | undefined {
  let best: T | undefined;
  for (const it of items) {
    const m = masks.get(it.idx);
    if (!m || !maskContains(it.bbox, m, x, y)) continue;
    if (!best || it.area < best.area) best = it;
  }
  return best;
}
