import { describe, expect, it } from "vitest";
import type { Candidate } from "../api";
import { candidateLabel, labelledPending, runSequential, splitCandidates, toggle } from "./candidates";
import { boxToPercent, clientToPhoto, maskContains, pickAt, type AlphaMask } from "./geometry";

const cand = (over: Partial<Candidate>): Candidate => ({
  idx: 0, yolo_label: null, category: null, category_score: null, score: 0.5, objectness: 0.9, hidden: false,
  status: "pending", bbox: [0, 0, 10, 10], area: 100, label_model: null, mask_model: "fastsam-s",
  mask_url: "", preview_url: "", overlay_url: "", ...over,
});

/** w x h alpha mask with the given cells opaque. */
function mask(w: number, h: number, opaque: (x: number, y: number) => boolean): AlphaMask {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[(y * w + x) * 4 + 3] = opaque(x, y) ? 255 : 0;
  return { width: w, height: h, data };
}

describe("candidates", () => {
  it("labels prefer a real category, then the YOLO label", () => {
    expect(candidateLabel(cand({ category: "滑鼠", yolo_label: "mouse" }))).toBe("滑鼠");
    expect(candidateLabel(cand({ category: "其他", yolo_label: "dining table" }))).toBe("dining table");
    expect(candidateLabel(cand({}))).toBe("未知物件");
  });

  it("collapses hidden candidates but never confirmed ones", () => {
    const list = [cand({ idx: 0 }), cand({ idx: 1, hidden: true }), cand({ idx: 2, hidden: true, status: "confirmed" })];
    const { shown, collapsed } = splitCandidates(list);
    expect(shown.map((c) => c.idx)).toEqual([0, 2]);
    expect(collapsed.map((c) => c.idx)).toEqual([1]);
  });

  it("select-labelled picks pending YOLO-labelled or confident candidates", () => {
    const list = [
      cand({ idx: 0, yolo_label: "cup" }),
      cand({ idx: 1, category: "筆電", category_score: 0.8 }),
      cand({ idx: 2, category: "筆電", category_score: 0.3 }),
      cand({ idx: 3, category: "其他", category_score: 0.9 }),
      cand({ idx: 4, yolo_label: "cup", status: "confirmed" }),
      cand({ idx: 5, category: "滑鼠", category_score: 0.9, hidden: true }),
    ];
    expect(labelledPending(list)).toEqual([0, 1]);
  });

  it("toggle returns a new set", () => {
    const a = new Set([1]);
    const b = toggle(a, 2);
    expect([...b].sort()).toEqual([1, 2]);
    expect([...toggle(b, 1)]).toEqual([2]);
    expect([...a]).toEqual([1]);
  });

  it("runs tasks one by one with progress and collects failures", async () => {
    const seen: string[] = [];
    const order: number[] = [];
    const { failed } = await runSequential([1, 2, 3], async (n) => {
      order.push(n);
      if (n === 2) throw new Error("boom");
    }, (d, t) => seen.push(`${d}/${t}`));
    expect(order).toEqual([1, 2, 3]);
    expect(seen).toEqual(["0/3", "1/3", "2/3", "3/3"]);
    expect(failed.map((f) => f.item)).toEqual([2]);
  });
});

describe("geometry", () => {
  it("converts boxes to percentages", () => {
    expect(boxToPercent([100, 50, 300, 150], 400, 200)).toEqual({ left: "25%", top: "25%", width: "50%", height: "50%" });
  });

  it("maps pointer coordinates to photo pixels", () => {
    expect(clientToPhoto(150, 70, { left: 50, top: 20, width: 200, height: 100 }, 4000, 2000)).toEqual([2000, 1000]);
  });

  it("tests the scaled mask, not just the box", () => {
    // bbox 100 px wide shown by a 10 px overlay: left half opaque
    const m = mask(10, 10, (x) => x < 5);
    expect(maskContains([0, 0, 100, 100], m, 20, 50)).toBe(true);
    expect(maskContains([0, 0, 100, 100], m, 80, 50)).toBe(false);
    expect(maskContains([0, 0, 100, 100], m, 120, 50)).toBe(false);
  });

  it("picks the smallest candidate whose mask contains the point", () => {
    const full = mask(4, 4, () => true);
    const items = [
      { idx: 0, bbox: [0, 0, 100, 100] as [number, number, number, number], area: 10000 },
      { idx: 1, bbox: [40, 40, 60, 60] as [number, number, number, number], area: 400 },
    ];
    const masks = new Map([[0, full], [1, full]]);
    expect(pickAt(items, masks, 50, 50)?.idx).toBe(1);
    expect(pickAt(items, masks, 10, 10)?.idx).toBe(0);
    expect(pickAt(items, masks, 150, 150)).toBeUndefined();
  });
});
