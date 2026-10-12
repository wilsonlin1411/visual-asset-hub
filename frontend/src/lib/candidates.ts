import type { Candidate } from "../api";

/** Display name of a candidate: CLIP category, else YOLO label, else "未知". */
export function candidateLabel(c: Candidate): string {
  return c.category && c.category !== "其他" ? c.category : c.yolo_label ?? "未知物件";
}

/** Split into the default list and the collapsed "looks like background" group, keeping API order
 *  (YOLO-labelled first, then by objectness). Confirmed candidates always stay visible. */
export function splitCandidates(cands: Candidate[]): { shown: Candidate[]; collapsed: Candidate[] } {
  const shown: Candidate[] = [];
  const collapsed: Candidate[] = [];
  for (const c of cands) (c.hidden && c.status !== "confirmed" ? collapsed : shown).push(c);
  return { shown, collapsed };
}

/** Candidates the "select all labelled" button picks: pending ones with a YOLO label or a confident category. */
export function labelledPending(cands: Candidate[], minCategoryScore = 0.5): number[] {
  return cands
    .filter((c) => c.status === "pending" && !c.hidden)
    .filter((c) => c.yolo_label !== null || (c.category !== null && c.category !== "其他" && (c.category_score ?? 0) >= minCategoryScore))
    .map((c) => c.idx);
}

export function toggle(selected: ReadonlySet<number>, idx: number): Set<number> {
  const next = new Set(selected);
  if (next.has(idx)) next.delete(idx);
  else next.add(idx);
  return next;
}

/** Run `task` for each item one at a time, reporting progress (confirm = ~1 s per candidate). */
export async function runSequential<T>(
  items: T[],
  task: (item: T) => Promise<void>,
  onProgress: (done: number, total: number) => void,
): Promise<{ failed: { item: T; error: unknown }[] }> {
  const failed: { item: T; error: unknown }[] = [];
  onProgress(0, items.length);
  for (let i = 0; i < items.length; i++) {
    try {
      await task(items[i]);
    } catch (error) {
      failed.push({ item: items[i], error });
    }
    onProgress(i + 1, items.length);
  }
  return { failed };
}
