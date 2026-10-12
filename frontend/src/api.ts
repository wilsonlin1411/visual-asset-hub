// Typed client for the FastAPI backend (backend/api/app.py). Shapes mirror the Pydantic models.
import type { Scene } from "./lib/scene";

export type BBox = [number, number, number, number];

export interface Tag {
  kind: "attribute" | "context";
  value: string;
  score: number | null;
  source: string;
}

export interface Photo {
  id: number;
  original_name: string;
  width: number;
  height: number;
  status: "processing" | "ready" | "failed";
  error: string | null;
  created_at: string;
  image_url: string;
  display_url: string;
  contexts: Tag[];
  counts: { pending?: number; confirmed?: number; discarded?: number; assets?: number };
  queue_position: number | null;
}

export interface Candidate {
  idx: number;
  yolo_label: string | null;
  category: string | null;
  category_score: number | null;
  score: number;
  objectness: number | null;
  hidden: boolean;
  status: "pending" | "confirmed" | "discarded";
  bbox: BBox;
  area: number;
  label_model: string | null;
  mask_model: string;
  mask_url: string;
  preview_url: string;
  overlay_url: string;
}

export interface CandidateList {
  photo_id: number;
  photo_status: Photo["status"];
  objectness_threshold: number;
  hidden_count: number;
  candidates: Candidate[];
}

export interface Asset {
  id: number;
  photo_id: number;
  category: string | null;
  category_score: number | null;
  category_source: string | null;
  bbox: BBox;
  origin: [number, number];
  width: number;
  height: number;
  label_model: string | null;
  mask_model: string;
  created_at: string;
  png_url: string;
  thumb_url: string;
  tags: Tag[];
}

export interface SearchResult {
  query: string;
  categories: string[];
  attributes: string[];
  contexts: string[];
  excluded_categories: string[];
  hits: { asset: Asset; score: number; cosine: number; reasons: string[] }[];
}

export interface Vocabulary {
  categories: { name: string; group: string }[];
  contexts: string[];
  attributes: { color: string[]; material: string[]; style: string[] };
  excluded_categories: string[];
}

// ---- canvas (stage 6); scene types live in lib/scene.ts

export type PresetKey = "product" | "ig" | "banner";

export interface Preset {
  key: PresetKey;
  name: string;
  width: number;
  height: number;
}

export interface DesignSummary {
  id: number;
  name: string;
  preset: PresetKey;
  width: number;
  height: number;
  created_at: string;
  updated_at: string;
  layer_count: number;
  thumb_url: string | null;
}

export interface DesignAsset {
  width: number;
  height: number;
  category: string | null;
  medium_url: string;
  thumb_url: string;
}

export interface Design extends DesignSummary {
  scene: Scene;
  assets: Record<string, DesignAsset>;
  missing_assets: number[];
}

export interface ExportRecord {
  id: number;
  design_id: number;
  format: "png" | "jpeg";
  width: number;
  height: number;
  bytes: number;
  created_at: string;
  url: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const qs = (params: Record<string, string | number | boolean | undefined | null>) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  photos: () => request<Photo[]>("/api/photos?limit=500"),
  photo: (id: number) => request<Photo>(`/api/photos/${id}`),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ photo: Photo; created: boolean }>("/api/photos", { method: "POST", body: form });
  },
  deletePhoto: (id: number) => request<{ deleted: number }>(`/api/photos/${id}`, { method: "DELETE" }),
  candidates: (photoId: number) =>
    request<CandidateList>(`/api/photos/${photoId}/candidates?include_hidden=true`),
  confirm: (photoId: number, idxs: number[]) =>
    request<Asset[]>("/api/assets/confirm", json("POST", { photo_id: photoId, idxs })),
  discard: (photoId: number, idxs: number[]) =>
    request<{ discarded: number }>("/api/candidates/discard", json("POST", { photo_id: photoId, idxs })),
  assets: (f: { category?: string; tag?: string; photo_id?: number }) =>
    request<Asset[]>(`/api/assets${qs({ ...f, limit: 1000 })}`),
  asset: (id: number) => request<Asset>(`/api/assets/${id}`),
  setCategory: (id: number, category: string | null) =>
    request<Asset>(`/api/assets/${id}`, json("PATCH", { category })),
  addTag: (id: number, kind: Tag["kind"], value: string) =>
    request<Asset>(`/api/assets/${id}/tags`, json("POST", { kind, value })),
  removeTag: (id: number, kind: Tag["kind"], value: string) =>
    request<Asset>(`/api/assets/${id}/tags${qs({ kind, value })}`, { method: "DELETE" }),
  deleteAsset: (id: number) => request<{ deleted: number }>(`/api/assets/${id}`, { method: "DELETE" }),
  search: (q: string, includePeople: boolean) =>
    request<SearchResult>(`/api/search${qs({ q, k: 30, include_people: includePeople })}`),
  vocabulary: () => request<Vocabulary>("/api/vocabulary"),

  presets: () => request<Preset[]>("/api/presets"),
  designs: () => request<DesignSummary[]>("/api/designs"),
  design: (id: number) => request<Design>(`/api/designs/${id}`),
  createDesign: (preset: PresetKey, name?: string) => request<Design>("/api/designs", json("POST", { preset, name })),
  saveDesign: (id: number, body: { scene: Scene; name?: string; preset?: PresetKey }) =>
    request<Design>(`/api/designs/${id}`, json("PUT", body)),
  deleteDesign: (id: number) => request<{ deleted: number }>(`/api/designs/${id}`, { method: "DELETE" }),
  exportDesign: (id: number, blob: Blob, format: "png" | "jpeg") => {
    const form = new FormData();
    form.append("file", blob, `design-${id}.${format === "png" ? "png" : "jpg"}`);
    form.append("format", format);
    return request<ExportRecord>(`/api/designs/${id}/export`, { method: "POST", body: form });
  },
  exports: (designId: number) => request<ExportRecord[]>(`/api/exports?design_id=${designId}`),
};

/** Canvas source for an asset: cut-out capped at 2048 px (enough for every preset). */
export const mediumUrl = (assetId: number) => `/api/assets/${assetId}/medium`;
