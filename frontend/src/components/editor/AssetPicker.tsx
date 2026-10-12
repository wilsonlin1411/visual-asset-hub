import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Asset } from "../../api";
import { useVocabulary } from "../AssetDetail";

export const DRAG_TYPE = "application/x-vah-asset";
export interface DraggedAsset { id: number; width: number; height: number }

/** Left panel: find assets by search or category; click or drag them onto the canvas. */
export default function AssetPicker({ onAdd }: { onAdd: (asset: Asset) => void }) {
  const [text, setText] = useState("");
  const [q, setQ] = useState("");
  const [category, setCategory] = useState("");
  const vocab = useVocabulary();

  const result = useQuery({
    queryKey: q ? ["search", q, false] : ["assets", { category }],
    queryFn: async () => (q ? (await api.search(q, false)).hits.map((h) => h.asset)
                            : api.assets({ category: category || undefined })),
  });

  return (
    <aside className="editor-side">
      <h2>素材</h2>
      <form className="row" onSubmit={(e) => { e.preventDefault(); setQ(text.trim()); }}>
        <input type="search" value={text} placeholder="搜尋，例如：咖啡杯" onChange={(e) => {
          setText(e.target.value);
          if (!e.target.value) setQ("");
        }} style={{ flex: 1 }} />
        <button type="submit">搜尋</button>
      </form>
      {!q && (
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="">全部類別</option>
          {vocab.data?.categories.map((c) => <option key={c.name}>{c.name}</option>)}
        </select>
      )}
      <div className="muted" style={{ fontSize: 12 }}>點一下加入畫布，或拖曳到想放的位置。</div>
      {result.isLoading && <span className="spinner" />}
      {result.error && <div className="error">{String(result.error)}</div>}
      <div className="picker-grid">
        {result.data?.map((a) => (
          <button key={a.id} className="picker-item checker" title={`${a.category ?? "未分類"} #${a.id}`}
            draggable
            onDragStart={(e) => {
              const payload: DraggedAsset = { id: a.id, width: a.width, height: a.height };
              e.dataTransfer.setData(DRAG_TYPE, JSON.stringify(payload));
              e.dataTransfer.effectAllowed = "copy";
            }}
            onClick={() => onAdd(a)}>
            <img src={a.thumb_url} alt={a.category ?? ""} loading="lazy" draggable={false} />
          </button>
        ))}
      </div>
      {result.data?.length === 0 && <div className="muted">沒有素材</div>}
    </aside>
  );
}
