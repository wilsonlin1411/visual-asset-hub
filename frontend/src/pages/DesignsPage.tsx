import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api, type PresetKey } from "../api";

export default function DesignsPage() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets, staleTime: Infinity });
  const designs = useQuery({ queryKey: ["designs"], queryFn: api.designs });
  const create = useMutation({
    mutationFn: (p: PresetKey) => api.createDesign(p),
    onSuccess: (d) => { qc.invalidateQueries({ queryKey: ["designs"] }); nav(`/designs/${d.id}`); },
  });
  const remove = useMutation({
    mutationFn: api.deleteDesign,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["designs"] }),
  });

  return (
    <>
      <div className="page-head">
        <h1>設計</h1>
        <span className="muted">把素材排進商品主圖、IG 貼文或 Banner，匯出精確尺寸的成品。</span>
      </div>

      <h2>新增設計</h2>
      <div className="preset-row">
        {presets.data?.map((p) => (
          <button key={p.key} className="preset-card" disabled={create.isPending} onClick={() => create.mutate(p.key)}>
            <div className="preset-shape" style={{ aspectRatio: `${p.width} / ${p.height}` }} />
            <strong>{p.name}</strong>
            <span className="muted">{p.width} × {p.height}</span>
          </button>
        ))}
      </div>
      {create.error && <div className="error">{String(create.error)}</div>}

      <h2 style={{ marginTop: 24 }}>我的設計（{designs.data?.length ?? 0}）</h2>
      {designs.data?.length === 0 && <div className="empty">還沒有設計，從上面選一種尺寸開始</div>}
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
        {designs.data?.map((d) => (
          <div key={d.id} className="card">
            <Link to={`/designs/${d.id}`} className="thumb" style={{ aspectRatio: "4 / 3", background: "var(--surface-2)" }}>
              {d.thumb_url
                ? <img src={`${d.thumb_url}${d.thumb_url.includes("?") ? "&" : "?"}v=${encodeURIComponent(d.updated_at)}`} alt={d.name} />
                : <div className="preset-shape" style={{ aspectRatio: `${d.width} / ${d.height}`, height: "70%" }} />}
            </Link>
            <div className="body">
              <div className="title"><span>{d.name}</span></div>
              <div className="meta">{d.width}×{d.height} · {d.layer_count} 個圖層 · {d.updated_at.slice(5, 16).replace("T", " ")}</div>
              <div className="row">
                <Link className="btn" to={`/designs/${d.id}`}>編輯</Link>
                <button className="danger" disabled={remove.isPending}
                  onClick={() => confirm(`刪除「${d.name}」？它的匯出檔也會刪除。`) && remove.mutate(d.id)}>刪除</button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
