import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api, type Asset, type Candidate } from "../api";
import PhotoStage from "../components/PhotoStage";
import { candidateLabel, labelledPending, runSequential, splitCandidates, toggle } from "../lib/candidates";

const STATUS_TEXT: Record<Candidate["status"], string> = { pending: "", confirmed: "已確認", discarded: "已捨棄" };

function CandidateCard({ c, selected, hovered, onToggle, onHover }: {
  c: Candidate; selected: boolean; hovered: boolean; onToggle: () => void; onHover: (h: boolean) => void;
}) {
  const obj = c.objectness ?? 0;
  return (
    <div className={`card${selected ? " selected" : ""}${c.status !== "pending" ? " dim" : ""}`}
      style={hovered ? { borderColor: "var(--hover)" } : undefined}
      onClick={c.status === "confirmed" ? undefined : onToggle}
      onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)}
      title={`#${c.idx}  YOLO: ${c.yolo_label ?? "—"}  CLIP: ${c.category ?? "—"} ${c.category_score?.toFixed(2) ?? ""}`}>
      <div className="thumb checker"><img src={c.preview_url} alt="" loading="lazy" /></div>
      <div className="body">
        <div className="title">
          <span>{candidateLabel(c)}</span>
          {c.status !== "pending" && <span className={`badge ${c.status === "confirmed" ? "ok" : ""}`}>{STATUS_TEXT[c.status]}</span>}
        </div>
        <div className="meta">{c.yolo_label ? `YOLO ${c.yolo_label}` : `像物件 ${(obj * 100).toFixed(0)}%`}</div>
        <div className="score-bar"><div style={{ width: `${obj * 100}%` }} /></div>
      </div>
    </div>
  );
}

export default function ReviewPage() {
  const photoId = Number(useParams().photoId);
  const qc = useQueryClient();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [hovered, setHovered] = useState<number | null>(null);
  const [showCollapsed, setShowCollapsed] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [created, setCreated] = useState<Asset[]>([]);
  const [error, setError] = useState<string | null>(null);

  const photo = useQuery({
    queryKey: ["photo", photoId],
    queryFn: () => api.photo(photoId),
    refetchInterval: (q) => (q.state.data?.status === "processing" ? 1500 : false),
  });
  const ready = photo.data?.status === "ready";
  const cands = useQuery({ queryKey: ["candidates", photoId], queryFn: () => api.candidates(photoId), enabled: ready });

  const all = cands.data?.candidates ?? [];
  const { shown, collapsed } = useMemo(() => splitCandidates(all), [all]);
  const pickable = showCollapsed ? [...shown, ...collapsed] : shown;
  const busy = progress !== null;

  async function refresh() {
    await Promise.all([
      qc.invalidateQueries({ queryKey: ["candidates", photoId] }),
      qc.invalidateQueries({ queryKey: ["photo", photoId] }),
      qc.invalidateQueries({ queryKey: ["photos"] }),
      qc.invalidateQueries({ queryKey: ["assets"] }),
    ]);
  }

  async function confirmSelected() {
    const order = all.map((c) => c.idx).filter((i) => selected.has(i));
    setError(null);
    const made: Asset[] = [];
    // One request per candidate: ~1 s each (refine + CLIP), which gives a natural progress bar.
    const { failed } = await runSequential(order, async (idx) => {
      made.push(...(await api.confirm(photoId, [idx])));
    }, (done, total) => setProgress({ done, total }));
    setProgress(null);
    setCreated((prev) => [...made, ...prev]);
    setSelected(new Set(failed.map((f) => f.item)));
    if (failed.length) setError(`${failed.length} 個候選確認失敗：${String(failed[0].error)}`);
    await refresh();
  }

  async function discardSelected() {
    const idxs = all.filter((c) => selected.has(c.idx) && c.status === "pending").map((c) => c.idx);
    try {
      await api.discard(photoId, idxs);
      setSelected(new Set());
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  if (photo.error) return <div className="error">讀取照片失敗：{String(photo.error)}</div>;
  if (!photo.data) return <div className="row"><span className="spinner" /> 載入中…</div>;
  const p = photo.data;

  return (
    <>
      <div className="page-head">
        <Link to="/upload">← 照片</Link>
        <h1>#{p.id} {p.original_name}</h1>
        <span className="muted">{p.width}×{p.height}</span>
        <div className="chips">
          {p.contexts.map((t) => (
            <span key={t.value} className="chip context" title={t.source}>{t.value}</span>
          ))}
        </div>
        <span className="spacer" />
        {(p.counts.assets ?? 0) > 0 && <Link className="btn" to={`/library?photo=${p.id}`}>這張照片的素材（{p.counts.assets}）</Link>}
      </div>

      {p.status === "processing" && (
        <div className="empty"><span className="spinner" /> 正在分割與分析這張照片
          {p.queue_position ? `（佇列第 ${p.queue_position} 位）` : ""}，完成後會自動顯示候選物件…</div>
      )}
      {p.status === "failed" && <div className="error">處理失敗：{p.error}</div>}
      {error && <div className="error">{error}</div>}

      {ready && (
        <div className="review">
          <PhotoStage photo={p} candidates={pickable} allCandidates={all} selected={selected} hovered={hovered}
            onHover={setHovered} onToggle={(idx) => !busy && setSelected((s) => toggle(s, idx))} />

          <div className="cand-panel">
            <div className="toolbar">
              {busy ? (
                <>
                  <span className="spinner" /> 去背與標註中 {progress.done}/{progress.total}
                  <div className="progress" style={{ flex: 1 }}><div style={{ width: `${(progress.done / progress.total) * 100}%` }} /></div>
                </>
              ) : (
                <>
                  <button onClick={() => setSelected(new Set(labelledPending(all)))}>全選有類別的</button>
                  <button onClick={() => setSelected(new Set())} disabled={!selected.size}>清除選取</button>
                  <span className="spacer" style={{ flex: 1 }} />
                  <button className="danger" onClick={discardSelected} disabled={!selected.size}>捨棄</button>
                  <button className="primary" onClick={confirmSelected} disabled={!selected.size}>
                    確認 {selected.size || ""} 個為素材
                  </button>
                </>
              )}
            </div>

            {created.length > 0 && (
              <div className="toolbar" style={{ position: "static" }}>
                <span>已建立 {created.length} 個素材：</span>
                {created.slice(0, 8).map((a) => (
                  <img key={a.id} className="checker" src={a.thumb_url} alt={a.category ?? ""} title={a.category ?? ""}
                    style={{ width: 40, height: 40, objectFit: "contain", borderRadius: 6 }} />
                ))}
                <Link to={`/library?photo=${p.id}`}>到素材庫 →</Link>
              </div>
            )}

            {cands.isLoading && <div className="row"><span className="spinner" /> 載入候選…</div>}
            <h3>候選物件（{shown.length}）· 有 YOLO 類別的排前面，其餘依「像物件」程度排序</h3>
            <div className="cand-grid">
              {shown.map((c) => (
                <CandidateCard key={c.idx} c={c} selected={selected.has(c.idx)} hovered={hovered === c.idx}
                  onToggle={() => !busy && setSelected((s) => toggle(s, c.idx))}
                  onHover={(h) => setHovered(h ? c.idx : null)} />
              ))}
            </div>

            {collapsed.length > 0 && (
              <>
                <div className="collapsed-head">
                  <button className="link" onClick={() => setShowCollapsed((v) => !v)}>
                    {showCollapsed ? "▾ 收合" : "▸ 顯示"}其他 {collapsed.length} 個
                  </button>
                  <span>（像背景或碎片，「像物件」低於 {Math.round((cands.data?.objectness_threshold ?? 0) * 100)}%）</span>
                </div>
                {showCollapsed && (
                  <div className="cand-grid">
                    {collapsed.map((c) => (
                      <CandidateCard key={c.idx} c={c} selected={selected.has(c.idx)} hovered={hovered === c.idx}
                        onToggle={() => !busy && setSelected((s) => toggle(s, c.idx))}
                        onHover={(h) => setHovered(h ? c.idx : null)} />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
