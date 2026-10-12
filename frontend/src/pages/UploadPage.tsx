import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, type Photo } from "../api";
import { DemoNotice } from "../components/DemoNotice";

function StatusBadge({ p }: { p: Photo }) {
  if (p.status === "processing")
    return (
      <span className="badge warn">
        <span className="spinner" style={{ width: 10, height: 10, marginRight: 4, verticalAlign: -1 }} />
        {p.queue_position ? `處理中（第 ${p.queue_position} 位）` : "處理中"}
      </span>
    );
  if (p.status === "failed") return <span className="badge danger" title={p.error ?? ""}>失敗</span>;
  return <span className="badge ok">完成</span>;
}

export default function UploadPage() {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const photos = useQuery({
    queryKey: ["photos"],
    queryFn: api.photos,
    // poll while anything is still being segmented / analysed
    refetchInterval: (q) => (q.state.data?.some((p) => p.status === "processing") ? 1500 : false),
  });

  const remove = useMutation({
    mutationFn: api.deletePhoto,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["photos"] }),
    onError: (e) => setError(String(e)),
  });

  async function uploadFiles(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!list.length) return;
    setError(null);
    setUploading({ done: 0, total: list.length });
    const failed: string[] = [];
    for (let i = 0; i < list.length; i++) {
      try {
        await api.upload(list[i]);
      } catch (e) {
        failed.push(`${list[i].name}：${(e as Error).message}`);
      }
      setUploading({ done: i + 1, total: list.length });
      qc.invalidateQueries({ queryKey: ["photos"] });
    }
    setUploading(null);
    if (failed.length) setError(`上傳失敗：${failed.join("；")}`);
  }

  return (
    <>
      <div className="page-head">
        <h1>上傳照片</h1>
        <span className="muted">上傳後會在背景自動分割、判斷類別與情境，完成後即可確認候選物件。</span>
      </div>
      <DemoNotice />

      <div
        className={`dropzone${over ? " over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); uploadFiles(e.dataTransfer.files); }}
      >
        <input ref={input} type="file" accept="image/*" multiple hidden
          onChange={(e) => { if (e.target.files) uploadFiles(e.target.files); e.target.value = ""; }} />
        {uploading ? (
          <div className="row" style={{ justifyContent: "center" }}>
            <span className="spinner" /> 上傳中 {uploading.done}/{uploading.total}
            <div className="progress"><div style={{ width: `${(uploading.done / uploading.total) * 100}%` }} /></div>
          </div>
        ) : (
          <><strong>拖放照片到這裡</strong>，或點一下選擇檔案（可多選；同一張照片重複上傳會直接沿用）</>
        )}
      </div>

      {error && <div className="error">{error}</div>}
      {photos.error && <div className="error">無法讀取照片清單：{String(photos.error)}</div>}

      <h2>照片（{photos.data?.length ?? 0}）</h2>
      {photos.data && photos.data.length === 0 && <div className="empty">還沒有照片</div>}
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
        {photos.data?.map((p) => (
          <div key={p.id} className="card photo-card">
            <Link to={`/photos/${p.id}`} className="thumb">
              <img src={p.display_url} alt={p.original_name} loading="lazy" />
            </Link>
            <div className="body">
              <div className="title">
                <span title={p.original_name} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  #{p.id} {p.original_name}
                </span>
                <StatusBadge p={p} />
              </div>
              <div className="meta">
                {p.width}×{p.height}
                {p.status === "ready" &&
                  ` · 待確認 ${p.counts.pending ?? 0} · 素材 ${p.counts.assets ?? 0}`}
              </div>
              <div className="row">
                <Link className="btn" to={`/photos/${p.id}`}>{p.status === "ready" ? "確認候選" : "查看"}</Link>
                <button className="danger" disabled={p.status === "processing" || remove.isPending}
                  onClick={() => {
                    if (confirm(`刪除照片 #${p.id}？它的 ${p.counts.assets ?? 0} 個素材也會一起刪除。`)) remove.mutate(p.id);
                  }}>刪除</button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
