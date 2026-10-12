import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, type Asset, type Tag, type Vocabulary } from "../api";
import { boxToPercent } from "../lib/geometry";

const CATEGORY_SOURCE: Record<string, string> = { yolo: "YOLO + CLIP", clip: "CLIP", fallback: "未能判斷", user: "手動設定" };

export function tagSourceLabel(source: string): string {
  if (source === "user") return "手動";
  if (source.startsWith("rule:")) return `規則：${source.slice(5)}`;
  if (source === "clip-scene") return "整張照片場景";
  if (source === "color") return "像素顏色";
  if (source === "clip-material") return "CLIP 材質";
  if (source === "clip-style") return "CLIP 風格";
  return source;
}

export function useVocabulary() {
  return useQuery({ queryKey: ["vocabulary"], queryFn: api.vocabulary, staleTime: Infinity });
}

function TagChips({ tags, onRemove, disabled }: { tags: Tag[]; onRemove: (t: Tag) => void; disabled: boolean }) {
  if (!tags.length) return <span className="muted">（無）</span>;
  return (
    <div className="chips">
      {tags.map((t) => (
        <span key={`${t.kind}-${t.value}`} className={`chip ${t.kind}${t.source === "user" ? " user" : ""}`}
          title={`${tagSourceLabel(t.source)}${t.score != null ? `（${t.score.toFixed(2)}）` : ""}`}>
          {t.value}
          <button aria-label={`移除 ${t.value}`} disabled={disabled} onClick={() => onRemove(t)}>×</button>
        </span>
      ))}
    </div>
  );
}

function AddTag({ asset, vocab, onAdd, disabled }: {
  asset: Asset; vocab: Vocabulary; onAdd: (kind: Tag["kind"], value: string) => void; disabled: boolean;
}) {
  const [value, setValue] = useState("");
  const have = new Set(asset.tags.map((t) => `${t.kind}:${t.value}`));
  const groups: [string, Tag["kind"], string[]][] = [
    ["情境", "context", vocab.contexts],
    ["顏色", "attribute", vocab.attributes.color],
    ["材質", "attribute", vocab.attributes.material],
    ["風格", "attribute", vocab.attributes.style],
  ];
  return (
    <div className="row">
      <select value={value} onChange={(e) => setValue(e.target.value)} disabled={disabled}>
        <option value="">新增標籤…</option>
        {groups.map(([label, kind, names]) => (
          <optgroup key={label} label={label}>
            {names.filter((n) => !have.has(`${kind}:${n}`)).map((n) => <option key={n} value={`${kind}:${n}`}>{n}</option>)}
          </optgroup>
        ))}
      </select>
      <button disabled={!value || disabled} onClick={() => {
        const [kind, name] = value.split(/:(.*)/s) as [Tag["kind"], string];
        onAdd(kind, name);
        setValue("");
      }}>加入</button>
    </div>
  );
}

export default function AssetDetail({ assetId, onClose }: { assetId: number; onClose: () => void }) {
  const qc = useQueryClient();
  const asset = useQuery({ queryKey: ["asset", assetId], queryFn: () => api.asset(assetId) });
  const vocab = useVocabulary();
  const photo = useQuery({
    queryKey: ["photo", asset.data?.photo_id],
    queryFn: () => api.photo(asset.data!.photo_id),
    enabled: !!asset.data,
  });

  const onChanged = (a: Asset) => {
    qc.setQueryData(["asset", assetId], a);
    qc.invalidateQueries({ queryKey: ["assets"] });
    qc.invalidateQueries({ queryKey: ["search"] });
  };
  const setCategory = useMutation({ mutationFn: (c: string | null) => api.setCategory(assetId, c), onSuccess: onChanged });
  const addTag = useMutation({ mutationFn: (t: { kind: Tag["kind"]; value: string }) => api.addTag(assetId, t.kind, t.value), onSuccess: onChanged });
  const removeTag = useMutation({ mutationFn: (t: Tag) => api.removeTag(assetId, t.kind, t.value), onSuccess: onChanged });
  const remove = useMutation({
    mutationFn: () => api.deleteAsset(assetId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["assets"] });
      qc.invalidateQueries({ queryKey: ["search"] });
      qc.invalidateQueries({ queryKey: ["photos"] });
      onClose();
    },
  });
  const pending = setCategory.isPending || addTag.isPending || removeTag.isPending || remove.isPending;
  const mutationError = setCategory.error || addTag.error || removeTag.error || remove.error;

  if (asset.error) return <aside className="drawer"><div className="error">{String(asset.error)}</div></aside>;
  if (!asset.data || !vocab.data) return <aside className="drawer"><span className="spinner" /></aside>;
  const a = asset.data;
  const groups = [...new Set(vocab.data.categories.map((c) => c.group))];

  return (
    <aside className="drawer">
      <div className="drawer-head">
        <h2 style={{ margin: 0 }}>素材 #{a.id}</h2>
        <button className="link" onClick={onClose}>關閉 ✕</button>
      </div>
      <div className="preview checker"><img src={a.thumb_url} alt={a.category ?? ""} /></div>
      <div className="row">
        <a className="btn" href={a.png_url} download>下載透明 PNG（{a.width}×{a.height}）</a>
        {pending && <span className="spinner" />}
      </div>
      {mutationError && <div className="error">{String(mutationError)}</div>}

      <div className="field">
        <h3>類別 <span className="badge">{CATEGORY_SOURCE[a.category_source ?? ""] ?? a.category_source ?? "—"}
          {a.category_score != null && a.category_source !== "user" ? ` ${a.category_score.toFixed(2)}` : ""}</span></h3>
        <select value={a.category ?? ""} disabled={pending}
          onChange={(e) => setCategory.mutate(e.target.value || null)}>
          <option value="">（未分類）</option>
          {groups.map((g) => (
            <optgroup key={g} label={g}>
              {vocab.data.categories.filter((c) => c.group === g).map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </optgroup>
          ))}
        </select>
        <span className="muted" style={{ fontSize: 12 }}>手動設定的類別會被情境規則採用（例如改成「筆電」後，同照片有咖啡杯就推出「遠距工作」）。</span>
      </div>

      <div className="field">
        <h3>情境</h3>
        <TagChips tags={a.tags.filter((t) => t.kind === "context")} disabled={pending} onRemove={(t) => removeTag.mutate(t)} />
      </div>
      <div className="field">
        <h3>屬性（顏色 / 材質 / 風格）</h3>
        <TagChips tags={a.tags.filter((t) => t.kind === "attribute")} disabled={pending} onRemove={(t) => removeTag.mutate(t)} />
      </div>
      <AddTag asset={a} vocab={vocab.data} disabled={pending} onAdd={(kind, value) => addTag.mutate({ kind, value })} />
      <span className="muted" style={{ fontSize: 12 }}>虛線外框 = 手動加入；手動加入或移除的標籤，重新標註時會保留。</span>

      {photo.data && (
        <div className="field">
          <h3>來源照片 <Link to={`/photos/${photo.data.id}`} style={{ fontWeight: 400 }}>#{photo.data.id} {photo.data.original_name}</Link></h3>
          <div className="source">
            <img src={photo.data.display_url} alt="" loading="lazy" />
            <div className="box" style={boxToPercent(a.bbox, photo.data.width, photo.data.height)} />
          </div>
        </div>
      )}

      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted" style={{ fontSize: 12 }}>遮罩：{a.mask_model}{a.label_model ? ` · 類別：${a.label_model}` : ""}</span>
        <button className="danger" disabled={pending}
          onClick={() => confirm(`刪除素材 #${a.id}？候選會回到「待確認」。`) && remove.mutate()}>刪除素材</button>
      </div>
    </aside>
  );
}
