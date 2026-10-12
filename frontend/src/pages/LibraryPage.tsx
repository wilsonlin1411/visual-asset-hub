import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import AssetCard from "../components/AssetCard";
import AssetDetail, { useVocabulary } from "../components/AssetDetail";

export default function LibraryPage() {
  const [params, setParams] = useSearchParams();
  const category = params.get("category") ?? "";
  const tag = params.get("tag") ?? "";
  const photo = params.get("photo") ?? "";
  const [open, setOpen] = useState<number | null>(null);
  const vocab = useVocabulary();

  const assets = useQuery({
    queryKey: ["assets", { category, tag, photo }],
    queryFn: () => api.assets({ category: category || undefined, tag: tag || undefined,
                                 photo_id: photo ? Number(photo) : undefined }),
  });

  function setFilter(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  }

  const groups = [...new Set(vocab.data?.categories.map((c) => c.group) ?? [])];
  const tagGroups: [string, string[]][] = vocab.data
    ? [["情境", vocab.data.contexts], ["顏色", vocab.data.attributes.color],
       ["材質", vocab.data.attributes.material], ["風格", vocab.data.attributes.style]]
    : [];

  return (
    <>
      <div className="page-head">
        <h1>素材庫</h1>
        <span className="muted">{assets.data ? `${assets.data.length} 個素材` : ""}</span>
        <span className="spacer" />
        <div className="filters">
          <select value={category} onChange={(e) => setFilter("category", e.target.value)}>
            <option value="">全部類別</option>
            {groups.map((g) => (
              <optgroup key={g} label={g}>
                {vocab.data!.categories.filter((c) => c.group === g).map((c) => <option key={c.name}>{c.name}</option>)}
              </optgroup>
            ))}
          </select>
          <select value={tag} onChange={(e) => setFilter("tag", e.target.value)}>
            <option value="">全部標籤</option>
            {tagGroups.map(([label, names]) => (
              <optgroup key={label} label={label}>{names.map((n) => <option key={n}>{n}</option>)}</optgroup>
            ))}
          </select>
          {photo && <span className="chip">照片 #{photo} <button onClick={() => setFilter("photo", "")}>×</button></span>}
          {(category || tag || photo) && <button className="link" onClick={() => setParams({}, { replace: true })}>清除篩選</button>}
        </div>
      </div>

      {assets.error && <div className="error">{String(assets.error)}</div>}
      <div className={`layout-with-drawer${open !== null ? " open" : ""}`}>
        <div>
          {assets.isLoading && <div className="row"><span className="spinner" /> 載入中…</div>}
          {assets.data?.length === 0 && <div className="empty">沒有符合的素材</div>}
          <div className="grid">
            {assets.data?.map((a) => (
              <AssetCard key={a.id} asset={a} selected={open === a.id} onClick={() => setOpen(a.id)} />
            ))}
          </div>
        </div>
        {open !== null && <AssetDetail key={open} assetId={open} onClose={() => setOpen(null)} />}
      </div>
    </>
  );
}
