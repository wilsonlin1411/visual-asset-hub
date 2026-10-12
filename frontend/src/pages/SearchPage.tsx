import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import AssetCard from "../components/AssetCard";
import AssetDetail from "../components/AssetDetail";

const EXAMPLES = ["和在咖啡廳工作有關的物件", "適合下班小酌的東西", "可愛的公仔", "藍色的東西", "騎車要用的", "晚餐的主菜"];

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const includePeople = params.get("people") === "1";
  const [text, setText] = useState(q);
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => setText(q), [q]);

  const result = useQuery({
    queryKey: ["search", q, includePeople],
    queryFn: () => api.search(q, includePeople),
    enabled: q.trim().length > 0,
  });

  function run(query: string, people = includePeople) {
    const next = new URLSearchParams();
    if (query.trim()) next.set("q", query.trim());
    if (people) next.set("people", "1");
    setParams(next);
    setOpen(null);
  }

  const r = result.data;
  const parsed = r ? [
    ...r.categories.map((v) => ["類別", v]),
    ...r.contexts.map((v) => ["情境", v]),
    ...r.attributes.map((v) => ["屬性", v]),
  ] : [];

  return (
    <>
      <div className="page-head"><h1>搜尋素材</h1><span className="muted">用中文描述你要的素材，例如用途、情境、顏色。</span></div>
      <form className="searchbar" onSubmit={(e) => { e.preventDefault(); run(text); }}>
        <input type="search" value={text} placeholder="例如：和在咖啡廳工作有關的物件" autoFocus
          onChange={(e) => setText(e.target.value)} />
        <button className="primary" type="submit" disabled={!text.trim()}>搜尋</button>
      </form>
      <div className="parsed">
        <label className="check"><input type="checkbox" checked={includePeople} onChange={(e) => run(q, e.target.checked)} />包含人物（手）</label>
        <span className="muted">試試：</span>
        {EXAMPLES.map((ex) => <button key={ex} className="link" onClick={() => run(ex)}>{ex}</button>)}
      </div>

      {result.isFetching && <div className="row"><span className="spinner" /> 搜尋中…</div>}
      {result.error && <div className="error">{String(result.error)}</div>}
      {r && (
        <div className="parsed">
          <span className="muted">理解為：</span>
          {parsed.length ? parsed.map(([k, v]) => <span key={k + v} className={`chip${k === "情境" ? " context" : ""}`}>{k} {v}</span>)
            : <span className="muted">（沒有對到詞彙表，只用語意相似度）</span>}
          {r.excluded_categories.length > 0 && <span className="muted">· 已排除 {r.excluded_categories.join("、")}</span>}
          <span className="muted">· {r.hits.length} 筆結果</span>
        </div>
      )}

      <div className={`layout-with-drawer${open !== null ? " open" : ""}`}>
        <div>
          {r && r.hits.length === 0 && <div className="empty">找不到相關素材</div>}
          <div className="grid">
            {r?.hits.map((h, i) => (
              <AssetCard key={h.asset.id} asset={h.asset} selected={open === h.asset.id} onClick={() => setOpen(h.asset.id)}>
                <div className="score">#{i + 1} · 分數 {h.score.toFixed(3)}</div>
                <div className="reasons">{h.reasons.slice(1).join("、") || h.reasons[0]}</div>
              </AssetCard>
            ))}
          </div>
        </div>
        {open !== null && <AssetDetail key={open} assetId={open} onClose={() => setOpen(null)} />}
      </div>
    </>
  );
}
