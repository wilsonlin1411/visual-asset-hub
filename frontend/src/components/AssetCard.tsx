import type { ReactNode } from "react";
import type { Asset } from "../api";

export default function AssetCard({ asset, selected, onClick, children }: {
  asset: Asset; selected?: boolean; onClick?: () => void; children?: ReactNode;
}) {
  const contexts = asset.tags.filter((t) => t.kind === "context").slice(0, 2);
  return (
    <div className={`card${selected ? " selected" : ""}`} onClick={onClick}>
      <div className="thumb checker"><img src={asset.thumb_url} alt={asset.category ?? ""} loading="lazy" /></div>
      <div className="body">
        <div className="title">
          <span>{asset.category ?? "未分類"}</span>
          <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>#{asset.id}</span>
        </div>
        {contexts.length > 0 && (
          <div className="chips">{contexts.map((t) => <span key={t.value} className="chip context">{t.value}</span>)}</div>
        )}
        {children}
      </div>
    </div>
  );
}
