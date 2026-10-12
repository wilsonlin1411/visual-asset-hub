import type { ExportRecord } from "../../api";
import { FONTS, type Background, type Layer, type Order } from "../../lib/scene";

function Num({ label, value, onChange, step = 1, min, max }: {
  label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number;
}) {
  return (
    <label className="field-inline">
      <span>{label}</span>
      <input type="number" value={Math.round(value * 100) / 100} step={step} min={min} max={max}
        onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) onChange(v); }} />
    </label>
  );
}

function Slider({ label, value, onChange, min, max, step, format }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number; format: (v: number) => string;
}) {
  return (
    <label className="field-inline">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="muted" style={{ minWidth: 40, textAlign: "right" }}>{format(value)}</span>
    </label>
  );
}

export function LayerInspector({ layer, onChange, onOrder, onDuplicate, onDelete, onFit }: {
  layer: Layer;
  onChange: (patch: Partial<Layer>) => void;
  onOrder: (op: Order) => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onFit?: () => void;
}) {
  return (
    <div className="inspector">
      <h2>{layer.type === "asset" ? `素材 #${layer.asset_id}` : "文字"}</h2>

      {layer.type === "text" && (
        <>
          <textarea rows={3} value={layer.text} onChange={(e) => onChange({ text: e.target.value })} />
          <label className="field-inline">
            <span>字型</span>
            <select value={layer.font_family} onChange={(e) => onChange({ font_family: e.target.value })}>
              {FONTS.map((f) => <option key={f.family} value={f.family} style={{ fontFamily: f.family }}>{f.label}</option>)}
            </select>
          </label>
          <Num label="字級" value={layer.font_size} min={4} onChange={(v) => onChange({ font_size: Math.max(4, v) })} />
          <label className="field-inline">
            <span>顏色</span>
            <input type="color" value={layer.fill} onChange={(e) => onChange({ fill: e.target.value })} />
          </label>
          <div className="seg">
            <button className={layer.font_weight === "bold" ? "on" : ""}
              onClick={() => onChange({ font_weight: layer.font_weight === "bold" ? "normal" : "bold" })}><b>粗體</b></button>
            {(["left", "center", "right"] as const).map((a) => (
              <button key={a} className={layer.align === a ? "on" : ""} onClick={() => onChange({ align: a })}>
                {{ left: "靠左", center: "置中", right: "靠右" }[a]}
              </button>
            ))}
          </div>
          <Slider label="行距" value={layer.line_height} min={0.8} max={2.5} step={0.05} format={(v) => v.toFixed(2)}
            onChange={(v) => onChange({ line_height: v })} />
        </>
      )}

      <div className="grid2">
        <Num label="X" value={layer.x} onChange={(v) => onChange({ x: v })} />
        <Num label="Y" value={layer.y} onChange={(v) => onChange({ y: v })} />
        <Num label="寬" value={layer.width} min={4} onChange={(v) => onChange({ width: Math.max(4, v) })} />
        <Num label="旋轉" value={layer.rotation} step={1} onChange={(v) => onChange({ rotation: v })} />
      </div>
      <Slider label="不透明" value={layer.opacity} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`}
        onChange={(v) => onChange({ opacity: v })} />

      {layer.type === "asset" && (
        <div className="row">
          <label className="check"><input type="checkbox" checked={layer.flip_x} onChange={(e) => onChange({ flip_x: e.target.checked })} />水平翻轉</label>
          <label className="check"><input type="checkbox" checked={layer.shadow} onChange={(e) => onChange({ shadow: e.target.checked })} />陰影</label>
          {onFit && <button onClick={onFit}>置中並填滿</button>}
        </div>
      )}

      <h3>圖層</h3>
      <div className="seg">
        <button onClick={() => onOrder("top")}>置頂</button>
        <button onClick={() => onOrder("up")}>上移</button>
        <button onClick={() => onOrder("down")}>下移</button>
        <button onClick={() => onOrder("bottom")}>置底</button>
      </div>
      <div className="row">
        <button onClick={onDuplicate}>複製（Ctrl+D）</button>
        <button className="danger" onClick={onDelete}>刪除（Del）</button>
      </div>
    </div>
  );
}

export function CanvasInspector({ name, onName, background, onBackground, exports }: {
  name: string;
  onName: (n: string) => void;
  background: Background;
  onBackground: (b: Background) => void;
  exports: ExportRecord[];
}) {
  const set = (p: Partial<Background>) => onBackground({ ...background, ...p });
  return (
    <div className="inspector">
      <h2>設計</h2>
      <label className="field-inline"><span>名稱</span>
        <input type="text" value={name} onChange={(e) => onName(e.target.value)} style={{ flex: 1 }} />
      </label>

      <h3>背景</h3>
      <div className="seg">
        {(["color", "gradient", "transparent"] as const).map((t) => (
          <button key={t} className={background.type === t ? "on" : ""} onClick={() => set({ type: t })}>
            {{ color: "純色", gradient: "漸層", transparent: "透明" }[t]}
          </button>
        ))}
      </div>
      {background.type !== "transparent" && (
        <div className="row">
          <input type="color" value={background.color} onChange={(e) => set({ color: e.target.value })} />
          {background.type === "gradient" && (
            <>
              <span className="muted">→</span>
              <input type="color" value={background.color2} onChange={(e) => set({ color2: e.target.value })} />
              <select value={background.angle} onChange={(e) => set({ angle: Number(e.target.value) })}>
                <option value={0}>左 → 右</option>
                <option value={90}>上 → 下</option>
                <option value={45}>左上 → 右下</option>
                <option value={135}>右上 → 左下</option>
              </select>
            </>
          )}
        </div>
      )}
      {background.type === "transparent" && <div className="muted" style={{ fontSize: 12 }}>匯出 PNG 時保留透明；JPEG 會以白色填滿。</div>}

      <h3>匯出紀錄</h3>
      {exports.length === 0 && <div className="muted">尚未匯出</div>}
      <ul className="export-list">
        {exports.slice(0, 8).map((e) => (
          <li key={e.id}>
            <a href={e.url} download>#{e.id} · {e.format.toUpperCase()} {e.width}×{e.height}</a>
            <span className="muted"> {(e.bytes / 1024).toFixed(0)} KB · {e.created_at.slice(5, 16).replace("T", " ")}</span>
          </li>
        ))}
      </ul>
      <div className="muted" style={{ fontSize: 12 }}>點畫布上的圖層可編輯；Ctrl+Z 復原、Ctrl+Y 重做、方向鍵微調（Shift 加速）。</div>
    </div>
  );
}
