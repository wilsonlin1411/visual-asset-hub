import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api, mediumUrl, type Asset, type PresetKey } from "../api";
import AssetPicker, { type DraggedAsset } from "../components/editor/AssetPicker";
import DesignCanvas, { type CanvasAsset, type CanvasHandle } from "../components/editor/DesignCanvas";
import { CanvasInspector, LayerInspector } from "../components/editor/Inspector";
import {
  TEMPLATES, duplicateLayer, emptyScene, fitAsset, historyReducer, newAssetLayer, newTextLayer, removeLayer,
  reorder, rescaleScene, updateLayer, type History, type Layer, type Scene, type Size,
} from "../lib/scene";

type SaveState = { kind: "idle" | "saving" | "saved" | "error"; at?: string; error?: string };

function isTyping(t: EventTarget | null) {
  const el = t as HTMLElement | null;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
}

export default function EditorPage() {
  const id = Number(useParams().designId);
  const qc = useQueryClient();
  const canvas = useRef<CanvasHandle>(null);
  const design = useQuery({ queryKey: ["design", id], queryFn: () => api.design(id), staleTime: Infinity });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets, staleTime: Infinity });
  const exportsQ = useQuery({ queryKey: ["exports", id], queryFn: () => api.exports(id) });

  const [hist, dispatch] = useReducer(historyReducer, { past: [], present: emptyScene(), future: [] } as History);
  const [loaded, setLoaded] = useState(false);
  const [name, setName] = useState("");
  const [preset, setPreset] = useState<PresetKey>("product");
  const [assets, setAssets] = useState<Record<number, CanvasAsset>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [save, setSave] = useState<SaveState>({ kind: "idle" });
  const [format, setFormat] = useState<"png" | "jpeg">("png");
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scene = hist.present;

  // Load the design once into local editing state.
  useEffect(() => {
    const d = design.data;
    if (!d || loaded) return;
    dispatch({ type: "reset", scene: d.scene });
    setName(d.name);
    setPreset(d.preset);
    setAssets(Object.fromEntries(Object.entries(d.assets).map(([k, a]) => [Number(k), { url: a.medium_url, w: a.width, h: a.height }])));
    setLoaded(true);
  }, [design.data, loaded]);

  const size: Size = useMemo(() => {
    const p = presets.data?.find((x) => x.key === preset);
    return p ? { w: p.width, h: p.height } : { w: design.data?.width ?? 800, h: design.data?.height ?? 800 };
  }, [presets.data, preset, design.data]);

  const commit = useCallback((s: Scene) => dispatch({ type: "commit", scene: s }), []);
  const sizeOf = useCallback((assetId: number) => (assets[assetId] ? { w: assets[assetId].w, h: assets[assetId].h } : undefined), [assets]);

  // Autosave 0.8 s after the last change.
  const firstSave = useRef(true);
  const saveNow = useCallback(async () => {
    setSave({ kind: "saving" });
    try {
      await api.saveDesign(id, { scene, name, preset });
      setSave({ kind: "saved", at: new Date().toLocaleTimeString("zh-TW", { hour12: false }) });
      qc.invalidateQueries({ queryKey: ["designs"] });
    } catch (e) {
      setSave({ kind: "error", error: String(e) });
    }
  }, [id, scene, name, preset, qc]);
  useEffect(() => {
    if (!loaded) return;
    if (firstSave.current) { firstSave.current = false; return; }
    const t = setTimeout(saveNow, 800);
    return () => clearTimeout(t);
  }, [loaded, saveNow]);

  function addAsset(a: { id: number; width: number; height: number }, at?: { x: number; y: number }) {
    setAssets((m) => ({ ...m, [a.id]: { url: mediumUrl(a.id), w: a.width, h: a.height } }));
    const layer = newAssetLayer(a.id, { w: a.width, h: a.height }, size, at);
    commit({ ...scene, layers: [...scene.layers, layer] });
    setSelectedId(layer.id);
  }

  function addText() {
    const layer = newTextLayer(size);
    commit({ ...scene, layers: [...scene.layers, layer] });
    setSelectedId(layer.id);
  }

  function changePreset(next: PresetKey) {
    const p = presets.data?.find((x) => x.key === next);
    if (!p || next === preset) return;
    commit(rescaleScene(scene, size, { w: p.width, h: p.height }));
    setPreset(next);
  }

  const del = useCallback(() => {
    if (!selectedId) return;
    commit(removeLayer(scene, selectedId));
    setSelectedId(null);
  }, [selectedId, scene, commit]);

  const dup = useCallback(() => {
    if (!selectedId) return;
    const r = duplicateLayer(scene, selectedId);
    commit(r.scene);
    setSelectedId(r.id);
  }, [selectedId, scene, commit]);

  // Keyboard shortcuts (ignored while typing in a field).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (isTyping(e.target)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); dispatch({ type: e.shiftKey ? "redo" : "undo" }); return; }
      if (mod && e.key.toLowerCase() === "y") { e.preventDefault(); dispatch({ type: "redo" }); return; }
      if (mod && e.key.toLowerCase() === "d") { e.preventDefault(); dup(); return; }
      if (e.key === "Escape") { setSelectedId(null); return; }
      if (!selectedId) return;
      if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); del(); return; }
      const step = e.shiftKey ? 10 : 1;
      const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
      if (d) {
        e.preventDefault();
        const l = scene.layers.find((x) => x.id === selectedId);
        if (l) commit(updateLayer(scene, l.id, { x: l.x + d[0], y: l.y + d[1] }));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, scene, commit, del, dup]);

  async function doExport() {
    if (!canvas.current) return;
    setExporting(true);
    setError(null);
    try {
      await saveNow();
      const t0 = performance.now();
      const blob = await canvas.current.exportImage(format, 0.92);
      const rec = await api.exportDesign(id, blob, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${name || "design"}-${rec.width}x${rec.height}.${format === "png" ? "png" : "jpg"}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      console.info(`export ${rec.width}x${rec.height} ${format} in ${Math.round(performance.now() - t0)} ms`);
      qc.invalidateQueries({ queryKey: ["exports", id] });
      qc.invalidateQueries({ queryKey: ["designs"] });
    } catch (e) {
      setError(`匯出失敗：${String(e)}`);
    } finally {
      setExporting(false);
    }
  }

  if (design.error) return <div className="error">讀取設計失敗：{String(design.error)}</div>;
  if (!loaded || !presets.data) return <div className="row"><span className="spinner" /> 載入中…</div>;

  const selected = scene.layers.find((l) => l.id === selectedId) ?? null;
  const templates = TEMPLATES.filter((t) => t.preset === preset);
  const saveText = { idle: "", saving: "儲存中…", saved: `已儲存 ${save.at ?? ""}`, error: "儲存失敗" }[save.kind];

  return (
    <div className="editor">
      <div className="editor-toolbar">
        <Link to="/designs">← 設計</Link>
        <select value={preset} onChange={(e) => changePreset(e.target.value as PresetKey)} title="換尺寸時圖層會等比例換算">
          {presets.data.map((p) => <option key={p.key} value={p.key}>{p.name} {p.width}×{p.height}</option>)}
        </select>
        <select value="" onChange={(e) => {
          const t = templates.find((x) => x.key === e.target.value);
          if (t) commit(t.apply(scene, size, sizeOf));
        }}>
          <option value="">套用版型…</option>
          {templates.map((t) => <option key={t.key} value={t.key}>{t.name}：{t.hint}</option>)}
        </select>
        <button onClick={addText}>＋ 文字</button>
        <button onClick={() => dispatch({ type: "undo" })} disabled={!hist.past.length} title="Ctrl+Z">↶ 復原</button>
        <button onClick={() => dispatch({ type: "redo" })} disabled={!hist.future.length} title="Ctrl+Y">↷ 重做</button>
        <span className={`muted${save.kind === "error" ? " error-text" : ""}`} title={save.error}>{saveText}</span>
        <span style={{ flex: 1 }} />
        <select value={format} onChange={(e) => setFormat(e.target.value as "png" | "jpeg")}>
          <option value="png">PNG</option>
          <option value="jpeg">JPEG</option>
        </select>
        <button className="primary" onClick={doExport} disabled={exporting}>
          {exporting ? <><span className="spinner" /> 匯出中…</> : `匯出 ${size.w}×${size.h}`}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {design.data!.missing_assets.length > 0 && (
        <div className="error">有 {design.data!.missing_assets.length} 個素材已從素材庫刪除，畫布上以紅框標示，可刪除該圖層或換成其他素材。</div>
      )}

      <div className="editor-body">
        <AssetPicker onAdd={(a: Asset) => addAsset(a)} />
        <DesignCanvas ref={canvas} scene={scene} size={size} assets={assets} selectedId={selectedId}
          onSelect={setSelectedId} onCommit={commit}
          onDropAsset={(a: DraggedAsset, x, y) => addAsset(a, { x, y })} />
        <aside className="editor-side">
          {selected ? (
            <LayerInspector layer={selected}
              onChange={(patch: Partial<Layer>) => commit(updateLayer(scene, selected.id, patch))}
              onOrder={(op) => commit(reorder(scene, selected.id, op))}
              onDuplicate={dup} onDelete={del}
              onFit={selected.type === "asset" && sizeOf(selected.asset_id) ? () => commit(updateLayer(scene, selected.id,
                fitAsset(selected, sizeOf(selected.asset_id)!, { cx: size.w / 2, cy: size.h / 2, w: size.w * 0.8, h: size.h * 0.8 }))) : undefined} />
          ) : (
            <CanvasInspector name={name} onName={setName} background={scene.background}
              onBackground={(b) => commit({ ...scene, background: b })} exports={exportsQ.data ?? []} />
          )}
        </aside>
      </div>
    </div>
  );
}
