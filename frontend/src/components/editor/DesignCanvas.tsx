import Konva from "konva";
import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import { Group, Image as KImage, Layer as KLayer, Line, Rect, Stage, Text as KText, Transformer } from "react-konva";
import useImage from "use-image";
import { gradientPoints, updateLayer, type AssetLayer, type Layer, type Scene, type Size, type TextLayer } from "../../lib/scene";
import { DRAG_TYPE, type DraggedAsset } from "./AssetPicker";

export interface CanvasAsset { url: string; w: number; h: number }

export interface CanvasHandle {
  /** Render the scene at exactly the design size (no selection handles / guides). */
  exportImage(format: "png" | "jpeg", quality?: number): Promise<Blob>;
}

interface Props {
  scene: Scene;
  size: Size;
  assets: Record<number, CanvasAsset>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onCommit: (scene: Scene) => void;
  onDropAsset: (asset: DraggedAsset, x: number, y: number) => void;
}

const SNAP_PX = 8; // screen pixels
// Shared node props (id, drag + transform handlers); react-konva's event prop types are not in NodeConfig.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type NodeProps = Record<string, any>;

function AssetNode({ layer, asset, onLoad, common }: {
  layer: AssetLayer; asset?: CanvasAsset; onLoad: (id: string, done: boolean) => void; common: NodeProps;
}) {
  const [img, status] = useImage(asset?.url ?? "", "anonymous");
  useEffect(() => {
    onLoad(layer.id, !asset || status !== "loading");
    return () => onLoad(layer.id, true);
  }, [asset, status, layer.id, onLoad]);

  if (!asset) {  // asset deleted from the library: keep the layer, show a placeholder
    const h = layer.width * 0.75;
    return (
      <Group {...common} x={layer.x} y={layer.y} offsetX={layer.width / 2} offsetY={h / 2} rotation={layer.rotation}>
        <Rect width={layer.width} height={h} stroke="#c23b30" dash={[12, 8]} strokeWidth={3} fill="rgba(194,59,48,0.06)" />
        <KText text={`素材 #${layer.asset_id} 已刪除`} width={layer.width} y={h / 2 - 14} align="center" fontSize={24} fill="#c23b30" />
      </Group>
    );
  }
  const h = layer.width * (asset.h / asset.w);
  return (
    <KImage {...common} image={img} x={layer.x} y={layer.y} width={layer.width} height={h}
      offsetX={layer.width / 2} offsetY={h / 2} rotation={layer.rotation} scaleX={layer.flip_x ? -1 : 1}
      opacity={layer.opacity}
      shadowEnabled={layer.shadow} shadowColor="#000" shadowOpacity={0.3}
      shadowBlur={layer.width * 0.06} shadowOffsetX={0} shadowOffsetY={layer.width * 0.03} />
  );
}

function TextNode({ layer, common }: { layer: TextLayer; common: NodeProps }) {
  const ref = useRef<Konva.Text>(null);
  // Centre the block vertically: its height is only known after Konva lays the text out.
  useLayoutEffect(() => {
    const n = ref.current;
    if (!n) return;
    n.offsetY(n.height() / 2);
    n.getLayer()?.batchDraw();
  }, [layer.text, layer.font_size, layer.font_family, layer.width, layer.line_height, layer.font_weight]);
  return (
    <KText {...common} ref={ref} text={layer.text} x={layer.x} y={layer.y} width={layer.width}
      offsetX={layer.width / 2} rotation={layer.rotation} fontSize={layer.font_size}
      fontFamily={layer.font_family} fontStyle={layer.font_weight === "bold" ? "bold" : "normal"}
      fill={layer.fill} align={layer.align} lineHeight={layer.line_height} opacity={layer.opacity} wrap="word" />
  );
}

const DesignCanvas = forwardRef<CanvasHandle, Props>(function DesignCanvas(
  { scene, size, assets, selectedId, onSelect, onCommit, onDropAsset }, ref) {
  const wrap = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const loading = useRef<Set<string>>(new Set());
  const [box, setBox] = useState({ w: 800, h: 600 });
  const [guides, setGuides] = useState({ v: false, h: false });

  // Fit the canvas into the available area (never upscale beyond 100%).
  useLayoutEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const scale = Math.min((box.w - 32) / size.w, (box.h - 32) / size.h, 1);

  const onLoad = useRef((id: string, done: boolean) => {
    if (done) loading.current.delete(id);
    else loading.current.add(id);
  }).current;

  // Attach the transformer to the selected node.
  const selected = scene.layers.find((l) => l.id === selectedId);
  useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    const node = selectedId ? stageRef.current?.findOne(`#${selectedId}`) : undefined;
    tr.nodes(node ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selectedId, scene]);

  useImperativeHandle(ref, () => ({
    async exportImage(format, quality = 0.92) {
      for (let i = 0; i < 100 && loading.current.size; i++) await new Promise((r) => setTimeout(r, 100));
      const stage = stageRef.current!;
      const tr = trRef.current;
      const nodes = tr?.nodes() ?? [];
      tr?.nodes([]);
      const raw = stage.toCanvas({ pixelRatio: 1 / scale });
      tr?.nodes(nodes);
      // Copy into a canvas of exactly the design size (guards against rounding of width * pixelRatio).
      const out = document.createElement("canvas");
      out.width = size.w;
      out.height = size.h;
      const ctx = out.getContext("2d")!;
      if (format === "jpeg") {  // JPEG has no alpha: transparent areas become white, not black
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, size.w, size.h);
      }
      ctx.drawImage(raw, 0, 0, size.w, size.h);
      return new Promise<Blob>((resolve, reject) =>
        out.toBlob((b) => (b ? resolve(b) : reject(new Error("export failed"))),
                   format === "png" ? "image/png" : "image/jpeg", quality));
    },
  }), [scale, size]);

  function snap(node: Konva.Node) {
    const t = SNAP_PX / scale;
    const v = Math.abs(node.x() - size.w / 2) < t;
    const h = Math.abs(node.y() - size.h / 2) < t;
    if (v) node.x(size.w / 2);
    if (h) node.y(size.h / 2);
    setGuides((g) => (g.v === v && g.h === h ? g : { v, h }));
  }

  function commitNode(layer: Layer, node: Konva.Node) {
    const patch: Partial<Layer> = { x: node.x(), y: node.y(), rotation: node.rotation() };
    if (layer.type === "asset") {
      patch.width = Math.max(4, layer.width * Math.abs(node.scaleX()));
      node.scaleX(layer.flip_x ? -1 : 1);
    } else {
      patch.width = Math.max(20, layer.width * node.scaleX());
      node.scaleX(1);
    }
    node.scaleY(1);
    onCommit(updateLayer(scene, layer.id, patch));
  }

  const bg = scene.background;
  const grad = gradientPoints(bg.angle, size.w, size.h);

  return (
    <div ref={wrap} className="canvas-area"
      onDragOver={(e) => { if (e.dataTransfer.types.includes(DRAG_TYPE)) e.preventDefault(); }}
      onDrop={(e) => {
        const raw = e.dataTransfer.getData(DRAG_TYPE);
        const stageEl = stageRef.current?.container();
        if (!raw || !stageEl) return;
        e.preventDefault();
        const r = stageEl.getBoundingClientRect();
        const x = Math.min(Math.max((e.clientX - r.left) / scale, 0), size.w);
        const y = Math.min(Math.max((e.clientY - r.top) / scale, 0), size.h);
        onDropAsset(JSON.parse(raw) as DraggedAsset, x, y);
      }}>
      <div className={`canvas-frame${bg.type === "transparent" ? " checker" : ""}`}
        style={{ width: size.w * scale, height: size.h * scale }}>
        <Stage ref={stageRef} width={size.w * scale} height={size.h * scale}
          onMouseDown={(e) => { if (e.target === e.target.getStage()) onSelect(null); }}>
          <KLayer scaleX={scale} scaleY={scale}>
            {bg.type !== "transparent" && (
              <Rect name="background" width={size.w} height={size.h} onMouseDown={() => onSelect(null)}
                {...(bg.type === "gradient"
                  ? { fillLinearGradientStartPoint: grad.start, fillLinearGradientEndPoint: grad.end,
                      fillLinearGradientColorStops: [0, bg.color, 1, bg.color2] }
                  : { fill: bg.color })} />
            )}
            {scene.layers.map((l) => {
              const common: NodeProps = {
                id: l.id, draggable: true,
                onMouseDown: () => onSelect(l.id),
                onTap: () => onSelect(l.id),
                onDragMove: (e: Konva.KonvaEventObject<DragEvent>) => snap(e.target),
                onDragEnd: (e: Konva.KonvaEventObject<DragEvent>) => {
                  setGuides({ v: false, h: false });
                  onCommit(updateLayer(scene, l.id, { x: e.target.x(), y: e.target.y() }));
                },
                onTransformEnd: (e: Konva.KonvaEventObject<Event>) => commitNode(l, e.target),
              };
              return l.type === "asset"
                ? <AssetNode key={l.id} layer={l} asset={assets[l.asset_id]} onLoad={onLoad} common={common} />
                : <TextNode key={l.id} layer={l} common={common} />;
            })}
            {guides.v && <Line points={[size.w / 2, 0, size.w / 2, size.h]} stroke="#f08a24" strokeWidth={1 / scale} dash={[6 / scale, 4 / scale]} listening={false} />}
            {guides.h && <Line points={[0, size.h / 2, size.w, size.h / 2]} stroke="#f08a24" strokeWidth={1 / scale} dash={[6 / scale, 4 / scale]} listening={false} />}
            <Transformer ref={trRef} rotateEnabled flipEnabled={false}
              keepRatio={selected?.type === "asset"}
              enabledAnchors={selected?.type === "text"
                ? ["middle-left", "middle-right"]
                : ["top-left", "top-right", "bottom-left", "bottom-right"]}
              anchorSize={9} borderStroke="#2f6fde" anchorStroke="#2f6fde" rotationSnaps={[0, 90, 180, 270]}
              boundBoxFunc={(oldBox, newBox) => (Math.abs(newBox.width) < 10 ? oldBox : newBox)} />
          </KLayer>
        </Stage>
      </div>
      <div className="canvas-caption">{size.w} × {size.h} px · 顯示 {Math.round(scale * 100)}%</div>
    </div>
  );
});

export default DesignCanvas;
