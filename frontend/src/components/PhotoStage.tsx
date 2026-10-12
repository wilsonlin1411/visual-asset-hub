import { useEffect, useMemo, useRef, useState } from "react";
import type { Candidate, Photo } from "../api";
import { boxToPercent, clientToPhoto, pickAt, type AlphaMask } from "../lib/geometry";

const HIT_SIDE = 256; // masks are downsampled to this for hit-testing (memory stays small)

/** Load each candidate's overlay PNG once into a small alpha bitmap for point-in-mask tests. */
function useHitMasks(cands: Candidate[]): Map<number, AlphaMask> {
  const [masks, setMasks] = useState<Map<number, AlphaMask>>(new Map());
  const key = cands.map((c) => c.idx).join(",");
  useEffect(() => {
    let cancelled = false;
    const loaded = new Map<number, AlphaMask>();
    cands.forEach((c) => {
      const img = new Image();
      img.onload = () => {
        if (cancelled) return;
        const s = Math.min(1, HIT_SIDE / Math.max(img.width, img.height));
        const w = Math.max(1, Math.round(img.width * s));
        const h = Math.max(1, Math.round(img.height * s));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
        ctx.drawImage(img, 0, 0, w, h);
        loaded.set(c.idx, { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data });
        setMasks(new Map(loaded));
      };
      img.src = c.overlay_url;
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return masks;
}

interface Props {
  photo: Photo;
  candidates: Candidate[];      // pickable (visible) candidates
  allCandidates: Candidate[];   // for confirmed overlays
  selected: ReadonlySet<number>;
  hovered: number | null;
  onHover: (idx: number | null) => void;
  onToggle: (idx: number) => void;
}

export default function PhotoStage({ photo, candidates, allCandidates, selected, hovered, onHover, onToggle }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const masks = useHitMasks(candidates);
  const pickable = useMemo(() => candidates.filter((c) => c.status !== "confirmed"), [candidates]);

  function at(e: React.MouseEvent): Candidate | undefined {
    const rect = ref.current!.getBoundingClientRect();
    const [x, y] = clientToPhoto(e.clientX, e.clientY, rect, photo.width, photo.height);
    return pickAt(pickable, masks, x, y);
  }

  const overlay = (c: Candidate, cls: string) => (
    <div key={`${cls}-${c.idx}`} className={`overlay ${cls}`}
      style={{ ...boxToPercent(c.bbox, photo.width, photo.height),
               WebkitMaskImage: `url(${c.overlay_url})`, maskImage: `url(${c.overlay_url})` }} />
  );
  const hoveredCand = allCandidates.find((c) => c.idx === hovered);

  return (
    <div className="stage-wrap">
      <div ref={ref} className="stage"
        // Portrait photos would otherwise fill several screens: cap the height at the viewport.
        style={{ aspectRatio: `${photo.width} / ${photo.height}`,
                 maxWidth: `calc((100vh - 150px) * ${photo.width / photo.height})`, margin: "0 auto" }}
        onMouseMove={(e) => onHover(at(e)?.idx ?? null)}
        onMouseLeave={() => onHover(null)}
        onClick={(e) => { const c = at(e); if (c) onToggle(c.idx); }}>
        <img className="photo" src={photo.display_url} alt={photo.original_name} draggable={false} />
        {allCandidates.filter((c) => c.status === "confirmed").map((c) => overlay(c, "confirmed"))}
        {allCandidates.filter((c) => selected.has(c.idx)).map((c) => overlay(c, "selected"))}
        {hoveredCand && overlay(hoveredCand, "hovered")}
        {hoveredCand && <div className="outline" style={boxToPercent(hoveredCand.bbox, photo.width, photo.height)} />}
      </div>
      <div className="stage-legend">
        <span><i style={{ background: "var(--hover)" }} />游標所在</span>
        <span><i style={{ background: "var(--accent)" }} />已選取</span>
        <span><i style={{ background: "var(--ok)" }} />已確認為素材</span>
        <span>點照片上的物件即可選取</span>
      </div>
    </div>
  );
}
