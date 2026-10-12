"""Asset library CLI (data stays in data/ across runs).

    python -m backend.cli.library ingest tests/images
    python -m backend.cli.library candidates 3
    python -m backend.cli.library confirm 3 --select 0,2,5      (or --labeled / --all)
    python -m backend.cli.library discard 3 --select 7,8
    python -m backend.cli.library list --category bowl
    python -m backend.cli.library delete-asset 12
    python -m backend.cli.library delete-photo 3
    python -m backend.cli.library stats
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from .. import config
from ..services.library import AssetLibrary
from .viz import collect_images


def ids(text: str) -> List[int]:
    return [int(t) for t in text.replace(" ", "").split(",") if t]


def cmd_ingest(lib: AssetLibrary, args) -> None:
    for path in collect_images(args.inputs):
        t0 = time.perf_counter()
        res = lib.add_photo(path)
        if not res.created and res.proposal is None:
            print(f"= {path.name}: already in library as photo {res.photo.id}")
            continue
        cands = lib.candidates(res.photo.id)
        visible = lib.candidates(res.photo.id, include_hidden=False)
        labeled = sum(c.class_name is not None for c in cands)
        contexts = [t[1] for t in lib.photo_contexts(res.photo.id)]
        print(f"+ {path.name}: photo {res.photo.id}, {len(cands)} candidates ({labeled} labeled, "
              f"{len(cands) - len(visible)} hidden)  contexts={contexts}  {(time.perf_counter() - t0) * 1000:.0f}ms")


def cmd_candidates(lib: AssetLibrary, args) -> None:
    photo = lib.get_photo(args.photo_id)
    threshold = lib.objectness_threshold
    print(f"photo {photo.id}: {photo.original_name} {photo.width}x{photo.height} [{photo.status}]"
          f"  contexts={[f'{t[1]}({t[3]})' for t in lib.photo_contexts(photo.id)]}")
    for c in lib.candidates(args.photo_id):
        yolo = f"{c.class_name} {c.score:.2f}" if c.class_name else "-"
        clip = f"{c.category} {c.category_score:.2f}" if c.category else "-"
        obj = f"{c.objectness:.2f}" if c.objectness is not None else "-"
        hidden = "hidden" if c.hidden(threshold) else ""
        print(f"  [{c.idx:2d}] yolo={yolo:<18} clip={clip:<14} obj={obj:<5} {c.status:<9} {hidden:<6}"
              f" bbox={list(c.bbox)}")


def cmd_confirm(lib: AssetLibrary, args) -> None:
    if args.select:
        selected = ids(args.select)
    elif args.labeled:
        selected = [c.idx for c in lib.candidates(args.photo_id, statuses=["pending"]) if c.class_name]
    else:
        selected = None  # --all: every pending candidate
    t0 = time.perf_counter()
    assets = lib.confirm(args.photo_id, selected)
    print(f"photo {args.photo_id}: {len(assets)} assets  {(time.perf_counter() - t0) * 1000:.0f}ms")
    for a in assets:
        print(f"  asset {a.id}: {a.category or '?':<14} {a.width}x{a.height}  {lib.path(a.png_path)}")


def cmd_discard(lib: AssetLibrary, args) -> None:
    print(f"discarded {lib.discard(args.photo_id, ids(args.select))} candidates")


def cmd_list(lib: AssetLibrary, args) -> None:
    assets = lib.list_assets(category=args.category, photo_id=args.photo, tag=args.tag, limit=args.limit)
    for a in assets:
        print(f"  asset {a.id:4d}  photo {a.photo_id:3d}  {a.category or '?':<14} {a.width}x{a.height}"
              f"  label={a.label_model or '-'} mask={a.mask_model}  {a.png_path}")
    print(f"{len(assets)} assets")


def cmd_delete_asset(lib: AssetLibrary, args) -> None:
    print("deleted" if lib.delete_asset(args.asset_id) else "not found")


def cmd_delete_photo(lib: AssetLibrary, args) -> None:
    print("deleted" if lib.delete_photo(args.photo_id) else "not found")


def cmd_stats(lib: AssetLibrary, args) -> None:
    print(json.dumps(lib.stats(), ensure_ascii=False, indent=2))


def cmd_tags(lib: AssetLibrary, args) -> None:
    a = lib.get_asset(args.asset_id)
    print(f"asset {a.id}: {a.category} ({a.category_source} {a.category_score or 0:.2f})  photo {a.photo_id}")
    for t in lib.tags(a.id):
        print(f"  {t.kind:<9} {t.value:<8} {t.score or 0:.2f}  {t.source}")


def cmd_retag(lib: AssetLibrary, args) -> None:
    t0 = time.perf_counter()
    res = lib.retag(args.photos or None, progress=print)
    print(f"retagged {res['photos']} photos / {res['assets']} assets  {time.perf_counter() - t0:.1f}s")


def cmd_thumbs(lib: AssetLibrary, args) -> None:
    t0 = time.perf_counter()
    n_assets = n_photos = 0
    for a in lib.list_assets(limit=1_000_000):
        if not lib.thumbs.asset_path(a.id).is_file():
            lib.thumbs.asset(a.id, lib.path(a.png_path))
            n_assets += 1
    for p in lib.list_photos(limit=1_000_000):
        if not lib.thumbs.photo_path(p.id).is_file():
            lib.thumbs.photo(p.id, lib.path(p.path))
            n_photos += 1
    print(f"created {n_assets} asset thumbnails, {n_photos} photo display images"
          f"  {time.perf_counter() - t0:.1f}s")


def cmd_reembed(lib: AssetLibrary, args) -> None:
    print(f"embedded {lib.reembed()} assets")


def cmd_search(lib: AssetLibrary, args) -> None:
    from ..inference import create_backend
    from ..services.embedding import query_encoder
    from ..services.library import embedding_space
    from ..services.search import SearchService
    from ..services.vocab import load_vocabulary

    svc = SearchService(lib, query_encoder(create_backend("onnx", "cpu")), load_vocabulary(), embedding_space())
    for q in args.queries:
        t0 = time.perf_counter()
        res = svc.search(q, k=args.k, include_excluded=args.include_people)
        p = res.query
        print(f"\n「{q}」  {(time.perf_counter() - t0) * 1000:.0f}ms  categories={p.categories} "
              f"contexts={p.contexts} attributes={p.attributes}")
        for h in res.hits:
            print(f"  {h.score:.3f}  asset {h.asset.id:4d} photo {h.asset.photo_id:3d} {h.asset.category or '?':<8}"
                  f"  {'; '.join(h.reasons)}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visual Asset Hub library")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="add photos and store their candidates")
    p.add_argument("inputs", nargs="+")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("candidates", help="show stored candidates of a photo")
    p.add_argument("photo_id", type=int)
    p.set_defaults(fn=cmd_candidates)

    p = sub.add_parser("confirm", help="turn candidates into assets")
    p.add_argument("photo_id", type=int)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", help="candidate indices, e.g. 0,2,5")
    g.add_argument("--labeled", action="store_true", help="all pending candidates with a class")
    g.add_argument("--all", action="store_true", help="all pending candidates")
    p.set_defaults(fn=cmd_confirm)

    p = sub.add_parser("discard", help="mark candidates as rejected")
    p.add_argument("photo_id", type=int)
    p.add_argument("--select", required=True)
    p.set_defaults(fn=cmd_discard)

    p = sub.add_parser("list", help="list assets")
    p.add_argument("--category")
    p.add_argument("--photo", type=int)
    p.add_argument("--tag")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("delete-asset")
    p.add_argument("asset_id", type=int)
    p.set_defaults(fn=cmd_delete_asset)

    p = sub.add_parser("delete-photo")
    p.add_argument("photo_id", type=int)
    p.set_defaults(fn=cmd_delete_photo)

    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    p = sub.add_parser("tags", help="show an asset's category and tags")
    p.add_argument("asset_id", type=int)
    p.set_defaults(fn=cmd_tags)

    p = sub.add_parser("retag", help="recompute CLIP analysis, contexts, tags and vectors")
    p.add_argument("photos", nargs="*", type=int, help="photo ids (default: all)")
    p.set_defaults(fn=cmd_retag)

    sub.add_parser("reembed", help="compute missing vectors for the current backend").set_defaults(fn=cmd_reembed)
    sub.add_parser("thumbs", help="create missing thumbnails / display images").set_defaults(fn=cmd_thumbs)

    p = sub.add_parser("search", help="Chinese natural-language search")
    p.add_argument("queries", nargs="+")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--include-people", action="store_true")
    p.set_defaults(fn=cmd_search)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    lib = AssetLibrary(args.data_dir)
    try:
        args.fn(lib, args)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1
    finally:
        lib.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
