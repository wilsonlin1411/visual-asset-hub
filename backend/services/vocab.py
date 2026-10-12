"""Tag vocabulary (backend/vocab/vocabulary.toml): categories, attributes, contexts, rules.

Also maps Chinese query text to vocabulary entries by name / synonym (longest match first).
"""
from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .. import config


@dataclass(frozen=True)
class Entry:
    kind: str                 # category | color | material | style | context
    name: str                 # Chinese display name, stored as the tag value
    prompts: Tuple[str, ...]  # English CLIP phrases (empty for colors)
    synonyms: Tuple[str, ...]
    key: str = ""             # color id (black, white, ...)
    group: str = ""
    coco: Tuple[str, ...] = ()
    exclude_from_search: bool = False
    fallback: bool = False
    scene: bool = True  # contexts only: also matched against the whole photo by CLIP

    @property
    def terms(self) -> Tuple[str, ...]:
        return (self.name, *self.synonyms)


@dataclass(frozen=True)
class Rule:
    contexts: Tuple[str, ...]
    all: Tuple[str, ...] = ()
    any: Tuple[str, ...] = ()
    min_any: int = 1

    def matches(self, categories: Set[str]) -> bool:
        if not set(self.all) <= categories:
            return False
        return not self.any or len(set(self.any) & categories) >= self.min_any

    def describe(self) -> str:
        parts = list(self.all)
        if self.any:
            parts.append(("(" if self.all else "") + "/".join(self.any) +
                         (f" x{self.min_any}" if self.min_any > 1 else "") + (")" if self.all else ""))
        return " + ".join(parts)


@dataclass
class Vocabulary:
    categories: List[Entry]
    colors: List[Entry]
    materials: List[Entry]
    styles: List[Entry]
    contexts: List[Entry]
    rules: List[Rule]
    templates: Dict[str, List[str]]
    objectness: dict
    tagging: dict
    scene_none: List[str]
    digest: str  # changes whenever the file changes (prompt-embedding cache key)
    _terms: List[Tuple[str, Entry]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        terms = []
        for e in (*self.categories, *self.colors, *self.materials, *self.styles, *self.contexts):
            terms += [(t, e) for t in e.terms if t]
        # Longest first so "咖啡廳工作" wins over "咖啡".
        self._terms = sorted(terms, key=lambda te: -len(te[0]))

    @property
    def fallback_category(self) -> Entry:
        return next(c for c in self.categories if c.fallback)

    def category(self, name: str) -> Optional[Entry]:
        return next((c for c in self.categories if c.name == name), None)

    def coco_to_category(self, coco_label: Optional[str]) -> Optional[str]:
        if coco_label is None:
            return None
        return next((c.name for c in self.categories if coco_label in c.coco), None)

    def context_categories(self, contexts: Sequence[str]) -> Set[str]:
        """Categories that take part in a rule producing any of `contexts` (the objects
        "related to" that context, e.g. 小酌 -> 酒瓶, 罐裝飲料)."""
        wanted = set(contexts)
        return {c for r in self.rules if wanted & set(r.contexts) for c in (*r.all, *r.any)}

    @property
    def excluded_categories(self) -> List[str]:
        return [c.name for c in self.categories if c.exclude_from_search]

    def match_terms(self, text: str) -> List[Tuple[str, Entry]]:
        """Vocabulary entries mentioned in `text`; each span of text is consumed once."""
        lowered = text.lower()
        used = [False] * len(lowered)
        found: List[Tuple[str, Entry]] = []
        seen = set()
        for term, entry in self._terms:
            t = term.lower()
            start = lowered.find(t)
            while start != -1:
                end = start + len(t)
                if not any(used[start:end]):
                    for i in range(start, end):
                        used[i] = True
                    if (entry.kind, entry.name) not in seen:
                        seen.add((entry.kind, entry.name))
                        found.append((term, entry))
                    break
                start = lowered.find(t, start + 1)
        return found


def _entries(raw: Sequence[dict], kind: str) -> List[Entry]:
    return [Entry(kind=kind, name=d["name"], prompts=tuple(d.get("prompts", ())),
                  synonyms=tuple(d.get("synonyms", ())), key=d.get("key", ""), group=d.get("group", ""),
                  coco=tuple(d.get("coco", ())), exclude_from_search=d.get("exclude_from_search", False),
                  fallback=d.get("fallback", False), scene=d.get("scene", True)) for d in raw]


def load_vocabulary(path: Path = config.VOCABULARY_PATH) -> Vocabulary:
    data = Path(path).read_bytes()
    raw = tomllib.loads(data.decode("utf-8"))
    vocab = Vocabulary(
        categories=_entries(raw["category"], "category"),
        colors=_entries(raw.get("color", []), "color"),
        materials=_entries(raw.get("material", []), "material"),
        styles=_entries(raw.get("style", []), "style"),
        contexts=_entries(raw["context"], "context"),
        rules=[Rule(contexts=tuple(r["contexts"]), all=tuple(r.get("all", ())), any=tuple(r.get("any", ())),
                    min_any=r.get("min_any", 1)) for r in raw.get("rule", [])],
        templates=raw["templates"],
        objectness=raw["objectness"],
        tagging=raw["tagging"],
        scene_none=list(raw.get("scene", {}).get("none", [])),
        digest=hashlib.sha256(data).hexdigest()[:16],
    )
    _validate(vocab)
    return vocab


def _validate(v: Vocabulary) -> None:
    cats = {c.name for c in v.categories}
    ctxs = {c.name for c in v.contexts}
    if len(cats) != len(v.categories):
        raise ValueError("duplicate category names in vocabulary")
    if sum(c.fallback for c in v.categories) != 1:
        raise ValueError("vocabulary needs exactly one fallback category")
    for c in (*v.categories, *v.materials, *v.styles, *v.contexts):
        if not c.prompts:
            raise ValueError(f"{c.kind} {c.name!r} has no prompts")
    for r in v.rules:
        unknown = (set(r.all) | set(r.any)) - cats
        if unknown:
            raise ValueError(f"rule {r.describe()} uses unknown categories {sorted(unknown)}")
        if set(r.contexts) - ctxs:
            raise ValueError(f"rule {r.describe()} yields unknown contexts {sorted(set(r.contexts) - ctxs)}")
    coco = [lbl for c in v.categories for lbl in c.coco]
    if len(coco) != len(set(coco)):
        raise ValueError("a COCO label is mapped to more than one category")
