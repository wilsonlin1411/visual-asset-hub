"""Chinese natural-language asset search.

1. Match vocabulary terms in the query (categories / attributes / contexts, by name or synonym).
2. Encode the query with the multilingual CLIP text encoder (same space as the image vectors).
3. Candidate set = vector top-N  UNION  assets carrying a matched category / tag.
4. Score = cosine + bonus for each matched category / context / attribute; return top-k
   with the reasons, so the UI (and tests) can show why something matched.
The "人物" category (hands etc.) is excluded unless the query asks for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .. import config
from ..db import Asset, Tag, repo
from ..db.vector_index import EmbeddingSpace
from .embedding import TextEncoder
from .vocab import Entry, Vocabulary


@dataclass
class ParsedQuery:
    text: str
    categories: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)  # colours, materials, styles
    contexts: List[str] = field(default_factory=list)
    matched_terms: Dict[str, str] = field(default_factory=dict)  # term -> "kind:name"


@dataclass
class SearchHit:
    asset: Asset
    score: float
    cosine: float
    reasons: List[str]
    tags: List[Tag]


@dataclass
class SearchResult:
    query: ParsedQuery
    hits: List[SearchHit]
    excluded_categories: List[str]


def parse_query(text: str, vocab: Vocabulary) -> ParsedQuery:
    q = ParsedQuery(text=text)
    for term, entry in vocab.match_terms(text):
        q.matched_terms[term] = f"{entry.kind}:{entry.name}"
        if entry.kind == "category":
            q.categories.append(entry.name)
        elif entry.kind == "context":
            q.contexts.append(entry.name)
        else:
            q.attributes.append(entry.name)
    return q


class SearchService:
    def __init__(self, library, query_encoder: TextEncoder, vocab: Vocabulary, space: EmbeddingSpace,
                 cfg: config.SearchConfig = config.SEARCH):
        self.library = library
        self.encoder = query_encoder
        self.vocab = vocab
        self.space = space
        self.cfg = cfg

    def search(self, text: str, k: Optional[int] = None, category: Optional[str] = None,
               tag: Optional[str] = None, include_excluded: bool = False) -> SearchResult:
        k = k or self.cfg.default_k
        db = self.library.db
        q = parse_query(text, self.vocab)
        excluded = [] if include_excluded else [c for c in self.vocab.excluded_categories if c not in q.categories]

        # Explicit filters restrict the candidate set; query terms only boost.
        allowed: Optional[set] = None
        if category is not None or tag is not None:
            allowed = {a.id for a in repo.list_assets(db, category=category, tag=tag, limit=1_000_000)}

        qvec = self.encoder.encode([text])[0]
        pool = {h.asset_id for h in self.library.index.search(qvec, self.space, self.cfg.candidates,
                                                               asset_ids=allowed)}
        related = list(self.vocab.context_categories(q.contexts))
        pool |= set(repo.asset_ids_matching(db, q.categories + related, q.contexts + q.attributes))
        if allowed is not None:
            pool &= allowed
        if not pool:
            return SearchResult(q, [], excluded)

        cosines = {h.asset_id: h.score for h in self.library.index.search(qvec, self.space, len(pool),
                                                                          asset_ids=pool)}
        context_objects = self.vocab.context_categories(q.contexts)
        assets = repo.get_assets(db, list(cosines))
        tags = repo.tags_for_assets(db, list(cosines))

        hits = []
        for aid, cos in cosines.items():
            asset = assets[aid]
            if asset.category in excluded:
                continue
            score, reasons = cos, [f"相似度 {cos:.3f}"]
            if asset.category in q.categories:
                score += self.cfg.category_bonus
                reasons.append(f"類別 {asset.category}")
            if asset.category in context_objects:
                score += self.cfg.context_object_bonus
                reasons.append(f"{asset.category} 屬於「{'、'.join(q.contexts)}」相關物件")
            values = {t.value: t for t in tags[aid]}
            for ctx in q.contexts:
                if ctx in values:
                    score += self.cfg.context_bonus
                    reasons.append(f"情境 {ctx}（{values[ctx].source}）")
            for attr in q.attributes:
                if attr in values:
                    score += self.cfg.attribute_bonus
                    reasons.append(f"屬性 {attr}")
            hits.append(SearchHit(asset, score, cos, reasons, tags[aid]))
        hits.sort(key=lambda h: -h.score)
        return SearchResult(q, hits[:k], excluded)
