from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.services.search import parse_query
from backend.services.tagging import COLOR_KEYS, Tagger, classify_colors, color_ratios
from backend.services.vocab import load_vocabulary
from fakes import FakeTextEncoder


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def tagger(vocab):
    return Tagger(vocab, FakeTextEncoder())


def test_vocabulary_loads_and_is_consistent(vocab):
    assert len(vocab.categories) == 44
    assert vocab.fallback_category.name == "其他"
    assert vocab.excluded_categories == ["人物"]
    assert vocab.coco_to_category("cup") == "咖啡杯" and vocab.coco_to_category("dining table") is None
    assert {c.name for c in vocab.contexts} >= {"遠距工作", "咖啡廳", "小酌", "旅行"}


@pytest.mark.parametrize("query,cats,ctxs,attrs", [
    ("和在咖啡廳工作有關的物件", [], ["遠距工作"], []),
    ("辦公桌上的電腦周邊", [], ["辦公桌面"], []),
    ("適合下班小酌的東西", [], ["小酌"], []),
    ("可愛的公仔", ["公仔"], [], ["可愛"]),
    ("晚餐的主菜", ["肉類主餐"], ["用餐"], []),
    ("藍色的東西", [], [], ["藍"]),
    ("帆布提袋", ["提袋"], [], ["布料"]),
    ("手機配件", ["手機殼"], [], []),     # synonym of 手機殼; longest match beats 手機 and 手(人物)
    ("新手機", ["手機"], [], []),         # "手機" must win over the 人物 synonym "手"
    ("保溫杯", ["隨行杯"], [], []),
])
def test_parse_query_longest_match(vocab, query, cats, ctxs, attrs):
    q = parse_query(query, vocab)
    assert (q.categories, q.contexts, q.attributes) == (cats, ctxs, attrs)


def test_rules_fire_on_category_combinations(tagger):
    ctx = lambda cats: {t[1] for t in tagger.rule_contexts(set(cats))}  # noqa: E731
    assert ctx(["筆電", "咖啡杯"]) == {"遠距工作", "咖啡廳"}
    assert ctx(["咖啡杯", "麵包"]) == {"咖啡廳", "甜點時光"}
    assert ctx(["滑鼠"]) == set()                      # needs two desk peripherals
    assert ctx(["滑鼠", "鍵盤"]) == {"辦公桌面"}
    assert ctx(["餐具"]) == set() and ctx(["餐具", "碗"]) == {"用餐"}
    assert ctx(["野生動物"]) == {"戶外", "旅行"}
    assert all(t[3].startswith("rule:") for t in tagger.rule_contexts({"安全帽"}))


def test_rule_evidence_needs_confidence_or_confirmation(tagger):
    row = lambda cat, p, obj=0.9, yolo=None, ys=0.0: SimpleNamespace(  # noqa: E731
        category=cat, category_score=p, objectness=obj, class_name=yolo, score=ys)
    rows = [
        row("筆電", 0.85),                          # confident CLIP: counts
        row("安全帽", 0.65),                        # misread fragment below rule_min_prob: ignored
        row("咖啡杯", 0.30, yolo="cup", ys=0.94),   # YOLO + CLIP agree: counts despite modest CLIP prob
        row("雕像", 0.55, yolo="keyboard", ys=0.8), # YOLO disagrees, CLIP not confident: ignored
        row("甜點", 0.95, obj=0.1),                 # hidden (low objectness): ignored
        row("人物", 0.99),                          # excluded category: ignored
    ]
    assert tagger.rule_categories(rows) == {"筆電", "咖啡杯"}
    asset = lambda cat, p, src: SimpleNamespace(category=cat, category_score=p, category_source=src)  # noqa: E731
    assets = [asset("安全帽", 0.9, "clip"),   # confident: counts
              asset("雕像", 0.4, "clip"),     # confirmed object, unsure category: ignored
              asset("藥品", 0.2, "user"),     # user-set category: counts
              asset("其他", 0.1, "fallback")]
    assert tagger.rule_categories(rows, assets) == {"筆電", "咖啡杯", "安全帽", "藥品"}


def test_scene_none_suppresses_generic_photos(tagger):
    none_vec = tagger._scene_matrix[-1]
    assert tagger.scene_contexts(none_vec) == []
    cafe = tagger.contexts.matrix[tagger.contexts.names.index("咖啡廳")]
    assert [t[1] for t in tagger.scene_contexts(cafe)][0] == "咖啡廳"


def test_category_prefers_trusted_yolo_label_else_clip(tagger):
    names = tagger.categories.names
    cup = tagger.categories.matrix[names.index("咖啡杯")]
    bowl = tagger.categories.matrix[names.index("碗")]
    assert tagger.category(cup).name == "咖啡杯" and tagger.category(cup).source == "clip"
    # YOLO says bowl with high confidence, and bowl is among CLIP's top-3 for a cup/bowl blend
    blend = (cup + bowl) / np.linalg.norm(cup + bowl)
    r = tagger.category(blend, "bowl", 0.9)
    assert (r.name, r.source) == ("碗", "yolo")
    # a low-confidence YOLO label is ignored
    assert tagger.category(cup, "bowl", 0.3).source == "clip"
    # a confident YOLO label cannot override a near-certain CLIP answer
    tumbler = tagger.categories.matrix[names.index("隨行杯")]
    assert tagger.category(tumbler, "cup", 0.95).name == "隨行杯"


def test_objectness_mass_on_object_prompts(tagger):
    obj = tagger._obj_matrix[0]
    bg = tagger._obj_matrix[-1]
    assert tagger.objectness(obj) > 0.9
    assert tagger.objectness(bg) < 0.1


@pytest.mark.parametrize("bgr,key", [
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "gray"), ((0, 0, 220), "red"),
    ((0, 140, 255), "orange"), ((0, 230, 230), "yellow"), ((0, 180, 0), "green"), ((220, 80, 0), "blue"),
    ((160, 40, 120), "purple"), ((190, 170, 255), "pink"), ((30, 70, 120), "brown"), ((170, 215, 235), "beige"),
])
def test_color_classification(bgr, key):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    assert COLOR_KEYS[classify_colors(hsv)[0]] == key


def test_color_tags_use_opaque_pixels_only(tagger):
    bgra = np.zeros((10, 10, 4), np.uint8)
    bgra[:, :6] = (220, 80, 0, 255)   # blue, opaque
    bgra[:, 6:] = (0, 0, 220, 0)      # red but transparent: ignored
    assert color_ratios(bgra) == {"blue": 1.0}
    assert [t[1] for t in tagger.colors(bgra)] == ["藍"]
