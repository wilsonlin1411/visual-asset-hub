"""Verify the runtime CLIP path (tokenizers + ONNX via InferenceBackend) against the
reference PyTorch models, and sanity-check zero-shot / cross-lingual behaviour.
Run in the export env (needs transformers + sentence-transformers):

    conda run -n vah-export python tools/check_clip_runtime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.imageio import imread  # noqa: E402
from backend.inference import create_backend  # noqa: E402
from backend.services.embedding import ClipImageEncoder, clip_text_encoder, query_encoder  # noqa: E402


def cos_rows(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def main() -> int:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    cpu = create_backend("onnx", "cpu")
    img_enc = ClipImageEncoder(create_backend("onnx", "cpu"))
    en = clip_text_encoder(cpu)
    zh = query_encoder(cpu)

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    st = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device="cpu")

    ok = True
    texts = ["a photo of a pudding.", "a laptop on a cafe table", "an empty gray image."]
    with torch.no_grad():
        t = proc.tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt")
        ref = clip.text_projection(clip.text_model(**t).pooler_output).numpy()
    c = cos_rows(en.encode(texts), ref).min()
    print(f"english text encoder vs transformers: min cos {c:.6f}")
    ok &= c > 0.9999

    zh_texts = ["和在咖啡廳工作有關的物件", "可愛的公仔", "a laptop"]
    c = cos_rows(zh.encode(zh_texts), st.encode(zh_texts)).min()
    print(f"multilingual encoder vs sentence-transformers: min cos {c:.6f}")
    ok &= c > 0.9999

    photo = imread(ROOT / "tests/images/pudding.jpg")
    ours = img_enc.encode_photo(photo)
    with torch.no_grad():
        pv = proc(images=Image.fromarray(photo[..., ::-1]), return_tensors="pt")["pixel_values"]
        ref = clip.visual_projection(clip.vision_model(pixel_values=pv).pooler_output).numpy()[0]
    c = float(ours @ (ref / np.linalg.norm(ref)))
    print(f"image encoder (our preprocessing) vs transformers pipeline: cos {c:.4f}")
    ok &= c > 0.97  # resize kernels differ slightly (cv2 vs PIL bicubic)

    labels_en = ["a photo of a pudding.", "a photo of a laptop.", "a photo of a cat.", "a photo of a helmet."]
    labels_zh = ["布丁", "筆電", "貓", "安全帽"]
    p_en = ours @ en.encode(labels_en).T
    p_zh = ours @ zh.encode(labels_zh).T
    print("zero-shot (english prompts):", dict(zip(labels_zh, np.round(p_en, 3))))
    print("zero-shot (chinese queries):", dict(zip(labels_zh, np.round(p_zh, 3))))
    ok &= int(p_en.argmax()) == 0 and int(p_zh.argmax()) == 0

    pair = zh.encode(["一個布丁"]) @ en.encode(["a pudding"]).T
    print(f"cross-lingual: '一個布丁' vs 'a pudding' cos {float(pair[0, 0]):.3f}")
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
