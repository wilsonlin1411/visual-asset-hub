"""Export the CLIP encoders to ONNX (run in the export env):

    conda run -n vah-export python tools/export_clip.py

Writes to models/:
    clip-vit-b32-image.onnx         pixel_values [B,3,224,224] -> image_embeds [B,512]
    clip-vit-b32-text.onnx          input_ids, attention_mask [B,77] -> text_embeds [B,512]   (English, tagging)
    clip-multilingual-text.onnx     input_ids, attention_mask [B,T] -> text_embeds [B,512]    (Chinese queries)
    clip-vit-b32-tokenizer/, clip-multilingual-tokenizer/   tokenizer.json for the `tokenizers` runtime
Each export is checked against the PyTorch model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CLIP_NAME = "openai/clip-vit-base-patch32"
MULTI_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def onnx_export(module, args, path: Path, input_names, output_names, dynamic_axes, opset: int) -> None:
    module.eval()
    kwargs = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` switch
        torch.onnx.export(module, args, str(path), **kwargs)
    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.0f} MB)")


def check(path: Path, feeds: dict, expected: np.ndarray, name: str) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    cos = (got * expected).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(expected, axis=1))
    print(f"  check {name}: max|diff|={np.abs(got - expected).max():.2e}  min cos={cos.min():.6f}")
    assert cos.min() > 0.9999, f"{name}: ONNX output deviates from PyTorch"


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.vision, self.proj = clip.vision_model, clip.visual_projection

    def forward(self, pixel_values):
        return self.proj(self.vision(pixel_values=pixel_values).pooler_output)


class TextEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.text, self.proj = clip.text_model, clip.text_projection

    def forward(self, input_ids, attention_mask):
        return self.proj(self.text(input_ids=input_ids, attention_mask=attention_mask).pooler_output)


class MultilingualEncoder(torch.nn.Module):
    """DistilmBERT -> mean pooling -> dense(768->512), i.e. the full SentenceTransformer stack."""

    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, input_ids, attention_mask):
        return self.st({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]


def export_clip(opset: int) -> None:
    from transformers import CLIPModel, CLIPTokenizerFast

    print(f"[clip] {CLIP_NAME}")
    clip = CLIPModel.from_pretrained(CLIP_NAME).eval()
    tok = CLIPTokenizerFast.from_pretrained(CLIP_NAME)
    tok.save_pretrained(MODELS_DIR / "clip-vit-b32-tokenizer")

    image_path = MODELS_DIR / "clip-vit-b32-image.onnx"
    pixels = torch.randn(2, 3, 224, 224)
    onnx_export(ImageEncoder(clip), (pixels,), image_path, ["pixel_values"], ["image_embeds"],
                {"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}}, opset)
    with torch.no_grad():
        ref = ImageEncoder(clip)(pixels).numpy()
    check(image_path, {"pixel_values": pixels.numpy()}, ref, "image")

    text_path = MODELS_DIR / "clip-vit-b32-text.onnx"
    enc = tok(["a photo of a cat", "a laptop on a cafe table"], padding="max_length", max_length=77,
              truncation=True, return_tensors="pt")
    onnx_export(TextEncoder(clip), (enc["input_ids"], enc["attention_mask"]), text_path,
                ["input_ids", "attention_mask"], ["text_embeds"],
                {"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "text_embeds": {0: "batch"}}, opset)
    with torch.no_grad():
        ref = TextEncoder(clip)(enc["input_ids"], enc["attention_mask"]).numpy()
    check(text_path, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
          ref, "text")


def export_multilingual(opset: int) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"[multilingual] {MULTI_NAME}")
    st = SentenceTransformer(MULTI_NAME, device="cpu").eval()
    st.tokenizer.save_pretrained(MODELS_DIR / "clip-multilingual-tokenizer")
    path = MODELS_DIR / "clip-multilingual-text.onnx"
    texts = ["和在咖啡廳工作有關的物件", "a laptop on a cafe table"]
    enc = st.tokenizer(texts, padding=True, truncation=True, max_length=st.max_seq_length, return_tensors="pt")
    onnx_export(MultilingualEncoder(st), (enc["input_ids"], enc["attention_mask"]), path,
                ["input_ids", "attention_mask"], ["text_embeds"],
                {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                 "text_embeds": {0: "batch"}}, opset)
    ref = st.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    check(path, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
          ref, "multilingual")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="*", help="clip and/or multilingual (default: both)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    parts = args.parts or ["clip", "multilingual"]
    unknown = set(parts) - {"clip", "multilingual"}
    if unknown:
        parser.error(f"unknown parts: {sorted(unknown)}")
    MODELS_DIR.mkdir(exist_ok=True)
    if "clip" in parts:
        export_clip(args.opset)
    if "multilingual" in parts:
        export_multilingual(args.opset)


if __name__ == "__main__":
    main()
