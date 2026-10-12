"""Host-side pre/post-processing for YOLOv8-seg style models (also FastSAM).

Pipeline: letterbox -> backend.infer -> decode_* -> select_detections (conf + NMS)
          -> scale_boxes -> process_masks.

`decode_*` is the only backend-format-specific step. Every decoder returns a
`HeadOutput`; everything after it is shared, so adding the Hailo raw-tensor decoder
later does not touch NMS or mask assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .masks import RegionMask

PAD_COLOR = (114, 114, 114)


# --------------------------------------------------------------------------- letterbox

@dataclass(frozen=True)
class LetterboxInfo:
    orig_shape: Tuple[int, int]     # (h, w) of the source image
    input_shape: Tuple[int, int]    # (h, w) fed to the model
    resized_shape: Tuple[int, int]  # (h, w) of the image content inside the padding
    gain: float                     # resized / original
    pad_top: int
    pad_left: int


def letterbox(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio and pad to `input_shape`. Works on any 3-channel uint8 image."""
    h, w = image.shape[:2]
    in_h, in_w = input_shape
    gain = min(in_h / h, in_w / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (in_w - new_w) / 2, (in_h - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    info = LetterboxInfo((h, w), (in_h, in_w), (new_h, new_w), gain, top, left)
    return padded, info


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from model-input pixels back to original image pixels."""
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_left) / info.gain
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_top) / info.gain
    h, w = info.orig_shape
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


# ------------------------------------------------------------------------------ decode

@dataclass
class HeadOutput:
    boxes: np.ndarray   # (A, 4) xyxy in model-input pixels
    scores: np.ndarray  # (A, C) per-class confidence in [0, 1]
    coeffs: np.ndarray  # (A, M) mask coefficients
    proto: np.ndarray   # (M, Hp, Wp) mask prototypes


def decode_yolov8_seg(outputs: Dict[str, np.ndarray]) -> HeadOutput:
    """Decode the standard Ultralytics ONNX export: preds [1, 4+C+M, A] and proto [1, M, Hp, Wp].

    Box regression (DFL) and class sigmoid are already inside this graph; the Hailo HEF
    does not include them and will need its own decoder producing the same HeadOutput.
    """
    preds = proto = None
    for arr in outputs.values():
        if arr.ndim == 3:
            preds = arr
        elif arr.ndim == 4:
            proto = arr
    if preds is None or proto is None:
        shapes = {k: v.shape for k, v in outputs.items()}
        raise ValueError(f"expected a rank-3 prediction and rank-4 proto tensor, got {shapes}")

    proto = proto[0]
    num_masks = proto.shape[0]
    p = preds[0].T  # Ultralytics layout is channels-first: (4+C+M, A) -> (A, 4+C+M)
    num_classes = p.shape[1] - 4 - num_masks
    if num_classes < 1:
        raise ValueError(f"prediction width {p.shape[1]} too small for {num_masks} mask coeffs")
    return HeadOutput(
        boxes=xywh_to_xyxy(p[:, :4]),
        scores=p[:, 4:4 + num_classes],
        coeffs=p[:, 4 + num_classes:],
        proto=proto,
    )


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xy, half = xywh[:, :2], xywh[:, 2:4] / 2
    return np.concatenate([xy - half, xy + half], axis=1)


# ------------------------------------------------------------------------- NMS / select

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[box_iou(boxes[i], boxes[rest]) <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                iou_threshold: float) -> np.ndarray:
    """Per-class NMS via the coordinate-offset trick (boxes of different classes never overlap)."""
    if boxes.size == 0:
        return np.zeros(0, dtype=np.int64)
    offset = class_ids[:, None].astype(np.float32) * (float(boxes.max()) + 1.0)
    return nms(boxes + offset, scores, iou_threshold)


@dataclass
class Selection:
    boxes: np.ndarray      # (N, 4) xyxy, model-input pixels
    scores: np.ndarray     # (N,)
    class_ids: np.ndarray  # (N,)
    coeffs: np.ndarray     # (N, M)


def select_detections(head: HeadOutput, conf_threshold: float, iou_threshold: float,
                      max_det: int, max_nms: int = 30000) -> Selection:
    class_ids = head.scores.argmax(axis=1)
    scores = head.scores[np.arange(len(class_ids)), class_ids]
    idx = np.nonzero(scores > conf_threshold)[0]
    if idx.size > max_nms:
        idx = idx[scores[idx].argsort()[::-1][:max_nms]]
    keep = idx[batched_nms(head.boxes[idx], scores[idx], class_ids[idx], iou_threshold)][:max_det]
    return Selection(head.boxes[keep], scores[keep], class_ids[keep], head.coeffs[keep])


# ------------------------------------------------------------------------------- masks

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def process_masks(proto: np.ndarray, coeffs: np.ndarray, boxes_orig: np.ndarray,
                  info: LetterboxInfo, threshold: float = 0.5) -> List[RegionMask]:
    """Assemble instance masks at original resolution, one crop per box.

    mask = sigmoid(coeffs @ proto), strip letterbox padding in proto space, upsample to
    the original image, keep only the instance box, then threshold. Only the box region
    is upsampled (an affine warp reproducing cv2.resize's pixel mapping), so cost scales
    with box size rather than image size. Proto is 1/4 input resolution, so edges are
    coarse; the capture pipeline refines them before compositing PNGs.
    """
    n = len(coeffs)
    if n == 0:
        return []
    oh, ow = info.orig_shape

    m, ph, pw = proto.shape
    probs = sigmoid((coeffs @ proto.reshape(m, -1)).reshape(n, ph, pw)).astype(np.float32)

    sy, sx = ph / info.input_shape[0], pw / info.input_shape[1]
    top, left = int(round(info.pad_top * sy)), int(round(info.pad_left * sx))
    bottom = int(round((info.pad_top + info.resized_shape[0]) * sy))
    right = int(round((info.pad_left + info.resized_shape[1]) * sx))
    # Source pixels per destination pixel when stretching the unpadded proto window to the original.
    kx, ky = (right - left) / ow, (bottom - top) / oh

    masks = []
    for i in range(n):
        x1, y1 = np.floor(boxes_orig[i, :2]).astype(int).clip(0)
        x2, y2 = np.ceil(boxes_orig[i, 2:]).astype(int)
        x2, y2 = min(x2, ow), min(y2, oh)
        if x2 <= x1 or y2 <= y1:
            masks.append(RegionMask(np.zeros((0, 0), dtype=bool), int(x1), int(y1)))
            continue
        # dst (u, v) inside the box -> src = ((x1 + u) + 0.5) * k - 0.5, same convention as cv2.resize.
        warp = np.float32([[kx, 0, (x1 + 0.5) * kx - 0.5],
                           [0, ky, (y1 + 0.5) * ky - 0.5]])
        prob = cv2.warpAffine(probs[i, top:bottom, left:right], warp, (int(x2 - x1), int(y2 - y1)),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        masks.append(RegionMask(prob > threshold, int(x1), int(y1)))
    return masks
