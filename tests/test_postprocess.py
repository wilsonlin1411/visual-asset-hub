import cv2
import numpy as np
import pytest

from backend.inference import postprocess as pp


def test_nms_suppresses_overlap_and_keeps_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    assert pp.nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_batched_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(pp.batched_nms(boxes, scores, np.array([0, 1]), 0.5).tolist()) == [0, 1]
    assert pp.batched_nms(boxes, scores, np.array([0, 0]), 0.5).tolist() == [0]


def test_batched_nms_empty():
    assert pp.batched_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), 0.5).size == 0


@pytest.mark.parametrize("h,w", [(1506, 2000), (2000, 1506), (480, 640), (640, 640)])
def test_letterbox_roundtrip(h, w):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    padded, info = pp.letterbox(image, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert info.pad_top + info.resized_shape[0] <= 640
    assert info.pad_left + info.resized_shape[1] <= 640

    # The full content area in input space maps back to the full original image.
    content = np.array([[info.pad_left, info.pad_top,
                         info.pad_left + info.resized_shape[1], info.pad_top + info.resized_shape[0]]])
    back = pp.scale_boxes(content, info)[0]
    np.testing.assert_allclose(back, [0, 0, w, h], atol=1.0)


def test_decode_splits_prediction_tensor():
    a, nc, nm = 6, 3, 2
    preds = np.zeros((1, 4 + nc + nm, a), dtype=np.float32)
    preds[0, :4, 0] = [50, 40, 20, 10]  # cx, cy, w, h
    preds[0, 4 + 1, 0] = 0.9
    preds[0, 4 + nc:, 0] = [1.0, -1.0]
    proto = np.zeros((1, nm, 8, 8), dtype=np.float32)
    head = pp.decode_yolov8_seg({"output0": preds, "output1": proto})
    assert head.scores.shape == (a, nc) and head.coeffs.shape == (a, nm) and head.proto.shape == (nm, 8, 8)
    np.testing.assert_allclose(head.boxes[0], [40, 35, 60, 45])
    np.testing.assert_allclose(head.coeffs[0], [1.0, -1.0])


def test_select_detections_thresholds_and_classes():
    head = pp.HeadOutput(
        boxes=np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        scores=np.array([[0.9, 0.1], [0.8, 0.0], [0.1, 0.2]], dtype=np.float32),
        coeffs=np.zeros((3, 1), dtype=np.float32),
        proto=np.zeros((1, 4, 4), dtype=np.float32),
    )
    sel = pp.select_detections(head, conf_threshold=0.25, iou_threshold=0.5, max_det=10)
    assert sel.class_ids.tolist() == [0]  # box 1 suppressed, box 2 below conf
    np.testing.assert_allclose(sel.scores, [0.9])


def test_process_masks_maps_proto_blob_to_original_and_crops_to_box():
    # 64x64 input, 16x16 proto (1/4 scale), original 32x64 -> gain 1, 16 px top/bottom padding.
    info = pp.LetterboxInfo(orig_shape=(32, 64), input_shape=(64, 64), resized_shape=(32, 64),
                            gain=1.0, pad_top=16, pad_left=0)
    proto = np.full((1, 16, 16), -10.0, dtype=np.float32)
    proto[0, 4:12, 0:8] = 10.0  # left half of the content rows (rows 4..11 = content area)
    coeffs = np.ones((1, 1), dtype=np.float32)

    full = pp.process_masks(proto, coeffs, np.array([[0, 0, 64, 32]], dtype=np.float32), info)[0]
    assert full.box == (0, 0, 64, 32)
    full = full.to_full((32, 64))
    assert full[:, :28].all() and not full[:, 36:].any()

    cropped = pp.process_masks(proto, coeffs, np.array([[0, 0, 16, 32]], dtype=np.float32), info)[0]
    assert cropped.box == (0, 0, 16, 32)
    assert cropped.data.all()


def test_process_masks_empty():
    info = pp.LetterboxInfo((10, 20), (64, 64), (32, 64), 3.2, 16, 0)
    assert pp.process_masks(np.zeros((32, 16, 16)), np.zeros((0, 32)), np.zeros((0, 4)), info) == []


def _process_masks_full_resize(proto, coeffs, boxes, info, threshold=0.5):
    """Reference (stage 1) implementation: resize the whole proto window, then crop."""
    oh, ow = info.orig_shape
    m, ph, pw = proto.shape
    probs = pp.sigmoid((coeffs @ proto.reshape(m, -1)).reshape(-1, ph, pw)).astype(np.float32)
    sy, sx = ph / info.input_shape[0], pw / info.input_shape[1]
    top, left = int(round(info.pad_top * sy)), int(round(info.pad_left * sx))
    bottom = int(round((info.pad_top + info.resized_shape[0]) * sy))
    right = int(round((info.pad_left + info.resized_shape[1]) * sx))
    out = np.zeros((len(coeffs), oh, ow), dtype=bool)
    for i in range(len(coeffs)):
        prob = cv2.resize(probs[i, top:bottom, left:right], (ow, oh), interpolation=cv2.INTER_LINEAR)
        x1, y1 = np.floor(boxes[i, :2]).astype(int).clip(0)
        x2, y2 = np.ceil(boxes[i, 2:]).astype(int)
        out[i, y1:y2, x1:x2] = prob[y1:y2, x1:x2] > threshold
    return out


@pytest.mark.parametrize("orig", [(1506, 2000), (4080, 3072)])
def test_process_masks_roi_matches_full_resize(orig):
    rng = np.random.default_rng(0)
    _, info = pp.letterbox(np.zeros((*orig, 3), np.uint8), (640, 640))
    proto = np.stack([cv2.GaussianBlur(rng.normal(size=(160, 160)).astype(np.float32), (0, 0), 3) * 8
                      for _ in range(8)])
    coeffs = rng.normal(size=(5, 8)).astype(np.float32)
    h, w = orig
    boxes = np.array([[0, 0, w, h], [w * .1, h * .2, w * .6, h * .7], [w * .5, h * .5, w, h],
                      [10.4, 20.6, 300.2, 250.9], [w - 50, h - 50, w, h]], dtype=np.float32)
    ref = _process_masks_full_resize(proto, coeffs, boxes, info)
    ours = pp.process_masks(proto, coeffs, boxes, info)
    for i in range(len(boxes)):
        agree = (ours[i].to_full(orig) == ref[i]).mean()
        assert agree > 0.9995, f"box {i}: agreement {agree:.5f}"
