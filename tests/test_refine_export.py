import dataclasses

import cv2
import numpy as np
import pytest

from backend.config import RefineConfig
from backend.inference import RegionMask
from backend.services.export_png import estimate_foreground, make_cutout
from backend.services.refine import GuidedFilterRefiner, edge_band_px

CFG = RefineConfig(edge_band_ratio=0.02, min_edge_band_px=4)


def scene():
    """Red disc on a green background, plus a blocky (low-res) version of its mask."""
    img = np.zeros((400, 400, 3), np.uint8)
    img[:] = (0, 160, 0)
    cv2.circle(img, (200, 200), 100, (0, 0, 220), -1)
    truth = np.zeros((400, 400), np.uint8)
    cv2.circle(truth, (200, 200), 100, 1, -1)
    coarse = cv2.resize(cv2.resize(truth, (25, 25), interpolation=cv2.INTER_AREA).astype(np.float32),
                        (400, 400), interpolation=cv2.INTER_LINEAR) > 0.5
    return img, truth.astype(bool), RegionMask.from_full(coarse)


@pytest.mark.parametrize("max_work_side", [2048, 160])  # 160 forces the downscaled path
def test_refine_snaps_coarse_mask_to_true_edge(max_work_side):
    img, truth, coarse = scene()
    matte = GuidedFilterRefiner(dataclasses.replace(CFG, max_work_side=max_work_side)).refine(img, coarse)
    full = np.zeros(truth.shape, np.float32)
    x0, y0, x1, y1 = matte.box
    full[y0:y1, x0:x1] = matte.alpha
    assert matte.alpha.dtype == np.float32 and 0 <= matte.alpha.min() and matte.alpha.max() <= 1

    coarse_err = np.mean(coarse.to_full(truth.shape) != truth)
    refined_err = np.mean((full > 0.5) != truth)
    assert refined_err < coarse_err * 0.5, (coarse_err, refined_err)
    assert full[200, 200] == 1.0 and full[5, 5] == 0.0


def test_refined_edge_is_sharp_on_a_sharp_photo_edge():
    img, truth, coarse = scene()
    matte = GuidedFilterRefiner(CFG).refine(img, coarse)
    full = np.zeros(truth.shape, np.float32)
    x0, y0, x1, y1 = matte.box
    full[y0:y1, x0:x1] = matte.alpha
    row = full[200, 90:110]  # true edge at x=100
    soft = np.count_nonzero((row > 0) & (row < 1))
    assert soft <= 6, row  # only the intentional feather (sigma 1.5) remains


def test_edge_band_scales_with_image():
    assert edge_band_px((4080, 3072), RefineConfig()) == round(0.006 * 4080)
    assert edge_band_px((100, 100), RefineConfig()) == RefineConfig().min_edge_band_px


def test_cutout_is_tight_rgba_with_padding():
    img, _, coarse = scene()
    matte = GuidedFilterRefiner(CFG).refine(img, coarse)
    cut = make_cutout(img, matte, padding=8)
    assert cut.rgba.dtype == np.uint8 and cut.rgba.shape[2] == 4
    x0, y0, x1, y1 = cut.bbox
    assert abs(x0 - 100) <= 3 and abs(x1 - 301) <= 3 and abs(y0 - 100) <= 3 and abs(y1 - 301) <= 3
    assert cut.rgba.shape[:2] == (y1 - y0 + 16, x1 - x0 + 16)
    assert cut.origin == (x0 - 8, y0 - 8)
    a = cut.rgba[..., 3]
    assert (a[:8] == 0).all() and (a[:, :8] == 0).all()
    assert (cut.rgba[a == 0, :3] == 0).all()


@pytest.mark.parametrize("sigma,height", [(4, 200), (16, 200), (16, 9000)])
def test_decontamination_removes_background_tint_from_edges(sigma, height):
    # sigma 16: low-resolution blur path; height 9000: >32767 edge pixels (remap size limit).
    roi = np.zeros((height, 200, 3), np.uint8)
    roi[:, :100] = (0, 0, 220)  # red object
    roi[:, 100:] = (0, 160, 0)  # green background
    alpha = np.zeros((height, 200), np.float32)
    alpha[:, :99] = 1.0
    alpha[:, 99:103] = 0.5  # soft edge sitting on green pixels
    out = estimate_foreground(roi, alpha, sigma=sigma)
    np.testing.assert_array_equal(out[:, :99], roi[:, :99])  # opaque untouched
    np.testing.assert_array_equal(out[:, 103:], roi[:, 103:])  # transparent untouched
    assert out[:, 100:103, 1].max() < 40 and out[:, 100:103, 2].min() > 180  # edge now red, not green
