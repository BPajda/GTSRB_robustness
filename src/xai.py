"""Grad-CAM, share of the CAM inside the sign ROI, confidence and the rejection threshold (KM3).

    python -m src.xai [--smoke] [--batch-size 128]

Models: the final model (resnet18_pretrained_A2) and the baseline (simple_cnn_scratch_A0).
Grad-CAM from pytorch-grad-cam on gradcam_target_layer(), target = predicted class, in
batches on the GPU; maps are upsampled to the 64x64 input and min-max normalized to [0, 1].
All predictions here run in fp32 (no autocast, no TF32), the numerics of CPU inference in
the app, so a few borderline images may differ from the bf16 predictions of src.evaluate
(the count is in run_info.json).

ROI indicator per image (box = the annotated sign ROI rescaled to 64x64, integer pixels):
    energy_in_roi = sum of the map inside the ROI / sum of the whole map
    roi_area      = ROI area / image area (what a uniform map would give)
    lift          = energy_in_roi / roi_area (1 = no preference for the sign)
computed on the whole clean test set and on the test set under each corruption at level 2.
Under occlusion the black box lies inside the ROI: those rows are flagged (occluder_in_roi)
and the share of the map on the box itself (occluder_energy) is reported as well.

The rejection threshold for the app (predictions whose max softmax is below it are rejected)
is chosen on the validation set only: the lowest threshold that keeps the accuracy of the
kept predictions on the corrupted validation set (15 corruptions pooled) at >= 99.5%.
On the test set its effect is only reported.

Outputs in results/xai/ (results/_smoke/xai/ with --smoke):
    roi_energy.csv       model x condition x correct/wrong: energy_in_roi, roi_area, lift, n
    roi_energy.png       energy_in_roi of correct vs wrong predictions, clean test set
    error_gallery.png    the 24 most confident errors of the final model, grouped by class pair
    gradcam_examples.png 8 correct examples from different sign groups, final vs baseline
    corruption_shift.png 4 test images, clean and 5 corruptions at levels 2 and 3, final model
    confidence.json/png  max softmax of correct and wrong predictions, reliability diagram, ECE
    threshold.json/png   coverage vs accuracy when rejecting low-confidence predictions
    run_info.json        settings, Grad-CAM layers, timings
--smoke: the 1-epoch smoke models of results/_smoke/ on ~500 test and ~500 validation images.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from src.class_names_pl import GROUP_OF, SIGN_GROUPS_PL, class_label  # noqa: E402
from src.corruptions import CORRUPTION_LABELS_PL, CORRUPTIONS, LEVELS, CorruptedDataset, level_label_pl, \
    roi_to_pixels  # noqa: E402
from src.data import DATA_DIR, IMG_SIZE, PROJECT_ROOT, SEED, GTSRBDataset, cached_dataset, \
    make_loader  # noqa: E402
from src.evaluate import load_model  # noqa: E402
from src.final_eval import BASELINE_RUN, FINAL_RUN, FINAL_TEST_DIR, RUN_LABELS, fmt_int  # noqa: E402
from src.models import gradcam_target_layer  # noqa: E402
from src.summarize import AXIS, INK, INK_2, pct_axis  # noqa: E402
from src.train import SMOKE_N_VAL, get_device, results_root  # noqa: E402

XAI_RUNS = (FINAL_RUN, BASELINE_RUN)
XAI_DIR = "xai"
CAM_LEVEL = 2             # corruption level of the ROI analysis
SHIFT_LEVELS = (2, 3)     # levels shown in corruption_shift.png
TARGET_ACC = 0.995        # accuracy of the kept predictions on the corrupted validation set
EXTRA_TARGETS = (0.99, 0.999)
N_BINS = 15               # ECE and reliability diagram
MIN_BIN_N = 20            # reliability bins with fewer predictions are drawn hollow
N_GALLERY, GALLERY_PER_ROW = 24, 4
N_SHIFT_STABLE, N_SHIFT_FLIP = 2, 2
BATCH_SIZE = 128

CONDITIONS = [(None, 0)] + [(c, lv) for c in CORRUPTIONS for lv in LEVELS]


def cond_key(ctype: str | None, level: int) -> str:
    return "clean" if ctype is None else f"{ctype}_{level}"


KEY_TO_COND = {cond_key(c, lv): (c, lv) for c, lv in CONDITIONS}
CAM_KEYS = ["clean"] + [cond_key(c, CAM_LEVEL) for c in CORRUPTIONS]
CORRUPTED_KEYS = [k for k in KEY_TO_COND if k != "clean"]

# Correct vs wrong is good/bad: blue (fill) and status red (outline) - distinct also under CVD.
CORRECT_COLOR, WRONG_COLOR = "#2a78d6", "#d03b3b"
CLEAN_COLOR, CORRUPT_COLOR = "#52514e", "#eb6834"
VAL_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"
ROI_COLOR = "#35d0f5"       # cyan does not occur in the CAM colormap
CAM_CMAP = plt.get_cmap("inferno")  # perceptually uniform, no rainbow


def cond_label_pl(key: str) -> str:
    ctype, level = KEY_TO_COND[key]
    if ctype is None:
        return "czyste"
    return f"{CORRUPTION_LABELS_PL[ctype]}, poziom {level} ({level_label_pl(ctype, level)})"


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


# ---------------------------------------------------------------------------
# Predictions and Grad-CAM over a dataset
# ---------------------------------------------------------------------------

@dataclass
class CondResult:
    """Per-image results of one model under one condition (clean or one corruption level)."""
    label: np.ndarray
    pred: np.ndarray
    conf: np.ndarray                          # max softmax, float64
    energy: np.ndarray | None = None          # energy_in_roi, NaN where the map is empty
    roi_area: np.ndarray | None = None
    occ_energy: np.ndarray | None = None      # share of the map on the occluder (occlusion only)
    occ_area: np.ndarray | None = None

    @property
    def correct(self) -> np.ndarray:
        return self.pred == self.label


def roi_boxes(base: GTSRBDataset) -> np.ndarray:
    """Integer pixel ROI boxes (x1, y1, x2, y2) in the 64x64 image, the same as used by occlusion."""
    return np.array([roi_to_pixels(r, IMG_SIZE, IMG_SIZE) for r in base.rois], dtype=np.int64)


def box_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / IMG_SIZE ** 2


def box_masks(boxes: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    r = np.arange(size)
    rows = (r >= boxes[:, 1, None]) & (r < boxes[:, 3, None])
    cols = (r >= boxes[:, 0, None]) & (r < boxes[:, 2, None])
    return rows[:, :, None] & cols[:, None, :]


def energy_share(maps: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Sum of each map inside its box / sum of the whole map (NaN for an all-zero map)."""
    total = maps.sum(axis=(1, 2), dtype=np.float64)
    inside = (maps * box_masks(boxes)).sum(axis=(1, 2), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, inside / total, np.nan)


def top1(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class and its softmax probability (computed in float64)."""
    conf, pred = torch.softmax(logits.double(), dim=1).max(dim=1)
    return pred.cpu().numpy(), conf.cpu().numpy()


def make_cam(model: nn.Module) -> GradCAM:
    return GradCAM(model=model, target_layers=[gradcam_target_layer(model)])


def gradcam(cam: GradCAM, x: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """Grad-CAM maps [B, 64, 64] in [0, 1] for the predicted class, and the fp32 logits."""
    maps = cam(input_tensor=x, targets=None)
    logits = cam.outputs.detach().float()
    cam.outputs = None  # drops the graph kept by retain_graph=True before the next batch
    return maps, logits


def cam_for(model: nn.Module, x: torch.Tensor, device: torch.device
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grad-CAM maps, predictions and confidences of a small batch of normalized images."""
    with make_cam(model) as cam:
        maps, logits = gradcam(cam, x.to(device).contiguous(memory_format=torch.channels_last))
    pred, conf = top1(logits)
    return maps, pred, conf


def run_pass(models: dict[str, nn.Module], base: GTSRBDataset, cam_keys: list[str], device: torch.device,
             batch_size: int, tag: str) -> dict[str, dict[str, CondResult]]:
    """Predictions of every model under every condition; Grad-CAM + ROI shares for cam_keys.

    Each batch is corrupted once on the CPU and passed to all models.
    """
    boxes = roi_boxes(base)
    areas = box_area(boxes)
    out = {run: {} for run in models}
    for ctype, level in CONDITIONS:
        key = cond_key(ctype, level)
        ds = base if ctype is None else CorruptedDataset(base, ctype, level)
        with_cam = key in cam_keys
        occ = np.array([ds.occluder(i) for i in range(len(ds))]) if with_cam and ctype == "occlusion" else None
        # hooks only while needed: they also fire (and copy activations) in plain forward passes
        cams = {run: make_cam(m) for run, m in models.items()} if with_cam else {}
        parts = {run: defaultdict(list) for run in models}
        t0, start = time.perf_counter(), 0
        for x, y, _roi in make_loader(ds, batch_size, shuffle=False, num_workers=0):
            sl = slice(start, start + len(y))
            start += len(y)
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            for run, model in models.items():
                p = parts[run]
                if with_cam:
                    maps, logits = gradcam(cams[run], x)
                    p["energy"].append(energy_share(maps, boxes[sl]))
                    if occ is not None:
                        p["occ_energy"].append(energy_share(maps, occ[sl]))
                else:
                    with torch.inference_mode():
                        logits = model(x).float()
                pred, conf = top1(logits)
                p["pred"].append(pred)
                p["conf"].append(conf)
        for cam in cams.values():
            cam.activations_and_grads.release()
        for run in models:
            p = {k: np.concatenate(v) for k, v in parts[run].items()}
            out[run][key] = CondResult(label=base.labels.copy(), pred=p["pred"], conf=p["conf"],
                                       energy=p.get("energy"), roi_area=areas if with_cam else None,
                                       occ_energy=p.get("occ_energy"),
                                       occ_area=box_area(occ) if occ is not None else None)
        accs = "  ".join(f"{run} {out[run][key].correct.mean():.4f}" for run in models)
        print(f"[{tag}] {key:<16} {'Grad-CAM' if with_cam else 'forward '} {time.perf_counter() - t0:5.1f} s | "
              f"acc {accs}")
    return out


def pooled(res: dict[str, CondResult], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Confidences and correctness of several conditions together."""
    return (np.concatenate([res[k].conf for k in keys]), np.concatenate([res[k].correct for k in keys]))


# ---------------------------------------------------------------------------
# ROI energy table
# ---------------------------------------------------------------------------

def _stat(f, a: np.ndarray) -> float:
    return float(f(a)) if len(a) else float("nan")


def roi_energy_table(test_res: dict[str, dict[str, CondResult]]) -> pd.DataFrame:
    rows = []
    for run in XAI_RUNS:
        for key in CAM_KEYS:
            r = test_res[run][key]
            ctype, level = KEY_TO_COND[key]
            for subset, mask in (("all", np.ones(len(r.label), bool)), ("correct", r.correct),
                                 ("wrong", ~r.correct)):
                e, a = r.energy[mask], r.roi_area[mask]
                ok = ~np.isnan(e)
                lift = e[ok] / a[ok]
                row = {
                    "run": run, "model": RUN_LABELS[run], "condition": key, "condition_pl": cond_label_pl(key),
                    "corruption": ctype or "none", "level": level, "subset": subset,
                    "n": int(mask.sum()), "n_empty_cam": int((~ok).sum()),
                    "energy_in_roi_mean": _stat(np.mean, e[ok]), "energy_in_roi_median": _stat(np.median, e[ok]),
                    "roi_area_mean": _stat(np.mean, a), "lift_mean": _stat(np.mean, lift),
                    "lift_median": _stat(np.median, lift),
                    "occluder_in_roi": ctype == "occlusion",
                }
                if r.occ_energy is not None:
                    oe, oa = r.occ_energy[mask], r.occ_area[mask]
                    ook = ~np.isnan(oe)
                    row |= {"occluder_energy_mean": _stat(np.mean, oe[ook]),
                            "occluder_energy_median": _stat(np.median, oe[ook]),
                            "occluder_area_mean": _stat(np.mean, oa),
                            "occluder_lift_median": _stat(np.median, oe[ook] / oa[ook])}
                rows.append(row)
    return pd.DataFrame(rows).round(5)


# ---------------------------------------------------------------------------
# Confidence, calibration and the rejection threshold
# ---------------------------------------------------------------------------

def calibration(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Accuracy vs confidence in n_bins equal-width bins (lo, hi] and the expected calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = bin_of == b
        n = int(m.sum())
        acc_b = float(correct[m].mean()) if n else None
        conf_b = float(conf[m].mean()) if n else None
        if n:
            ece += n / len(conf) * abs(acc_b - conf_b)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n, "accuracy": acc_b, "mean_conf": conf_b})
    wrong = conf[~correct]
    return {
        "n": int(len(conf)),
        "accuracy": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "ece": float(ece),
        "mean_conf_correct": _stat(np.mean, conf[correct]),
        "mean_conf_wrong": _stat(np.mean, wrong),
        "median_conf_wrong": _stat(np.median, wrong),
        "n_wrong": int(len(wrong)),
        "share_wrong_conf_ge_0.9": _stat(np.mean, wrong >= 0.9),
        "share_wrong_conf_ge_0.99": _stat(np.mean, wrong >= 0.99),
        "bins": bins,
    }


def confidence_summary(test_res: dict[str, dict[str, CondResult]]) -> dict:
    out = {}
    for run in XAI_RUNS:
        res = test_res[run]
        out[run] = {
            "model": RUN_LABELS[run],
            "clean": calibration(*pooled(res, ["clean"])),
            "corrupted_all": calibration(*pooled(res, CORRUPTED_KEYS)),
            "per_corruption": {c: {k: v for k, v in calibration(*pooled(res, [cond_key(c, lv) for lv in LEVELS]))
                                   .items() if k != "bins"} for c in CORRUPTIONS},
        }
    return out


def coverage_curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every distinct confidence t (descending): t, coverage and accuracy of the predictions with conf >= t."""
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order].astype(np.float64)
    k = np.arange(1, len(c) + 1)
    acc = np.cumsum(ok) / k
    last = np.r_[c[1:] != c[:-1], True]  # end of each run of equal confidences
    return c[last], k[last] / len(c), acc[last]


def lowest_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float | None:
    """The lowest t for which the predictions with conf >= t have accuracy >= target (None if no t has)."""
    thr, _cov, acc = coverage_curve(conf, correct)
    ok = np.flatnonzero(acc >= target)
    return float(thr[ok[-1]]) if len(ok) else None


def rejection_stats(conf: np.ndarray, correct: np.ndarray, t: float | None) -> dict:
    keep = conf >= t if t is not None else np.ones(len(conf), bool)
    n_keep, errors = int(keep.sum()), int((~correct).sum())
    errors_kept = int((~correct & keep).sum())
    return {
        "n": int(len(conf)),
        "n_kept": n_keep,
        "coverage": n_keep / len(conf),
        "accuracy_all": float(correct.mean()),
        "accuracy_kept": float(correct[keep].mean()) if n_keep else None,
        "errors_total": errors,
        "errors_kept": errors_kept,
        "share_errors_rejected": 1 - errors_kept / errors if errors else None,
        "share_correct_rejected": float((correct & ~keep).sum() / max(correct.sum(), 1)),
    }


def threshold_report(val_res: dict[str, CondResult], test_res: dict[str, CondResult]) -> dict:
    conf_v, ok_v = pooled(val_res, CORRUPTED_KEYS)
    t = lowest_threshold(conf_v, ok_v, TARGET_ACC)

    def by_condition(res: dict[str, CondResult]) -> dict:
        report = {"clean": rejection_stats(*pooled(res, ["clean"]), t),
                  "corrupted_all": rejection_stats(*pooled(res, CORRUPTED_KEYS), t)}
        for c in CORRUPTIONS:
            report[c] = rejection_stats(*pooled(res, [cond_key(c, lv) for lv in LEVELS]), t)
        return report

    return {
        "model": FINAL_RUN,
        "rule": "najniższy próg t, przy którym accuracy zachowanych predykcji (max softmax >= t) na zakłóconej "
                f"walidacji (15 zakłóceń razem) wynosi co najmniej {fmt_pct(TARGET_ACC)}; wyznaczony wyłącznie na "
                "walidacji, test tylko raportuje efekt",
        "chosen_on": "val, 15 corruptions pooled",
        "confidence": "max softmax of fp32 logits (softmax in float64); a prediction is kept when conf >= threshold",
        "target_accuracy": TARGET_ACC,
        "threshold": t,
        "val_thresholds_other_targets": {str(a): lowest_threshold(conf_v, ok_v, a) for a in EXTRA_TARGETS},
        "val": by_condition(val_res),
        "test": by_condition(test_res),
    }


def clean_json(obj):
    """NaN -> None and NumPy scalars -> Python. Floats are not rounded: the threshold must stay exact."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    return obj


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

TO_FLOAT = v2.ToDtype(torch.float32, scale=True)


def as_hwc(img: torch.Tensor) -> np.ndarray:
    return img.permute(1, 2, 0).numpy()


def overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """CAM in the colormap over a grayscale copy of the image; stronger map = more opaque."""
    gray = img.mean(axis=2, keepdims=True)
    heat = CAM_CMAP(cam)[..., :3]
    alpha = (0.3 + 0.55 * cam)[..., None]
    return np.clip((1 - alpha) * gray + alpha * heat, 0, 1)


def show(ax, img: np.ndarray, box=None) -> None:
    ax.imshow(img, interpolation="nearest")
    if box is not None:
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, ec=ROI_COLOR, lw=1.1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS)
        spine.set_linewidth(0.6)


@functools.lru_cache(maxsize=None)
def pictogram(c: int) -> np.ndarray:
    """Class pictogram from data/Meta on a white background."""
    with Image.open(DATA_DIR / "Meta" / f"{c}.png") as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return np.asarray(bg.convert("RGB"))


def image_grid(width_ratios: list[float], height_ratios: list[float], cell: float, wspace: float = 0.06,
               hspace: float = 0.06, margins: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.2)):
    """Figure and GridSpec in which a ratio-1 cell is exactly `cell` inches, so square images fill their
    cells and caption rows keep their height. margins = (left, right, top, bottom) in inches."""
    left, right, top, bottom = margins
    grid_w = cell * (sum(width_ratios) + wspace * (len(width_ratios) - 1) * np.mean(width_ratios))
    grid_h = cell * (sum(height_ratios) + hspace * (len(height_ratios) - 1) * np.mean(height_ratios))
    w, h = grid_w + left + right, grid_h + top + bottom
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(len(height_ratios), len(width_ratios), width_ratios=width_ratios,
                          height_ratios=height_ratios, wspace=wspace, hspace=hspace,
                          left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h)
    return fig, gs


def cam_colorbar(fig, label: str = "Grad-CAM (0 = brak wpływu, 1 = największy)") -> None:
    """Colorbar of the CAM colormap in the right margin of an image_grid figure."""
    w, h = fig.get_size_inches()
    right = fig.subplotpars.right if not fig.axes else max(ax.get_position().x1 for ax in fig.axes)
    cax = fig.add_axes((right + 0.2 / w, 0.3, 0.12 / w, 0.4))
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CAM_CMAP), cax=cax)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    comma_axis(cbar.ax, "y")


def caption(ax, text: str, ok: bool = True, fontsize: float = 7.0) -> None:
    """Text under an image axis (in the caption row reserved by image_grid)."""
    ax.text(0.5, -0.04, text, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize,
            color=INK if ok else WRONG_COLOR, linespacing=1.15)


def comma_axis(ax, axis: str = "x", digits: int = 1) -> None:
    fmt_ = matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:.{digits}f}".replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt_)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roi_energy(test_res: dict[str, dict[str, CondResult]], cam_info: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(XAI_RUNS), figsize=(11.5, 4.4), sharex=True)
    lowest = min(np.nanmin(test_res[run]["clean"].energy) for run in XAI_RUNS)
    lo = min(0.5, math.floor(lowest * 10) / 10)  # the axis starts at the lowest observed share
    bins = np.arange(lo, 1.0 + 1e-9, 0.025)
    for ax, run in zip(axes, XAI_RUNS):
        r = test_res[run]["clean"]
        ok = ~np.isnan(r.energy)
        for name, mask, color, filled in (("poprawne", r.correct, CORRECT_COLOR, True),
                                          ("błędne", ~r.correct, WRONG_COLOR, False)):
            e = r.energy[mask & ok]
            if not len(e):
                continue
            label = f"{name}: n = {fmt_int(len(e))}, mediana {fmt(np.median(e))}"
            if filled:
                ax.hist(e, bins=bins, density=True, color=color, alpha=0.3, label=label)
                ax.hist(e, bins=bins, density=True, histtype="step", color=color, lw=1)
            else:
                ax.hist(e, bins=bins, density=True, histtype="step", color=color, lw=2, label=label)
            ax.axvline(np.median(e), color=color, lw=1.4)
        area = float(np.mean(r.roi_area))
        ax.axvline(area, color=INK_2, lw=1.2, ls=(0, (1, 1.5)),
                   label=f"mapa równomierna = pole ROI / pole obrazu (średnio {fmt(area)})")
        ax.set_xlim(lo, 1)
        comma_axis(ax, "x")
        ax.set_xlabel("energy_in_roi: udział mapy Grad-CAM wewnątrz ramki ROI")
        ax.set_title(f"{RUN_LABELS[run]}\nGrad-CAM z warstwy {cam_info[run]['layer']} "
                     f"({cam_info[run]['map_size']}, przeskalowane do 64×64)", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7.8)
    axes[0].set_ylabel("gęstość (pole pod histogramem = 1)")
    n = len(test_res[FINAL_RUN]["clean"].label)
    fig.suptitle(f"Ile uwagi modelu przypada na znak? Czysty zbiór testowy (n = {fmt_int(n)}), "
                 "predykcje poprawne i błędne; pionowe linie = mediany", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pair_groups(idx: np.ndarray, r: CondResult) -> list[list[int]]:
    """Images grouped by (true, predicted) pair: larger groups first, then by highest confidence."""
    groups = defaultdict(list)
    for i in idx:
        groups[(int(r.label[i]), int(r.pred[i]))].append(int(i))
    return sorted(groups.values(), key=lambda g: (-len(g), -r.conf[g[0]]))


def flow_rows(groups: list[list[int]], per_row: int) -> list[list[list[int]]]:
    """First-fit layout of groups into rows of per_row slots; a group is split only if longer than a row."""
    rows: list[list[list[int]]] = []
    for g in groups:
        for chunk in (g[k:k + per_row] for k in range(0, len(g), per_row)):
            for row in rows:
                if sum(map(len, row)) + len(chunk) <= per_row:
                    row.append(chunk)
                    break
            else:
                rows.append([chunk])
    return rows


def plot_error_gallery(model: nn.Module, base: GTSRBDataset, r: CondResult, device: torch.device,
                       path: Path) -> dict:
    wrong = np.flatnonzero(~r.correct)
    idx = wrong[np.argsort(-r.conf[wrong], kind="stable")][:N_GALLERY]
    if not len(idx):
        print(f"no errors, skipping {path.name}")
        return {}
    maps, pred, conf = cam_for(model, torch.stack([base[i][0] for i in idx]), device)
    cam_of = dict(zip(idx.tolist(), maps))
    boxes = roi_boxes(base)
    groups = pair_groups(idx, r)
    group_size = {i: len(g) for g in groups for i in g}
    rows = flow_rows(groups, GALLERY_PER_ROW)

    slot_cols = 4  # photo, Grad-CAM, pictogram, gap
    width_ratios = ([1, 1, 0.8, 0.3] * GALLERY_PER_ROW)[:-1]
    fig, gs = image_grid(width_ratios, [0.62, 1, 0.24] * len(rows), cell=1.0, hspace=0.04,
                         margins=(0.15, 0.15, 0.75, 0.1))
    for ri, row in enumerate(rows):
        slot = 0
        for chunk in row:
            first, last = slot * slot_cols, (slot + len(chunk) - 1) * slot_cols + 3
            hax = fig.add_subplot(gs[3 * ri, first:last])
            hax.axis("off")
            t, p = int(r.label[chunk[0]]), int(r.pred[chunk[0]])
            width = 50 * len(chunk) + 12 * (len(chunk) - 1)
            n_pair = group_size[chunk[0]]
            text = (textwrap.fill(f"prawda: {class_label(t, code=True)}", width) + "\n"
                    + textwrap.fill(f"→ predykcja: {class_label(p, code=True)}"
                                    + (f"  ({n_pair} z {len(idx)})" if n_pair > 1 else ""), width))
            hax.text(0.5, 0.06, text, ha="center", va="bottom", fontsize=7.6, color=INK, transform=hax.transAxes,
                     linespacing=1.15)
            hax.plot([0.01, 0.99], [0.0, 0.0], color=AXIS, lw=0.8, transform=hax.transAxes, clip_on=False)
            for i in chunk:
                c0 = slot * slot_cols
                img = as_hwc(TO_FLOAT(base.image_uint8(i)))
                ax_img, ax_cam, ax_pic = (fig.add_subplot(gs[3 * ri + 1, c0 + k]) for k in range(3))
                show(ax_img, img, boxes[i])
                show(ax_cam, overlay(img, cam_of[i]), boxes[i])
                show(ax_pic, pictogram(int(r.pred[i])))
                e = energy_share(cam_of[i][None], boxes[i][None])[0]
                caption(ax_cam, f"pewność {fmt(r.conf[i], 3)} · ROI {fmt(e)}")
                slot += 1
    fig.suptitle(f"{len(idx)} najbardziej pewnych siebie błędów finalnego modelu ({RUN_LABELS[FINAL_RUN]}) "
                 f"na czystym zbiorze testowym, pogrupowane według pary klas\n"
                 "w każdej trójce: zdjęcie z ramką ROI · mapa Grad-CAM dla klasy przewidzianej · "
                 "wzorzec klasy przewidzianej; ROI = energy_in_roi", fontsize=10.5, y=1 - 0.08 / fig.get_figheight(),
                 va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not np.array_equal(pred, r.pred[idx]):
        print("warning: error gallery predictions differ from the main pass")
    return {"indices": idx.tolist(), "paths": [base.paths[i] for i in idx]}


def pick_examples(fin: CondResult, bas: CondResult, base: GTSRBDataset, rng: np.random.Generator) -> list[int]:
    """One random image per visual sign group, preferring images both models classify correctly."""
    chosen = []
    for classes in SIGN_GROUPS_PL.values():
        in_group = np.isin(base.labels, classes)
        for pool in (in_group & fin.correct & bas.correct, in_group & fin.correct, in_group):
            cand = np.flatnonzero(pool)
            if len(cand):
                chosen.append(int(rng.choice(cand)))
                break
    return chosen


def plot_gradcam_examples(models: dict[str, nn.Module], base: GTSRBDataset, test_res, device: torch.device,
                          path: Path) -> dict:
    idx = pick_examples(test_res[FINAL_RUN]["clean"], test_res[BASELINE_RUN]["clean"], base,
                        np.random.default_rng(SEED))
    x = torch.stack([base[i][0] for i in idx])
    cams = {run: cam_for(models[run], x, device) for run in XAI_RUNS}
    boxes = roi_boxes(base)
    fig, gs = image_grid([1] * len(idx), [1, 0.12, 1, 0.42, 1, 0.42], cell=1.45, wspace=0.08, hspace=0.04,
                         margins=(1.35, 1.0, 1.1, 0.1))
    for j, i in enumerate(idx):
        c = int(base.labels[i])
        img = as_hwc(TO_FLOAT(base.image_uint8(i)))
        ax = fig.add_subplot(gs[0, j])
        show(ax, img, boxes[i])
        ax.set_title(f"{GROUP_OF[c]}\n{class_label(c, short=True)}", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("zdjęcie\n(ramka ROI)", fontsize=8.5)
        for row, run in ((2, FINAL_RUN), (4, BASELINE_RUN)):
            maps, pred, conf = cams[run]
            ax = fig.add_subplot(gs[row, j])
            show(ax, overlay(img, maps[j]), boxes[i])
            e = energy_share(maps[j][None], boxes[i][None])[0]
            ok = pred[j] == c
            verdict = "✓ poprawnie" if ok else f"✗ {class_label(int(pred[j]), short=True, width=18)}"
            caption(ax, f"{verdict}\npewność {fmt(conf[j])} · ROI {fmt(e)}", ok, fontsize=7.2)
            if j == 0:
                ax.set_ylabel(textwrap.fill(RUN_LABELS[run], 18), fontsize=8.5)
    cam_colorbar(fig)
    fig.suptitle("Grad-CAM dla poprawnych predykcji: po jednym losowym zdjęciu testowym z każdej grupy znaków\n"
                 "pod mapą: wynik, pewność (max softmax), ROI = energy_in_roi", fontsize=10.5,
                 y=1 - 0.08 / fig.get_figheight(), va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"indices": idx, "paths": [base.paths[i] for i in idx]}


def pick_shift_images(res: dict[str, CondResult], base: GTSRBDataset, rng: np.random.Generator
                      ) -> list[tuple[int, str]]:
    """2 random clean-correct images that stay correct under all 10 shown corruptions and 2 that turn
    wrong under at least 2 of them, all from different sign groups."""
    keys = [cond_key(c, lv) for c in CORRUPTIONS for lv in SHIFT_LEVELS]
    n_wrong = np.sum([~res[k].correct for k in keys], axis=0)
    ok = res["clean"].correct
    chosen, groups = [], set()
    for kind, pool, n in (("stabilne", ok & (n_wrong == 0), N_SHIFT_STABLE),
                          ("zmienia predykcję", ok & (n_wrong >= 2), N_SHIFT_FLIP)):
        taken = 0
        for fallback in (pool, ok, np.ones_like(ok)):  # smoke models may have too few candidates
            if taken == n:
                break
            for i in rng.permutation(np.flatnonzero(fallback)):
                g = GROUP_OF[int(base.labels[i])]
                if g not in groups and int(i) not in [c for c, _ in chosen]:
                    chosen.append((int(i), kind if fallback is pool else f"{kind}: brak, zastępcze"))
                    groups.add(g)
                    taken += 1
                    if taken == n:
                        break
    return chosen


def plot_corruption_shift(model: nn.Module, base: GTSRBDataset, res: dict[str, CondResult], device: torch.device,
                          path: Path) -> dict:
    picks = pick_shift_images(res, base, np.random.default_rng(SEED))
    versions = [(None, 0)] + [(c, lv) for c in CORRUPTIONS for lv in SHIFT_LEVELS]
    boxes = roi_boxes(base)
    fig, gs = image_grid([1] * len(versions), [1, 1, 0.5] * len(picks), cell=1.12, wspace=0.07, hspace=0.04,
                         margins=(1.25, 1.0, 1.1, 0.05))
    for k, (i, kind) in enumerate(picks):
        c = int(base.labels[i])
        dsets = [None if ct is None else CorruptedDataset(base, ct, lv) for ct, lv in versions]
        imgs = [TO_FLOAT(base.image_uint8(i)) if d is None else d.corrupted_image(i) for d in dsets]
        x = torch.stack([base[i][0] if d is None else d[i][0] for d in dsets])
        maps, pred, conf = cam_for(model, x, device)
        for j, ((ctype, lv), img) in enumerate(zip(versions, imgs)):
            img = as_hwc(img)
            ax_img, ax_cam = fig.add_subplot(gs[3 * k, j]), fig.add_subplot(gs[3 * k + 1, j])
            show(ax_img, img)
            show(ax_cam, overlay(img, maps[j]), boxes[i])
            ok = pred[j] == c
            e = energy_share(maps[j][None], boxes[i][None])[0]
            label = class_label(int(pred[j]), short=True, width=17)
            caption(ax_cam, f"{'✓' if ok else '✗'} {label}\n{fmt(conf[j])} · ROI {fmt(e)}", ok, fontsize=6.6)
            if k == 0:
                title = "czyste" if ctype is None else \
                    f"{CORRUPTION_LABELS_PL[ctype].replace(' ', chr(10), 1)}\npoz. {lv}: {level_label_pl(ctype, lv)}"
                ax_img.set_title(title, fontsize=7.2)
            if j == 0:
                ax_img.set_ylabel(f"{class_label(c, short=True, code=True, width=16)}\n({kind})", fontsize=7.5)
                ax_cam.set_ylabel("Grad-CAM", fontsize=7.5)
    cam_colorbar(fig)
    fig.suptitle(f"Jak przesuwa się uwaga finalnego modelu ({RUN_LABELS[FINAL_RUN]}) przy zakłóceniach "
                 f"poziomu {' i '.join(map(str, SHIFT_LEVELS))}\n{N_SHIFT_STABLE} losowe zdjęcia testowe, które "
                 f"model rozpoznaje zawsze, i {N_SHIFT_FLIP}, przy których myli się co najmniej 2 razy; "
                 "pod mapą: predykcja, pewność, energy_in_roi", fontsize=10.5,
                 y=1 - 0.08 / fig.get_figheight(), va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"indices": [i for i, _ in picks], "paths": [base.paths[i] for i, _ in picks],
            "kinds": [kind for _, kind in picks]}


def plot_confidence(test_res: dict[str, dict[str, CondResult]], summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(XAI_RUNS), 3, figsize=(14, 4.0 * len(XAI_RUNS)),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    hist_bins = np.linspace(0, 1, 26)
    for r_i, run in enumerate(XAI_RUNS):
        res = test_res[run]
        for c_i, (keys, name, part) in enumerate(((["clean"], "czysty test", "clean"),
                                                  (CORRUPTED_KEYS, "test z 15 zakłóceniami razem",
                                                   "corrupted_all"))):
            ax = axes[r_i, c_i]
            conf, ok = pooled(res, keys)
            s = summary[run][part]
            for mask, label, color, filled in ((ok, "poprawne", CORRECT_COLOR, True),
                                               (~ok, "błędne", WRONG_COLOR, False)):
                if not mask.any():
                    continue
                lab = f"{label} (n = {fmt_int(int(mask.sum()))})"
                if filled:
                    ax.hist(conf[mask], bins=hist_bins, color=color, alpha=0.3, label=lab)
                    ax.hist(conf[mask], bins=hist_bins, histtype="step", color=color, lw=1)
                else:
                    ax.hist(conf[mask], bins=hist_bins, histtype="step", color=color, lw=2, label=lab)
            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            comma_axis(ax, "x")
            ax.set_xlabel("pewność = max softmax")
            ax.set_title(f"{name}: accuracy {fmt_pct(s['accuracy'], 2)}, ECE {fmt(s['ece'], 4)}", fontsize=9.5)
            ax.legend(loc="upper left", fontsize=8)
            if c_i == 0:
                ax.set_ylabel(f"{textwrap.fill(RUN_LABELS[run], 24)}\n\nliczba predykcji (skala log.)", fontsize=9)
        ax = axes[r_i, 2]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, label="idealna kalibracja")
        for part, label, color, marker in (("clean", "czyste", CLEAN_COLOR, "o"),
                                           ("corrupted_all", "zakłócone (15)", CORRUPT_COLOR, "s")):
            bins = [b for b in summary[run][part]["bins"] if b["n"]]
            xs, ys = [b["mean_conf"] for b in bins], [b["accuracy"] for b in bins]
            ax.plot(xs, ys, color=color, lw=1.4)
            ax.plot([], [], color=color, lw=1.4, marker=marker, ms=5.5,
                    label=f"{label}: ECE {fmt(summary[run][part]['ece'], 4)}")
            for x_, y_, b in zip(xs, ys, bins):  # bins with few predictions: hollow markers
                ax.plot(x_, y_, marker=marker, ms=5.5, color=color, ls="none",
                        mfc=color if b["n"] >= MIN_BIN_N else "white", mew=1.3)
        ax.plot([], [], marker="o", ms=5.5, color=INK_2, mfc="white", ls="none",
                label=f"pusty znacznik: < {MIN_BIN_N} predykcji w przedziale")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        comma_axis(ax, "x")
        comma_axis(ax, "y")
        ax.set_aspect("equal")
        ax.set_xlabel("średnia pewność w przedziale")
        ax.set_ylabel("accuracy w przedziale")
        ax.set_title(f"diagram niezawodności ({N_BINS} równych przedziałów)", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Pewność modeli na zbiorze testowym: rozkład max softmax dla predykcji poprawnych i błędnych "
                 "oraz kalibracja", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold(val_res: dict[str, CondResult], test_res: dict[str, CondResult], report: dict,
                   path: Path) -> None:
    t = report["threshold"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    curves = [("walidacja, zakłócona (podstawa wyboru progu)", val_res, CORRUPTED_KEYS, VAL_COLOR, "-", 2.2),
              ("walidacja, czysta", val_res, ["clean"], VAL_COLOR, (0, (3, 2)), 1.4),
              ("test, zakłócony", test_res, CORRUPTED_KEYS, TEST_COLOR, "-", 2.2),
              ("test, czysty", test_res, ["clean"], TEST_COLOR, (0, (3, 2)), 1.4)]
    lows, cov_at_t = [], []
    for label, res, keys, color, ls, lw in curves:
        conf, ok = pooled(res, keys)
        _thr, cov, acc = coverage_curve(conf, ok)
        ax.plot(cov * 100, acc * 100, color=color, ls=ls, lw=lw, label=label)
        lows.append(acc[-1] * 100)
        s = rejection_stats(conf, ok, t)
        if t is not None and s["n_kept"]:
            cov_at_t.append(s["coverage"] * 100)
            ax.plot(s["coverage"] * 100, s["accuracy_kept"] * 100, marker="o", ms=7, color=color, mec="white",
                    mew=1.2, zorder=4)
    ax.axhline(TARGET_ACC * 100, color=INK_2, lw=0.9, ls=(0, (1, 1.5)))
    ax.text(0.99, TARGET_ACC * 100 - 0.05, f"cel {fmt_pct(TARGET_ACC)} (walidacja zakłócona)",
            transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=8, color=INK_2)
    x_lo = max(0.0, math.floor((min(cov_at_t) if cov_at_t else 50) / 10) * 10 - 10)
    ax.set_xlim(x_lo, 100.5)
    ax.set_ylim(min(lows) - 0.5, 100.15)
    pct_axis(ax)
    pct_axis(ax, "x")
    ax.invert_xaxis()
    ax.set_xlabel("pokrycie = odsetek zachowanych predykcji (im dalej w prawo, tym wyższy próg)")
    ax.set_ylabel("accuracy zachowanych predykcji")
    thr_text = "nie osiągnięto celu" if t is None else f"wybrany próg t = {fmt(t, 4)} (kółka)"
    ax.set_title(f"Pokrycie a accuracy przy odrzucaniu predykcji o pewności < t; {thr_text}", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8)

    names = [("clean", "czyste")] + [(c, CORRUPTION_LABELS_PL[c]) for c in CORRUPTIONS] + \
            [("corrupted_all", "wszystkie zakłócone")]
    test_rep = report["test"]
    y = np.arange(len(names))[::-1]
    for yi, (key, name) in zip(y, names):
        s = test_rep[key]
        a0 = s["accuracy_all"] * 100
        bx.plot(a0, yi, marker="o", ms=7, mfc="white", mec=TEST_COLOR, mew=1.6, zorder=3)
        if s["accuracy_kept"] is not None:
            a1 = s["accuracy_kept"] * 100
            bx.plot([a0, a1], [yi, yi], color=AXIS, lw=2, zorder=1)
            bx.plot(a1, yi, marker="o", ms=7, color=TEST_COLOR, mec="white", mew=1.0, zorder=3)
        bx.text(101.2, yi, f"pokrycie {fmt_pct(s['coverage'])}\nodrzucone błędy {fmt_pct(s['share_errors_rejected'] or 0)}",
                va="center", fontsize=7.5, color=INK_2)
    bx.set_yticks(y, [n for _, n in names])
    lo = min(test_rep[k]["accuracy_all"] for k, _ in names) * 100
    bx.set_xlim(math.floor(lo / 5) * 5 - 1, 100.8)
    pct_axis(bx, "x")
    bx.set_xlabel("accuracy na zbiorze testowym")
    bx.grid(axis="y", visible=False)
    bx.set_title("Efekt progu na teście (poziomy 1–3 razem):\nwszystkie predykcje (pusty) → zachowane (pełny)",
                 fontsize=9.5)
    fig.suptitle(f"Próg odrzucania dla aplikacji – wybrany na walidacji, na teście tylko raportowany "
                 f"({RUN_LABELS[FINAL_RUN]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def layer_info(model: nn.Module, device: torch.device) -> dict:
    layer = gradcam_target_layer(model)
    name = next(n for n, m in model.named_modules() if m is layer)
    shapes = []
    handle = layer.register_forward_hook(lambda _m, _i, o: shapes.append(tuple(o.shape[2:])))
    with torch.inference_mode():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device).contiguous(memory_format=torch.channels_last))
    handle.remove()
    return {"layer": name, "map_size": "×".join(map(str, shapes[0]))}


def project_relative(path: Path) -> str:
    """Path relative to the repository (no local directories in committed results)."""
    path = Path(path).resolve()
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.name


def prediction_diff(root: Path, base: GTSRBDataset, test_res: dict[str, dict[str, CondResult]]) -> dict:
    """Clean-test images whose fp32 prediction here differs from the bf16 prediction of src.final_eval."""
    out = {}
    for run in XAI_RUNS:
        path = root / FINAL_TEST_DIR / run / "predictions_test.csv"
        if not path.exists():
            out[run] = None
            continue
        ev = pd.read_csv(path).set_index("path")["pred"]
        ours = pd.Series(test_res[run]["clean"].pred, index=base.paths)
        common = ev.index.intersection(ours.index)
        out[run] = {"n_compared": int(len(common)), "n_different": int((ev[common] != ours[common]).sum())}
    return out


def run_xai(smoke: bool = False, batch_size: int = BATCH_SIZE, report_split: str = "test",
            weights_root: Path | None = None, out_dir: Path | None = None, n: int | None = None) -> Path:
    """report_split="val", weights_root and n exist for checks that must not touch the test set."""
    root = results_root(smoke)
    weights_root = Path(weights_root) if weights_root is not None else root
    out = Path(out_dir) if out_dir is not None else root / XAI_DIR
    out.mkdir(parents=True, exist_ok=True)
    n = n if n is not None else (SMOKE_N_VAL if smoke else None)
    torch.backends.cudnn.allow_tf32 = False  # true fp32, like CPU inference in the app
    torch.backends.cuda.matmul.allow_tf32 = False
    # Measured on the RTX 4070 Laptop (Grad-CAM, ResNet18, batch 128): benchmark=True is ~10% faster per
    # batch but spends ~20 s autotuning every new batch shape and peaks at 5.3 GB instead of 3.1 GB.
    torch.backends.cudnn.benchmark = False
    device = get_device()
    timing = {}

    t0 = time.perf_counter()
    models = {run: load_model(weights_root / run, device)[0] for run in XAI_RUNS}
    cam_info = {run: layer_info(models[run], device) for run in XAI_RUNS}
    test = cached_dataset(report_split, n)
    val = cached_dataset("val", n)
    timing["load_s"] = time.perf_counter() - t0
    print(f"models from {weights_root}, {report_split}: {len(test)} images, val: {len(val)} images, "
          f"device {device}, batch {batch_size}, Grad-CAM layers {cam_info}")

    t0 = time.perf_counter()
    test_res = run_pass(models, test, CAM_KEYS, device, batch_size, report_split)
    timing["test_pass_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    val_res = run_pass({FINAL_RUN: models[FINAL_RUN]}, val, [], device, batch_size, "val")[FINAL_RUN]
    timing["val_pass_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    roi_energy_table(test_res).to_csv(out / "roi_energy.csv", index=False)
    plot_roi_energy(test_res, cam_info, out / "roi_energy.png")
    conf_summary = confidence_summary(test_res)
    write_json({"description": "max softmax (fp32 logits) on the test set; ECE with "
                               f"{N_BINS} equal-width bins (lo, hi]; corrupted_all = 15 corruptions pooled",
                "split": report_split, "n_bins": N_BINS, "models": conf_summary}, out / "confidence.json")
    plot_confidence(test_res, conf_summary, out / "confidence.png")
    report = threshold_report(val_res, test_res[FINAL_RUN])
    write_json(report, out / "threshold.json")
    plot_threshold(val_res, test_res[FINAL_RUN], report, out / "threshold.png")
    shown = {
        "error_gallery": plot_error_gallery(models[FINAL_RUN], test, test_res[FINAL_RUN]["clean"], device,
                                            out / "error_gallery.png"),
        "gradcam_examples": plot_gradcam_examples(models, test, test_res, device, out / "gradcam_examples.png"),
        "corruption_shift": plot_corruption_shift(models[FINAL_RUN], test, test_res[FINAL_RUN], device,
                                                  out / "corruption_shift.png"),
    }
    timing["outputs_s"] = time.perf_counter() - t0

    write_json({
        "split": report_split, "smoke": smoke,
        "weights": {run: project_relative(weights_root / run / "best.pt") for run in XAI_RUNS},
        "n_report": len(test), "n_val": len(val), "batch_size": batch_size, "device": str(device),
        "precision": "fp32 (autocast off, TF32 off)",
        "gradcam": {run: {**cam_info[run], "target": "predicted class", "library": "pytorch-grad-cam GradCAM",
                          "normalization": "min-max per image to [0, 1], upsampled to 64x64"} for run in XAI_RUNS},
        "cam_conditions": CAM_KEYS,
        "roi": "annotated ROI rescaled to 64x64 and rounded to integer pixels (as in the occlusion corruption)",
        "pred_diff_vs_final_eval": prediction_diff(root, test, test_res) if report_split == "test" else None,
        "threshold": report["threshold"],
        "shown_images": shown,
        "timing_s": {k: round(v, 1) for k, v in timing.items()},
    }, out / "run_info.json")

    print(f"\nROI energy (clean, median energy_in_roi / lift):")
    for run in XAI_RUNS:
        r = test_res[run]["clean"]
        for name, m in (("correct", r.correct), ("wrong", ~r.correct)):
            e = r.energy[m & ~np.isnan(r.energy)]
            if len(e):
                print(f"  {run:<24} {name:<8} n {len(e):>6}  energy {np.median(e):.3f}  "
                      f"lift {np.median(e / r.roi_area[m & ~np.isnan(r.energy)]):.3f}")
    t = report["threshold"]
    print(f"threshold (val, {TARGET_ACC:.1%} on corrupted): {t}")
    if t is not None:
        for part in ("clean", "corrupted_all"):
            s = report["test"][part]
            print(f"  {report_split} {part:<14} coverage {s['coverage']:.4f}  acc kept {s['accuracy_kept']:.4f} "
                  f"(all {s['accuracy_all']:.4f})")
    print(f"timing: {timing} -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM, ROI energy, confidence and rejection threshold (KM3).")
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch smoke models from results/_smoke/ on ~500 test/val images -> results/_smoke/xai/")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    run_xai(smoke=args.smoke, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
