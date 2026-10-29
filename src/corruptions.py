"""Deterministic image corruptions for robustness evaluation, 3 severity levels each:
gaussian_blur, gaussian_noise, darken, occlusion, jpeg (jpeg is never used in training augmentation).

Corruptions act on the resized 64x64 RGB image as a float tensor in [0, 1], before
ImageNet normalization. Random parts (noise, occluder position) come from a generator
seeded with (seed, image index, corruption type, level), so every model is evaluated
on exactly the same corrupted images.

Run as a module to check the pipeline and save a preview:
    python -m src.corruptions     # writes results/corruptions/preview.png
"""

from __future__ import annotations

import argparse
import io
import math
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from src.data import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT, SEED, GTSRBDataset, cached_dataset

CORRUPTIONS = ("gaussian_blur", "gaussian_noise", "darken", "occlusion", "jpeg")
LEVELS = (1, 2, 3)
# Fixed in KM1 (docs/KM1_sprawozdanie.md, section 5.4) - do not change.
SEVERITY = {
    "gaussian_blur": {1: 0.8, 2: 2.0, 3: 3.0},       # sigma in pixels
    "gaussian_noise": {1: 0.03, 2: 0.06, 3: 0.10},   # sigma, image in [0, 1]
    "darken": {1: 0.6, 2: 0.4, 3: 0.25},             # brightness factor
    "occlusion": {1: 0.10, 2: 0.20, 3: 0.30},        # black box area as a fraction of the ROI area
    "jpeg": {1: 30, 2: 15, 3: 8},                    # JPEG quality
}
PREVIEW_PATH = PROJECT_ROOT / "results" / "corruptions" / "preview.png"


def corruption_rng(seed: int, index: int, ctype: str, level: int) -> np.random.Generator:
    """Generator for one (image, corruption, level); independent of the process and of other images."""
    return np.random.default_rng([seed, index, CORRUPTIONS.index(ctype), level])


# ---------------------------------------------------------------------------
# Corruptions: float image [3, H, W] in [0, 1] -> float image [3, H, W] in [0, 1]
# ---------------------------------------------------------------------------

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma]).clamp(0.0, 1.0)  # float rounding


def gaussian_noise(img: torch.Tensor, sigma: float, rng: np.random.Generator) -> torch.Tensor:
    noise = torch.from_numpy(rng.standard_normal(tuple(img.shape), dtype=np.float32))
    return (img + noise * sigma).clamp(0.0, 1.0)


def darken(img: torch.Tensor, factor: float) -> torch.Tensor:
    return img * factor


def roi_to_pixels(roi, height: int, width: int) -> tuple[int, int, int, int]:
    """Float (x1, y1, x2, y2) box (x2, y2 exclusive) -> integer pixel box clipped to the image."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in roi)
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def occlusion(img: torch.Tensor, frac: float, roi, rng: np.random.Generator) -> torch.Tensor:
    """Black box with the ROI's aspect ratio and frac of its area, placed at random fully inside the ROI."""
    x1, y1, x2, y2 = roi_to_pixels(roi, img.shape[-2], img.shape[-1])
    rw, rh = x2 - x1, y2 - y1
    w = min(rw, max(1, round(rw * math.sqrt(frac))))
    h = min(rh, max(1, round(rh * math.sqrt(frac))))
    ox = x1 + int(rng.integers(0, rw - w + 1))
    oy = y1 + int(rng.integers(0, rh - h + 1))
    out = img.clone()
    out[:, oy:oy + h, ox:ox + w] = 0.0
    return out


def jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Encode and decode with PIL at the given quality (image quantized to 8 bits first)."""
    arr = (img.clamp(0.0, 1.0) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        out = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


def corrupt(img: torch.Tensor, ctype: str, param: float, roi=None,
            rng: np.random.Generator | None = None) -> torch.Tensor:
    """Apply one corruption with an explicit parameter (sigma, factor, area fraction or quality)."""
    if ctype in ("gaussian_noise", "occlusion") and rng is None:
        raise ValueError(f"{ctype} needs an rng (see corruption_rng)")
    if ctype == "gaussian_blur":
        return gaussian_blur(img, param)
    if ctype == "gaussian_noise":
        return gaussian_noise(img, param, rng)
    if ctype == "darken":
        return darken(img, param)
    if ctype == "occlusion":
        if roi is None:
            raise ValueError("occlusion needs the sign ROI")
        return occlusion(img, param, roi, rng)
    if ctype == "jpeg":
        return jpeg(img, int(param))
    raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {ctype!r}")


def apply_corruption(img: torch.Tensor, ctype: str, level: int, roi=None,
                     rng: np.random.Generator | None = None) -> torch.Tensor:
    """Apply a corruption at severity level 1, 2 or 3 (parameters in SEVERITY)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return corrupt(img, ctype, SEVERITY[ctype][level], roi, rng)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class CorruptedDataset(Dataset):
    """Corrupted view of a GTSRBDataset (validation or test) for one (type, level).

    Yields (image [3, 64, 64] normalized, label, roi) like GTSRBDataset with A0:
    resize -> [0, 1] -> corruption -> ImageNet normalization. The base dataset's own
    transform is ignored. Image idx is corrupted with corruption_rng(seed, idx, type, level).
    """

    def __init__(self, base: GTSRBDataset, ctype: str, level: int, seed: int = SEED):
        if ctype not in CORRUPTIONS:
            raise ValueError(f"corruption must be one of {CORRUPTIONS}, got {ctype!r}")
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.base, self.ctype, self.level, self.seed = base, ctype, level, seed
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.base)

    def corrupted_image(self, idx: int) -> torch.Tensor:
        """Corrupted image in [0, 1], before normalization (for display)."""
        img = self.to_float(self.base.image_uint8(idx))
        rng = corruption_rng(self.seed, idx, self.ctype, self.level)
        return apply_corruption(img, self.ctype, self.level, roi=self.base.rois[idx], rng=rng)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        x = self.normalize(self.corrupted_image(idx))
        return x, int(self.base.labels[idx]), torch.from_numpy(self.base.rois[idx])


# ---------------------------------------------------------------------------
# Preview and checks
# ---------------------------------------------------------------------------

CORRUPTION_LABELS_PL = {
    "gaussian_blur": "Rozmycie Gaussa",
    "gaussian_noise": "Szum Gaussa",
    "darken": "Przyciemnienie",
    "occlusion": "Zasłonięcie ROI",
    "jpeg": "Kompresja JPEG",
}


def level_label_pl(ctype: str, level: int) -> str:
    p = SEVERITY[ctype][level]
    text = {
        "gaussian_blur": f"σ = {p}",
        "gaussian_noise": f"σ = {p}",
        "darken": f"×{p}",
        "occlusion": f"{round(p * 100)}% ROI",
        "jpeg": f"jakość {p}",
    }[ctype]
    return text.replace(".", ",")


def pick_preview_images(base: GTSRBDataset, n_normal: int = 4, n_dark: int = 2,
                        dark_below: float = 30.0, seed: int = SEED) -> list[int]:
    """Indices of n_normal ordinary and n_dark dark (mean brightness < dark_below) images, all classes distinct."""
    brightness = base.cache.float().mean(dim=(1, 2, 3)).numpy()
    rng = np.random.default_rng(seed)
    chosen, classes = [], set()
    for mask, n in ((brightness >= 60, n_normal), (brightness < dark_below, n_dark)):
        for idx in rng.permutation(np.flatnonzero(mask)):
            if len([i for i in chosen if mask[i]]) == n:
                break
            if base.labels[idx] not in classes:
                chosen.append(int(idx))
                classes.add(base.labels[idx])
    return chosen


def save_preview(base: GTSRBDataset, indices: list[int], path: Path = PREVIEW_PATH, seed: int = SEED) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_cols = 1 + len(CORRUPTIONS) * len(LEVELS)
    fig, axes = plt.subplots(len(indices), n_cols, figsize=(1.15 * n_cols, 1.2 * len(indices) + 1.2),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.08})
    to_float = v2.ToDtype(torch.float32, scale=True)
    for r, idx in enumerate(indices):
        clean = to_float(base.image_uint8(idx))
        x1, y1, x2, y2 = roi_to_pixels(base.rois[idx], clean.shape[-2], clean.shape[-1])
        images = [clean] + [CorruptedDataset(base, c, lv, seed).corrupted_image(idx)
                            for c in CORRUPTIONS for lv in LEVELS]
        for c, img in enumerate(images):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, ec="#f28e2b", lw=1.0))
                name = textwrap.fill(CLASS_NAMES[base.labels[idx]], 18)
                ax.set_ylabel(f"klasa {base.labels[idx]}\n{name}\njasność {clean.mean() * 255:.0f}/255",
                              fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4 if c else 1.2)
    # column headers: corruption name over its three levels, level parameter over each column
    axes[0, 0].set_title("Oryginał\n(ramka ROI)", fontsize=8, pad=4, fontweight="bold")
    for k, ctype in enumerate(CORRUPTIONS):
        for j, lv in enumerate(LEVELS):
            col = 1 + k * len(LEVELS) + j
            axes[0, col].set_title(f"poziom {lv}\n{level_label_pl(ctype, lv)}", fontsize=7.5, pad=4)
        left = axes[0, 1 + k * len(LEVELS)].get_position()
        right = axes[0, (k + 1) * len(LEVELS)].get_position()
        name = CORRUPTION_LABELS_PL[ctype] + (" (nie ma w treningu)" if ctype == "jpeg" else "")
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.045, name, ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#c0392b" if ctype == "jpeg" else "black")
    fig.suptitle("Zakłócenia do oceny odporności – 6 zdjęć walidacyjnych (w tym 2 ciemne), "
                 "5 typów × 3 poziomy", fontsize=11, y=1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def check_pipeline(base: GTSRBDataset, n: int = 64) -> None:
    """Corrupted view is deterministic and, apart from the corruption, identical to A0."""
    to_float, normalize = v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(n):
        assert torch.equal(normalize(to_float(base.image_uint8(i))), base[i][0]), "A0 pipeline mismatch"
    for ctype in CORRUPTIONS:
        for lv in LEVELS:
            a, b = CorruptedDataset(base, ctype, lv), CorruptedDataset(base, ctype, lv)
            for i in range(0, len(base), max(1, len(base) // n)):
                x = a.corrupted_image(i)
                assert torch.equal(x, b.corrupted_image(i)), f"{ctype}/{lv} not deterministic"
                assert x.shape == (3, 64, 64) and 0.0 <= x.min() and x.max() <= 1.0, f"{ctype}/{lv} out of range"
    # occluded area matches the requested fraction of the ROI
    for lv in LEVELS:
        ds = CorruptedDataset(base, "occlusion", lv)
        fracs = []
        for i in range(n):
            x1, y1, x2, y2 = roi_to_pixels(base.rois[i], 64, 64)
            changed = (ds.corrupted_image(i) != to_float(base.image_uint8(i))).any(dim=0)
            outside = changed.clone()
            outside[y1:y2, x1:x2] = False
            assert not outside.any(), "occluder leaves the ROI"
            fracs.append(changed.sum().item() / ((x2 - x1) * (y2 - y1)))
        print(f"occlusion level {lv}: target {SEVERITY['occlusion'][lv]:.0%} of ROI, "
              f"changed pixels {np.mean(fracs):.1%} of ROI on average (black pixels already 0 are not counted)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check corruptions and save a preview on validation images.")
    parser.add_argument("--out", type=Path, default=PREVIEW_PATH)
    args = parser.parse_args()

    val = cached_dataset("val")
    check_pipeline(val)
    print("checks passed: A0-equivalent pipeline, determinism, value range, occluder inside ROI")

    for ctype in CORRUPTIONS:
        ds = CorruptedDataset(val, ctype, 3)
        t0 = time.perf_counter()
        for i in range(500):
            ds[i]
        print(f"{ctype:<15} {(time.perf_counter() - t0) / 500 * 1e3:.3f} ms per image")

    indices = pick_preview_images(val)
    save_preview(val, indices, args.out)
    print(f"preview images (val indices): {indices}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
