"""GTSRB data: CSV loading, track-based train/val split, Dataset and augmentations.

Run as a module to create the validation split once and check it:
    python -m src.data            # create splits/val_tracks.json if missing, print checks
    python -m src.data --force    # recreate the split
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = PROJECT_ROOT / "splits" / "val_tracks.json"

SEED = 42
IMG_SIZE = 64
NUM_CLASSES = 43
N_SPLITS = 5  # one fold of StratifiedGroupKFold = ~20% of tracks
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AUGMENTATIONS = ("A0", "A1", "A2")

CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5t",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5t",
]


# ---------------------------------------------------------------------------
# CSV and split
# ---------------------------------------------------------------------------

def track_id(path: str) -> str:
    """'Train/20/00020_00003_00017.png' -> '00020_00003' (<class>_<track>)."""
    cls, track, _frame = Path(path).stem.split("_")
    return f"{cls}_{track}"


def load_csv(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read Train.csv or Test.csv. Training rows also get `track` and `frame` columns."""
    name = name.lower()
    if name not in ("train", "test"):
        raise ValueError(f"name must be 'train' or 'test', got {name!r}")
    df = pd.read_csv(Path(data_dir) / f"{name.capitalize()}.csv")
    if name == "train":
        df["track"] = df["Path"].map(track_id)
        df["frame"] = df["Path"].map(lambda p: int(Path(p).stem.split("_")[2]))
    return df


def make_val_split(train_df: pd.DataFrame, seed: int = SEED) -> tuple[list[str], int]:
    """Pick validation tracks with StratifiedGroupKFold (y = class, groups = track).

    Returns the sorted validation track ids and the fold used: the first fold
    in which every class has at least one validation track.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = sgkf.split(train_df, train_df["ClassId"], groups=train_df["track"])
    for fold, (_, val_idx) in enumerate(folds):
        val = train_df.iloc[val_idx]
        if val["ClassId"].nunique() == NUM_CLASSES:
            return sorted(val["track"].unique()), fold
    raise RuntimeError("no fold has every class in validation")


def save_split(val_tracks: list[str], fold: int, train_df: pd.DataFrame,
               path: Path = SPLIT_PATH) -> None:
    is_val = train_df["track"].isin(val_tracks)
    payload = {
        "description": "Validation tracks (<class>_<track>) held out from data/Train; "
                       "all other training tracks are used for training.",
        "method": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}), "
                  "y=ClassId, groups=track",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "fold": fold,
        "n_tracks": {"train": int(train_df.loc[~is_val, "track"].nunique()), "val": len(val_tracks)},
        "n_images": {"train": int((~is_val).sum()), "val": int(is_val.sum())},
        "val_tracks": list(val_tracks),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path = SPLIT_PATH) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - create it with: python -m src.data")
    return set(json.loads(path.read_text(encoding="utf-8"))["val_tracks"])


def get_train_val(data_dir: Path = DATA_DIR, split_path: Path = SPLIT_PATH
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation frames according to the saved split."""
    df = load_csv("train", data_dir)
    is_val = df["track"].isin(load_split(split_path))
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def split_stats(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Sizes and sanity checks of a train/val split."""
    val_imgs = val_df["ClassId"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    val_tracks = val_df.groupby("ClassId")["track"].nunique().reindex(range(NUM_CLASSES), fill_value=0)
    n_tracks = train_df["track"].nunique() + val_df["track"].nunique()
    return {
        "n_images": {"train": len(train_df), "val": len(val_df)},
        "n_tracks": {"train": int(train_df["track"].nunique()), "val": int(val_df["track"].nunique())},
        "val_fraction_tracks": round(val_df["track"].nunique() / n_tracks, 4),
        "val_fraction_images": round(len(val_df) / (len(train_df) + len(val_df)), 4),
        "shared_tracks": len(set(train_df["track"]) & set(val_df["track"])),
        "val_classes": int((val_imgs > 0).sum()),
        "val_min_images_per_class": int(val_imgs.min()),
        "val_min_images_class": int(val_imgs.idxmin()),
        "val_min_tracks_per_class": int(val_tracks.min()),
    }


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class RandomDarken(torch.nn.Module):
    """With probability p, multiply a float image in [0, 1] by a factor drawn from `factor`."""

    def __init__(self, factor: tuple[float, float] = (0.5, 0.9), p: float = 0.3):
        super().__init__()
        self.factor = factor
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return img
        return img * torch.empty(1).uniform_(*self.factor).item()

    def extra_repr(self) -> str:
        return f"factor={self.factor}, p={self.p}"


class RandomGaussianNoise(torch.nn.Module):
    """With probability p, add N(0, sigma^2) noise to a float image in [0, 1], sigma drawn from `sigma`."""

    def __init__(self, sigma: tuple[float, float] = (0.01, 0.05), p: float = 0.3):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return img
        sigma = torch.empty(1).uniform_(*self.sigma).item()
        return (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, p={self.p}"


def build_transform(aug: str = "A0") -> v2.Compose:
    """PIL image -> normalized [3, 64, 64] tensor.

    A0: resize + normalization only (also used for validation and test).
    A1: A0 + RandomAffine (rotation +-10 deg, shift, scale) + ColorJitter (brightness, contrast).
    A2: A1 + Gaussian blur, darkening, Gaussian noise, RandomErasing.
    No flips (they change the meaning of some signs) and no JPEG (held out for evaluation).
    """
    if aug not in AUGMENTATIONS:
        raise ValueError(f"aug must be one of {AUGMENTATIONS}, got {aug!r}")
    ops = [v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)]
    if aug in ("A1", "A2"):
        ops += [
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    if aug == "A2":
        ops.append(v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3))
    ops.append(v2.ToDtype(torch.float32, scale=True))
    if aug == "A2":
        ops += [
            RandomDarken(factor=(0.5, 0.9), p=0.3),
            RandomGaussianNoise(sigma=(0.01, 0.05), p=0.3),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        ]
    ops.append(v2.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return v2.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization ([..., 3, H, W]) and clip to [0, 1] for display."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------------

class GTSRBDataset(Dataset):
    """Yields (image [3, 64, 64] float32, label int, roi [4] float32).

    `roi` is (x1, y1, x2, y2) of the annotated sign box rescaled from the original
    image size to 64x64 pixels (x2, y2 exclusive). It does not follow random
    geometric augmentation, so it is exact only with A0 (validation, test, XAI).

    cache=True decodes and resizes every image to uint8 64x64 once, in memory
    (~12 KB per image). Resizing is the first, deterministic step of every
    transform, so the outputs are identical to cache=False, only faster.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path = DATA_DIR, transform: v2.Compose | None = None,
                 cache: bool = False):
        self.data_dir = Path(data_dir)
        self.paths = df["Path"].tolist()
        self.labels = df["ClassId"].to_numpy(dtype=np.int64)
        sx = IMG_SIZE / df["Width"].to_numpy(dtype=np.float64)
        sy = IMG_SIZE / df["Height"].to_numpy(dtype=np.float64)
        self.rois = np.stack([
            df["Roi.X1"].to_numpy() * sx,
            df["Roi.Y1"].to_numpy() * sy,
            df["Roi.X2"].to_numpy() * sx,
            df["Roi.Y2"].to_numpy() * sy,
        ], axis=1).astype(np.float32)
        self.transform = transform if transform is not None else build_transform("A0")
        self.cache = self._decode_all() if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def load_image(self, idx: int) -> Image.Image:
        with Image.open(self.data_dir / self.paths[idx]) as im:
            return im.convert("RGB")

    def _decode_all(self) -> torch.Tensor:
        resize = v2.Compose([v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])
        with ThreadPoolExecutor(max_workers=8) as pool:
            imgs = pool.map(lambda i: resize(self.load_image(i)).as_subclass(torch.Tensor), range(len(self)))
            return torch.stack(list(imgs))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        img = self.cache[idx] if self.cache is not None else self.load_image(idx)
        x = self.transform(img).as_subclass(torch.Tensor)
        return x, int(self.labels[idx]), torch.from_numpy(self.rois[idx])

    def image_uint8(self, idx: int) -> torch.Tensor:
        """The resized image before any augmentation: uint8 [3, 64, 64]."""
        if self.cache is not None:
            return self.cache[idx]
        resize = v2.Compose([v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])
        return resize(self.load_image(idx)).as_subclass(torch.Tensor)

    def with_transform(self, transform: v2.Compose) -> "GTSRBDataset":
        """Shallow copy with another transform; the image cache is shared, not copied."""
        ds = copy.copy(self)
        ds.transform = transform
        return ds


SPLITS = ("train", "val", "test")


def stratified_subset(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """About n rows sampled per class in proportion to class size (smoke tests)."""
    frac = min(1.0, n / len(df))
    sub = df.groupby("ClassId", group_keys=False).sample(frac=frac, random_state=seed)
    return sub.sort_index().reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def cached_dataset(split: str, n: int | None = None, seed: int = SEED) -> GTSRBDataset:
    """GTSRBDataset(cache=True) with the A0 transform for 'train', 'val' or 'test', decoded once per process.

    n: optional stratified subset of about n images (smoke tests). Use with_transform()
    for augmentation, so that several runs in one process share the decoded images.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "test":
        df = load_csv("test")
    else:
        train_df, val_df = get_train_val()
        df = train_df if split == "train" else val_df
    if n is not None:
        df = stratified_subset(df, n, seed)
    return GTSRBDataset(df, transform=build_transform("A0"), cache=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: seed NumPy and `random` from the per-worker torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2,
                seed: int = SEED) -> DataLoader:
    """Seeded DataLoader with persistent workers.

    On Windows every worker re-imports torch at start-up (~10 s each, one after
    another), so workers pay off only for random augmentation (A1/A2) and are
    kept alive between epochs. Measured with GTSRBDataset(cache=True):
    A0 runs fastest with num_workers=0, A2 with 4 workers. Delete the loader
    before the interpreter exits to avoid a harmless atexit traceback.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create (once) and check the track-based train/val split.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--force", action="store_true", help="recreate the split even if it exists")
    args = parser.parse_args()

    if args.force or not args.split.exists():
        full = load_csv("train", args.data_dir)
        val_tracks, fold = make_val_split(full)
        save_split(val_tracks, fold, full, args.split)
        print(f"Saved {args.split} (StratifiedGroupKFold fold {fold})")
    else:
        print(f"Using existing {args.split}")

    train_df, val_df = get_train_val(args.data_dir, args.split)
    s = split_stats(train_df, val_df)
    print(f"train: {s['n_images']['train']} images, {s['n_tracks']['train']} tracks")
    print(f"val:   {s['n_images']['val']} images, {s['n_tracks']['val']} tracks "
          f"({s['val_fraction_tracks']:.1%} of tracks, {s['val_fraction_images']:.1%} of images)")
    print(f"tracks shared by train and val: {s['shared_tracks']}")
    print(f"classes in val: {s['val_classes']}/{NUM_CLASSES}, "
          f"min val tracks per class: {s['val_min_tracks_per_class']}, "
          f"min val images per class: {s['val_min_images_per_class']} (class {s['val_min_images_class']})")
    assert s["shared_tracks"] == 0, "train and val share tracks"
    assert s["val_classes"] == NUM_CLASSES and s["val_min_tracks_per_class"] >= 1, "class missing in val"

    for aug in AUGMENTATIONS:
        x, y, roi = GTSRBDataset(train_df, args.data_dir, build_transform(aug))[0]
        assert x.shape == (3, IMG_SIZE, IMG_SIZE) and x.dtype == torch.float32
        assert 0 <= roi[0] < roi[2] <= IMG_SIZE and 0 <= roi[1] < roi[3] <= IMG_SIZE
        print(f"{aug}: image {tuple(x.shape)}, label {y}, roi {[round(float(v), 1) for v in roi]}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
