"""Training entry point: python -m src.train --config configs/<run>.yaml [--smoke] [--force]

Config (YAML): run_name, model, pretrained, aug (A0/A1/A2), seed, epochs, batch_size, lr,
weight_decay, patience. Missing keys take the values in DEFAULTS.

AdamW + CosineAnnealingLR (stepped per epoch), cross-entropy, mixed precision on GPU
(torch.amp autocast, bf16 + channels_last where supported, see precision_for).
After every epoch: validation loss, accuracy and macro-F1 -> history.csv; the best
weights by validation macro-F1 -> best.pt; early stopping after `patience` epochs
without improvement. Outputs go to results/<run_name>/ (results/_smoke/<run_name>/
with --smoke: 1 epoch on ~2000 training and ~500 validation images).

A run is skipped when its metrics.json exists or its training is already complete
(train_summary.json + best.pt), unless --force. An interrupted run starts over.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data import AUGMENTATIONS, NUM_CLASSES, PROJECT_ROOT, build_transform, cached_dataset, make_loader, \
    seed_everything
from src.models import MODEL_NAMES, build_model, count_params

RESULTS_DIR = PROJECT_ROOT / "results"
SMOKE_DIR = RESULTS_DIR / "_smoke"
SMOKE_N_TRAIN, SMOKE_N_VAL = 2000, 500
EVAL_BATCH_SIZE = 512

DEFAULTS = {
    "pretrained": False,
    "aug": "A0",
    "seed": 42,
    "epochs": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 5,
}
CONFIG_KEYS = ("run_name", "model", *DEFAULTS)
# Files written by train/evaluate; removed before a run starts over.
RUN_FILES = ("config.json", "history.csv", "best.pt", "train_summary.json", "metrics.json",
             "per_class_f1.csv", "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Config and run directories
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(cfg) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    return validate_config({**DEFAULTS, **cfg})


def validate_config(cfg: dict) -> dict:
    missing = [k for k in CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config is missing {missing}")
    if cfg["model"] not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {cfg['model']!r}")
    if cfg["aug"] not in AUGMENTATIONS:
        raise ValueError(f"aug must be one of {AUGMENTATIONS}, got {cfg['aug']!r}")
    if cfg["model"] == "simple_cnn" and cfg["pretrained"]:
        raise ValueError("simple_cnn is trained from scratch only")
    return {k: cfg[k] for k in CONFIG_KEYS}


def save_config(cfg: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def results_root(smoke: bool, results_dir: Path | None = None) -> Path:
    return Path(results_dir) if results_dir is not None else (SMOKE_DIR if smoke else RESULTS_DIR)


def is_trained(out_dir: Path) -> bool:
    return (out_dir / "train_summary.json").exists() and (out_dir / "best.pt").exists()


def is_evaluated(out_dir: Path) -> bool:
    return (out_dir / "metrics.json").exists()


def num_workers_for(aug: str) -> int:
    """A0 is fastest in the main process; random augmentation pays for 4 persistent workers (see make_loader)."""
    return 0 if aug == "A0" else 4


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precision_for(device: torch.device) -> tuple[torch.dtype, torch.memory_format]:
    """Autocast dtype and memory format for mixed precision.

    bf16 + channels_last where supported: measured on the RTX 4070 Laptop (training
    steps/s, batch 128) it is the fastest setting, e.g. ResNet18 1330 img/s vs 990 with
    NCHW, while fp16 + channels_last drops to 300 img/s. Otherwise fp16 + NCHW.
    On CPU autocast is off (float32).
    """
    if device.type != "cuda":
        return torch.float32, torch.contiguous_format
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, torch.channels_last
    return torch.float16, torch.contiguous_format


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Predicted labels, true labels and mean cross-entropy over a loader."""
    dtype, memory_format = precision_for(device)
    model.eval()
    preds, labels = [], []
    loss_sum = torch.zeros((), device=device)
    for x, y, _roi in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=memory_format)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits.float(), y, reduction="sum")
        preds.append(logits.argmax(1))
        labels.append(y)
    preds, labels = torch.cat(preds).cpu().numpy(), torch.cat(labels).cpu().numpy()
    return preds, labels, loss_sum.item() / len(labels)


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(f1_score(labels, preds, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0))


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device) -> tuple[float, float]:
    dtype, memory_format = precision_for(device)
    model.train()
    # running sums stay on the GPU: a .item() per step would make the CPU wait for every batch
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    n = 0
    for x, y, _roi in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=memory_format)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * len(y)
        correct += (logits.argmax(1) == y).sum()
        n += len(y)
    return loss_sum.item() / n, correct.item() / n


def train(cfg: dict, smoke: bool = False, force: bool = False, results_dir: Path | None = None) -> Path:
    """Train one run; returns its output directory. Skips finished runs unless force."""
    cfg = validate_config(cfg)
    out_dir = results_root(smoke, results_dir) / cfg["run_name"]
    if not force and (is_evaluated(out_dir) or is_trained(out_dir)):
        print(f"[{cfg['run_name']}] already trained ({out_dir}), skipping (use --force to retrain)")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (out_dir / name).unlink(missing_ok=True)

    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = get_device()
    epochs = 1 if smoke else cfg["epochs"]
    train_base = cached_dataset("train", SMOKE_N_TRAIN if smoke else None)
    val_ds = cached_dataset("val", SMOKE_N_VAL if smoke else None)
    train_ds = train_base.with_transform(build_transform(cfg["aug"]))
    workers = num_workers_for(cfg["aug"])
    train_loader = make_loader(train_ds, cfg["batch_size"], shuffle=True, num_workers=workers, seed=cfg["seed"])
    val_loader = make_loader(val_ds, EVAL_BATCH_SIZE, shuffle=False, num_workers=0, seed=cfg["seed"])

    amp_dtype, memory_format = precision_for(device)
    model = build_model(cfg["model"], cfg["pretrained"]).to(device, memory_format=memory_format)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)  # bf16 needs no scaling

    run_info = {
        **cfg,
        "epochs_effective": epochs,
        "num_workers": workers,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR(T_max=epochs), stepped per epoch",
        "loss": "CrossEntropyLoss",
        "early_stopping": "validation macro-F1",
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if device.type == "cuda" else None,
        "channels_last": memory_format == torch.channels_last,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "smoke": smoke,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "params": count_params(model),
    }
    (out_dir / "config.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(f"[{cfg['run_name']}] {cfg['model']} {'ImageNet' if cfg['pretrained'] else 'scratch'} {cfg['aug']} "
          f"seed {cfg['seed']} | {run_info['params']:,} params | train {len(train_ds)}, val {len(val_ds)} | "
          f"{run_info['device']}, workers {workers}")

    history_path = out_dir / "history.csv"
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_time_s"]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fields)

    best_f1, best_epoch, stale, epoch = -1.0, 0, 0, 0
    t_start = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
            scheduler.step()
            preds, labels, val_loss = predict(model, val_loader, device)
            val_acc, val_f1 = float((preds == labels).mean()), macro_f1(labels, preds)
            epoch_time = time.perf_counter() - t0
            row = [epoch, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{val_loss:.5f}", f"{val_acc:.5f}",
                   f"{val_f1:.5f}", f"{lr:.3e}", f"{epoch_time:.2f}"]
            with history_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            improved = val_f1 > best_f1
            if improved:
                best_f1, best_epoch, stale = val_f1, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1, "config": cfg},
                           out_dir / "best.pt")
            else:
                stale += 1
            print(f"[{cfg['run_name']}] epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f} F1 {val_f1:.4f}{' *' if improved else ''} "
                  f"| {epoch_time:.1f} s")
            if stale >= cfg["patience"]:
                print(f"[{cfg['run_name']}] early stopping: no improvement for {stale} epochs")
                break
    finally:
        del train_loader, val_loader  # stop persistent workers

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 6),
        "epochs_run": epoch,
        "stopped_early": epoch < epochs,
        "train_time_s": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[{cfg['run_name']}] done: best val macro-F1 {best_f1:.4f} at epoch {best_epoch}, "
          f"{summary['train_time_s']:.0f} s")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one run from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="1 epoch on ~2000/500 images, results/_smoke/")
    parser.add_argument("--force", action="store_true", help="retrain even if the run is finished")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="parent of the run directory (default: results/ or results/_smoke/)")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke, force=args.force, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
