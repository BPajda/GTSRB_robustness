"""Training entry point: python -m src.train --config configs/<run>.yaml [--smoke] [--force]

Config (YAML): run_name, model, pretrained, aug (A0/A1/A2), seed, epochs, batch_size, lr,
weight_decay, patience, finetune, backbone_lr, freeze_epochs. Missing keys take the
values in DEFAULTS.

AdamW + CosineAnnealingLR (stepped per epoch), cross-entropy, mixed precision on GPU
(torch.amp autocast, bf16 + channels_last where supported, see precision_for).
After every epoch: validation loss, accuracy and macro-F1 -> history.csv; the best
weights by validation macro-F1 -> best.pt; early stopping after `patience` epochs
without improvement. Outputs go to results/<run_name>/ (results/_smoke/<run_name>/
with --smoke: 1 epoch on ~2000 training and ~500 validation images; two_stage gets
2 epochs there, 1 per stage).

Fine-tuning strategies (`finetune`, pretrained models only except "full"):
- full (default): the whole network, one AdamW group with lr (the original setup).
- linear: backbone frozen (requires_grad=False, its BatchNorm layers stay in eval mode),
  AdamW on the head only (ResNet fc / MobileNetV3 classifier[-1]) with lr.
- discriminative: the whole network, two AdamW groups: backbone backbone_lr, head lr;
  cosine on both.
- two_stage: freeze_epochs epochs as "linear" with a constant head lr, then the backbone
  is unfrozen and training goes on as "discriminative" with cosine over the remaining
  epochs (the head keeps its AdamW state). Early stopping counts from epoch 1 but can
  stop the run only in stage 2, after at least one epoch with the backbone unfrozen.

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
from src.models import MODEL_NAMES, backbone_parameters, build_model, count_params, count_trainable_params, \
    head_parameters, set_backbone_trainable, set_train_mode

RESULTS_DIR = PROJECT_ROOT / "results"
SMOKE_DIR = RESULTS_DIR / "_smoke"
SMOKE_N_TRAIN, SMOKE_N_VAL = 2000, 500
EVAL_BATCH_SIZE = 512
FINETUNE_STRATEGIES = ("full", "linear", "discriminative", "two_stage")

DEFAULTS = {
    "pretrained": False,
    "aug": "A0",
    "seed": 42,
    "epochs": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 5,
    "finetune": "full",
    "backbone_lr": 1e-4,   # discriminative, two_stage
    "freeze_epochs": 3,    # two_stage
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
    strategy = cfg["finetune"]
    if strategy not in FINETUNE_STRATEGIES:
        raise ValueError(f"finetune must be one of {FINETUNE_STRATEGIES}, got {strategy!r}")
    if strategy != "full" and cfg["model"] == "simple_cnn":
        raise ValueError(f"finetune {strategy!r} needs a pretrained backbone; simple_cnn is trained from scratch "
                         f"only (use finetune: full)")
    if strategy != "full" and not cfg["pretrained"]:
        raise ValueError(f"finetune {strategy!r} is for pretrained models only (pretrained: true, or finetune: full)")
    lr = cfg["backbone_lr"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"backbone_lr must be a positive number, got {lr!r} "
                         f"(YAML reads 1e-4 as a string: write 1.0e-4 or 0.0001)")
    n = cfg["freeze_epochs"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"freeze_epochs must be an integer >= 1, got {n!r}")
    if strategy == "two_stage" and n >= cfg["epochs"]:
        raise ValueError(f"two_stage needs freeze_epochs < epochs ({n} >= {cfg['epochs']}): "
                         f"at least one epoch with the backbone unfrozen")
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
    set_train_mode(model)  # model.train(), but BatchNorm of a frozen backbone stays in eval mode
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


# ---------------------------------------------------------------------------
# Fine-tuning strategies
# ---------------------------------------------------------------------------

Scheduler = torch.optim.lr_scheduler.LRScheduler


def setup_finetune(model: nn.Module, cfg: dict, epochs: int) -> tuple[torch.optim.Optimizer, Scheduler | None]:
    """Optimizer and scheduler of cfg["finetune"]; freezes the backbone for linear and stage 1 of two_stage.

    With two parameter groups the head group is the last one. In stage 1 of two_stage the
    backbone group is already in the optimizer but gets no gradients, so AdamW skips it
    (no update, no weight decay); the scheduler is None there (constant lr) and comes from
    unfreeze_backbone() at the start of stage 2.
    """
    strategy, wd = cfg["finetune"], cfg["weight_decay"]
    if strategy == "full":  # the original setup, unchanged
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=wd)
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if strategy in ("linear", "two_stage"):
        set_backbone_trainable(model, False)
    if strategy == "linear":
        optimizer = torch.optim.AdamW(head_parameters(model), lr=cfg["lr"], weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW([{"params": backbone_parameters(model), "lr": cfg["backbone_lr"]},
                                       {"params": head_parameters(model), "lr": cfg["lr"]}],
                                      lr=cfg["lr"], weight_decay=wd)
    if strategy == "two_stage":
        return optimizer, None
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def unfreeze_backbone(model: nn.Module, optimizer: torch.optim.Optimizer, remaining_epochs: int) -> Scheduler:
    """Stage 2 of two_stage: the whole network trains, cosine from backbone_lr / lr over the remaining epochs."""
    set_backbone_trainable(model, True)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)


def finetune_description(cfg: dict) -> dict:
    """Human-readable optimizer / scheduler / early-stopping setup for config.json."""
    strategy = cfg["finetune"]
    info = {
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR(T_max=epochs), stepped per epoch",
        "loss": "CrossEntropyLoss",
        "early_stopping": "validation macro-F1",
    }
    if strategy == "linear":
        info["optimizer"] = "AdamW on the head only (lr), backbone frozen with BatchNorm in eval mode"
    elif strategy == "discriminative":
        info["optimizer"] = "AdamW, 2 param groups: backbone (backbone_lr), head (lr)"
        info["scheduler"] = "CosineAnnealingLR(T_max=epochs) on both groups, stepped per epoch"
    elif strategy == "two_stage":
        info["optimizer"] = ("AdamW, 2 param groups: backbone (backbone_lr), head (lr); stage 1 (epochs "
                             "1..freeze_epochs): backbone frozen with BatchNorm in eval mode, only the head trains")
        info["scheduler"] = ("stage 1: constant lr; stage 2: CosineAnnealingLR(T_max=epochs - freeze_epochs) on "
                             "both groups, stepped per epoch")
        info["early_stopping"] = "validation macro-F1, counted from epoch 1, may stop the run only in stage 2"
    return info


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
    strategy = cfg["finetune"]
    two_stage = strategy == "two_stage"
    epochs, freeze_epochs = cfg["epochs"], cfg["freeze_epochs"]
    if smoke:  # two_stage needs one epoch per stage to exercise the unfreezing
        epochs, freeze_epochs = (2, 1) if two_stage else (1, freeze_epochs)
    train_base = cached_dataset("train", SMOKE_N_TRAIN if smoke else None)
    val_ds = cached_dataset("val", SMOKE_N_VAL if smoke else None)
    train_ds = train_base.with_transform(build_transform(cfg["aug"]))
    workers = num_workers_for(cfg["aug"])
    train_loader = make_loader(train_ds, cfg["batch_size"], shuffle=True, num_workers=workers, seed=cfg["seed"])
    val_loader = make_loader(val_ds, EVAL_BATCH_SIZE, shuffle=False, num_workers=0, seed=cfg["seed"])

    amp_dtype, memory_format = precision_for(device)
    model = build_model(cfg["model"], cfg["pretrained"]).to(device, memory_format=memory_format)
    optimizer, scheduler = setup_finetune(model, cfg, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)  # bf16 needs no scaling
    two_groups = len(optimizer.param_groups) == 2  # discriminative, two_stage: [backbone, head]

    run_info = {
        **cfg,
        "epochs_effective": epochs,
        **({"freeze_epochs_effective": freeze_epochs} if two_stage else {}),
        "num_workers": workers,
        "eval_batch_size": EVAL_BATCH_SIZE,
        **finetune_description(cfg),
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if device.type == "cuda" else None,
        "channels_last": memory_format == torch.channels_last,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "smoke": smoke,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "params": count_params(model),
        # parameters updated at some point of the training (two_stage: all, after unfreezing)
        "trainable_params": count_params(model) if two_stage else count_trainable_params(model),
        **({"trainable_params_stage1": count_trainable_params(model)} if two_stage else {}),
    }
    (out_dir / "config.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    finetune_note = "" if strategy == "full" else \
        f" | finetune {strategy}, {count_trainable_params(model):,} trainable"
    print(f"[{cfg['run_name']}] {cfg['model']} {'ImageNet' if cfg['pretrained'] else 'scratch'} {cfg['aug']} "
          f"seed {cfg['seed']} | {run_info['params']:,} params{finetune_note} | train {len(train_ds)}, "
          f"val {len(val_ds)} | {run_info['device']}, workers {workers}")

    history_path = out_dir / "history.csv"
    # lr = head lr (the only group for full / linear); lr_backbone = 0 while the backbone is frozen
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr",
              *(["lr_backbone"] if two_groups else []), "epoch_time_s"]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fields)

    best_f1, best_epoch, stale, epoch = -1.0, 0, 0, 0
    t_start = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            if two_stage and epoch == freeze_epochs + 1:
                scheduler = unfreeze_backbone(model, optimizer, epochs - freeze_epochs)
                print(f"[{cfg['run_name']}] epoch {epoch}: backbone unfrozen, "
                      f"{count_trainable_params(model):,} trainable params")
            backbone_frozen = two_stage and epoch <= freeze_epochs
            lr = optimizer.param_groups[-1]["lr"]
            lr_backbone = [0.0 if backbone_frozen else optimizer.param_groups[0]["lr"]] if two_groups else []
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
            if scheduler is not None:
                scheduler.step()
            preds, labels, val_loss = predict(model, val_loader, device)
            val_acc, val_f1 = float((preds == labels).mean()), macro_f1(labels, preds)
            epoch_time = time.perf_counter() - t0
            row = [epoch, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{val_loss:.5f}", f"{val_acc:.5f}",
                   f"{val_f1:.5f}", f"{lr:.3e}", *(f"{v:.3e}" for v in lr_backbone), f"{epoch_time:.2f}"]
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
            if stale >= cfg["patience"] and not (two_stage and epoch <= freeze_epochs):
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
