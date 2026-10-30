"""Clean and corrupted evaluation of a trained run; writes results/<run_name>/metrics.json.

    python -m src.evaluate --config configs/<run>.yaml [--smoke] [--force]
    python -m src.evaluate --run <run_name or results dir> [--force]
    python -m src.evaluate --latency-only [--results-dir results/_smoke]

Uses best.pt of the run. Computes on the validation set: accuracy, macro-F1,
per_class_f1.csv, confusion_matrix.png, accuracy under each of the 15 corruptions
(5 types x 3 levels), mCA, relative drop, parameter count and CPU latency
(median of 100 single-image predictions after warm-up).

--latency-only re-measures just cpu_latency_ms and cpu_latency_ms_1thread of every
evaluated run directly under --results-dir (default results/, so results/_smoke/ is
not included) and overwrites only these two keys of metrics.json: no training, no
evaluation, only best.pt is loaded on CPU.

The test set is locked: --split test also needs --i-know-this-is-final and writes
separate *_test files. It is used once, in KM3, for the final models only.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from src.corruptions import CORRUPTIONS, LEVELS, CorruptedDataset
from src.data import CLASS_NAMES, IMG_SIZE, NUM_CLASSES, cached_dataset, make_loader
from src.models import build_model, count_params
from src.train import EVAL_BATCH_SIZE, SMOKE_N_VAL, get_device, is_evaluated, is_trained, load_config, \
    macro_f1, precision_for, predict, results_root

LATENCY_WARMUP, LATENCY_REPEATS = 20, 100
# Fixed thread count: with the default (24 on the i9-13900HX) small models get slower
# (MobileNetV3-Small ~17 ms vs ~6 ms at 1-4 threads), so the comparison would depend on
# thread scheduling. 4 threads are close to the fastest setting for all three models.
LATENCY_THREADS = 4
METRIC_KEYS = ("clean", "corrupted", "mca", "relative_drop", "params", "cpu_latency_ms", "best_epoch",
               "train_time_s")


def metrics_complete(metrics: dict) -> bool:
    """All keys of the metrics.json format, with all 15 corruption accuracies."""
    if any(k not in metrics for k in METRIC_KEYS):
        return False
    if not {"acc", "macro_f1"} <= set(metrics["clean"]):
        return False
    return all(str(lv) in metrics["corrupted"].get(c, {}) for c in CORRUPTIONS for lv in LEVELS)


def load_model(out_dir: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device, memory_format=precision_for(device)[1]).eval(), ckpt


@torch.inference_mode()
def cpu_latency_ms(model: nn.Module, threads: int = LATENCY_THREADS, warmup: int = LATENCY_WARMUP,
                   repeats: int = LATENCY_REPEATS) -> float:
    """Median wall time of one batch-1 prediction on CPU (fp32) with `threads` torch threads."""
    model = copy.deepcopy(model).to("cpu", memory_format=torch.contiguous_format).eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, generator=torch.Generator().manual_seed(0))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(old_threads)
    return float(np.median(times) * 1e3)


def save_per_class(labels: np.ndarray, preds: np.ndarray, path: Path) -> None:
    p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=np.arange(NUM_CLASSES),
                                                        zero_division=0)
    pd.DataFrame({
        "class_id": np.arange(NUM_CLASSES),
        "class_name": CLASS_NAMES,
        "precision": p.round(5),
        "recall": r.round(5),
        "f1": f1.round(5),
        "support": support,
    }).to_csv(path, index=False)


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cm = confusion_matrix(labels, preds, labels=np.arange(NUM_CLASSES))
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    masked = np.ma.masked_equal(share, 0)
    im = ax.imshow(masked, cmap="Blues", norm=LogNorm(vmin=1e-3, vmax=1), interpolation="nearest")
    ax.set_facecolor("#fcfcfb")
    ticks = np.arange(NUM_CLASSES)
    ax.set_xticks(ticks, ticks, fontsize=7)
    ax.set_yticks(ticks, ticks, fontsize=7)
    ax.set_xlabel("Klasa przewidziana")
    ax.set_ylabel("Klasa rzeczywista")
    ax.set_title(title, fontsize=11)
    for k in range(NUM_CLASSES + 1):  # hairline grid between cells
        ax.axhline(k - 0.5, color="#e1e0d9", lw=0.3)
        ax.axvline(k - 0.5, color="#e1e0d9", lw=0.3)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    for i, j in zip(*np.nonzero(off)):  # count of each confusion, so small errors are readable
        ax.text(j, i, str(off[i, j]), ha="center", va="center", fontsize=4.5,
                color="white" if share[i, j] > 0.1 else "#0b0b0b")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("odsetek zdjęć klasy rzeczywistej (skala log.; liczby w komórkach = liczba pomyłek)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_on(model: nn.Module, dataset, device: torch.device) -> float:
    preds, labels, _ = predict(model, make_loader(dataset, EVAL_BATCH_SIZE, shuffle=False, num_workers=0), device)
    return float((preds == labels).mean())


def evaluate(out_dir: Path, split: str = "val", smoke: bool = False, force: bool = False,
             final: bool = False) -> dict:
    """Evaluate best.pt of the run in out_dir and write metrics(.json|_test.json)."""
    if split == "test" and not final:
        raise PermissionError("the test set is used once, in KM3: pass final=True / --i-know-this-is-final")
    out_dir = Path(out_dir)
    if not is_trained(out_dir):
        raise FileNotFoundError(f"{out_dir} has no finished training (best.pt + train_summary.json)")
    suffix = "" if split == "val" else f"_{split}"
    metrics_path = out_dir / f"metrics{suffix}.json"
    if metrics_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_complete(metrics):
            print(f"[{out_dir.name}] {metrics_path.name} exists, skipping (use --force to recompute)")
            return metrics

    device = get_device()
    model, ckpt = load_model(out_dir, device)
    train_summary = json.loads((out_dir / "train_summary.json").read_text(encoding="utf-8"))
    base = cached_dataset(split, SMOKE_N_VAL if smoke else None)
    t0 = time.perf_counter()

    preds, labels, _ = predict(model, make_loader(base, EVAL_BATCH_SIZE, shuffle=False, num_workers=0), device)
    acc, f1 = float((preds == labels).mean()), macro_f1(labels, preds)
    save_per_class(labels, preds, out_dir / f"per_class_f1{suffix}.csv")
    split_pl = {"val": "walidacja", "test": "zbiór testowy"}[split]
    save_confusion_matrix(labels, preds, f"Macierz pomyłek – {out_dir.name} ({split_pl}, n = {len(labels)}, "
                                         f"acc = {acc:.2%}, macro-F1 = {f1:.4f})".replace(".", ","),
                          out_dir / f"confusion_matrix{suffix}.png")
    print(f"[{out_dir.name}] {split} clean: acc {acc:.4f}, macro-F1 {f1:.4f}")

    corrupted = {}
    for ctype in CORRUPTIONS:
        corrupted[ctype] = {str(lv): round(accuracy_on(model, CorruptedDataset(base, ctype, lv), device), 6)
                            for lv in LEVELS}
        print(f"[{out_dir.name}] {ctype:<15} " + "  ".join(f"L{k}: {v:.4f}" for k, v in corrupted[ctype].items()))
    mca = float(np.mean([v for levels in corrupted.values() for v in levels.values()]))

    metrics = {
        "clean": {"acc": round(acc, 6), "macro_f1": round(f1, 6)},
        "corrupted": corrupted,
        "mca": round(mca, 6),
        "relative_drop": round((acc - mca) / acc, 6),
        "params": count_params(model),
        "cpu_latency_ms": round(cpu_latency_ms(model), 3),
        "best_epoch": int(ckpt["epoch"]),
        "train_time_s": train_summary["train_time_s"],
        "split": split,
        "n_images": int(len(labels)),
        "cpu_latency_threads": LATENCY_THREADS,
        "cpu_latency_ms_1thread": round(cpu_latency_ms(model, threads=1), 3),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[{out_dir.name}] mCA {mca:.4f}, relative drop {metrics['relative_drop']:.4f}, "
          f"CPU latency {metrics['cpu_latency_ms']:.2f} ms, evaluation {time.perf_counter() - t0:.0f} s "
          f"-> {metrics_path}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def remeasure_latency(results_dir: Path) -> list[Path]:
    """Re-measure CPU latency (LATENCY_THREADS and 1 thread) of every evaluated run in results_dir.

    Only the run directories directly under results_dir are taken (so results/_smoke/ is
    skipped for results/). Overwrites cpu_latency_ms and cpu_latency_ms_1thread in
    metrics.json; all other keys, and their order, stay as they were.
    """
    run_dirs = [p for p in sorted(Path(results_dir).iterdir()) if is_trained(p) and is_evaluated(p)]
    if not run_dirs:
        raise FileNotFoundError(f"{results_dir} has no evaluated runs (best.pt + train_summary.json + metrics.json)")
    for out_dir in run_dirs:
        metrics_path = out_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        old = metrics.get("cpu_latency_ms"), metrics.get("cpu_latency_ms_1thread")
        model, _ = load_model(out_dir, torch.device("cpu"))
        metrics["cpu_latency_ms"] = round(cpu_latency_ms(model), 3)
        metrics["cpu_latency_ms_1thread"] = round(cpu_latency_ms(model, threads=1), 3)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"[{out_dir.name}] CPU latency {LATENCY_THREADS} threads {old[0]} -> {metrics['cpu_latency_ms']} ms, "
              f"1 thread {old[1]} -> {metrics['cpu_latency_ms_1thread']} ms")
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best weights of a run on clean and corrupted data.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--config", type=Path, help="run config (YAML); the run directory follows from run_name")
    which.add_argument("--run", help="run name (under results/ or results/_smoke/) or a run directory")
    which.add_argument("--latency-only", action="store_true",
                       help="only re-measure CPU latency of all evaluated runs in --results-dir")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="--latency-only: parent of the run directories (default: results/ or results/_smoke/)")
    parser.add_argument("--smoke", action="store_true", help="results/_smoke/ and ~500 validation images")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--i-know-this-is-final", dest="final", action="store_true",
                        help="unlock --split test (KM3 only, final models only)")
    parser.add_argument("--force", action="store_true", help="recompute even if metrics exist")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        parser.error("the test set is locked until KM3; add --i-know-this-is-final to use it")
    if args.latency_only:
        if args.split != "val" or args.force:
            parser.error("--latency-only only rewrites the latency keys of metrics.json (no --split/--force)")
        remeasure_latency(results_root(args.smoke, args.results_dir))
        return
    if args.results_dir is not None:
        parser.error("--results-dir is used with --latency-only")
    if args.config is not None:
        out_dir = results_root(args.smoke) / load_config(args.config)["run_name"]
    else:
        out_dir = Path(args.run)
        if not out_dir.is_dir():
            out_dir = results_root(args.smoke) / args.run
    evaluate(out_dir, split=args.split, smoke=args.smoke, force=args.force, final=args.final)


if __name__ == "__main__":
    main()
