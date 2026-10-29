"""Summary table and comparison plots across runs in results/.

    python -m src.summarize                              # results/ -> results/summary/
    python -m src.summarize --results-dir results/_smoke # smoke runs -> results/_smoke/summary/

Writes summary.csv (every run with metrics.json, E4 seed repeats included) and PNG plots
(Polish labels) for the 11 planned variants (E1 + E2 runs listed in e2_selection.json):
    h1_learning_curves.png   validation macro-F1 per epoch, scratch vs ImageNet (H1)
    f1_mca_bars.png          clean macro-F1 and mCA of every variant
    robustness_curves.png    accuracy vs corruption level, A0/A1/A2 per architecture (H2, H4)
    params_vs_f1.png         parameter count (log) vs macro-F1 (H3)
    corruption_heatmap.png   variant x corruption, accuracy averaged over levels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.corruptions import CORRUPTION_LABELS_PL, CORRUPTIONS, LEVELS, level_label_pl  # noqa: E402
from src.train import RESULTS_DIR  # noqa: E402

MODEL_ORDER = ("simple_cnn", "resnet18", "mobilenet_v3_small")
MODEL_LABELS = {"simple_cnn": "SimpleCNN", "resnet18": "ResNet18", "mobilenet_v3_small": "MobileNetV3-Small"}
# Categorical slots 1-3 of the reference palette (validated all-pairs for 3 series), one per architecture.
MODEL_COLORS = {"simple_cnn": "#2a78d6", "resnet18": "#eb6834", "mobilenet_v3_small": "#1baf7a"}
# Augmentation strength is ordered: one-hue ordinal ramp (light -> dark) plus a marker per level.
AUG_COLORS = {"A0": "#86b6ef", "A1": "#2a78d6", "A2": "#104281"}
AUG_MARKERS = {"A0": "o", "A1": "s", "A2": "^"}
AUG_LABELS = {"A0": "A0 (brak)", "A1": "A1 (geometryczna)", "A2": "A2 (odpornościowa)"}
AUG_JITTER = {"A0": 0.93, "A1": 1.0, "A2": 1.075}  # horizontal offset factor on the log params axis
SCRATCH_COLOR = "#898781"
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
UNSEEN_BG = "#f0efec"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "figure.dpi": 100,
})


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def stage_of(cfg: dict) -> str:
    if cfg["seed"] != 42:
        return "E4"
    return "E1" if cfg["aug"] == "A0" else "E2"


def variant_label(row) -> str:
    init = "ImageNet" if row["pretrained"] else "od zera"
    return f"{MODEL_LABELS[row['model']]} · {init} · {row['aug']}"


def collect_runs(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        mpath, cpath = run_dir / "metrics.json", run_dir / "config.json"
        if not (mpath.exists() and cpath.exists()):
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        cfg = json.loads(cpath.read_text(encoding="utf-8"))
        ts_path = run_dir / "train_summary.json"
        ts = json.loads(ts_path.read_text(encoding="utf-8")) if ts_path.exists() else {}
        row = {
            "run_name": cfg["run_name"],
            "stage": stage_of(cfg),
            "model": cfg["model"],
            "pretrained": bool(cfg["pretrained"]),
            "aug": cfg["aug"],
            "seed": cfg["seed"],
            "best_epoch": m["best_epoch"],
            "epochs_run": ts.get("epochs_run"),
            "acc": m["clean"]["acc"],
            "macro_f1": m["clean"]["macro_f1"],
            "mca": m["mca"],
            "relative_drop": m["relative_drop"],
            "params": m["params"],
            "cpu_latency_ms": m["cpu_latency_ms"],
            "cpu_latency_ms_1thread": m.get("cpu_latency_ms_1thread"),
            "train_time_s": m["train_time_s"],
        }
        for c in CORRUPTIONS:
            for lv in LEVELS:
                row[f"acc_{c}_{lv}"] = m["corrupted"][c][str(lv)]
        for c in CORRUPTIONS:
            row[f"acc_{c}_mean"] = float(np.mean([m["corrupted"][c][str(lv)] for lv in LEVELS]))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no runs with metrics.json in {root}")
    df = pd.DataFrame(rows)
    df["_order"] = df["model"].map(MODEL_ORDER.index)
    df = df.sort_values(["seed", "_order", "pretrained", "aug"]).drop(columns="_order").reset_index(drop=True)
    df["label"] = df.apply(variant_label, axis=1)
    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    out = df.drop(columns="label")
    float_cols = out.select_dtypes("float").columns
    out[float_cols] = out[float_cols].round(6)
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def pct_axis(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(
        lambda v, _p: f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def plot_learning_curves(df: pd.DataFrame, root: Path, path: Path) -> None:
    """H1: validation macro-F1 per epoch, scratch vs ImageNet, one panel per torchvision architecture."""
    a0 = df[df["aug"] == "A0"]
    models = [m for m in ("resnet18", "mobilenet_v3_small") if m in set(a0["model"])]
    if not models:
        print(f"no A0 runs of resnet18 / mobilenet_v3_small, skipping {path.name}")
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        max_epoch = 1
        for pretrained in (False, True):
            sel = df[(df["model"] == model) & (df["pretrained"] == pretrained) & (df["aug"] == "A0")]
            if sel.empty:
                continue
            row = sel.iloc[0]
            hist = pd.read_csv(root / row["run_name"] / "history.csv")
            max_epoch = max(max_epoch, int(hist["epoch"].max()))
            color = MODEL_COLORS[model] if pretrained else SCRATCH_COLOR
            label = "ImageNet (pretrenowany)" if pretrained else "od zera"
            y = hist["val_macro_f1"] * 100
            ax.plot(hist["epoch"], y, color=color, lw=2, marker="o" if pretrained else "s", ms=4, label=label)
            best = hist.loc[hist["epoch"] == row["best_epoch"]]
            ax.plot(best["epoch"], best["val_macro_f1"] * 100, marker="*", ms=13, color=color, mec="white",
                    mew=0.8, ls="none")
            ax.annotate(f"{row['macro_f1'] * 100:.2f}%".replace(".", ","), (best["epoch"].iloc[0], y.max()),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.set_title(f"{MODEL_LABELS[model]}, A0")
        ax.set_xlabel("Epoka")
        ax.set_xlim(0.5, max_epoch + 0.5)
        ax.set_xticks(range(1, max_epoch + 1, 1 if max_epoch <= 10 else 2))
        pct_axis(ax)
        ax.legend(loc="lower right")
    axes[0, 0].set_ylabel("Macro-F1 na walidacji")
    fig.suptitle("H1: uczenie od zera vs wagi ImageNet – krzywe uczenia (★ = najlepsza epoka, zapisane wagi)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_f1_mca_bars(df: pd.DataFrame, path: Path) -> None:
    """Clean macro-F1 and mCA per variant, two panels (different scales, no dual axis)."""
    d = df.iloc[::-1]  # first variant on top
    ypos = np.arange(len(d))
    colors = [MODEL_COLORS[m] for m in d["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(d) + 1.6), sharey=True)
    for ax, col, title in ((axes[0], "macro_f1", "Macro-F1 na czystej walidacji"),
                           (axes[1], "mca", "mCA – średnia accuracy po 15 zakłóceniach")):
        vals = d[col].to_numpy() * 100
        ax.barh(ypos, vals, height=0.72, color=colors, edgecolor="white", linewidth=1.5)
        for y, v in zip(ypos, vals):
            ax.text(v + 0.8, y, f"{v:.2f}%".replace(".", ","), va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 112)
        ax.set_xticks(range(0, 101, 20))
        pct_axis(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[0].set_yticks(ypos, d["label"])
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=8, label=MODEL_LABELS[m]) for m in MODEL_ORDER
               if m in set(d["model"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Skuteczność na czystych danych i odporność na zakłócenia – wszystkie warianty", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def robustness_rows(df: pd.DataFrame) -> list[tuple[str, bool]]:
    """(model, pretrained) pairs to show: those trained with A1/A2 (E2), or all of them before E2."""
    combos = list(dict.fromkeys(zip(df["model"], df["pretrained"])))
    with_aug = [c for c in combos if ((df["model"] == c[0]) & (df["pretrained"] == c[1]) & (df["aug"] != "A0")).any()]
    rows = with_aug or combos
    return sorted(rows, key=lambda c: (MODEL_ORDER.index(c[0]), c[1]))


def plot_robustness_curves(df: pd.DataFrame, path: Path) -> None:
    """H2/H4: accuracy vs corruption level (0 = clean), A0/A1/A2, rows = architectures, columns = corruptions."""
    rows = robustness_rows(df)
    # sharex is off on purpose: every column has its own level labels (sigma, factor, quality...)
    fig, axes = plt.subplots(len(rows), len(CORRUPTIONS), figsize=(3.0 * len(CORRUPTIONS), 2.5 * len(rows) + 1.0),
                             sharey=True, squeeze=False)
    x = [0, *LEVELS]
    for r, (model, pretrained) in enumerate(rows):
        for c, ctype in enumerate(CORRUPTIONS):
            ax = axes[r, c]
            if ctype == "jpeg":
                ax.set_facecolor(UNSEEN_BG)
            for aug in ("A0", "A1", "A2"):
                sel = df[(df["model"] == model) & (df["pretrained"] == pretrained) & (df["aug"] == aug)]
                if sel.empty:
                    continue
                row = sel.iloc[0]
                y = [row["acc"] * 100] + [row[f"acc_{ctype}_{lv}"] * 100 for lv in LEVELS]
                ax.plot(x, y, color=AUG_COLORS[aug], marker=AUG_MARKERS[aug], ms=5, lw=2, label=AUG_LABELS[aug])
            if r == len(rows) - 1:
                ax.set_xticks(x, ["czyste"] + [f"{lv}\n{level_label_pl(ctype, lv)}" for lv in LEVELS], fontsize=7)
            else:
                ax.set_xticks(x, ["czyste", *map(str, LEVELS)], fontsize=7)
            pct_axis(ax)
            if r == 0:
                title = CORRUPTION_LABELS_PL[ctype]
                if ctype == "jpeg":
                    ax.set_title(f"{title}\nNIEWIDZIANA W TRENINGU (H4)", fontweight="bold")
                else:
                    ax.set_title(f"{title}\n(w augmentacji A2)")
            if c == 0:
                init = "ImageNet" if pretrained else "od zera"
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n({init})\naccuracy", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("poziom zakłócenia")
    handles = [Line2D([], [], color=AUG_COLORS[a], marker=AUG_MARKERS[a], lw=2, label=AUG_LABELS[a])
               for a in ("A0", "A1", "A2") if a in set(df["aug"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01),
               title="Augmentacja w treningu")
    fig.suptitle("H2 i H4: accuracy na walidacji w zależności od poziomu zakłócenia "
                 "(szare tło = zakłócenie nieobecne w żadnej augmentacji)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Push sorted label positions apart by at least min_gap, keeping their mean."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    placed -= (placed.mean() - np.mean(ys))
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def plot_params_vs_f1(df: pd.DataFrame, path: Path) -> None:
    """H3: parameter count (log scale) vs clean macro-F1."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y_all = df["macro_f1"].to_numpy() * 100
    span = max(y_all.max() - y_all.min(), 1.0)
    for model in MODEL_ORDER:
        d = df[df["model"] == model]
        if d.empty:
            continue
        x = d["params"] * d["aug"].map(AUG_JITTER)  # same size: spread A0/A1/A2 slightly so markers don't hide
        for (_, row), xi in zip(d.iterrows(), x):
            ax.scatter(xi, row["macro_f1"] * 100, s=70, marker=AUG_MARKERS[row["aug"]], zorder=3,
                       color=MODEL_COLORS[model] if row["pretrained"] else "white",
                       edgecolor=MODEL_COLORS[model], linewidth=1.8)
        label_y = spread_labels(list(d["macro_f1"] * 100), min_gap=span * 0.07)
        for (_, row), xi, ly in zip(d.iterrows(), x, label_y):
            text = f"{row['aug']}, {'ImageNet' if row['pretrained'] else 'od zera'}: " \
                   f"{row['macro_f1'] * 100:.2f}%".replace(".", ",")
            ax.annotate(text, (xi, row["macro_f1"] * 100), xytext=(row["params"] * 1.4, ly),
                        textcoords="data", va="center", fontsize=7.5, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6})
    ax.set_xscale("log")
    ax.set_xlim(df["params"].min() / 2, df["params"].max() * 6)
    ax.set_ylim(y_all.min() - span * 0.25, y_all.max() + span * 0.25)
    pct_axis(ax)
    ax.set_xlabel("Liczba parametrów (skala logarytmiczna; A0/A1/A2 lekko rozsunięte w poziomie)")
    ax.set_ylabel("Macro-F1 na czystej walidacji")
    model_handles = [Line2D([], [], color=MODEL_COLORS[m], marker="o", ls="none", ms=8,
                            label=f"{MODEL_LABELS[m]} ({int(df.loc[df['model'] == m, 'params'].iloc[0]):,} parametrów)"
                            .replace(",", " ")) for m in MODEL_ORDER if m in set(df["model"])]
    style_handles = [Line2D([], [], color=INK_2, marker=AUG_MARKERS[a], ls="none", ms=7, label=a)
                     for a in ("A0", "A1", "A2") if a in set(df["aug"])]
    style_handles += [Line2D([], [], color=INK_2, marker="o", ls="none", ms=7, label="ImageNet (pełny)"),
                      Line2D([], [], color=INK_2, marker="o", ls="none", ms=7, mfc="white", label="od zera (pusty)")]
    leg = ax.legend(handles=model_handles, loc="upper left", fontsize=8)
    ax.add_artist(leg)
    ax.legend(handles=style_handles, loc="lower right", fontsize=8, ncol=2)
    ax.set_title("H3: rozmiar modelu a macro-F1", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corruption_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Variant x corruption accuracy (mean over the 3 levels), with clean accuracy and mCA as reference columns."""
    cols = ["acc"] + [f"acc_{c}_mean" for c in CORRUPTIONS] + ["mca"]
    names = ["Czyste"] + [CORRUPTION_LABELS_PL[c].replace(" ", "\n", 1) for c in CORRUPTIONS] + ["mCA"]
    names[CORRUPTIONS.index("jpeg") + 1] += "\n(niewidziana)"
    data = df[cols].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 2))
    im = ax.imshow(data, cmap="Blues", vmin=max(0.0, np.floor(data.min() / 10) * 10), vmax=100, aspect="auto")
    lo, hi = im.norm.vmin, im.norm.vmax
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.6
            ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK, fontweight="bold" if j in (0, len(cols) - 1) else None)
    ax.set_xticks(range(len(cols)), names, fontsize=8)
    ax.set_yticks(range(len(df)), df["label"], fontsize=8)
    ax.tick_params(length=0)
    ax.grid(False)
    for x in (0.5, len(cols) - 1.5):
        ax.axvline(x, color="white", lw=4)
    for k in range(len(df) + 1):
        ax.axhline(k - 0.5, color="white", lw=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.tick_top()
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("accuracy [%]")
    ax.set_title("Accuracy na walidacji [%] dla każdego zakłócenia (średnia z 3 poziomów)", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plan_variants(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """The 11 planned variants (E1 + the E2 runs in e2_selection.json); all seed-42 runs before E2."""
    seed42 = df[df["seed"] == 42]
    selection = root / "e2_selection.json"
    if not selection.exists():
        return seed42.reset_index(drop=True)
    e2_runs = json.loads(selection.read_text(encoding="utf-8"))["e2_runs"]
    keep = (seed42["stage"] == "E1") | seed42["run_name"].isin(e2_runs)
    return seed42[keep].reset_index(drop=True)


def summarize(root: Path = RESULTS_DIR, out_dir: Path | None = None) -> pd.DataFrame:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_runs(root)
    save_table(df, out_dir / "summary.csv")
    variants = plan_variants(df, root)
    plot_learning_curves(variants, root, out_dir / "h1_learning_curves.png")
    plot_f1_mca_bars(variants, out_dir / "f1_mca_bars.png")
    plot_robustness_curves(variants, out_dir / "robustness_curves.png")
    plot_params_vs_f1(variants, out_dir / "params_vs_f1.png")
    plot_corruption_heatmap(variants, out_dir / "corruption_heatmap.png")
    print(f"{len(df)} runs ({len(variants)} variants in the plots) -> {out_dir}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[["run_name", "stage", "best_epoch", "acc", "macro_f1", "mca", "relative_drop", "params",
                  "cpu_latency_ms", "train_time_s"]].to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summary table and plots of all runs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: <results-dir>/summary")
    args = parser.parse_args()
    summarize(args.results_dir, args.out)


if __name__ == "__main__":
    main()
