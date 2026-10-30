"""Final evaluation on the test set (KM3): python -m src.final_eval [--smoke] [--force]

Evaluates the three KM3 models on the 12 630 test images, clean and under the 15
corruptions, with src.evaluate (the same metrics as on validation), and writes to
results/final_test/:
    <run>/metrics_test.json, per_class_f1_test.csv, confusion_matrix_test.png,
          predictions_test.csv (path, label, pred of the clean test set)
    comparison.csv       validation vs test: acc, macro-F1, mCA, relative drop, and val - test
    corruption_acc.csv   validation vs test accuracy for every corruption and level
    top_confusions.csv   15 most frequent (true -> predicted) pairs, final model and baseline
    weakest_classes.csv  10 classes with the lowest test F1 of the final model
    robustness_test.png  test accuracy vs corruption level, 3 models (validation dashed)

The test set is only reported: nothing computed here feeds back into the choice of a
model, a threshold or any other setting. Existing test metrics are reused unless --force.
--smoke: the 1-epoch smoke models of results/_smoke/ on ~500 test images
-> results/_smoke/final_test/ (no test result of the real models is computed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.class_names_pl import CLASS_NAMES_PL, GROUP_OF, SIGN_CODES  # noqa: E402
from src.corruptions import CORRUPTION_LABELS_PL, CORRUPTIONS, LEVELS, level_label_pl  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.summarize import INK_2, UNSEEN_BG, pct_axis  # noqa: E402
from src.train import results_root  # noqa: E402

FINAL_RUN = "resnet18_pretrained_A2"      # winner of the KM1 selection rule (results/e4_selection.json), seed 42
BASELINE_RUN = "simple_cnn_scratch_A0"    # baseline fixed in KM1
REFERENCE_RUN = "resnet18_pretrained_A0"  # the final network without augmentation: effect of A2 on the test set
KM3_RUNS = (FINAL_RUN, BASELINE_RUN, REFERENCE_RUN)
RUN_ROLES = {FINAL_RUN: "final", BASELINE_RUN: "baseline", REFERENCE_RUN: "reference"}
RUN_LABELS = {
    FINAL_RUN: "ResNet18 · ImageNet · A2 (finalny)",
    BASELINE_RUN: "SimpleCNN · od zera · A0 (baseline)",
    REFERENCE_RUN: "ResNet18 · ImageNet · A0 (bez augmentacji)",
}
# Categorical slots 1-3 of the reference palette (validated all-pairs); ResNet18 keeps its KM2 orange
# and SimpleCNN its KM2 blue, the reference run takes slot 3.
RUN_COLORS = {FINAL_RUN: "#eb6834", BASELINE_RUN: "#2a78d6", REFERENCE_RUN: "#1baf7a"}
RUN_MARKERS = {FINAL_RUN: "^", BASELINE_RUN: "o", REFERENCE_RUN: "s"}
FINAL_TEST_DIR = "final_test"
COMPARED_METRICS = ("acc", "macro_f1", "mca", "relative_drop")
N_TOP_CONFUSIONS, N_WEAKEST = 15, 10


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def check_final_run(root: Path) -> None:
    """FINAL_RUN must be the winner recorded when the KM1 rule was applied on validation."""
    path = root / "e4_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - the final model is chosen in E4 (src.run_experiments)")
    winner = json.loads(path.read_text(encoding="utf-8"))["winner"]
    if winner != FINAL_RUN:
        raise RuntimeError(f"{path} names {winner} as the winner, but FINAL_RUN is {FINAL_RUN}")


def load_val_metrics(root: Path) -> dict[str, dict]:
    metrics = {}
    for run in KM3_RUNS:
        path = root / run / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - evaluate the run on validation first")
        metrics[run] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def headline(m: dict) -> dict[str, float]:
    return {"acc": m["clean"]["acc"], "macro_f1": m["clean"]["macro_f1"], "mca": m["mca"],
            "relative_drop": m["relative_drop"]}


def comparison_table(val: dict, test: dict) -> pd.DataFrame:
    """Long format: one row per (model, metric) with the validation and test value."""
    rows = []
    for run in KM3_RUNS:
        v, t = headline(val[run]), headline(test[run])
        for metric in COMPARED_METRICS:
            rows.append({"run": run, "role": RUN_ROLES[run], "model": RUN_LABELS[run], "metric": metric,
                         "val": v[metric], "test": t[metric], "val_minus_test": v[metric] - t[metric],
                         "n_val": val[run]["n_images"], "n_test": test[run]["n_images"]})
    return pd.DataFrame(rows).round(6)


def corruption_table(val: dict, test: dict) -> pd.DataFrame:
    """Accuracy on validation and test for the clean data and each of the 15 corruptions."""
    rows = []
    for run in KM3_RUNS:
        rows.append({"run": run, "role": RUN_ROLES[run], "corruption": "clean", "level": 0, "param": "",
                     "val": val[run]["clean"]["acc"], "test": test[run]["clean"]["acc"]})
        for ctype in CORRUPTIONS:
            for lv in LEVELS:
                rows.append({"run": run, "role": RUN_ROLES[run], "corruption": ctype, "level": lv,
                             "param": level_label_pl(ctype, lv), "val": val[run]["corrupted"][ctype][str(lv)],
                             "test": test[run]["corrupted"][ctype][str(lv)]})
    df = pd.DataFrame(rows)
    df["val_minus_test"] = df["val"] - df["test"]
    return df.round(6)


def load_predictions(out: Path, run: str) -> pd.DataFrame:
    return pd.read_csv(out / run / "predictions_test.csv")


def top_confusions(preds: pd.DataFrame, run: str, n: int = N_TOP_CONFUSIONS) -> pd.DataFrame:
    """The n most frequent (true -> predicted) pairs among the errors on the clean test set."""
    support = preds["label"].value_counts()
    errors = preds[preds["label"] != preds["pred"]]
    pairs = (errors.groupby(["label", "pred"]).size().reset_index(name="count")
             .sort_values(["count", "label", "pred"], ascending=[False, True, True]).head(n))
    return pd.DataFrame({
        "run": run,
        "role": RUN_ROLES[run],
        "rank": range(1, len(pairs) + 1),
        "true_id": pairs["label"].to_numpy(),
        "true_code": [SIGN_CODES[c] for c in pairs["label"]],
        "true_name": [CLASS_NAMES_PL[c] for c in pairs["label"]],
        "pred_id": pairs["pred"].to_numpy(),
        "pred_code": [SIGN_CODES[c] for c in pairs["pred"]],
        "pred_name": [CLASS_NAMES_PL[c] for c in pairs["pred"]],
        "count": pairs["count"].to_numpy(),
        "share_of_errors": (pairs["count"] / max(len(errors), 1)).round(5).to_numpy(),
        "share_of_true_class": (pairs["count"] / pairs["label"].map(support)).round(5).to_numpy(),
        "errors_total": len(errors),
    })


def weakest_classes(per_class: pd.DataFrame, preds: pd.DataFrame, n: int = N_WEAKEST) -> pd.DataFrame:
    """The n classes with the lowest F1, with their most frequent wrong prediction."""
    weak = per_class.sort_values(["f1", "support"], ascending=[True, False], kind="stable").head(n)
    errors = preds[preds["label"] != preds["pred"]]
    rows = []
    for rank, (_, r) in enumerate(weak.iterrows(), start=1):
        c = int(r["class_id"])
        wrong_as = errors.loc[errors["label"] == c, "pred"].value_counts()
        top = int(wrong_as.index[0]) if len(wrong_as) else None
        rows.append({
            "rank": rank, "class_id": c, "code": SIGN_CODES[c], "class_name": CLASS_NAMES_PL[c],
            "group": GROUP_OF[c], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "support": int(r["support"]), "missed": int(wrong_as.sum()),
            "false_alarms": int((errors["pred"] == c).sum()),
            "most_confused_with_id": top,
            "most_confused_with": CLASS_NAMES_PL[top] if top is not None else "",
            "most_confused_count": int(wrong_as.iloc[0]) if len(wrong_as) else 0,
        })
    return pd.DataFrame(rows)


def plot_robustness_test(val: dict, test: dict, path: Path) -> None:
    """Accuracy vs corruption level on the test set, 3 models; validation as thin dashed lines."""
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(3.0 * len(CORRUPTIONS), 4.1), sharey=True)
    x = [0, *LEVELS]
    for ax, ctype in zip(axes, CORRUPTIONS):
        if ctype == "jpeg":
            ax.set_facecolor(UNSEEN_BG)
        for run in KM3_RUNS:
            for m, is_test in ((val[run], False), (test[run], True)):
                y = [m["clean"]["acc"] * 100] + [m["corrupted"][ctype][str(lv)] * 100 for lv in LEVELS]
                if is_test:
                    ax.plot(x, y, color=RUN_COLORS[run], marker=RUN_MARKERS[run], ms=5.5, lw=2, zorder=3)
                else:
                    ax.plot(x, y, color=RUN_COLORS[run], lw=1.1, ls=(0, (3, 2)), alpha=0.75, zorder=2)
        ax.set_xticks(x, ["czyste"] + [f"{lv}\n{level_label_pl(ctype, lv)}" for lv in LEVELS], fontsize=7)
        ax.set_xlabel("poziom zakłócenia")
        pct_axis(ax)
        if ctype == "jpeg":
            ax.set_title(f"{CORRUPTION_LABELS_PL[ctype]}\nNIEWIDZIANA W TRENINGU", fontweight="bold")
        else:
            ax.set_title(f"{CORRUPTION_LABELS_PL[ctype]}\n(w augmentacji A2)")
    axes[0].set_ylabel("accuracy")
    handles = [Line2D([], [], color=RUN_COLORS[r], marker=RUN_MARKERS[r], lw=2, label=RUN_LABELS[r])
               for r in KM3_RUNS]
    handles += [Line2D([], [], color=INK_2, lw=2, label="zbiór testowy"),
                Line2D([], [], color=INK_2, lw=1.1, ls=(0, (3, 2)), label="walidacja (KM2)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    n_test = next(iter(test.values()))["n_images"]
    fig.suptitle(f"Accuracy na zbiorze testowym (n = {fmt_int(n_test)}) w zależności od poziomu zakłócenia "
                 "(linie przerywane = walidacja, szare tło = zakłócenie nieobecne w augmentacji)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def final_eval(smoke: bool = False, force: bool = False) -> dict[str, dict]:
    root = results_root(smoke)
    if not smoke:
        check_final_run(root)
    out = root / FINAL_TEST_DIR
    out.mkdir(parents=True, exist_ok=True)
    val = load_val_metrics(root)
    t0 = time.perf_counter()

    test = {run: evaluate(root / run, split="test", smoke=smoke, force=force, final=True, dest_dir=out / run,
                          save_predictions=True)
            for run in KM3_RUNS}

    comparison = comparison_table(val, test)
    comparison.to_csv(out / "comparison.csv", index=False)
    corruption_table(val, test).to_csv(out / "corruption_acc.csv", index=False)
    preds = {run: load_predictions(out, run) for run in KM3_RUNS}
    pd.concat([top_confusions(preds[run], run) for run in (FINAL_RUN, BASELINE_RUN)]).to_csv(
        out / "top_confusions.csv", index=False)
    per_class = pd.read_csv(out / FINAL_RUN / "per_class_f1_test.csv")
    weakest_classes(per_class, preds[FINAL_RUN]).to_csv(out / "weakest_classes.csv", index=False)
    plot_robustness_test(val, test, out / "robustness_test.png")

    print(f"\nvalidation vs test ({time.perf_counter() - t0:.0f} s) -> {out}")
    table = comparison.pivot(index="run", columns="metric", values=["val", "test"]).reindex(KM3_RUNS)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(table)
    return test


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the KM3 models once on the clean and corrupted test set.")
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch smoke models from results/_smoke/ on ~500 test images -> results/_smoke/")
    parser.add_argument("--force", action="store_true", help="recompute test metrics that already exist")
    args = parser.parse_args()
    final_eval(smoke=args.smoke, force=args.force)


if __name__ == "__main__":
    main()
