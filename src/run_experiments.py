"""Run the KM2 experiment plan: python -m src.run_experiments --stage e1|e2|e3|e4|all [--smoke]

E1: simple_cnn + resnet18 / mobilenet_v3_small x scratch / ImageNet, all A0 (5 runs).
E2: simple_cnn and the better resnet18 and mobilenet variant of E1 (higher validation
    macro-F1), each with A1 and A2 (6 runs); the choice -> results/e2_selection.json.
E3: make sure all 11 runs have complete clean + corrupted validation metrics.
E4: pick the winner of the 11 runs with the rule fixed in KM1 (docs/KM1_sprawozdanie.md,
    section 5.7) -> results/e4_selection.json, train it with seeds 43 and 44 and
    summarize the 3 seeds -> results/e4_summary.json.

Configs are written to configs/<run>.yaml (an existing config is reused as is).
Finished runs are skipped, so an interrupted stage can simply be started again.
--smoke runs the same plan with 1-epoch smoke runs in results/_smoke/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.corruptions import CORRUPTIONS
from src.data import PROJECT_ROOT
from src.evaluate import evaluate, metrics_complete
from src.train import DEFAULTS, load_config, results_root, save_config, train

CONFIGS_DIR = PROJECT_ROOT / "configs"
STAGES = ("e1", "e2", "e3", "e4")
SHORT_NAMES = {"simple_cnn": "simple_cnn", "resnet18": "resnet18", "mobilenet_v3_small": "mobilenet"}
E1_VARIANTS = [("simple_cnn", False), ("resnet18", False), ("resnet18", True),
               ("mobilenet_v3_small", False), ("mobilenet_v3_small", True)]
E2_AUGS = ("A1", "A2")
BASE_SEED, E4_SEEDS = 42, (43, 44)
F1_MARGIN = 0.01   # rule step 1: drop variants more than 1 p.p. of macro-F1 below the best
MCA_TIE = 0.005    # rule step 3: mCA differences below 0.5 p.p. are a tie
RULE_KM1 = [
    "1. Odrzucamy warianty, których macro-F1 na czystej walidacji jest niższe od najlepszego "
    "o więcej niż 1 p.p.",
    "2. Spośród pozostałych wybieramy wariant z najwyższym mCA.",
    "3. Jeśli różnica mCA jest mniejsza niż 0,5 p.p., wygrywa model mniejszy i szybszy.",
]


def run_name(model: str, pretrained: bool, aug: str, seed: int = BASE_SEED) -> str:
    name = f"{SHORT_NAMES[model]}_{'pretrained' if pretrained else 'scratch'}_{aug}"
    return name if seed == BASE_SEED else f"{name}_s{seed}"


def make_config(model: str, pretrained: bool, aug: str, seed: int = BASE_SEED) -> dict:
    return {"run_name": run_name(model, pretrained, aug, seed), "model": model, "pretrained": pretrained,
            "aug": aug, **{k: v for k, v in DEFAULTS.items() if k not in ("pretrained", "aug")}, "seed": seed}


def pp(x: float) -> str:
    """Fraction -> percentage points with a Polish decimal comma."""
    return f"{x * 100:.2f}".replace(".", ",") + " p.p."


def pct(x: float) -> str:
    return f"{x * 100:.2f}".replace(".", ",") + "%"


def params_pl(n: int) -> str:
    return f"{n:,}".replace(",", " ") + " parametrów"


def ms_pl(x: float) -> str:
    return f"{x:.2f}".replace(".", ",") + " ms"


class Plan:
    """Paths and runs of one plan (normal or smoke)."""

    def __init__(self, smoke: bool = False):
        self.smoke = smoke
        self.root = results_root(smoke)
        self.configs = self.root / "configs" if smoke else CONFIGS_DIR

    def config(self, cfg: dict) -> dict:
        """Reuse configs/<run>.yaml if it exists, otherwise write the generated one."""
        path = self.configs / f"{cfg['run_name']}.yaml"
        if path.exists():
            return load_config(path)
        save_config(cfg, path)
        return cfg

    def run(self, cfg: dict) -> dict:
        """Train (if needed) and evaluate (if needed) one run; returns its metrics."""
        cfg = self.config(cfg)
        out_dir = train(cfg, smoke=self.smoke)
        return evaluate(out_dir, smoke=self.smoke)

    def metrics(self, name: str) -> dict:
        path = self.root / name / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing - run the earlier stage first")
        return json.loads(path.read_text(encoding="utf-8"))

    def run_config(self, name: str) -> dict:
        return json.loads((self.root / name / "config.json").read_text(encoding="utf-8"))

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved {path}")
        return path

    def read_json(self, name: str) -> dict:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing - run the earlier stage first")
        return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# E1 and E2
# ---------------------------------------------------------------------------

def e1_configs() -> list[dict]:
    return [make_config(m, p, "A0") for m, p in E1_VARIANTS]


def stage_e1(plan: Plan) -> None:
    for cfg in e1_configs():
        plan.run(cfg)


def select_e2(plan: Plan) -> dict:
    """Better E1 variant (scratch vs ImageNet) of resnet18 and mobilenet by clean validation macro-F1."""
    selection = {
        "rule": "Dla ResNet18 i MobileNetV3-Small do E2 przechodzi wariant z E1 (od zera albo ImageNet, A0) "
                "o wyższym macro-F1 na czystej walidacji. Przy remisie decyduje wyższe mCA.",
        "split": "val",
    }
    for model in ("resnet18", "mobilenet_v3_small"):
        candidates = {}
        for pretrained in (False, True):
            name = run_name(model, pretrained, "A0")
            m = plan.metrics(name)
            candidates[name] = {"pretrained": pretrained, "val_macro_f1": m["clean"]["macro_f1"],
                                "val_acc": m["clean"]["acc"], "mca": m["mca"], "best_epoch": m["best_epoch"]}
        ranked = sorted(candidates, key=lambda n: (candidates[n]["val_macro_f1"], candidates[n]["mca"]),
                        reverse=True)
        best, other = ranked
        b, o = candidates[best], candidates[other]
        diff = b["val_macro_f1"] - o["val_macro_f1"]
        init = lambda c: "pretrenowany (ImageNet)" if c["pretrained"] else "uczony od zera"  # noqa: E731
        reason = (f"{init(b)}: macro-F1 = {pct(b['val_macro_f1'])}, {init(o)}: {pct(o['val_macro_f1'])}, "
                  f"różnica {pp(diff)} na korzyść wariantu {best}.")
        if diff == 0:
            reason += f" Remis macro-F1, rozstrzyga mCA: {pct(b['mca'])} wobec {pct(o['mca'])}."
        selection[model] = {"candidates": candidates, "selected": best, "pretrained": b["pretrained"],
                            "macro_f1_margin": round(diff, 6), "reason": reason}
    selection["e2_runs"] = [c["run_name"] for c in e2_configs(selection)]
    return selection


def e2_configs(selection: dict) -> list[dict]:
    variants = [("simple_cnn", False)] + [(m, selection[m]["pretrained"]) for m in ("resnet18", "mobilenet_v3_small")]
    return [make_config(m, p, aug) for m, p in variants for aug in E2_AUGS]


def stage_e2(plan: Plan) -> None:
    selection = select_e2(plan)
    plan.write_json("e2_selection.json", selection)
    for model in ("resnet18", "mobilenet_v3_small"):
        print(f"E2 {model}: {selection[model]['reason']}")
    for cfg in e2_configs(selection):
        plan.run(cfg)


def eleven_runs(plan: Plan) -> list[str]:
    """The 5 E1 runs and the 6 E2 runs chosen in e2_selection.json."""
    return [c["run_name"] for c in e1_configs()] + plan.read_json("e2_selection.json")["e2_runs"]


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------

def stage_e3(plan: Plan) -> None:
    names = eleven_runs(plan)
    print(f"{'run':<28} {'acc':>7} {'F1':>7} {'mCA':>7} {'drop':>7}")
    for name in names:
        out_dir = plan.root / name
        path = out_dir / "metrics.json"
        complete = path.exists() and metrics_complete(json.loads(path.read_text(encoding="utf-8")))
        m = plan.metrics(name) if complete else evaluate(out_dir, smoke=plan.smoke, force=True)
        print(f"{name:<28} {m['clean']['acc']:7.4f} {m['clean']['macro_f1']:7.4f} {m['mca']:7.4f} "
              f"{m['relative_drop']:7.4f}")
    print(f"E3: all {len(names)} runs have complete clean and corrupted validation metrics")


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------

def select_e4(plan: Plan) -> dict:
    """Winner of the 11 runs with the KM1 rule (section 5.7)."""
    rows = []
    for name in eleven_runs(plan):
        m, c = plan.metrics(name), plan.run_config(name)
        rows.append({"run_name": name, "model": c["model"], "pretrained": c["pretrained"], "aug": c["aug"],
                     "val_macro_f1": m["clean"]["macro_f1"], "val_acc": m["clean"]["acc"], "mca": m["mca"],
                     "relative_drop": m["relative_drop"], "params": m["params"],
                     "cpu_latency_ms": m["cpu_latency_ms"]})
    by_name = {r["run_name"]: r for r in rows}

    # Step 1: within 1 p.p. of the best clean macro-F1
    best = max(rows, key=lambda r: r["val_macro_f1"])
    threshold = best["val_macro_f1"] - F1_MARGIN
    kept = [r for r in rows if best["val_macro_f1"] - r["val_macro_f1"] <= F1_MARGIN + 1e-9]
    rejected = {r["run_name"]: f"macro-F1 {pct(r['val_macro_f1'])}, o {pp(best['val_macro_f1'] - r['val_macro_f1'])} "
                               f"poniżej najlepszego" for r in rows if r not in kept}
    step1 = {"best_run": best["run_name"], "best_macro_f1": best["val_macro_f1"],
             "threshold_macro_f1": round(threshold, 6), "kept": [r["run_name"] for r in kept], "rejected": rejected}

    # Step 2: highest mCA among the kept variants
    top = max(kept, key=lambda r: r["mca"])
    step2 = {"top_mca_run": top["run_name"], "top_mca": top["mca"],
             "mca_ranking": [{"run_name": r["run_name"], "mca": r["mca"]}
                             for r in sorted(kept, key=lambda r: r["mca"], reverse=True)]}

    # Step 3: mCA tie (< 0.5 p.p.) -> smaller and faster model
    tied = [r for r in kept if top["mca"] - r["mca"] < MCA_TIE - 1e-9]
    step3 = {"tie_threshold_pp": MCA_TIE * 100,
             "tied": [{"run_name": r["run_name"], "mca_gap_pp": round((top["mca"] - r["mca"]) * 100, 3),
                       "params": r["params"], "cpu_latency_ms": r["cpu_latency_ms"]} for r in tied]}
    winner = top
    if len(tied) == 1:
        step3["applied"] = False
        step3["note"] = (f"Żaden inny wariant nie ma mCA bliżej niż 0,5 p.p. od {top['run_name']}, "
                         f"krok 3 nie zmienia wyniku.")
    else:
        # Variants of one architecture have the same size and cost, so for them steps 1-2 decide:
        # keep the highest-mCA variant of each architecture, then compare architectures.
        per_arch = {}
        for r in sorted(tied, key=lambda r: r["mca"], reverse=True):
            per_arch.setdefault(r["model"], r)
        archs = list(per_arch.values())
        smallest = min(archs, key=lambda r: r["params"])
        fastest = min(archs, key=lambda r: r["cpu_latency_ms"])
        winner = smallest
        step3["applied"] = True
        step3["best_variant_per_architecture"] = {k: v["run_name"] for k, v in per_arch.items()}
        step3["smallest"] = smallest["run_name"]
        step3["fastest"] = fastest["run_name"]
        if len(archs) == 1:
            step3["note"] = (f"Remisują tylko warianty jednej architektury ({smallest['model']}), które mają ten sam "
                             f"rozmiar i koszt, więc rozstrzyga mCA: {top['run_name']}.")
        elif smallest is fastest:
            step3["note"] = (f"{smallest['run_name']} jest jednocześnie najmniejszy ({params_pl(smallest['params'])}) "
                             f"i najszybszy ({ms_pl(smallest['cpu_latency_ms'])} na CPU) spośród remisujących.")
        else:
            step3["note"] = (f"UWAGA: reguła niejednoznaczna - najmniejszy jest {smallest['run_name']} "
                             f"({params_pl(smallest['params'])}, {ms_pl(smallest['cpu_latency_ms'])}), "
                             f"a najszybszy {fastest['run_name']} ({params_pl(fastest['params'])}, "
                             f"{ms_pl(fastest['cpu_latency_ms'])}). Przyjęto mniejszy model (liczba parametrów jest "
                             f"pomiarem bez szumu); decyzję warto potwierdzić.")

    justification = (
        f"Krok 1: najlepsze macro-F1 ma {best['run_name']} ({pct(best['val_macro_f1'])}); próg {pct(threshold)}, "
        f"zostaje {len(kept)} z {len(rows)} wariantów. "
        f"Krok 2: najwyższe mCA wśród nich ma {top['run_name']} ({pct(top['mca'])}). "
        f"Krok 3: {step3['note']} "
        f"Zwycięzca: {winner['run_name']} (macro-F1 {pct(winner['val_macro_f1'])}, mCA {pct(winner['mca'])}, "
        f"{params_pl(winner['params'])}, {ms_pl(winner['cpu_latency_ms'])} na CPU)."
    )
    return {
        "rule_km1": RULE_KM1,
        "split": "val",
        "candidates": sorted(rows, key=lambda r: r["val_macro_f1"], reverse=True),
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "winner": winner["run_name"],
        "winner_config": {k: by_name[winner["run_name"]][k] for k in ("model", "pretrained", "aug")},
        "justification": justification,
    }


def summarize_seeds(plan: Plan, names: list[str], seeds: list[int]) -> dict:
    per_seed = {}
    for name, seed in zip(names, seeds):
        m = plan.metrics(name)
        per_seed[name] = {
            "seed": seed,
            "acc": m["clean"]["acc"],
            "macro_f1": m["clean"]["macro_f1"],
            "mca": m["mca"],
            "relative_drop": m["relative_drop"],
            **{f"acc_{c}": round(float(np.mean(list(m["corrupted"][c].values()))), 6) for c in CORRUPTIONS},
            "best_epoch": m["best_epoch"],
            "train_time_s": m["train_time_s"],
            "cpu_latency_ms": m["cpu_latency_ms"],
        }
    keys = [k for k in next(iter(per_seed.values())) if k != "seed"]
    values = {k: np.array([v[k] for v in per_seed.values()], dtype=float) for k in keys}
    return {
        "seeds": seeds,
        "runs": names,
        "std_definition": "sample standard deviation (ddof=1) over the seeds",
        "per_seed": per_seed,
        "mean": {k: round(float(v.mean()), 6) for k, v in values.items()},
        "std": {k: round(float(v.std(ddof=1)), 6) for k, v in values.items()},
    }


def stage_e4(plan: Plan) -> None:
    selection = select_e4(plan)
    plan.write_json("e4_selection.json", selection)
    print(f"E4: {selection['justification']}")
    w = selection["winner_config"]
    names, seeds = [selection["winner"]], [BASE_SEED]
    for seed in E4_SEEDS:
        cfg = make_config(w["model"], w["pretrained"], w["aug"], seed)
        plan.run(cfg)
        names.append(cfg["run_name"])
        seeds.append(seed)
    summary = {"winner": selection["winner"], **summarize_seeds(plan, names, seeds)}
    plan.write_json("e4_summary.json", summary)
    mean, std = summary["mean"], summary["std"]
    print(f"E4 over seeds {seeds}: macro-F1 {mean['macro_f1']:.4f} ± {std['macro_f1']:.4f}, "
          f"mCA {mean['mca']:.4f} ± {std['mca']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KM2 experiment stages E1-E4.")
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--smoke", action="store_true", help="1-epoch smoke runs in results/_smoke/")
    args = parser.parse_args()
    # Polish justifications are printed too; never crash on a console/log code page without them
    sys.stdout.reconfigure(errors="replace")

    plan = Plan(smoke=args.smoke)
    stages = STAGES if args.stage == "all" else (args.stage,)
    t0 = time.perf_counter()
    for stage in stages:
        print(f"===== {stage.upper()} =====")
        {"e1": stage_e1, "e2": stage_e2, "e3": stage_e3, "e4": stage_e4}[stage](plan)
    print(f"finished {', '.join(s.upper() for s in stages)} in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
