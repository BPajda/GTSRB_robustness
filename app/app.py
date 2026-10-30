"""Gradio app (KM3): recognize a traffic sign, test the model's robustness to corruptions, browse the results.

    python -m app.app [--device auto|cpu|cuda] [--port 7860] [--share]

Model: the final model results/resnet18_pretrained_A2/best.pt, on the GPU when available (--device auto),
otherwise on the CPU. Inference runs in fp32 with TF32 off - the numerics in which the rejection threshold
in results/xai/threshold.json was chosen (on the validation set). As in training, an image is resized to
64x64 without keeping the aspect ratio, so the model expects a crop with the sign in the middle (like the
GTSRB images), not a whole street scene.

Tabs:
    Rozpoznaj         an uploaded (optionally cropped) photo or a random test image: prediction, top-5,
                      Grad-CAM, a warning below the threshold, the true class of a test image
    Test odporności   the same image under one of the 5 corruptions of src.corruptions, level 0-3
    Wyniki            tables and plots from results/summary, results/final_test and results/xai
    O projekcie       short description, repository link, limitations
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import v2

from src.class_names_pl import CLASS_NAMES_PL, SIGN_CODES, category_pl, class_label
from src.corruptions import CORRUPTION_LABELS_PL, CORRUPTIONS, apply_corruption, corruption_rng, level_label_pl, \
    roi_to_pixels
from src.data import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SEED, GTSRBDataset, load_csv
from src.evaluate import load_model
from src.final_eval import BASELINE_RUN, FINAL_RUN, FINAL_TEST_DIR, KM3_RUNS, RUN_LABELS, fmt_int
from src.summarize import AXIS, INK, INK_2, MODEL_LABELS, UNSEEN_BG, pct_axis
from src.train import RESULTS_DIR, precision_for
from src.xai import ROI_COLOR, XAI_DIR, as_hwc, fmt, fmt_pct, gradcam, make_cam, overlay

FINAL_DIR = RESULTS_DIR / FINAL_RUN
SUMMARY_DIR = RESULTS_DIR / "summary"
TEST_RESULTS_DIR = RESULTS_DIR / FINAL_TEST_DIR
XAI_RESULTS_DIR = RESULTS_DIR / XAI_DIR
THRESHOLD_PATH = XAI_RESULTS_DIR / "threshold.json"
REPO_URL = "https://github.com/BPajda/GTSRB_robustness"

TOP_K = 5
APP_LEVELS = (0, 1, 2, 3)        # 0 = no corruption
VIEW_SIZE = 256                  # 64x64 images are shown 4x enlarged, pixel for pixel
UPLOAD_ROI_MARGIN = 0.1          # an uploaded crop is assumed to have the sign in its middle 80% (like GTSRB)
UPLOAD_RNG_INDEX = 0             # corruption_rng index of an uploaded image; test images use their own index
# Categorical slots 1-2 of the project palette: true (or reference) class and predicted class.
TRUE_COLOR, PRED_COLOR = "#2a78d6", "#eb6834"
METRIC_LABELS = {"acc": "Accuracy", "macro_f1": "Macro-F1", "mca": "mCA", "relative_drop": "Wzgl. spadek"}

RESIZE = v2.Compose([v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])  # as GTSRBDataset
TO_FLOAT = v2.ToDtype(torch.float32, scale=True)
NORMALIZE = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)


# ---------------------------------------------------------------------------
# Model and images
# ---------------------------------------------------------------------------

def softmax(logits: torch.Tensor) -> np.ndarray:
    """Softmax in float64, as in src.xai (the threshold was chosen on these values)."""
    return torch.softmax(logits.double(), dim=1).cpu().numpy()


class Recognizer:
    """The final model and its rejection threshold. Calls are serialized: Grad-CAM hooks are not thread-safe."""

    def __init__(self, device: torch.device):
        for path, hint in ((FINAL_DIR / "best.pt", f"python -m src.train --config configs/{FINAL_RUN}.yaml"),
                           (THRESHOLD_PATH, "python -m src.xai")):
            if not path.exists():
                raise SystemExit(f"{path} not found - create it with: {hint}")
        torch.backends.cudnn.allow_tf32 = False  # true fp32 also on the GPU
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model, _ = load_model(FINAL_DIR, device)
        self.device, self.memory_format = device, precision_for(device)[1]
        self.threshold = float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])
        self.lock = threading.Lock()

    def _batch(self, images: torch.Tensor) -> torch.Tensor:
        return NORMALIZE(images).to(self.device).contiguous(memory_format=self.memory_format)

    def probs(self, images: torch.Tensor) -> np.ndarray:
        """Softmax [B, 43] of float images [B, 3, 64, 64] in [0, 1]."""
        with self.lock, torch.inference_mode():
            logits = self.model(self._batch(images)).float()
        return softmax(logits)

    def explain(self, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Softmax [43] and the Grad-CAM map [64, 64] of the predicted class for one float image in [0, 1]."""
        with self.lock, make_cam(self.model) as cam:
            maps, logits = gradcam(cam, self._batch(image[None]))
        return softmax(logits)[0], maps[0]


@dataclass(frozen=True)
class Sample:
    """The image both tabs work on: a test image or an uploaded crop."""
    image: torch.Tensor   # uint8 [3, 64, 64], resized like the A0 pipeline
    roi: np.ndarray       # sign box (x1, y1, x2, y2) in the 64x64 image, used by occlusion
    rng_index: int        # image index for corruption_rng
    label: int | None     # true class (test images only)
    source: str

    @property
    def clean(self) -> torch.Tensor:
        return TO_FLOAT(self.image)

    def corrupted(self, ctype: str, level: int) -> torch.Tensor:
        """Float image in [0, 1], level 0 = clean. For a test image it is exactly the image of the evaluation
        (CorruptedDataset over the test set uses the same seed, index and ROI)."""
        if level == 0:
            return self.clean
        rng = corruption_rng(SEED, self.rng_index, ctype, level)
        return apply_corruption(self.clean, ctype, level, roi=self.roi, rng=rng)


class TestImages:
    """Random test images with labels and ROIs, read from disk one at a time."""

    def __init__(self):
        self.available = (DATA_DIR / "Test.csv").exists()
        if self.available:
            self.df = load_csv("test")
            self.ds = GTSRBDataset(self.df)
        self.rng = np.random.default_rng()  # unseeded on purpose: every click shows another image

    def random(self) -> Sample:
        i = int(self.rng.integers(len(self.ds)))
        row = self.df.iloc[i]
        return Sample(self.ds.image_uint8(i), self.ds.rois[i], i, int(self.ds.labels[i]),
                      f"zdjęcie testowe {row['Path']} (oryginał {row['Width']}×{row['Height']} px)")


def upload_sample(img: Image.Image) -> Sample:
    img = ImageOps.exif_transpose(img).convert("RGB")
    m = UPLOAD_ROI_MARGIN * IMG_SIZE
    roi = np.array([m, m, IMG_SIZE - m, IMG_SIZE - m], dtype=np.float32)
    return Sample(RESIZE(img).as_subclass(torch.Tensor), roi, UPLOAD_RNG_INDEX, None,
                  f"wgrane zdjęcie ({img.width}×{img.height} px)")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view(img: np.ndarray, roi=None) -> Image.Image:
    """HxWx3 float image in [0, 1] enlarged to VIEW_SIZE (nearest neighbour), optionally with the ROI box."""
    out = Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8))
    out = out.resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.NEAREST)
    if roi is not None:
        s = VIEW_SIZE // IMG_SIZE
        x1, y1, x2, y2 = roi_to_pixels(roi, IMG_SIZE, IMG_SIZE)
        ImageDraw.Draw(out).rectangle((x1 * s, y1 * s, x2 * s - 1, y2 * s - 1), outline=ROI_COLOR, width=2)
    return out


def cam_view(img: torch.Tensor, cam: np.ndarray, sample: Sample) -> Image.Image:
    """Grad-CAM over the image; the ROI box only for test images (for uploads it is a guess)."""
    return view(overlay(as_hwc(img), cam), sample.roi if sample.label is not None else None)


@functools.lru_cache(maxsize=None)
def pictogram_uri(c: int) -> str:
    """data/Meta/<c>.png (transparent background) as a data URI, empty if the file is missing."""
    path = DATA_DIR / "Meta" / f"{c}.png"
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else ""


def pictogram_html(c: int, css_class: str = "pict") -> str:
    uri = pictogram_uri(c)
    return f'<img class="{css_class}" src="{uri}" alt="wzorcowy piktogram klasy {c}">' if uri else ""


def result_html(probs: np.ndarray, threshold: float, sample: Sample, condition: str = "") -> str:
    pred = int(probs.argmax())
    conf = float(probs[pred])
    if conf >= threshold:
        status = (f'<div class="status ok">✔ Model jest pewny: pewność {fmt_pct(conf, 2)} ≥ próg '
                  f'{fmt_pct(threshold, 2)}.</div>')
    else:
        status = (f'<div class="status warn"><strong>⚠ Model nie jest pewny</strong>'
                  f'Pewność {fmt_pct(conf, 2)} jest poniżej progu {fmt_pct(threshold, 2)}, wyznaczonego na '
                  'walidacji. Wynik może być błędny – w działającym systemie ta predykcja zostałaby '
                  'odrzucona.</div>')
    truth = ""
    if sample.label is not None:
        c, hit = sample.label, sample.label == pred
        truth = (f'<div class="truth {"hit" if hit else "miss"}">'
                 f'<span class="verdict">{"✅ Trafienie" if hit else "❌ Pomyłka"}</span>'
                 f'{pictogram_html(c, "pict-small")}'
                 f'<span>prawdziwa klasa: <strong>{SIGN_CODES[c]}</strong> {CLASS_NAMES_PL[c]}</span></div>')
    source = html.escape(sample.source + (f" · {condition}" if condition else ""))
    return (f'<div class="sign-card">{pictogram_html(pred)}<div>'
            f'<div class="sign-meta">{SIGN_CODES[pred]} · znak {category_pl(pred)} · klasa GTSRB nr {pred}</div>'
            f'<div class="sign-name">{CLASS_NAMES_PL[pred]}</div>'
            f'<div class="sign-conf">pewność modelu: <strong>{fmt_pct(conf, 2)}</strong></div></div></div>'
            f'{status}{truth}<div class="source">Źródło: {source}</div>')


def image_view(label: str) -> gr.Image:
    """Output image component for the enlarged 64x64 views (PNG: the default WebP is lossy)."""
    return gr.Image(label=label, type="pil", format="png", interactive=False, height=VIEW_SIZE,
                    buttons=["download"])


def empty_html(text: str) -> str:
    return f'<div class="empty">{text}</div>'


def pct_label(p: float) -> str:
    return "< 0,1%" if p < 0.001 else fmt_pct(p)


def top5_figure(probs: np.ndarray, threshold: float, label: int | None) -> Figure:
    """Top-5 classes; the prediction highlighted, the rest gray, the true class (if known) named in its label."""
    top = np.argsort(probs)[::-1][:TOP_K]
    y = np.arange(TOP_K)
    fig = Figure(figsize=(6.4, 2.7), dpi=120, layout="constrained")
    ax = fig.subplots()
    ax.barh(y, probs[top] * 100, height=0.62, color=[TRUE_COLOR] + [AXIS] * (TOP_K - 1))
    for yi, c in zip(y, top):
        ax.text(probs[c] * 100 + 1.5, yi, pct_label(probs[c]), va="center", fontsize=8.5, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=3)
    names = [class_label(int(c), short=True, code=True) + ("  ✓ prawdziwa" if c == label else "") for c in top]
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.set_ylim(TOP_K - 0.5, -1.15)
    ax.axvline(threshold * 100, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(threshold * 100 - 1, -0.8, f"próg {fmt_pct(threshold)}", ha="right", va="center", fontsize=8,
            color=INK_2)
    ax.set_xlim(0, 113)
    ax.set_xticks([0, 25, 50, 75, 100])
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("prawdopodobieństwo (softmax)")
    ax.set_title("Top-5 klas", loc="left", fontweight="bold")
    return fig


def robustness_figure(probs: np.ndarray, ctype: str, level: int, threshold: float, ref: int,
                      ref_is_true: bool) -> Figure:
    """probs [4, 43]: softmax at levels 0-3. Probability of the reference class (true class, or the class
    predicted without corruption) and the confidence of the predicted class; hollow = another class."""
    x = np.array(APP_LEVELS)
    pred, conf, p_ref = probs.argmax(axis=1), probs.max(axis=1) * 100, probs[:, ref] * 100
    other = pred != ref
    fig = Figure(figsize=(6.4, 3.6), dpi=120, layout="constrained")
    ax = fig.subplots()
    ax.axvspan(level - 0.3, level + 0.3, color=UNSEEN_BG, lw=0, zorder=0)
    ax.axhline(threshold * 100, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(x, conf, color=PRED_COLOR, lw=2, marker="s", ms=9, zorder=2)
    ax.plot(x, p_ref, color=TRUE_COLOR, lw=2, marker="o", ms=5.5, zorder=3)
    ax.scatter(x[other], conf[other], marker="s", s=81, facecolor="white", edgecolor=PRED_COLOR, lw=2, zorder=4)
    for i in np.flatnonzero(other):
        below = conf[i] > 75
        ax.annotate(class_label(int(pred[i]), short=True, width=20), (x[i], conf[i]),
                    xytext=(0, -9 if below else 9), textcoords="offset points", ha="center",
                    va="top" if below else "bottom", fontsize=7.5, color=INK_2)
    ref_name = class_label(ref, short=True)
    ref_role = "prawdziwa klasa" if ref_is_true else "klasa rozpoznana bez zakłóceń"
    handles = [Line2D([], [], color=TRUE_COLOR, lw=2, marker="o", ms=5.5, label=f"{ref_role}: {ref_name}"),
               Line2D([], [], color=PRED_COLOR, lw=2, marker="s", ms=8, label="przewidziana klasa (pewność modelu)"),
               Line2D([], [], color=INK_2, lw=1, ls=(0, (4, 3)), label=f"próg pewności {fmt_pct(threshold)}")]
    if other.any():
        handles.insert(2, Line2D([], [], color=PRED_COLOR, lw=0, marker="s", ms=8, markerfacecolor="white",
                                 markeredgewidth=2, label=f"przewidziana klasa inna niż {ref_name} (podpisana)"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=1)
    ax.set_xticks(x, [f"{lv}\n{'bez zakłóceń' if lv == 0 else level_label_pl(ctype, lv)}" for lv in APP_LEVELS])
    ax.set_xlim(-0.5, len(APP_LEVELS) - 0.5)
    name = CORRUPTION_LABELS_PL[ctype]
    ax.set_xlabel(f"poziom zakłócenia: {name[0].lower() + name[1:]}")
    ax.set_ylim(0, 105)
    pct_axis(ax)
    ax.set_ylabel("prawdopodobieństwo (softmax)")
    fig.suptitle("Pewność w funkcji poziomu zakłócenia", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------

def variant_name(row) -> str:
    name = f"{MODEL_LABELS[row.model]} · {'ImageNet' if row.pretrained else 'od zera'} · {row.aug}"
    return name + (" ★" if row.run_name == FINAL_RUN else "")


def summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Wariant": [variant_name(r) for r in summary.itertuples()],
        "Etap": summary["stage"],
        "Seed": summary["seed"],
        "Accuracy": summary["acc"].map(lambda v: fmt_pct(v, 2)),
        "Macro-F1": summary["macro_f1"].map(lambda v: fmt_pct(v, 2)),
        "mCA": summary["mca"].map(lambda v: fmt_pct(v, 2)),
        "Wzgl. spadek": summary["relative_drop"].map(lambda v: fmt_pct(v, 1)),
        "Parametry": (summary["params"] / 1e6).map(lambda v: f"{fmt(v)} mln"),
        "CPU [ms]": summary["cpu_latency_ms"].map(lambda v: fmt(v, 1)),
        "Najl. epoka": [f"{b} z {e}" for b, e in zip(summary["best_epoch"], summary["epochs_run"])],
    })


def comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """One row per model: every metric on validation and on the test set."""
    rows = []
    for _run, g in comparison.groupby("run", sort=False):
        row = {"Model": g["model"].iloc[0]}
        for r in g.itertuples():
            row[f"{METRIC_LABELS[r.metric]}: walidacja"] = fmt_pct(r.val, 2)
            row[f"{METRIC_LABELS[r.metric]}: test"] = fmt_pct(r.test, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_block(path: Path, title: str, caption: str) -> None:
    gr.Markdown(f"#### {title}\n{caption}")
    gr.Image(str(path), show_label=False, interactive=False, buttons=["fullscreen", "download"])


def results_tab() -> None:
    summary = pd.read_csv(SUMMARY_DIR / "summary.csv")
    comparison = pd.read_csv(TEST_RESULTS_DIR / "comparison.csv")
    thr = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    conf = json.loads((XAI_RESULTS_DIR / "confidence.json").read_text(encoding="utf-8"))["models"][FINAL_RUN]
    roi = pd.read_csv(XAI_RESULTS_DIR / "roi_energy.csv")
    roi = roi[(roi["run"] == FINAL_RUN) & (roi["condition"] == "clean") & (roi["subset"] == "all")].iloc[0]
    mca = summary.set_index("run_name")["mca"]
    test = comparison.set_index(["run", "metric"])

    with gr.Tabs():
        with gr.Tab("Porównanie wariantów (walidacja)"):
            gr.Markdown("Wszystkie wytrenowane warianty, oceniane na walidacji (serie zdjęć wydzielone z "
                        "`Train/`). mCA – średnia accuracy po 5 zakłóceniach × 3 poziomach, względny spadek = "
                        "(accuracy − mCA) / accuracy, CPU – mediana czasu predykcji jednego zdjęcia. "
                        "★ – model finalny, używany w aplikacji.")
            gr.Dataframe(summary_table(summary), interactive=False, wrap=True, show_row_numbers=False,
                         max_height=800)
            plot_block(SUMMARY_DIR / "f1_mca_bars.png", "Skuteczność i odporność wszystkich wariantów",
                       "Macro-F1 na czystej walidacji i mCA. Na czystych danych różnice są niewielkie, "
                       "w odporności duże: ResNet18 z wagami ImageNet i augmentacją A2 ma mCA "
                       f"{fmt_pct(mca[FINAL_RUN])}, baseline SimpleCNN bez augmentacji – "
                       f"{fmt_pct(mca[BASELINE_RUN])}.")
            plot_block(SUMMARY_DIR / "robustness_curves.png", "Accuracy a poziom zakłócenia (H2, H4)",
                       "Augmentacje A0, A1 i A2 w każdej architekturze. A2 najmocniej pomaga przy szumie i "
                       "zasłonięciu; przy kompresji JPEG, której nie było w treningu, zysk jest znacznie mniejszy.")
            plot_block(SUMMARY_DIR / "corruption_heatmap.png", "Accuracy dla każdego zakłócenia",
                       "Średnia z 3 poziomów. Najtrudniejsze jest zasłonięcie części znaku, "
                       "najmniej szkodzi przyciemnienie.")
            plot_block(SUMMARY_DIR / "h1_learning_curves.png", "Uczenie od zera a wagi ImageNet (H1)",
                       "Macro-F1 walidacji po kolejnych epokach, bez augmentacji. Sieci z wagami ImageNet "
                       "osiągają wyższy wynik w mniejszej liczbie epok (★ – epoka, z której zapisano wagi).")
            plot_block(SUMMARY_DIR / "params_vs_f1.png", "Rozmiar modelu a macro-F1 (H3)",
                       "Liczba parametrów (skala logarytmiczna) a macro-F1. ResNet18 jest najdokładniejszy, "
                       "ale ma 7 razy więcej parametrów niż MobileNetV3-Small i jest kilka razy wolniejszy na CPU.")

        with gr.Tab("Zbiór testowy"):
            gr.Markdown(f"Ocena na zbiorze testowym ({fmt_int(int(comparison['n_test'].iloc[0]))} zdjęć, użyty "
                        "jeden raz, po wyborze modelu na walidacji): model finalny, baseline i ten sam ResNet18 "
                        "bez augmentacji (A0), który pokazuje efekt augmentacji A2.")
            gr.Dataframe(comparison_table(comparison), interactive=False, wrap=True, show_row_numbers=False)
            plot_block(TEST_RESULTS_DIR / "robustness_test.png", "Accuracy na teście a poziom zakłócenia",
                       "Linie ciągłe – zbiór testowy, przerywane – walidacja. Wyniki testu są bliskie "
                       f"walidacji (mCA modelu finalnego: {fmt_pct(test.loc[(FINAL_RUN, 'mca'), 'val'])} na "
                       f"walidacji, {fmt_pct(test.loc[(FINAL_RUN, 'mca'), 'test'])} na teście).")
            with gr.Accordion("Macierze pomyłek (czysty zbiór testowy)", open=False):
                for run in KM3_RUNS:
                    plot_block(TEST_RESULTS_DIR / run / "confusion_matrix_test.png", RUN_LABELS[run], "")

        with gr.Tab("Grad-CAM i pewność"):
            plot_block(XAI_RESULTS_DIR / "gradcam_examples.png", "Na co patrzy model?",
                       "Grad-CAM dla poprawnie rozpoznanych zdjęć testowych, po jednym z każdej grupy znaków: "
                       "model finalny i baseline. Jasne obszary najmocniej wpłynęły na decyzję, turkusowa ramka "
                       "to oznaczony obszar znaku (ROI).")
            plot_block(XAI_RESULTS_DIR / "roi_energy.png", "Jaka część uwagi przypada na znak",
                       "Udział mapy Grad-CAM wewnątrz ROI na czystym teście. U modelu finalnego mediana wynosi "
                       f"{fmt_pct(roi['energy_in_roi_median'], 0)}, choć ROI zajmuje średnio "
                       f"{fmt_pct(roi['roi_area_mean'], 0)} obrazu – model patrzy na znak, nie na tło.")
            plot_block(XAI_RESULTS_DIR / "corruption_shift.png", "Uwaga modelu przy zakłóceniach",
                       "Cztery zdjęcia testowe bez zakłóceń i z każdym z 5 zakłóceń na poziomach 2 i 3 "
                       "(model finalny).")
            plot_block(XAI_RESULTS_DIR / "error_gallery.png", "Najbardziej pewne błędy",
                       "Błędy modelu finalnego z najwyższą pewnością, pogrupowane według par klas "
                       "(prawdziwa → przewidziana).")
            plot_block(XAI_RESULTS_DIR / "confidence.png", "Pewność i kalibracja",
                       "Rozkład pewności (max softmax) dla predykcji poprawnych i błędnych oraz diagram "
                       f"niezawodności. Model finalny jest dobrze skalibrowany (ECE {fmt(conf['clean']['ece'], 4)} "
                       "na czystym teście), a jego błędy mają zwykle niską pewność (mediana "
                       f"{fmt_pct(conf['clean']['median_conf_wrong'], 0)}).")
            clean, corrupted = thr["test"]["clean"], thr["test"]["corrupted_all"]
            plot_block(XAI_RESULTS_DIR / "threshold.png", "Próg niepewności używany w aplikacji",
                       "Najniższy próg, przy którym zachowane predykcje na zakłóconej walidacji mają co najmniej "
                       f"{fmt_pct(thr['target_accuracy'])} accuracy: {fmt_pct(thr['threshold'], 2)}. Na czystym "
                       f"teście odrzuca {fmt_pct(1 - clean['coverage'])} zdjęć, a z zachowanych "
                       f"{fmt_pct(clean['accuracy_kept'], 2)} jest poprawnych; na zakłóconym teście zachowuje "
                       f"{fmt_pct(corrupted['coverage'])} predykcji z accuracy "
                       f"{fmt_pct(corrupted['accuracy_kept'], 2)}.")


def about_tab() -> None:
    test = pd.read_csv(TEST_RESULTS_DIR / "comparison.csv").set_index(["run", "metric"])["test"]
    gr.Markdown(f"""
### O projekcie

Projekt bada, czy uczenie transferowe i odpowiednia augmentacja ograniczają spadek skuteczności klasyfikatora
znaków drogowych na zakłóconych zdjęciach: rozmytych, zaszumionych, przyciemnionych, częściowo zasłoniętych
i silnie skompresowanych (JPEG). Porównano własną sieć SimpleCNN z ResNet18 i MobileNetV3-Small, uczonymi od zera
i z wagami ImageNet, oraz trzy warianty augmentacji na zbiorze GTSRB (43 klasy niemieckich znaków).
Model finalny – ResNet18 z wagami ImageNet i augmentacją odpornościową A2 – osiąga na zbiorze testowym
{fmt_pct(test[(FINAL_RUN, 'acc')])} accuracy i {fmt_pct(test[(FINAL_RUN, 'mca')])} średnio na zakłóconych zdjęciach
(baseline SimpleCNN: {fmt_pct(test[(BASELINE_RUN, 'acc')])} i {fmt_pct(test[(BASELINE_RUN, 'mca')])}).
Grad-CAM pokazuje, które piksele zdecydowały o wyniku, a próg pewności wyznaczony na walidacji oznacza
predykcje, którym nie należy ufać.

**Repozytorium:** [{REPO_URL.removeprefix('https://')}]({REPO_URL})

### Ograniczenia

- **Znaki niemieckie.** Model zna tylko 43 klasy ze zbioru GTSRB. Polskie nazwy to odpowiedniki o tym samym
  znaczeniu, ale polskie znaki mogą wyglądać nieco inaczej, a znaku spoza tych 43 klas model nie rozpozna –
  zawsze wskaże którąś z nich (zwykle z niską pewnością).
- **Wycinki zamiast detekcji.** Model klasyfikuje wycinek 64×64 px ze znakiem na środku; nie szuka znaku na
  zdjęciu całej ulicy. Takie zdjęcie trzeba najpierw przyciąć do znaku.
- **Zdjęcia, nie rysunki.** Model uczył się na zdjęciach z kamery samochodowej; na czystych grafikach znaków
  (np. wzorcowych piktogramach z internetu) myli się wyraźnie częściej.
- **Model nie zastępuje kierowcy.** To projekt badawczy: wyniki zmierzono na jednym zbiorze danych, w kontrolowanych
  warunkach. Aplikacja służy do demonstracji, a nie do podejmowania decyzji na drodze.
""")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

CSS = """
.sign-card {display: flex; gap: 16px; align-items: center;}
.sign-card .pict {width: 96px; height: 96px; object-fit: contain; flex: none;}
.sign-meta {font-size: 0.85rem; color: var(--body-text-color-subdued);}
.sign-name {font-size: 1.35rem; font-weight: 600; line-height: 1.25; margin: 2px 0 4px;}
.status {margin-top: 12px; padding: 10px 14px; border-radius: 8px; border-left: 5px solid;}
.status strong {display: block; font-size: 1.1rem; margin-bottom: 2px;}
.status.ok {border-color: #1baf7a; background: rgba(27, 175, 122, 0.10);}
.status.warn {border-color: #d03b3b; background: rgba(208, 59, 59, 0.13);}
.truth {display: flex; gap: 10px; align-items: center; margin-top: 10px; padding: 8px 14px; border-radius: 8px;
        background: var(--background-fill-secondary);}
.truth .verdict {font-weight: 600; white-space: nowrap;}
.truth .pict-small {width: 40px; height: 40px; object-fit: contain; flex: none;}
.source {margin-top: 8px; font-size: 0.8rem; color: var(--body-text-color-subdued);}
.empty {padding: 24px; text-align: center; color: var(--body-text-color-subdued);}
.hint {font-size: 0.9rem;}
.note {font-size: 0.8rem; color: var(--body-text-color-subdued);}
"""
CAM_NOTE = ("Grad-CAM: jasne obszary najmocniej wpłynęły na decyzję modelu. Turkusowa ramka (tylko zdjęcia "
            "testowe) – oznaczony w zbiorze obszar znaku (ROI).")


def build_app(rec: Recognizer, tests: TestImages, device_label: str) -> gr.Blocks:
    no_image = empty_html("Wgraj zdjęcie znaku albo wylosuj zdjęcie testowe.")

    def recognize(sample: Sample | None):
        """Tab 'Rozpoznaj': result card, the 64x64 input, Grad-CAM, top-5."""
        if sample is None:
            return no_image, None, None, None
        img = sample.clean
        probs, cam = rec.explain(img)
        return (result_html(probs, rec.threshold, sample), view(as_hwc(img)), cam_view(img, cam, sample),
                top5_figure(probs, rec.threshold, sample.label))

    def robustness(sample: Sample | None, ctype: str, level: int):
        """Tab 'Test odporności': clean and corrupted image, Grad-CAM, result card, confidence vs level."""
        if sample is None:
            return None, None, None, no_image, None
        level = int(level)
        images = torch.stack([sample.corrupted(ctype, lv) for lv in APP_LEVELS])
        img = images[level]
        probs, cam = rec.explain(img)
        curve = rec.probs(images)
        curve[level] = probs  # the card and the plot show the very same prediction
        ref = sample.label if sample.label is not None else int(curve[0].argmax())
        condition = ("bez zakłóceń" if level == 0 else
                     f"{CORRUPTION_LABELS_PL[ctype]}, poziom {level} ({level_label_pl(ctype, level)})")
        return (view(as_hwc(sample.clean)), view(as_hwc(img)), cam_view(img, cam, sample),
                result_html(probs, rec.threshold, sample, condition),
                robustness_figure(curve, ctype, level, rec.threshold, ref, sample.label is not None))

    def on_upload(value: dict | None):
        img = (value or {}).get("composite")
        if img is None:  # cleared editor (also after a random test image)
            return (gr.skip(),) * 5
        sample = upload_sample(img)
        return sample, *recognize(sample)

    def on_random():
        if not tests.available:
            raise gr.Error("Brak zbioru testowego w data/ – wgraj własne zdjęcie.")
        sample = tests.random()
        return sample, None, *recognize(sample)

    with gr.Blocks(title="Rozpoznawanie znaków drogowych") as demo:
        gr.Markdown("# 🚦 Rozpoznawanie znaków drogowych\n"
                    f"ResNet18 (wagi ImageNet, augmentacja A2) · 43 klasy znaków ze zbioru GTSRB · "
                    f"obliczenia: {device_label}")
        sample_state = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Rozpoznaj"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.Markdown("**Wgraj zdjęcie znaku** i w razie potrzeby przytnij je (narzędzie kadrowania "
                                    "w edytorze). Model oczekuje wycinka, na którym znak jest na środku i zajmuje "
                                    "większość kadru – nie szuka znaku na zdjęciu całej ulicy.",
                                    elem_classes="hint")
                        editor = gr.ImageEditor(type="pil", image_mode="RGB", sources=("upload", "clipboard"),
                                                transforms=("crop",), brush=False, eraser=False, layers=False,
                                                format="png", label="Zdjęcie znaku", height=380,
                                                placeholder="Upuść tu zdjęcie znaku\nlub kliknij, aby wybrać plik")
                        random_1 = gr.Button("🎲 Losowe zdjęcie testowe", variant="primary",
                                             interactive=tests.available)
                    with gr.Column(scale=6):
                        card_1 = gr.HTML(no_image)
                        with gr.Row():
                            input_1 = image_view("Wejście modelu (64×64 px)")
                            cam_1 = image_view("Grad-CAM")
                        gr.Markdown(CAM_NOTE, elem_classes="note")
                        top5 = gr.Plot(show_label=False)

            with gr.Tab("Test odporności"):
                gr.Markdown("To samo zdjęcie co w zakładce **Rozpoznaj**, poddane wybranemu zakłóceniu. Poziomy "
                            "(1 = słabe, 3 = silne) i ich parametry są takie same jak w ocenie modeli; dla zdjęcia "
                            "testowego to dokładnie te obrazy, na których liczono wyniki.", elem_classes="hint")
                with gr.Row():
                    choices = [(CORRUPTION_LABELS_PL[c] + (" (nie było w treningu)" if c == "jpeg" else ""), c)
                               for c in CORRUPTIONS]
                    ctype_radio = gr.Radio(choices, value=CORRUPTIONS[0], label="Typ zakłócenia", scale=3)
                    level_slider = gr.Slider(0, 3, value=2, step=1, label="Poziom (0 = bez zakłóceń)", scale=2)
                    random_2 = gr.Button("🎲 Losowe zdjęcie testowe", interactive=tests.available, scale=1)
                with gr.Row():
                    clean_2 = image_view("Oryginał (64×64 px)")
                    corrupted_2 = image_view("Po zakłóceniu")
                    cam_2 = image_view("Grad-CAM")
                gr.Markdown(CAM_NOTE, elem_classes="note")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        card_2 = gr.HTML(no_image)
                    with gr.Column(scale=6):
                        level_plot = gr.Plot(show_label=False)

            with gr.Tab("Wyniki"):
                results_tab()

            with gr.Tab("O projekcie"):
                about_tab()

        tab_1 = [card_1, input_1, cam_1, top5]
        tab_2 = [clean_2, corrupted_2, cam_2, card_2, level_plot]
        robustness_args = dict(fn=robustness, inputs=[sample_state, ctype_radio, level_slider], outputs=tab_2)
        editor.change(on_upload, editor, [sample_state, *tab_1], api_name="wgraj").then(**robustness_args,
                                                                                 api_visibility="private")
        for button, visibility in ((random_1, "public"), (random_2, "private")):
            button.click(on_random, None, [sample_state, editor, *tab_1], api_name="losuj", api_visibility=visibility
                         ).then(**robustness_args, api_visibility="private")
        ctype_radio.change(**robustness_args, api_name="odpornosc")
        level_slider.change(**robustness_args, api_visibility="private")
        if tests.available:
            demo.load(on_random, None, [sample_state, editor, *tab_1], api_visibility="private").then(
                **robustness_args, api_visibility="private")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app: traffic sign recognition, robustness, results.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto = GPU if available, otherwise CPU")
    parser.add_argument("--port", type=int, default=None, help="default: 7860 or the next free port")
    parser.add_argument("--share", action="store_true", help="also create a temporary public gradio.live link")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: CUDA is not available")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    device_label = f"GPU ({torch.cuda.get_device_name(device)})" if use_cuda else "CPU"
    demo = build_app(Recognizer(device), TestImages(), device_label)
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft(primary_hue="blue"), css=CSS)


if __name__ == "__main__":
    main()
