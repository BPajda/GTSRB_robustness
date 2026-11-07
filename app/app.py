"""Gradio app: recognize a traffic sign, test the model's robustness to corruptions, browse the results.

    python -m app.app [--device auto|cpu|cuda] [--port 7860] [--share]

Model: the final model results/resnet18_pretrained_A2/best.pt, on the GPU when available (--device auto),
otherwise on the CPU. Inference runs in fp32 with TF32 off - the numerics in which the rejection threshold
in results/xai/threshold.json was chosen (on the validation set). As in training, an image is resized to
64x64 without keeping the aspect ratio, so the model expects a crop with the sign in the middle (like the
GTSRB images), not a whole street scene.

Tabs:
    Rozpoznaj         an uploaded (optionally cropped) photo, an example or a random test image: prediction,
                      Grad-CAM, top-5, a warning below the threshold, the true class of a test image
    Test odporności   the same image under one of the 5 corruptions of src.corruptions, level 0-3
    Wyniki            conclusions, tables and plots from results/summary, results/final_test and results/xai
    O projekcie       short description, links to the repository and the full report, limitations

Look: the road at night rather than a dashboard. Dark asphalt with cards one tone lighter; the only light
element is the white result plate, like a sign in the headlights. The other colours are those of the signs (the
blue of information signs, the yellow of warning signs, the red of prohibition signs) and the typeface is
Barlow, based on DIN 1451 of the German signs in GTSRB; the charts use the same fonts (app/assets/fonts).
There is one dark theme regardless of the system setting, so the interface and the charts always match.
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import io
import json
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import v2

from src.class_names_pl import CLASS_NAMES_PL, SIGN_CODES, category_pl, class_label
from src.corruptions import CORRUPTION_LABELS_PL, CORRUPTIONS, apply_corruption, corruption_rng, level_label_pl, \
    roi_to_pixels
from src.data import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SEED, GTSRBDataset, load_csv
from src.evaluate import load_model
from src.final_eval import BASELINE_RUN, FINAL_RUN, FINAL_TEST_DIR, KM3_RUNS, REFERENCE_RUN, RUN_LABELS, fmt_int
from src.summarize import MODEL_LABELS, pct_axis
from src.train import RESULTS_DIR, precision_for
from src.xai import ROI_COLOR, XAI_DIR, as_hwc, fmt, fmt_pct, gradcam, make_cam, overlay

FINAL_DIR = RESULTS_DIR / FINAL_RUN
SUMMARY_DIR = RESULTS_DIR / "summary"
TEST_RESULTS_DIR = RESULTS_DIR / FINAL_TEST_DIR
XAI_RESULTS_DIR = RESULTS_DIR / XAI_DIR
THRESHOLD_PATH = XAI_RESULTS_DIR / "threshold.json"
E5_SUMMARY_PATH = RESULTS_DIR / "e5_summary.json"
FONTS_DIR = Path(__file__).parent / "assets" / "fonts"
REPO_URL = "https://github.com/BPajda/GTSRB_robustness"
REPORT_URL = f"{REPO_URL}/blob/main/reports/GTSRB_robustness_report_PL.pdf"
WEIGHTS_URL = f"{REPO_URL}/releases/latest"

TOP_K = 5
APP_LEVELS = (0, 1, 2, 3)        # 0 = no corruption
VIEW_SIZE = 256                  # 64x64 images are shown 4x enlarged, pixel for pixel
UPLOAD_ROI_MARGIN = 0.1          # an uploaded crop is assumed to have the sign in its middle 80% (like GTSRB)
UPLOAD_RNG_INDEX = 0             # corruption_rng index of an uploaded image; test images use their own index
# Clickable examples (test set indices): one per visual sign group, the last two are hard - 11866 is right but
# below the threshold, 6879 is a blurred Stop read as "Zakaz wjazdu".
EXAMPLE_TEST_IMAGES = (864, 209, 6975, 1652, 9126, 5652, 11866, 6879)
THUMB_SIZE = 128
METRIC_LABELS = {"acc": "Accuracy", "macro_f1": "Macro-F1", "mca": "mCA", "relative_drop": "Wzgl. spadek"}

# Design tokens: the road at night, used by the CSS and the charts. Asphalt, cards one tone lighter, the colours
# of the signs; the only light element is the result plate, like a sign lit by headlights. Yellow and red carry
# one meaning each. Contrast ratios are WCAG ratios against the background the colour is used on.
ASPHALT = "#1E2328"      # page background; text on the plate and on the yellow bar
CARD = "#272D33"         # cards and panels
RAISED = "#313840"       # raised elements: inputs, the image editor, inactive segments
LINE = "#414A53"         # lines and borders, chart grid
TEXT = "#E6EAED"         # main text (11.5:1 on a card)
MUTED = "#A3ADB6"        # secondary text, chart axes (6.1:1 on a card, 5.2:1 on a raised element)
BLUE = "#3D8BD9"         # accent: primary button, active segment, slider, keyboard focus, highlighted bar or line
BLUE_HOVER = "#62A1E3"   # hover of the blue elements
BLUE_INK = "#121518"     # text on BLUE (5.1:1; ASPHALT would be 4.4:1, below AA)
SIGN_BLUE = "#0B5394"    # information signs: the "model is confident" bar, white text (7.8:1)
YELLOW = "#F2B705"       # warning signs: only "the model is not confident" (and the threshold behind it)
RED = "#C8102E"          # prohibition signs: only a wrong prediction or an error, white text (5.9:1)
WHITE = "#FFFFFF"        # the result plate, the board under the static plots, text on the blue and red bars
PLATE_MUTED = "#596067"  # secondary text on the plate (6.4:1)
PLATE_LINE = "#C9CED3"   # lines on the plate
BAR_GRAY = "#5C6670"     # other classes in the charts
BODY_FONT, DISPLAY_FONT = "Barlow", "Barlow Semi Condensed"

RESIZE = v2.Compose([v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])  # as GTSRBDataset
TO_FLOAT = v2.ToDtype(torch.float32, scale=True)
NORMALIZE = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)

for _font in sorted(FONTS_DIR.glob("*.ttf")):
    font_manager.fontManager.addfont(str(_font))
matplotlib.rcParams.update({
    "font.family": [BODY_FONT, "DejaVu Sans"],  # DejaVu only for glyphs Barlow lacks (the σ of the blur levels)
    "font.size": 10,
    "text.color": MUTED,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": LINE,
    "axes.facecolor": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": LINE,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "figure.facecolor": "none",
    "svg.fonttype": "path",  # glyphs as paths: the SVG looks the same without the font in the browser
})


# ---------------------------------------------------------------------------
# Model and images
# ---------------------------------------------------------------------------

def softmax(logits: torch.Tensor) -> np.ndarray:
    """Softmax in float64, as in src.xai (the threshold was chosen on these values)."""
    return torch.softmax(logits.double(), dim=1).cpu().numpy()


class Recognizer:
    """The final model and its rejection threshold. Calls are serialized: Grad-CAM hooks are not thread-safe."""

    def __init__(self, device: torch.device):
        weights = FINAL_DIR / "best.pt"
        if not weights.exists():
            raise SystemExit(f"{weights} not found. The model weights are not in the repository; either\n"
                             f"  - download best.pt from {WEIGHTS_URL} into results/{FINAL_RUN}/, or\n"
                             f"  - train the model: python -m src.train --config configs/{FINAL_RUN}.yaml")
        if not THRESHOLD_PATH.exists():
            raise SystemExit(f"{THRESHOLD_PATH} not found - create it with: python -m src.xai")
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
    """Test images with labels and ROIs, read from disk one at a time."""

    def __init__(self):
        self.available = (DATA_DIR / "Test.csv").exists()
        if self.available:
            self.df = load_csv("test")
            self.ds = GTSRBDataset(self.df)
        self.rng = np.random.default_rng()  # unseeded on purpose: every click shows another image

    def sample(self, i: int) -> Sample:
        row = self.df.iloc[i]
        return Sample(self.ds.image_uint8(i), self.ds.rois[i], i, int(self.ds.labels[i]),
                      f"zdjęcie testowe {row['Path']} (oryginał {row['Width']}×{row['Height']} px)")

    def random(self) -> Sample:
        return self.sample(int(self.rng.integers(len(self.ds))))

    def thumbnail(self, i: int) -> tuple[Image.Image, str]:
        """The original test image as a square thumbnail, with its class name as the caption (alt text)."""
        img = Image.open(DATA_DIR / self.df.iloc[i]["Path"]).convert("RGB")
        c = int(self.ds.labels[i])
        return img.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS), f"{SIGN_CODES[c]} {CLASS_NAMES_PL[c]}"


def upload_sample(img: Image.Image) -> Sample:
    img = ImageOps.exif_transpose(img).convert("RGB")
    m = UPLOAD_ROI_MARGIN * IMG_SIZE
    roi = np.array([m, m, IMG_SIZE - m, IMG_SIZE - m], dtype=np.float32)
    return Sample(RESIZE(img).as_subclass(torch.Tensor), roi, UPLOAD_RNG_INDEX, None,
                  f"wgrane zdjęcie ({img.width}×{img.height} px)")


# ---------------------------------------------------------------------------
# Views: images, the result plate, charts
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


def png_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # lossless: the 64x64 pixels stay exact
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@functools.lru_cache(maxsize=None)
def pictogram_uri(c: int) -> str:
    """data/Meta/<c>.png (transparent background) as a data URI, empty if the file is missing."""
    path = DATA_DIR / "Meta" / f"{c}.png"
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else ""


def pictogram_html(c: int, css_class: str) -> str:
    uri = pictogram_uri(c)
    return f'<img class="{css_class}" src="{uri}" alt="Piktogram: {CLASS_NAMES_PL[c]}">' if uri else ""


def lower_first(text: str) -> str:
    """'Zasłonięcie ROI' -> 'zasłonięcie ROI' (str.lower would also lower the acronym)."""
    return text[:1].lower() + text[1:]


def pct_label(p: float) -> str:
    return "< 0,01%" if p < 0.0001 else fmt_pct(p, 2)


def result_html(probs: np.ndarray, threshold: float, sample: Sample) -> str:
    """The result plate (pictogram, code, name, confidence) and one status bar under it:
    a wrong prediction (known true class) > below the threshold > confident."""
    pred = int(probs.argmax())
    conf = float(probs[pred])
    below = conf < threshold
    conf_s, thr_s = fmt_pct(conf, 2), fmt_pct(threshold, 2)
    hit_note = " Wynik zgadza się z prawdziwą klasą zdjęcia testowego." if sample.label == pred else ""
    if sample.label is not None and sample.label != pred:
        c = sample.label
        state, title = "miss", "Pomyłka"
        detail = (f'{pictogram_html(c, "status-pict")}<span>Prawdziwa klasa to <strong>{SIGN_CODES[c]} '
                  f'{CLASS_NAMES_PL[c]}</strong>. ' +
                  (f"Model nie był pewny (pewność {conf_s}, próg {thr_s}), więc w działającym systemie ta "
                   "predykcja zostałaby odrzucona.</span>" if below else
                   f"Model był pewny (pewność {conf_s}, próg {thr_s}), więc próg tego błędu nie wychwycił.</span>"))
    elif below:
        state, title = "warn", "Model nie jest pewny – wynik wymaga potwierdzenia"
        detail = (f"<span>Pewność {conf_s} jest poniżej progu {thr_s}, wyznaczonego na walidacji. W działającym "
                  f"systemie ta predykcja zostałaby odrzucona.{hit_note}</span>")
    else:
        state, title = "ok", "Model jest pewny"
        detail = f"<span>Pewność {conf_s} jest powyżej progu {thr_s}, wyznaczonego na walidacji.{hit_note}</span>"
    name = CLASS_NAMES_PL[pred]
    return (f'<div class="result"><div class="plate">{pictogram_html(pred, "plate-pict")}'
            f'<div class="plate-text"><div class="plate-meta"><span class="sign-code">{SIGN_CODES[pred]}</span>'
            f'<span>znak {category_pl(pred)}, klasa {pred} w GTSRB</span></div>'
            f'<div class="plate-name{" long" if len(name) > 36 else ""}">{name}</div></div>'
            f'<div class="plate-conf"><div class="plate-number">{conf_s}</div>'
            f'<div class="plate-label">pewność modelu</div></div></div>'
            f'<div class="status {state}" role="status"><div class="status-title">{title}</div>'
            f'<div class="status-detail">{detail}</div></div></div>')


def figure_html(img: Image.Image, title: str, note: str, alt: str) -> str:
    return (f'<figure class="view"><figcaption><span class="view-title">{title}</span>'
            f'<span class="view-note">{note}</span></figcaption>'
            f'<img src="{png_uri(img)}" alt="{html.escape(alt)}" width="{VIEW_SIZE}" height="{VIEW_SIZE}"></figure>')


def views_html(figures: list[str], sample: Sample) -> str:
    roi_note = " Turkusowa ramka to obszar znaku oznaczony w zbiorze." if sample.label is not None else ""
    return (f'<div class="card views views-{len(figures)}">{"".join(figures)}</div>'
            f'<p class="footnote">Źródło: {html.escape(sample.source)}.{roi_note}</p>')


CAM_NOTE = "jasne obszary najbardziej wpłynęły na wynik"


def chart_uri(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=True)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def chart_html(title: str, fig: Figure, alt: str) -> str:
    return (f'<section class="card chart"><h3 class="card-title">{title}</h3>'
            f'<img src="{chart_uri(fig)}" alt="{html.escape(alt)}"></section>')


def threshold_line(ax, threshold: float, vertical: bool) -> None:
    """The rejection threshold in warning-sign yellow."""
    (ax.axvline if vertical else ax.axhline)(threshold * 100, color=YELLOW, lw=2, zorder=1)


def top5_figure(probs: np.ndarray, threshold: float, label: int | None) -> Figure:
    """Top-5 classes: the prediction in blue (its name and value in the main text colour), the rest gray;
    a missed true class outlined in blue."""
    top = np.argsort(probs)[::-1][:TOP_K]
    pred = int(top[0])
    y = np.arange(TOP_K)
    fig = Figure(figsize=(7.2, 2.5), layout="constrained")
    ax = fig.subplots()
    thr = threshold * 100
    for yi, c in zip(y, top):
        missed_truth = c == label and c != pred
        p = probs[c] * 100
        ax.barh(yi, p, height=0.62, color=CARD if missed_truth else BLUE if c == pred else BAR_GRAY,
                edgecolor=BLUE if missed_truth else "none", lw=1.5, zorder=2)
        # a bar ending just before the threshold gets its label behind the line, not across it
        ax.text((thr if thr - 10 < p < thr else p) + 1.2, yi, pct_label(probs[c]), va="center", fontsize=10,
                color=TEXT if c == pred else MUTED, fontweight=500 if c == pred else 400, zorder=3)
    names = [class_label(int(c), short=True, code=True) + (" (prawdziwa)" if c == label and c != pred else "")
             for c in top]
    ax.set_yticks(y, names)
    for tick, c in zip(ax.get_yticklabels(), top):
        tick.set_fontweight(500 if c == pred else 400)
        tick.set_color(TEXT if c == pred else MUTED)
    ax.tick_params(axis="y", pad=8)
    ax.set_ylim(TOP_K - 0.5, -1.15)
    threshold_line(ax, threshold, vertical=True)
    ax.text(thr - 1, -0.85, f"próg {fmt_pct(threshold, 2)}", ha="right", va="center", fontsize=9.5, color=TEXT)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 25, 50, 75, 100])
    pct_axis(ax, "x")
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("prawdopodobieństwo według modelu", labelpad=6)
    return fig


def top5_alt(probs: np.ndarray) -> str:
    top = np.argsort(probs)[::-1][:TOP_K]
    return "Pięć najbardziej prawdopodobnych klas: " + "; ".join(
        f"{class_label(int(c), code=True)} {pct_label(probs[c])}" for c in top) + "."


def robustness_figure(probs: np.ndarray, ctype: str, level: int, threshold: float, ref: int,
                      ref_is_true: bool) -> Figure:
    """probs [4, 43]: softmax at levels 0-3. Probability of the reference class (true class, or the class
    predicted without corruption) and the confidence of the predicted class; a marked square = another class
    (filled red when the reference is the true class, i.e. a mistake, otherwise hollow)."""
    x = np.array(APP_LEVELS)
    pred, conf, p_ref = probs.argmax(axis=1), probs.max(axis=1) * 100, probs[:, ref] * 100
    other = pred != ref
    # red is too dark on a card for a thin outline (2.4:1), so a mistake is a red square edged in the text colour
    other_face, other_edge, other_lw = (RED, TEXT, 1.5) if ref_is_true else (CARD, MUTED, 2)
    fig = Figure(figsize=(6.6, 3.5), layout="constrained")
    ax = fig.subplots()
    ax.axvspan(level - 0.32, level + 0.32, color=RAISED, lw=0, zorder=0)
    threshold_line(ax, threshold, vertical=False)
    ax.plot(x, conf, color=MUTED, lw=2, marker="s", ms=8, zorder=2)
    ax.plot(x, p_ref, color=BLUE, lw=2, marker="o", ms=7, zorder=3)
    ax.scatter(x[other], conf[other], marker="s", s=80, facecolor=other_face, edgecolor=other_edge, lw=other_lw,
               zorder=4)
    for i in np.flatnonzero(other):
        below = conf[i] > 75
        ax.annotate(class_label(int(pred[i]), short=True, width=20), (x[i], conf[i]),
                    xytext=(0, -10 if below else 10), textcoords="offset points", ha="center",
                    va="top" if below else "bottom", fontsize=9, color=TEXT if ref_is_true else MUTED)
    ref_name = class_label(ref, short=True)
    ref_role = "prawdziwa klasa" if ref_is_true else "klasa rozpoznana bez zakłóceń"
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=7, label=f"{ref_role}: {ref_name}"),
               Line2D([], [], color=MUTED, lw=2, marker="s", ms=7, label="pewność modelu (przewidziana klasa)")]
    if other.any():
        other_label = "pomyłka (podpisana)" if ref_is_true else "inna klasa (podpisana)"
        handles.append(Line2D([], [], color=other_edge, lw=0, marker="s", ms=7, markerfacecolor=other_face,
                              markeredgewidth=other_lw, label=other_label))
    handles.append(Line2D([], [], color=YELLOW, lw=2, label=f"próg {fmt_pct(threshold, 2)}"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=9.5,
              handlelength=2.2, columnspacing=1.6, borderaxespad=0)
    ax.set_xticks(x, [f"{lv}\n{'bez zakłóceń' if lv == 0 else level_label_pl(ctype, lv)}" for lv in APP_LEVELS])
    ax.set_xlim(-0.5, len(APP_LEVELS) - 0.5)
    ax.set_xlabel("poziom zakłócenia", labelpad=6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    pct_axis(ax)
    ax.tick_params(axis="y", colors=MUTED)
    ax.grid(axis="y", visible=True)
    ax.set_ylabel("prawdopodobieństwo")
    return fig


def robustness_alt(probs: np.ndarray, ctype: str) -> str:
    conf = probs.max(axis=1)
    return (f"Pewność modelu dla poziomów 0–3 ({CORRUPTION_LABELS_PL[ctype]}): " +
            "; ".join(f"poziom {lv}: {class_label(int(probs[lv].argmax()), code=True)} {fmt_pct(conf[lv], 2)}"
                      for lv in APP_LEVELS) + ".")


def empty_html(text: str) -> str:
    return f'<div class="empty">{text}</div>'


def html_block(value: str = "", **kwargs) -> gr.HTML:
    """gr.HTML without Gradio's prose styles: the app CSS styles its own markup."""
    return gr.HTML(value, apply_default_css=False, **kwargs)


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------

def file_url(path: Path) -> str:
    """URL of a file under a static path (gr.set_static_paths), served as is."""
    return "/gradio_api/file=" + urllib.parse.quote(path.resolve().as_posix())


def table_html(head: str, rows: list[str], css_class: str = "") -> str:
    return (f'<div class="table-wrap"><table class="data-table {css_class}"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def variant_cells(row) -> str:
    return (f'<td>{MODEL_LABELS[row.model]}</td><td>{"ImageNet" if row.pretrained else "od zera"}</td>'
            f'<td>{row.aug}</td>')


def finetune_cell(row) -> str:
    """Fine-tuning strategy of the ImageNet weights (E5); '–' for models trained from scratch.

    summary.csv written before E5 has no finetune column: all its runs are full fine-tuning.
    """
    return f'<td>{getattr(row, "finetune", "full") if row.pretrained else "–"}</td>'


def finetune_note(summary: pd.DataFrame) -> str:
    """Explanation of the Fine-tuning column, only when the E5 strategies are in the table."""
    if "finetune" not in summary or set(summary["finetune"]) <= {"full"}:
        return ""
    return (" Kolumna „Fine-tuning” mówi, jak dostrajano wagi ImageNet (etap E5): full – cała sieć z jednym lr, "
            "linear – tylko ostatnia warstwa, reszta sieci zamrożona, discriminative – cała sieć z niższym lr "
            "wcześniejszych warstw, two_stage – najpierw sama ostatnia warstwa, potem cała sieć.")


def summary_table(summary: pd.DataFrame) -> str:
    head = ('<tr><th>Model</th><th>Wagi startowe</th><th>Augmentacja</th><th>Fine-tuning</th><th>Etap</th>'
            '<th class="num">Seed</th>'
            '<th class="num">Accuracy</th><th class="num">Macro-F1</th><th class="num">mCA</th>'
            '<th class="num">Wzgl. spadek</th><th class="num">Parametry</th><th class="num">CPU [ms]</th>'
            '<th class="num">Najlepsza epoka</th></tr>')
    rows = [f'<tr{" class=final" if r.run_name == FINAL_RUN else ""}>{variant_cells(r)}{finetune_cell(r)}'
            f'<td>{r.stage}</td>'
            f'<td class="num">{r.seed}</td><td class="num">{fmt_pct(r.acc, 2)}</td>'
            f'<td class="num">{fmt_pct(r.macro_f1, 2)}</td><td class="num">{fmt_pct(r.mca, 2)}</td>'
            f'<td class="num">{fmt_pct(r.relative_drop, 1)}</td><td class="num">{fmt(r.params / 1e6)} mln</td>'
            f'<td class="num">{fmt(r.cpu_latency_ms, 1)}</td><td class="num">{r.best_epoch} z {r.epochs_run}</td></tr>'
            for r in summary.itertuples()]
    return table_html(head, rows)


def comparison_table(comparison: pd.DataFrame, summary: pd.DataFrame) -> str:
    """One row per model: every metric on validation and on the test set."""
    roles = {FINAL_RUN: "model finalny", BASELINE_RUN: "baseline", REFERENCE_RUN: "punkt odniesienia"}
    variants = summary.set_index("run_name")
    metrics = list(dict.fromkeys(comparison["metric"]))
    head = ('<tr><th rowspan="2">Model</th><th rowspan="2">Wagi startowe</th><th rowspan="2">Augmentacja</th>' +
            "".join(f'<th colspan="2" class="group">{METRIC_LABELS[m]}</th>' for m in metrics) + "</tr><tr>" +
            '<th class="num">walidacja</th><th class="num">test</th>' * len(metrics) + "</tr>")
    rows = []
    for run, g in comparison.groupby("run", sort=False):
        v = variants.loc[run]
        cells = "".join(f'<td class="num">{fmt_pct(r.val, 2)}</td><td class="num test">{fmt_pct(r.test, 2)}</td>'
                        for r in g.itertuples())
        rows.append(f'<tr{" class=final" if run == FINAL_RUN else ""}><td>{MODEL_LABELS[v.model]}'
                    f'<span class="role">{roles.get(run, "")}</span></td>'
                    f'<td>{"ImageNet" if v.pretrained else "od zera"}</td><td>{v.aug}</td>{cells}</tr>')
    return table_html(head, rows, "comparison")


def plot_html(path: Path, title: str, caption: str) -> str:
    url = file_url(path)
    return (f'<figure class="card result-figure"><figcaption><h3 class="card-title">{title}</h3>'
            f'{f"<p>{caption}</p>" if caption else ""}</figcaption>'
            f'<a href="{url}" target="_blank" rel="noopener" title="Otwórz w pełnym rozmiarze">'
            f'<img src="{url}" alt="{html.escape(title)}"></a></figure>')


def results_html() -> str:
    summary = pd.read_csv(SUMMARY_DIR / "summary.csv")
    comparison = pd.read_csv(TEST_RESULTS_DIR / "comparison.csv")
    corruption = pd.read_csv(TEST_RESULTS_DIR / "corruption_acc.csv")
    thr = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    conf = json.loads((XAI_RESULTS_DIR / "confidence.json").read_text(encoding="utf-8"))["models"][FINAL_RUN]
    roi = pd.read_csv(XAI_RESULTS_DIR / "roi_energy.csv")
    roi = roi[(roi["run"] == FINAL_RUN) & (roi["condition"] == "clean") & (roi["subset"] == "all")].iloc[0]
    mca = summary.set_index("run_name")["mca"]
    test = comparison.set_index(["run", "metric"])
    t = test["test"]
    final_corr = corruption[(corruption["run"] == FINAL_RUN) & (corruption["level"] > 0)]
    hardest = final_corr.groupby("corruption", sort=False)["test"].mean()
    clean, corrupted = thr["test"]["clean"], thr["test"]["corrupted_all"]
    # E5 (fine-tuning strategies, validation only) came after the final evaluation: optional
    e5 = (json.loads(E5_SUMMARY_PATH.read_text(encoding="utf-8"))["per_strategy"]
          if E5_SUMMARY_PATH.exists() else None)
    e5_plot = SUMMARY_DIR / "e5_finetuning.png"
    e5_lead = (f"Liczy się też sposób dostrajania wag ImageNet: uczenie samej ostatniej warstwy daje na walidacji "
               f"tylko {fmt_pct(e5['linear']['macro_f1'], 0)} macro-F1 (etap E5). " if e5 else "")

    lead = (f"Model finalny, ResNet18 z wagami ImageNet i augmentacją odpornościową A2, rozpoznaje poprawnie "
            f"{fmt_pct(t[(FINAL_RUN, 'acc')])} zdjęć testowych, a zdjęć zakłóconych średnio "
            f"{fmt_pct(t[(FINAL_RUN, 'mca')])}. Własny baseline SimpleCNN osiąga odpowiednio "
            f"{fmt_pct(t[(BASELINE_RUN, 'acc')])} i {fmt_pct(t[(BASELINE_RUN, 'mca')])}, a ta sama sieć ResNet18 "
            f"bez augmentacji {fmt_pct(t[(REFERENCE_RUN, 'mca')])} na zdjęciach zakłóconych, więc odporność daje "
            f"głównie augmentacja. {e5_lead}Najtrudniejsze pozostaje zakłócenie "
            f"„{lower_first(CORRUPTION_LABELS_PL[hardest.idxmin()])}” (średnio {fmt_pct(hardest.min())} na trzech "
            f"poziomach). Grad-CAM pokazuje, że model patrzy na znak, a próg pewności "
            f"{fmt_pct(thr['threshold'], 2)} odrzuca {fmt_pct(1 - clean['coverage'])} zdjęć testowych i zostawia "
            f"predykcje poprawne w {fmt_pct(clean['accuracy_kept'], 2)}.")

    validation = [
        '<section class="results-section"><h2>Porównanie wariantów na walidacji</h2>',
        "<p>Wszystkie wytrenowane warianty, oceniane na walidacji (serie zdjęć wydzielone ze zbioru treningowego). "
        "mCA to średnia accuracy po 5 zakłóceniach i 3 poziomach, względny spadek to (accuracy − mCA) / accuracy, "
        "CPU to mediana czasu predykcji jednego zdjęcia. Wyróżniony wiersz to model finalny, używany w aplikacji."
        f"{finetune_note(summary)}</p>",
        summary_table(summary),
        plot_html(SUMMARY_DIR / "f1_mca_bars.png", "Skuteczność i odporność wszystkich wariantów",
                  "Macro-F1 na czystej walidacji i mCA. Na czystych danych różnice są niewielkie, w odporności duże: "
                  f"ResNet18 z wagami ImageNet i augmentacją A2 ma mCA {fmt_pct(mca[FINAL_RUN])}, baseline SimpleCNN "
                  f"bez augmentacji {fmt_pct(mca[BASELINE_RUN])}."),
        plot_html(SUMMARY_DIR / "robustness_curves.png", "Accuracy a poziom zakłócenia (H2, H4)",
                  "Augmentacje A0, A1 i A2 w każdej architekturze. A2 najmocniej pomaga przy szumie i zasłonięciu; "
                  "przy kompresji JPEG, której nie było w treningu, zysk jest znacznie mniejszy."),
        plot_html(SUMMARY_DIR / "corruption_heatmap.png", "Accuracy dla każdego zakłócenia",
                  "Średnia z 3 poziomów. Najtrudniejsze jest zasłonięcie części znaku, najmniej szkodzi "
                  "przyciemnienie."),
        plot_html(SUMMARY_DIR / "h1_learning_curves.png", "Uczenie od zera a wagi ImageNet (H1)",
                  "Macro-F1 walidacji po kolejnych epokach, bez augmentacji. Sieci z wagami ImageNet osiągają wyższy "
                  "wynik w mniejszej liczbie epok (gwiazdka oznacza epokę, z której zapisano wagi)."),
        plot_html(SUMMARY_DIR / "params_vs_f1.png", "Rozmiar modelu a macro-F1 (H3)",
                  "Liczba parametrów (skala logarytmiczna) a macro-F1. ResNet18 jest najdokładniejszy, ale ma 7 razy "
                  "więcej parametrów niż MobileNetV3-Small i jest kilka razy wolniejszy na CPU."),
        *([plot_html(e5_plot, "Strategie fine-tuningu ResNet18 (E5)",
                     "ResNet18 z wagami ImageNet i augmentacją A2, na walidacji. Uczenie samej głowicy (linear) daje "
                     f"tylko {fmt_pct(e5['linear']['macro_f1'], 0)} macro-F1, a niższe lr dla backbone'u "
                     f"(discriminative) – najwyższe mCA, {fmt_pct(e5['discriminative']['mca'])}. E5 przeprowadzono "
                     "po ocenie na teście, więc finalny model się nie zmienił.")]
          if e5 and e5_plot.exists() else []),
        "</section>",
    ]
    matrices = "".join(plot_html(TEST_RESULTS_DIR / run / "confusion_matrix_test.png", RUN_LABELS[run]
                                 .replace(" · ", ", "), "") for run in KM3_RUNS)
    test_set = [
        '<section class="results-section"><h2>Zbiór testowy</h2>',
        f"<p>Ocena na zbiorze testowym ({fmt_int(int(comparison['n_test'].iloc[0]))} zdjęć, użyty jeden raz, po "
        "wyborze modelu na walidacji): model finalny, baseline i ten sam ResNet18 bez augmentacji, który pokazuje "
        "efekt augmentacji A2.</p>",
        comparison_table(comparison, summary),
        plot_html(TEST_RESULTS_DIR / "robustness_test.png", "Accuracy na teście a poziom zakłócenia",
                  "Linie ciągłe to zbiór testowy, przerywane to walidacja. Wyniki testu są bliskie walidacji (mCA "
                  f"modelu finalnego: {fmt_pct(test.loc[(FINAL_RUN, 'mca'), 'val'])} na walidacji, "
                  f"{fmt_pct(test.loc[(FINAL_RUN, 'mca'), 'test'])} na teście)."),
        '<details class="card"><summary>Macierze pomyłek na czystym zbiorze testowym</summary>'
        '<p class="hint">Kliknij macierz, aby otworzyć ją w pełnym rozmiarze.</p>'
        f'<div class="matrices">{matrices}</div></details>',
        "</section>",
    ]
    xai = [
        '<section class="results-section"><h2>Na co patrzy model i kiedy mu ufać</h2>',
        plot_html(XAI_RESULTS_DIR / "gradcam_examples.png", "Grad-CAM na przykładach",
                  "Poprawnie rozpoznane zdjęcia testowe, po jednym z każdej grupy znaków: model finalny i baseline. "
                  "Jasne obszary najmocniej wpłynęły na decyzję, turkusowa ramka to oznaczony obszar znaku (ROI)."),
        plot_html(XAI_RESULTS_DIR / "roi_energy.png", "Jaka część uwagi przypada na znak",
                  "Udział mapy Grad-CAM wewnątrz ROI na czystym teście. U modelu finalnego mediana wynosi "
                  f"{fmt_pct(roi['energy_in_roi_median'], 0)}, choć ROI zajmuje średnio "
                  f"{fmt_pct(roi['roi_area_mean'], 0)} obrazu, więc model patrzy na znak, a nie na tło."),
        plot_html(XAI_RESULTS_DIR / "corruption_shift.png", "Uwaga modelu przy zakłóceniach",
                  "Cztery zdjęcia testowe bez zakłóceń i z każdym z 5 zakłóceń na poziomach 2 i 3 (model finalny)."),
        plot_html(XAI_RESULTS_DIR / "error_gallery.png", "Najbardziej pewne błędy",
                  "Błędy modelu finalnego z najwyższą pewnością, pogrupowane według par klas: prawdziwej "
                  "i przewidzianej."),
        plot_html(XAI_RESULTS_DIR / "confidence.png", "Pewność i kalibracja",
                  "Rozkład pewności dla predykcji poprawnych i błędnych oraz diagram niezawodności. Model finalny "
                  f"jest dobrze skalibrowany (ECE {fmt(conf['clean']['ece'], 4)} na czystym teście), a jego błędy "
                  f"mają zwykle niską pewność (mediana {fmt_pct(conf['clean']['median_conf_wrong'], 0)})."),
        plot_html(XAI_RESULTS_DIR / "threshold.png", "Próg pewności używany w aplikacji",
                  "Najniższy próg, przy którym zachowane predykcje na zakłóconej walidacji mają co najmniej "
                  f"{fmt_pct(thr['target_accuracy'])} accuracy: {fmt_pct(thr['threshold'], 2)}. Na czystym teście "
                  f"odrzuca {fmt_pct(1 - clean['coverage'])} zdjęć, a z zachowanych "
                  f"{fmt_pct(clean['accuracy_kept'], 2)} jest poprawnych; na zakłóconym teście zachowuje "
                  f"{fmt_pct(corrupted['coverage'])} predykcji z accuracy {fmt_pct(corrupted['accuracy_kept'], 2)}."),
        "</section>",
    ]
    return f'<div class="results"><p class="lead">{lead}</p>{"".join(validation + test_set + xai)}</div>'


def about_markdown(device_label: str) -> str:
    test = pd.read_csv(TEST_RESULTS_DIR / "comparison.csv").set_index(["run", "metric"])["test"]
    threshold = float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])
    return f"""
## O projekcie

Projekt sprawdza, czy uczenie transferowe i augmentacja danych pomagają klasyfikatorowi znaków drogowych
rozpoznawać zdjęcia rozmyte, zaszumione, przyciemnione, częściowo zasłonięte i mocno skompresowane. Porównano
własną sieć SimpleCNN z ResNet18 i MobileNetV3-Small, uczonymi od zera i z wagami ImageNet, oraz trzy warianty
augmentacji na zbiorze GTSRB (43 klasy niemieckich znaków).

W aplikacji działa model finalny: ResNet18 z wagami ImageNet i augmentacją odpornościową A2. Na zbiorze testowym
rozpoznaje poprawnie {fmt_pct(test[(FINAL_RUN, 'acc')])} zdjęć, a zdjęć zakłóconych średnio
{fmt_pct(test[(FINAL_RUN, 'mca')])}. Grad-CAM pokazuje, które fragmenty obrazu zdecydowały o wyniku, a próg
pewności {fmt_pct(threshold, 2)}, wyznaczony na walidacji, oznacza wyniki, którym nie należy ufać.

Kod i wyniki: [{REPO_URL.removeprefix('https://')}]({REPO_URL})

Pełne sprawozdanie (PDF): [GTSRB_robustness_report_PL.pdf]({REPORT_URL})

## Ograniczenia

- Model zna tylko 43 klasy niemieckich znaków ze zbioru GTSRB. Polskie nazwy to odpowiedniki o tym samym
  znaczeniu, ale polskie znaki mogą wyglądać trochę inaczej. Znaku spoza tych klas model nie rozpozna, tylko
  wskaże najbardziej podobny, zwykle z niską pewnością.
- Model ocenia wycinek ze znakiem na środku i nie szuka znaku na zdjęciu całej ulicy. Takie zdjęcie trzeba
  najpierw przyciąć do znaku.
- Model uczył się na zdjęciach z kamery samochodowej, więc na rysunkach i wzorcowych piktogramach myli się
  częściej.
- To projekt badawczy, a nie system do użytku na drodze. Wyniki zmierzono na jednym zbiorze danych.

<p class="device">Obliczenia w tej sesji: {device_label}.</p>
"""


# ---------------------------------------------------------------------------
# Look: theme, CSS, dark mode
# ---------------------------------------------------------------------------

def sign_theme() -> gr.themes.Base:
    """Gradio theme in the night road tokens, the same in light and dark mode (one look in every mode). A value
    set here is used in both modes; any other value takes Gradio's dark default, built from the asphalt scale."""
    blue = gr.themes.Color(c50="#EAF3FC", c100="#D3E6F8", c200="#A9CDF0", c300="#7DB3E8", c400=BLUE_HOVER,
                           c500=BLUE, c600="#2F74BC", c700="#245E9C", c800=SIGN_BLUE, c900="#083E6F",
                           c950="#052A4A", name="sign_blue")
    gray = gr.themes.Color(c50="#F5F7F8", c100=TEXT, c200="#CBD2D8", c300="#B6BEC6", c400=MUTED, c500="#7A848E",
                           c600=BAR_GRAY, c700=LINE, c800=RAISED, c900=CARD, c950=ASPHALT, name="asphalt")
    body = gr.themes.GoogleFont(BODY_FONT, weights=(400, 500))
    theme = gr.themes.Base(primary_hue=blue, secondary_hue=blue, neutral_hue=gray,
                           radius_size=gr.themes.sizes.radius_sm,
                           font=[body, gr.themes.GoogleFont(DISPLAY_FONT, weights=(600, 700)), "sans-serif"],
                           font_mono=[body, "sans-serif"])
    values = dict(
        body_background_fill=ASPHALT, body_text_color=TEXT, body_text_color_subdued=MUTED, body_text_size="16px",
        background_fill_primary=CARD, background_fill_secondary=RAISED,
        border_color_primary=LINE, border_color_accent=BLUE, border_color_accent_subdued=LINE,
        color_accent=BLUE, color_accent_soft=RAISED,
        block_background_fill=CARD, block_border_color=LINE, block_border_width="1px", block_radius="8px",
        block_shadow="none", block_padding="16px",
        block_label_background_fill=CARD, block_label_text_color=MUTED, block_label_border_width="0px",
        block_label_text_weight="500", block_label_text_size="15px",
        block_title_background_fill="transparent", block_title_text_color=TEXT, block_title_text_weight="500",
        block_title_text_size="15px", block_title_padding="0", block_title_radius="0",
        block_info_text_color=MUTED, block_info_text_size="14px",
        panel_background_fill=CARD, panel_border_color=LINE,
        input_background_fill=RAISED, input_background_fill_focus=RAISED, input_border_color=LINE,
        input_border_color_focus=BLUE, input_shadow="none", input_shadow_focus=f"0 0 0 2px {BLUE}",
        input_placeholder_color=MUTED,
        button_border_width="1.5px", button_large_radius="6px", button_large_text_size="17px",
        button_large_text_weight="500", button_large_padding="11px 20px",
        button_transform_hover="none", button_transform_active="none",
        button_primary_background_fill=BLUE, button_primary_background_fill_hover=BLUE_HOVER,
        button_primary_border_color=BLUE, button_primary_border_color_hover=BLUE_HOVER,
        button_primary_text_color=BLUE_INK, button_primary_text_color_hover=BLUE_INK,
        button_primary_shadow="none", button_primary_shadow_hover="none", button_primary_shadow_active="none",
        # blue text is below AA on a card (3.9:1): a secondary button is outlined in blue, its text stays light
        button_secondary_background_fill=CARD, button_secondary_background_fill_hover=RAISED,
        button_secondary_border_color=BLUE, button_secondary_border_color_hover=BLUE_HOVER,
        button_secondary_text_color=TEXT, button_secondary_text_color_hover=TEXT,
        button_secondary_shadow="none", button_secondary_shadow_hover="none", button_secondary_shadow_active="none",
        checkbox_label_background_fill=RAISED, checkbox_label_background_fill_hover=RAISED,
        checkbox_label_background_fill_selected=BLUE, checkbox_label_text_color=TEXT,
        checkbox_label_text_color_selected=BLUE_INK, checkbox_label_border_color=LINE,
        checkbox_label_border_color_hover=BLUE, checkbox_label_border_color_selected=BLUE,
        checkbox_label_shadow="none", checkbox_label_text_weight="500",
        slider_color=BLUE, loader_color=BLUE,
        link_text_color=TEXT, link_text_color_hover=TEXT, link_text_color_visited=TEXT,
        link_text_color_active=TEXT,
        table_border_color=LINE, table_even_background_fill=CARD, table_odd_background_fill=CARD,
        table_text_color=TEXT,
        shadow_drop="none", shadow_drop_lg="none", shadow_spread="0px", shadow_inset="none",
        error_background_fill=CARD, error_border_color=RED, error_text_color=TEXT, error_icon_color=RED,
    )
    theme.set(**values)
    for name in list(vars(theme)):
        base = name.removesuffix("_dark")
        if name.endswith("_dark") and hasattr(theme, base):
            if base in values:
                setattr(theme, name, getattr(theme, base))
            elif getattr(theme, name) is not None:  # None = no dark default, the light value is used
                setattr(theme, base, getattr(theme, name))
    return theme


# Keep the dark look even if Gradio (system light mode or ?__theme=light) removes the "dark" class from <body>.
# The image editor draws its canvas with WebGL on a fixed background (#27272a in dark mode) that CSS cannot reach;
# Gradio exposes the editor's PIXI app as globalThis.__PIXI_APP__, so the raised colour is set there.
FORCE_DARK_JS = f"""() => {{
    const dark = () => document.body.classList.contains("dark") || document.body.classList.add("dark");
    dark();
    new MutationObserver(dark).observe(document.body, {{attributes: true, attributeFilter: ["class"]}});
    let canvas_app = null;
    setInterval(() => {{
        const app = globalThis.__PIXI_APP__;
        if (app && app !== canvas_app && app.renderer) {{
            app.renderer.background.color = "{RAISED}";
            app.render();
            canvas_app = app;
        }}
    }}, 500);
}}"""

CSS_TOKENS = f"""
:root, .dark {{
    --asphalt: {ASPHALT}; --card: {CARD}; --raised: {RAISED}; --line: {LINE}; --text: {TEXT}; --muted: {MUTED};
    --blue: {BLUE}; --blue-hover: {BLUE_HOVER}; --blue-ink: {BLUE_INK}; --sign-blue: {SIGN_BLUE};
    --yellow: {YELLOW}; --red: {RED}; --white: {WHITE}; --plate-muted: {PLATE_MUTED}; --plate-line: {PLATE_LINE};
    --font-display: '{DISPLAY_FONT}', '{BODY_FONT}', sans-serif;
    color-scheme: dark;
}}
"""
CSS = CSS_TOKENS + """
body, gradio-app, .gradio-container { background: var(--asphalt) !important; color: var(--text); }
.gradio-container { width: 100% !important; max-width: 1360px !important; margin: 0 auto !important;
    padding: 0 24px !important; }
.gradio-container > .main { width: 100%; max-width: none !important; padding: 8px 0 40px !important; }
.gradio-container .html-container { padding: 0 !important; }
footer { display: none !important; }

/* Keyboard focus: one visible blue ring everywhere */
.gradio-container :is(button, a, input, select, textarea, summary, [tabindex]):focus-visible {
    outline: 3px solid var(--blue) !important; outline-offset: 2px; box-shadow: none !important;
}

/* Header */
.app-header { padding: 12px 0 4px; }
.app-header h1 { font-family: var(--font-display); font-weight: 700; font-size: 2.6rem; line-height: 1.05;
    letter-spacing: 0; color: var(--text); margin: 0 0 8px; }
.app-header p { margin: 0; font-size: 1.125rem; color: var(--muted); }

/* Tabs: blue text would be below AA on asphalt, so the selected tab is marked by the blue line under it */
.app-tabs > .tab-wrapper { margin-bottom: 20px; }
.app-tabs > .tab-wrapper button[role="tab"] { font-size: 1.0625rem; font-weight: 500; color: var(--muted);
    padding: 12px 18px; }
.app-tabs > .tab-wrapper button[role="tab"]:hover { color: var(--text); }
.app-tabs > .tab-wrapper button[role="tab"].selected { color: var(--text); }
/* the tab bar clips anything outside the buttons, so their focus ring is drawn inside */
.app-tabs > .tab-wrapper button[role="tab"]:focus-visible { outline-offset: -3px; }

/* Cards and panels: told apart from the page by their fill and a thin line, no shadows */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; }
.card-title { font-family: var(--font-display); font-weight: 600; font-size: 1.25rem; color: var(--text);
    margin: 0 0 10px; }
.input-panel { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
    padding: 18px 20px !important; gap: 14px !important; align-self: flex-start; }
.input-panel .block { border-color: var(--line) !important; }
.panel-title { font-family: var(--font-display); font-weight: 600; font-size: 1.25rem; margin: 0;
    color: var(--text); }
.panel-subtitle { font-family: var(--font-display); font-weight: 600; font-size: 1.0625rem; margin: 4px 0 0;
    color: var(--text); }
.hint { margin: 0; font-size: 0.9375rem; color: var(--muted); line-height: 1.45; }
.empty { padding: 48px 24px; text-align: center; color: var(--muted); font-size: 1.0625rem; background: var(--card);
    border: 1px dashed var(--line); border-radius: 8px; }
.footnote { margin: 8px 2px 0; font-size: 0.875rem; color: var(--muted); }

/* Result plate: the most important element of the page and the only light one, like a sign in the headlights */
.result { container-type: inline-size; }
/* the dark border sits inside a white rim, as on a real sign: on asphalt an outer dark border would vanish */
.plate { display: grid; grid-template-columns: 112px minmax(0, 1fr) auto; gap: 22px; align-items: center;
    background: var(--white); border: 3px solid var(--white); box-shadow: inset 0 0 0 3px var(--asphalt);
    border-radius: 12px; padding: 23px 27px; }
.plate-pict { width: 112px; height: 112px; max-width: none; object-fit: contain; display: block; }
.plate-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; color: var(--plate-muted);
    font-size: 0.9375rem; }
.sign-code { font-family: var(--font-display); font-weight: 700; font-size: 1.125rem; color: var(--asphalt);
    border: 2px solid var(--asphalt); border-radius: 4px; padding: 0 8px; line-height: 1.5; }
.plate-name { font-family: var(--font-display); font-weight: 700; font-size: 2.125rem; line-height: 1.1;
    color: var(--asphalt); margin-top: 8px; }
.plate-name.long { font-size: 1.625rem; }
.plate-conf { text-align: right; padding-left: 22px; border-left: 1px solid var(--plate-line); }
.plate-number { font-family: var(--font-display); font-weight: 700; font-size: 3rem; line-height: 1;
    color: var(--asphalt); font-variant-numeric: tabular-nums; white-space: nowrap; }
.plate-label { font-size: 0.9375rem; color: var(--plate-muted); margin-top: 6px; }

/* One status bar under the plate */
.status { margin-top: 12px; border-radius: 8px; padding: 14px 20px; }
.status-title { font-family: var(--font-display); font-weight: 700; font-size: 1.375rem; line-height: 1.2; }
.status-detail { display: flex; align-items: center; gap: 12px; margin-top: 4px; font-size: 1rem; line-height: 1.45; }
.status-detail strong { font-weight: 500; }
.status.ok { background: var(--sign-blue); color: var(--white); }
.status.warn { background: var(--yellow); color: var(--asphalt); }
.status.miss { background: var(--red); color: var(--white); }
.status-pict { width: 44px; height: 44px; object-fit: contain; flex: none; background: var(--white);
    border-radius: 6px; padding: 3px; }

/* Images: same size side by side */
.views { display: grid; grid-template-rows: auto auto; column-gap: 20px; row-gap: 8px; }
.views-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.views-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.view { margin: 0; display: grid; grid-row: span 2; grid-template-rows: subgrid; align-items: start; }
.view figcaption { display: flex; flex-direction: column; }
.view-title { font-family: var(--font-display); font-weight: 600; font-size: 1.125rem; color: var(--text); }
.view-note { font-size: 0.875rem; color: var(--muted); min-height: 1.3em; }
.view img { width: 100%; max-width: 320px; height: auto; aspect-ratio: 1; image-rendering: pixelated;
    border-radius: 4px; background: var(--raised); }

/* Charts */
.chart img { display: block; width: 100%; height: auto; }

/* Upload panel; the image editor is a raised element (its canvas background is set in FORCE_DARK_JS) */
.editor, .editor .empty, .editor .image-container.dark-bg { background: var(--raised) !important; }
.editor .empty, .editor .empty p { color: var(--muted) !important; }
.editor .crop-confirm-button button { background: var(--blue) !important; border-color: var(--blue) !important; }
.editor .crop-confirm-button button:hover { background: var(--blue-hover) !important;
    border-color: var(--blue-hover) !important; }
.editor .crop-confirm-button button, .editor .crop-confirm-button button * { color: var(--blue-ink) !important; }
.examples, .examples .grid-wrap { height: auto !important; max-height: none !important; min-height: 0 !important;
    overflow: visible !important; padding: 0 !important; }
.examples .grid-container { grid-template-columns: repeat(8, minmax(0, 1fr)) !important;
    grid-auto-rows: auto !important; height: auto !important; gap: 6px !important; }
.examples .gallery-item { height: auto !important; min-height: 0 !important; }
.examples button.thumbnail-item { width: 100% !important; height: auto !important; aspect-ratio: 1;
    border: 1px solid var(--line) !important; border-radius: 6px !important; box-shadow: none !important; }
.examples button.thumbnail-item:hover { border-color: var(--blue) !important; }
/* no "selected" look: after a random image or an upload it would point at the wrong thumbnail */
.examples button.thumbnail-item.selected:not(:hover) { border-color: var(--line) !important; }
.examples .caption-label { display: none !important; }

/* Robustness controls: one bar */
.control-bar { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px 20px;
    align-items: flex-end !important; gap: 24px !important; }
.control-bar .form { flex: 1 1 auto !important; gap: 24px !important; border: 0 !important; background: none !important;
    box-shadow: none !important; }
.control-bar .block { border: 0 !important; padding: 0 !important; background: none !important; }
.control-bar .segmented { flex: 0 1 auto !important; width: auto !important; max-width: 100%; }
.control-bar .level { flex: 1 1 220px !important; width: auto !important; }
.segmented .wrap { gap: 0 !important; flex-wrap: wrap !important; row-gap: 6px !important; }
.segmented .wrap label { margin: 0 !important; border-radius: 0 !important; border: 1.5px solid var(--line) !important;
    margin-left: -1.5px !important; padding: 9px 14px !important; }
.segmented .wrap label:first-child { border-radius: 6px 0 0 6px !important; margin-left: 0 !important; }
.segmented .wrap label:last-child { border-radius: 0 6px 6px 0 !important; }
.segmented .wrap label.selected { border-color: var(--blue) !important; position: relative; z-index: 1; }
.segmented .wrap label input { position: absolute; opacity: 0; width: 1px; height: 1px; }
.segmented .wrap label:has(input:focus-visible) { outline: 3px solid var(--blue); outline-offset: 2px; z-index: 2; }
.level .tab-like-container, .level input[type="number"] { display: none !important; }
.level input[type="range"] { -webkit-appearance: none; appearance: none; height: 22px !important;
    background: transparent !important; }
.level input[type="range"]::-webkit-slider-runnable-track { height: 6px !important; border-radius: 3px;
    border: 0 !important; box-shadow: none !important;
    background: linear-gradient(to right, var(--blue) var(--range_progress),
                                var(--line) var(--range_progress)) !important; }
.level input[type="range"]::-moz-range-track { height: 6px; border-radius: 3px; border: 0; background: var(--line); }
.level input[type="range"]::-moz-range-progress { height: 6px; border-radius: 3px; background: var(--blue); }
.level input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px !important;
    height: 22px !important; margin-top: -8px !important; border-radius: 50%; background: var(--card) !important;
    border: 3px solid var(--blue) !important; box-shadow: none !important; }
.level input[type="range"]::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: var(--card);
    border: 3px solid var(--blue); }

/* Results tab */
.results { display: flex; flex-direction: column; gap: 20px; }
.results .lead { font-size: 1.1875rem; line-height: 1.55; margin: 0; max-width: 980px; color: var(--text); }
.results-section { display: flex; flex-direction: column; gap: 16px; }
.results-section > h2 { font-family: var(--font-display); font-weight: 700; font-size: 1.75rem; margin: 20px 0 0;
    padding-bottom: 8px; border-bottom: 2px solid var(--line); color: var(--text); }
.results-section > p { margin: 0; max-width: 980px; color: var(--text); line-height: 1.5; }
.result-figure { margin: 0; }
.result-figure p { margin: 0 0 12px; color: var(--muted); line-height: 1.5; max-width: 980px; }
/* the static plots have a white background: shown on a white board, like a printout pinned to a board */
.result-figure img { display: block; width: 100%; height: auto; box-sizing: border-box; padding: 12px;
    background: var(--white); border-radius: 4px; }
details.card summary { cursor: pointer; font-family: var(--font-display); font-weight: 600; font-size: 1.125rem;
    color: var(--text); }
details.card[open] { display: flex; flex-direction: column; gap: 16px; }
details.card .result-figure { border: 0; padding: 0; }
.matrices { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.matrices .card-title { font-size: 1.0625rem; }
.table-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.9375rem; font-variant-numeric: tabular-nums;
    color: var(--text); }
.data-table th { font-weight: 500; color: var(--muted); text-align: left; padding: 10px 12px; white-space: nowrap;
    border-bottom: 2px solid var(--line); vertical-align: bottom; background: var(--card); }
.data-table th.group { text-align: center; border-bottom: 1px solid var(--line); }
.data-table td { padding: 8px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.data-table tr:last-child td { border-bottom: 0; }
.data-table .num { text-align: right; }
.data-table tr.final td { background: var(--raised); font-weight: 500; }
.data-table tr.final td:first-child { box-shadow: inset 4px 0 0 var(--blue); }
.data-table .role { display: block; font-size: 0.8125rem; color: var(--muted); font-weight: 400; }
.data-table.comparison td.test { font-weight: 500; }

/* About tab */
.about > * { max-width: 820px; }
.about h2 { font-family: var(--font-display); font-weight: 700; font-size: 1.75rem; margin: 8px 0 10px; }
.about p, .about li { font-size: 1.0625rem; line-height: 1.55; }
.about ul { list-style: disc; }
/* links in the text colour, marked by a blue underline (blue text would be below AA) */
.about a { color: var(--text) !important; text-decoration: underline; text-decoration-color: var(--blue);
    text-decoration-thickness: 2px; text-underline-offset: 3px; }
.about a:hover { text-decoration-color: var(--blue-hover); }
.about .device { font-size: 0.875rem; color: var(--muted); margin-top: 28px; }
"""
# ?__theme=dark makes Gradio itself use dark mode (also the image editor canvas, which ignores the CSS).
# Gradio scopes the css= rules and drops @container blocks, so the responsive rules go to <head> as they are
# (with !important against the higher specificity of the scoped rules).
HEAD = """<meta name="color-scheme" content="dark">
<script>
(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "dark") {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url.href);
    }
})();
</script>
<style>
@container (max-width: 640px) {
    .plate { grid-template-columns: 88px minmax(0, 1fr) !important; gap: 16px 20px !important; }
    .plate-pict { width: 88px !important; height: 88px !important; }
    .plate-name, .plate-name.long { font-size: 1.625rem !important; }
    .plate-conf { grid-column: 1 / -1; display: flex; align-items: baseline; gap: 12px; text-align: left !important;
        border-left: 0 !important; padding-left: 0 !important; border-top: 1px solid var(--plate-line);
        padding-top: 12px; }
    .plate-number { font-size: 2.5rem !important; }
    .plate-label { margin-top: 0 !important; }
}
@media (max-width: 900px) { .matrices { grid-template-columns: 1fr !important; } }
@media (max-width: 640px) {
    .views-3 { grid-template-columns: 1fr 1fr !important; }
    .app-header h1 { font-size: 2rem !important; }
}
</style>"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

HEADER = ('<header class="app-header"><h1>Rozpoznawanie znaków drogowych</h1>'
          '<p>Model ResNet18 rozpoznaje 43 rodzaje znaków ze zbioru GTSRB i pokazuje, na co patrzył.</p></header>')


def build_app(rec: Recognizer, tests: TestImages, device_label: str) -> gr.Blocks:
    no_image = empty_html("Wgraj zdjęcie znaku, kliknij przykład albo wylosuj zdjęcie testowe.")

    def recognize(sample: Sample | None):
        """Tab 'Rozpoznaj': result plate, input image and Grad-CAM, top-5."""
        if sample is None:
            return no_image, "", ""
        img = sample.clean
        probs, cam = rec.explain(img)
        figures = [figure_html(view(as_hwc(img)), "Obraz wejściowy", "64×64 px, tak jak widzi go model",
                               "Obraz wejściowy modelu"),
                   figure_html(cam_view(img, cam, sample), "Na co patrzył model", f"Grad-CAM: {CAM_NOTE}",
                               "Mapa Grad-CAM nałożona na obraz")]
        return (result_html(probs, rec.threshold, sample), views_html(figures, sample),
                chart_html("Pięć najbardziej prawdopodobnych klas", top5_figure(probs, rec.threshold, sample.label),
                           top5_alt(probs)))

    def robustness(sample: Sample | None, ctype: str, level: int):
        """Tab 'Test odporności': clean and corrupted image, Grad-CAM, result plate, confidence vs level."""
        if sample is None:
            return "", no_image, ""
        level = int(level)
        images = torch.stack([sample.corrupted(ctype, lv) for lv in APP_LEVELS])
        img = images[level]
        probs, cam = rec.explain(img)
        curve = rec.probs(images)
        curve[level] = probs  # the plate and the plot show the very same prediction
        ref = sample.label if sample.label is not None else int(curve[0].argmax())
        name = CORRUPTION_LABELS_PL[ctype]
        condition = ("poziom 0, bez zakłóceń" if level == 0 else
                     f"{name}, poziom {level} ({level_label_pl(ctype, level)})")
        if ctype == "jpeg":
            condition += "; tego zakłócenia nie było w treningu"
        figures = [figure_html(view(as_hwc(sample.clean)), "Oryginał", "to samo zdjęcie co w zakładce Rozpoznaj",
                               "Oryginalny obraz 64×64 px"),
                   figure_html(view(as_hwc(img)), "Po zakłóceniu", condition, f"Obraz po zakłóceniu: {condition}"),
                   figure_html(cam_view(img, cam, sample), "Na co patrzył model", f"Grad-CAM: {CAM_NOTE}",
                               "Mapa Grad-CAM nałożona na obraz po zakłóceniu")]
        chart = robustness_figure(curve, ctype, level, rec.threshold, ref, sample.label is not None)
        return (views_html(figures, sample), result_html(probs, rec.threshold, sample),
                chart_html(f"Pewność przy kolejnych poziomach: {lower_first(name)}", chart,
                           robustness_alt(curve, ctype)))

    def on_upload(value: dict | None):
        img = (value or {}).get("composite")
        if img is None:  # cleared editor (also after a test image)
            return (gr.skip(),) * 4
        sample = upload_sample(img)
        return sample, *recognize(sample)

    def on_random():
        if not tests.available:
            raise gr.Error("Brak zbioru testowego w data/ – wgraj własne zdjęcie.")
        sample = tests.random()
        return sample, None, *recognize(sample)

    def on_example(evt: gr.SelectData):
        sample = tests.sample(EXAMPLE_TEST_IMAGES[evt.index])
        return sample, None, *recognize(sample)

    gr.set_static_paths([RESULTS_DIR])
    with gr.Blocks(title="Rozpoznawanie znaków drogowych") as demo:
        html_block(HEADER)
        sample_state = gr.State(None)
        with gr.Tabs(elem_classes="app-tabs"):
            with gr.Tab("Rozpoznaj"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340, elem_classes="input-panel"):
                        html_block('<h2 class="panel-title">Zdjęcie znaku</h2>')
                        # Brush and eraser stay enabled: in Gradio 6.30 brush=False or eraser=False breaks the
                        # editor toolbar after an upload (Svelte props_invalid_value), and the crop tool with it.
                        editor = gr.ImageEditor(type="pil", image_mode="RGB", sources=("upload", "clipboard"),
                                                transforms=("crop",), layers=False, format="png",
                                                label="Zdjęcie znaku", show_label=False, height=220,
                                                elem_classes="editor",
                                                placeholder="Upuść tu zdjęcie znaku\nlub kliknij, aby wybrać plik")
                        html_block('<p class="hint">Znak powinien wypełniać większość kadru, więc zdjęcie całej '
                                   'ulicy przytnij: na pasku edytora wybierz ikonę obrazu, a potem kadrowanie.</p>')
                        random_1 = gr.Button("Losuj zdjęcie testowe", variant="primary",
                                             interactive=tests.available)
                        html_block('<h3 class="panel-subtitle">Przykładowe zdjęcia testowe</h3>',
                                   visible=tests.available)
                        examples = gr.Gallery([tests.thumbnail(i) for i in EXAMPLE_TEST_IMAGES]
                                              if tests.available else None,
                                              show_label=False, container=False, columns=8, height="auto",
                                              allow_preview=False, object_fit="cover", format="png", buttons=[],
                                              interactive=False, elem_classes="examples", visible=tests.available)
                    with gr.Column(scale=8, min_width=420):
                        card_1 = html_block(no_image)
                        views_1 = html_block()
                        top5_1 = html_block()

            with gr.Tab("Test odporności"):
                with gr.Row(elem_classes="control-bar"):
                    choices = [(CORRUPTION_LABELS_PL[c], c) for c in CORRUPTIONS]
                    ctype_radio = gr.Radio(choices, value=CORRUPTIONS[0], label="Typ zakłócenia", scale=6,
                                           elem_classes="segmented")
                    level_slider = gr.Slider(0, 3, value=2, step=1, label="Poziom zakłócenia", scale=3,
                                             info="0 to brak zakłóceń, 3 to zakłócenie silne", buttons=[],
                                             elem_classes="level")
                    random_2 = gr.Button("Losuj zdjęcie testowe", interactive=tests.available, scale=0,
                                         min_width=220)
                views_2 = html_block()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=360):
                        card_2 = html_block(no_image)
                    with gr.Column(scale=6, min_width=420):
                        level_plot = html_block()

            with gr.Tab("Wyniki"):
                html_block(results_html())

            with gr.Tab("O projekcie"):
                gr.Markdown(about_markdown(device_label), elem_classes="about")

        tab_1 = [card_1, views_1, top5_1]
        tab_2 = [views_2, card_2, level_plot]
        robustness_args = dict(fn=robustness, inputs=[sample_state, ctype_radio, level_slider], outputs=tab_2)
        editor.change(on_upload, editor, [sample_state, *tab_1], api_name="wgraj").then(**robustness_args,
                                                                                 api_visibility="private")
        for button, visibility in ((random_1, "public"), (random_2, "private")):
            button.click(on_random, None, [sample_state, editor, *tab_1], api_name="losuj", api_visibility=visibility
                         ).then(**robustness_args, api_visibility="private")
        examples.select(on_example, None, [sample_state, editor, *tab_1], api_visibility="private").then(
            **robustness_args, api_visibility="private")
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
    demo.launch(server_port=args.port, share=args.share, theme=sign_theme(), css=CSS, js=FORCE_DARK_JS,
                head=HEAD, footer_links=[])


if __name__ == "__main__":
    main()
