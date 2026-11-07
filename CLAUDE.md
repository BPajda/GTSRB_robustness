# GTSRB_robustness – kontekst projektu dla asystenta kodu

Klasyfikacja znaków drogowych (GTSRB, 43 klasy) sieciami konwolucyjnymi i analiza ich odporności na zakłócenia obrazu: rozmycie, szum, przyciemnienie, zasłonięcie i kompresję JPEG. Projekt jest zakończony. Opis wyników jest w `README.md`, a pełne sprawozdanie w `reports/GTSRB_robustness_report_PL.pdf`.

## Dane

Zbiór GTSRB z Kaggle, rozpakowany do `data/` (poza gitem):

```
data/
├── Train/<ClassId>/<class>_<track>_<frame>.png   # 39 209 zdjęć, 43 klasy (0–42)
├── Test/<nnnnn>.png                              # 12 630 zdjęć
├── Meta/<ClassId>.png                            # wzorcowe piktogramy klas
├── Train.csv   # Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
├── Test.csv    # te same kolumny
└── Meta.csv
```

- `Path` w CSV jest względna wobec `data/`.
- Zdjęcia treningowe tworzą serie (tracki): `<class>_<track>` to jeden fizyczny znak, ok. 30 niemal identycznych klatek, razem 1 307 serii.
- Ramki `Roi.*` są używane w zakłóceniu „occlusion” i w analizie Grad-CAM, po przeskalowaniu do 64×64.

## Niezmienniki

1. Zbiór testowy służy wyłącznie do jednorazowej oceny końcowej (`src/final_eval.py`). Ewaluacja na nim wymaga flagi `--i-know-this-is-final`. Nie używać go do wyboru modelu ani strojenia.
2. Walidacja jest wydzielana z `Train/` po seriach (`StratifiedGroupKFold`, seed 42) i zapisana w `splits/val_tracks.json`. Wszystkie skrypty czytają ten plik.
3. Augmentacje nie zawierają odbić lustrzanych (zmieniają znaczenie znaków) ani kompresji JPEG (zakłócenie testowe).
4. Wejście wszystkich modeli: 64×64 px, RGB, normalizacja ImageNet.
5. Seed 42 wszędzie, a zakłócenia są deterministyczne (`corruption_rng(seed, index, typ, poziom)`).
6. `data/` i wagi (`*.pt`) nie trafiają do repozytorium. Wyniki (JSON/CSV/PNG) są w repo.

## Eksperymenty

- Modele (`src/models.py`): `simple_cnn` (baseline), `resnet18`, `mobilenet_v3_small`, od zera albo z wagami ImageNet. ResNet18 i MobileNetV3 mają stride 1 w pierwszej konwolucji (ResNet18 także bez max poolingu), żeby ostatnia mapa cech miała 8×8 / 4×4.
- Augmentacje (`src/data.py`): `A0` brak, `A1` geometryczna, `A2` odpornościowa (A1 + rozmycie, przyciemnienie, szum, RandomErasing).
- Zakłócenia (`src/corruptions.py`), 3 poziomy:

| typ | poziom 1 | poziom 2 | poziom 3 |
| --- | --- | --- | --- |
| gaussian_blur (sigma) | 0.8 | 2.0 | 3.0 |
| gaussian_noise (sigma, obraz w [0,1]) | 0.03 | 0.06 | 0.10 |
| darken (mnożnik jasności) | 0.6 | 0.4 | 0.25 |
| occlusion (% pola ROI, wewnątrz ROI) | 10% | 20% | 30% |
| jpeg (quality) | 30 | 15 | 8 |

- Etapy (`src/run_experiments.py --stage e1…e5|all`): E1 architektura i transfer (A0), E2 augmentacja A1/A2, E3 ocena wszystkich 11 modeli na zakłóconej walidacji, E4 zwycięzca na 3 seedach, E5 strategie fine-tuningu zwycięzcy. Regułę wyboru zwycięzcy stosuje `select_e4()` i zapisuje uzasadnienie w `results/e4_selection.json`.
- Finalny model: `resnet18_pretrained_A2` (seed 42). Aplikacja i XAI korzystają z tego runu.
- Trening: AdamW, cosine, maks. 20 epok, early stopping na macro-F1 walidacji (patience 5), bf16 + channels_last na GPU.
- Fine-tuning (klucze configu, domyślne w `DEFAULTS` w `src/train.py`): `finetune` = `full` (domyślnie, cała sieć z jednym lr), `linear` (tylko głowica, BatchNorm backbone'u w trybie eval), `discriminative` (backbone `backbone_lr`, głowica `lr`), `two_stage` (`freeze_epochs` epok jak `linear`, potem jak `discriminative`). Strategie inne niż `full` tylko dla modeli pretrenowanych. Wariant `full` musi dawać dokładnie te same wyniki co wcześniej.
- Metryki: accuracy, macro-F1, mCA (średnia accuracy po 15 zakłóceniach), względny spadek, liczba parametrów, czas predykcji na CPU (4 wątki, mediana ze 100 powtórzeń).

## Struktura

```
src/        data, models, corruptions, train, evaluate, run_experiments, summarize, final_eval, xai, class_names_pl
app/        app.py (Gradio, interfejs po polsku), assets/fonts (OFL)
configs/    YAML na każdy trening
splits/     val_tracks.json
notebooks/  01_eda.ipynb
results/    <run>/ (config.json, history.csv, metrics.json, per_class_f1.csv, confusion_matrix.png, best.pt poza gitem),
            summary/, final_test/, xai/, eda/, corruptions/, e*_selection.json, e*_summary.json
reports/    sprawozdania PDF (pełne + kamienie milowe)
docs/img/   zrzuty ekranu aplikacji do README
```

## Środowisko

- Python 3.12, PyTorch 2.14 (CUDA 13.2 albo CPU), biblioteki w `requirements.txt`.
- Rozwijane na Windowsie, a ścieżka projektu może zawierać spacje: zawsze `pathlib.Path`. Kod z DataLoaderem pod `if __name__ == "__main__":`.
- Skrypty treningu i ewaluacji (`train`, `evaluate`, `run_experiments`, `final_eval`, `xai`) mają tryb `--smoke` (1 epoka, mały podzbiór, wyniki w `results/_smoke/`); `summarize` czyta te wyniki przez `--results-dir results/_smoke`. Zmiany w kodzie treningu i ewaluacji sprawdzaj nim przed pełnym uruchomieniem. Długich treningów nie uruchamiaj bez prośby, podaj gotowe polecenie.

## Konwencje

- Kod, nazwy i komentarze po angielsku. README, sprawozdania i interfejs aplikacji po polsku.
- Commity wykonuje autor repozytorium. Asystent nie uruchamia `git commit`, `git push`, `git reset`, `git rebase` ani innych poleceń zmieniających historię. Może czytać stan (`git status`, `git diff`, `git log`) i zaproponować podział zmian na commity.
- Komunikaty commitów: Conventional Commits, po angielsku, tryb rozkazujący, do ~72 znaków (np. `feat(data): add track-based train/val split`). Typy: `feat`, `fix`, `refactor`, `docs`, `chore`, `build`, `test`, `exp` (wyniki eksperymentów). Bez stopek typu `Co-Authored-By`.
