# GTSRB_robustness

Rozpoznawanie znaków drogowych z wykorzystaniem konwolucyjnych sieci neuronowych oraz analiza odporności modeli
na zakłócenia obrazu (rozmycie, szum, ciemność, zasłonięcie, kompresja JPEG).

**Pytanie badawcze:** czy uczenie transferowe i odpowiednia augmentacja ograniczają spadek skuteczności klasyfikatora
znaków na zakłóconych zdjęciach?

Porównywane modele: własna sieć `simple_cnn` (baseline), `resnet18` i `mobilenet_v3_small` (od zera i wstępnie
wytrenowane na ImageNet), z trzema wariantami augmentacji: A0 (brak), A1 (geometryczna), A2 (odpornościowa).
Wejście wszystkich modeli: 64×64 px RGB, normalizacja ImageNet.

| Etap | Zakres | Stan |
|------|--------|------|
| KM1 | analiza zbioru, podział danych, plan eksperymentów | dane, podział, EDA – gotowe |
| KM2 | implementacja, trening i porównanie wariantów | kod gotowy i sprawdzony smoke-testami, treningi – |
| KM3 | ocena na zbiorze testowym, XAI (Grad-CAM), aplikacja Gradio | – |

## Dane

Zbiór **GTSRB – German Traffic Sign Recognition Benchmark** z Kaggle:
<https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>

Pobierz archiwum (przez stronę albo `kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`)
i rozpakuj je do katalogu `data/` w głównym folderze repozytorium, tak żeby powstała struktura:

```
data/
├── Train/<ClassId>/<klasa>_<seria>_<klatka>.png   # 39 209 zdjęć, 43 klasy (0–42)
├── Test/<nnnnn>.png                               # 12 630 zdjęć
├── Meta/<ClassId>.png                             # wzorcowe piktogramy klas
├── Train.csv
├── Test.csv
└── Meta.csv
```

Katalog `data/` jest w `.gitignore` i nie trafia do repozytorium.

## Środowisko

Python 3.12, Windows (PowerShell). Polecenia uruchamiaj w głównym katalogu repozytorium.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Najpierw zainstaluj **PyTorch**, w wersji dopasowanej do sprzętu. Sprawdź `nvidia-smi`:

- **Karta NVIDIA** (wybór w tym projekcie: RTX 4070 Laptop, sterownik z obsługą CUDA 13.4, build CUDA 13.2):
  ```powershell
  pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
  ```
- **Brak karty NVIDIA** – wersja CPU:
  ```powershell
  pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
  ```

Potem pozostałe biblioteki:

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Na karcie NVIDIA ostatnie polecenie powinno wypisać `2.14.1+cu132 True`.

## Podział train/val

Walidacja to ok. 20% **serii** (tracków) z `Train/`, a nie losowych zdjęć: jedna seria to ok. 30 niemal identycznych
klatek tego samego znaku, więc losowy podział zdjęć dałby przeciek między train i val. Podział robi
`StratifiedGroupKFold` (stratyfikacja po klasie, grupy = id serii, seed 42), każda klasa ma co najmniej 1 serię w walidacji.

Podział jest już zapisany w [`splits/val_tracks.json`](splits/val_tracks.json) i wszystkie skrypty go wczytują.
Sprawdzenie (oraz utworzenie pliku, jeśli go brakuje):

```powershell
python -m src.data           # wypisuje liczności i kontrole podziału
python -m src.data --force   # tworzy podział od nowa (wynik jest deterministyczny)
```

Wynik: train 31 349 zdjęć / 1 045 serii, val 7 860 zdjęć / 262 serie, 0 wspólnych serii, min. 30 zdjęć walidacyjnych na klasę.

**Zbiór testowy** (`Test/`) jest używany tylko raz, w KM3, do oceny finalnych modeli – nigdy do wyboru modelu ani strojenia.

## Analiza danych (EDA)

Notatnik [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (zapisany z wynikami). Wykresy trafiają do
[`results/eda/`](results/eda/), kluczowe liczby do `results/eda/summary.json`, liczności klas do `results/eda/class_counts.csv`.

Ponowne uruchomienie:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Eksperymenty (KM2)

Cały plan E1–E4 jednym poleceniem (przerwany da się wznowić tym samym poleceniem – gotowe runy są pomijane):

```powershell
python -m src.run_experiments --stage all     # albo --stage e1 / e2 / e3 / e4
python -m src.summarize                       # tabela i wykresy -> results/summary/
```

Pojedynczy run i jego ewaluacja:

```powershell
python -m src.train --config configs/resnet18_pretrained_A0.yaml
python -m src.evaluate --config configs/resnet18_pretrained_A0.yaml
```

Szybki test (1 epoka, ok. 2000 zdjęć treningowych i 500 walidacyjnych, wyniki w `results/_smoke/`, poza gitem):
flaga `--smoke` w każdym z powyższych poleceń, np. `python -m src.run_experiments --stage all --smoke`,
a potem `python -m src.summarize --results-dir results/_smoke`.

- `python -m src.models` – liczba parametrów i rozmiar ostatniej mapy cech każdego modelu.
  ResNet18 i MobileNetV3-Small są dostosowane do wejścia 64×64: w ResNet18 pierwsza konwolucja ma stride 1
  i nie ma max poolingu (mapa 8×8), w MobileNetV3-Small pierwsza konwolucja ma stride 1 (mapa 4×4).
  Zmieniają się tylko kroki, więc wagi ImageNet wczytują się bez zmian.
- `python -m src.corruptions` – kontrola zakłóceń i podgląd `results/corruptions/preview.png`.
- Zbiór testowy jest zablokowany: `evaluate --split test` wymaga dodatkowo `--i-know-this-is-final` (KM3).

Wyniki runu w `results/<run>/`: `config.json`, `history.csv`, `train_summary.json`, `metrics.json`,
`per_class_f1.csv`, `confusion_matrix.png`, `best.pt` (poza gitem). Wybory etapów: `results/e2_selection.json`,
`results/e4_selection.json`, `results/e4_summary.json`.

## Struktura repozytorium

```
├── configs/          # konfiguracje YAML runów (KM2)
├── splits/           # val_tracks.json – podział train/val
├── notebooks/        # 01_eda.ipynb
├── src/
│   ├── data.py       # wczytywanie CSV, podział, Dataset, augmentacje A0/A1/A2
│   ├── models.py     # simple_cnn, resnet18, mobilenet_v3_small (64×64)
│   ├── corruptions.py  # 5 zakłóceń × 3 poziomy, deterministyczne
│   ├── train.py, evaluate.py, run_experiments.py, summarize.py   # KM2
│   ├── xai.py        # KM3
├── results/          # wyniki runów (JSON/CSV/PNG), results/eda/ – wykresy EDA
├── app/app.py        # aplikacja Gradio (KM3)
└── docs/             # sprawozdania KM1–KM3
```

## Uwagi praktyczne

- Augmentacje nie zawierają odbić lustrzanych (zmieniają znaczenie znaków, np. „nakaz skrętu w prawo/lewo”) ani
  kompresji JPEG (zakłócenie testowane wyłącznie w ewaluacji).
- `GTSRBDataset(..., cache=True)` dekoduje i skaluje wszystkie zdjęcia raz, do pamięci (ok. 370 MB dla train).
  Wynik jest identyczny jak bez cache, a epoka szybsza (zmierzone: A0 ok. 13 s zamiast ok. 70 s, A2 ok. 19 s zamiast
  ok. 30 s). Na Windows start każdego workera DataLoadera trwa ok. 10 s, więc dla A0 najszybciej jest `num_workers=0`,
  a dla A1/A2 `num_workers=4` z trwałymi workerami (`make_loader`).
