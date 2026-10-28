# GTSRB_robustness – kontekst projektu

Projekt zaliczeniowy z przedmiotu „Sztuczna inteligencja w systemach informatycznych”.
Temat: **Rozpoznawanie znaków drogowych z wykorzystaniem konwolucyjnych sieci neuronowych oraz analiza odporności modeli na zakłócenia obrazu.**

Pytanie badawcze: czy uczenie transferowe i odpowiednia augmentacja ograniczają spadek skuteczności klasyfikatora znaków na zakłóconych zdjęciach (rozmycie, szum, ciemność, zasłonięcie, kompresja JPEG)?

Projekt ma 3 kamienie milowe (KM), każdy z osobnym sprawozdaniem w `docs/`:
- **KM1** – problem, analiza datasetu, podział danych, plan eksperymentów (hipotezy, baseline, metryki)
- **KM2** – implementacja, trening i porównanie wariantów (etapy E1–E4)
- **KM3** – ocena na zbiorze testowym, XAI (Grad-CAM), aplikacja Gradio

Podział pracy: Claude Code pisze kod i robi szybkie testy. Długie treningi uruchamia użytkownik. Analiza wyników i sprawozdania powstają osobno, na podstawie plików w `results/`.

## Dane (zweryfikowane)

Dataset GTSRB z Kaggle, rozpakowany do `data/` (folder poza gitem):

```
data/
├── Train/<ClassId>/<class>_<track>_<frame>.png   # 39 209 zdjęć, 43 klasy (0–42)
├── Test/<nnnnn>.png                              # 12 630 zdjęć
├── Meta/<ClassId>.png                            # wzorcowe piktogramy klas
├── Train.csv   # Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
├── Test.csv    # te same kolumny, etykiety testu są w ClassId
└── Meta.csv    # Path,ClassId,ShapeId,ColorId,SignId
```

- Kolumna `Path` jest względna wobec `data/` (np. `Train/20/00020_00000_00000.png`).
- **Serie (tracki):** nazwa pliku treningowego to `<class>_<track>_<frame>`. Jest 1 307 serii po ok. 30 niemal identycznych klatek tego samego fizycznego znaku. Identyfikator serii = `<class>_<track>`.
- Nierównowaga klas: od 210 zdjęć (klasa 0, 7 serii) do 2 250 (klasa 2, 75 serii).
- Rozmiary: od 25 do ok. 250 px, mediana 43 px.
- Ramki `Roi.*` (położenie znaku na zdjęciu) są w Train i Test. Używamy ich w XAI (KM3), po przeskalowaniu do rozmiaru wejścia.
- W `data/Meta/` są śmieciowe pliki `.~lock.*.ods#` z archiwum – ignorować.

## Zasady, których nie wolno łamać

1. **Zbiór testowy (`Test/`) jest używany tylko raz, w KM3**, do oceny finalnego modelu i baseline'u. Nigdy do wyboru modelu, early stoppingu ani strojenia.
2. **Walidacja jest wydzielana z `Train/` po seriach, nie po zdjęciach**: ok. 20% serii, stratyfikowane po klasie (`StratifiedGroupKFold`, groups = id serii), seed 42. Każda klasa musi mieć co najmniej 1 serię w walidacji. Podział zapisany raz do `splits/val_tracks.json` (w repo) i wczytywany przez wszystkie skrypty.
3. **Żadnych odbić lustrzanych** (horizontal/vertical flip) w augmentacji – zmieniają znaczenie znaków (np. nakaz skrętu w lewo/prawo).
4. Wejście sieci: **64×64 px, RGB, normalizacja ImageNet** – identycznie dla wszystkich modeli, żeby porównanie było uczciwe.
5. Seed 42 wszędzie (Python, NumPy, PyTorch, DataLoader). Zakłócenia do ewaluacji są deterministyczne – każdy model widzi identyczne zakłócone obrazy.
6. Nigdy nie commitować `data/` ani wag (`*.pt`). Wyniki (JSON/CSV/PNG) commitujemy.

## Plan eksperymentów (KM2)

Modele (`src/models.py`):
- `simple_cnn` – własny baseline: 3 bloki (Conv-BN-ReLU-MaxPool) + klasyfikator, tylko od zera
- `resnet18` – torchvision, `scratch` lub `pretrained` (ImageNet)
- `mobilenet_v3_small` – torchvision, `scratch` lub `pretrained`

Augmentacje (`src/data.py`):
- `A0` – brak (resize + normalizacja)
- `A1` – geometryczna: obrót ±10°, przesunięcie/skala (RandomAffine), ColorJitter jasność/kontrast
- `A2` – odpornościowa: A1 + GaussianBlur, szum Gaussa, przyciemnienie, RandomErasing

Zakłócenia do ewaluacji (`src/corruptions.py`), 3 poziomy nasilenia (1 = słabe, 2 = średnie, 3 = silne):
`gaussian_blur`, `gaussian_noise`, `darken`, `occlusion`, `jpeg`. **JPEG nigdy nie występuje w augmentacji treningowej** (test uogólnienia, hipoteza H4).

Etapy:
- **E1**: simple_cnn + 4 warianty (resnet18/mobilenet × scratch/pretrained), wszystkie A0 → 5 runów
- **E2**: simple_cnn + najlepszy resnet18 + najlepszy mobilenet z E1, każdy z A1 i A2 → 6 runów
- **E3**: ewaluacja wszystkich 11 modeli na zakłóconej walidacji (bez treningu)
- **E4**: zwycięzca, 3 seedy (42, 43, 44)

Trening: AdamW, maks. 20 epok, early stopping na macro-F1 walidacji (patience 5), zapis najlepszych wag.

Metryki: accuracy, macro-F1, mCA (średnia accuracy po 5 zakłóceniach × 3 poziomach), względny spadek `(acc_clean − mCA) / acc_clean`, liczba parametrów, czas predykcji 1 zdjęcia na CPU (ms, mediana ze 100 powtórzeń).

## Struktura repo

```
GTSRB_robustness/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/            # YAML na każdy run
├── splits/             # val_tracks.json
├── notebooks/          # 01_eda.ipynb
├── src/
│   ├── data.py         # Dataset, podział, augmentacje
│   ├── corruptions.py
│   ├── models.py
│   ├── train.py        # python -m src.train --config configs/<run>.yaml
│   ├── evaluate.py     # czyste + zakłócone metryki
│   ├── summarize.py    # tabela zbiorcza i wykresy porównawcze
│   └── xai.py          # Grad-CAM, wskaźnik energii w ROI
├── results/<run_name>/ # config.json, history.csv, metrics.json, confusion_matrix.png
├── app/app.py          # Gradio
└── docs/               # sprawozdania KM1, KM2, KM3
```

## Format wyników każdego runu

`results/<run_name>/`:
- `config.json` – pełna konfiguracja
- `history.csv` – epoch, train_loss, val_loss, val_acc, val_macro_f1, czas epoki
- `metrics.json` – `{"clean": {"acc", "macro_f1"}, "corrupted": {"<typ>": {"1": acc, "2": acc, "3": acc}}, "mca", "relative_drop", "params", "cpu_latency_ms", "best_epoch", "train_time_s"}`
- `confusion_matrix.png`, `per_class_f1.csv`
- `best.pt` – wagi (poza gitem)

## Środowisko

- Windows, ścieżka projektu zawiera spacje – zawsze `pathlib.Path`, nigdy sklejanie stringów.
- Wirtualne środowisko `.venv` w katalogu repo. Przed instalacją PyTorcha sprawdź `nvidia-smi`: jest karta NVIDIA → build z CUDA ze strony pytorch.org; brak → wersja CPU. Zapisz wybór w README.
- DataLoader na Windows: kod uruchamiany pod `if __name__ == "__main__":`, `num_workers` 2–4, przy problemach 0.
- Biblioteki: torch, torchvision, numpy, pandas, scikit-learn, matplotlib, seaborn, pyyaml, tqdm, pillow, grad-cam (pytorch-grad-cam), gradio, jupyter.

## Sposób pracy

- Kod, nazwy i komentarze w kodzie po angielsku. Sprawozdania i README po polsku.
- Zanim użytkownik puści pełny trening: każdy skrypt sprawdzony szybkim testem (1 epoka, mały podzbiór, flaga `--smoke`).
- Długich treningów nie uruchamiaj sam bez prośby – podaj gotowe polecenie.
- Małe, opisowe commity. Po każdym etapie krótko podsumuj, co powstało i jak to uruchomić.
