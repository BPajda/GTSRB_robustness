# Plan projektu: rozpoznawanie znaków drogowych i odporność modeli (GTSRB)

## W skrócie

Projekt odpowiada na pytanie: **czy uczenie transferowe i odpowiednio dobrana augmentacja sprawiają, że klasyfikator znaków drogowych nie traci skuteczności w złych warunkach (rozmycie, szum, ciemność, zasłonięcie, kompresja)?** Odpowiedź opiera się na liczbach: spadku skuteczności na celowo zakłóconych zdjęciach, porównanym między wariantami modeli.

| | |
| --- | --- |
| Temat | Rozpoznawanie znaków drogowych z wykorzystaniem konwolucyjnych sieci neuronowych oraz analiza odporności modeli na zakłócenia obrazu |
| Dataset | GTSRB: 43 klasy, 39 209 zdjęć treningowych, 12 630 testowych ([Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), [artykuł](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)) |
| Modele | własna prosta CNN (baseline), ResNet18, MobileNetV3-Small |
| Stack | Python, PyTorch z torchvision, pytorch-grad-cam, scikit-learn, matplotlib, Gradio |
| Sprzęt | laptop z RTX 4070 Laptop (8 GB), wejście 64×64 px, epoka około 13 do 19 s |

Projekt jest podzielony na trzy kamienie milowe. Każdy kończy się osobnym sprawozdaniem w katalogu `docs/`.

## KM1: problem, dane i plan eksperymentu

Status: **zakończony**, szczegóły w [KM1_sprawozdanie.md](KM1_sprawozdanie.md).

- cel projektu i pytanie badawcze,
- analiza eksploracyjna GTSRB (`notebooks/01_eda.ipynb`, wykresy w `results/eda/`),
- podział train/val **po seriach** (StratifiedGroupKFold, seed 42, zapis w `splits/val_tracks.json`): 31 349 / 7 860 zdjęć, zero wspólnych serii,
- przetwarzanie wstępne (64×64, normalizacja ImageNet) i augmentacje A0, A1, A2 w `src/data.py`,
- hipotezy H1 do H4, baseline, warianty, zakłócenia, metryki i reguła wyboru modelu.

### Hipotezy

1. **H1 (transfer):** modele z wagami ImageNet osiągają wyższe macro-F1 i szybciej się zbiegają niż te same architektury uczone od zera.
2. **H2 (augmentacja):** augmentacja odpornościowa A2 zmniejsza spadek skuteczności na zakłóconych danych bardziej niż geometryczna A1, przy stracie poniżej 1 p.p. macro-F1 na danych czystych.
3. **H3 (rozmiar modelu):** MobileNetV3-Small osiąga wynik zbliżony do ResNet18 (różnica poniżej 1 p.p. macro-F1) przy kilkukrotnie mniejszej liczbie parametrów.
4. **H4 (uogólnienie):** odporność z augmentacji przenosi się częściowo na zakłócenie niewidziane w treningu (JPEG).

## KM2: implementacja i eksperymenty

Wspólne ustawienia: ten sam podział danych, wejście 64×64, AdamW, maksymalnie 20 epok, early stopping na macro-F1 walidacji (cierpliwość 5), seed 42.

### Augmentacje

| Poziom | Operacje |
| --- | --- |
| A0 | przeskalowanie i normalizacja |
| A1 | A0 + obrót ±10°, przesunięcie do 10%, skala 0,9 do 1,1, jasność i kontrast ±30% |
| A2 | A1 + (p = 0,3 każda) rozmycie σ 0,1 do 1,5, przyciemnienie ×0,5 do 0,9, szum σ 0,01 do 0,05, RandomErasing 2% do 15% |

Bez odbić lustrzanych (zmieniają znaczenie znaków) i bez JPEG (zakłócenie testowe dla H4).

### Zakłócenia do oceny odporności

Deterministyczne, nakładane na zbiór walidacyjny, każde w 3 poziomach.

| Zakłócenie | Poziom 1 | Poziom 2 | Poziom 3 | W A2 |
| --- | --- | --- | --- | --- |
| Rozmycie Gaussa | σ = 0,8 | σ = 2,0 | σ = 3,0 | tak |
| Szum Gaussa | σ = 0,03 | σ = 0,06 | σ = 0,10 | tak |
| Przyciemnienie | ×0,6 | ×0,4 | ×0,25 | tak |
| Zasłonięcie (w ramce ROI) | 10% ROI | 20% ROI | 30% ROI | tak |
| Kompresja JPEG | jakość 30 | jakość 15 | jakość 8 | nie |

### Macierz eksperymentów

| Etap | Cel | Warianty | Treningów | Hipoteza |
| --- | --- | --- | --- | --- |
| E1 | architektura i transfer | SimpleCNN; ResNet18 i MobileNetV3-Small, od zera i z ImageNet; A0 | 5 | H1, H3 |
| E2 | augmentacja | SimpleCNN + najlepszy ResNet18 i MobileNetV3 z E1; A1 i A2 | 6 | H2 |
| E3 | odporność | 11 modeli z E1 i E2 na zakłóconej walidacji | 0 | H2, H4 |
| E4 | stabilność | zwycięzca, 3 seedy | 3 | rzetelność |

### Metryki

Accuracy i macro-F1 (czyste dane), mCA (średnia accuracy po 5 zakłóceniach i 3 poziomach), względny spadek (acc czyste minus mCA, podzielone przez acc czyste), liczba parametrów, czas predykcji na CPU, macierz pomyłek, F1 dla każdej klasy.

### Reguła wyboru modelu (ustalona przed treningami)

1. Odrzucamy warianty z macro-F1 na czystej walidacji niższym od najlepszego o więcej niż 1 p.p.
2. Spośród pozostałych wybieramy najwyższe mCA.
3. Przy różnicy mCA poniżej 0,5 p.p. wygrywa model mniejszy i szybszy.

### Wyniki każdego treningu

`results/<run>/`: `config.json`, `history.csv`, `metrics.json`, `confusion_matrix.png`, `per_class_f1.csv`, wagi `best.pt` (poza gitem).

### Do oddania

- kod treningu i ewaluacji uruchamiany jednym poleceniem z konfiguracją,
- wyniki E1 do E4 i tabela zbiorcza z wykresami,
- sprawozdanie KM2: weryfikacja H1 do H4 i uzasadniony wybór finalnego modelu.

## KM3: finalne rozwiązanie, ocena i XAI

- **Ocena końcowa:** zwycięski model i baseline oceniane **raz** na 12 630 zdjęciach testowych, czystych i zakłóconych. Accuracy, macro-F1, mCA, F1 dla klas, macierz pomyłek, najczęściej mylone pary klas.
- **XAI:**
  1. Grad-CAM dla każdej predykcji,
  2. ilościowy wskaźnik: jaka część mapy Grad-CAM leży wewnątrz ramki ROI, porównany dla predykcji poprawnych i błędnych oraz obrazów czystych i zakłóconych,
  3. galeria najpewniejszych błędów pogrupowana według typu,
  4. rozkład pewności modelu dla predykcji poprawnych i błędnych.
- **Aplikacja Gradio:**
  - zakładka „Rozpoznaj”: top-5 klas i mapa Grad-CAM,
  - zakładka „Test odporności”: suwaki typu i siły zakłócenia na żywo,
  - zakładka „Wyniki”: wykresy z KM2 i KM3.
- **Analiza ograniczeń:** tylko znaki niemieckie, wycinki ze znakiem na środku (bez detekcji), sztuczne zakłócenia jako przybliżenie warunków rzeczywistych.

## Zgodność z wytycznymi prowadzącego

| Wymaganie | Gdzie w projekcie |
| --- | --- |
| Cel i problem rozwiązywany z AI | KM1, pytanie badawcze |
| Wybór, analiza i uzasadnienie datasetu | KM1, EDA i podział danych |
| DoE: hipotezy, metryki, baseline, plan | KM1, H1 do H4, metryki, baseline, macierz E1 do E4 |
| Sieci konwolucyjne | SimpleCNN, ResNet18, MobileNetV3 |
| Trenowanie, fine-tuning, porównanie architektur, augmentacje | KM2, E1, E2, E4 |
| Eksperymenty wynikają z celu | każdy etap przypisany do hipotezy, reguła wyboru ustalona z góry |
| Ocena na danych niewykorzystanych w uczeniu | KM3, oficjalny test użyty raz |
| Aplikacja z modułem AI | KM3, Gradio |
| XAI i analiza błędów | Grad-CAM, wskaźnik ROI, galeria błędów |
| Mocne i słabe strony | KM3, analiza ograniczeń |

## Ryzyka

| Ryzyko | Co robimy |
| --- | --- |
| Wszystkie modele mają około 99% na czystych danych | spodziewane; wnioski opieramy na mCA, gdzie różnice są większe |
| Wynik zależy od seeda | E4: 3 seedy dla zwycięzcy, średnia i odchylenie |
| Wyciek danych między train a val | podział po seriach, zapisany do pliku i używany wszędzie |
| Hipoteza się nie potwierdzi | to też wynik; opisujemy dlaczego, na podstawie liczb i Grad-CAM |
