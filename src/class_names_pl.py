"""Polish names of the 43 GTSRB classes, following the Polish road sign regulations.

GTSRB contains German signs; each class is named after its Polish equivalent
(the sign with the same meaning and pictogram) together with the Polish sign code.
The directions were checked against the class pictograms in data/Meta/ (e.g. class 21
is the double curve turning left first, so A-4).

    CLASS_NAMES_PL[c]        full name, e.g. "Ograniczenie prędkości do 50 km/h"
    CLASS_NAMES_PL_SHORT[c]  at most 26 characters, for tight plot captions
    SIGN_CODES[c]            Polish sign code, e.g. "B-33"
    SIGN_GROUPS_PL           8 visual groups (shape and colour), used to pick varied examples
"""

from __future__ import annotations

import textwrap

from src.data import NUM_CLASSES

# (code, full name, short name)
_SIGNS = [
    ("B-33", "Ograniczenie prędkości do 20 km/h", "Ograniczenie 20 km/h"),
    ("B-33", "Ograniczenie prędkości do 30 km/h", "Ograniczenie 30 km/h"),
    ("B-33", "Ograniczenie prędkości do 50 km/h", "Ograniczenie 50 km/h"),
    ("B-33", "Ograniczenie prędkości do 60 km/h", "Ograniczenie 60 km/h"),
    ("B-33", "Ograniczenie prędkości do 70 km/h", "Ograniczenie 70 km/h"),
    ("B-33", "Ograniczenie prędkości do 80 km/h", "Ograniczenie 80 km/h"),
    ("B-34", "Koniec ograniczenia prędkości do 80 km/h", "Koniec ograniczenia 80"),
    ("B-33", "Ograniczenie prędkości do 100 km/h", "Ograniczenie 100 km/h"),
    ("B-33", "Ograniczenie prędkości do 120 km/h", "Ograniczenie 120 km/h"),
    ("B-25", "Zakaz wyprzedzania", "Zakaz wyprzedzania"),
    ("B-26", "Zakaz wyprzedzania przez samochody ciężarowe", "Zakaz wyprz. ciężarówek"),
    ("A-6a", "Skrzyżowanie z drogą podporządkowaną występującą po obu stronach", "Skrzyżowanie z podporz."),
    ("D-1", "Droga z pierwszeństwem", "Droga z pierwszeństwem"),
    ("A-7", "Ustąp pierwszeństwa", "Ustąp pierwszeństwa"),
    ("B-20", "Stop", "Stop"),
    ("B-1", "Zakaz ruchu w obu kierunkach", "Zakaz ruchu"),
    ("B-5", "Zakaz wjazdu samochodów ciężarowych", "Zakaz wjazdu ciężarówek"),
    ("B-2", "Zakaz wjazdu", "Zakaz wjazdu"),
    ("A-30", "Inne niebezpieczeństwo", "Inne niebezpieczeństwo"),
    ("A-2", "Niebezpieczny zakręt w lewo", "Zakręt w lewo"),
    ("A-1", "Niebezpieczny zakręt w prawo", "Zakręt w prawo"),
    ("A-4", "Niebezpieczne zakręty – pierwszy w lewo", "Zakręty, pierwszy w lewo"),
    ("A-11", "Nierówna droga", "Nierówna droga"),
    ("A-15", "Śliska jezdnia", "Śliska jezdnia"),
    ("A-12b", "Zwężenie jezdni – prawostronne", "Zwężenie prawostronne"),
    ("A-14", "Roboty na drodze", "Roboty na drodze"),
    ("A-29", "Sygnały świetlne", "Sygnały świetlne"),
    ("A-16", "Przejście dla pieszych", "Przejście dla pieszych"),
    ("A-17", "Dzieci", "Dzieci"),
    ("A-24", "Rowerzyści", "Rowerzyści"),
    ("A-32", "Oszronienie jezdni", "Oszronienie jezdni"),
    ("A-18b", "Zwierzęta dzikie", "Zwierzęta dzikie"),
    ("B-42", "Koniec zakazów", "Koniec zakazów"),
    ("C-2", "Nakaz jazdy w prawo za znakiem", "W prawo za znakiem"),
    ("C-4", "Nakaz jazdy w lewo za znakiem", "W lewo za znakiem"),
    ("C-5", "Nakaz jazdy prosto", "Nakaz jazdy prosto"),
    ("C-6", "Nakaz jazdy prosto lub w prawo", "Prosto lub w prawo"),
    ("C-7", "Nakaz jazdy prosto lub w lewo", "Prosto lub w lewo"),
    ("C-9", "Nakaz jazdy z prawej strony znaku", "Z prawej strony znaku"),
    ("C-10", "Nakaz jazdy z lewej strony znaku", "Z lewej strony znaku"),
    ("C-12", "Ruch okrężny", "Ruch okrężny"),
    ("B-27", "Koniec zakazu wyprzedzania", "Koniec zakazu wyprz."),
    ("B-28", "Koniec zakazu wyprzedzania przez samochody ciężarowe", "Koniec zakazu wyprz. cięż."),
]

SIGN_CODES = [code for code, _, _ in _SIGNS]
CLASS_NAMES_PL = [name for _, name, _ in _SIGNS]
CLASS_NAMES_PL_SHORT = [short for _, _, short in _SIGNS]

# Category in the Polish regulations, from the letter of the code.
CATEGORY_PL = {"A": "ostrzegawczy", "B": "zakazu", "C": "nakazu", "D": "informacyjny"}

# Visual groups (shape and colour), not legal categories: e.g. "Ustąp pierwszeństwa" is a
# warning sign (A-7) in the regulations but looks like no other sign.
SIGN_GROUPS_PL = {
    "Ograniczenia prędkości": [0, 1, 2, 3, 4, 5, 7, 8],
    "Inne znaki zakazu": [9, 10, 15, 16, 17],
    "Znaki odwołujące": [6, 32, 41, 42],
    "Znaki ostrzegawcze": [11, *range(18, 32)],
    "Znaki nakazu": list(range(33, 41)),
    "Droga z pierwszeństwem": [12],
    "Ustąp pierwszeństwa": [13],
    "Stop": [14],
}
GROUP_OF = {c: group for group, classes in SIGN_GROUPS_PL.items() for c in classes}


def category_pl(c: int) -> str:
    return CATEGORY_PL[SIGN_CODES[c][0]]


def class_label(c: int, short: bool = False, code: bool = False, width: int | None = None) -> str:
    """Polish class name, optionally with its code ("B-33 Ograniczenie...") and wrapped to `width`."""
    name = CLASS_NAMES_PL_SHORT[c] if short else CLASS_NAMES_PL[c]
    if code:
        name = f"{SIGN_CODES[c]} {name}"
    return textwrap.fill(name, width) if width else name


def _check() -> None:
    assert len(_SIGNS) == NUM_CLASSES
    assert sorted(GROUP_OF) == list(range(NUM_CLASSES)), "every class must be in exactly one visual group"
    assert max(map(len, CLASS_NAMES_PL_SHORT)) <= 26
    assert all(code[0] in CATEGORY_PL for code in SIGN_CODES)


_check()


if __name__ == "__main__":
    for c in range(NUM_CLASSES):
        print(f"{c:2d}  {SIGN_CODES[c]:<6} {CLASS_NAMES_PL[c]:<66} {CLASS_NAMES_PL_SHORT[c]:<26} {GROUP_OF[c]}")
