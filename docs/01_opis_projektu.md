# Opis projektu

## Kontekst i motywacja

Japonia leży na styku czterech płyt tektonicznych (pacyficznej, filipińskiej,
euroazjatyckiej i północnoamerykańskiej), co czyni ją jednym z najbardziej
aktywnych sejsmicznie regionów świata. Trzęsienia ziemi są tam codziennością, a
ich rozkład przestrzenny jest silnie niejednorodny: wschodnie wybrzeże (strefy
subdukcji wzdłuż Rowu Japońskiego i Rowu Nankai) generuje wielokrotnie więcej
zdarzeń niż wnętrze kraju i jego zachodnia część. Wybraliśmy ten region właśnie
dlatego, że jest **bogaty w dane** (stabilne posteriory), ma **dobrze znaną
strukturę przestrzenną** (możemy zweryfikować, czy model „odkrywa" rzeczywistość)
oraz zawiera **spektakularny outlier** — trzęsienie Tohoku z 11 marca 2011 r.
(magnituda 9.0), które wraz z sekwencją wstrząsów wtórnych radykalnie podniosło
liczbę zdarzeń w tym roku. Outlier ten jest doskonałym studium przypadku dla
omówienia spójności danych z modelem.

## Pytanie badawcze

Chcemy odpowiedzieć na pytanie: **jaka jest roczna intensywność trzęsień ziemi
(M ≥ 4.0) w poszczególnych rejonach Japonii i jak pewni jesteśmy tych
oszacowań?** Formalnie modelujemy liczbę zdarzeń w komórce siatki w danym roku
jako zmienną losową Poissona o nieznanej intensywności `lambda`. Interesuje nas
nie tylko punktowe oszacowanie `lambda` dla każdej komórki, ale **pełny rozkład
posteriori** — bo właśnie niepewność decyduje o użyteczności mapy ryzyka.
Dodatkowo pytamy, czy **dzielenie informacji między sąsiednimi komórkami**
(partial pooling) poprawia oszacowania, zwłaszcza tam, gdzie danych jest mało.

## Opis danych

Dane pochodzą z **USGS Earthquake Catalog API** (publiczny serwis FDSN, format
CSV, bez rejestracji). Pobieramy zdarzenia w prostokącie obejmującym Japonię
(szerokość 24–50°N, długość 122–154°E), o magnitudzie ≥ 4.0, z lat 2000–2023.
Ponieważ API zwraca maksymalnie 20 000 rekordów na zapytanie, pobieramy dane
**rok po roku** w pętli (24 zapytania z odstępem 1 s). Otrzymujemy ~33 100
zdarzeń. Każdy rekord zawiera m.in. czas, szerokość i długość geograficzną,
głębokość, magnitudę oraz metadane jakości pomiaru. W roku 2011 jest 5746 zdarzeń
wobec mediany ~1275 rocznie — widać Tohoku gołym okiem.

## Opis siatki przestrzennej

Punktowe epicentra agregujemy do **regularnej siatki 2°×2°**. Komórka to kwadrat
2 stopnie na 2 stopnie (~220 km na ~180 km w tych szerokościach). Dla każdej
komórki i każdego roku liczymy liczbę zdarzeń — to jedna obserwacja modelu.
Otrzymujemy 154 aktywne komórki i 2086 obserwacji typu „komórka-rok". Dlaczego
2°×2°? To kompromis: siatka drobniejsza (1°×1°) dawałaby więcej komórek, ale z
mniejszą liczbą danych każda (niestabilne estymacje), a grubsza zacierałaby
strukturę przestrzenną. Przy 2°×2° większość komórek ma sensowną liczbę zdarzeń,
a mapa zachowuje wyraźny gradient wschód–zachód.

## Use cases — do czego służą takie mapy ryzyka

- **Ubezpieczenia i reasekuracja** — wycena polis i ekspozycji w zależności od
  rejonu; niepewność jest tu kluczowa dla rezerw.
- **Planowanie infrastruktury** — normy budowlane, lokalizacja elektrowni,
  mostów, szpitali.
- **Priorytetyzacja monitoringu** — gdzie zagęścić sieć sejsmometrów.
- **Komunikacja ryzyka** — mapy z przedziałami niepewności są uczciwsze niż same
  wartości punktowe i lepiej wspierają decyzje publiczne.

## Powiązane pliki

- [../notebooks/01_data_acquisition.ipynb](../notebooks/01_data_acquisition.ipynb) — pobranie i opis danych
- [../notebooks/02_preprocessing.ipynb](../notebooks/02_preprocessing.ipynb) — agregacja do siatki + EDA
- Modele: [02_modele.md](02_modele.md) · Wyniki: [04_wyniki.md](04_wyniki.md)
