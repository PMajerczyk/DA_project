# Priory

## Co to jest prior i po co go dobieramy

W podejściu bayesowskim **prior** to rozkład prawdopodobieństwa wyrażający naszą
wiedzę o parametrze *przed* zobaczeniem danych. Po połączeniu z funkcją
wiarygodności (danymi) daje rozkład *posteriori*. Dobry prior powinien być
**słabo informujący** (weakly-informative): kodować właściwą skalę i znak
zjawiska, ale nie narzucać odpowiedzi tam, gdzie dane mają głos. W modelu Poissona
z linkiem logarytmicznym prior nakładamy na log-intensywność `alpha = log(lambda)`
— a tu trzeba uważać, bo łagodny z pozoru rozkład normalny na `alpha` może po
przejściu przez `exp()` implikować ekstremalne intensywności.

## Najważniejsza zasada: prior z wiedzy zewnętrznej, nie z danych (double-dipping)

Prowadzący zwrócił uwagę na typowy błąd: **nie wolno wyznaczać środka i
odchylenia priora ze średniej/mediany tych samych liczb (`count`), które potem
modelujemy**. Te same dane policzyłyby się dwa razy (raz w priorze, raz w
wiarygodności) → sztucznie zawężony, zbyt pewny posterior. To jest *data-leakage*
(double-dipping). Wcześniejsza wersja tego projektu popełniała ten błąd
(„mediana ~6, średnia ~16 → centrujemy niżej") i została **poprawiona**.

Podejście zgodne z artykułami prowadzącego — Räty i in. (2023, NHESS) stosują
priory „weakly informative / conventional" o dużych wariancjach i weryfikują je
**analizą wrażliwości**; Tu i in. (2025, Annals of GIS) idą według listy WAMBS
(prior predictive + prior sensitivity).

## Jak wyznaczyliśmy środek — stawka zewnętrzna + geometria siatki

Żadna liczba poniżej nie pochodzi z naszej próby 2000–2023:

1. **Publikowana stawka dla Japonii** (długie archiwum, źródło zewnętrzne):
   ~**1200 trzęsień M ≥ 4 rocznie**.
2. **Liczba komórek z samej geometrii** siatki (nie z danych): szerokość 24–50°
   → 26/2 = 13 wierszy, długość 122–154° → 32/2 = 16 kolumn → **208 komórek**.
3. **Stawka na komórkę-rok** (równomiernie): `1200 / 208 ≈ 5,8` zdarzenia/rok,
   stąd `mu_0 = log(5,8) ≈ 1,8`.

Równomierne rozłożenie celowo *zaniża* komórki aktywne i *zawyża* spokojne —
dlatego odchylenie `sigma_0` jest szerokie (sejsmiczność jest silnie
skoncentrowana, więc „typowa komórka" jest z natury rozmyta).

## Skąd kształt i górna granica

- **Kształt (prawoskośność)** wynika z **prawa Gutenberga-Richtera**
  `log₁₀N(≥M) = a − bM` z `b ≈ 1` — a nie z naszego histogramu. Potwierdzają to
  pasma dla Japonii: M≥4 ≈ 1200, M≥5 ≈ 149, M≥6 ≈ 12,7 (`1200/149 ≈ 8`,
  `149/12,7 ≈ 12` → `b ≈ 1`). Na skali log-intensywności uzasadnia to symetryczny
  rozkład normalny na `alpha` (czyli log-normalny na `lambda`).
- **Górna granica** `U ≈ 3000`/rok pochodzi z udokumentowanej produktywności
  wstrząsów wtórnych dużych trzęsień subdukcyjnych (sekwencja klasy Tohoku
  osiąga rząd 10³ M≥4 w najaktywniejszym rejonie), a dolna `L = 1`. Na skali log:
  `[0, 8]`. Odchylenie dobieramy tak, by ±3 sigma pokryło ten zakres:
  `sigma_0 = (8 − 1,8)/3 ≈ 2,07`.

**Priory końcowe:** Model 1 `alpha ~ Normal(1,8, 2,07)` (sigma ustalone);
Model 2 `mu_global ~ Normal(1,8, 1)`, `sigma_global ~ HalfNormal(1)`.

## Prior predictive checks — co pokazują

1. **Parametrów**: losujemy `alpha`, patrzymy na `lambda = exp(alpha)`. Mediana
   ~6/rok, 95% do kilkuset, ogon do niskich tysięcy, wszystko dodatnie. Sensowne.
2. **Pomiarów**: `count ~ Poisson(exp(alpha))` — nieujemne liczby całkowite,
   prawoskośne (konsekwencja G-R, nie odczyt z danych), prawie nic > 3000.
3. **Plauzybilność** (osobny krok, NIE uzasadnienie priora): nakładamy obwiednię
   priorową na dane obserwowane, by sprawdzić brak rażącego konfliktu. Priora nie
   stroimy pod dane.

## Analiza wrażliwości priora (dowód braku double-dippingu)

Przefitowaliśmy Model 1 z przesuniętym środkiem (±1 na skali log) i ze
zmniejszonym/zwiększonym odchyleniem. Posterior `lambda` praktycznie się nie
zmienia: komórka aktywna 120,0 → 120,1, średnia 3,0 (bez zmian), a tylko najbardziej
uboga w dane drgnie nieznacznie (1,1–2,1 — bo tam słaba wiarygodność pozwala
priorowi mówić więcej). To dowodzi, że prior jest słabo informujący, a **wybór
środka nie napędza wyników** — dokładnie odporność, którą raportują Räty i in.

## Źródła

- Gutenberg & Richter (1944), *BSSA* 34(4):185–188 — prawo częstość-magnituda.
- Publikowane stawki sejsmiczności Japonii (~1200 M≥4, 149 M≥5, 12,7 M≥6 / rok; b≈1).
- Räty i in. (2023), *NHESS* 23:2403 — priory weakly-informative + analiza wrażliwości.
- Tu, Yu & Tu (2025), *Annals of GIS* 31:1 — lista kontrolna WAMBS.
- Alternatywa: sprzężony prior **Gamma–Poisson** na `lambda` (parametry jako
  pseudo-obserwacje; np. Gupta & Baker 2017) — standardowa rama dla siatek.

## Powiązane pliki

- [../notebooks/03_priors.ipynb](../notebooks/03_priors.ipynb) — pełne wyprowadzenie + sensitivity
- [../utils/data_prep.py](../utils/data_prep.py) — `PRIOR_MU`, `PRIOR_SIGMA_M1`
- Modele: [02_modele.md](02_modele.md)
