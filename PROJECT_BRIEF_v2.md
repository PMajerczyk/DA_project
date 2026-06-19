# Project Brief v2 — Bayesian Spatial Modelling of Seismic Activity
# Wariant B: siatka przestrzenna + dane USGS

> Dokument kontekstowy dla Claude Code. Wersja 2 — po analizie przykładowego projektu
> (traffic incidents). Czytać razem z PROJECT_OVERVIEW (kryteria oceny).
> Przedmiot: Data Analytics (AIR ISZ). Zespół: Paweł Majerczyk, Jakub Gicala.
> Narzędzie: CmdStanPy + ArviZ (jak w projekcie referencyjnym) lub PyMC + ArviZ.

---

## 1. Temat w jednym zdaniu

Bayesowskie modelowanie liczby trzęsień ziemi (>= M4.0) per komórka siatki przestrzennej,
porównując model bez efektu przestrzennego (Model 1) z modelem hierarchicznym z partial
pooling po strefach aktywności (Model 2). Wynikiem jest kolorowa mapa posteriorowej
intensywności z przedziałami niepewności.

---

## 2. Dane — Wariant B (siatka + USGS catalog)

### Źródło
USGS Earthquake Catalog API — jeden endpoint, format CSV, brak rejestracji.

```python
import pandas as pd

url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=csv"
    "&starttime=2000-01-01"
    "&endtime=2023-12-31"
    "&minmagnitude=4.0"
    "&orderby=time"
    # Opcjonalnie: bbox dla konkretnego regionu
    # "&minlatitude=24&maxlatitude=50&minlongitude=122&maxlongitude=154"  # Japonia
)
df = pd.read_csv(url)
```

### Preprocessing — agregacja do siatki
```python
import numpy as np

# Definiujemy siatkę np. 2°×2°
lat_bins = np.arange(24, 50, 2)   # Japonia: 24-50°N
lon_bins = np.arange(122, 154, 2) # Japonia: 122-154°E

df['lat_cell'] = pd.cut(df['latitude'], bins=lat_bins, labels=False)
df['lon_cell'] = pd.cut(df['longitude'], bins=lon_bins, labels=False)
df['cell_id'] = df['lat_cell'].astype(str) + '_' + df['lon_cell'].astype(str)

# Roczna liczba zdarzeń per komórka
annual_counts = (df.groupby(['cell_id', df['time'].str[:4]])
                   .size()
                   .reset_index(name='count'))
```

### Wynikowa struktura danych (wejście do modelu)
| cell_id | year | count | lat_center | lon_center |
|---------|------|-------|------------|------------|
| 3_2     | 2000 | 47    | 30.0       | 128.0      |
| 3_2     | 2001 | 31    | 30.0       | 128.0      |
| 8_5     | 2000 | 312   | 40.0       | 138.0      |

Każda obserwacja = jedna komórka siatki w jednym roku.

### Region — Japonia (rekomendowany)
- Bardzo aktywna sejsmicznie → dużo danych per komórka → stabilne posteriory
- Dobrze znane zróżnicowanie przestrzenne (wschodnie wybrzeże >> zachodnie)
- Rok 2011 (Tohoku M9.0) → interesujący outlier do omówienia w Data Consistency
- Alternatywa: cały Pacyfic Ring of Fire (więcej zróżnicowania przestrzennego)

---

## 3. Dwa modele — uzasadniona różnica

### Kluczowa lekcja z projektu referencyjnego
W projekcie traffic accidents różnica między modelami to tylko dodanie covariatu
(humidity vs humidity+temperature). To słabe uzasadnienie. Nasza różnica jest
strukturalna — zmienia się założenie o niezależności komórek.

### Model 1 — Poisson bez poolingu (baseline)
Każda komórka estymowana niezależnie. Brak dzielenia informacji między sąsiednimi
komórkami. Problem: komórki z małą liczbą zdarzeń → szeroki CI, niestabilna estymacja.

```
# Matematycznie:
Y_c ~ Poisson(lambda_c)        # c = cell index
log(lambda_c) = alpha_c        # osobny intercept per komórka
alpha_c ~ Normal(mu, sigma)    # wspólny prior ALE niezależne — no pooling
```

W Stanie:
```stan
data {
  int<lower=0> N;          // liczba obserwacji (cell × year)
  int<lower=0> C;          // liczba komórek
  array[N] int cell_id;    // indeks komórki dla każdej obs.
  array[N] int count;      // liczba trzęsień
}
parameters {
  vector[C] alpha;         // log-intensywność per komórka
  real mu;                 // globalny prior mean
  real<lower=0> sigma;     // globalny prior sd
}
model {
  mu ~ normal(2, 1);       // prior na log(~7 trzęsień/rok)
  sigma ~ half_normal(0, 1);
  alpha ~ normal(mu, sigma);  // niezależne — no pooling
  for (n in 1:N)
    count[n] ~ poisson_log(alpha[cell_id[n]]);
}
generated quantities {
  array[N] int count_pred;
  array[N] real log_lik;
  for (n in 1:N) {
    count_pred[n] = poisson_log_rng(alpha[cell_id[n]]);
    log_lik[n] = poisson_log_lpmf(count[n] | alpha[cell_id[n]]);
  }
}
```

### Model 2 — Hierarchiczny Poisson z partial pooling
Komórki współdzielą hyperprior — "pożyczają siłę" od sąsiadów. Komórki z małą liczbą
danych są przyciągane do globalnej średniej (shrinkage). Fizyczne uzasadnienie:
sąsiednie komórki leżą w tej samej strefie tektonicznej → podobna aktywność bazowa.

```
# Matematycznie:
Y_c ~ Poisson(lambda_c)
log(lambda_c) = alpha_c
alpha_c ~ Normal(mu_global, sigma_global)   // partial pooling
mu_global ~ Normal(2, 1)
sigma_global ~ HalfNormal(0, 1)
```

Kluczowa różnica od Modelu 1: sigma_global jest ESTYMOWANE z danych, nie zakładane.
Przy małym sigma_global → komórki zbliżają się do global mean (silny shrinkage).
Przy dużym sigma_global → komórki są niezależne (odpada do no-pooling).

```stan
data {
  int<lower=0> N;
  int<lower=0> C;
  array[N] int cell_id;
  array[N] int count;
}
parameters {
  real mu_global;              // hyperprior mean
  real<lower=0> sigma_global;  // hyperprior sd — KEY PARAMETER
  vector[C] alpha;             // per-cell log-intensity
}
model {
  mu_global ~ normal(2, 1);
  sigma_global ~ half_normal(0, 1);
  alpha ~ normal(mu_global, sigma_global);  // partial pooling
  for (n in 1:N)
    count[n] ~ poisson_log(alpha[cell_id[n]]);
}
generated quantities {
  array[N] int count_pred;
  array[N] real log_lik;
  for (n in 1:N) {
    count_pred[n] = poisson_log_rng(alpha[cell_id[n]]);
    log_lik[n] = poisson_log_lpmf(count[n] | alpha[cell_id[n]]);
  }
}
```

### Tabela różnic (do raportu)
| Aspekt | Model 1 (No Pooling) | Model 2 (Partial Pooling) |
|--------|---------------------|--------------------------|
| Założenie | Komórki niezależne | Komórki uczą się od siebie |
| sigma_global | Zakładane z góry | Estymowane z danych |
| Problem | Duży CI dla rzadkich komórek | Kontrolowany przez shrinkage |
| Fizyczne uzas. | Brak struktury przestrzennej | Sąsiednie strefy tektoniczne |
| Efekt | Wilde wyniki na peryferiach | Gładka, sensowna mapa |

---

## 4. Priory — jak uzasadnić (wzorując się na projekcie referencyjnym)

Projekt traffic incidents pokazał jak to zrobić: obliczają górną granicę z danych,
potem liczą sigma analitycznie. My robimy to samo:

### Krok 1: Górna granica
Historycznie max roczna liczba trzęsień M>=4.0 w jednej komórce 2°×2° w Japonii
wynosi ~500 (okolice Tohoku 2011). Górna granica: 2× = 1000.

### Krok 2: Uzasadnienie prioru na alpha (log-intensywność)
```
Średnia komórka w Japonii: ~30 trzęsień/rok → log(30) ≈ 3.4
Prior: alpha ~ Normal(3.4, 1.5)
→ 95% prior CI: [0.5, 6.3] → [exp(0.5), exp(6.3)] = [1.6, 544] trzęsień/rok ✓
```

### Krok 3: Prior predictive check (WYMAGANY w kryteriach)
```python
# Parametry
alpha_sim = np.random.normal(3.4, 1.5, size=1000)
count_sim = np.random.poisson(np.exp(alpha_sim))

# Sprawdzamy:
# - Czy max symulowany count < 1000? ✓
# - Czy rozkład ma sens (0-500)? ✓
# - Czy nie ma ujemnych wartości? ✓ (exp zawsze dodatnie)
```

### Prior predictive check (pomiarów — WYMAGANY):
```python
# Symulujemy całe "fałszywe" datasety z prioru
# i sprawdzamy czy wyglądają jak prawdziwe dane sejsmiczne
```

---

## 5. Posterior analysis — co robić (wzorując się na projekcie referencyjnym)

### Co projekt referencyjny zrobił (minimum):
- Histogram parametru 'a' — posterior vs prior
- Bar chart: observed vs predicted per miesiąc

### Co MY robimy (minimum + to czego brakło tamtemu projektowi):
1. **Trace plots** — `az.plot_trace(idata)` — OBOWIĄZKOWE (tamten projekt tego nie ma!)
2. **R-hat i ESS** — `az.summary(idata)` — diagnostyka konwergencji
3. **Posterior per parametr** — `az.plot_posterior(idata)` — histogram z HDI
4. **Posterior predictive** — observed vs predicted per komórka (bar chart lub mapa)
5. **Data consistency** — czy posterior predictive pokrywa obserwowane wartości?
   - Rok 2011 Tohoku → będzie outlier → UZASADNIAMY że model stacjonarny tego nie capturuje
6. **Efekt shrinkage** — porównanie CI dla komórek z małą i dużą liczbą obs.
7. **Mapa posteriorowa** — kolorowa mapa lambda_c + mapa sigma (niepewność)

---

## 6. Model Comparison — dokładnie jak w projekcie referencyjnym

```python
import arviz as az

comp_dict = {"model_1_nopool": idata1, "model_2_partial": idata2}

# WAIC
comp_waic = az.compare(comp_dict, ic="waic")
print(comp_waic)
az.plot_compare(comp_waic)

# PSIS-LOO
comp_loo = az.compare(comp_dict, ic="loo")
print(comp_loo)
az.plot_compare(comp_loo)
```

### Co opisać (per kryterium):
- **WAIC:** który model wygrywa, jaki elpd_diff, czy warning (jak w projekcie ref.)?
- **PSIS-LOO:** czy Pareto k-hat > 0.7? Które komórki? (Rok 2011 to kandydat)
- **Final assessment:** Model 2 powinien wygrać — uzasadniamy że shrinkage fizycznie
  sensowny, komórki w tym samym regionie tektonicznym mają podobną aktywność bazową

---

## 7. Narzędzia

### Rekomendowane: CmdStanPy + ArviZ (jak projekt referencyjny)
```
pip install cmdstanpy arviz pandas numpy geopandas matplotlib
cmdstanpy.install_cmdstan()
```

### Alternatywa: PyMC + ArviZ
```
pip install pymc arviz pandas numpy geopandas matplotlib
```

Oba dają WAIC i PSIS-LOO przez ArviZ. CmdStanPy jest szybszy dla dużych modeli,
PyMC ma czytelniejszą składnię Python.

---

## 8. Struktura repo i notebooków

```
project/
  PROJECT_OVERVIEW.pdf          # kryteria oceny
  PROJECT_BRIEF.md              # ten plik
  README.md
  data/
    raw/
      earthquakes_japan.csv     # surowe dane z USGS API
    processed/
      grid_annual_counts.csv    # count per cell per year
      grid_metadata.csv         # lat/lon center per cell
  models/
    model1_nopool.stan          # Stan model 1
    model2_partial.stan         # Stan model 2
  notebooks/
    01_data_acquisition.ipynb   # pobranie USGS + opis źródła (kryt. 1 pkt 3)
    02_preprocessing.ipynb      # agregacja do siatki + EDA + mapa surowa (kryt. 1 pkt 4)
    03_priors.ipynb             # uzasadnienie + prior predictive checks (kryt. 3)
    04_model1_posterior.ipynb   # fit M1 + diagnostyka + posterior predictive (kryt. 4)
    05_model2_posterior.ipynb   # fit M2 + diagnostyka + shrinkage + mapa (kryt. 5)
    06_comparison.ipynb         # WAIC + LOO + final assessment + mapa końcowa (kryt. 6)
  report/
    main_report.ipynb           # złożony raport (run_notebook pattern jak w ref.)
  utils/
    display.py                  # display_df, display_image (jak w projekcie ref.)
    notebook.py                 # run_notebook helper
```

---

## 9. Czego UNIKAĆ (błędy projektu referencyjnego)

1. NIE pomijać trace plots i diagnostyki MCMC — to osobny punkt kryteriów (Sampling Issues)
2. NIE robić dwóch modeli które różnią się tylko dodaniem jednej zmiennej — różnica
   musi być STRUKTURALNA (no-pooling vs hierarchiczny) z uzasadnieniem
3. NIE ignorować ostrzeżeń WAIC/LOO — opisać je i wytłumaczyć (projekt ref. tylko wspomniał)
4. NIE używać abs() do wymuszania dodatnich wartości lambda — używać poisson_log i log link
5. NIE porównywać tylko jednej próbki z każdego modelu — używać wszystkich obserwacji

---

## 10. Otwarte decyzje

1. Narzędzie: CmdStanPy czy PyMC? (CmdStanPy = jak projekt ref., PyMC = prostszy Python)
2. Rozdzielczość siatki: 1°×1° (więcej komórek, mniej danych per komórka) vs 2°×2°?
3. Region: Japonia (znana, aktywna) vs Ring of Fire (więcej zróżnicowania)?
4. Lata: 2000–2023 (bez Tohoku) vs 1990–2023 (z Tohoku jako omawianym outlierem)?
5. Czy dodać Negative Binomial jako wariant (overdispersion) zamiast hierarchii?
   → NB vs Poisson to alternatywna para modeli, prostsza ale mniej oryginalna

---

## 11. Pobieranie danych przez Claude Code — instrukcja

Claude Code może pobrać dane sam podczas sesji. USGS API ma limit 20 000 rekordów
per zapytanie, więc dane muszą być pobrane rok po roku w pętli.

### Prompt startowy do Claude Code (wkleić na początku sesji):

```
Mamy projekt bayesowskiego modelowania aktywnosci sejsmicznej.
Pliki kontekstowe: PROJECT_OVERVIEW.pdf, PROJECT_BRIEF_v2.md

Krok 1 — pobierz dane:
Pobierz trzesienia ziemi z USGS API dla Japonii (lat 24-50N, lon 122-154E),
magnitudy >= 4.0, lata 2000-2023. API ma limit 20000 rekordow per zapytanie
— pobieraj rok po roku w petli i polacz w jeden DataFrame. Zapisz do
data/raw/earthquakes_japan.csv. Wypisz shape i pierwsze wiersze.

Krok 2 — preprocessing:
Zagreguj do siatki 2x2 stopnie, oblicz roczna liczbe zdarzen per komorka.
Zapisz do data/processed/grid_annual_counts.csv. Narysuj mape surowych danych.

Potem przejdz do implementacji notebookow zgodnie z PROJECT_BRIEF_v2.md.
```

### Kod do pobrania danych (Claude Code uruchomi automatycznie):

```python
import pandas as pd
import time

BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
PARAMS = (
    "?format=csv&minmagnitude=4.0&orderby=time"
    "&minlatitude=24&maxlatitude=50"
    "&minlongitude=122&maxlongitude=154"
)

frames = []
for year in range(2000, 2024):
    url = f"{BASE}{PARAMS}&starttime={year}-01-01&endtime={year}-12-31"
    try:
        df = pd.read_csv(url)
        frames.append(df)
        print(f"{year}: {len(df)} events")
        time.sleep(1)  # grzeczne odpytywanie API
    except Exception as e:
        print(f"{year}: ERROR {e}")

all_events = pd.concat(frames, ignore_index=True)
all_events.to_csv("data/raw/earthquakes_japan.csv", index=False)
print(f"Total: {len(all_events)} events")
```

### Oczekiwany wynik:
- ~150 000-200 000 zdarzen dla Japonii 2000-2023 M>=4.0
- Czas pobierania: ~3-5 minut (24 zapytania × sleep 1s)
- Plik: ~30-50 MB CSV
