# Kluczowe fragmenty kodu

## Pobieranie danych rok po roku

USGS API zwraca **maksymalnie 20 000 rekordów na zapytanie**. Gdyby pobrać całe
2000–2023 jednym żądaniem, milcząco straciłybyśmy część danych (zwłaszcza rok
2011 z tysiącami zdarzeń). Dlatego pętla pobiera **osobno każdy rok** i łączy
wyniki:

```python
for year in range(2000, 2024):
    url = f"{BASE}{PARAMS}&starttime={year}-01-01&endtime={year}-12-31"
    df = pd.read_csv(url)        # jedno zapytanie = jeden rok
    frames.append(df)
    time.sleep(1)               # grzeczne odstępy między zapytaniami
all_events = pd.concat(frames, ignore_index=True)
```

`time.sleep(1)` chroni publiczne API przed przeciążeniem. Plik: `utils/download_data.py`.

## Agregacja do siatki — pd.cut() i cell_id

`pd.cut()` przypisuje każdą współrzędną do przedziału (kosza) siatki, zwracając
indeks całkowity. Łączymy indeksy szerokości i długości w jeden identyfikator
komórki:

```python
df["lat_idx"] = pd.cut(df["latitude"],  bins=LAT_BINS, labels=False, right=False)
df["lon_idx"] = pd.cut(df["longitude"], bins=LON_BINS, labels=False, right=False)
df["cell_id"] = df["lat_idx"].astype(str) + "_" + df["lon_idx"].astype(str)
annual = df.groupby(["cell_id", "year"]).size().reset_index(name="count")
```

`labels=False` daje numery koszy zamiast etykiet; `groupby(...).size()` zlicza
zdarzenia w każdej komórce-roku. Plik: `utils/data_prep.py`, `notebooks/02`.

## Model Stan — bloki

```stan
data {            // co podajemy z zewnątrz
  int N; int C;
  array[N] int cell_id;   // przypisanie obserwacji do komórki
  array[N] int count;     // liczby zdarzeń
}
parameters {      // co estymujemy
  real mu_global; real<lower=0> sigma_global; vector[C] alpha;
}
model {           // priory + wiarygodność
  mu_global ~ normal(2, 1);
  sigma_global ~ normal(0, 1);              // half-normal przez <lower=0>
  alpha ~ normal(mu_global, sigma_global);  // partial pooling
  count ~ poisson_log(alpha[cell_id]);
}
generated quantities {   // wielkości pochodne liczone po próbkowaniu
  vector[N] log_lik;     // POTRZEBNE do WAIC/LOO
  array[N] int count_pred;
  for (n in 1:N) {
    count_pred[n] = poisson_log_rng(alpha[cell_id[n]]);
    log_lik[n] = poisson_log_lpmf(count[n] | alpha[cell_id[n]]);
  }
}
```

`data` to wejście, `parameters` to niewiadome, `model` definiuje rozkład
(priory + wiarygodność), a `generated quantities` liczy rzeczy potrzebne do
analizy — w tym **`log_lik`, bez którego `az.compare` nie zadziała**.

## poisson_log — dlaczego lepszy niż Poisson z abs()

`poisson_log(eta)` to Poisson, którego logarytm intensywności podajemy wprost:
`lambda = exp(eta)`. Dzięki temu `lambda` jest **zawsze dodatnia** w naturalny
sposób, a obliczenia są numerycznie stabilne (Stan pracuje w przestrzeni log).
Alternatywa „zwykły Poisson + `abs()` na lambda" jest błędna: `abs()` tworzy załom
w funkcji gęstości (nieróżniczkowalność), psuje gradienty wykorzystywane przez
sampler i może powodować dwuznaczność znaku. Dlatego w całym projekcie używamy
`poisson_log` i linku logarytmicznego, nigdy `abs()`.

## az.compare() — WAIC, LOO, elpd w prostych słowach

```python
comp_dict = {"model_1_nopool": idata1, "model_2_partial": idata2}
az.compare(comp_dict, ic="waic")   # ranking wg WAIC
az.compare(comp_dict, ic="loo")    # ranking wg PSIS-LOO
```

- **elpd** — miara, jak dobrze model przewidziałby *nowe* dane; wyższa = lepsza.
- **WAIC** — szybkie przybliżenie elpd liczone z `log_lik`.
- **PSIS-LOO** — sprytne przybliżenie walidacji „zostaw-jeden-poza" (leave-one-out)
  metodą ważenia próbek; dodatkowo zwraca diagnostykę **Pareto k** sygnalizującą,
  dla których obserwacji przybliżenie jest niewiarygodne (`k > 0,7`).

## Jak działa shrinkage w kodzie

Cała magia tkwi w jednej linijce różniącej modele:

```stan
// Model 1: sigma USTALONE -> brak sprzężenia między komórkami
alpha ~ normal(2, 2);
// Model 2: sigma_global ESTYMOWANE -> komórki dzielą wspólny rozkład
alpha ~ normal(mu_global, sigma_global);
```

Gdy `sigma_global` jest estymowane, wszystkie `alpha` „widzą" wspólne
`mu_global` i `sigma_global`. Komórka uboga w dane ma słabą wiarygodność, więc
jej posteriori jest przyciągane do `mu_global` (shrinkage). Komórka bogata w dane
ma silną wiarygodność, która „przebija" hyperprior — zostaje przy swoim. Efekt
mierzymy na skali `alpha` (log), porównując `sd(alpha)` między modelami:

```python
m1_sd = idata1.posterior["alpha"].std(("chain", "draw"))
m2_sd = idata2.posterior["alpha"].std(("chain", "draw"))
reduction = 1 - m2_sd / m1_sd     # dodatnie = Model 2 pewniejszy
```

## Powiązane pliki

- [../utils/download_data.py](../utils/download_data.py) · [../utils/data_prep.py](../utils/data_prep.py)
- [../models/model2_partial.stan](../models/model2_partial.stan)
- [../notebooks/05_model2_posterior.ipynb](../notebooks/05_model2_posterior.ipynb) · [../notebooks/06_comparison.ipynb](../notebooks/06_comparison.ipynb)
