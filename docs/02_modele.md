# Modele

## Model Poissona — dlaczego pasuje do danych zliczeniowych

Naszą zmienną objaśnianą jest **liczba zdarzeń** (trzęsień) w komórce w ciągu
roku — liczba całkowita, nieujemna, często mała. Rozkład Poissona jest naturalnym
modelem takich danych: opisuje liczbę niezależnych zdarzeń zachodzących ze stałą
średnią intensywnością `lambda` w ustalonym oknie czasu. Ma jeden parametr
(`lambda` = średnia = wariancja) i nie dopuszcza wartości ujemnych. Ponieważ
`lambda` musi być dodatnia, modelujemy ją przez **link logarytmiczny**:
`log(lambda) = alpha`, gdzie `alpha` może być dowolną liczbą rzeczywistą.
Dzięki temu nigdy nie musimy sztucznie wymuszać dodatniości (np. przez `abs()`),
a parametry działają na wygodnej skali logarytmicznej.

## Model 1 — bez poolingu (no pooling)

**Założenie:** każda komórka jest niezależna; nie ma żadnego dzielenia informacji
między komórkami. Każda komórka `c` ma własną log-intensywność `alpha_c` z
ustalonym, wspólnym priorem:

```
count[c,y] ~ Poisson(lambda_c)
log(lambda_c) = alpha_c
alpha_c ~ Normal(2, 2)        # sigma USTALONE z góry
```

Kluczowe: parametr skali `sigma = 2` jest **ustalony**, nie estymowany. Każda
komórka „radzi sobie sama". **Kiedy zawodzi:** dla komórek z bardzo małą liczbą
obserwacji oszacowanie opiera się głównie na priorze i danych z jednej komórki,
co daje szerokie, niestabilne przedziały. Brak struktury przestrzennej oznacza
„dziki" wynik na peryferiach.

## Model 2 — partial pooling (hierarchiczny)

**Założenie:** komórki współdzielą hyperprior i „pożyczają sobie siłę".
Log-intensywności pochodzą ze wspólnego rozkładu, którego parametry są
**estymowane z danych**:

```
count[c,y] ~ Poisson(lambda_c)
log(lambda_c) = alpha_c
alpha_c ~ Normal(mu_global, sigma_global)   # partial pooling
mu_global    ~ Normal(2, 1)
sigma_global ~ HalfNormal(1)                 # KLUCZOWY parametr, estymowany
```

**Co oznacza `sigma_global`:** to oszacowana z danych zmienność log-intensywności
między komórkami. Gdy `sigma_global` jest małe → komórki są podobne i silnie
przyciągane do średniej globalnej (silny shrinkage). Gdy duże → komórki są
niezależne (model degeneruje się do no-poolingu). U nas posteriori
`sigma_global` ≈ 1,33 (przedział ~[1,18, 1,48]) — komórki realnie się różnią, ale
w skończonym stopniu, co daje umiarkowany shrinkage.

## Efekt shrinkage na konkretnym przykładzie

Shrinkage mierzymy na **skali logarytmicznej** (`alpha`), bo tam działa pooling;
na skali `lambda` skośność rozkładu zaciemnia obraz. Dla komórek ubogich w dane
(< 10 zdarzeń łącznie, n = 57 komórek):

- **punkt centralny**: Model 1 daje średnie `alpha ≈ 0,09`, Model 2 podciąga je
  do `alpha ≈ 0,20`, czyli **w stronę średniej globalnej** `mu_global ≈ 1,34`;
- **niepewność**: posteriorowe `sd(alpha)` spada średnio o **~10%**.

Dla komórek bogatych w dane (> 200 zdarzeń) shrinkage wynosi ~0% — mają dość
informacji, by zignorować hyperprior. Efekt jest więc **umiarkowany, ale dokładnie
ukierunkowany**: pomaga tam, gdzie trzeba, i nie psuje tego, co już dobrze
oszacowane. Umiarkowana skala wynika z tego, że dane Poissona są
„samoinformujące" (nawet jedna obserwacja ogranicza `lambda`), a większość komórek
w Japonii ma dużo danych.

## Tabela porównawcza

| Aspekt | Model 1 (no pooling) | Model 2 (partial pooling) |
|---|---|---|
| Założenie o komórkach | niezależne | uczą się od siebie |
| `sigma` (skala) | ustalone = 2 | estymowane (`sigma_global` ≈ 1,33) |
| Parametry | `alpha` (154) | `alpha` (154) + `mu_global`, `sigma_global` |
| Komórki ubogie w dane | szerokie, niestabilne | shrinkage do średniej |
| Uzasadnienie fizyczne | brak struktury | wspólna strefa tektoniczna |
| Parametryzacja MCMC | bezpośrednia | centered (silne dane) |

## Powiązane pliki

- [../models/model1_nopool.stan](../models/model1_nopool.stan) · [../models/model2_partial.stan](../models/model2_partial.stan)
- [../notebooks/04_model1_posterior.ipynb](../notebooks/04_model1_posterior.ipynb) · [../notebooks/05_model2_posterior.ipynb](../notebooks/05_model2_posterior.ipynb)
- Priory: [03_priory.md](03_priory.md) · Kod: [05_kluczowy_kod.md](05_kluczowy_kod.md)
