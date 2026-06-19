# Bayesowskie modelowanie aktywności sejsmicznej — Japonia

## Co robimy i po co

Projekt buduje **bayesowski model przestrzenny liczby trzęsień ziemi** (magnituda
≥ 4.0) na obszarze Japonii w latach 2000–2023. Powierzchnię kraju dzielimy na
siatkę komórek 2°×2° i dla każdej komórki szacujemy roczną intensywność
sejsmiczną `lambda` wraz z pełną niepewnością. Porównujemy dwa modele: prosty
model **bez poolingu** (każda komórka niezależnie) z modelem **hierarchicznym z
partial poolingiem** (komórki „pożyczają sobie siłę"). Efektem końcowym jest
kolorowa mapa posteriorowej aktywności z mapą niepewności — narzędzie przydatne
w ocenie ryzyka sejsmicznego, ubezpieczeniach i planowaniu infrastruktury.

## Struktura repozytorium

```
data/
  raw/earthquakes_japan.csv        # ~33 tys. zdarzeń pobranych z USGS API
  processed/grid_annual_counts.csv # liczba zdarzeń per komórka per rok
  processed/grid_metadata.csv      # metadane komórek (środki, sumy)
  processed/idata_model{1,2}.nc    # zapisane wyniki MCMC (ArviZ)
models/
  model1_nopool.stan               # model 1 — sigma ustalone (no pooling)
  model2_partial.stan              # model 2 — sigma_global estymowane
notebooks/
  01_data_acquisition.ipynb        # źródło danych + mapa surowa
  02_preprocessing.ipynb           # agregacja do siatki + EDA
  03_priors.ipynb                  # uzasadnienie priorów + prior predictive
  04_model1_posterior.ipynb        # fit + diagnostyka modelu 1
  05_model2_posterior.ipynb        # fit + shrinkage + mapa modelu 2
  06_comparison.ipynb              # WAIC + LOO + ocena końcowa
report/main_report.ipynb           # skonsolidowany raport
utils/                             # display.py, data_prep.py, download_data.py
docs/                              # ta dokumentacja (PL)
```

## Jak uruchomić projekt

```bash
python3 -m venv venv && source venv/bin/activate
pip install cmdstanpy arviz geopandas pandas numpy matplotlib \
            ipykernel jupyter nbconvert h5netcdf netcdf4
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

python utils/download_data.py            # pobranie danych z USGS
python run_notebook.py notebooks/01_data_acquisition.ipynb \
  notebooks/02_preprocessing.ipynb notebooks/03_priors.ipynb \
  notebooks/04_model1_posterior.ipynb notebooks/05_model2_posterior.ipynb \
  notebooks/06_comparison.ipynb
```

Notebooki uruchamiamy **w kolejności numerycznej** — notebooki 04/05 czytają
przetworzoną siatkę z 02 i zapisują wyniki MCMC, które konsumuje notebook 06.

## Krótkie wyniki

Kryteria informacyjne dają **niejednoznaczny** obraz: WAIC nieznacznie preferuje
model hierarchiczny (Model 2), a PSIS-LOO odwraca ranking i wskazuje Model 1 —
przy czym **oba dają ostrzeżenia**, bo zdominowane są przez ekstremalny rok 2011
(Tohoku M9.0, Pareto k do ~9). Różnice są na poziomie ~1,6 błędu standardowego,
więc statystycznie modele są blisko. **Wybieramy Model 2** ze względów
merytorycznych: zapewnia regularyzację (shrinkage), jego hierarchia jest poparta
danymi (`sigma_global` ≈ 1,33 > 0), a wynikowa mapa jest gładsza i ma uczciwie
oszacowaną niepewność. Wschodnie wybrzeże Japonii ma ~4× wyższą aktywność niż
zachodnie — co jest fizycznie poprawne.

## Powiązane pliki

- Pełny opis: [01_opis_projektu.md](01_opis_projektu.md)
- Modele: [02_modele.md](02_modele.md)
- Priory: [03_priory.md](03_priory.md)
- Wyniki: [04_wyniki.md](04_wyniki.md)
- Kluczowy kod: [05_kluczowy_kod.md](05_kluczowy_kod.md)
- Notebooki: [../notebooks/](../notebooks/) · Raport: [../report/main_report.ipynb](../report/main_report.ipynb)
