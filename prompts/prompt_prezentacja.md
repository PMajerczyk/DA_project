## Prompt dla Claude — prezentacja HTML (Reveal.js)

Zrób mi prezentację w formacie HTML (jeden plik, self-contained) na temat bayesowskiego modelowania aktywności sejsmicznej Japonii. Prezentacja ma 10 slajdów na ~8 minut obrony akademickiej na kursie Data Analytics (AGH).

Użyj biblioteki Reveal.js (CDN). Styl: ciemne tło (motyw "black" lub "night"), czytelne wzory matematyczne przez MathJax (CDN). Gdzie zaznaczono [FIGURA] wstaw estetyczny placeholder — szary prostokąt z opisem. Język angielski, styl akademicki.

---

**Slajd 1 — Tytuł**
- Bayesian Spatial Modelling of Seismic Activity — Japan
- Paweł Majerczyk, Jakub Gicala
- Data Analytics, AGH WIEiT ISZ, June 2026

**Slajd 2 — Motywacja i cel**
- Japonia na styku 4 płyt tektonicznych → jeden z najbardziej aktywnych sejsmicznie regionów świata
- Cel: mapa posteriorowej intensywności λ_c z kwantyfikowaną niepewnością
- Zastosowania: ocena ryzyka sejsmicznego, ubezpieczenia, planowanie infrastruktury
- Podejście: bayesowski model przestrzenny na siatce 2°×2°

**Slajd 3 — Dane**
- Źródło: USGS Earthquake Catalog API, 2000-2023, M≥4.0
- 33 106 zdarzeń, siatka 2°×2° (208 komórek geometrycznie, 154 aktywnych)
- 2086 obserwacji komórka-rok
- Outlier: 2011 Tohoku M9.0 → 5746 zdarzeń (mediana innych lat: 1251)
- [FIGURA: mapa surowych epicentrów nałożona na siatkę 2°×2°]

**Slajd 4 — Dwa modele**
Wspólna wiarygodność: count_{c,y} ~ Poisson(λ_c), log(λ_c) = α_c

Tabela porównawcza:
| | Model 1 — No Pooling | Model 2 — Partial Pooling |
|---|---|---|
| Prior α_c | N(1.8, 2.07) | N(μ_global, σ_global) |
| Skala σ | ustalona = 2.07 | estymowana (HalfNormal(1)) |
| Komórki | niezależne | pożyczają siłę (shrinkage) |
| Parametry | α (154) | α (154) + μ_global, σ_global |

Kluczowa różnica strukturalna: σ_global wolny vs. ustalony

**Slajd 5 — DAG — struktura generatywna**
Opis słowny struktury obu modeli (nie rysuj — opisz tekstowo z formatowaniem):

Model 1: mu_0=1.8, sigma_0=2.07 (fixed) → alpha_c → lambda_c → count_{c,y}

Model 2: mu_0=1.8 → mu_global → alpha_c → lambda_c → count_{c,y}
         sigma_hp=1.0 → sigma_global ↗

Plate notation: c=1..154, y=2000..2023

Kluczowy wniosek: jedyna różnica to czy sigma jest węzłem obliczanym czy wolnym parametrem

**Slajd 6 — Priory: podejście split-in-time**
Problem: prior z tych samych danych = double-dipping → zbyt wąskie posteriory

Rozwiązanie — split-in-time:
- Dane analizy: 2000-2023 (USGS)
- Prior z niezależnych katalogów historycznych (pre-2000): JMA od 1923, ISC od 1960, ~1200 M≥4/rok

Wyprowadzenie bez data-leakage:
- μ_0 = log(1200 / 208 komórek) = log(5.77) ≈ 1.8
- σ_0 = (8 - 1.8) / 3 ≈ 2.07  [prawo Gutenberga-Richtera, b≈1]

Analiza wrażliwości: posterior stabilny przy zmianie μ_0 ±1 i σ_0 ×2 (Räty et al. 2023, WAMBS)

**Slajd 7 — Wyniki: Model 1**
Diagnostyki MCMC (4 łańcuchy, 4000 próbek):
- Dywergencje: 0
- max R-hat: 1.010 (próg < 1.01) ✓
- min ESS bulk: 6841 (próg > 400) ✓

Obserwacja: komórki ubogie w dane (<10 zdarzeń) mają szerokie posteriory zdominowane przez prior — brak mechanizmu regularyzacji

**Slajd 8 — Wyniki: Model 2 + shrinkage**
Diagnostyki: 0 dywergencji, R-hat=1.000 dla hiperparametrów, ESS=3895

Hiperparametry posteriori:
- μ_global = 0.28 [HDI: -0.07; 0.63]
- σ_global = 2.35 [HDI: 2.11; 2.61] >> 0 → hierarchia potwierdzona danymi

Shrinkage (57 komórek ubogich, <10 zdarzeń):
- Redukcja SD posteriora: ~10.6%
- α przesuwa się od -2.19 (M1) ku μ_global=0.28

[FIGURA: mapa prior (szara) vs posterior (kolorowa) — co dane nauczyły model]

**Slajd 9 — Porównanie modeli**
WAIC: Model 1 lepszy o |Δelpd|=7.2
LOO:  Model 1 lepszy o |Δelpd|=13.4

ALE: Pareto-k do ~8, ~55 obserwacji z k>0.7 → kryteria niewiarygodne
Wszystkie skrajne k: klaster Tohoku 2011 → model stacjonarny nie opisuje jednorazowego skoku

Decyzja: wybieramy Model 2 z powodów merytorycznych:
- regularyzacja komórek ubogich ✓
- σ_global >> 0 poparte danymi ✓
- tylko 1 dodatkowy parametr ✓
- lepsza, gładsza mapa z uczciwą niepewnością ✓

**Slajd 10 — Wnioski**
[FIGURA: mapa posteriorowa Model 2 — intensywność sejsmiczna Japonii]

Kluczowe wyniki:
- Wschód Japonii ~4× aktywniejszy od zachodu (strefy subdukcji Pacyfiku)
- Model 2 (partial pooling) daje lepszy produkt przy niemal zerowym koszcie predykcyjnym
- Priory wyznaczone bez data-leakage (split-in-time, JMA/ISC pre-2000)

Ograniczenia i kierunki dalsze:
- Model stacjonarny nie opisuje roku 2011 (PPC coverage: 84.7% zamiast ~95%)
- Następny krok: model niestacjonarny z trendem czasowym lub wskaźnikiem Tohoku

---

**Wymagania techniczne:**
- Jeden plik HTML, wszystkie zasoby przez CDN (Reveal.js, MathJax)
- Motyw ciemny (black lub night)
- Tabele z CSS — czytelne na ciemnym tle
- Placeholdery [FIGURA] jako div z szarym tłem i opisem tekstowym
- Nawigacja klawiaturą (strzałki), licznik slajdów widoczny
- Rozmiar czcionki czytelny przy projektorze (min 24px dla treści)
