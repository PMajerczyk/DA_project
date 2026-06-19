# Wyniki i ich interpretacja

## Wyniki WAIC i LOO — co oznaczają liczby

Kryteria informacyjne szacują **przewidywaną trafność modelu poza próbą** przez
tzw. `elpd` (expected log pointwise predictive density) — im wyższe, tym lepiej.

- **WAIC:** wygrywa Model 2 z przewagą `elpd_diff ≈ 61` przy błędzie
  `dse ≈ 37` — czyli tylko ~1,6 błędu standardowego. To różnica realna, ale
  **nierozstrzygająca**.
- **PSIS-LOO:** ranking **się odwraca** — wygrywa Model 1 z `elpd_diff ≈ 26`
  (`dse ≈ 16`, też ~1,6 SE).

Co znaczą poszczególne wielkości:
- `elpd_diff` — różnica trafności względem najlepszego modelu (0 dla zwycięzcy);
- `dse` — błąd standardowy tej różnicy; przewaga jest „pewna" dopiero, gdy
  `elpd_diff` przekracza 2–3× `dse`. Tu jest tylko ~1,6×, więc modele są blisko;
- `p_waic`/`p_loo` — efektywna liczba parametrów (miara złożoności);
- `weight` — waga modelu w uśrednianiu (tu ~0,5/0,5, czyli remis).

## Dlaczego wybraliśmy Model 2 (nietechnicznie)

Kryteria są remisowe i niewiarygodne (patrz niżej), więc decyzję opieramy na
**merytoryce, nie na rankingu**. Model 2 lepiej radzi sobie z komórkami ubogimi w
dane (regularyzuje je zamiast dawać dzikie przedziały), jego hierarchia jest
poparta danymi, kosztuje tylko jeden dodatkowy parametr i daje gładszą,
bardziej wiarygodną mapę z uczciwą niepewnością — a tego właśnie potrzebuje mapa
ryzyka. Innymi słowy: za niemal zerową cenę w trafności predykcyjnej dostajemy
lepszy, stabilniejszy produkt.

## Co oznacza mapa posteriorowej aktywności

Mapa choropletowa (notebook 05) pokazuje posteriorową **średnią intensywność**
`lambda` dla każdej komórki. Widać wyraźny gradient: wschodnie wybrzeże ma średnio
~17,8 zdarzenia/rok na komórkę, zachodnie ~4,3 — czyli ~4× więcej na wschodzie.
To zgadza się z geofizyką (strefy subdukcji wzdłuż wschodniego wybrzeża).
Towarzysząca mapa szerokości przedziałów pokazuje, **gdzie jesteśmy niepewni** —
zwykle tam, gdzie danych jest mało.

## Interpretacja sigma_global — jak bardzo komórki się różnią

`sigma_global` (posteriori ≈ 1,33; przedział ~[1,18, 1,48]) to oszacowana
zmienność log-intensywności między komórkami. Wartość wyraźnie większa od zera
**potwierdza, że komórki realnie się różnią** — gdyby były identyczne,
`sigma_global` dążyłoby do 0. Na skali intensywności `sigma_global ≈ 1,33`
oznacza, że typowa komórka różni się od średniej globalnej o czynnik ~`e^1.33 ≈
3,8` w górę lub w dół. To umiarkowana heterogeniczność: na tyle duża, że warto
mieć osobne `alpha` per komórka, ale na tyle skończona, że pooling pomaga.

## Omówienie roku 2011 (Tohoku) — outlier

11 marca 2011 r. trzęsienie Tohoku (M9.0) i jego wstrząsy wtórne podniosły liczbę
zdarzeń w komórkach wschodniego wybrzeża wielokrotnie ponad ich długoterminową
średnią (komórka `6_10`: 1403 zdarzenia w 2011 r.). Nasze modele są
**stacjonarne** — zakładają jedną stałą intensywność per komórka na wszystkie
lata — więc z definicji nie potrafią uchwycić jednorazowego skoku. W posterior
predictive 2011 r. wyróżnia się jako największe odchylenie. **Co z tym robimy:**
nie usuwamy go ani nie „naprawiamy" — traktujemy jako udokumentowany outlier i
uczciwie komentujemy, że poprawny krok dalej to model niestacjonarny lub z
członem na wstrząsy wtórne.

## Ostrzeżenia Pareto k-hat — co oznaczają i czy są problemem

PSIS-LOO dla każdej obserwacji liczy diagnostykę **Pareto k**. Gdy `k > 0,7`,
oszacowanie LOO dla tego punktu jest niewiarygodne. U nas `k` sięga ~9 (!), a
51 obserwacji ma `k > 0,7`. Najgorsze punkty to komórki o najwyższych liczbach
zdarzeń, na czele z klastrem Tohoku 2011 (~22% punktów >0,7 pochodzi z 2011, a
absolutnie najwyższe `k` to właśnie komórki Tohoku). **Czy to problem?** Tak — i
dlatego ranking LOO traktujemy z rezerwą. Te same wpływowe, źle dopasowane
obserwacje wywołują też ostrzeżenie WAIC. Wniosek: oba kryteria są zdominowane
przez garść komórek, których model stacjonarny nie opisuje — stąd ich
rozbieżność i niewiarygodność.

## Wnioski końcowe

**Roczna intensywność trzęsień M ≥ 4.0 w Japonii jest silnie zróżnicowana
przestrzennie (wschód ~4× aktywniejszy od zachodu), a hierarchiczny model z
partial poolingiem (Model 2) daje najlepszy kompromis między trafnością a
stabilnością oszacowań — przy zastrzeżeniu, że żaden ze stacjonarnych modeli nie
opisuje wyjątkowego roku 2011.**

## Powiązane pliki

- [../notebooks/06_comparison.ipynb](../notebooks/06_comparison.ipynb) — WAIC, LOO, ocena końcowa
- [../notebooks/05_model2_posterior.ipynb](../notebooks/05_model2_posterior.ipynb) — mapa i shrinkage
- Modele: [02_modele.md](02_modele.md) · Kod: [05_kluczowy_kod.md](05_kluczowy_kod.md)
