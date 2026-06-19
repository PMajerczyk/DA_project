# Priory

## Co to jest prior i po co go dobieramy

W podejściu bayesowskim **prior** to rozkład prawdopodobieństwa wyrażający naszą
wiedzę o parametrze *przed* zobaczeniem danych. Po połączeniu z funkcją
wiarygodności (danymi) daje rozkład *posteriori*. Dobry prior powinien być
**słabo informujący**: kodować właściwą skalę i znak zjawiska (żeby model nie
rozważał wartości absurdalnych), ale nie narzucać odpowiedzi tam, gdzie dane mają
głos. W modelu Poissona z linkiem logarytmicznym prior nakładamy na
log-intensywność `alpha = log(lambda)`. To miejsce wymaga ostrożności: łagodny z
pozoru rozkład normalny na `alpha` może implikować ekstremalne intensywności po
przejściu przez `exp()`.

## Jak obliczyliśmy górną granicę z danych

Punktem wyjścia jest fizyczna górna granica liczby zdarzeń w komórce. Najbardziej
aktywna obserwowana komórka-rok to rejon Tohoku w 2011 r. z ~1400 zdarzeniami
M ≥ 4.0. Dając zapas na jeszcze większą sekwencję wstrząsów wtórnych, przyjmujemy
górną granicę `U ≈ 3000` zdarzeń/rok. Dolna sensowna granica dla aktywnej komórki
to ~1 zdarzenie/rok, `L = 1`. Na skali logarytmicznej daje to przedział
`log(L) = 0` do `log(U) ≈ 8`.

## Dlaczego Normal(2, 2) na log-intensywność

Pierwotny brief sugerował `Normal(3.4, 1.5)`, zakładając „typową" komórkę o ~30
zdarzeniach/rok (`log(30) ≈ 3.4`). Po analizie *rzeczywistych* danych okazało się,
że typowa komórka jest znacznie spokojniejsza: mediana wynosi ~6, a średnia ~16
zdarzeń na komórkę-rok. Dlatego **wyśrodkowaliśmy prior niżej**, na
`mu_0 = 2` (czyli `exp(2) ≈ 7,4` zdarzenia/rok — wartość między medianą a
średnią). Odchylenie dobieramy tak, by ~±3 sigma pokryło zakres `[0, 8]` na skali
log:

```
sigma_0 = (8 - 2) / 3 = 2
```

Stąd końcowy prior **`alpha ~ Normal(2, 2)`** (ustalony w Modelu 1). W Modelu 2
ten sam rozsądek przenosi się na hyperprior: `mu_global ~ Normal(2, 1)`,
`sigma_global ~ HalfNormal(1)` (większość masy na `[0, 2.5]`, zgodnie z ustaloną
wartością 2 w Modelu 1). Krótko: liczby pochodzą z fizyki i danych, nie z
arbitralnego wyboru.

## Co pokazują prior predictive checks i jak je interpretować

Wykonujemy dwa sprawdzenia (oba wymagane przez kryteria):

1. **Prior predictive PARAMETRÓW** — losujemy `alpha` z priora i patrzymy na
   implikowaną intensywność `lambda = exp(alpha)`. Interpretacja: mediana powinna
   leżeć w okolicach kilku–kilkunastu zdarzeń/rok, górny ogon (95–99,9%) sięgać
   setek do niskich tysięcy (żeby *dopuścić* Tohoku, ale nie czynić go normą),
   a wszystkie wartości być dodatnie. Tak właśnie jest.
2. **Prior predictive POMIARÓW** — przepuszczamy prior przez wiarygodność:
   losujemy `count ~ Poisson(exp(alpha))`. Interpretacja: symulowane liczby to
   nieujemne liczby całkowite, silnie prawoskośne (większość komórek spokojna,
   nieliczne bardzo aktywne), prawie nigdy nie przekraczające granicy ~3000.
   Kształt odpowiada rzeczywistemu rozkładowi z notebooka 02.

Dodatkowo symulujemy całe „fałszywe" zbiory danych i nakładamy ich rozkład na
dane obserwowane — obwiednia priorowa powinna *zawierać* obserwacje. Uwaga: nasz
prior dopuszcza w ogonie wartości do ~3000 (powyżej orientacyjnego progu 1000 z
listy kontrolnej), i jest to **celowe** — bez tego nie objęlibyśmy fizycznie
realnego Tohoku (~1400).

## Kod z wyjaśnieniem

```python
NSIM = 20000
alpha = np.random.normal(2, 2, NSIM)   # losujemy log-intensywność z priora
lam   = np.exp(alpha)                   # przejście na skalę intensywności (>0)
count = np.random.poisson(lam)          # symulowane pomiary (liczby całkowite)
# sprawdzenia:
(count < 0).any()        # False — exp() gwarantuje dodatniość
np.quantile(lam, 0.999)  # ~niskie tysiące — dopuszcza Tohoku, nie czyni normą
np.median(count)         # kilka zdarzeń — typowa spokojna komórka
```

Każda linijka odwzorowuje strukturę modelu: prior na `alpha` → deterministyczne
`exp` → losowy Poisson. To dokładnie ten sam łańcuch, który Stan próbkuje przy
estymacji posteriori.

## Powiązane pliki

- [../notebooks/03_priors.ipynb](../notebooks/03_priors.ipynb) — pełne sprawdzenia prior predictive
- [../utils/data_prep.py](../utils/data_prep.py) — stałe `PRIOR_MU`, `PRIOR_SIGMA_M1`
- Modele: [02_modele.md](02_modele.md)
