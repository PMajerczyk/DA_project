# Zadanie: usunąć data-leakage z priora i podeprzeć go literaturą (Japonia)

## Cel
Środek priora na log-intensywność jest dziś wyprowadzony z naszej własnej próby
(double-dipping). Trzeba zastąpić jego **uzasadnienie** wiedzą dziedzinową dla
Japonii, nie zmieniając elementów priora, które są poprawne (skala, kształt,
ogon). Uwaga: docelowa wartość liczbowa wyjdzie zbliżona — celem jest poprawna
proweniencja, nie inny wynik.

## Problem (dokładna lokalizacja)
W `03_priors.md` środek `mu_0 = 2` dobrano na podstawie **mediany ~6 i średniej
~16 policzonych z danych**, które potem wchodzą do modelu przez wiarygodność
(Poisson na tych samych `count`). Te same dane liczą się dwa razy → sztucznie
zawężony, zbyt pewny posterior. Zdanie „liczby pochodzą z fizyki **i danych**" to
opis usterki — człon „i danych" jest problemem. Skażenie propaguje się do
Modelu 2 (`mu_global ~ Normal(2, 1)`).

## Zmiany do wprowadzenia
1. **Środek priora — wyprowadzić z zewnętrznej stawki dla Japonii, nie z próby.**
   Wyprowadzenie do udokumentowania w `03_priors.md`:
   - Publikowana roczna stawka dla Japonii: **~1 200 zdarzeń M ≥ 4 / rok**
     (źródło zewnętrzne, archiwum od 1900 — NIE nasza próba 2000–2023).
   - Liczba komórek z samej geometrii siatki (nie z danych):
     lat 24–50° → 26°/2 = 13, lon 122–154° → 32°/2 = 16, razem **208 komórek**.
   - Stawka na komórkę-rok (uniform): `1200 / 208 ≈ 5,8` → `mu_0 = log(5,8) ≈ 1,8`.
   - Stąd `mu_0 ≈ 1,8` (exp ≈ 5,8). Wartość ~2,0 też jest do obrony w granicach
     zaokrąglenia; sigma zostaje szeroka, bo sejsmiczność jest silnie
     skoncentrowana i „typowa komórka" jest z natury rozmyta.
   - **Krytyczne:** w uzasadnieniu cytować stawkę zewnętrzną i geometrię siatki.
     Ani jedna liczba w wyprowadzeniu środka nie może pochodzić z `count` w próbie.
2. **Uzasadnienie kształtu** — zamienić „kształt odpowiada rzeczywistemu
   rozkładowi z notebooka 02" (argument kołowy) na wyprowadzenie z prawa G-R:
   prawoskośność i reguła „×10 na jednostkę magnitudy" wynikają z b ≈ 1.
   Potwierdzenie dla Japonii: proporcje pasm M≥4 / M≥5 / M≥6 (~1200 / ~149 /
   ~12,7) dają b ≈ 1.
3. **Model 2** — przeliczyć hyperprior tak, by `mu_global` brał środek z pkt 1
   (≈1,8), a nie z odziedziczonej „dwójki".
4. **Tekst** — usunąć/zmienić każde zdanie uzasadniające parametry priora
   zgodnością z obserwacjami (np. obwiednia „zawiera obserwacje" jako argument za
   doborem środka). Zgodność z danymi to osobny krok (PPC / posterior), nie
   uzasadnienie priora.
5. **Stałe** — zaktualizować `PRIOR_MU`, `PRIOR_SIGMA_M1` w `utils/data_prep.py`
   i wszystkie miejsca, które z nich korzystają (modele, notebook 03). Jeśli
   `mu_0` zmienia się z 2 na 1,8, przeliczyć też `sigma_0 = (8 − 1,8)/3 ≈ 2,07`.
6. (Opcjonalnie) udokumentować alternatywę **Gamma–Poisson** (sprzężoną) na
   `lambda` jako wariant z interpretacją parametrów jako pseudo-obserwacji —
   to standardowa rama w sejsmologii dla regionów modelowanych na siatce.

## Czego NIE ruszać (to jest poprawne)
- Górna granica `U ≈ 3000` (Tohoku ~1400 + zapas) i dolna `L = 1` — skala z fizyki
  i ze znanego ekstremum, legalne.
- Dodatniość przez `exp()`, log-link, ogon dopuszczający Tohoku.
- Szeroka sigma pokrywająca ~`[0, 8]` na skali log.
- Geometria siatki 2°×2° i bounding box — pokrywają się z domeną katalogu JMA,
  więc są zewnętrznie umocowane.
- Sama struktura sprawdzeń prior predictive (parametrów i pomiarów).

## Źródła do zacytowania w uzasadnieniu
- Gutenberg & Richter (1944): `log10 N(M) = a − bM`; b ≈ 1 dla płytkich zdarzeń
  tektonicznych; `a` = regionalny poziom sejsmiczności.
- Stawka dla Japonii (zewnętrzna, archiwum od 1900): M≥4 ~1 200/rok,
  M≥5 ~149/rok, M≥6 ~12,7/rok — kotwica środka i potwierdzenie b ≈ 1.
- Globalna stawka M≥4 ~10^4/rok — kontekst rzędu wielkości dla ogona.
- Gupta & Baker (2017), *Structural Safety* — siatka komórek, gamma-prior na
  tempo Poissona w komórce (rama Gamma–Poisson).
- Bayesowski ETAS (np. arXiv:2109.05894) — priory gamma o różnej
  nieinformatywności; wynik odporny na rozsądne zmiany priorów → środka nie
  trzeba (i nie należy) wyciągać z danych.

## Weryfikacja po zmianach
- Wyprowadzenie środka wskazuje wyłącznie źródło zewnętrzne + geometrię siatki;
  żadna liczba nie pochodzi z `count` w próbie.
- Prior predictive nadal: wartości dodatnie, mediana rzędu kilku zdarzeń/rok,
  ogon (99,9%) w niskich tysiącach, prawie nic > ~3000.
- `grep` na repo: brak `np.median`/`np.mean`/`.describe()` na danych
  obserwowanych w ścieżce wyznaczania parametrów priora.
- Spójność `PRIOR_MU`/`PRIOR_SIGMA_M1` z `03_priors.md` i z hyperpriorem Modelu 2.
