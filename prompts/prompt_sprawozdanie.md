## Prompt dla Claude — sprawozdanie DOCX

Napisz mi sprawozdanie akademickie w formacie Word (DOCX) z projektu bayesowskiego modelowania aktywności sejsmicznej Japonii. Sprawozdanie ma być gotowe do oddania prowadzącemu na kursie Data Analytics (AGH).

**Wygeneruj plik DOCX używając biblioteki python-docx. Daj mi kompletny skrypt Python który uruchomię lokalnie i który wygeneruje plik sprawozdanie.docx.**

Struktura i treść sprawozdania:

**1. Strona tytułowa**
- Tytuł: Bayesowskie modelowanie przestrzenne aktywności sejsmicznej — Japonia
- Autorzy: Paweł Majerczyk, Jakub Gicala
- Kurs: Data Analytics, AGH WIEiT ISZ, czerwiec 2026

**2. Wstęp**
Japonia leży na styku 4 płyt tektonicznych. Cel: mapa posteriorowej intensywności lambda_c na siatce 2x2 stopnie z pełną niepewnością bayesowską. Dane: USGS API 2000-2023, M>=4.0, ~33 106 zdarzeń. Zastosowania: ocena ryzyka, ubezpieczenia, infrastruktura.

**3. Dane i preprocessing**
- Siatka: 13x16 komórek = 208 geometrycznie, 154 aktywnych, 2086 obserwacji komórka-rok
- Tabela: łączna liczba zdarzeń 33106, mediana roczna 1251, rok 2011: 5746 zdarzeń (outlier Tohoku M9.0, komórka 6_10: 1403 zdarzeń)
- Wzmianka o mapie surowych epicentrów (Rysunek 1 w notebooku seismic_activity.ipynb)

**4. Modele**
Wiarygodność obu modeli: count_{c,y} ~ Poisson(lambda_c), log(lambda_c) = alpha_c

Model 1 (no pooling): alpha_c ~ N(1.8, 2.07), sigma ustalona z góry, komórki niezależne

Model 2 (partial pooling): alpha_c ~ N(mu_global, sigma_global), mu_global ~ N(1.8, 1), sigma_global ~ HalfNormal(1) — sigma_global estymowana z danych, efekt shrinkage

Tabela porównawcza obu modeli z kolumnami: sigma, pooling, liczba parametrów, komórki ubogie w dane, parametryzacja MCMC (centered dla M2).

**5. Priory — podejście split-in-time**
Prior nie może pochodzić z tych samych danych co likelihood (double-dipping). Nasze dane: 2000-2023. Stawka ~1200 M>=4/rok pochodzi z niezależnych katalogów historycznych:
- JMA: 1923-present, ~1000-1400/rok
- ISC: 1960-present
- USGS/NEIC pre-2000: ~1150/rok
- Utsu 1971, Kasahara 1981: 1100-1300/rok

Wyprowadzenie: mu_0 = log(1200/208) = log(5.77) ≈ 1.8. Prawo Gutenberga-Richtera (b≈1): sigma_0 = (8-1.8)/3 ≈ 2.07.

Tabela wrażliwości priora (5 scenariuszy — posterior stabilny: komórka aktywna zawsze lambda≈120, średnia lambda≈1.6).

Cytowania: Räty et al. 2023 (NHESS), Tu et al. 2025 (Annals of GIS), WAMBS checklist.

**6. Wyniki — Model 1**
Diagnostyki: 0 dywergencji, max R-hat=1.010, min ESS bulk=6841. Komórki ubogie w dane mają szerokie posteriory zdominowane przez prior.

**7. Wyniki — Model 2**
Diagnostyki: 0 dywergencji, max R-hat(alpha)=1.010, max R-hat(hypers)=1.000, min ESS=3895.
Hiperparametry posteriori (tabela): mu_global=0.28 (SD=0.189, HDI [-0.073; 0.627]), sigma_global=2.35 (SD=0.137, HDI [2.108; 2.612]).
sigma_global >> 0 potwierdza że hierarchia jest poparta danymi.
Shrinkage: 57 komórek ubogich (<10 zdarzeń), redukcja SD posteriora o ~10.6%.
PPC coverage: 84.7% (oczekiwane 95%) — niedopasowanie z powodu 2011.
Wzmianka o mapie prior vs posterior (Rysunek 2 w notebooku).

**8. Porównanie modeli**
WAIC: Model 1 lepszy o |delta_elpd|=7.2. LOO: Model 1 lepszy o |delta_elpd|=13.4.
Ale: Pareto-k do ~8, ~55 obserwacji z k>0.7 — wszystkie z klastra Tohoku 2011. Kryteria zdominowane przez outlier, wyniki niewiarygodne.
Wybieramy Model 2 z powodów merytorycznych: regularyzacja komórek ubogich, sigma_global>0 poparte danymi, jeden dodatkowy parametr, lepsza mapa z uczciwą niepewnością.
Wzmianka o mapie posteriorowej (Rysunek 3 w notebooku).

**9. Wnioski**
Wschód Japonii ~4x aktywniejszy od zachodu (strefy subdukcji). Ograniczenie: model stacjonarny nie opisuje 2011. Kolejny krok: model niestacjonarny lub z wskaźnikiem dla roku 2011.

**10. Literatura**
- Gutenberg & Richter (1944). Bulletin of the Seismological Society of America, 34(4), 185-188.
- Utsu (1971). Journal of the Faculty of Science, Hokkaido University, 3(3), 129-195.
- Räty et al. (2023). Natural Hazards and Earth System Sciences, 23, 2403-2423.
- Tu et al. (2025). Annals of GIS.
- USGS Earthquake Catalog API: https://earthquake.usgs.gov/fdsnws/event/1/

**Wymagania dotyczące skryptu:**
- Użyj python-docx
- Nagłówki sekcji jako Heading 1, podsekcje jako Heading 2
- Tabele z obramowaniem, nagłówki pogrubione
- Wzory matematyczne jako tekst (zapisz czytelnie np. alpha_c ~ N(1.8, 2.07), nie używaj LaTeX)
- Marginesy 2.5cm, czcionka Calibri 11pt, interlinia 1.15
- Strona tytułowa oddzielona page break
- Na końcu skryptu: doc.save('sprawozdanie.docx')
