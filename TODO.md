# TODO - Lista Zadań dla Pracy Magisterskiej

## ✅ Gotowe (Platforma Bazowa)

- [x] Struktura projektu
- [x] Model DTW/OTW (baseline)
- [x] System metryk
- [x] Ewaluator
- [x] Narzędzia do audio i MIDI
- [x] Dokumentacja i przewodniki

## 📝 Do Zrobienia - Kolejność Priorytetowa

### Etap 1: Setup i Podstawowe Testy (Tydzień 1)

- [ ] Zainstaluj zależności: `pip install -r requirements.txt`
- [ ] Pobierz dane testowe (MAESTRO lub własne)
- [ ] Uruchom `quickstart.py` na testowych danych
- [ ] Przejdź przez `notebooks/tutorial.ipynb`
- [ ] Sprawdź czy DTW działa poprawnie

**Deliverable**: Działający DTW na przynajmniej 1 utworze

### Etap 2: Ewaluacja Baseline (Tydzień 2)

- [ ] Przygotuj 10-20 utworów do testów
- [ ] Uruchom ewaluację DTW na wszystkich
- [ ] Uruchom ewaluację OTW na wszystkich
- [ ] Zbierz wyniki do tabeli
- [ ] Stwórz wykresy porównawcze

**Deliverable**: Wyniki DTW vs OTW na wielu utworach

### Etap 3: Analiza Robustness (Tydzień 3)

- [ ] Test tempo robustness (0.8x - 1.3x)
- [ ] Test z szumem
- [ ] Identyfikacja przypadków brzegowych:
  - [ ] Ozdobniki (tryle, arpeggia)
  - [ ] Fermaty
  - [ ] Skoki tempa
  - [ ] Powtórzenia
- [ ] Dokumentuj problemy które znalazłeś

**Deliverable**: Analiza gdzie i dlaczego modele zawodzą

### Etap 4: Implementacja CNN (Tydzień 4-6) - OPCJONALNE

⚠️ **UWAGA**: To jest najtrudniejsza część. Możesz pominąć jeśli brakuje czasu.

- [ ] Przygotuj dataset treningowy z MAESTRO
  - [ ] Wczytaj pary audio-MIDI
  - [ ] Stwórz DataLoader
  - [ ] Zaimplementuj augmentacje
- [ ] Zaimplementuj training loop w `models/cnn_model.py`
  - [ ] Loss function (zacznij od MSE)
  - [ ] Optimizer (Adam, lr=0.001)
  - [ ] Ewaluacja w trakcie treningu
- [ ] Trenuj model (GPU!)
  - [ ] Monitoring (loss, accuracy)
  - [ ] Early stopping
  - [ ] Zapisywanie checkpointów
- [ ] Ewaluuj wytrenowany model

**Deliverable**: Wytrenowany model CNN (lub uzasadnienie dlaczego został pominięty)

### Etap 5: Ostateczna Ewaluacja (Tydzień 7)

- [ ] Finalny zestaw testowy (najlepiej osobny od treningowego)
- [ ] Ewaluacja wszystkich dostępnych modeli
- [ ] Statystyki (średnia, mediana, std)
- [ ] Testy statystyczne (t-test między modelami)
- [ ] Wszystkie wykresy i tabele do pracy

**Deliverable**: Kompletne wyniki do rozdziału eksperymentalnego

### Etap 6: Praca Pisemna (Tydzień 8+)

- [ ] Rozdział: Metodologia
  - [ ] Opis datasetu
  - [ ] Opis metryk
  - [ ] Setup eksperymentu
- [ ] Rozdział: Wyniki
  - [ ] Tabele porównawcze
  - [ ] Wykresy
  - [ ] Analiza przypadków
- [ ] Rozdział: Dyskusja
  - [ ] Interpretacja wyników
  - [ ] Porównanie z literaturą
  - [ ] Ograniczenia
- [ ] Rozdział: Wnioski
  - [ ] Podsumowanie
  - [ ] Future work

**Deliverable**: Gotowy draft pracy

## 🎯 Ścieżki Alternatywne

### Ścieżka MINIMALNA (jeśli brakuje czasu)
- Etap 1 + Etap 2 + Etap 5 (z DTW/OTW tylko)
- Propozycja ulepszeń teoretyczna (bez implementacji)
- ~4 tygodnie pracy

### Ścieżka ŚREDNIA (zalecana)
- Etap 1 + Etap 2 + Etap 3 + Etap 5
- Dokładna analiza problemów
- Propozycja Transformer (teoretyczna)
- ~6 tygodni pracy

### Ścieżka PEŁNA (dla ambitnych)
- Wszystkie etapy włącznie z CNN
- Ewentualnie implementacja Transformer
- ~8-10 tygodni pracy

## 📊 Metryki Sukcesu

**Minimum do obrony**:
- ✅ Działający DTW i OTW
- ✅ Ewaluacja na >10 utworach
- ✅ Wykresy i tabele
- ✅ Analiza wyników

**Dobra praca**:
- ✅ Wszystko powyżej +
- ✅ Testy robustness
- ✅ Analiza przypadków brzegowych
- ✅ Propozycje ulepszeń

**Znakomita praca**:
- ✅ Wszystko powyżej +
- ✅ Wytrenowany CNN lub inny model ML
- ✅ Oryginalna propozycja (Transformer)
- ✅ Głęboka analiza teoretyczna

## 🚨 Red Flags - Kiedy Prosić o Pomoc

- DTW nie działa po 2 dniach prób → poproś promotora
- Brak danych po tygodniu → użyj syntetycznych
- Training CNN nie konwerguje → zacznij od prostszego problemu
- Brakuje czasu → przejdź na ścieżkę minimalną

## 💡 Wskazówki

1. **Dokumentuj wszystko na bieżąco** - commit do git codziennie
2. **Zapisuj wyniki eksperymentów** - użyj `ExperimentLogger` ze SNIPPETS.md
3. **Twórz wykresy od razu** - łatwiej się motywować widząc postępy
4. **Nie perfekjonizuj** - 80% jakości to wystarczająco dobre
5. **Proś o feedback wcześnie** - pokaż promotorowi wyniki co 2 tygodnie

## 📅 Przykładowy Timeline (8 tygodni)

| Tydzień | Zadania | Cel |
|---------|---------|-----|
| 1 | Setup + DTW test | Działająca platforma |
| 2 | Ewaluacja baseline | Pierwsze wyniki |
| 3 | Robustness tests | Zrozumienie problemów |
| 4-6 | CNN (opcjonalnie) | Model ML |
| 7 | Finalna ewaluacja | Wszystkie wyniki |
| 8+ | Pisanie | Draft pracy |

## ✨ Bonus Tasks (jeśli masz czas)

- [ ] Implementuj Transformer (z SKILL.md)
- [ ] Stwórz demo aplikację (Streamlit/Gradio)
- [ ] Opublikuj kod na GitHub
- [ ] Napisz blog post o wynikach
- [ ] Przygotuj prezentację dla konferencji

---

**Pamiętaj**: Ukończona praca jest lepsza niż perfekcyjna ale nieukończona!

Powodzenia! 🎓
