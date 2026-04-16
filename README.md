# NBA Predictor

Aplikace předpovídá výsledky zápasů NBA pomocí modelu strojového učení trénovaného na reálných historických datech. Výsledky zobrazuje ve webovém dashboardu, který se spustí automaticky.

---

## Spuštění

### Varianta A — EXE (bez Pythonu)

1. Stáhni `NBAPredictor.zip` z [Releases](../../releases/latest)
2. Rozbal archiv
3. Spusť **`NBAPredictor.exe`**
4. Prohlížeč se otevře automaticky na `http://127.0.0.1:5000`

Při prvním spuštění aplikace stáhne data (~2 minuty). Při dalším spuštění se otevře okamžitě.

### Varianta B — ze zdrojového kódu

Požadavky: Python 3.10+ ([python.org](https://www.python.org/downloads/))

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Sestavení EXE

```
vendor\build_exe.bat
```

Vytvoří `dist/NBAPredictor/NBAPredictor.exe` včetně ZIP archivu pro distribuci.

---

## Základní funkce

| Tab v aplikaci | Co zobrazuje |
|----------------|-------------|
| **Predictions** | Predikce výsledků nadcházejících zápasů s pravděpodobností výhry |
| **Model Comparison** | Porovnání přesnosti natrénovaných modelů |
| **Injury Report** | Aktuální zranění hráčů a dostupnost sestavy |

Tlačítko **Refresh predictions** aktualizuje data a vygeneruje nové předpovědi.

---

## Zdroje dat

Data jsou 100% reálná, sesbíraná vlastním kódem ze tří veřejných zdrojů:

### 1. NBA Stats API (`data/nba_stats.py`)
- URL: `https://stats.nba.com/stats/leaguegamelog`
- Co stahuje: herní statistiky všech týmů za aktuální sezónu — body, procento střelby, doskoky, asistence, ztráty, krádeže, bloky
- Jak: HTTP požadavky přes knihovnu `nba_api`

### 2. Basketball Reference (`data/basketball_ref.py`)
- URL: `https://www.basketball-reference.com/teams/{TYM}/{ROK}/gamelog/`
- Co stahuje: historické game logy posledních 5 sezón (2021–2026) pro všech 30 týmů
- Jak: web scraping pomocí `cloudscraper` + `pandas.read_html()`

### 3. ESPN Injury API (`data/injuries.py`)
- URL: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`
- Co stahuje: aktuální zprávy o zraněních a dostupnosti hráčů
- Jak: HTTP GET požadavek, parsování JSON odpovědi

**Celkový objem dat:** přibližně 12 000+ herních záznamů

---

## Předzpracování dat (`data/processor.py`)

### Čištění
- Odstranění řádků s chybějícím datem nebo výsledkem
- Normalizace zkratek týmů — NBA API a Basketball Reference používají různé zkratky (např. `BKN` → `BRK`, `PHX` → `PHO`)
- Převod výsledku `W/L` na binární proměnnou `Win` (1 = výhra, 0 = prohra)

### Sloučení zdrojů
- Data z obou zdrojů jsou spojena přes sloupce `Team` + `Date`
- Duplicity odstraněny, preferována NBA API data (aktuálnější)

### Feature engineering (`features/`)
Ze surových statistik jsou vypočítány příznaky:

| Příznak | Výpočet |
|---------|---------|
| `roll_PTS` | Klouzavý průměr bodů za posledních 10 zápasů (`shift(1)` zabraňuje data leakage) |
| `form` | Podíl výher v posledních 5 zápasech |
| `streak` | Délka aktuální série výher (+) nebo proher (-) |
| `rest_days` | Počet dní od posledního zápasu |
| `elo` | ELO rating (začíná 1500, K=20, domácí bonus 100 bodů) |
| `diff_*` | Diferenciál každého příznaku: tým − soupeř |
| `is_home` | 1 = domácí tým |
| `is_b2b` | 1 = back-to-back zápas (rest ≤ 1 den) |

Celkem **16 příznaků**, každý jako diferenciál (pohled z perspektivy domácího týmu).

---

## Modely (`models/`)

Trénují se automaticky 2 modely, porovnají se a uloží se nejlepší:

| Model | Třída | Ladění |
|-------|-------|--------|
| Logistic Regression | `LogisticModel` | GridSearchCV (parametr C, solver) |
| Random Forest | `RandomForestModel` | RandomizedSearchCV (50 iterací) |

Hodnocení probíhá na **časovém splitu 80/20** — starší zápasy trénují, novější testují.  
Metriky: Accuracy, ROC-AUC, F1 Score, Log Loss, 5-fold Cross-validation.  
Nejlepší model (podle ROC-AUC) se uloží do `storage/trained/best_model.pkl`.

---

## Postup tvorby modelu

Kompletní postup — načtení dat, předzpracování, feature engineering, trénování a vyhodnocení — je zdokumentován v notebooku:

**`docs/NBA_Predictor_Colab.ipynb`** — otevři na [colab.research.google.com](https://colab.research.google.com)

---

## Struktura projektu

```
NBA_Predictor/
├── main.py                  vstupní bod aplikace
├── app.py                   Flask server
├── config.py                konfigurace (cesty, konstanty, zkratky týmů)
├── pipeline.py              orchestrace pipeline (sběr dat → trénink → predikce)
│
├── data/                    sběr a čištění dat
│   ├── nba_stats.py         NBA Stats API klient
│   ├── basketball_ref.py    Basketball Reference scraper
│   ├── injuries.py          ESPN injury report
│   └── processor.py         čištění a sloučení zdrojů
│
├── features/                feature engineering
│   ├── team_features.py     rolling statistiky, form, streak, rest, ELO
│   ├── player_features.py   hodnota hráče, síla sestavy
│   └── builder.py           sestavení finálního datasetu
│
├── models/                  ML modely
│   ├── base.py              abstraktní třída
│   ├── logistic.py          Logistic Regression
│   ├── random_forest.py     Random Forest
│   └── evaluator.py         porovnání modelů
│
├── web/                     Flask API
│   └── routes.py            API endpointy
│
├── vendor/                  neautorský kód
│   ├── frontend/            webový frontend (HTML/CSS/JS)
│   ├── build/               PyInstaller konfigurace
│   └── build_exe.bat        sestavení EXE
│
├── docs/                    dokumentace
│   ├── README.md
│   └── NBA_Predictor_Colab.ipynb
│
└── storage/                 runtime data (generováno automaticky)
    ├── raw/                 stažená data (CSV)
    ├── processed/           features.csv, predictions.csv
    └── trained/             best_model.pkl
```

---

## API endpointy

| Endpoint | Popis |
|----------|-------|
| `GET /api/predictions` | Predikce nadcházejících zápasů |
| `GET /api/models` | Porovnání natrénovaných modelů |
| `GET /api/injuries` | Zpráva o zraněních (`?team=BOS` pro filtr) |
| `GET /api/status` | Stav systému |
| `GET /api/update-status` | Průběh aktualizace dat |
