from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
RAW_DIR = STORAGE_DIR / "raw"
PROCESSED_DIR = STORAGE_DIR / "processed"
MODEL_DIR = STORAGE_DIR / "trained"
LOG_DIR = STORAGE_DIR / "logs"

for directory in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CURRENT_SEASON = 2026
SEASONS_BACK = 5
REQUEST_DELAY = 1.5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 25

ROLLING_WINDOW = 10
FORM_WINDOW = 5

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

TEAM_ABBREVIATION_FIXES = {
    "ATL": "ATL",
    "BOS": "BOS",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHA": "CHO",
    "CHO": "CHO",
    "CHI": "CHI",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GSW": "GSW",
    "HOU": "HOU",
    "IND": "IND",
    "LAC": "LAC",
    "LAL": "LAL",
    "MEM": "MEM",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NOP": "NOP",
    "NOH": "NOP",
    "NOK": "NOP",
    "NYK": "NYK",
    "OKC": "OKC",
    "ORL": "ORL",
    "PHI": "PHI",
    "PHX": "PHO",
    "PHO": "PHO",
    "POR": "POR",
    "SAC": "SAC",
    "SAS": "SAS",
    "TOR": "TOR",
    "UTA": "UTA",
    "WAS": "WAS",
}

TEAMS = sorted(set(TEAM_ABBREVIATION_FIXES.values()))

NBA_TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BRK": 1610612751,
    "CHO": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHO": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}

# ESPN uses different 2-3 letter codes that must be mapped to our canonical abbreviations
ESPN_ABBR_FIXES = {
    "BKN": "BRK", "GS": "GSW", "NY": "NYK", "SA": "SAS",
    "NO": "NOP", "CHA": "CHO", "PHX": "PHO", "WSH": "WAS",
    "UTA": "UTA", "UTAH": "UTA", "ATL": "ATL", "BOS": "BOS", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "POR": "POR",
    "SAC": "SAC", "TOR": "TOR",
}

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Reverse mapping: abbreviation → full team name
TEAM_ABBR_TO_NAME = {v: k for k, v in TEAM_NAME_TO_ABBR.items() if k != "LA Clippers"}
