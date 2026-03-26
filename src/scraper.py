import pandas as pd
import cloudscraper
import time
import os
import logging
from io import StringIO
from requests.exceptions import RequestException
from datetime import datetime

class NBAScraper:
    BASE_URL = "https://www.basketball-reference.com/teams/{team}/{year}/gamelog/"
    
    REQUEST_DELAY = 3.1 
    MAX_RETRIES = 3
    TIMEOUT = 25

    START_SEASON = 2018
    CURRENT_SEASON = 2026 

    ALL_TEAMS = [
        "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
        "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]

    def __init__(self, storage_path="data/raw/"):
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, "nba_raw.csv")
        
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self._setup_logging()
        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler("logs/scraper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("NBA_SCRAPER")

    def _request_page(self, url):
        for attempt in range(self.MAX_RETRIES):
            try:
                res = self.scraper.get(url, timeout=self.TIMEOUT)
                
                if res.status_code == 200:
                    return res.text
                
                if res.status_code == 429:
                    self.logger.warning("Rate limit (429) hit! Sleeping for 120s...")
                    time.sleep(120)
                    continue

                self.logger.error(f"Status {res.status_code} for {url}")
            except RequestException as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
            
            time.sleep(self.REQUEST_DELAY * (attempt + 2))
        return None

    def _make_unique(self, cols):
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        return new_cols

    def _parse_table(self, html, team, year):
        try:
            tables = pd.read_html(StringIO(html), header=[0, 1], match="Date")
            if not tables: return None

            df = tables[0].copy()
            new_cols = []
            for i, col in enumerate(df.columns):
                if i == 3: new_cols.append("HomeAway")
                else: new_cols.append(col[1])

            df.columns = self._make_unique(new_cols)
            df = df[df["Date"] != "Date"].copy()
            
            df.loc[:, "Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            
            df.loc[:, "Team"] = team
            df.loc[:, "Season"] = year
            return df.reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Parse error {team}-{year}: {e}")
            return None

    def get_completed_tasks(self):
        if not os.path.exists(self.file_path):
            return set()
        try:
            df = pd.read_csv(self.file_path, usecols=["Team", "Season"])
            completed = set(zip(df["Team"].astype(str), df["Season"].astype(int)))
            return completed
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return set()

    def run(self):
        self.logger.info("=== NBA SCRAPER STARTING ===")
        
        completed = self.get_completed_tasks()
        
        tasks = []
        for year in range(self.START_SEASON, self.CURRENT_SEASON + 1):
            for team in self.ALL_TEAMS:
                if (team, year) not in completed:
                    tasks.append((team, year))
        
        if not tasks:
            self.logger.info("Všechna data jsou aktuální. Končím.")
            return

        self.logger.info(f"K vyřízení zbývá {len(tasks)} úkolů z celkových {len(self.ALL_TEAMS) * (self.CURRENT_SEASON - self.START_SEASON + 1)}")

        for i, (team, year) in enumerate(tasks):
            self.logger.info(f"[{i+1}/{len(tasks)}] Fetching {team} {year}")
            
            url = self.BASE_URL.format(team=team, year=year)
            html = self._request_page(url)
            
            if html:
                df = self._parse_table(html, team, year)
                if df is not None and not df.empty:
                    file_exists = os.path.isfile(self.file_path)
                    df.to_csv(self.file_path, mode='a', index=False, header=not file_exists)
                    self.logger.info(f"Successfully saved {len(df)} rows for {team}-{year}")
                else:
                    self.logger.warning(f"No data parsed for {team}-{year}")
            
            time.sleep(self.REQUEST_DELAY)

        self.logger.info("=== SCRAPING COMPLETED ===")

    def _request_page(self, url):
        for attempt in range(self.MAX_RETRIES):
            try:
                res = self.scraper.get(url, timeout=self.TIMEOUT)
                
                if res.status_code == 200:
                    return res.text
                
                if res.status_code == 404:
                    self.logger.warning(f"Page not found (404): {url}")
                    return None

                if res.status_code == 429:
                    self.logger.warning("Rate limit (429) hit! Sleeping for 120s...")
                    time.sleep(120)
                    continue

                self.logger.error(f"Status {res.status_code} for {url}")
            except RequestException as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
            
            time.sleep(self.REQUEST_DELAY * (attempt + 2))
        return None

    def save_future_games(self):
        months = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]
        current_month_name = datetime.now().strftime("%B").lower()
        
        try:
            start_idx = months.index(current_month_name)
        except ValueError:
            start_idx = 0
            
        future_months = months[start_idx:]
        all_future_games = []

        for month in future_months:
            url = f"https://www.basketball-reference.com/leagues/NBA_{self.CURRENT_SEASON}_games-{month}.html"
            html = self._request_page(url)
            
            if not html:
                continue

            try:
                tables = pd.read_html(StringIO(html))
                df = next((t for t in tables if "Date" in t.columns), None)
                
                if df is None: continue

                df = df.rename(columns={"Visitor/Neutral": "Away", "Home/Neutral": "Home"})
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])

                today = pd.Timestamp.today().normalize()
                future_only = df[df["Date"] >= today].copy()

                if not future_only.empty:
                    all_future_games.append(future_only[["Date", "Home", "Away"]])
                    self.logger.info(f"Found {len(future_only)} future games in {month}")

            except Exception as e:
                self.logger.warning(f"Could not parse month {month}: {e}")
            
            time.sleep(self.REQUEST_DELAY)

        if all_future_games:
            final_df = pd.concat(all_future_games).drop_duplicates()
            path = os.path.join(self.storage_path, "future_games.csv")
            final_df.to_csv(path, index=False)
            self.logger.info(f"SUCCESS: Total {len(final_df)} future games saved.")
            return final_df
        
        self.logger.warning("No future games found.")
        return pd.DataFrame()

if __name__ == "__main__":
    scraper = NBAScraper()
    scraper.run()
    scraper.save_future_games()