import pandas as pd
import cloudscraper
import time
import os
import logging
from io import StringIO
from requests.exceptions import RequestException
from datetime import datetime



class NBAScraper:
    """
    Production-grade NBA Game Log Scraper.

    Downloads team game logs from Basketball Reference and stores them
    as a structured dataset for analysis or machine learning pipelines.
    """

    BASE_URL = "https://www.basketball-reference.com/teams/{team}/{year}/gamelog/"

    REQUEST_DELAY = 4
    MAX_RETRIES = 3
    TIMEOUT = 20

    def __init__(self, storage_path="data/raw/"):

        self.storage_path = storage_path

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            filename="logs/scraper.log"
        )

        self.logger = logging.getLogger("NBA_SCRAPER")

        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )

    def _request_page(self, url):
        """
        Request page with retry logic.
        """

        for attempt in range(self.MAX_RETRIES):

            try:
                response = self.scraper.get(url, timeout=self.TIMEOUT)

                if response.status_code == 200:
                    return response.text

                self.logger.warning(
                    f"Server returned status {response.status_code}"
                )

            except RequestException as e:
                self.logger.warning(f"Request failed: {e}")

            self.logger.info(f"Retrying... ({attempt+1}/{self.MAX_RETRIES})")
            time.sleep(3)

        self.logger.error("Max retries exceeded.")
        return None

    def _parse_table(self, html, team_code, year):
        """
        Parse game log table from HTML.
        """

        tables = pd.read_html(StringIO(html), match="Date")

        if not tables:
            self.logger.warning(f"No tables found for {team_code} {year}")
            return None

        df = tables[0]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        if "Rk" not in df.columns:
            df.rename(columns={df.columns[0]: "Rk"}, inplace=True)

        df = df[pd.to_numeric(df["Rk"], errors="coerce").notna()]

        df["Team"] = team_code
        df["Season"] = year

        return df

    def fetch_team_data(self, team_code, year):
        """
        Fetch and parse a single team's season data.
        """

        url = self.BASE_URL.format(team=team_code, year=year)

        self.logger.info(f"Fetching {team_code} season {year}")

        html = self._request_page(url)

        if html is None:
            return None

        return self._parse_table(html, team_code, year)

    def run_bulk_collection(self, teams, years):
        """
        Run scraping across multiple teams and seasons.
        """

        all_data = []

        total_jobs = len(teams) * len(years)
        current_job = 0

        for year in years:

            for team in teams:

                current_job += 1
                self.logger.info(f"[{current_job}/{total_jobs}] Processing {team} {year}")

                data = self.fetch_team_data(team, year)

                if data is not None:
                    all_data.append(data)
                    self.logger.info(f"{len(data)} games collected")

                time.sleep(self.REQUEST_DELAY)

        if not all_data:
            self.logger.error("No data collected.")
            return

        final_df = pd.concat(all_data, ignore_index=True)

        filename = os.path.join(self.storage_path, "nba_raw.csv")
        final_df.to_csv(filename, index=False, encoding="utf-8")

        self.logger.info(f"Dataset saved: {filename}")
        self.logger.info(f"Total rows: {len(final_df)}")


if __name__ == "__main__":

    scraper = NBAScraper()

    target_teams = [
        "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
        "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]

    current_year = datetime.now().year
    target_years = list(range(2023, current_year))

    scraper.run_bulk_collection(target_teams, target_years)