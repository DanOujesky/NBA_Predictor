import pandas as pd
import cloudscraper
import time
import os
from io import StringIO


class NBAScraper:
    """
    NBA Game Log Scraper

    This class downloads NBA team game logs from Basketball Reference
    and stores them as a CSV dataset for further data analysis or processing.
    """

    BASE_URL = "https://www.basketball-reference.com/teams/{team}/{year}/gamelog/"

    def __init__(self, storage_path="data/raw/"):
        """
        Initialize the scraper and ensure the storage directory exists.
        """
        self.storage_path = storage_path

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
        )

    def fetch_team_data(self, team_code, year):
        """
        Download game log data for a specific NBA team and season.

        Args:
            team_code (str): Three-letter team abbreviation (e.g., BOS, LAL).
            year (int): Season year.

        Returns:
            pandas.DataFrame or None
        """
        url = self.BASE_URL.format(team=team_code, year=year)
        print(f"Fetching data: Team {team_code}, Season {year}...")

        try:
            response = self.scraper.get(url)

            if response.status_code != 200:
                print(f"Server returned status code {response.status_code}")
                return None

            tables = pd.read_html(StringIO(response.text), match="Date")

            if not tables:
                print(f"No game log table found for {team_code} ({year}).")
                return None

            df = tables[0]

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)

            if 'Rk' not in df.columns:
                df.rename(columns={df.columns[0]: 'Rk'}, inplace=True)

            df = df[pd.to_numeric(df['Rk'], errors='coerce').notna()]

            df['Team'] = team_code
            df['Season'] = year

            return df

        except Exception as e:
            print(f"Critical error for {team_code} ({year}): {e}")
            return None

    def run_bulk_collection(self, teams, years):
        """
        Download game logs for multiple teams across multiple seasons.

        Args:
            teams (list): List of NBA team codes.
            years (list): List of seasons.
        """
        all_data = []

        for year in years:
            for team in teams:
                data = self.fetch_team_data(team, year)

                if data is not None:
                    all_data.append(data)
                    print(f"Downloaded {len(data)} games.")

                time.sleep(4)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)

            filename = os.path.join(self.storage_path, "nba_raw.csv")
            final_df.to_csv(filename, index=False, encoding='utf-8')

            print(f"\nSuccess! Saved {len(final_df)} rows to {filename}")
        else:
            print("No data was collected.")


if __name__ == "__main__":
    scraper = NBAScraper()

    target_teams = [
        "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
        "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]

    target_years = [2023, 2024, 2025]

    scraper.run_bulk_collection(target_teams, target_years)