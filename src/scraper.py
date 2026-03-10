import pandas as pd
import cloudscraper
import time
import os
from io import StringIO


class NBAScraper:
    BASE_URL = "https://www.basketball-reference.com/teams/{team}/{year}/gamelog/"

    def __init__(self, storage_path="data/raw/"):
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
        )

    def fetch_team_data(self, team_code, year):
        url = self.BASE_URL.format(team=team_code, year=year)
        print(f"📡 Stahuji data: {team_code} pro rok {year}...")

        try:
            response = self.scraper.get(url)
            if response.status_code != 200:
                print(f"Server vrátil chybu {response.status_code}")
                return None

            tables = pd.read_html(StringIO(response.text), match="Date")

            if not tables:
                print(f"Na stránce {team_code} nebyla nalezena tabulka dat.")
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
            print(f"Kritická chyba u {team_code} {year}: {e}")
            return None

    def run_bulk_collection(self, teams, years):
        all_data = []
        for year in years:
            for team in teams:
                data = self.fetch_team_data(team, year)
                if data is not None:
                    all_data.append(data)
                    print(f"Staženo {len(data)} zápasů.")
                time.sleep(4)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            filename = os.path.join(self.storage_path, "nba_raw.csv")
            final_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\nÚspěch! Celkem uloženo {len(final_df)} řádků do {filename}")
        else:
            print("Žádná data nebyla stažena.")


if __name__ == "__main__":
    scraper = NBAScraper()
    target_teams = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]
    target_years = [2023, 2024, 2025]
    scraper.run_bulk_collection(target_teams, target_years)