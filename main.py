import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from processor import NBADataProcessor
from predictor import NBAPredictor, PredictorConfig

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_latest_team_stats(df: pd.DataFrame, team_code: str) -> pd.Series:
    """
    Retrieves the most recent rolling averages for a specific team.
    """
    team_data = df[df['Team'] == team_code].sort_values(by='Date', ascending=False)
    if team_data.empty:
        raise ValueError(f"Team code '{team_code}' not found in the database.")
    return team_data.iloc[0]

def run_prediction(home_team: str, away_team: str, market_odds: float):
    """
    The core prediction engine: 
    Lookup -> Diff Calculation -> Inference -> Value Analysis.
    """
    predictor = NBAPredictor()
    
    try:
        if not Path("data/processed/nba_processed.csv").exists():
            print(f"{Colors.FAIL}Error: Processed data missing. Run --update first.{Colors.ENDC}")
            return

        df = pd.read_csv("data/processed/nba_processed.csv")
        
        home_stats = get_latest_team_stats(df, home_team)
        away_stats = get_latest_team_stats(df, away_team)

        matchup_data = pd.DataFrame([{
            "Diff_Points": home_stats["Roll_Team_Points"] - away_stats["Roll_Team_Points"],
            "Diff_FG_pct": home_stats["Roll_Team_FG_pct"] - away_stats["Roll_Team_FG_pct"],
            "Diff_AST": home_stats["Roll_Team_AST"] - away_stats["Roll_Team_AST"],
            "Diff_TOV": home_stats["Roll_Team_TOV"] - away_stats["Roll_Team_TOV"],
            "Is_Home": 1
        }])

        win_prob = predictor.predict_win_probability(matchup_data)
        analysis = predictor.calculate_betting_value(win_prob, market_odds)

        print(f"\n{Colors.BOLD}{'='*50}")
        print(f" NBA MATCHUP ANALYSIS: {home_team} vs {away_team}")
        print(f"{'='*50}{Colors.ENDC}")
        
        prob_color = Colors.OKGREEN if win_prob > 0.6 else Colors.OKBLUE
        print(f"Win Probability ({home_team}): {prob_color}{win_prob:.2%}{Colors.ENDC}")
        print(f"Fair Odds (0% Margin):   {Colors.BOLD}{analysis['fair_odds']}{Colors.ENDC}")
        print(f"Bookmaker Odds:         {market_odds}")
        
        ev_color = Colors.OKGREEN if analysis['has_value'] else Colors.FAIL
        print(f"Expected Value (EV):    {ev_color}{analysis['expected_value']:+.2%}{Colors.ENDC}")
        
        print(f"{Colors.BOLD}{'-'*50}{Colors.ENDC}")
        if analysis['has_value']:
            print(f"{Colors.OKGREEN}RESULT: VALUE DETECTED. The bet is mathematically profitable.{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}RESULT: NO VALUE. The bookmaker odds are too low for this risk.{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*50}{Colors.ENDC}\n")

    except Exception as e:
        print(f"{Colors.FAIL}Analysis Failed: {e}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="NBA Edge: Professional Predictive Analytics & Value Betting Tool")
    
    parser.add_argument("--update", action="store_true", help="Run the ETL pipeline to refresh team stats")
    parser.add_argument("--train", action="store_true", help="Retrain the XGBoost model on the latest data")
    
    parser.add_argument("--home", type=str, help="Home Team Code (e.g., GSW)")
    parser.add_argument("--away", type=str, help="Away Team Code (e.g., DAL)")
    parser.add_argument("--odds", type=float, help="Decimal odds from the bookmaker (e.g., 1.95)")

    args = parser.parse_args()

    if args.update:
        NBADataProcessor().run_pipeline()
    
    if args.train:
        NBAPredictor().run_training_pipeline()

    if args.home and args.away and args.odds:
        run_prediction(args.home.upper(), args.away.upper(), args.odds)
    elif not (args.update or args.train):
        parser.print_help()

if __name__ == "__main__":
    main()