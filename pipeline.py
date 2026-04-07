"""End-to-end pipeline: collect data, engineer features, train models, predict.

Usage:
    python pipeline.py              Run the full pipeline
    python pipeline.py --skip-scrape   Skip data collection, use existing data
    python pipeline.py --train-only    Only retrain models on existing features
"""

import argparse
import logging
import sys

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR
from data.basketball_ref import BasketballRefScraper
from data.nba_stats import NBAStatsClient
from data.processor import DataProcessor
from features.builder import ALL_FEATURES, build_dataset
from models.evaluator import ModelEvaluator
from models.logistic import LogisticModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect data from both sources and return game logs and player stats."""
    nba_client = NBAStatsClient()
    logger.info("=== Fetching data from NBA Stats API ===")
    nba_games = nba_client.fetch_team_gamelogs()
    player_stats = nba_client.fetch_player_stats()
    nba_client.fetch_injury_report(player_stats)

    logger.info("=== Fetching data from Basketball Reference ===")
    bref = BasketballRefScraper()
    bref_games = bref.scrape_gamelogs()
    bref.scrape_schedule()

    return nba_games, bref_games, player_stats


def process_data(nba_games: pd.DataFrame, bref_games: pd.DataFrame) -> pd.DataFrame:
    """Clean and merge raw data from both sources."""
    processor = DataProcessor()
    nba_clean = processor.process_nba_api_data(nba_games)
    bref_clean = processor.process_bref_data(bref_games)
    merged = processor.merge_sources(nba_clean, bref_clean)
    processor.save(merged)
    return merged


def train_models(features: pd.DataFrame) -> pd.DataFrame:
    """Train all models and return comparison results."""
    feature_cols = [c for c in ALL_FEATURES if c in features.columns]
    logger.info("Training with %d features: %s", len(feature_cols), feature_cols)

    models = [LogisticModel(), XGBoostModel(auto_tune=True), RandomForestModel(auto_tune=True)]
    evaluator = ModelEvaluator(models)
    comparison = evaluator.run(features, feature_cols)
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    logger.info("\n=== MODEL COMPARISON ===\n%s", comparison.to_string(index=False))
    return comparison


def generate_predictions(features: pd.DataFrame) -> None:
    """Generate predictions for upcoming games using the best trained model."""
    import joblib
    from config import MODEL_DIR, TEAM_ABBR_TO_NAME, TEAM_NAME_TO_ABBR

    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        logger.warning("No trained model found, skipping predictions")
        return

    model = joblib.load(model_path)
    schedule_path = RAW_DIR / "schedule.csv"
    if not schedule_path.exists():
        logger.warning("No schedule data found, skipping predictions")
        return

    schedule = pd.read_csv(schedule_path)
    schedule["Date"] = pd.to_datetime(schedule["Date"])

    feature_cols = [c for c in ALL_FEATURES if c in features.columns]
    latest = features.sort_values("Date").groupby("Team").last().reset_index()

    predictions = []
    for _, game in schedule.iterrows():
        home_name = game["Home"]
        away_name = game["Away"]
        home = TEAM_NAME_TO_ABBR.get(home_name, home_name)
        away = TEAM_NAME_TO_ABBR.get(away_name, away_name)

        home_feats = latest[latest["Team"] == home]
        away_feats = latest[latest["Team"] == away]

        if home_feats.empty or away_feats.empty:
            continue

        matchup = {}
        for col in feature_cols:
            if col in ("is_home", "is_b2b"):
                matchup[col] = 1 if col == "is_home" else 0
            elif col in home_feats.columns:
                matchup[col] = float(home_feats[col].iloc[0])
            else:
                matchup[col] = 0.0

        X = pd.DataFrame([matchup])
        try:
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            prob = 0.5

        predictions.append({
            "date": game["Date"].strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_prob": round(prob, 4),
            "away_prob": round(1 - prob, 4),
            "predicted_winner": home if prob >= 0.5 else away,
        })

    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(PROCESSED_DIR / "predictions.csv", index=False)
        logger.info("Generated %d game predictions", len(predictions))


def main():
    parser = argparse.ArgumentParser(description="NBA Predictor Pipeline")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip data collection")
    parser.add_argument("--train-only", action="store_true", help="Only retrain models")
    args = parser.parse_args()

    if args.train_only:
        features_path = PROCESSED_DIR / "features.csv"
        if not features_path.exists():
            logger.error("No feature data found. Run full pipeline first.")
            sys.exit(1)
        features = pd.read_csv(features_path)
        train_models(features)
        generate_predictions(features)
        return

    if args.skip_scrape:
        games_path = PROCESSED_DIR / "games_clean.csv"
        if not games_path.exists():
            logger.error("No processed game data found. Run full pipeline first.")
            sys.exit(1)
        games = pd.read_csv(games_path)
        games["Date"] = pd.to_datetime(games["Date"])
        player_stats = pd.DataFrame()
        ps_path = RAW_DIR / "player_stats.csv"
        if ps_path.exists():
            player_stats = pd.read_csv(ps_path)
    else:
        nba_games, bref_games, player_stats = collect_data()
        games = process_data(nba_games, bref_games)

    logger.info("=== Building features ===")
    features = build_dataset(games, player_stats)

    logger.info("=== Training models ===")
    train_models(features)

    logger.info("=== Generating predictions ===")
    generate_predictions(features)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
