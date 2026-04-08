"""End-to-end pipeline: collect data, engineer features, train models, predict.

Usage:
    python pipeline.py              Run the full pipeline
    python pipeline.py --skip-scrape   Skip data collection, use existing data
    python pipeline.py --train-only    Only retrain models on existing features
"""

import argparse
import logging
import sys

import joblib
import pandas as pd

from config import (
    MODEL_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TEAM_ABBR_TO_NAME,
    TEAM_NAME_TO_ABBR,
)
from data.basketball_ref import BasketballRefScraper
from data.injuries import InjuryReportFetcher, compute_team_availability
from data.nba_stats import NBAStatsClient
from data.processor import DataProcessor
from features.builder import ALL_FEATURES, ROLL_COLS, STATE_COLS, build_dataset
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


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect data from all sources.

    Returns:
        nba_games: Raw game log DataFrame from NBA Stats API.
        bref_games: Raw game log DataFrame from Basketball Reference.
        player_stats: Per-game player stats used for injury fallback.
    """
    nba_client = NBAStatsClient()
    logger.info("=== Fetching data from NBA Stats API ===")
    nba_games = nba_client.fetch_team_gamelogs()
    player_stats = nba_client.fetch_player_stats()

    logger.info("=== Fetching injury report ===")
    fetcher = InjuryReportFetcher()
    fetcher.fetch(player_stats)  # persists injury_report.csv

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
    """Train all models and return a comparison table."""
    feature_cols = [c for c in ALL_FEATURES if c in features.columns]
    logger.info("Training with %d features: %s", len(feature_cols), feature_cols)

    models = [LogisticModel(), XGBoostModel(auto_tune=True), RandomForestModel(auto_tune=True)]
    evaluator = ModelEvaluator(models)
    comparison = evaluator.run(features, feature_cols)
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    logger.info("\n=== MODEL COMPARISON ===\n%s", comparison.to_string(index=False))
    return comparison


def _resolve_abbr(name: str) -> str:
    """Convert a full team name or any abbreviation to our canonical form."""
    from config import TEAM_ABBREVIATION_FIXES
    fixed = TEAM_NAME_TO_ABBR.get(name)
    if fixed:
        return fixed
    return TEAM_ABBREVIATION_FIXES.get(name, name)


def _get_team_state(latest: pd.DataFrame, team: str) -> dict | None:
    """Return the latest rolling-stats row for a team, or None if missing."""
    row = latest[latest["Team"] == team]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _is_b2b(team: str, game_date: pd.Timestamp, schedule: pd.DataFrame) -> int:
    """Return 1 if the team played the previous calendar day, else 0."""
    prev = game_date - pd.Timedelta(days=1)
    played_prev = schedule[schedule["Date"].dt.date == prev.date()]
    for _, g in played_prev.iterrows():
        home = _resolve_abbr(g.get("Home", ""))
        away = _resolve_abbr(g.get("Away", ""))
        if team in (home, away):
            return 1
    return 0


def generate_predictions(features: pd.DataFrame) -> None:
    """Generate win-probability predictions for upcoming games.

    For each scheduled game, computes matchup differentials from the most
    recent rolling stats of each team and adjusts the roster feature using
    the current injury report. The model then produces a win probability
    for the home team.
    """
    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        logger.warning("No trained model found — skipping predictions")
        return

    model = joblib.load(model_path)
    schedule_path = RAW_DIR / "schedule.csv"
    if not schedule_path.exists():
        logger.warning("No schedule data found — skipping predictions")
        return

    schedule = pd.read_csv(schedule_path)
    schedule["Date"] = pd.to_datetime(schedule["Date"])

    # Use only upcoming games
    today = pd.Timestamp.today().normalize()
    upcoming = schedule[schedule["Date"] >= today].copy()
    if upcoming.empty:
        logger.warning("No upcoming games in schedule")
        return

    # Latest per-team rolling stats snapshot (no data leakage — these are
    # already shifted features from the last completed game)
    latest = features.sort_values("Date").groupby("Team").last().reset_index()

    # Load current injury report to compute real-time roster availability.
    # player_stats provides the full roster as denominator for availability ratio.
    injury_path = RAW_DIR / "injury_report.csv"
    team_availability: dict[str, float] = {}
    if injury_path.exists():
        try:
            injury_df = pd.read_csv(injury_path)
            player_stats = pd.DataFrame()
            ps_path = RAW_DIR / "player_stats.csv"
            if ps_path.exists():
                player_stats = pd.read_csv(ps_path)
            avail = compute_team_availability(injury_df, player_stats)
            abbr_col = "team_abbr" if "team_abbr" in avail.columns else "Team"
            for _, row in avail.iterrows():
                team_availability[str(row[abbr_col])] = float(row["availability_ratio"])
            logger.info("Loaded injury data for %d teams", len(team_availability))
        except Exception as exc:
            logger.warning("Could not load injury report: %s", exc)

    roll_to_diff = [
        ("roll_PTS",     "diff_pts"),
        ("roll_OPP_PTS", "diff_opp_pts"),
        ("roll_FG_PCT",  "diff_fg_pct"),
        ("roll_REB",     "diff_reb"),
        ("roll_AST",     "diff_ast"),
        ("roll_TOV",     "diff_tov"),
        ("form",         "diff_form"),
        ("streak",       "diff_streak"),
        ("rest_days",    "diff_rest"),
    ]

    feature_cols = [c for c in ALL_FEATURES if c in features.columns]
    predictions = []

    for _, game in upcoming.iterrows():
        home = _resolve_abbr(game["Home"])
        away = _resolve_abbr(game["Away"])
        game_date = game["Date"]

        home_state = _get_team_state(latest, home)
        away_state = _get_team_state(latest, away)

        if home_state is None or away_state is None:
            logger.debug("Skipping %s vs %s — no feature data", home, away)
            continue

        matchup: dict[str, float] = {}

        # Compute differential features from each team's latest rolling stats
        for roll_col, diff_col in roll_to_diff:
            if diff_col not in feature_cols:
                continue
            home_val = float(home_state.get(roll_col, 0.0) or 0.0)
            away_val = float(away_state.get(roll_col, 0.0) or 0.0)
            matchup[diff_col] = home_val - away_val

        # Home court and back-to-back flags
        matchup["is_home"] = 1.0
        matchup["is_b2b"] = float(_is_b2b(home, game_date, upcoming))

        # Roster availability differential from current injury data
        if "diff_roster" in feature_cols:
            home_avail = team_availability.get(home, 1.0)
            away_avail = team_availability.get(away, 1.0)
            matchup["diff_roster"] = home_avail - away_avail

        # Build feature vector in the exact order the model expects
        X = pd.DataFrame([[matchup.get(c, 0.0) for c in feature_cols]], columns=feature_cols)

        try:
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception as exc:
            logger.warning("Prediction failed for %s vs %s: %s", home, away, exc)
            prob = 0.5

        home_avail_pct = round(team_availability.get(home, 1.0) * 100, 1)
        away_avail_pct = round(team_availability.get(away, 1.0) * 100, 1)

        predictions.append({
            "date": game_date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_team_name": TEAM_ABBR_TO_NAME.get(home, home),
            "away_team_name": TEAM_ABBR_TO_NAME.get(away, away),
            "home_prob": round(prob, 4),
            "away_prob": round(1 - prob, 4),
            "predicted_winner": home if prob >= 0.5 else away,
            "predicted_winner_name": (
                TEAM_ABBR_TO_NAME.get(home, home) if prob >= 0.5
                else TEAM_ABBR_TO_NAME.get(away, away)
            ),
            "home_roster_availability": home_avail_pct,
            "away_roster_availability": away_avail_pct,
        })

    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(PROCESSED_DIR / "predictions.csv", index=False)
        logger.info("Generated %d game predictions", len(predictions))
    else:
        logger.warning("No predictions could be generated")


def main():
    parser = argparse.ArgumentParser(description="NBA Predictor Pipeline")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip data collection, reprocess from existing raw CSVs")
    parser.add_argument("--train-only", action="store_true",
                        help="Only retrain models on existing features.csv")
    args = parser.parse_args()

    if args.train_only:
        features_path = PROCESSED_DIR / "features.csv"
        if not features_path.exists():
            logger.error("No feature data found. Run the full pipeline first.")
            sys.exit(1)
        features = pd.read_csv(features_path)
        features["Date"] = pd.to_datetime(features["Date"])
        train_models(features)
        generate_predictions(features)
        return

    if args.skip_scrape:
        # Reprocess from the cached raw CSVs — no HTTP requests made.
        nba_path = RAW_DIR / "nba_api_gamelogs.csv"
        bref_path = RAW_DIR / "bref_gamelogs.csv"
        if not nba_path.exists() and not bref_path.exists():
            logger.error("No raw game log CSVs found. Run the full pipeline first.")
            sys.exit(1)
        nba_games = pd.read_csv(nba_path) if nba_path.exists() else pd.DataFrame()
        bref_games = pd.read_csv(bref_path) if bref_path.exists() else pd.DataFrame()
        games = process_data(nba_games, bref_games)
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
