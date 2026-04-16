import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from config import (
    LAST_UPDATED_FILE,
    MODEL_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TEAM_ABBR_TO_NAME,
    TEAM_ABBREVIATION_FIXES,
    TEAM_NAME_TO_ABBR,
    UPDATE_CACHE_HOURS,
)
from data.basketball_ref import BasketballRefScraper
from data.injuries import InjuryReportFetcher, compute_team_availability
from data.nba_stats import NBAStatsClient
from data.processor import DataProcessor
from features.builder import ALL_FEATURES, ROLL_COLS, STATE_COLS, build_dataset
from models.evaluator import ModelEvaluator
from models.logistic import LogisticModel
from models.random_forest import RandomForestModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")

_update_status: dict = {"running": False, "last_updated": None, "progress": "", "error": None}


def get_update_status() -> dict:
    return _update_status.copy()


def _set_status(**kwargs) -> None:
    _update_status.update(kwargs)


def _load_last_updated() -> datetime | None:
    if LAST_UPDATED_FILE.exists():
        try:
            return datetime.fromisoformat(LAST_UPDATED_FILE.read_text().strip())
        except ValueError:
            return None
    return None


def _save_last_updated() -> None:
    LAST_UPDATED_FILE.write_text(datetime.now().isoformat())


def _gamelogs_last_date(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(path, usecols=["GAME_DATE"])
        dates = pd.to_datetime(df["GAME_DATE"], errors="coerce").dropna()
        return dates.max() if not dates.empty else None
    except Exception:
        return None


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nba_client = NBAStatsClient()
    _set_status(progress="Fetching NBA API data...")
    logger.info("=== Fetching data from NBA Stats API ===")
    nba_games = nba_client.fetch_team_gamelogs()
    player_stats = nba_client.fetch_player_stats()

    _set_status(progress="Fetching injury report...")
    logger.info("=== Fetching injury report ===")
    fetcher = InjuryReportFetcher()
    fetcher.fetch(player_stats)

    _set_status(progress="Fetching Basketball Reference data...")
    logger.info("=== Fetching data from Basketball Reference ===")
    bref = BasketballRefScraper()
    bref_games = bref.scrape_gamelogs()
    bref.scrape_schedule()

    return nba_games, bref_games, player_stats


def smart_update_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Incremental update: only fetch games added since the last run.

    Uses LeagueGameLog to fetch all new games in a single API call instead
    of iterating over all 30 teams. Falls back to full collection on first run.
    """
    gamelogs_path = RAW_DIR / "nba_api_gamelogs.csv"
    bref_path = RAW_DIR / "bref_gamelogs.csv"

    if not gamelogs_path.exists():
        logger.info("No existing gamelogs — running full data collection")
        return collect_data()

    # Check if existing data is already fresh enough to skip the API entirely
    last_updated = _load_last_updated()
    if last_updated is not None:
        age_hours = (datetime.now() - last_updated).total_seconds() / 3600
        if age_hours < UPDATE_CACHE_HOURS:
            logger.info("Data is %.1fh old (< %dh threshold) — using cache", age_hours, UPDATE_CACHE_HOURS)
            existing = pd.read_csv(gamelogs_path)
            bref_games = pd.read_csv(bref_path) if bref_path.exists() else pd.DataFrame()
            ps_path = RAW_DIR / "player_stats.csv"
            player_stats = pd.read_csv(ps_path) if ps_path.exists() else pd.DataFrame()
            return existing, bref_games, player_stats

    nba_client = NBAStatsClient()

    # Determine the earliest date we need to (re)fetch
    last_game_date = _gamelogs_last_date(gamelogs_path)
    if last_game_date is None:
        logger.warning("Could not determine last game date — falling back to full fetch")
        return collect_data()

    since_date = (last_game_date + pd.Timedelta(days=1)).strftime("%m/%d/%Y")
    _set_status(progress=f"Fetching new games since {since_date}...")
    logger.info("=== Incremental update: games since %s ===", since_date)

    new_games = nba_client.fetch_incremental_gamelogs(since_date)
    existing = pd.read_csv(gamelogs_path)

    if not new_games.empty:
        combined = pd.concat([existing, new_games], ignore_index=True)
        # Deduplicate by GAME_ID + Team (each team has its own row per game)
        if "GAME_ID" in combined.columns:
            combined = combined.drop_duplicates(subset=["Team", "GAME_ID"])
        else:
            combined = combined.drop_duplicates(subset=["Team", "GAME_DATE", "MATCHUP"])
        combined.to_csv(gamelogs_path, index=False)
        logger.info("Added %d new game rows; total %d", len(new_games), len(combined))
    else:
        combined = existing
        logger.info("No new games to add")

    _set_status(progress="Fetching player stats and injuries...")
    player_stats = nba_client.fetch_player_stats()

    fetcher = InjuryReportFetcher()
    fetcher.fetch(player_stats)

    _set_status(progress="Updating schedule...")
    bref = BasketballRefScraper()
    bref.scrape_schedule()

    bref_games = pd.read_csv(bref_path) if bref_path.exists() else pd.DataFrame()
    _save_last_updated()
    return combined, bref_games, player_stats


def quick_update_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fast refresh: current-season NBA API + injuries + schedule only."""
    nba_client = NBAStatsClient()
    logger.info("=== Quick update: NBA API current season ===")
    nba_games = nba_client.fetch_team_gamelogs()
    player_stats = nba_client.fetch_player_stats()

    logger.info("=== Quick update: injury report ===")
    fetcher = InjuryReportFetcher()
    fetcher.fetch(player_stats)

    logger.info("=== Quick update: schedule ===")
    bref = BasketballRefScraper()
    bref.scrape_schedule()

    bref_path = RAW_DIR / "bref_gamelogs.csv"
    bref_games = pd.read_csv(bref_path) if bref_path.exists() else pd.DataFrame()

    return nba_games, bref_games, player_stats


def process_data(nba_games: pd.DataFrame, bref_games: pd.DataFrame) -> pd.DataFrame:
    processor = DataProcessor()
    nba_clean = processor.process_nba_api_data(nba_games)
    bref_clean = processor.process_bref_data(bref_games)
    merged = processor.merge_sources(nba_clean, bref_clean)
    processor.save(merged)
    return merged


def train_models(features: pd.DataFrame) -> pd.DataFrame:
    _set_status(progress="Training models...")
    feature_cols = [c for c in ALL_FEATURES if c in features.columns]
    logger.info("Training with %d features: %s", len(feature_cols), feature_cols)

    models = [
        LogisticModel(),
        RandomForestModel(auto_tune=True),
    ]

    evaluator = ModelEvaluator(models)
    comparison = evaluator.run(features, feature_cols)
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    logger.info("\n=== MODEL COMPARISON ===\n%s", comparison.to_string(index=False))
    return comparison


def _resolve_abbr(name: str) -> str:
    fixed = TEAM_NAME_TO_ABBR.get(name)
    if fixed:
        return fixed
    return TEAM_ABBREVIATION_FIXES.get(name, name)


def _get_team_state(latest: pd.DataFrame, team: str) -> dict | None:
    row = latest[latest["Team"] == team]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _is_b2b(team: str, game_date: pd.Timestamp, schedule: pd.DataFrame) -> int:
    prev = game_date - pd.Timedelta(days=1)
    played_prev = schedule[schedule["Date"].dt.date == prev.date()]
    for _, g in played_prev.iterrows():
        home = _resolve_abbr(g.get("Home", ""))
        away = _resolve_abbr(g.get("Away", ""))
        if team in (home, away):
            return 1
    return 0


def generate_predictions(features: pd.DataFrame) -> None:
    _set_status(progress="Generating predictions...")
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

    today = pd.Timestamp.today().normalize()
    upcoming = schedule[schedule["Date"] >= today].copy()
    if upcoming.empty:
        logger.warning("No upcoming games in schedule")
        return

    latest = features.sort_values("Date").groupby("Team").last().reset_index()

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
            logger.warning("Could not load injury report: %s", exc, exc_info=True)

    roll_to_diff = [
        ("roll_PTS",     "diff_pts"),
        ("roll_OPP_PTS", "diff_opp_pts"),
        ("roll_FG_PCT",  "diff_fg_pct"),
        ("roll_FG3_PCT", "diff_fg3_pct"),
        ("roll_REB",     "diff_reb"),
        ("roll_AST",     "diff_ast"),
        ("roll_TOV",     "diff_tov"),
        ("roll_STL",     "diff_stl"),
        ("roll_BLK",     "diff_blk"),
        ("form",         "diff_form"),
        ("streak",       "diff_streak"),
        ("rest_days",    "diff_rest"),
        ("elo",          "diff_elo"),
    ]

    if hasattr(model, "feature_names_in_"):
        feature_cols = [c for c in model.feature_names_in_ if c in features.columns]
    elif hasattr(model, "get_booster") and model.get_booster().feature_names:
        feature_cols = [c for c in model.get_booster().feature_names if c in features.columns]
    else:
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

        for roll_col, diff_col in roll_to_diff:
            if diff_col not in feature_cols:
                continue
            home_val = float(home_state.get(roll_col, 0.0) or 0.0)
            away_val = float(away_state.get(roll_col, 0.0) or 0.0)
            matchup[diff_col] = home_val - away_val

        matchup["is_home"] = 1.0
        matchup["is_b2b"] = float(_is_b2b(home, game_date, upcoming))

        if "diff_roster" in feature_cols:
            home_avail = team_availability.get(home, 1.0)
            away_avail = team_availability.get(away, 1.0)
            matchup["diff_roster"] = home_avail - away_avail

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
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--quick-update", action="store_true")
    parser.add_argument("--smart-update", action="store_true")
    parser.add_argument("--serve", action="store_true")
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

    elif args.smart_update:
        nba_games, bref_games, player_stats = smart_update_data()
        games = process_data(nba_games, bref_games)
        features = build_dataset(games, player_stats)
        generate_predictions(features)

    elif args.quick_update:
        nba_games, bref_games, player_stats = quick_update_data()
        games = process_data(nba_games, bref_games)
        features = build_dataset(games, player_stats)
        generate_predictions(features)

    elif args.skip_scrape:
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
        features = build_dataset(games, player_stats)
        train_models(features)
        generate_predictions(features)

    else:
        nba_games, bref_games, player_stats = collect_data()
        games = process_data(nba_games, bref_games)
        features = build_dataset(games, player_stats)
        train_models(features)
        generate_predictions(features)

    logger.info("=== Pipeline complete ===")

    if args.serve:
        from app import create_app
        app = create_app()
        logger.info("Starting web server at http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
