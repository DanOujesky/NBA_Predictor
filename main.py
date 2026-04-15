import logging
import sys
import threading
import webbrowser
from datetime import datetime
from threading import Timer

import pandas as pd

from config import FLASK_HOST, FLASK_PORT, MODEL_DIR, PROCESSED_DIR, RAW_DIR
from features.builder import build_dataset
from pipeline import (
    _save_last_updated,
    _set_status,
    collect_data,
    generate_predictions,
    process_data,
    smart_update_data,
    train_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

RETRAIN_GAME_THRESHOLD = 30


def _open_browser():
    webbrowser.open(f"http://{FLASK_HOST}:{FLASK_PORT}")


def _count_new_games(new_df: pd.DataFrame) -> int:
    if new_df is None or new_df.empty:
        return 0
    return max(len(new_df) // 2, 0)


def _run_background_update():
    """Execute the smart data update and regenerate predictions in a background thread."""
    try:
        _set_status(running=True, progress="Starting update...", error=None)
        logger.info("Background update started")

        gamelogs_path = RAW_DIR / "nba_api_gamelogs.csv"
        rows_before = 0
        if gamelogs_path.exists():
            try:
                rows_before = len(pd.read_csv(gamelogs_path, usecols=["GAME_DATE"]))
            except Exception:
                pass

        nba_games, bref_games, player_stats = smart_update_data()

        rows_after = len(nba_games) if not nba_games.empty else 0
        new_game_rows = max(rows_after - rows_before, 0)
        new_games_count = new_game_rows // 2

        _set_status(progress="Processing data...")
        games = process_data(nba_games, bref_games)
        features = build_dataset(games, player_stats)

        model_path = MODEL_DIR / "best_model.pkl"
        should_retrain = not model_path.exists() or new_games_count >= RETRAIN_GAME_THRESHOLD

        if should_retrain:
            logger.info(
                "Retraining models (%s new games since last train)",
                new_games_count if new_games_count > 0 else "first run",
            )
            train_models(features)

        generate_predictions(features)
        _save_last_updated()
        _set_status(
            running=False,
            progress="",
            last_updated=datetime.now().isoformat(),
            error=None,
        )
        logger.info("Background update completed")
    except Exception as exc:
        logger.error("Background update failed: %s", exc, exc_info=True)
        _set_status(running=False, progress="", error=str(exc))


def run():
    has_predictions = (PROCESSED_DIR / "predictions.csv").exists()
    has_model = (MODEL_DIR / "best_model.pkl").exists()

    if has_predictions and has_model:
        # Cached data exists — start Flask immediately and update in background
        update_thread = threading.Thread(target=_run_background_update, daemon=True)
        update_thread.start()
        Timer(1.5, _open_browser).start()
    else:
        logger.info("First run or missing model — running full pipeline")
        _set_status(running=True, progress="First-time setup...")
        try:
            nba_games, bref_games, player_stats = collect_data()
            games = process_data(nba_games, bref_games)
            features = build_dataset(games, player_stats)
            train_models(features)
            generate_predictions(features)
            _save_last_updated()
            _set_status(running=False, progress="", last_updated=datetime.now().isoformat())
        except Exception as exc:
            logger.error("Initial pipeline failed: %s", exc, exc_info=True)
            _set_status(running=False, error=str(exc))
        Timer(1.0, _open_browser).start()

    from app import create_app
    app = create_app()
    logger.info("Starting web server at http://%s:%d", FLASK_HOST, FLASK_PORT)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
