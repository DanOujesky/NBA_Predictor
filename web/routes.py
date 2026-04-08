"""Flask routes serving the web dashboard and prediction API."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from config import MODEL_DIR, PROCESSED_DIR, RAW_DIR, TEAM_ABBR_TO_NAME
from features.builder import ALL_FEATURES

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


def _load_model():
    """Load the best trained model from disk."""
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_model_meta() -> str:
    """Read which model type was saved as best."""
    meta = MODEL_DIR / "best_model_meta.txt"
    return meta.read_text().strip() if meta.exists() else "Unknown"


def _load_comparison() -> list[dict]:
    """Load model comparison results if available."""
    path = PROCESSED_DIR / "model_comparison.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _load_injury_report(team: str = "") -> list[dict]:
    """Load the injury report and enrich it with per-game stats (PTS/REB/AST).

    Joins player_stats.csv to attach PPG, RPG, APG for each injured player.
    Returns only unavailable players, sorted by player_value descending.
    """
    path = RAW_DIR / "injury_report.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    # Normalise column names
    df = df.rename(columns={"PLAYER_NAME": "player_name", "TEAM_ABBREVIATION": "team_abbr"})

    # Join per-game stats so the frontend can show PPG / RPG / APG
    ps_path = RAW_DIR / "player_stats.csv"
    if ps_path.exists():
        ps = pd.read_csv(ps_path, usecols=lambda c: c in (
            "PLAYER_NAME", "PTS", "REB", "AST"
        )).rename(columns={"PLAYER_NAME": "player_name"})
        df = df.merge(ps, on="player_name", how="left")

    # Compute player_value if still missing
    if "player_value" not in df.columns or df["player_value"].fillna(0).eq(0).all():
        from features.player_features import PLAYER_VALUE_WEIGHTS
        df["player_value"] = sum(
            df[col].fillna(0) * w
            for col, w in PLAYER_VALUE_WEIGHTS.items()
            if col in df.columns
        )

    # Filter to players who are not fully available
    if "is_available" in df.columns:
        mask = ~df["is_available"].map(lambda x: str(x).strip().lower() == "true")
        unavailable = df[mask]
    elif "availability_factor" in df.columns:
        unavailable = df[df["availability_factor"].lt(1.0)]
    else:
        unavailable = df

    if team:
        unavailable = unavailable[
            unavailable["team_abbr"].str.upper() == team.upper()
        ]

    unavailable = unavailable.sort_values("player_value", ascending=False)
    # Expose team_abbr as TEAM_ABBREVIATION so the frontend fallback works
    unavailable = unavailable.rename(columns={"team_abbr": "TEAM_ABBREVIATION"})
    return unavailable.head(100).fillna("").to_dict("records")


def _load_predictions() -> list[dict]:
    """Load pre-computed game predictions if available."""
    path = PROCESSED_DIR / "predictions.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


@bp.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


@bp.route("/api/predictions")
def api_predictions():
    """Return game predictions as JSON."""
    return jsonify({
        "predictions": _load_predictions(),
        "model_name": _load_model_meta(),
    })


@bp.route("/api/models")
def api_models():
    """Return model comparison metrics."""
    return jsonify({"models": _load_comparison()})


@bp.route("/api/injuries")
def api_injuries():
    """Return current injury report for all or a specific team."""
    team = request.args.get("team", "").upper()
    return jsonify({"injuries": _load_injury_report(team)})


@bp.route("/api/teams")
def api_teams():
    """Return list of all NBA teams with abbreviation and full name."""
    teams = [{"abbr": k, "name": v} for k, v in TEAM_ABBR_TO_NAME.items()]
    return jsonify({"teams": sorted(teams, key=lambda t: t["name"])})


@bp.route("/api/status")
def api_status():
    """Return system status: model trained, data available, predictions ready."""
    model_exists = (MODEL_DIR / "best_model.pkl").exists()
    features_exist = (PROCESSED_DIR / "features.csv").exists()
    predictions_exist = (PROCESSED_DIR / "predictions.csv").exists()
    injury_exists = (RAW_DIR / "injury_report.csv").exists()
    return jsonify({
        "model_trained": model_exists,
        "data_processed": features_exist,
        "predictions_ready": predictions_exist,
        "injury_report_available": injury_exists,
        "best_model": _load_model_meta() if model_exists else None,
    })
