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


def _load_injury_report() -> list[dict]:
    """Load injury/availability report if available."""
    path = RAW_DIR / "injury_report.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    injured = df[~df["is_available"]].sort_values("player_value", ascending=False)
    return injured.head(50).to_dict("records")


def _load_predictions() -> list[dict]:
    """Load pre-computed game predictions if available."""
    path = PROCESSED_DIR / "predictions.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


@bp.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


@bp.route("/api/predictions")
def api_predictions():
    """Return game predictions as JSON."""
    predictions = _load_predictions()
    return jsonify({
        "predictions": predictions,
        "model_name": _load_model_meta(),
    })


@bp.route("/api/models")
def api_models():
    """Return model comparison metrics."""
    return jsonify({"models": _load_comparison()})


@bp.route("/api/injuries")
def api_injuries():
    """Return current injury/availability report."""
    team = request.args.get("team", "").upper()
    report = _load_injury_report()
    if team:
        report = [r for r in report if r.get("TEAM_ABBREVIATION", "") == team]
    return jsonify({"injuries": report})


@bp.route("/api/teams")
def api_teams():
    """Return list of all NBA teams."""
    teams = [{"abbr": k, "name": v} for k, v in TEAM_ABBR_TO_NAME.items()]
    return jsonify({"teams": sorted(teams, key=lambda t: t["name"])})


@bp.route("/api/status")
def api_status():
    """Return system status (model trained, data available, etc.)."""
    model_exists = (MODEL_DIR / "best_model.pkl").exists()
    features_exist = (PROCESSED_DIR / "features.csv").exists()
    predictions_exist = (PROCESSED_DIR / "predictions.csv").exists()
    return jsonify({
        "model_trained": model_exists,
        "data_processed": features_exist,
        "predictions_ready": predictions_exist,
        "best_model": _load_model_meta() if model_exists else None,
    })
