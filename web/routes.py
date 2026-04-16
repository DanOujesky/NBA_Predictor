"""Flask Blueprint s API endpointy pro webový dashboard NBA Predictoru.

Endpointy:
  GET /                     - Hlavní stránka (index.html z vendor/frontend/)
  GET /api/predictions      - Predikce nadcházejících zápasů + jméno modelu
  GET /api/models           - Porovnávací tabulka všech natrénovaných modelů
  GET /api/injuries         - Zpráva o zraněních (volitelný parametr ?team=XXX)
  GET /api/teams            - Seznam všech 30 NBA týmů (zkratka + název)
  GET /api/status           - Stav systému (model, data, predikce)
  GET /api/update-status    - Průběh probíhající aktualizace dat
"""

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
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_model_meta() -> str:
    meta = MODEL_DIR / "best_model_meta.txt"
    return meta.read_text().strip() if meta.exists() else "Unknown"


def _load_comparison() -> list[dict]:
    path = PROCESSED_DIR / "model_comparison.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _load_injury_report(team: str = "") -> list[dict]:
    path = RAW_DIR / "injury_report.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    df = df.rename(columns={"PLAYER_NAME": "player_name", "TEAM_ABBREVIATION": "team_abbr"})

    ps_path = RAW_DIR / "player_stats.csv"
    if ps_path.exists():
        ps = pd.read_csv(ps_path, usecols=lambda c: c in (
            "PLAYER_NAME", "PTS", "REB", "AST"
        )).rename(columns={"PLAYER_NAME": "player_name"})
        df = df.merge(ps, on="player_name", how="left")

    if "player_value" not in df.columns or df["player_value"].fillna(0).eq(0).all():
        from features.player_features import PLAYER_VALUE_WEIGHTS
        df["player_value"] = sum(
            df[col].fillna(0) * w
            for col, w in PLAYER_VALUE_WEIGHTS.items()
            if col in df.columns
        )

    if "is_available" in df.columns:
        mask = ~df["is_available"].map(lambda x: str(x).strip().lower() == "true")
        unavailable = df[mask]
    elif "availability_factor" in df.columns:
        unavailable = df[df["availability_factor"].lt(1.0)]
    else:
        unavailable = df

    if team:
        unavailable = unavailable[unavailable["team_abbr"].str.upper() == team.upper()]

    unavailable = unavailable.sort_values("player_value", ascending=False)
    unavailable = unavailable.rename(columns={"team_abbr": "TEAM_ABBREVIATION"})
    return unavailable.head(100).fillna("").to_dict("records")


def _load_predictions() -> list[dict]:
    path = PROCESSED_DIR / "predictions.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/predictions")
def api_predictions():
    return jsonify({
        "predictions": _load_predictions(),
        "model_name": _load_model_meta(),
    })


@bp.route("/api/models")
def api_models():
    return jsonify({"models": _load_comparison()})


@bp.route("/api/injuries")
def api_injuries():
    team = request.args.get("team", "").upper()
    return jsonify({"injuries": _load_injury_report(team)})


@bp.route("/api/teams")
def api_teams():
    teams = [{"abbr": k, "name": v} for k, v in TEAM_ABBR_TO_NAME.items()]
    return jsonify({"teams": sorted(teams, key=lambda t: t["name"])})


@bp.route("/api/status")
def api_status():
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


@bp.route("/api/update-status")
def api_update_status():
    from pipeline import get_update_status
    return jsonify(get_update_status())


