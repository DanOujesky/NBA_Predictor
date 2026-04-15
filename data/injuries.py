"""Real-time NBA injury report fetching from ESPN public API."""

import logging
import time

import pandas as pd
import requests

from config import (
    ESPN_ABBR_FIXES,
    INJURY_STATUS_MAP,
    RAW_DIR,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    TEAM_ABBREVIATION_FIXES,
)

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

STATUS_TO_AVAILABILITY = INJURY_STATUS_MAP

PLAYER_VALUE_WEIGHTS = {
    "PTS": 1.0, "REB": 0.7, "AST": 1.0,
    "STL": 1.5, "BLK": 1.2, "TOV": -1.0,
}


def _normalize_espn_abbr(espn_abbr: str) -> str:
    """Convert ESPN team abbreviation to our canonical 3-letter form."""
    abbr = ESPN_ABBR_FIXES.get(espn_abbr, espn_abbr)
    return TEAM_ABBREVIATION_FIXES.get(abbr, abbr)


def _compute_player_values(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Add a composite player_value column to a player stats DataFrame."""
    df = player_stats.copy()
    df["player_value"] = sum(
        df[col].fillna(0) * w
        for col, w in PLAYER_VALUE_WEIGHTS.items()
        if col in df.columns
    )
    return df


def _normalise_player_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Rename NBA API column names to a unified schema and normalise abbreviations."""
    df = _compute_player_values(player_stats).copy()
    df = df.rename(columns={
        "PLAYER_NAME": "player_name",
        "TEAM_ABBREVIATION": "team_abbr",
    })
    if "team_abbr" in df.columns:
        df["team_abbr"] = df["team_abbr"].map(
            lambda x: TEAM_ABBREVIATION_FIXES.get(str(x).strip(), str(x).strip())
        )
    return df


class InjuryReportFetcher:
    """Fetches the current NBA injury report from ESPN's public API.

    Falls back to GP-based availability derived from player stats if the
    ESPN request fails or returns no data.
    """

    def fetch(self, player_stats: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return a player-level injury DataFrame and persist it to disk.

        Columns: player_name, team_abbr, status, injury_type, comment,
                 availability_factor, is_available, player_value.
        """
        try:
            df = self._fetch_espn(player_stats)
        except Exception as exc:
            logger.error("Unexpected error in ESPN fetch: %s", exc, exc_info=True)
            df = None

        if df is not None and not df.empty:
            df.to_csv(RAW_DIR / "injury_report.csv", index=False)
            logger.info("Saved ESPN injury report: %d entries", len(df))
            return df

        logger.warning("ESPN injury fetch failed — using GP-based fallback")
        if player_stats is not None and not player_stats.empty:
            return self._gp_based_fallback(player_stats)

        logger.error("No player stats available for fallback injury report")
        return pd.DataFrame()

    def _fetch_espn(self, player_stats: pd.DataFrame | None) -> pd.DataFrame | None:
        """Request injury data from ESPN and return a normalised DataFrame."""
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(ESPN_INJURIES_URL, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("ESPN API returned HTTP %d", resp.status_code)
                return None
            data = resp.json()
        except Exception as exc:
            logger.error("ESPN injury request failed: %s", exc)
            return None

        rows = []
        for team_block in data.get("injuries", []):
            for entry in team_block.get("injuries", []):
                athlete = entry.get("athlete", {})
                athlete_team = athlete.get("team", {})
                espn_abbr = athlete_team.get("abbreviation", "")
                if not espn_abbr:
                    from config import TEAM_NAME_TO_ABBR
                    espn_abbr = TEAM_NAME_TO_ABBR.get(
                        athlete_team.get("displayName", team_block.get("displayName", "")), ""
                    )
                team_abbr = _normalize_espn_abbr(espn_abbr) if espn_abbr else ""

                raw_status = entry.get("status", "").strip()
                avail = STATUS_TO_AVAILABILITY.get(raw_status.lower(), 1.0)
                details = entry.get("details", {})
                injury_type = details.get("type", entry.get("type", {}).get("abbreviation", ""))
                rows.append({
                    "player_name": athlete.get("displayName", ""),
                    "team_abbr": team_abbr,
                    "status": raw_status,
                    "injury_type": injury_type,
                    "comment": entry.get("shortComment", details.get("detail", "")),
                    "availability_factor": float(avail),
                    "is_available": bool(avail > 0.5),
                    "player_value": 0.0,
                })

        if not rows:
            logger.warning("ESPN returned 0 injury entries")
            return None

        df = pd.DataFrame(rows)
        df = self._attach_player_values(df, player_stats)
        return df

    def _attach_player_values(
        self,
        injury_df: pd.DataFrame,
        player_stats: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Join per-game player values from player_stats onto the injury report."""
        if player_stats is None or player_stats.empty:
            return injury_df

        ps = _normalise_player_stats(player_stats)
        if "player_name" not in ps.columns:
            return injury_df

        merged = injury_df.merge(
            ps[["player_name", "player_value"]].rename(columns={"player_value": "_pv"}),
            on="player_name",
            how="left",
        )
        merged["player_value"] = merged["_pv"].fillna(merged["player_value"])
        return merged.drop(columns=["_pv"])

    def _gp_based_fallback(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Estimate availability from games-played ratio when ESPN is unavailable."""
        ps = _normalise_player_stats(player_stats)
        max_gp = ps["GP"].max() if "GP" in ps.columns else 1
        ps["availability_factor"] = (ps.get("GP", max_gp) / max_gp).clip(0, 1)
        ps["is_available"] = ps["availability_factor"] > 0.7
        ps["status"] = ps["is_available"].map({True: "Active", False: "Out"})
        ps["injury_type"] = ""
        ps["comment"] = ""

        keep = [
            "player_name", "team_abbr", "status", "injury_type",
            "comment", "availability_factor", "is_available", "player_value",
        ]
        result = ps[[c for c in keep if c in ps.columns]]
        result.to_csv(RAW_DIR / "injury_report.csv", index=False)
        logger.info("GP-based injury fallback: %d players", len(result))
        return result


def compute_team_availability(
    injury_df: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute team-level roster availability ratios.

    Uses player_stats (full roster) as the denominator and the injury report
    to identify unavailable players. If player_stats is not provided, falls back
    to using only the players listed in the injury report.

    Returns a DataFrame with columns:
        team_abbr, roster_strength, available_strength, availability_ratio.
    """
    if injury_df.empty and (player_stats is None or player_stats.empty):
        return pd.DataFrame(
            columns=["team_abbr", "roster_strength", "available_strength", "availability_ratio"]
        )

    if player_stats is not None and not player_stats.empty:
        ps = _normalise_player_stats(player_stats)
        if "team_abbr" not in ps.columns:
            logger.warning("player_stats missing team column; falling back to injury-only mode")
            return _availability_from_injury_only(injury_df)

        full = ps.groupby("team_abbr")["player_value"].sum().rename("roster_strength")

        if not injury_df.empty:
            inj = injury_df.copy()
            if inj["is_available"].dtype == object:
                inj["is_available"] = inj["is_available"].map(
                    lambda x: str(x).strip().lower() == "true"
                )
            inj["availability_factor"] = pd.to_numeric(inj["availability_factor"], errors="coerce").fillna(1.0)

            if inj["player_value"].fillna(0).eq(0).all():
                if "player_name" in inj.columns and "player_name" in ps.columns:
                    inj = inj.merge(
                        ps[["player_name", "team_abbr", "player_value"]].rename(
                            columns={"player_value": "_pv"}
                        ),
                        on=["player_name", "team_abbr"],
                        how="left",
                    )
                    inj["player_value"] = inj["_pv"].fillna(0.0)
                    inj = inj.drop(columns=["_pv"])

            inj["lost_value"] = inj["player_value"] * (1.0 - inj["availability_factor"].clip(0, 1))
            lost = inj.groupby("team_abbr")["lost_value"].sum()
        else:
            lost = pd.Series(dtype=float)

        result = pd.DataFrame({"roster_strength": full})
        result["lost_value"] = lost.reindex(result.index).fillna(0.0)
        result["available_strength"] = (result["roster_strength"] - result["lost_value"]).clip(lower=0)
        safe_total = result["roster_strength"].replace(0, 1)
        result["availability_ratio"] = (result["available_strength"] / safe_total).clip(0, 1)
        return result.reset_index()

    return _availability_from_injury_only(injury_df)


def _availability_from_injury_only(injury_df: pd.DataFrame) -> pd.DataFrame:
    """Compute availability using only injured players when no full roster is available."""
    df = injury_df.copy()
    if df.empty:
        return pd.DataFrame(
            columns=["team_abbr", "roster_strength", "available_strength", "availability_ratio"]
        )

    if df["is_available"].dtype == object:
        df["is_available"] = df["is_available"].map(
            lambda x: str(x).strip().lower() == "true"
        )
    df["availability_factor"] = pd.to_numeric(df["availability_factor"], errors="coerce").fillna(1.0)
    df["player_value"] = pd.to_numeric(df["player_value"], errors="coerce").fillna(0.0)
    df["weighted_value"] = df["player_value"] * df["availability_factor"]

    total = df.groupby("team_abbr")["player_value"].sum().rename("roster_strength")
    available = (
        df[df["is_available"]]
        .groupby("team_abbr")["weighted_value"]
        .sum()
        .rename("available_strength")
    )
    result = pd.DataFrame({"roster_strength": total, "available_strength": available}).fillna(0)
    safe_total = result["roster_strength"].replace(0, 1)
    result["availability_ratio"] = (result["available_strength"] / safe_total).clip(0, 1)
    return result.reset_index()
