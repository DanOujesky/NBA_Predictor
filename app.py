import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath("src"))

from predictor import NBAPredictor
from scraper import NBAScraper
from processor import NBADataProcessor



st.set_page_config(
    page_title="NBA Edge Predictor",
    page_icon="🏀",
    layout="wide"
)

RAW_PATH = "data/raw/nba_raw.csv"
PROCESSED_PATH = "data/processed/nba_processed.csv"

st.markdown("""
<style>
.main {background-color: #0e1117;}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Edge Predictor")
st.caption("ML Betting Analytics Dashboard")
st.markdown("---")

st.sidebar.header("⚙️ Control Panel")

if st.sidebar.button("🔄 Full Data Pipeline"):
    with st.spinner("Running scraper + processor..."):
        scraper = NBAScraper()
        processor = NBADataProcessor()

        teams = [
            "ATL","BOS","BRK","CHO","CHI","CLE","DAL","DEN","DET","GSW",
            "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
            "OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"
        ]

        years = [2023, 2024, 2025]

        scraper.run_bulk_collection(teams, years)
        processor.run_pipeline()

        st.sidebar.success("✅ Data ready!")

@st.cache_data
def load_data():
    if os.path.exists(PROCESSED_PATH):
        return pd.read_csv(PROCESSED_PATH)
    return None

df = load_data()

teams = sorted(df["Team"].unique()) if df is not None else []

@st.cache_resource
def load_model():
    model = NBAPredictor()
    model.load_model()
    return model

predictor = load_model()

def build_feature_vector(df, home_team, away_team):

    latest_home = df[df["Team"] == home_team].sort_values("Date").iloc[-1]
    latest_away = df[df["Team"] == away_team].sort_values("Date").iloc[-1]

    features = {
        "Diff_Points": latest_home["Roll_Team_Points"] - latest_away["Roll_Team_Points"],
        "Diff_FG_pct": latest_home["Roll_Team_FG_pct"] - latest_away["Roll_Team_FG_pct"],
        "Diff_AST": latest_home["Roll_Team_AST"] - latest_away["Roll_Team_AST"],
        "Diff_TOV": latest_home["Roll_Team_TOV"] - latest_away["Roll_Team_TOV"],
        "Is_Home": 1
    }

    return pd.DataFrame([features])

col1, col2, col3 = st.columns(3)

with col1:
    home_team = st.selectbox("🏠 Home Team", teams)

with col2:
    away_team = st.selectbox("✈️ Away Team", teams)

with col3:
    odds = st.number_input("💰 Odds", min_value=1.01, value=1.90)

if st.button("🔍 Analyze Match", use_container_width=True):

    if home_team == away_team:
        st.error("Teams must be different")
        st.stop()

    with st.spinner("Running model..."):

        feature_vector = build_feature_vector(df, home_team, away_team)

        win_prob = predictor.predict_win_probability(feature_vector)

        result = predictor.calculate_betting_value(win_prob, odds)

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Win Probability", f"{result['win_probability']:.1%}")
    c2.metric("Fair Odds", result["fair_odds"])
    c3.metric("EV", f"{result['expected_value']:.2%}")

    if result["has_value"]:
        st.success("🔥 VALUE BET")
    else:
        st.warning("❌ No value")

    st.subheader("🧠 Feature Breakdown")

    fv = feature_vector.iloc[0]
    st.write(fv)

st.markdown("---")
st.header("📊 Analytics")

st.subheader("📈 Win Rate")

df["Win"] = df["Target"]

winrates = df.groupby("Team")["Win"].mean().sort_values(ascending=False)

fig, ax = plt.subplots()
winrates.head(10).plot(kind="bar", ax=ax)
st.pyplot(fig)

st.subheader("🔥 Team Form")

team_form = st.selectbox("Team Form", teams)

team_df = df[df["Team"] == team_form].tail(20)

fig, ax = plt.subplots()
ax.plot(team_df["Target"].values)
st.pyplot(fig)

st.subheader("🧠 Feature Importance")

importance = predictor.model.feature_importances_
features = predictor.config.features

fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)

st.subheader("💰 EV Curve")

odds_range = np.linspace(1.1, 3.0, 50)
current_prob = win_prob if 'win_prob' in locals() else 0.5

ev_values = [(current_prob * o) - 1 for o in odds_range]

fig, ax = plt.subplots()
ax.plot(odds_range, ev_values)
ax.axhline(0)
st.pyplot(fig)

st.subheader("⚔️ Team Comparison")

t1 = st.selectbox("Team A", teams, key="A")
t2 = st.selectbox("Team B", teams, key="B")

def get_latest(team):
    return df[df["Team"] == team].sort_values("Date").iloc[-1]

if t1 and t2:
    s1 = get_latest(t1)
    s2 = get_latest(t2)

    comp = pd.DataFrame({
        "Points": [s1["Roll_Team_Points"], s2["Roll_Team_Points"]],
        "FG%": [s1["Roll_Team_FG_pct"], s2["Roll_Team_FG_pct"]],
        "AST": [s1["Roll_Team_AST"], s2["Roll_Team_AST"]],
        "TOV": [s1["Roll_Team_TOV"], s2["Roll_Team_TOV"]],
    }, index=[t1, t2])

    st.dataframe(comp)

st.markdown("---")
st.caption(f"NBA Edge Predictor © 2026 | {datetime.now().strftime('%Y-%m-%d')}")