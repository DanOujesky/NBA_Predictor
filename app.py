import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath("src"))
from predictor import NBAPredictor

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="centered")

PROCESSED_PATH = "data/processed/nba_processed.csv"

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 40px; color: #00ffcc; }
    .stSelectbox label { font-size: 18px; font-weight: bold; }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #1c1f26;
        border: 1px solid #3d4452;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists(PROCESSED_PATH):
        return pd.read_csv(PROCESSED_PATH)
    return None

@st.cache_resource
def load_model():
    model = NBAPredictor()
    try:
        model.load_model()
        return model
    except:
        return None

st.title("🏀 NBA Edge Predictor")
st.write("Select match to calculate AI win probability and fair market odds.")

df = load_data()
predictor = load_model()

if df is None or predictor is None:
    st.error("Missing data or model. Please run the data pipeline first.")
    st.stop()

teams = sorted(df["Team"].unique())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 HOME TEAM", teams, index=0)
with col2:
    away_team = st.selectbox("✈️ AWAY TEAM", teams, index=1)

st.markdown("---")

if home_team == away_team:
    st.warning("Please select two different teams to analyze the match.")
else:
    latest_matchup = df[df["Team"] == home_team].sort_values("Date").iloc[-1:]
    
    prob = predictor.predict_win_probability(latest_matchup[list(predictor.config.features)])
    
    fair_odds = 1 / prob if prob > 0 else 0

    st.markdown(f"### Analysis: {home_team} vs {away_team}")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Win Probability", f"{prob:.1%}")
    res_col2.metric("Minimum Fair Odds", f"{fair_odds:.2f}")


st.markdown("---")
st.caption(f"NBA Predictor Pro | {datetime.now().strftime('%Y-%m-%d %H:%M')}")