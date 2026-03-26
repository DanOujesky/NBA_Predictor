import streamlit as st
import pandas as pd
import os
import joblib
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from src.scraper import NBAScraper
from src.processor import NBADataProcessor
from src.predictor import NBAPredictor

RAW_DATA_PATH = Path("data/raw/nba_raw.csv")
FUTURE_GAMES_PATH = Path("data/raw/future_games.csv")
PROCESSED_DATA_PATH = Path("data/processed/nba_processed.csv")
SCALER_PATH = Path("models/scaler.pkl")

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

class NBAAppManager:
    def __init__(self):
        self.scraper = NBAScraper()
        self.processor = NBADataProcessor()
        self.predictor = NBAPredictor()
        
    def run_full_pipeline(self):
        with st.status("🔄 Aktualizace systému a AI modelu...", expanded=True) as status:
            st.write("1️⃣ Scrapování nejnovějších výsledků...")
            self.scraper.run()
            
            if RAW_DATA_PATH.exists():
                os.utime(RAW_DATA_PATH, None)
            
            st.write("2️⃣ Aktualizace rozpisu budoucích zápasů...")
            self.scraper.save_future_games()
            
            st.write("3️⃣ Přepočítávání statistik a diferenciálů...")
            self.processor.run_pipeline()
            
            st.write("4️⃣ Retraining predikčního modelu...")
            self.predictor.train_model()
            self.predictor.save_model()
            
            status.update(label="Všechna data jsou aktuální!", state="complete", expanded=False)

    def get_predictions_for_future(self):
        if not FUTURE_GAMES_PATH.exists() or not PROCESSED_DATA_PATH.exists():
            return None

        team_map = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
            "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
        }

        try:
            future_df = pd.read_csv(FUTURE_GAMES_PATH)
            processed_df = pd.read_csv(PROCESSED_DATA_PATH)
            processed_df['Date'] = pd.to_datetime(processed_df['Date'])
            
            scaler = joblib.load(SCALER_PATH)
            self.predictor.load_model()

            latest_stats = processed_df.sort_values('Date').groupby('Team').tail(1)
            
            predictions = []
            for _, row in future_df.iterrows():
                home_full = row['Home']
                away_full = row['Away']
                home_code = team_map.get(home_full)
                away_code = team_map.get(away_full)
                
                h_data = latest_stats[latest_stats['Team'] == home_code]
                a_data = latest_stats[latest_stats['Team'] == away_code]
                
                if not h_data.empty and not a_data.empty:
                    diff_features = pd.DataFrame([{
                        "Diff_Points": h_data['Roll_Team_Points'].values[0] - a_data['Roll_Team_Points'].values[0],
                        "Diff_FG_pct": h_data['Roll_Team_FG_pct'].values[0] - a_data['Roll_Team_FG_pct'].values[0],
                        "Diff_AST": h_data['Roll_Team_AST'].values[0] - a_data['Roll_Team_AST'].values[0],
                        "Diff_TOV": h_data['Roll_Team_TOV'].values[0] - a_data['Roll_Team_TOV'].values[0],
                        "Diff_Form": h_data['Form'].values[0] - a_data['Form'].values[0],
                        "Diff_Streak": h_data['Streak'].values[0] - a_data['Streak'].values[0]
                    }])
                    
                    diff_scaled = scaler.transform(diff_features)
                    features_final = pd.DataFrame(diff_scaled, columns=diff_features.columns)
                    features_final["Is_Home"] = 1
                    
                    prob = self.predictor.predict_win_probability(features_final)
                    predictions.append({
                        "Datum": row['Date'],
                        "Domácí": home_full,
                        "Hosté": away_full,
                        "Pravděpodobnost výhry": prob,
                        "Fair Kurz": 1/prob if prob > 0 else 0
                    })
            return pd.DataFrame(predictions)
        except Exception:
            return None

app_manager = NBAAppManager()

st.title("🏀 NBA Edge Predictor Pro")
st.markdown("Analýza zápasů na základě pokročilých rolling statistik a AI modelu.")

with st.sidebar:
    st.image("https://cdn.nba.com/logos/nba/nba-logoman-75.svg", width=50)
    st.header("Ovládací panel")
    
    if 'last_upd' not in st.session_state:
        if RAW_DATA_PATH.exists():
            st.session_state.last_upd = datetime.fromtimestamp(os.path.getmtime(RAW_DATA_PATH)).strftime('%Y-%m-%d %H:%M')
        else:
            st.session_state.last_upd = "Nikdy"

    st.info(f"Poslední aktualizace: \n{st.session_state.last_upd}")
    
    if st.button("🔄 Aktualizovat všechna data", width="stretch"):
        app_manager.run_full_pipeline()
        st.session_state.last_upd = datetime.now().strftime('%Y-%m-%d %H:%M')
        st.success("Aktualizace hotova!")
        st.rerun()

st.subheader("Predikce pro nadcházející utkání")
predictions_df = app_manager.get_predictions_for_future()

if predictions_df is not None and not predictions_df.empty:
    display_df = predictions_df.copy()
    display_df['Pravděpodobnost výhry'] = (display_df['Pravděpodobnost výhry'] * 100).map('{:.1f}%'.format)
    display_df['Fair Kurz'] = display_df['Fair Kurz'].map('{:.2f}'.format)
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(predictions_df.head(3).iterrows()):
        with cols[i % 3]:
            prob = row['Pravděpodobnost výhry']
            color = "#00ffcc" if prob > 0.6 else "#ff4b4b"
            st.markdown(f"""
            <div style="border: 1px solid #444; padding: 15px; border-radius: 10px; text-align: center;">
                <small>{row['Datum']}</small><br>
                <strong>{row['Domácí']}</strong> vs {row['Hosté']}<br>
                <h2 style="color: {color};">{prob:.1%}</h2>
                <small>Fair kurz: {row['Fair Kurz']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("---")
    st.dataframe(display_df, width="stretch", hide_index=True)
else:
    st.warning("Zatím nejsou k dispozici žádné predikce. Spusťte aktualizaci dat v bočním panelu.")