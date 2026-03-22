import streamlit as st
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import math
from datetime import date, timedelta
import os

# ====================== SAFE CONVERSION HELPER ======================
def safe_float(val, default=0.0):
    """Convert API values (strings like '2.3' or '55%' or numbers) safely to float."""
    if val is None:
        return default
    try:
        s = str(val).replace("%", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return default

st.set_page_config(page_title="SportyBet AI Predictor v2", layout="wide")
st.title("⚽ SportyBet AI Predictor v2 – Stats + Poisson + XGBoost Learning")

# ====================== SETUP ======================
API_KEY = st.sidebar.text_input("API-Football Key", type="password")
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY} if API_KEY else None

HISTORY_FILE = "prediction_history.csv"
if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=["date", "fixture_id", "match", "market", "selection", "prob", 
                                    "form_diff", "att_diff", "def_diff", "poisson_home", "actual", "correct"])

le_market = LabelEncoder()

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k) if lam > 0 else 0

def calculate_poisson_markets(lambda_h, lambda_a):
    # BTTS
    p0_h = poisson_pmf(0, lambda_h)
    p0_a = poisson_pmf(0, lambda_a)
    btts_yes = 1 - p0_h - p0_a + p0_h * p0_a
    # Top 3 correct scores
    scores = []
    for i in range(0, 6):
        for j in range(0, 6):
            p = poisson_pmf(i, lambda_h) * poisson_pmf(j, lambda_a)
            scores.append(((i, j), round(p*100, 1)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return round(btts_yes*100, 1), scores[:3]

# ====================== FETCH & PREDICT ======================
def get_prediction(fixture_id):
    r = requests.get(f"{BASE_URL}/predictions?fixture={fixture_id}", headers=headers)
    if r.status_code != 200 or not r.json()["response"]:
        return None
    data = r.json()["response"][0]
    
    teams = data["teams"]
    pred = data.get("predictions", {})
    comp = data.get("comparison", {})
    match_str = f"{teams['home']['name']} vs {teams['away']['name']}"
    
    # Extract lambdas from last_5 averages (FIXED: safe float conversion)
    home_last5_for = safe_float(
        teams["home"].get("last_5", {}).get("goals", {}).get("for", {}).get("average", 1.3), 1.3
    )
    away_last5_against = safe_float(
        teams["away"].get("last_5", {}).get("goals", {}).get("against", {}).get("average", 1.3), 1.3
    )
    lambda_h = (home_last5_for + away_last5_against) / 2

    away_last5_for = safe_float(
        teams["away"].get("last_5", {}).get("goals", {}).get("for", {}).get("average", 1.3), 1.3
    )
    home_last5_against = safe_float(
        teams["home"].get("last_5", {}).get("goals", {}).get("against", {}).get("average", 1.3), 1.3
    )
    lambda_a = (away_last5_for + home_last5_against) / 2
    
    btts_prob, top_scores = calculate_poisson_markets(lambda_h, lambda_a)
    
    markets = []
    # 1X2 (FIXED: selection now matches auto-update logic)
    percent = pred.get("percent", {})
    for outcome, p in percent.items():
        if outcome == "home":
            selection = "Home Win"
        elif outcome == "away":
            selection = "Away Win"
        else:
            selection = "Draw"
        markets.append({
            "market": "1X2",
            "selection": selection,
            "prob": int(safe_float(p, 50)),
            "advice": pred.get("advice", "")
        })
    
    # Under/Over (API)
    if "under_over" in pred:
        markets.append({"market": "Over/Under", "selection": pred["under_over"], "prob": 65, "advice": ""})
    
    # BTTS
    markets.append({"market": "BTTS", "selection": "Yes" if btts_prob > 50 else "No", "prob": btts_prob, "advice": ""})
    
    # Correct Score (top one)
    cs = top_scores[0][0]
    markets.append({"market": "Correct Score", "selection": f"{cs[0]}-{cs[1]}", "prob": top_scores[0][1], "advice": ""})
    
    # Features for XGBoost (FIXED: fully robust parsing)
    features = {
        "form_diff": safe_float(comp.get("form", {}).get("home", "50")) - safe_float(comp.get("form", {}).get("away", "50")),
        "att_diff": safe_float(comp.get("att", {}).get("home", "50")) - safe_float(comp.get("att", {}).get("away", "50")),
        "def_diff": safe_float(comp.get("def", {}).get("home", "50")) - safe_float(comp.get("def", {}).get("away", "50")),
        "poisson_home": safe_float(comp.get("poisson_distribution", {}).get("home", "50"))
    }
    
    return {
        "match": match_str,
        "fixture_id": fixture_id,
        "markets": markets,
        "features": features,
        "lambda_h": lambda_h,
        "lambda_a": lambda_a
    }

# ====================== HISTORY & AUTO UPDATE ======================
def save_to_history(row_dict):
    global history
    new_row = pd.DataFrame([row_dict])
    history = pd.concat([history, new_row], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

def auto_update_history():
    if history.empty:
        st.info("No history yet.")
        return
    with st.spinner("Fetching real results and updating history..."):
        for idx, row in history.iterrows():
            if pd.isna(row.get("actual")) or row.get("correct") is None:
                r = requests.get(f"{BASE_URL}/fixtures?id={row['fixture_id']}", headers=headers)
                if r.status_code == 200:
                    fix = r.json()["response"][0]
                    if fix["fixture"]["status"]["short"] == "FT":
                        goals_h = fix["goals"]["home"]
                        goals_a = fix["goals"]["away"]
                        actual_result = "Home Win" if goals_h > goals_a else "Away Win" if goals_a > goals_h else "Draw"
                        
                        # FIXED: proper correct logic for ALL markets
                        if row["market"] == "1X2":
                            correct = (row["selection"] == actual_result)
                        elif row["market"] == "Over/Under":
                            total = goals_h + goals_a
                            if row["selection"] == "Over 2.5":
                                correct = total > 2.5
                            elif row["selection"] == "Under 2.5":
                                correct = total <= 2.5
                            else:
                                correct = False
                        elif row["market"] == "BTTS":
                            btts_yes = (goals_h > 0 and goals_a > 0)
                            correct = (row["selection"] == "Yes" and btts_yes) or (row["selection"] == "No" and not btts_yes)
                        elif row["market"] == "Correct Score":
                            correct = (row["selection"] == f"{goals_h}-{goals_a}")
                        else:
                            correct = False
                        
                        history.at[idx, "actual"] = f"{goals_h}-{goals_a}"
                        history.at[idx, "correct"] = correct
        history.to_csv(HISTORY_FILE, index=False)
        st.success("History updated with real results! XGBoost will now be smarter.")

# ====================== XGBoost LEARNING ======================
def train_xgboost():
    if len(history) < 15:
        return None, None
    df = history.copy()
    df = df.dropna(subset=["correct"])
    if len(df) < 15:
        return None, None
    
    df["market_enc"] = le_market.fit_transform(df["market"].astype(str))
    
    X = df[["prob", "form_diff", "att_diff", "def_diff", "poisson_home", "market_enc"]]
    y = df["correct"].astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model, le_market

# ====================== MAIN APP ======================
date_to_check = st.sidebar.date_input("Date", value=date.today() + timedelta(days=1))
colA, colB = st.sidebar.columns(2)
generate_btn = colA.button("Generate Predictions")
update_btn = colB.button("🔄 Auto-Update History")

if update_btn and API_KEY:
    auto_update_history()

if generate_btn and API_KEY:
    with st.spinner("Fetching fixtures + predictions + Poisson..."):
        fixtures = requests.get(f"{BASE_URL}/fixtures?date={date_to_check.strftime('%Y-%m-%d')}", headers=headers).json().get("response", [])
        
        predictions = []
        for f in fixtures[:20]:
            if f["fixture"]["status"]["short"] == "NS":
                pred = get_prediction(f["fixture"]["id"])
                if pred:
                    predictions.append(pred)
        
        model, encoder = train_xgboost()
        
        all_bets = []
        for p in predictions:
            for m in p["markets"]:
                base_prob = m["prob"]
                if model is not None:
                    feat = pd.DataFrame([{
                        "prob": base_prob,
                        "form_diff": p["features"]["form_diff"],
                        "att_diff": p["features"]["att_diff"],
                        "def_diff": p["features"]["def_diff"],
                        "poisson_home": p["features"]["poisson_home"],
                        "market_enc": encoder.transform([m["market"]])[0]
                    }])
                    ml_prob = model.predict_proba(feat)[0][1] * 100
                    final_conf = round((base_prob * 0.6 + ml_prob * 0.4), 1)
                else:
                    final_conf = base_prob
                
                all_bets.append({
                    "match": p["match"],
                    "market": m["market"],
                    "selection": m["selection"],
                    "confidence": final_conf,
                    "fixture_id": p["fixture_id"],
                    "advice": m["advice"]
                })
        
        df = pd.DataFrame(all_bets).sort_values("confidence", ascending=False)
        
        # ====================== DISPLAY GROUPS ======================
        st.header("📋 Ready-to-Paste SportyBet Selections (XGBoost boosted)")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("🔒 Safe Acca (≥68%)")
            for _, r in df[df["confidence"] >= 68].iterrows():
                st.write(f"**{r['match']}** → **{r['market']}: {r['selection']}** ({r['confidence']}%)")
        
        with c2:
            st.subheader("⚖️ Medium Acca")
            for _, r in df[(df["confidence"] >= 55) & (df["confidence"] < 68)].iterrows():
                st.write(f"**{r['match']}** → **{r['market']}: {r['selection']}** ({r['confidence']}%)")
        
        with c3:
            st.subheader("🎲 High-Odds + Poisson Specials")
            for _, r in df[df["confidence"] < 55].iterrows():
                st.write(f"**{r['match']}** → **{r['market']}: {r['selection']}** ({r['confidence']}%)")
        
        st.info("**How to get SportyBet code:** Paste each line into SportyBet → Book bet → copy code.")
        
        # ====================== LOG OUTCOME (for learning) ======================
        st.subheader("Log a prediction outcome (or use Auto-Update)")
        fid = st.number_input("Fixture ID", step=1)
        market_sel = st.selectbox("Market", ["1X2", "Over/Under", "BTTS", "Correct Score"])
        sel = st.text_input("Your selection (e.g. Home Win / Yes / 2-1)")
        was_correct = st.radio("Was it correct?", ["Yes", "No"])
        if st.button("Save to history & retrain"):
            save_to_history({
                "date": str(date.today()),
                "fixture_id": fid,
                "match": "Manual log",
                "market": market_sel,
                "selection": sel,
                "prob": 60,
                "form_diff": 0, "att_diff": 0, "def_diff": 0, "poisson_home": 50,
                "actual": "Logged",
                "correct": was_correct == "Yes"
            })
            st.success("Saved! Model is now learning from this.")
        
        st.header("📚 Your Learning History")
        st.dataframe(history.sort_values("date", ascending=False))
        
        st.caption("✅ Auto-results updater + Poisson BTTS/Correct Score + XGBoost that improves every day. Free tier still enough.")
