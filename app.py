import streamlit as st
import requests
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import math
from datetime import date
from supabase import create_client

# ====================== SUPABASE ======================
@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

def load_history():
    client = get_supabase()
    response = client.table("prediction_history").select("*").execute()
    data = response.data
    cols = ["id", "date", "fixture_id", "match", "market", "selection", "prob",
            "form_diff", "att_diff", "def_diff", "poisson_home", "actual", "correct"]
    return pd.DataFrame(data) if data else pd.DataFrame(columns=cols)

def save_to_history(row_dict):
    client = get_supabase()
    check = client.table("prediction_history").select("id") \
        .eq("date", row_dict["date"]) \
        .eq("fixture_id", row_dict["fixture_id"]) \
        .eq("market", row_dict["market"]) \
        .eq("selection", row_dict["selection"]) \
        .execute()
    if not check.data:
        client.table("prediction_history").insert(row_dict).execute()
        return True
    return False

# ====================== THEME & SAFE FLOAT ======================
st.markdown("""
<style>
    .stApp { background-color: #1E1E2E; color: #E0E0E0; }
    h1, h2, h3, .stSubheader { color: #9C27B0 !important; }
    .stButton>button { background-color: #9C27B0; color: white; border: none; padding: 12px; font-weight: bold; }
    .stButton>button:hover { background-color: #7B1FA2; }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input { background-color: #2C2C3E; color: #E0E0E0; }
</style>
""", unsafe_allow_html=True)

def safe_float(val, default=0.0):
    if val is None: return default
    try:
        return float(str(val).replace("%", "").strip())
    except:
        return default

st.set_page_config(page_title="SportyBet AI Predictor v2", layout="wide")
st.title("⚽ SportyBet AI Predictor v2 – Stats + Poisson + XGBoost")

# ====================== API SETUP ======================
API_KEY = st.sidebar.text_input("API-Football Key", type="password")
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY} if API_KEY else None

# ====================== RAPIDAPI FOR FULL LEAGUES (2,100+ leagues) ======================
try:
    RAPID_API_KEY = st.secrets["rapidapi"]["key"]
except (KeyError, TypeError):
    RAPID_API_KEY = None

if RAPID_API_KEY:
    st.sidebar.success("✅ RapidAPI loaded – full leagues enabled! (lower divisions, cups, etc.)")
else:
    st.sidebar.info("💡 Add [rapidapi] key to secrets.toml for unlimited leagues worldwide")

if not API_KEY:
    st.sidebar.error("⚠️ Enter your API-Football key")
    st.stop()

le_market = LabelEncoder()

# ====================== POISSON & PREDICTION FUNCTIONS ======================
def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k) if lam > 0 else 0

def calculate_poisson_markets(lambda_h, lambda_a):
    p0_h = poisson_pmf(0, lambda_h)
    p0_a = poisson_pmf(0, lambda_a)
    btts_yes = 1 - p0_h - p0_a + p0_h * p0_a
    scores = []
    for i in range(0, 6):
        for j in range(0, 6):
            p = poisson_pmf(i, lambda_h) * poisson_pmf(j, lambda_a)
            scores.append(((i, j), round(p*100, 1)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return round(btts_yes*100, 1), scores[:3]

def get_prediction(fixture_id):
    r = requests.get(f"{BASE_URL}/predictions?fixture={fixture_id}", headers=headers)
    if r.status_code != 200 or not r.json().get("response"):
        return None
    data = r.json()["response"][0]
    teams = data["teams"]
    pred = data.get("predictions", {})
    comp = data.get("comparison", {})
    match_str = f"{teams['home']['name']} vs {teams['away']['name']}"
    
    home_last5_for = safe_float(teams["home"].get("last_5", {}).get("goals", {}).get("for", {}).get("average", 1.3))
    away_last5_against = safe_float(teams["away"].get("last_5", {}).get("goals", {}).get("against", {}).get("average", 1.3))
    lambda_h = (home_last5_for + away_last5_against) / 2

    away_last5_for = safe_float(teams["away"].get("last_5", {}).get("goals", {}).get("for", {}).get("average", 1.3))
    home_last5_against = safe_float(teams["home"].get("last_5", {}).get("goals", {}).get("against", {}).get("average", 1.3))
    lambda_a = (away_last5_for + home_last5_against) / 2
    
    btts_prob, top_scores = calculate_poisson_markets(lambda_h, lambda_a)
    lam_total = lambda_h + lambda_a
    p0 = poisson_pmf(0, lam_total)
    p1 = poisson_pmf(1, lam_total)
    over_15_prob = round((1 - p0 - p1) * 100, 1)
    
    markets = []
    percent = pred.get("percent", {})
    for outcome, p in percent.items():
        selection = "Home Win" if outcome == "home" else "Away Win" if outcome == "away" else "Draw"
        markets.append({"market": "1X2", "selection": selection, "prob": int(safe_float(p, 50)), "advice": pred.get("advice", "")})
    
    if "under_over" in pred:
        markets.append({"market": "Over/Under", "selection": pred["under_over"], "prob": 65, "advice": ""})
    
    markets.append({"market": "BTTS", "selection": "Yes" if btts_prob > 50 else "No", "prob": btts_prob, "advice": ""})
    markets.append({"market": "Over 1.5", "selection": "Over 1.5", "prob": over_15_prob, "advice": ""})
    
    cs = top_scores[0][0]
    markets.append({"market": "Correct Score", "selection": f"{cs[0]}-{cs[1]}", "prob": top_scores[0][1], "advice": ""})
    
    features = {
        "form_diff": safe_float(comp.get("form", {}).get("home", 50)) - safe_float(comp.get("form", {}).get("away", 50)),
        "att_diff": safe_float(comp.get("att", {}).get("home", 50)) - safe_float(comp.get("att", {}).get("away", 50)),
        "def_diff": safe_float(comp.get("def", {}).get("home", 50)) - safe_float(comp.get("def", {}).get("away", 50)),
        "poisson_home": safe_float(comp.get("poisson_distribution", {}).get("home", 50))
    }
    return {"match": match_str, "fixture_id": fixture_id, "markets": markets, "features": features}

def train_xgboost():
    df = load_history()
    if len(df) < 15: return None, None
    df = df.dropna(subset=["correct"]).copy()
    if len(df) < 15: return None, None
    possible_markets = ["1X2", "Over/Under", "BTTS", "Over 1.5", "Correct Score"]
    le_market.fit(possible_markets)
    df["market_enc"] = le_market.transform(df["market"].astype(str))
    X = df[["prob", "form_diff", "att_diff", "def_diff", "poisson_home", "market_enc"]]
    y = df["correct"].astype(int)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model, le_market

def auto_update_history():
    df = load_history()
    if df.empty:
        st.info("No history yet.")
        return
    with st.spinner("Fetching real results..."):
        updated = 0
        client = get_supabase()
        for idx, row in df.iterrows():
            if pd.isna(row.get("actual")) or pd.isna(row.get("correct")):
                r = requests.get(f"{BASE_URL}/fixtures?id={row['fixture_id']}", headers=headers)
                if r.status_code == 200 and r.json().get("response"):
                    fix = r.json()["response"][0]
                    if fix["fixture"]["status"]["short"] == "FT":
                        goals_h = fix["goals"]["home"]
                        goals_a = fix["goals"]["away"]
                        actual_result = "Home Win" if goals_h > goals_a else "Away Win" if goals_a > goals_h else "Draw"
                        
                        if row["market"] == "1X2":
                            correct = (row["selection"] == actual_result)
                        elif row["market"] == "Over/Under":
                            total = goals_h + goals_a
                            sel = str(row["selection"]).strip()
                            if "Over" in sel:
                                line = float(sel.split()[-1])
                                correct = total > line
                            elif "Under" in sel:
                                line = float(sel.split()[-1])
                                correct = total <= line
                            else:
                                correct = False
                        elif row["market"] == "BTTS":
                            btts_yes = (goals_h > 0 and goals_a > 0)
                            correct = (row["selection"] == "Yes" and btts_yes) or (row["selection"] == "No" and not btts_yes)
                        elif row["market"] == "Over 1.5":
                            correct = (goals_h + goals_a) > 1.5
                        elif row["market"] == "Correct Score":
                            correct = (row["selection"] == f"{goals_h}-{goals_a}")
                        else:
                            correct = False
                        
                        update_data = {"actual": f"{goals_h}-{goals_a}", "correct": correct}
                        client.table("prediction_history").update(update_data).eq("id", int(row["id"])).execute()
                        updated += 1
        st.success(f"✅ Updated {updated} past predictions!")

# ====================== MAIN APP ======================
date_to_check = st.sidebar.date_input("Date", value=date(2026, 3, 28))

colA, colB = st.sidebar.columns(2)
generate_btn = colA.button("Generate Predictions")
update_btn = colB.button("🔄 Auto-Update History")

if update_btn and API_KEY:
    auto_update_history()

if generate_btn and API_KEY:
    with st.spinner("Fetching fixtures..."):
        if RAPID_API_KEY:
            rapid_headers = {
                "x-rapidapi-key": RAPID_API_KEY,
                "x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com"
            }
            resp = requests.get(
                f"https://free-api-live-football-data.p.rapidapi.com/fixtures/date/{date_to_check.strftime('%Y-%m-%d')}",
                headers=rapid_headers
            )
            source = "RapidAPI (ALL leagues)"
        else:
            resp = requests.get(f"{BASE_URL}/fixtures?date={date_to_check.strftime('%Y-%m-%d')}", headers=headers)
            source = "API-Football (limited leagues)"
        
        fixtures = resp.json().get("response", []) if resp.status_code == 200 else []
        
        ns_fixtures = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "NS"]
        
        st.info(f"📅 **Date checked:** {date_to_check} | Total fixtures: {len(fixtures)} | Upcoming (NS): {len(ns_fixtures)} | Source: {source}")
        
        if len(ns_fixtures) == 0:
            st.warning("⚠️ No upcoming matches (NS) on this date. Try tomorrow or the next weekend. Free plans show more matches closer to kick-off.")
            st.stop()
        
        predictions = []
        for f in ns_fixtures[:20]:
            pred = get_prediction(f["fixture"]["id"])
            if pred:
                predictions.append(pred)
        
        model, encoder = train_xgboost()
        
        all_bets = []
        for p in predictions:
            for m in p["markets"]:
                base_prob = m["prob"]
                if model is not None:
                    feat = pd.DataFrame([{"prob": base_prob, 
                                          "form_diff": p["features"]["form_diff"],
                                          "att_diff": p["features"]["att_diff"], 
                                          "def_diff": p["features"]["def_diff"],
                                          "poisson_home": p["features"]["poisson_home"],
                                          "market_enc": encoder.transform([m["market"]])[0]}])
                    ml_prob = model.predict_proba(feat)[0][1] * 100
                    final_conf = round(base_prob * 0.6 + ml_prob * 0.4, 1)
                else:
                    final_conf = base_prob
                
                all_bets.append({
                    "match": p["match"], "market": m["market"], "selection": m["selection"],
                    "confidence": final_conf, "fixture_id": p["fixture_id"],
                    "form_diff": p["features"]["form_diff"], "att_diff": p["features"]["att_diff"],
                    "def_diff": p["features"]["def_diff"], "poisson_home": p["features"]["poisson_home"]
                })
        
        if all_bets:
            df = pd.DataFrame(all_bets).sort_values("confidence", ascending=False)
        else:
            df = pd.DataFrame(columns=["match", "market", "selection", "confidence", "fixture_id",
                                       "form_diff", "att_diff", "def_diff", "poisson_home"])
        
        logged_count = 0
        today = str(date.today())
        for bet in all_bets:
            row_dict = {
                "date": today, "fixture_id": bet["fixture_id"], "match": bet["match"],
                "market": bet["market"], "selection": bet["selection"], "prob": bet["confidence"],
                "form_diff": bet["form_diff"], "att_diff": bet["att_diff"],
                "def_diff": bet["def_diff"], "poisson_home": bet["poisson_home"],
                "actual": None, "correct": None
            }
            if save_to_history(row_dict):
                logged_count += 1
        
        st.success(f"✅ Generated {len(predictions)} matches • Auto-logged {logged_count} new predictions (total history: {len(load_history())} rows)")

        # AI Insights + Display
        if model is not None:
            st.sidebar.subheader("🔍 AI Learned Insights")
            importances = pd.Series(model.feature_importances_, 
                                  index=["Base Prob", "Form Diff", "Attack Diff", "Defense Diff", "Poisson Home", "Market Type"])
            for feat, imp in importances.sort_values(ascending=False).items():
                st.sidebar.write(f"• **{feat}** → {imp:.2f}")

        st.header("📋 Ready-to-Paste SportyBet Selections")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("🔒 Safe Acca (≥68%)")
            for _, r in df[df["confidence"] >= 68].iterrows():
                st.markdown(f"**{r['match']}** → **{r['market']}: {r['selection']}** ({r['confidence']}%)")
        with c2:
            st.subheader("⚖️ Medium Acca")
            for _, r in df[(df["confidence"] >= 55) & (df["confidence"] < 68)].iterrows():
                st.markdown(f"**{r['match']}** → **{r['market']}: {r['selection']}** ({r['confidence']}%)")
        with c3:
            st.subheader("🎲 High-Odds + Poisson Specials")
            for _, r in df[df["confidence"] < 55].iterrows():
                st.markdown(f"**{r['match']}** → **{r['market']}: {r['selection']}** ({r['confidence']}%)")

# ====================== HISTORY DISPLAY ======================
st.header("📚 Learning History – Past Predictions + Real Results")
df = load_history()
if not df.empty:
    display_df = df.sort_values("date", ascending=False).copy()
    display_df["Status"] = display_df["correct"].apply(
        lambda x: "✅ Correct" if x is True else "❌ Wrong" if x is False else "⏳ Pending")
    st.dataframe(
        display_df[["date", "match", "market", "selection", "prob", "actual", "Status"]],
        width='stretch', hide_index=True,
        column_config={"prob": st.column_config.NumberColumn("Confidence %")}
    )
else:
    st.info("Generate predictions — they are now saved forever in Supabase!")

st.caption("✅ Permanent Supabase storage • XGBoost retrains on every result • Full-league support via RapidAPI")
