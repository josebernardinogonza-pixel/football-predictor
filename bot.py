"""
EDGE BOT PRO v3.0 - Bot Multi-Liga (8 Deportes) con Modelos Auto-Cargados
"""
import os, requests, pickle, numpy as np, pandas as pd
from datetime import datetime
from scipy.stats import poisson

# Config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BANKROLL_MXN = 24360.00
KELLY_FRACTION = 0.12

ESPN_URLS = {
    "Premier League": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
    "LaLiga": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
    "Serie A": "https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard",
    "Bundesliga": "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard",
    "Ligue 1": "https://site.api.espn.com/apis/site/v2/sports/soccer/fra.1/scoreboard",
    "Liga MX": "https://site.api.espn.com/apis/site/v2/sports/soccer/ligamx/scoreboard",
    "NBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "MLB": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
}

SPORT_MODELS = {'soccer': 'soccer_model.pkl', 'nba': 'nba_model.pkl', 'mlb': 'mlb_model.pkl'}
ODDS_KEYS = {
    'Premier League': 'soccer_epl', 'LaLiga': 'soccer_spain_la_liga', 'Serie A': 'soccer_italy_serie_a',
    'Bundesliga': 'soccer_germany_bundesliga', 'Ligue 1': 'soccer_france_ligue_one', 'Liga MX': 'soccer_mexico_liga_mx',
    'NBA': 'basketball_nba', 'MLB': 'baseball_mlb'
}

# Load models
MODELS = {}
for sport, path in SPORT_MODELS.items():
    try:
        MODELS[sport] = pickle.load(open(f'models/{path}', 'rb'))
    except: print(f"⚠️ No model for {sport}")

def get_odds(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={os.getenv('ODDS_API_KEY')}&regions=us&markets=h2h&oddsFormat=decimal"
    try: return requests.get(url).json()
    except: return []

def prepare_features(event, league):
    sport = 'soccer' if 'soccer' in ESPN_URLS[league] else league.lower()
    # Generic approx features (real: from hist data)
    return pd.DataFrame({
        'scored_rolling_5': [1.4], 'conceded_rolling_5': [1.2], 'scored_rolling_10': [1.3], 'conceded_rolling_10': [1.25],
        'home_advantage': [0.15], 'rest_days': [3], 'rest_diff': [0], 'strength_diff_5': [0.2], 'xg_home': [1.5], 'xg_away': [1.2]
    })

def analyze_match(model, features, cuota):
    probs = model.predict_proba(features)[0]
    prob_home_win = probs[0] if model.sport == 'soccer' else probs[0]
    if prob_home_win > 0.45:  # Criterio APROBADO
        ev = (prob_home_win * cuota - 1)
        if ev > 0.08:
            f = max(0, ((cuota * prob_home_win - 1)/(cuota-1)) * KELLY_FRACTION * BANKROLL_MXN)
            return prob_home_win, ev, round(f, 2), "APROBADO"
    return 0, 0, 0, "NO"

def main():
    mensaje = f"🏦 ORDEN - {datetime.now().strftime('%d/%m %H:%M')} | v3.0 Multi-Liga\n💰 ${BANKROLL_MXN:,.0f}\n"
    all_odds = {k: get_odds(v) for k,v in ODDS_KEYS.items()}
    
    picks = 0
    for league, url in ESPN_URLS.items():
        try:
            events = requests.get(url).json()['events']
            mensaje += f"\n<b>{league}</b>\n"
            sport = 'soccer' if league != 'NBA' and league != 'MLB' else league.lower()
            model = MODELS.get(sport)
            
            for event in events:
                if event['status']['type']['state'] == 'pre':
                    comp = event['competitions'][0]
                    h_team = next(c['team']['name'] for c in comp['competitors'] if c['homeAway']=='home')
                    odds_data = all_odds.get(league, [])
                    cuota_home = next((o['price'] for m in odds_data for bk in m.get('bookmakers',[]) for o in bk[0]['markets'][0]['outcomes'] if o['name']==h_team), None)
                    
                    if cuota_home and model:
                        features = prepare_features(event, league)
                        prob, ev, stake, rec = analyze_match(model, features, cuota_home)
                        if rec == "APROBADO" and stake > 50:
                            picks +=1
                            mensaje += f"🎯 <b>{event['name']}</b> | {h_team} | Stake: ${stake} | Prob: {prob:.1%} | EV: {ev:.1%} | {rec}\n────\n"
                    elif not model: mensaje += "<i>Sin modelo entrenado</i>\n"
            if picks == 0 for league: mensaje += "<i>Sin +EV</i>\n"
        except: continue
    
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "HTML"})
    print("✅ Enviado!")

if __name__ == "__main__": main()
