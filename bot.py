"""
EDGE BOT PRO v3.3 - Bot Multi-Liga (8 Deportes)
Carga modelos entrenados | ESPN + Odds API | Telegram Alerts
"""
import os
import requests
import pickle
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
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

SPORT_MODELS = {
    'soccer': 'models/soccer_model.pkl',
    'nba': 'models/nba_model.pkl',
    'mlb': 'models/mlb_model.pkl'
}

ODDS_KEYS = {
    'Premier League': 'soccer_epl',
    'LaLiga': 'soccer_spain_la_liga',
    'Serie A': 'soccer_italy_serie_a',
    'Bundesliga': 'soccer_germany_bundesliga',
    'Ligue 1': 'soccer_france_ligue_one',
    'Liga MX': 'soccer_mexico_liga_mx',
    'NBA': 'basketball_nba',
    'MLB': 'baseball_mlb'
}

# Carga modelos
MODELS = {}
for sport_key, model_path in SPORT_MODELS.items():
    try:
        MODELS[sport_key] = pickle.load(open(model_path, 'rb'))
        print(f"✅ {sport_key} modelo cargado")
    except Exception as e:
        print(f"⚠️ {sport_key} modelo no encontrado: {e}")
        print("   Ejecuta: python training.py")

def get_odds(sport_key):
    if not ODDS_API_KEY:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        return resp if isinstance(resp, list) else []
    except Exception as e:
        print(f"Error obteniendo odds: {e}")
        return []

def prepare_features(event, league):
    soccer_leagues = ['League', 'Liga', 'Serie', 'Bundesliga', 'Ligue', 'MX']
    sport = 'soccer' if any(x in league for x in soccer_leagues) else league.lower()
    sport_map = {'soccer': 1.4, 'nba': 110, 'mlb': 4.6}
    base_scored = sport_map.get(sport, 1.4)
    return pd.DataFrame({
        'scored_rolling_5': [base_scored],
        'conceded_rolling_5': [base_scored - 0.2],
        'scored_rolling_10': [base_scored],
        'conceded_rolling_10': [base_scored - 0.2],
        'opponent_scored_rolling_5': [base_scored + 0.1],
        'opponent_conceded_rolling_5': [base_scored + 0.15],
        'opponent_scored_rolling_10': [base_scored + 0.05],
        'opponent_conceded_rolling_10': [base_scored + 0.1],
        'home_advantage': [0.15 if 'MX' in league else 0.10],
        'rest_days': [3.0],
        'rest_diff': [0.5],
        'rest_advantage': [0.1],
        'strength_diff_5': [0.3],
        'strength_diff_10': [0.2]
    })

def analyze_match(model, features, cuota):
    try:
        # Verificar que el modelo tiene predict_proba y que las features coinciden
        if not hasattr(model, 'predict_proba'):
            print("⚠️ El modelo no tiene predict_proba")
            return 0, 0, 0, "NO"
        # Asegurar que el número de features coincida (ignorar nombres)
        # Si las features no coinciden en número, el modelo fallará
        probs = model.predict_proba(features)[0]
        # El modelo debe tener al menos 2 clases. Clase 1 es "local gana"
        if len(probs) < 2:
            return 0, 0, 0, "NO"
        prob_home_win = probs[1]  # Clase 1 = local gana (según training.py)
        if prob_home_win > 0.45:
            ev = (prob_home_win * cuota - 1)
            if ev > 0.08:
                f_star = max(0, (cuota * prob_home_win - 1) / (cuota - 1))
                stake = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
                if stake > 50:
                    return prob_home_win, ev, stake, "APROBADO"
        return 0, 0, 0, "NO"
    except Exception as e:
        print(f"Error en predict: {e}")
        return 0, 0, 0, "NO"

def get_home_team_odds(event, odds_data):
    try:
        comp = event['competitions'][0]
        home_comp = next(c for c in comp['competitors'] if c['homeAway'] == 'home')
        h_team = home_comp['team']['name']
        for game in odds_data:
            if h_team in game.get('home_team', '') or h_team in game.get('away_team', ''):
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == h_team:
                                    return outcome.get('price'), h_team
        return None, h_team
    except Exception as e:
        print(f"Error extrayendo odds: {e}")
        return None, None

def main():
    print("🔄 Iniciando análisis...")
    mensaje = f"🏦 <b>ORDEN DE INVERSIÓN v3.3</b> - {datetime.now().strftime('%d/%m %H:%M')}\n"
    mensaje += f"💰 Bankroll: ${BANKROLL_MXN:,.2f} MXN\n"
    mensaje += "━━━━━━━━━━━━━━━━━━━━━━\n"
    total_picks = 0
    all_odds = {league: get_odds(key) for league, key in ODDS_KEYS.items()}
    for league, url in ESPN_URLS.items():
        try:
            resp = requests.get(url, timeout=10).json()
            events = resp.get('events', [])
            soccer_leagues = ['League', 'Liga', 'Serie', 'Bundesliga', 'Ligue', 'MX']
            sport_key = 'soccer' if any(x in league for x in soccer_leagues) else league.lower()
            model = MODELS.get(sport_key)
            mensaje += f"\n<b>{league}</b>\n"
            league_picks = 0
            for event in events:
                if event.get('status', {}).get('type', {}).get('state') != 'pre':
                    continue
                cuota_home, h_team = get_home_team_odds(event, all_odds.get(league, []))
                if cuota_home and model and cuota_home > 1.01:
                    features = prepare_features(event, league)
                    prob, ev, stake, rec = analyze_match(model, features, cuota_home)
                    if rec == "APROBADO":
                        league_picks += 1
                        total_picks += 1
                        mensaje += f"⚽ <b>{event.get('name', 'N/A')}</b>\n"
                        mensaje += f"🎯 {h_team} | Stake: <b>${stake:,.0f}</b>\n"
                        mensaje += f"📊 Prob: {prob:.1%} | Cuota: {cuota_home:.2f} | EV: {ev:.1%}\n"
                        mensaje += "────────────────────\n"
            if league_picks == 0:
                mensaje += "<i>Sin +EV detectado hoy</i>\n"
            else:
                mensaje += f"📈 {league_picks} picks encontrados\n"
        except Exception as e:
            mensaje += f"<i>Error {league}: {str(e)[:50]}...</i>\n"
            print(f"Error procesando {league}: {e}")
    mensaje += f"\n🔄 Total: {total_picks} picks | Próxima ejecución: 1h 🚀"
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "HTML"},
                timeout=10
            )
            if resp.status_code == 200:
                print("✅ Mensaje enviado a Telegram")
            else:
                print(f"⚠️ Error Telegram: {resp.status_code}")
        except Exception as e:
            print(f"Error enviando Telegram: {e}")
    else:
        print("⚠️ Variables Telegram no configuradas")
        print(mensaje)
    print(f"✅ Análisis completado: {total_picks} picks")

if __name__ == "__main__":
    main()
