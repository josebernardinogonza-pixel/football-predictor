"""
EDGE BOT PRO v4.0 - Sistema de Predicción ML (8 Ligas)
Carga modelos entrenados, obtiene cuotas en tiempo real, aplica Kelly Criterion y envía alertas a Telegram.
Versión con calibración de probabilidades y registro de historial.
"""
import os
import requests
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Configuración (desde variables de entorno)
# -------------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BANKROLL_MXN = 24360.00
KELLY_FRACTION = 0.12          # Fracción de Kelly conservadora
MIN_STAKE = 50                 # Stake mínimo en MXN
EDGE_THRESHOLD = 0.08          # Valor esperado mínimo

# -------------------------------------------------------------------
# URLs de ESPN y mapeo a IDs de Odds API
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Carga de modelos
# -------------------------------------------------------------------
SPORT_MODELS = {
    'soccer': 'models/soccer_model.pkl',
    'nba': 'models/nba_model.pkl',
    'mlb': 'models/mlb_model.pkl'
}

MODELS = {}
for sport_key, model_path in SPORT_MODELS.items():
    try:
        MODELS[sport_key] = pickle.load(open(model_path, 'rb'))
        print(f"✅ {sport_key} modelo cargado")
    except Exception as e:
        print(f"⚠️ {sport_key} modelo no encontrado: {e}")
        print("   Ejecuta: python training.py")

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------
def get_odds(sport_key):
    """Obtiene cuotas de The Odds API para un deporte dado."""
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

def get_home_team_odds(event, odds_data):
    """Obtiene la cuota del equipo local desde los datos de Odds API."""
    try:
        comp = event['competitions'][0]
        home_comp = next(c for c in comp['competitors'] if c['homeAway'] == 'home')
        h_team = home_comp['team']['name']
        for game in odds_data:
            # Comparar nombres normalizados
            if h_team.lower() in game.get('home_team', '').lower() or h_team.lower() in game.get('away_team', '').lower():
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name').lower() == h_team.lower():
                                    return outcome.get('price'), h_team
        return None, h_team
    except Exception as e:
        print(f"Error extrayendo odds: {e}")
        return None, None

def prepare_features(event, league):
    """
    Construye un DataFrame con las 15 features que espera el modelo.
    En una implementación real, estos valores se obtendrían de un histórico de partidos.
    Para este ejemplo se usan valores sintéticos realistas.
    """
    # Determinar deporte
    soccer_keywords = ['Premier', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Liga MX']
    if any(kw in league for kw in soccer_keywords):
        sport = 'soccer'
        base_scored = 1.4
        base_conceded = 1.2
        home_adv = 0.15 if 'Liga MX' in league else 0.10
    elif league == 'NBA':
        sport = 'nba'
        base_scored = 110
        base_conceded = 108
        home_adv = 5.0
    elif league == 'MLB':
        sport = 'mlb'
        base_scored = 4.5
        base_conceded = 4.3
        home_adv = 0.3
    else:
        sport = 'soccer'
        base_scored = 1.4
        base_conceded = 1.2
        home_adv = 0.10

    return pd.DataFrame({
        'scored_rolling_5': [base_scored],
        'conceded_rolling_5': [base_conceded],
        'scored_rolling_10': [base_scored + 0.05],
        'conceded_rolling_10': [base_conceded - 0.05],
        'opponent_scored_rolling_5': [base_conceded + 0.1],
        'opponent_conceded_rolling_5': [base_scored - 0.1],
        'opponent_scored_rolling_10': [base_conceded + 0.05],
        'opponent_conceded_rolling_10': [base_scored - 0.05],
        'home_advantage': [home_adv],
        'rest_days': [3.0],
        'rest_diff': [0.5],
        'rest_advantage': [0.1],
        'strength_diff_5': [0.3],
        'strength_diff_10': [0.2],
        'form_rating': [0.55]             # Rating de forma estimado (0-1)
    })

def analyze_match(model, features, cuota):
    """
    Usa el modelo calibrado para predecir probabilidad de victoria local,
    calcula el Valor Esperado y el stake según Kelly.
    """
    try:
        if not hasattr(model, 'predict_proba'):
            return 0, 0, 0, "NO"
        proba = model.predict_proba(features)[0]
        if len(proba) != 2:
            return 0, 0, 0, "NO"
        prob_home_win = proba[1]  # Clase 1 = local gana

        # Cálculo de EV y Kelly
        ev = prob_home_win * cuota - 1
        if prob_home_win > 0.45 and ev > EDGE_THRESHOLD:
            # Fórmula de Kelly: f* = (p * b - 1) / (b - 1) donde b = cuota - 1
            b = cuota - 1
            if b <= 0:
                return 0, 0, 0, "NO"
            f_star = (prob_home_win * cuota - 1) / b
            stake = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
            if stake >= MIN_STAKE:
                return prob_home_win, ev, stake, "APROBADO"
        return 0, 0, 0, "NO"
    except Exception as e:
        print(f"Error en predict: {e}")
        return 0, 0, 0, "NO"

def save_to_history(pick):
    """Guarda el pick en un CSV de historial para backtesting."""
    history_file = "historial.csv"
    header = ["fecha", "liga", "partido", "equipo_local", "cuota", "prob_modelo", "ev", "stake"]
    data = [datetime.now().strftime("%Y-%m-%d %H:%M"), pick.get('liga'), pick.get('partido'),
            pick.get('equipo'), pick.get('cuota'), pick.get('prob'), pick.get('ev'), pick.get('stake')]
    try:
        df = pd.read_csv(history_file)
        pd.DataFrame([data], columns=header).to_csv(history_file, mode='a', header=False, index=False)
    except FileNotFoundError:
        pd.DataFrame([data], columns=header).to_csv(history_file, index=False)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("🔄 Iniciando análisis con modelo ML calibrado...")
    mensaje = f"🏦 <b>ORDEN DE INVERSIÓN v4.0 - ML</b> - {datetime.now().strftime('%d/%m %H:%M')}\n"
    mensaje += f"💰 Bankroll: ${BANKROLL_MXN:,.2f} MXN\n"
    mensaje += "━━━━━━━━━━━━━━━━━━━━━━\n"
    total_picks = 0

    # Obtener cuotas de todas las ligas
    all_odds = {league: get_odds(key) for league, key in ODDS_KEYS.items()}

    for league, url in ESPN_URLS.items():
        try:
            resp = requests.get(url, timeout=10).json()
            events = resp.get('events', [])
            # Determinar modelo deportivo
            soccer_keywords = ['Premier', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Liga MX']
            sport_key = 'soccer' if any(kw in league for kw in soccer_keywords) else league.lower()
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
                        pick = {
                            'liga': league,
                            'partido': event.get('name', 'N/A'),
                            'equipo': h_team,
                            'cuota': cuota_home,
                            'prob': round(prob, 3),
                            'ev': round(ev, 3),
                            'stake': stake
                        }
                        save_to_history(pick)
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

    # Enviar a Telegram
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
