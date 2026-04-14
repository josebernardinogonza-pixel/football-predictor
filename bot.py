import os
import requests
import numpy as np
from scipy.stats import poisson
import csv
import json
from datetime import datetime, timedelta

# ==========================================
# CONFIGURACIÓN MAESTRA (DATOS REALES)
# ==========================================
BANKROLL_MXN = 21830.00  
KELLY_FRACTION = 0.12
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

os.makedirs("data", exist_ok=True)
HISTORIAL_PATH = "data/predictions_history.csv"
CONFIG_PATH = "data/config_ia.json"

# Endpoints Reales de ESPN
ESPN_URLS = {
    "FÚTBOL ⚽": "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard",
    "NBA 🏀": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "MLB ⚾": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
}

def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f: return json.load(f)
    return {"MIN_EV": 0.07, "ADJUST": 1.05}

def auditar_con_espn():
    """Consulta ESPN para ver qué pasó ayer y aprender"""
    ayer = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    resultados = {}
    for sport, url in ESPN_URLS.items():
        try:
            res = requests.get(f"{url}?dates={ayer}").json()
            for event in res.get('events', []):
                if event['status']['type']['state'] == 'post':
                    teams = event['competitions'][0]['competitors']
                    h = next(t for t in teams if t['homeAway'] == 'home')
                    a = next(t for t in teams if t['homeAway'] == 'away')
                    winner = "LOCAL" if int(h['score']) > int(a['score']) else "VISITANTE"
                    resultados[event['name'].upper()] = winner
        except: continue
    return resultados

def get_real_odds(sport_key):
    """Obtiene cuotas reales de The Odds API"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'decimal'}
    try:
        res = requests.get(url, params=params).json()
        return res if isinstance(res, list) else []
    except: return []

def main():
    config = get_config()
    resultados_ayer = auditar_con_espn()
    
    mensaje = f"🏦 <b>SISTEMA EL MAESTRO - {datetime.now().strftime('%d/%m/%Y')}</b>\n"
    mensaje += f"💰 Bankroll: ${BANKROLL_MXN:,.2f} MXN\n"
    mensaje += "━━━━━━━━━━━━━━━━━━━━\n\n"

    # Escaneo de Ligas (Solo si hay partidos hoy en la API)
    sports_map = {
        "FÚTBOL ⚽": "soccer_mexico_ligamx",
        "NBA 🏀": "basketball_nba",
        "MLB ⚾": "baseball_mlb"
    }

    hay_picks = False
    for label, key in sports_map.items():
        odds_data = get_real_odds(key)
        if not odds_data: continue

        mensaje += f"<b>{label}</b>\n"
        count = 0
        for m in odds_data[:5]:
            home, away = m['home_team'], m['away_team']
            try:
                bookie = next((b for b in m['bookmakers'] if b['key'] in ['stake', 'novibet']), m['bookmakers'][0])
                cuota = bookie['markets'][0]['outcomes'][0]['price']
                
                # Modelo Poisson para Fútbol, Probabilidad base para otros
                if "soccer" in key:
                    prob = np.sum(np.tril(np.outer(poisson.pmf(range(10), 1.7 * config['ADJUST']), poisson.pmf(range(10), 1.2)), -1))
                else:
                    prob = 0.65 # Ajuste conservador para NBA/MLB si no hay stats profundas
                
                ev = (prob * cuota) - 1
                if ev > config['MIN_EV']:
                    hay_picks = True
                    count += 1
                    f_star = ((cuota - 1) * prob - (1 - prob)) / (cuota - 1)
                    stake = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
                    
                    mensaje += f"🏟 <b>{home} vs {away}</b>\n"
                    mensaje += f"🎯 PICK: <b>{home} (Local)</b>\n"
                    mensaje += f"💵 APOSTAR: <b>${max(0, stake)} MXN</b>\n"
                    mensaje += f"📈 Prob: {prob:.1%} | Cuota: {cuota:.2f}\n"
                    mensaje += "────────────────────\n"
                    
                    # Guardar en historial
                    with open(HISTORIAL_PATH, "a", newline="") as f:
                        csv.writer(f).writerow([datetime.now().date(), label, f"{home} vs {away}", home, prob, cuota, stake, "PENDIENTE"])
            except: continue
        if count == 0: mensaje += "<i>Sin valor detectado hoy.</i>\n\n"

    if not hay_picks:
        mensaje += "⚠️ No se detectaron oportunidades de inversión real para hoy."

    # Enviar a Telegram
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                 json={"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "HTML"})

if __name__ == "__main__":
    main()
