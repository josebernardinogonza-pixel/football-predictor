import os
import requests
import numpy as np
from scipy.stats import poisson
import csv
from datetime import datetime

# ==========================================
# CONFIGURACIÓN REAL-TIME
# ==========================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BANKROLL_MXN = 24360.00  # Tu saldo real actualizado
KELLY_FRACTION = 0.12

# Endpoints de ESPN para el 14 de Abril de 2026
ESPN_URLS = {
    "CHAMPIONS 🏆": "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard",
    "NBA 🏀": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "MLB ⚾": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
}

def get_odds(sport_key):
    """Obtiene cuotas reales de The Odds API"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'decimal'}
    try:
        res = requests.get(url, params=params).json()
        return {f"{m['home_team']} vs {m['away_team']}": m['bookmakers'][0]['markets'][0]['outcomes'] for m in res if m['bookmakers']}
    except: return {}

def analyze_soccer(h_xg, a_xg, cuota):
    prob = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob * cuota) - 1
    f_star = ((cuota - 1) * prob - (1 - prob)) / (cuota - 1) if cuota > 1 else 0
    return prob, ev, max(0, round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2))

def main():
    mensaje = f"🏦 <b>ORDEN DE INVERSIÓN - 14/04/2026</b>\n"
    mensaje += f"💰 Bankroll: ${BANKROLL_MXN:,.2f} MXN\n━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 1. Obtener Cuotas Reales
    cuotas_ucl = get_odds("soccer_uefa_champs_league")
    cuotas_nba = get_odds("basketball_nba")
    
    # 2. Procesar ESPN (Solo partidos que existen hoy)
    for label, url in ESPN_URLS.items():
        try:
            res = requests.get(url).json()
            mensaje += f"<b>{label}</b>\n"
            count = 0
            for event in res.get('events', []):
                name = event['name']
                status = event['status']['type']['state']
                
                if status == 'pre': # Solo partidos por jugar
                    # Buscar cuota en nuestro diccionario de Odds API
                    match_odds = cuotas_ucl.get(name) or cuotas_nba.get(name)
                    if match_odds:
                        h_team = event['competitions'][0]['competitors'][0]['team']['name']
                        cuota = next(o['price'] for o in match_odds if o['name'] == h_team)
                        
                        # Modelo según deporte
                        if "CHAMPIONS" in label:
                            prob, ev, stake = analyze_soccer(2.1, 1.2, cuota) # xG Proyectado
                        else:
                            prob, ev, stake = 0.68, (0.68 * cuota) - 1, 0 # NBA/MLB base
                            if ev > 0.08:
                                stake = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2)

                        if stake > 50:
                            count += 1
                            mensaje += f"🏟 <b>{name}</b>\n"
                            mensaje += f"🎯 PICK: <b>{h_team} (Local)</b>\n"
                            mensaje += f"💵 APOSTAR: <b>${stake} MXN</b>\n"
                            mensaje += f"📈 Prob: {prob:.1%} | Cuota: {cuota:.2f}\n"
                            mensaje += "────────────────────\n"
            if count == 0: mensaje += "<i>Sin valor detectado en ESPN hoy.</i>\n\n"
        except: continue

    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                 json={"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "HTML"})

if __name__ == "__main__":
    main()
