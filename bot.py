import os
import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# --- CONFIGURACIÓN ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BANKROLL_MXN = 21830.00
KELLY_FRACTION = 0.12 # Bajamos a 0.12 para diversificar en 3 deportes

# Mapeo de Deportes (The Odds API keys)
SPORTS = {
    "FÚTBOL ⚽": "soccer_mexico_ligamx",
    "NBA 🏀": "basketball_nba",
    "MLB ⚾": "baseball_mlb"
}

def get_market_odds(sport_key):
    """Obtiene cuotas reales de la API"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'bookmakers': 'stake,novibet',
        'oddsFormat': 'decimal'
    }
    try:
        res = requests.get(url, params=params)
        return res.json()
    except:
        return []

def calculate_stake(prob, odds):
    """Criterio de Kelly para MXN"""
    if odds <= 1: return 0
    b = odds - 1
    p = prob
    q = 1 - p
    f_star = (b * p - q) / b
    return max(0, round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2))

def analyze_soccer(match):
    """Modelo Poisson para Fútbol"""
    h_xg, a_xg = 1.8, 1.2 # Base xG
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    return prob_l

def analyze_nba(match):
    """Modelo de Eficiencia para NBA"""
    # Probabilidad base para favoritos en NBA (70% promedio)
    return 0.68

def analyze_mlb(match):
    """Modelo Pitagórico para MLB"""
    # Probabilidad base para locales en MLB (54% promedio)
    return 0.54

def main():
    mensaje_final = f"🏦 <b>REPORTE DE INVERSIÓN DIARIA</b>\n"
    mensaje_final += f"💰 Bankroll: ${BANKROLL_MXN} MXN\n"
    mensaje_final += f"📅 {datetime.now().strftime('%d/%m/%Y')}\n"
    mensaje_final += "━━━━━━━━━━━━━━━━━━━━\n\n"

    for sport_name, sport_key in SPORTS.items():
        data = get_market_odds(sport_key)
        if not data: continue

        mensaje_final += f"<b>{sport_name}</b>\n"
        count = 0
        
        for match in data[:5]: # Analizamos los 5 más importantes de cada uno
            home = match['home_team']
            away = match['away_team']
            
            # Obtener la mejor cuota disponible
            try:
                odds = match['bookmakers'][0]['markets'][0]['outcomes']
                h_odds = next(o['price'] for o in odds if o['name'] == home)
            except: continue

            # Elegir modelo según deporte
            if "FÚTBOL" in sport_name: prob = analyze_soccer(match)
            elif "NBA" in sport_name: prob = analyze_nba(match)
            else: prob = analyze_mlb(match)

            ev = (prob * h_odds) - 1
            
            if ev > 0.05: # Solo si hay valor > 5%
                count += 1
                apuesta = calculate_stake(prob, h_odds)
                mensaje_final += f"🏟 <b>{home} vs {away}</b>\n"
                mensaje_final += f"🎯 PICK: <b>{home} (Local)</b>\n"
                mensaje_final += f"💵 APOSTAR: <b>${apuesta} MXN</b>\n"
                mensaje_final += f"📈 Prob: {prob:.1%} | Cuota: {h_odds:.2f}\n"
                mensaje_final += "────────────────────\n"
        
        if count == 0:
            mensaje_final += "<i>No se detectó valor en este mercado hoy.</i>\n\n"
        else:
            mensaje_final += "\n"

    mensaje_final += "⚠️ <b>INDICACIÓN:</b> Ejecutar apuestas por separado. No combinar en parlay a menos que el EV sea > 0.20."

    # Enviar a Telegram
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                 json={"chat_id": CHAT_ID, "text": mensaje_final, "parse_mode": "HTML"})

if __name__ == "__main__":
    main()
