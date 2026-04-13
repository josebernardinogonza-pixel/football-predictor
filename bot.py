import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
BANKROLL_MXN = 21830.00  
KELLY_FRACTION = 0.15
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("ALL_SPORTS_API_KEY") # Tu llave de AllSportsApi.com

os.makedirs("data", exist_ok=True)
HISTORIAL_PATH = "data/predictions_history.csv"
CONFIG_IA_PATH = "data/config_ia.json"

def get_today_matches():
    """Obtiene partidos reales de hoy desde la API"""
    hoy_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://apiv2.allsportsapi.com/football/?met=Fixtures&APIkey={API_KEY}&from={hoy_str}&to={hoy_str}"
    try:
        res = requests.get(url).json()
        if "result" in res:
            # Filtramos solo ligas importantes para no saturar
            top_leagues = ["Liga MX", "Premier League", "LaLiga", "Serie A", "Bundesliga"]
            matches = []
            for m in res["result"]:
                if m["league_name"] in top_leagues:
                    matches.append({
                        "match": f"{m['event_home_team']} vs {m['event_away_team']}",
                        "h": 1.8, "a": 1.2, # xG base (la IA lo ajustará)
                        "cuota": 2.00,
                        "liga": m["league_name"]
                    })
            return matches[:6] # Top 6 del día
        return []
    except: return []

def get_config():
    default = {"MIN_EV": 0.07, "ADJUST": 1.10}
    if os.path.exists(CONFIG_IA_PATH):
        with open(CONFIG_IA_PATH, "r") as f:
            return {**default, **json.load(f)}
    return default

def predict_match(h_xg, a_xg, cuota, config):
    adj = config.get("ADJUST", 1.10)
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg*adj), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > config.get("MIN_EV", 0.07) else 0
    return prob_l, ev, max(0, apuesta)

def generate_card(picks):
    img = Image.new('RGB', (1000, 1000), color=(10, 10, 15))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 100], fill="#D4AF37")
    draw.text((500, 50), f"INVERSIÓN DIARIA - {datetime.now().strftime('%d/%m/%Y')}", fill="black", anchor="mm")
    y = 150
    for p in picks:
        draw.rectangle([50, y, 950, y+120], outline="#D4AF37", width=2)
        draw.text((80, y+30), f"{p['liga']}: {p['match']}", fill="white")
        draw.text((80, y+70), f"PROB: {p['prob']:.1%} | STAKE: ${p['apuesta']} MXN", fill="#00FF00")
        y += 150
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Inicializar archivos
    with open("audit_report.txt", "w") as f: f.write("Analizando jornada...")
    
    # 2. Obtener partidos REALES de hoy
    hoy_matches = get_today_matches()
    config = get_config()
    
    final_picks = []
    if hoy_matches:
        for p in hoy_matches:
            prob, ev, stake = predict_match(p['h'], p['a'], p['cuota'], config)
            if stake > 0:
                res = {**p, "prob": prob, "apuesta": stake}
                final_picks.append(res)
                with open(HISTORIAL_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now().date(), p['match'], p['liga'], "Local", prob, 2.0, stake, "PENDIENTE"])
        
        if final_picks:
            generate_card(final_picks)
            with open("audit_report.txt", "w") as f:
                f.write(f"<b>✅ JORNADA DETECTADA</b>\nSe encontraron {len(final_picks)} oportunidades de valor.")
    else:
        with open("audit_report.txt", "w") as f:
            f.write("<b>⏳ SIN ACTIVIDAD</b>\nNo hay partidos de ligas TOP hoy o la API está en mantenimiento.")
