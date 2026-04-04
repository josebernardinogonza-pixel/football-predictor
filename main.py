import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.25
HOME_ADVANTAGE = 1.10
AWAY_PENALTY = 0.90
NEWS_PENALTY = 0.85
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_stats_from_serp(team_name):
    """Obtiene xG de ataque y defensa"""
    query = f"{team_name} xG stats 2024 per game"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        text = str(res.get("organic_results", ""))
        nums = [float(v) for v in re.findall(r'(\d\.\d+)', text) if 0.4 < float(v) < 3.5]
        return (nums[0], nums[1]) if len(nums) >= 2 else (1.4, 1.4)
    except: return 1.4, 1.4

def get_news_impact(team_name):
    """Analiza noticias de lesiones"""
    query = f"{team_name} team news injuries"
    url = f"https://serpapi.com/search.json?engine=google_news&q={query}&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        keywords = ["injury", "out", "suspended", "baja", "lesion"]
        count = 0
        for art in res.get("news_results", [])[:5]:
            text = (art.get("title", "") + art.get("snippet", "")).lower()
            if any(w in text for w in keywords): count += 1
        return NEWS_PENALTY if count >= 2 else 1.0
    except: return 1.0

def get_market_odds(home, away):
    """Busca cuotas reales"""
    query = f"odds {home} vs {away}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        text = str(res.get("organic_results", ""))
        odds = re.findall(r'(\d\.\d{2})', text)
        return [float(o) for o in odds[:3]] if len(odds) >= 3 else [2.10, 3.20, 3.40]
    except: return [2.00, 3.00, 3.50]

def save_to_csv(data):
    """Guarda resultados en historial"""
    file = "predictions_history.csv"
    exists = os.path.isfile(file)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(file, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Fecha", "Partido", "Prob_L", "Cuota", "EV", "Apuesta_MXN"])
        w.writerow([fecha, f"{data['home']} vs {data['away']}", f"{data['prob_l']:.1%}", data['cuota'], f"{data['ev']:+.2f}", f"${data['apuesta']} MXN"])

def generate_card(data):
    """Crea la imagen para Telegram"""
    img = Image.new('RGB', (800, 450), color=(15, 15, 15))
    draw = ImageDraw.Draw(img)
    
    # Descargar logos
    def load_logo(url):
        try:
            r = requests.get(url)
            return Image.open(BytesIO(r.content)).convert("RGBA").resize((130, 130))
        except: return Image.new('RGBA', (130, 130), color=(40, 40, 40))

    logo_h = load_logo(data['home_logo'])
    logo_a = load_logo(data['away_logo'])
    img.paste(logo_h, (80, 100), logo_h)
    img.paste(logo_a, (590, 100), logo_a)

    # Textos
    draw.text((400, 40), "PRONÓSTICO LIGA MX", fill="gold", anchor="mm")
    draw.text((145, 250), data['home'][:12], fill="white", anchor="mm")
    draw.text((655, 250), data['away'][:12], fill="white", anchor="mm")
    
    # Probabilidades
    draw.text((400, 150), f"{data['prob_l']:.1%}", fill="#00FF00", anchor="mm")
    draw.text((400, 180), "Prob. Victoria Local", fill="gray", anchor="mm")
    
    # Cuadro Apuesta
    draw.rectangle([150, 300, 650, 410], outline="gold", width=2)
    draw.text((400, 335), f"APUESTA: ${data['apuesta']} MXN", fill="gold", anchor="mm")
    draw.text((400, 375), f"Cuota: {data['cuota']} | EV: {data['ev']:+.2f}", fill="white", anchor="mm")
    
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Obtener partidos
    url = f"https://serpapi.com/search.json?q=proximos+partidos+Liga+MX&api_key={SERPAPI_KEY}"
    res = requests.get(url).json()
    matches = res.get("sports_results", {}).get("games", [])[:5]
    
    results = []
    for m in matches:
        h_name = m["teams"][0]["name"]
        a_name = m["teams"][1]["name"]
        h_logo = m["teams"][0].get("thumbnail")
        a_logo = m["teams"][1].get("thumbnail")
        
        # Lógica Poisson
        h_att, h_def = get_stats_from_serp(h_name)
        a_att, a_def = get_stats_from_serp(a_name)
        h_xg = (h_att * a_def) * HOME_ADVANTAGE * get_news_impact(h_name)
        a_xg = (a_att * h_def) * AWAY_PENALTY * get_news_impact(a_name)
        
        prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
        odds = get_market_odds(h_name, a_name)
        ev = (prob_l * odds[0]) - 1
        
        apuesta = 0
        if ev > 0:
            f = ((odds[0]-1)*prob_l - (1-prob_l))/(odds[0]-1)
            apuesta = round(f * KELLY_FRACTION * BANKROLL_MXN, 2)
        
        data = {"home": h_name, "away": a_name, "home_logo": h_logo, "away_logo": a_logo, "prob_l": prob_l, "cuota": odds[0], "ev": ev, "apuesta": max(0, apuesta)}
        save_to_csv(data)
        results.append(data)

    # Generar imagen del mejor pick
    if results:
        best = max(results, key=lambda x: x['ev'])
        generate_card(best)
