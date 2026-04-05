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
KELLY_FRACTION = 0.20 # Bajamos a 0.20 para ser más conservadores
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_stats(team):
    """Obtiene xG Ataque/Defensa con manejo de errores"""
    url = f"https://serpapi.com/search.json?q={team}+xG+stats+2024&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        snippets = " ".join([r.get("snippet", "") for r in res.get("organic_results", [])[:3]])
        nums = [float(v) for v in re.findall(r'(\d\.\d+)', snippets) if 0.5 < float(v) < 3.0]
        return (nums[0], nums[1]) if len(nums) >= 2 else (1.4, 1.3)
    except: return 1.4, 1.3

def get_news(team):
    """Analiza bajas de última hora"""
    url = f"https://serpapi.com/search.json?engine=google_news&q={team}+injuries+bajas&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        count = 0
        for art in res.get("news_results", [])[:5]:
            if any(w in art.get("title", "").lower() for w in ["injury", "out", "baja", "lesion"]): count += 1
        return 0.85 if count >= 2 else 1.0
    except: return 1.0

def generate_card(data):
    """Genera la imagen para Telegram"""
    img = Image.new('RGB', (800, 450), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    
    # Cargar Logos
    def load_img(url):
        try:
            r = requests.get(url, timeout=5)
            return Image.open(BytesIO(r.content)).convert("RGBA").resize((140, 140))
        except: return Image.new('RGBA', (140, 140), color=(40, 40, 40))

    img.paste(load_img(data['h_logo']), (70, 100), load_img(data['h_logo']))
    img.paste(load_img(data['a_logo']), (590, 100), load_img(data['a_logo']))

    # Textos (Sin fuentes externas para evitar errores en GitHub Actions)
    draw.text((400, 40), "TOP PICK LIGA MX", fill="gold", anchor="mm")
    draw.text((140, 260), data['home'][:12], fill="white", anchor="mm")
    draw.text((660, 260), data['away'][:12], fill="white", anchor="mm")
    draw.text((400, 160), f"{data['prob']:.1%}", fill="#00FF00", anchor="mm")
    draw.text((400, 190), "Prob. Victoria", fill="gray", anchor="mm")
    
    # Cuadro de Apuesta
    draw.rectangle([150, 310, 650, 420], outline="gold", width=3)
    draw.text((400, 345), f"APUESTA: ${data['apuesta']} MXN", fill="gold", anchor="mm")
    draw.text((400, 385), f"Cuota: {data['cuota']} | EV: {data['ev']:+.2f}", fill="white", anchor="mm")
    
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Buscar partidos
    url = f"https://serpapi.com/search.json?q=proximos+partidos+Liga+MX&api_key={SERPAPI_KEY}"
    matches = requests.get(url).json().get("sports_results", {}).get("games", [])[:3] # Solo top 3 para ahorrar créditos
    
    results = []
    for m in matches:
        h, a = m["teams"][0]["name"], m["teams"][1]["name"]
        h_logo, a_logo = m["teams"][0].get("thumbnail"), m["teams"][1].get("thumbnail")
        
        # Proyección
        h_att, h_def = get_stats(h)
        a_att, a_def = get_stats(a)
        h_xg = (h_att * a_def) * 1.10 * get_news(h)
        a_xg = (a_att * h_def) * 0.90 * get_news(a)
        
        prob = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
        cuota = 2.05 # Cuota base (puedes automatizar esto también)
        ev = (prob * cuota) - 1
        
        apuesta = 0
        if ev > 0:
            f = ((cuota-1)*prob - (1-prob))/(cuota-1)
            apuesta = round(f * KELLY_FRACTION * BANKROLL_MXN, 2)
        
        res_data = {"home": h, "away": a, "h_logo": h_logo, "a_logo": a_logo, "prob": prob, "cuota": cuota, "ev": ev, "apuesta": max(0, apuesta)}
        
        # Guardar en CSV
        with open("predictions_history.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), f"{h} vs {a}", f"{prob:.1%}", ev, apuesta])
        
        results.append(res_data)

    # Generar imagen del mejor
    if results:
        best = max(results, key=lambda x: x['ev'])
        generate_card(best)
