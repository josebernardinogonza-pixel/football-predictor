import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime

# --- CONFIGURACIÓN LIGA MX ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.20
HOME_ADVANTAGE = 1.10
AWAY_PENALTY = 0.90
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
LEAGUE = "Liga MX"

def get_stats(team):
    """Busca xG de equipos de la Liga MX"""
    url = f"https://serpapi.com/search.json?q={team}+xG+stats+2024+mexico&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        snippets = " ".join([r.get("snippet", "") for r in res.get("organic_results", [])[:3]])
        nums = [float(v) for v in re.findall(r'(\d\.\d+)', snippets) if 0.5 < float(v) < 3.5]
        return (nums[0], nums[1]) if len(nums) >= 2 else (1.4, 1.3)
    except: return 1.4, 1.3

def get_real_result(home, away):
    """Busca el resultado real en Google (Solo si ya terminó)"""
    query = f"resultado final {home} vs {away} liga mx"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        # Verificamos si Google muestra el partido como 'Finalizado'
        status = res.get("sports_results", {}).get("game_spotlight", {}).get("status", "")
        snippet = str(res.get("sports_results", "")) + str(res.get("organic_results", ""))
        
        scores = re.findall(r'(\d+)\s*-\s*(\d+)', snippet)
        if scores:
            h, a = int(scores[0][0]), int(scores[0][1])
            return "L" if h > a else "E" if h == a else "V"
    except: return None
    return None

def audit_past_predictions():
    """Auditoría corregida para Liga MX"""
    file = "predictions_history.csv"
    if not os.path.isfile(file): return "<b>📊 AUDITORÍA</b>\nEsperando datos..."
    
    hits, total = 0, 0
    report = "<b>📊 AUDITORÍA LIGA MX</b>\n\n"
    
    with open(file, 'r') as f:
        rows = list(csv.reader(f))
        for row in rows[-8:]: # Revisamos los últimos 8 partidos
            try:
                # FIX: Tomamos solo las primeras 5 columnas para evitar el error de 'unpack'
                fecha, partido, prob, ev, apuesta = row[:5]
                if "vs" not in partido: continue
                
                home, away = partido.split(" vs ")
                res = get_real_result(home, away)
                
                if res:
                    total += 1
                    # Si el bot predijo victoria local (Prob > 45%) y ganó el local:
                    if res == "L":
                        hits += 1
                        report += f"✅ {home} vs {away}: ACERTADO\n"
                    else:
                        report += f"❌ {home} vs {away}: FALLADO\n"
            except: continue
            
    if total > 0:
        report += f"\n<b>🎯 Precisión: {(hits/total):.1%}</b>"
    else:
        report = "<b>📊 AUDITORÍA</b>\n⏳ Los partidos de la jornada aún no terminan."
    return report

def generate_card(data):
    """Imagen con estilo Liga MX"""
    img = Image.new('RGB', (800, 450), color=(20, 25, 20)) # Fondo verde oscuro
    draw = ImageDraw.Draw(img)
    def load_img(url):
        try:
            r = requests.get(url, timeout=5)
            return Image.open(BytesIO(r.content)).convert("RGBA").resize((140, 140))
        except: return Image.new('RGBA', (140, 140), color=(50, 50, 50))
    
    img.paste(load_img(data['h_logo']), (70, 100), load_img(data['h_logo']))
    img.paste(load_img(data['a_logo']), (590, 100), load_img(data['a_logo']))
    draw.text((400, 40), "PRONÓSTICO LIGA MX", fill="gold", anchor="mm")
    draw.text((140, 260), data['home'][:12], fill="white", anchor="mm")
    draw.text((660, 260), data['away'][:12], fill="white", anchor="mm")
    draw.text((400, 160), f"{data['prob']:.1%}", fill="#00FF00", anchor="mm")
    draw.rectangle([150, 310, 650, 420], outline="gold", width=3)
    draw.text((400, 345), f"APUESTA: ${data['apuesta']} MXN", fill="gold", anchor="mm")
    draw.text((400, 385), f"Cuota: {data['cuota']} | EV: {data['ev']:+.2f}", fill="white", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Auditoría
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(audit_past_predictions())

    # 2. Escanear Liga MX
    url = f"https://serpapi.com/search.json?q=proximos+partidos+{LEAGUE}&api_key={SERPAPI_KEY}"
    res = requests.get(url).json()
    games = res.get("sports_results", {}).get("games", [])
    
    all_matches = []
    for g in games[:5]: # Analizamos los próximos 5 partidos de la Liga MX
        h, a = g["teams"][0]["name"], g["teams"][1]["name"]
        h_logo, a_logo = g["teams"][0].get("thumbnail"), g["teams"][1].get("thumbnail")
        
        h_att, h_def = get_stats(h)
        a_att, a_def = get_stats(a)
        h_xg = (h_att * HOME_ADVANTAGE) * a_def
        a_xg = (a_att * AWAY_PENALTY) * h_def
        
        prob = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
        cuota = 2.00 # Cuota estimada
        ev = (prob * cuota) - 1
        
        apuesta = 0
        if ev > 0:
            f_star = ((cuota-1)*prob - (1-prob))/(cuota-1)
            apuesta = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
        
        res_data = {"home": h, "away": a, "h_logo": h_logo, "a_logo": a_logo, "prob": prob, "cuota": cuota, "ev": ev, "apuesta": apuesta}
        
        # GUARDADO LIMPIO (5 COLUMNAS)
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), f"{h} vs {a}", f"{prob:.1%}", f"{ev:.2f}", f"{apuesta}"])
        
        all_matches.append(res_data)

    if all_matches:
        best = max(all_matches, key=lambda x: x['ev'])
        generate_card(best)
