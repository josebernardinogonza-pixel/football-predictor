import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.20
HOME_ADVANTAGE = 1.10
AWAY_PENALTY = 0.90
NEWS_PENALTY = 0.85
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# ==========================================
# FUNCIONES DE AUDITORÍA (APRENDIZAJE)
# ==========================================

def get_real_result(home, away):
    """Busca el resultado real de un partido pasado"""
    query = f"resultado final {home} vs {away}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        snippet = str(res.get("sports_results", "")) + str(res.get("organic_results", ""))
        scores = re.findall(r'(\d+)\s*-\s*(\d+)', snippet)
        if scores:
            h, a = int(scores[0][0]), int(scores[0][1])
            return "L" if h > a else "E" if h == a else "V"
    except: return None
    return None

def audit_past_predictions():
    """Revisa el historial y calcula precisión (CON FILTRO DE SEGURIDAD)"""
    file = "predictions_history.csv"
    if not os.path.isfile(file): return "Esperando primer historial..."
    
    hits, total = 0, 0
    report = "📊 *AUDITORÍA DE RESULTADOS*\n"
    
    with open(file, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
        # Analizamos las últimas 5 filas para ver rendimiento reciente
        for row in rows[-5:]:
            # --- FILTRO DE SEGURIDAD: Solo procesar filas con 6 columnas (formato actual) ---
            if len(row) != 6: 
                continue 
            
            fecha, partido, prob, cuota, ev, apuesta = row
            if "vs" not in partido: continue
            
            try:
                home, away = partido.split(" vs ")
                res_real = get_real_result(home, away)
                if res_real:
                    total += 1
                    if res_real == "L": # Asumimos que apostamos al Local (L)
                        hits += 1
                        report += f"✅ {home}: ACERTADO\n"
                    else:
                        report += f"❌ {home}: FALLADO\n"
            except: continue

    if total > 0:
        report += f"\n🎯 *Precisión Reciente: {(hits/total):.1%}"
    else:
        report = "⏳ Analizando partidos en curso..."
    return report

# ==========================================
# FUNCIONES DE PREDICCIÓN
# ==========================================

def get_stats(team):
    url = f"https://serpapi.com/search.json?q={team}+xG+stats+2024&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        text = str(res.get("organic_results", ""))
        nums = [float(v) for v in re.findall(r'(\d\.\d+)', text) if 0.5 < float(v) < 3.5]
        return (nums[0], nums[1]) if len(nums) >= 2 else (1.4, 1.3)
    except: return 1.4, 1.3

def get_news(team):
    url = f"https://serpapi.com/search.json?engine=google_news&q={team}+injuries&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        count = sum(1 for a in res.get("news_results", [])[:5] if any(w in a.get("title","").lower() for w in ["injury","out","baja"]))
        return NEWS_PENALTY if count >= 2 else 1.0
    except: return 1.0

def generate_card(data):
    img = Image.new('RGB', (800, 450), color=(15, 15, 15))
    draw = ImageDraw.Draw(img)
    def load_img(url):
        try: return Image.open(BytesIO(requests.get(url).content)).convert("RGBA").resize((130, 130))
        except: return Image.new('RGBA', (130, 130), color=(40, 40, 40))
    
    img.paste(load_img(data['h_logo']), (80, 100), load_img(data['h_logo']))
    img.paste(load_img(data['a_logo']), (590, 100), load_img(data['a_logo']))
    draw.text((400, 40), "PRONÓSTICO LIGA MX", fill="gold", anchor="mm")
    draw.text((400, 160), f"{data['prob']:.1%}", fill="#00FF00", anchor="mm")
    draw.rectangle([150, 300, 650, 410], outline="gold", width=2)
    draw.text((400, 335), f"APUESTA: ${data['apuesta']} MXN", fill="gold", anchor="mm")
    img.save("prediction_card.png")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # 1. Auditoría
    resumen = audit_past_predictions()
    with open("audit_report.txt", "w", encoding="utf-8") as f: f.write(resumen)

    # 2. Nuevas Predicciones
    url = f"https://serpapi.com/search.json?q=proximos+partidos+Liga+MX&api_key={SERPAPI_KEY}"
    matches = requests.get(url).json().get("sports_results", {}).get("games", [])[:3]
    
    results = []
    for m in matches:
        h, a = m["teams"][0]["name"], m["teams"][1]["name"]
        h_logo, a_logo = m["teams"][0].get("thumbnail"), m["teams"][1].get("thumbnail")
        
        h_att, h_def = get_stats(h)
        a_att, a_def = get_stats(a)
        h_xg = (h_att * a_def) * HOME_ADVANTAGE * get_news(h)
        a_xg = (a_att * h_def) * AWAY_PENALTY * get_news(a)
        
        prob = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
        cuota = 2.10 # Ejemplo
        ev = (prob * cuota) - 1
        
        apuesta = 0
        if ev > 0:
            f_star = ((cuota - 1) * prob - (1 - prob)) / (cuota - 1)
            apuesta = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
        
        res_data = {"home": h, "away": a, "h_logo": h_logo, "a_logo": a_logo, "prob": prob, "cuota": cuota, "ev": ev, "apuesta": max(0, apuesta)}
        
        # GUARDAR EN CSV (Formato estandarizado de 6 columnas)
        with open("predictions_history.csv", 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d"), f"{h} vs {a}", f"{prob:.1%}", cuota, f"{ev:.2f}", f"{apuesta}"])
        
        results.append(res_data)

    if results:
        best = max(results, key=lambda x: x['ev'])
        generate_card(best)
