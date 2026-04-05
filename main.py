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
    # Crear imagen con mejor resolución (1000x600)
    img = Image.new('RGB', (1000, 600), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    
    # Intentar cargar logos con un tiempo de espera mayor
    def load_img(url):
        try:
            if not url: return Image.new('RGBA', (180, 180), color=(30, 30, 30))
            r = requests.get(url, timeout=10)
            return Image.open(BytesIO(r.content)).convert("RGBA").resize((180, 180))
        except:
            return Image.new('RGBA', (180, 180), color=(30, 30, 30))

    logo_h = load_img(data['h_logo'])
    logo_a = load_img(data['a_logo'])
    
    # Posicionar Logos
    img.paste(logo_h, (100, 120), logo_h)
    img.paste(logo_a, (720, 120), logo_a)

    # --- TEXTOS ---
    # Título Superior
    draw.text((500, 50), "PRONÓSTICO PROFESIONAL", fill="gold", anchor="mm")
    
    # Nombres de Equipos (Más grandes)
    draw.text((190, 320), data['home'][:15].upper(), fill="white", anchor="mm")
    draw.text((810, 320), data['away'][:15].upper(), fill="white", anchor="mm")
    
    # Etiquetas debajo de logos
    draw.text((190, 350), "LOCAL", fill="gray", anchor="mm")
    draw.text((810, 350), "VISITANTE", fill="gray", anchor="mm")

    # PROBABILIDAD (Centro y Gigante)
    # Dibujamos el texto varias veces para simular "Negrita" si no hay fuentes
    prob_text = f"{data['prob']:.1%}"
    pos_centro = (500, 200)
    draw.text(pos_centro, prob_text, fill="#00FF00", anchor="mm")
    draw.text((500, 240), "PROBABILIDAD DE VICTORIA", fill="white", anchor="mm")

    # CUADRO DE APUESTA (Llamativo)
    draw.rectangle([150, 420, 850, 550], outline="gold", width=5)
    
    apuesta_text = f"APUESTA SUGERIDA: ${data['apuesta']} MXN"
    draw.text((500, 460), apuesta_text, fill="gold", anchor="mm")
    
    info_text = f"CUOTA: {data['cuota']}  |  VALOR (EV): {data['ev']:+.2f}"
    draw.text((500, 510), info_text, fill="white", anchor="mm")

    # Guardar imagen
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
