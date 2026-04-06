import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.20
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_today_matches():
    """Busca SOLO los partidos de la jornada de HOY"""
    url = f"https://serpapi.com/search.json?q=liga+mx+partidos+hoy&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        games = res.get("sports_results", {}).get("games", [])
        
        today_games = []
        for g in games:
            status = g.get("status", "").lower()
            # FILTRO CRÍTICO: Solo tomamos partidos que NO han terminado
            # Buscamos estados como "12:00", "En vivo", "70'", "Hoy"
            if "final" not in status and "postp" not in status:
                h = g["teams"][0]["name"]
                a = g["teams"][1]["name"]
                h_logo = g["teams"][0].get("thumbnail")
                a_logo = g["teams"][1].get("thumbnail")
                today_games.append({"home": h, "away": a, "h_logo": h_logo, "a_logo": a_logo, "status": status})
        
        return today_games
    except:
        return []

def get_stats(team):
    """Estadísticas frescas para la jornada actual"""
    url = f"https://serpapi.com/search.json?q={team}+xG+stats+2024+mexico&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        snippets = " ".join([r.get("snippet", "") for r in res.get("organic_results", [])[:3]])
        nums = [float(v) for v in re.findall(r'(\d\.\d+)', snippets) if 0.5 < float(v) < 3.5]
        return (nums[0], nums[1]) if len(nums) >= 2 else (1.4, 1.3)
    except: return 1.4, 1.3

def audit_recent_results():
    """Audita solo los partidos de las últimas 24 horas"""
    file = "predictions_history.csv"
    if not os.path.isfile(file): return "<b>📊 AUDITORÍA</b>\nSin datos."
    
    report = "<b>📊 RESULTADOS DE LA JORNADA</b>\n\n"
    with open(file, 'r') as f:
        rows = list(csv.reader(f))
        # Solo revisamos los últimos partidos guardados
        for row in rows[-6:]:
            try:
                fecha_str, partido, prob, ev, apuesta = row[:5]
                # Si la predicción es de hoy o ayer, la auditamos
                home, away = partido.split(" vs ")
                
                # Buscamos el resultado real
                url_res = f"https://serpapi.com/search.json?q=resultado+final+{home}+vs+{away}+liga+mx&api_key={SERPAPI_KEY}"
                res_data = requests.get(url_res).json()
                snippet = str(res_data.get("sports_results", "")) + str(res_data.get("organic_results", ""))
                
                scores = re.findall(r'(\d+)\s*-\s*(\d+)', snippet)
                if scores:
                    h, a = int(scores[0][0]), int(scores[0][1])
                    res_final = "L" if h > a else "E" if h == a else "V"
                    icon = "✅" if res_final == "L" else "❌" # Asumiendo apuesta al local
                    report += f"{icon} {home} {h}-{a} {away}\n"
            except: continue
    return report

def generate_card(data):
    """Imagen con el estado del partido (En vivo o la hora)"""
    img = Image.new('RGB', (800, 450), color=(10, 30, 10))
    draw = ImageDraw.Draw(img)
    def load_img(url):
        try:
            r = requests.get(url, timeout=5)
            return Image.open(BytesIO(r.content)).convert("RGBA").resize((140, 140))
        except: return Image.new('RGBA', (140, 140), color=(50, 50, 50))
    
    img.paste(load_img(data['h_logo']), (70, 100), load_img(data['h_logo']))
    img.paste(load_img(data['a_logo']), (590, 100), load_img(data['a_logo']))
    
    draw.text((400, 40), f"JORNADA EN VIVO - {data['status'].upper()}", fill="gold", anchor="mm")
    draw.text((140, 260), data['home'][:12], fill="white", anchor="mm")
    draw.text((660, 260), data['away'][:12], fill="white", anchor="mm")
    draw.text((400, 160), f"{data['prob']:.1%}", fill="#00FF00", anchor="mm")
    draw.rectangle([150, 310, 650, 420], outline="gold", width=3)
    draw.text((400, 345), f"APUESTA: ${data['apuesta']} MXN", fill="gold", anchor="mm")
    draw.text((400, 385), f"Cuota: {data['cuota']} | EV: {data['ev']:+.2f}", fill="white", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Auditoría de lo que ya terminó
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(audit_recent_results())

    # 2. Análisis de los partidos de HOY
    today_matches = get_today_matches()
    
    all_results = []
    for m in today_matches:
        h, a = m["home"], m["away"]
        h_att, h_def = get_stats(h)
        a_att, a_def = get_stats(a)
        
        # Poisson
        h_xg, a_xg = (h_att * 1.1) * a_def, (a_att * 0.9) * h_def
        prob = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
        
        cuota = 2.05 # Aquí podrías buscar cuotas reales en vivo
        ev = (prob * cuota) - 1
        apuesta = round((((cuota-1)*prob - (1-prob))/(cuota-1)) * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0 else 0
        
        res_data = {**m, "prob": prob, "cuota": cuota, "ev": ev, "apuesta": max(0, apuesta)}
        
        # Guardar en historial
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), f"{h} vs {a}", f"{prob:.1%}", f"{ev:.2f}", f"{apuesta}"])
        
        all_results.append(res_data)

    if all_results:
        # Enviamos el que tenga más valor (EV) de los partidos de hoy
        best = max(all_results, key=lambda x: x['ev'])
        generate_card(best)
    else:
        # Si no hay partidos hoy, generar una imagen de aviso
        img = Image.new('RGB', (800, 450), color=(20, 20, 20))
        ImageDraw.Draw(img).text((400, 225), "NO HAY PARTIDOS DE LIGA MX HOY", fill="white", anchor="mm")
        img.save("prediction_card.png")
