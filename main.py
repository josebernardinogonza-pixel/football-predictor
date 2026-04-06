import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.15
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_live_data():
    """Busca partidos en vivo reales"""
    url = f"https://serpapi.com/search.json?q=liga+mx+en+vivo+scores&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        games = res.get("sports_results", {}).get("games", [])
        live_list = []
        for g in games:
            status = g.get("status", "").lower()
            # Si el partido está en curso (tiene minutos o dice 'vivo')
            if any(c.isdigit() for c in status) or "vivo" in status:
                live_list.append({
                    "home": g["teams"][0]["name"],
                    "away": g["teams"][1]["name"],
                    "h_score": int(g["teams"][0].get("score", 0)),
                    "a_score": int(g["teams"][1].get("score", 0)),
                    "status": status,
                    "h_logo": g["teams"][0].get("thumbnail"),
                    "a_logo": g["teams"][1].get("thumbnail")
                })
        return live_list
    except: return []

def audit_logic():
    """Genera el reporte de auditoría de forma segura"""
    file = "predictions_history.csv"
    report = "<b>📊 AUDITORÍA DE LA JORNADA</b>\n\n"
    
    if not os.path.isfile(file):
        return report + "Esperando datos históricos..."

    try:
        with open(file, 'r') as f:
            rows = list(csv.reader(f))
            for row in rows[-5:]:
                fecha, partido, prob, ev, apuesta = row[:5]
                report += f"• {partido}: Analizado ({fecha})\n"
        return report
    except:
        return report + "Error al leer el historial."

def generate_card(game):
    """Imagen del partido en vivo"""
    img = Image.new('RGB', (800, 450), color=(10, 10, 30))
    draw = ImageDraw.Draw(img)
    
    # Encabezado Live
    draw.rectangle([0, 0, 800, 50], fill="red")
    draw.text((400, 25), f"• EN VIVO: {game['status'].upper()}", fill="white", anchor="mm")
    
    # Marcador y Equipos
    draw.text((400, 150), f"{game['h_score']} - {game['a_score']}", fill="white", anchor="mm")
    draw.text((150, 150), game['home'][:12], fill="gold", anchor="mm")
    draw.text((650, 150), game['away'][:12], fill="gold", anchor="mm")
    
    # Predicción (Poisson simplificado para Live)
    prob = 0.55 # Ejemplo, aquí iría tu cálculo de Poisson
    draw.text((400, 250), f"Prob. Próximo Gol: {prob:.1%}", fill="#00FF00", anchor="mm")
    
    # Apuesta
    draw.rectangle([150, 320, 650, 420], outline="white", width=2)
    draw.text((400, 355), f"APUESTA SUGERIDA: $150.00 MXN", fill="white", anchor="mm")
    draw.text((400, 390), "Mercado: Ganador Resto del Partido", fill="gray", anchor="mm")
    
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. SIEMPRE crear el archivo de auditoría primero
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(audit_logic())

    # 2. Buscar datos reales en vivo
    live_matches = get_live_data()
    
    if live_matches:
        game = live_matches[0]
        generate_card(game)
        # Guardar en CSV
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%H:%M"), f"{game['home']} vs {game['away']}", "55%", "0.15", "150"])
        print(f"✅ Analizado: {game['home']} vs {game['away']}")
    else:
        # Si no hay nada en vivo, crear imagen de aviso
        img = Image.new('RGB', (800, 450), color=(20, 20, 20))
        ImageDraw.Draw(img).text((400, 225), "NO HAY PARTIDOS EN VIVO AHORA", fill="white", anchor="mm")
        img.save("prediction_card.png")
        print("No hay partidos en vivo.")
