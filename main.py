import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# --- CONFIGURACIÓN DE RECUPERACIÓN ---
BANKROLL_MXN = 4875.00  # Actualizado tras pérdida del Real Madrid
KELLY_FRACTION = 0.15   # Reducimos riesgo para estabilizar la banca
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def predict_football(h_xg, a_xg, cuota):
    """Modelo Poisson para Liga MX / Champions"""
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0.05 else 0
    return prob_l, ev, max(0, apuesta)

def predict_nba(h_rtg, a_rtg, cuota):
    """Modelo de Eficiencia para NBA"""
    # Simplificación: Probabilidad basada en diferencial de Rating
    diff = h_rtg - a_rtg
    prob = 1 / (1 + np.exp(-diff/10)) # Sigmoide para probabilidad
    ev = (prob * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0.05 else 0
    return prob, ev, max(0, apuesta)

def generate_master_card(picks):
    """Genera la infografía de inversión"""
    img = Image.new('RGB', (1000, 1000), color=(10, 10, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 100], fill="#D4AF37")
    draw.text((500, 50), "CARTELERA DE RECUPERACIÓN - 2026", fill="black", anchor="mm")

    y = 150
    for p in picks:
        color = "#00FF00" if p['ev'] > 0.10 else "white"
        draw.text((100, y), f"{p['match']} ({p['type']})", fill="white")
        draw.text((100, y+40), f"Pick: {p['pick']} | Prob: {p['prob']:.1%}", fill=color)
        draw.text((700, y+40), f"Stake: ${p['apuesta']} MXN", fill="gold")
        draw.line([100, y+90, 900, y+90], fill=(50, 50, 50))
        y += 120

    draw.text((500, 950), f"BANKROLL ACTUAL: ${BANKROLL_MXN} MXN", fill="gold", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. REPORTE DE AUDITORÍA (Real Madrid vs Bayern)
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write("<b>📊 AUDITORÍA CRÍTICA</b>\n"
                "❌ Real Madrid 1-2 Bayern\n"
                "Pérdida: -$125.00 MXN\n"
                "Estado: Recalibrando modelo...\n\n"
                "<b>🚀 NUEVOS PICKS DE RECUPERACIÓN</b>")

    # 2. GENERAR NUEVOS PICKS (NBA + LIGA MX)
    picks = [
        {"match": "DEN Nuggets vs POR Blazers", "type": "NBA", "pick": "Denver ML", "prob": 0.88, "cuota": 1.25},
        {"match": "Tigres vs Guadalajara", "type": "LMX", "pick": "Tigres ML", "prob": 0.52, "cuota": 2.10},
        {"match": "NY Knicks vs ATL Hawks", "type": "NBA", "pick": "Knicks ML", "prob": 0.66, "cuota": 1.70}
    ]
    
    final_picks = []
    for p in picks:
        if p['type'] == "NBA":
            prob, ev, apuesta = predict_nba(115, 105, p['cuota']) # Ratings simulados
        else:
            prob, ev, apuesta = predict_football(1.9, 1.1, p['cuota'])
        
        res = {**p, "prob": prob, "ev": ev, "apuesta": apuesta}
        final_picks.append(res)
        
        # Guardar en CSV (Fix de columnas)
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), p['match'], f"{prob:.1%}", f"{ev:.2f}", f"{apuesta}"])

    generate_master_card(final_picks)
