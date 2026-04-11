import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# --- CONFIGURACIÓN DE ÉLITE ---
BANKROLL_REAL = 21830.00  # TU SALDO REAL TRAS GANANCIAS MX
KELLY_FRACTION = 0.15
API_KEY = os.getenv("ALL_SPORTS_API_KEY")

def force_initialize_files():
    """Crea los archivos de inmediato para que el Workflow no falle"""
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write("<b>📊 ESCÁNER LIGA MX - JORNADA 14</b>\nBuscando partidos en vivo o próximos...")
    
    # Imagen de respaldo por si no hay partidos
    img = Image.new('RGB', (800, 450), color=(15, 15, 15))
    img.save("prediction_card.png")

def predict_elite(h_xg, a_xg, cuota):
    """Cálculo real de Poisson"""
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_REAL, 2) if ev > 0.05 else 0
    return prob_l, ev, max(0, apuesta)

def generate_card_mx(data):
    """Genera la imagen profesional para la Liga MX"""
    img = Image.new('RGB', (1000, 500), color=(10, 25, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 70], fill="#D4AF37")
    draw.text((500, 35), f"LIGA MX - JORNADA 14 - {datetime.now().strftime('%d/%m/%Y')}", fill="black", anchor="mm")
    
    y = 150
    for p in data:
        draw.text((100, y), f"{p['match']}", fill="white")
        draw.text((100, y+40), f"Pick: Victoria Local | Prob: {p['prob']:.1%}", fill="#00FF00")
        draw.text((700, y+40), f"Stake: ${p['apuesta']} MXN", fill="gold")
        draw.line([100, y+90, 900, y+90], fill=(50, 50, 50))
        y += 120
    
    draw.text((500, 470), f"BANKROLL: ${BANKROLL_REAL} MXN", fill="white", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. EVITAR ERROR DE WORKFLOW
    force_initialize_files()

    # 2. CARGA DE DATOS LIGA MX (SÁBADO 11 ABRIL 2026)
    # Cargamos los xG proyectados manualmente para asegurar precisión hoy
    picks_hoy = [
        {"match": "Tigres vs Chivas", "h_xg": 2.24, "a_xg": 1.06, "cuota": 2.10},
        {"match": "América vs Cruz Azul", "h_xg": 1.65, "a_xg": 1.38, "cuota": 2.25},
        {"match": "Pachuca vs Santos", "h_xg": 1.95, "a_xg": 1.02, "cuota": 1.85}
    ]
    
    final_results = []
    reporte = "<b>🏆 PICKS DE ÉLITE LIGA MX</b>\n\n"
    
    for p in picks_hoy:
        prob, ev, apuesta = predict_elite(p['h_xg'], p['a_xg'], p['cuota'])
        res = {**p, "prob": prob, "ev": ev, "apuesta": apuesta}
        final_results.append(res)
        
        reporte += f"🔹 {p['match']}\n   Prob: {prob:.1%} | EV: {ev:+.2f}\n   <b>Apuesta: ${apuesta} MXN</b>\n\n"
        
        # Guardar en el CSV histórico
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), p['match'], f"{prob:.1%}", f"{ev:.2f}", f"{apuesta}"])

    # 3. ACTUALIZAR ARCHIVOS PARA TELEGRAM
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(reporte)
    
    generate_card_mx(final_results)
    print("✅ Jornada de Liga MX procesada correctamente.")
