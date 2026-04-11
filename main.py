import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# --- CONFIGURACIÓN CORREGIDA ---
BANKROLL_REAL = 21830.00  # SALDO REAL TRAS GANANCIAS MX
KELLY_FRACTION = 0.15
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def predict_elite(h_xg, a_xg, cuota):
    """Fuerza el cálculo real de Poisson, sin valores por defecto"""
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    # Solo apostar si hay valor real (EV > 0.05)
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_REAL, 2) if ev > 0.05 else 0
    return prob_l, ev, max(0, apuesta)

if __name__ == "__main__":
    # --- PICKS DE ÉLITE PARA HOY (SÁBADO 11 ABRIL 2026) ---
    # He cargado manualmente los xG proyectados para evitar el error de la API
    picks_hoy = [
        {"match": "Tigres vs Chivas", "h_xg": 2.2, "a_xg": 1.1, "cuota": 2.10},
        {"match": "América vs Cruz Azul", "h_xg": 1.7, "a_xg": 1.4, "cuota": 2.25},
        {"match": "Pachuca vs Santos", "h_xg": 1.9, "a_xg": 1.0, "cuota": 1.85}
    ]
    
    resultados = []
    for p in picks_hoy:
        prob, ev, apuesta = predict_elite(p['h_xg'], p['a_xg'], p['cuota'])
        resultados.append({**p, "prob": prob, "ev": ev, "apuesta": apuesta})
        
        # Guardar en CSV con el saldo correcto
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), p['match'], f"{prob:.1%}", f"{ev:.2f}", f"${apuesta} MXN"])

    # (Aquí sigue tu función de generar imagen con los resultados de 'resultados')
    print("✅ Modelo recalibrado con Bankroll de $21,830 MXN")
