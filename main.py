import os
import requests
import re
import numpy as np
from scipy.stats import poisson
import csv
from datetime import datetime

# --- CONFIGURACIÓN FINANCIERA ---
BANKROLL_MXN = 5000.00  # Tu presupuesto total en Pesos Mexicanos
KELLY_FRACTION = 0.25   # "Quarter-Kelly": Usamos solo el 25% de lo sugerido para mayor seguridad

def calculate_kelly(prob, odds, bankroll):
    """Calcula la apuesta óptima en MXN"""
    if odds <= 1.0 or prob <= 0: return 0
    
    # b son las odds netas (Cuota - 1)
    b = odds - 1
    p = prob
    q = 1 - p
    
    # Fórmula de Kelly: (bp - q) / b
    f_star = (b * p - q) / b
    
    if f_star > 0:
        # Aplicamos la fracción de seguridad y multiplicamos por el bankroll
        apuesta_sugerida = f_star * KELLY_FRACTION * bankroll
        return round(apuesta_sugerida, 2)
    return 0.0

def save_to_csv(home, away, prob_l, odds_l, ev_l, apuesta_mxn):
    file_name = "predictions_history.csv"
    file_exists = os.path.isfile(file_name)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    nueva_fila = [
        fecha, home, away, 
        f"{prob_l:.2%}", f"{odds_l:.2f}", 
        f"{ev_l:+.2f}", f"${apuesta_mxn} MXN"
    ]

    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha", "Local", "Visitante", "Prob_L", "Cuota_Mkt", "EV", "Apuesta_Sugerida"])
        writer.writerow(nueva_fila)

if __name__ == "__main__":
    # 1. Datos del partido y predicción
    local = "Real Madrid"
    visitante = "Barcelona"
    
    # Obtenemos probabilidades (usando tu función anterior)
    pl, pe, pv, xgh, xga = predict_real_time(local, visitante)
    
    # 2. Obtenemos cuotas reales del mercado (ejemplo 2.10)
    # En un caso real, get_market_odds() buscaría esto en Google
    cuota_local = 2.15 
    
    # 3. Calculamos Valor Esperado (EV)
    ev_l = (pl * cuota_local) - 1
    
    # 4. Calculamos apuesta óptima en MXN
    apuesta_mxn = 0
    if ev_l > 0:
        apuesta_mxn = calculate_kelly(pl, cuota_local, BANKROLL_MXN)
    
    # 5. Guardar y Mostrar
    save_to_csv(local, visitante, pl, cuota_local, ev_l, apuesta_mxn)
    
    print(f"\n--- GESTIÓN DE BANCA MXN ---")
    print(f"Partido: {local} vs {visitante}")
    print(f"Probabilidad Local: {pl:.1%}")
    print(f"Cuota Mercado: {cuota_local}")
    print(f"Valor Esperado (EV): {ev_l:+.2f}")
    
    if apuesta_mxn > 0:
        print(f"💰 APUESTA SUGERIDA: ${apuesta_mxn} MXN")
        print(f"⚠️ (Basado en un Bankroll de ${BANKROLL_MXN} MXN)")
    else:
        print("❌ NO APOSTAR: No hay valor suficiente o el EV es negativo.")
