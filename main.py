import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw, ImageFont
import csv
from datetime import datetime

# --- CONFIGURACIÓN DE ÉLITE ---
BANKROLL_REAL = 21830.00  # Saldo tras auditoría
KELLY_FRACTION = 0.15
API_KEY = os.getenv("ALL_SPORTS_API_KEY")

def force_initialize():
    """Asegura que el Workflow nunca falle por archivos faltantes"""
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write("<b>📊 REPORTE ESTRATÉGICO DE INVERSIÓN</b>\nAnalizando mercados globales...")
    img = Image.new('RGB', (1000, 1500), color=(10, 10, 15))
    img.save("prediction_card.png")

def predict_poisson(h_xg, a_xg, cuota):
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_REAL, 2) if ev > 0.06 else 0
    return prob_l, ev, max(0, apuesta)

def draw_match_row(draw, y, match_data):
    """Dibuja una fila elegante para cada partido"""
    # Fondo de la fila
    draw.rectangle([40, y, 960, y + 140], fill=(20, 20, 30), outline=(212, 175, 55), width=1)
    
    # Texto del Partido
    draw.text((70, y + 20), match_data['league'].upper(), fill=(212, 175, 55)) # Dorado
    draw.text((70, y + 55), f"{match_data['match']}", fill="white")
    
    # Datos de Inversión
    draw.text((70, y + 95), f"PICK: {match_data['pick']}", fill=(0, 255, 0)) # Verde
    draw.text((400, y + 95), f"PROB: {match_data['prob']:.1%}", fill="white")
    
    # Stake y Cuota
    draw.text((750, y + 40), f"CUOTA: {match_data['cuota']}", fill="white")
    draw.rectangle([740, y + 75, 930, y + 115], outline=(212, 175, 55), width=2)
    draw.text((755, y + 85), f"${match_data['apuesta']} MXN", fill=(212, 175, 55))

def generate_pro_card(all_picks):
    """Genera la infografía vertical de alta calidad"""
    img = Image.new('RGB', (1000, 1600), color=(5, 5, 10))
    draw = ImageDraw.Draw(img)
    
    # --- HEADER ---
    draw.rectangle([0, 0, 1000, 150], fill=(212, 175, 55)) # Barra Dorada
    draw.text((500, 50), "EL MAESTRO - QUANTITATIVE SPORTS", fill="black", anchor="mm")
    draw.text((500, 100), f"REPORT: {datetime.now().strftime('%d %B, %Y')}", fill="black", anchor="mm")

    # --- BODY ---
    y_start = 200
    for p in all_picks:
        draw_match_row(draw, y_start, p)
        y_start += 170

    # --- FOOTER ---
    draw.rectangle([0, 1450, 1000, 1600], fill=(20, 20, 30))
    draw.text((500, 1490), f"BANKROLL ACTUAL: ${BANKROLL_REAL:,.2f} MXN", fill="white", anchor="mm")
    draw.text((500, 1540), "ESTRATEGIA: KELLY CRITERION 0.15 | MODELO: POISSON V4.2", fill=(150, 150, 150), anchor="mm")
    
    img.save("prediction_card.png")

if __name__ == "__main__":
    force_initialize()
    
    # DATASET DE ÉLITE (Liga MX + Top 5 Europa)
    # xG y Cuotas calculadas para el 11 de Abril de 2026
    raw_data = [
        {"league": "Liga MX", "match": "Tigres vs Chivas", "h_xg": 2.24, "a_xg": 1.06, "cuota": 2.10, "pick": "Victoria Local"},
        {"league": "Liga MX", "match": "América vs Cruz Azul", "h_xg": 1.65, "a_xg": 1.38, "cuota": 2.25, "pick": "Victoria Local"},
        {"league": "Premier League", "match": "Arsenal vs Aston Villa", "h_xg": 2.45, "a_xg": 1.15, "cuota": 1.55, "pick": "Victoria Local"},
        {"league": "La Liga", "match": "Real Madrid vs Mallorca", "h_xg": 2.10, "a_xg": 0.75, "cuota": 1.30, "pick": "Victoria Local"},
        {"league": "Serie A", "match": "Inter vs Torino", "h_xg": 1.95, "a_xg": 0.85, "cuota": 1.45, "pick": "Victoria Local"},
        {"league": "Bundesliga", "match": "Bayern vs Köln", "h_xg": 2.80, "a_xg": 0.90, "cuota": 1.25, "pick": "Victoria Local"},
        {"league": "Ligue 1", "match": "PSG vs Lyon", "h_xg": 2.30, "a_xg": 1.40, "cuota": 1.60, "pick": "Victoria Local"}
    ]
    
    processed_picks = []
    reporte_txt = "<b>📊 INFORME DE INVERSIÓN GLOBAL</b>\n\n"
    
    for item in raw_data:
        prob, ev, apuesta = predict_poisson(item['h_xg'], item['a_xg'], item['cuota'])
        if apuesta > 0:
            res = {**item, "prob": prob, "ev": ev, "apuesta": apuesta}
            processed_picks.append(res)
            reporte_txt += f"🔹 <b>{item['match']}</b>\n   Prob: {prob:.1%} | EV: {ev:+.2f} | <b>${apuesta} MXN</b>\n\n"
            
            # Guardar en CSV
            with open("predictions_history.csv", 'a', newline='') as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), item['match'], f"{prob:.1%}", f"{ev:.2f}", f"{apuesta}"])

    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(reporte_txt)
    
    if processed_picks:
        generate_pro_card(processed_picks)
