import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from datetime import datetime

# --- CONFIGURACIÓN DE CRITERIOS ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.15
MAX_ODD = 4.0        # Filtro estricto para evitar cuotas "imposibles"
MIN_PROB = 0.40      # Solo entrar si hay al menos 40% de certeza
API_KEY = os.getenv("ALL_SPORTS_API_KEY")

def calculate_live_probability(home_score, away_score, status):
    """
    Calcula la probabilidad de que ocurra al menos un gol más
    basado en el promedio de goles por minuto del partido actual.
    """
    try:
        # Extraer el minuto actual del status (ej: "65'")
        current_minute = int(''.join(filter(str.isdigit, status)))
    except:
        return 0.0

    remaining_time = 90 - current_minute
    if remaining_time <= 0: return 0.0

    # Lambda (λ): Promedio de goles actuales por minuto
    total_goals = home_score + away_score
    goals_per_minute = total_goals / max(current_minute, 1)
    
    # λ para el tiempo que queda
    expected_lambda = goals_per_minute * remaining_time
    
    # Probabilidad de 0 goles adicionales: P(0) = (λ^0 * e^-λ) / 0!
    prob_zero_goals = poisson.pmf(0, expected_lambda)
    
    # Probabilidad de que SÍ haya más goles (Over)
    return 1 - prob_zero_goals

def get_real_data():
    """Conexión directa a All Sports API"""
    url = f"https://apiv2.allsportsapi.com/football/?met=Livescore&APIkey={API_KEY}"
    try:
        response = requests.get(url).json()
        raw_matches = response.get("result", [])
        clean_matches = []

        for m in raw_matches:
            # Extraer cuotas reales si el plan las incluye en el Livescore
            # Si no, All Sports API las tiene en 'bookmakers' o un met=Odds
            odds_list = m.get("odds", [])
            actual_odd = 1.0
            
            # Buscamos la cuota del Over 0.5 o la victoria local según disponibilidad
            if odds_list:
                actual_odd = float(odds_list[0].get("value", 1.0))
            else:
                # Si el plan es básico, a veces no trae cuotas en Livescore.
                # Aquí podrías hacer un segundo fetch a met=Odds, 
                # por ahora saltamos si no hay cuota para no "simular".
                continue

            if actual_odd > MAX_ODD or actual_odd <= 1.1:
                continue

            h_score = int(m.get("event_final_result", "0-0").split("-")[0].strip())
            a_score = int(m.get("event_final_result", "0-0").split("-")[1].strip())
            
            prob = calculate_live_probability(h_score, a_score, m.get("event_status", "0"))

            if prob >= MIN_PROB:
                clean_matches.append({
                    "home": m.get("event_home_team"),
                    "away": m.get("event_away_team"),
                    "score": f"{h_score}-{a_score}",
                    "status": m.get("event_status"),
                    "odd": actual_odd,
                    "prob": prob
                })
        return clean_matches
    except:
        return []

def run_bot():
    # 1. Crear archivo de auditoría para evitar error de GitHub Actions
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(f"<b>🔍 ESCANEO REAL - {datetime.now().strftime('%H:%M')}</b>\n")

    matches = get_real_data()

    if not matches:
        with open("audit_report.txt", "a") as f:
            f.write("No se encontraron oportunidades que superen los filtros de riesgo.")
        return

    # 2. Seleccionar el mejor pick basado en el criterio de Kelly
    # f* = (bp - q) / b
    for pick in matches:
        b = pick['odd'] - 1
        p = pick['prob']
        q = 1 - p
        edge = (p * b - q) / b
        
        if edge > 0:
            stake = edge * BANKROLL_MXN * KELLY_FRACTION
            
            # Generar reporte detallado
            with open("audit_report.txt", "a", encoding="utf-8") as f:
                f.write(f"🔥 <b>VALOR DETECTADO</b>\n")
                f.write(f"Partido: {pick['home']} vs {pick['away']}\n")
                f.write(f"Cuota Real: {pick['odd']}\n")
                f.write(f"Probabilidad (Poisson): {p:.1%}\n")
                f.write(f"Sugerencia: ${stake:.2f} MXN")
            
            # Crear imagen de la tarjeta
            img = Image.new('RGB', (800, 400), color=(15, 15, 25))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 800, 50], fill="gold")
            d.text((400, 25), "OPORTUNIDAD DE ALTA PROBABILIDAD", fill="black", anchor="mm")
            d.text((400, 150), f"{pick['home']} {pick['score']} {pick['away']}", fill="white", anchor="mm")
            d.text((400, 250), f"Cuota: {pick['odd']} | Confianza: {p:.1%}", fill="gold", anchor="mm")
            d.text((400, 320), f"STAKE RECOMENDADO: ${stake:.2f}", fill="#00FF00", anchor="mm")
            img.save("prediction_card.png")
            break

if __name__ == "__main__":
    run_bot()
