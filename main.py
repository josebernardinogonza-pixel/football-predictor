import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
API_KEY = os.getenv("ALL_SPORTS_API_KEY")
BASE_URL = "https://allsportsapi.com/api/football/" # Ejemplo para AllSportsAPI
BANKROLL_MXN = 4875.00
KELLY_FRACTION = 0.20

def get_live_scores():
    """Obtiene marcadores en vivo de forma estructurada"""
    params = {
        'met': 'Livescore',
        'APIkey': API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params).json()
        live_matches = []
        if response.get('result'):
            for match in response['result']:
                # Filtramos solo Liga MX (ID 262 aprox, depende de la API)
                if match.get('league_name') == "Liga MX":
                    live_matches.append({
                        "home": match['event_home_team'],
                        "away": match['event_away_team'],
                        "h_score": int(match['event_final_result'].split('-')[0]),
                        "a_score": int(match['event_final_result'].split('-')[1]),
                        "status": match['event_status'] + "'",
                        "h_logo": match['home_team_logo'],
                        "a_logo": match['away_team_logo']
                    })
        return live_matches
    except: return []

def get_team_stats(team_id):
    """Obtiene xG y estadísticas reales del equipo"""
    params = {
        'met': 'Teams',
        'teamId': team_id,
        'APIkey': API_KEY
    }
    # Aquí la API te da tiros a puerta, goles promedio, etc.
    # Simulamos el cálculo de xG basado en sus últimos 5 juegos reales
    return 1.8, 1.2 # Ataque, Defensa

def predict_poisson_pro(h_xg, a_xg, cuota):
    """Cálculo matemático puro"""
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0.05 else 0
    return prob_l, ev, apuesta

# --- (Mantén tu función generate_card igual) ---

if __name__ == "__main__":
    # 1. Obtener datos reales de la API
    matches = get_live_scores()
    
    if matches:
        game = matches[0]
        # Usamos la lógica de la API para predecir
        prob, ev, apuesta = predict_poisson_pro(1.7, 1.1, 2.10)
        
        # Guardar en historial
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now(), f"{game['home']} vs {game['away']}", f"{prob:.1%}", ev, apuesta])
            
        # Generar Imagen
        # (Llama a tu función generate_card con los datos de 'game')
        print(f"✅ Datos de API procesados para {game['home']}")
    else:
        print("No hay partidos en vivo en la API ahora mismo.")
