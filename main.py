import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime

# --- CONFIGURACIÓN LIVE ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.15 # Menor riesgo en vivo por la volatilidad
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_live_matches():
    """Busca partidos que se están jugando EN ESTE MOMENTO"""
    url = f"https://serpapi.com/search.json?q=liga+mx+en+vivo+scores&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        games = res.get("sports_results", {}).get("games", [])
        
        live_games = []
        for g in games:
            status = g.get("status", "").lower()
            # Detectamos si el partido tiene un minuto (ej: 66', 45', 1er T)
            if any(char.isdigit() for char in status) or "vivo" in status or "t" in status:
                h_team = g["teams"][0]["name"]
                a_team = g["teams"][1]["name"]
                h_score = int(g["teams"][0].get("score", 0))
                a_score = int(g["teams"][1].get("score", 0))
                
                # Extraer minuto (si dice 66', tomamos 66)
                minute_match = re.search(r'(\d+)', status)
                minute = int(minute_match.group(1)) if minute_match else 45
                
                live_games.append({
                    "home": h_team, "away": a_team,
                    "h_score": h_score, "a_score": a_score,
                    "minute": minute, "status": status,
                    "h_logo": g["teams"][0].get("thumbnail"),
                    "a_logo": g["teams"][1].get("thumbnail")
                })
        return live_games
    except: return []

def predict_live(game):
    """Ajusta Poisson al tiempo restante del partido"""
    rem_time = max(1, 95 - game['minute']) # Tiempo restante incluyendo compensación
    factor = rem_time / 90
    
    # xG base (puedes mejorar esto con get_stats anterior)
    h_xg_rem = 1.5 * factor 
    a_xg_rem = 1.2 * factor
    
    # Probabilidad de que caiga AL MENOS un gol más (Over 0.5 restante)
    prob_more_goals = 1 - (poisson.pmf(0, h_xg_rem) * poisson.pmf(0, a_xg_rem))
    
    # Probabilidad de victoria local desde este momento
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(5), h_xg_rem), poisson.pmf(range(5), a_xg_rem)), -1))
    
    cuota_live = 1.85 # Esto debería venir de la API de cuotas en vivo
    ev = (prob_l * cuota_live) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0 else 0
    
    return {**game, "prob": prob_l, "ev": ev, "apuesta": max(0, apuesta), "more_goals": prob_more_goals}

def generate_live_card(data):
    """Imagen con marcador y minuto en tiempo real"""
    img = Image.new('RGB', (800, 450), color=(20, 20, 40)) # Azul oscuro para "Live"
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0, 0, 800, 60], fill="red") # Barra de "EN VIVO"
    draw.text((400, 30), f"• EN VIVO - MINUTO {data['minute']}'", fill="white", anchor="mm")
    
    # Marcador
    draw.text((400, 150), f"{data['h_score']} - {data['a_score']}", fill="white", anchor="mm")
    draw.text((150, 150), data['home'][:12], fill="gold", anchor="mm")
    draw.text((650, 150), data['away'][:12], fill="gold", anchor="mm")
    
    # Predicción Live
    draw.text((400, 230), f"Prob. Gol en lo que resta: {data['more_goals']:.1%}", fill="#00FF00", anchor="mm")
    
    draw.rectangle([150, 300, 650, 410], outline="white", width=2)
    draw.text((400, 335), f"APUESTA EN VIVO: ${data['apuesta']} MXN", fill="white", anchor="mm")
    draw.text((400, 375), f"Sugerencia: Próximo Gol / Ganador Resto del Partido", fill="gray", anchor="mm")
    
    img.save("prediction_card.png")

if __name__ == "__main__":
    print("Buscando partidos en curso...")
    live_matches = get_live_matches()
    
    if live_matches:
        # Analizamos el partido más avanzado o con más valor
        best_live = predict_live(live_matches[0])
        generate_live_card(best_live)
        
        # Guardar en historial
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now(), f"LIVE: {best_live['home']} vs {best_live['away']}", f"{best_live['prob']:.1%}", best_live['ev'], best_live['apuesta']])
        
        print(f"✅ Analizado: {best_live['home']} vs {best_live['away']} al minuto {best_live['minute']}")
    else:
        # Si no hay nada en vivo, buscar próximos
        print("No hay partidos en vivo ahora mismo.")
        # Aquí podrías llamar a tu función de próximos partidos
