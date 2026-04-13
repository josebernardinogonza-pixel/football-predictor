import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
STAKE_API_KEY = os.getenv("STAKE_API_KEY")
STAKE_URL = "https://api.stake.com/graphql" # Endpoint oficial de Stake
BANKROLL_MXN = 21830.00
KELLY_FRACTION = 0.15

def get_direct_stake_odds():
    """Consulta cuotas de fútbol directamente en Stake.com usando GraphQL"""
    # Esta es la consulta técnica que entiende el servidor de Stake
    query = """
    {
      activeSports(sport: "soccer") {
        groups {
          events {
            name
            id
            specifiers
            markets(names: ["winner"]) {
              outcomes {
                name
                value
              }
            }
          }
        }
      }
    }
    """
    headers = {
        "x-access-token": STAKE_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(STAKE_URL, json={'query': query}, headers=headers)
        data = response.json()
        
        odds_results = []
        # Navegamos por la estructura de datos de Stake
        for group in data['data']['activeSports'][0]['groups']:
            for event in group['events']:
                try:
                    match_name = event['name']
                    # Extraer cuota del local (Home)
                    h_odds = event['markets'][0]['outcomes'][0]['value']
                    odds_results.append({
                        "match": match_name,
                        "cuota": float(h_odds)
                    })
                except: continue
        return odds_results
    except Exception as e:
        print(f"Error conectando a Stake: {e}")
        return []

def predict_and_save(match_data):
    """Aplica Poisson y guarda en el historial"""
    h_xg, a_xg = 1.8, 1.2 # xG base
    cuota = match_data['cuota']
    
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    
    apuesta = 0
    if ev > 0.07:
        f_star = ((cuota - 1) * prob_l - (1 - prob_l)) / (cuota - 1)
        apuesta = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
        
    return {
        "match": match_data['match'],
        "prob": prob_l,
        "cuota": cuota,
        "apuesta": max(0, apuesta),
        "ev": ev
    }

if __name__ == "__main__":
    print("🛰️ Conectando directamente a Stake.com...")
    partidos_stake = get_direct_stake_odds()
    
    final_picks = []
    for p in partidos_stake[:8]: # Analizamos los primeros 8
        res = predict_and_save(p)
        if res['apuesta'] > 0:
            final_picks.append(res)
            # Guardar en CSV
            with open("data/predictions_history.csv", "a", newline="") as f:
                csv.writer(f).writerow([datetime.now().date(), res['match'], "Stake Direct", "Local", res['prob'], res['cuota'], res['apuesta'], "PENDIENTE"])

    if final_picks:
        # (Aquí llamas a tu función de generar imagen que ya tienes)
        print(f"✅ {len(final_picks)} oportunidades encontradas en Stake.")
    else:
        print("❌ No hay apuestas con valor en Stake ahora mismo.")
