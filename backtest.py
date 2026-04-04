import os
import requests
import re
import numpy as np
from scipy.stats import poisson

# --- CONFIGURACIÓN DEL BACKTEST ---
STAKE_POR_APUESTA = 10  # Euros/Unidades por apuesta
EV_MINIMO = 0.05        # Solo apostamos si el valor es > 5%

def get_past_results(league):
    """Busca resultados reales de la última jornada"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"results {league} last week"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    response = requests.get(url).json()
    results = []
    
    # Extraemos partidos terminados y sus marcadores
    if "sports_results" in response and "games" in response["sports_results"]:
        for game in response["sports_results"]["games"]:
            if game.get("status") == "Final":
                home = game["teams"][0]["name"]
                away = game["teams"][1]["name"]
                score_h = int(game["teams"][0]["score"])
                score_a = int(game["teams"][1]["score"])
                
                # Determinar resultado real (H=Home, D=Draw, A=Away)
                res = "H" if score_h > score_a else "A" if score_a > score_h else "D"
                results.append({"home": home, "away": away, "res": res, "score": f"{score_h}-{score_a}"})
    
    return results

def predict_logic(h_xg, a_xg):
    """Misma lógica de Poisson que el modelo principal"""
    h_probs = poisson.pmf(range(10), h_xg)
    a_probs = poisson.pmf(range(10), a_xg)
    matrix = np.outer(h_probs, a_probs)
    return np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

def run_backtest():
    # 1. Obtener partidos pasados
    past_matches = get_past_results("Premier League")
    
    total_invested = 0
    total_return = 0
    bets_made = 0

    print(f"--- INICIANDO BACKTESTING ---")
    
    for match in past_matches[:10]: # Analizamos los últimos 10
        # Simulamos xG histórico (en un modelo pro, esto vendría de una base de datos)
        # Aquí usamos 1.5 como base para el ejemplo
        h_xg, a_xg = 1.6, 1.2 
        prob_l, prob_e, prob_v = predict_logic(h_xg, a_xg)
        
        # Cuota ficticia de la casa de apuestas (ej: 2.10 para el local)
        odds_l = 2.10 
        ev_l = (prob_l * odds_l) - 1
        
        if ev_l > EV_MINIMO:
            bets_made += 1
            total_invested += STAKE_POR_APUESTA
            print(f"Apuesta realizada: {match['home']} (Cuota {odds_l})")
            
            # ¿Ganamos la apuesta?
            if match['res'] == "H":
                profit = STAKE_POR_APUESTA * odds_l
                total_return += profit
                print(f"✅ GANADA: {match['score']} (+{profit - STAKE_POR_APUESTA:.2f}€)")
            else:
                print(f"❌ PERDIDA: {match['score']} (-{STAKE_POR_APUESTA}€)")
        
    # --- RESUMEN FINAL ---
    roi = ((total_return - total_invested) / total_invested * 100) if total_invested > 0 else 0
    print(f"\n--- RESUMEN DEL BACKTEST ---")
    print(f"Partidos analizados: {len(past_matches)}")
    print(f"Apuestas realizadas: {bets_made}")
    print(f"Inversión Total: {total_invested}€")
    print(f"Retorno Total: {total_return:.2f}€")
    print(f"Beneficio Neto: {total_return - total_invested:.2f}€")
    print(f"ROI: {roi:.2%}")

if __name__ == "__main__":
    run_backtest()
