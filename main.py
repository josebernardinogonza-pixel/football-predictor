import os
import requests
import re
import numpy as np
from scipy.stats import poisson

# --- CONFIGURACIÓN ---
HOME_ADVANTAGE = 1.10  # Aumenta 10% al local
AWAY_PENALTY = 0.90    # Reduce 10% al visitante

def get_stats_from_serp(team_name):
    """Busca xG a favor y xG en contra (defensa)"""
    api_key = os.getenv("SERPAPI_KEY")
    # Buscamos xG anotado y xG concedido (conceded)
    query = f"{team_name} xG stats 2024 per game"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    response = requests.get(url).json()
    text_blob = ""
    if "organic_results" in response:
        for result in response["organic_results"]:
            text_blob += " " + result.get("snippet", "") + " " + result.get("title", "")

    # Extraer todos los números decimales
    numbers = [float(v) for v in re.findall(r'(\d\.\d+)', text_blob) if 0.4 < float(v) < 3.0]
    
    if len(numbers) >= 2:
        attack_xg = numbers[0]  # El primero suele ser ataque
        defense_xg = numbers[1] # El segundo suele ser defensa
        return attack_xg, defense_xg
    return 1.3, 1.3 # Valores neutros si falla

def get_market_odds(home_team, away_team):
    """Busca las cuotas (odds) reales en Google"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"odds {home_team} vs {away_team}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url).json()
    
    # Intentamos buscar números mayores a 1.20 que parezcan cuotas
    text = str(response.get("organic_results", ""))
    odds = re.findall(r'(\d\.\d{2})', text)
    
    # Si encontramos cuotas, devolvemos las 3 primeras (Local, Empate, Visitante)
    if len(odds) >= 3:
        return [float(o) for o in odds[:3]]
    return [2.0, 3.2, 3.5] # Cuotas ficticias si no encuentra

def predict_advanced(h_team, a_team):
    # Obtener fuerza de ataque y defensa
    h_att, h_def = get_stats_from_serp(h_team)
    a_att, a_def = get_stats_from_serp(a_team)
    
    # Proyectar xG del partido: (Ataque Local * Defensa Visitante)
    # Aplicamos factor de localía
    projected_h_xg = (h_att * a_def) * HOME_ADVANTAGE
    projected_a_xg = (a_att * h_def) * AWAY_PENALTY
    
    # Distribución de Poisson
    h_probs = poisson.pmf(range(10), projected_h_xg)
    a_probs = poisson.pmf(range(10), projected_a_xg)
    matrix = np.outer(h_probs, a_probs)
    
    prob_l = np.sum(np.tril(matrix, -1))
    prob_e = np.sum(np.diag(matrix))
    prob_v = np.sum(np.triu(matrix, 1))
    
    return prob_l, prob_e, prob_v, projected_h_xg, projected_a_xg

if __name__ == "__main__":
    # Lista de partidos a analizar
    partidos = [("Real Madrid", "Atletico Madrid"), ("Liverpool", "Chelsea")]
    
    for local, visitante in partidos:
        pl, pe, pv, xgh, xga = predict_advanced(local, visitante)
        odds = get_market_odds(local, visitante)
        
        # Calcular Valor Esperado (EV)
        # EV = (Probabilidad * Cuota) - 1
        ev_l = (pl * odds[0]) - 1
        
        print(f"\nANÁLISIS: {local} vs {visitante}")
        print(f"xG Proyectado: {xgh:.2f} - {xga:.2f}")
        print(f"Probabilidades: L:{pl:.1%} | E:{pe:.1%} | V:{pv:.1%}")
        print(f"Cuotas Mercado: L:{odds[0]} | E:{odds[1]} | V:{odds[2]}")
        
        if ev_l > 0.05: # Si el valor es mayor al 5%
            print(f"¡ALERTA DE VALOR! -> Apostar a {local} (EV: {ev_l:+.2f})")
        else:
            print("Sin valor claro en este mercado.")
        print("-" * 40)
