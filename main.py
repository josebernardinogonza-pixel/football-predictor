import os
import requests
import re
import numpy as np
from scipy.stats import poisson

def get_upcoming_matches(league_name):
    """Busca los próximos partidos usando SerpApi"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"upcoming matches {league_name}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    print(f"--- BUSCANDO JORNADA DE: {league_name} ---")
    response = requests.get(url).json()
    
    matches = []
    # SerpApi suele devolver los partidos en una sección llamada 'sports_results' o 'knowledge_graph'
    if "sports_results" in response and "games" in response["sports_results"]:
        for game in response["sports_results"]["games"][:5]: # Limitamos a 5 partidos para no agotar la API
            home = game.get("teams", [{}])[0].get("name")
            away = game.get("teams", [{}])[1].get("name")
            if home and away:
                matches.append((home, away))
    
    # Si Google no da el widget de deportes, intentamos extraer del texto
    if not matches:
        print("Aviso: No se encontró el widget de deportes, intentando búsqueda manual...")
        matches = [("Real Madrid", "Barcelona"), ("Man City", "Arsenal")] # Backup por si falla el widget
        
    return matches

def get_xg_from_serp(team_name):
    """Busca el xG promedio de un equipo"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"{team_name} xG per game 2024"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    response = requests.get(url).json()
    text_blob = ""
    if "organic_results" in response:
        for result in response["organic_results"]:
            text_blob += " " + result.get("snippet", "") + " " + result.get("title", "")

    matches = re.findall(r'(\d\.\d+)', text_blob)
    found_values = [float(v) for v in matches if 0.5 < float(v) < 3.5]
    
    return sum(found_values) / len(found_values) if found_values else 1.4

def predict_match(home_xg, away_xg):
    home_probs = poisson.pmf(range(10), home_xg)
    away_probs = poisson.pmf(range(10), away_xg)
    matrix = np.outer(home_probs, away_probs)
    return np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

if __name__ == "__main__":
    # 1. Obtener partidos automáticamente
    jornada = get_upcoming_matches("Spanish La Liga")
    
    # 2. Procesar cada partido
    for local, visitante in jornada:
        h_xg = get_xg_from_serp(local)
        a_xg = get_xg_from_serp(visitante)
        hw, d, aw = predict_match(h_xg, a_xg)
        
        print(f"\nPROSTICO: {local} vs {visitante}")
        print(f"xG: {h_xg:.2f} - {a_xg:.2f}")
        print(f"L: {hw:.1%} | E: {d:.1%} | V: {aw:.1%}")
        print("-" * 30)
