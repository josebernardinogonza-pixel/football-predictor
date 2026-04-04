import os
import requests
import re
import numpy as np
from scipy.stats import poisson

def get_xg_from_serp(team_name):
    """
    Busca en Google el xG promedio del equipo usando SerpApi
    """
    api_key = os.getenv("SERPAPI_KEY")
    query = f"{team_name} xG per game 2024 2025"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    print(f"Buscando estadísticas para: {team_name}...")
    response = requests.get(url).json()
    
    # Unimos todos los textos que Google encontró (títulos y descripciones)
    text_blob = ""
    if "organic_results" in response:
        for result in response["organic_results"]:
            text_blob += " " + result.get("snippet", "") + " " + result.get("title", "")

    # Buscamos números decimales (ej: 1.75) que estén cerca de la palabra 'xG'
    # Esta regex busca un número después o antes de la mención de xG
    matches = re.findall(r'xG.*?(\d\.\d+)|(\d\.\d+).*?xG', text_blob, re.IGNORECASE)
    
    found_values = []
    for m in matches:
        for val in m:
            if val: found_values.append(float(val))
    
    if found_values:
        avg_xg = sum(found_values) / len(found_values)
        # Limitamos el xG para que sea realista (entre 0.5 y 3.0)
        return max(0.5, min(3.0, avg_xg))
    
    print(f"No se encontró xG claro para {team_name}, usando promedio por defecto.")
    return 1.5 # Valor por defecto si falla la búsqueda

def predict_match(home_xg, away_xg):
    home_probs = poisson.pmf(range(10), home_xg)
    away_probs = poisson.pmf(range(10), away_xg)
    matrix = np.outer(home_probs, away_probs)
    
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    return home_win, draw, away_win

if __name__ == "__main__":
    # CONFIGURA AQUÍ LOS EQUIPOS
    equipo_local = "Real Madrid"
    equipo_visitante = "Barcelona"
    
    h_xg = get_xg_from_serp(equipo_local)
    a_xg = get_xg_from_serp(equipo_visitante)
    
    hw, d, aw = predict_match(h_xg, a_xg)
    
    print(f"\n--- RESULTADOS PARA: {equipo_local} vs {equipo_visitante} ---")
    print(f"xG Proyectado {equipo_local}: {h_xg:.2f}")
    print(f"xG Proyectado {equipo_visitante}: {a_xg:.2f}")
    print(f"--------------------------------------------")
    print(f"Probabilidad Victoria Local: {hw:.2%}")
    print(f"Probabilidad Empate: {d:.2%}")
    print(f"Probabilidad Victoria Visitante: {aw:.2%}")
