import os
import requests
import numpy as np
from scipy.stats import poisson

# --- TU LÓGICA DE POISSON ---
def predict_match(home_xg, away_xg):
    home_probs = poisson.pmf(range(10), home_xg)
    away_probs = poisson.pmf(range(10), away_xg)
    matrix = np.outer(home_probs, away_probs)
    
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    return home_win, draw, away_win

# --- OBTENER DATOS DE SERPAPI ---
def get_match_data(query):
    api_key = os.getenv("SERPAPI_KEY")
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url).json()
    
    # Aquí podrías extraer xG real si aparece en los snippets de Google
    # Por ahora, simularemos que extrae un xG basado en resultados recientes
    print(f"Buscando datos para: {query}...")
    return 1.8, 1.2  # Valores de ejemplo (Home xG, Away xG)

if __name__ == "__main__":
    h_xg, a_xg = get_match_data("Manchester City vs Arsenal xG stats")
    hw, d, aw = predict_match(h_xg, a_xg)
    
    print(f"\nPredicción Final:")
    print(f"Local: {hw:.2%}")
    print(f"Empate: {d:.2%}")
    print(f"Visitante: {aw:.2%}")
