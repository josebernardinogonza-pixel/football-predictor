import os
import requests
import re
import numpy as np
from scipy.stats import poisson

# --- CONFIGURACIÓN AVANZADA ---
HOME_ADVANTAGE = 1.10
AWAY_PENALTY = 0.90
NEWS_PENALTY = 0.85  # Reduce el ataque un 15% si hay bajas importantes

def get_news_impact(team_name):
    """Analiza noticias de última hora para detectar bajas"""
    api_key = os.getenv("SERPAPI_KEY")
    # Buscamos en Google News las últimas noticias de lesiones/bajas
    query = f"{team_name} team news injuries out"
    url = f"https://serpapi.com/search.json?engine=google_news&q={query}&api_key={api_key}"
    
    print(f"Analizando noticias en tiempo real para: {team_name}...")
    response = requests.get(url).json()
    
    bad_news_keywords = ["injury", "out for", "suspended", "doubtful", "missing", "absent", "hamstring", "acl"]
    impact_score = 1.0
    news_count = 0

    if "news_results" in response:
        for article in response["news_results"][:5]: # Analizar los 5 titulares más recientes
            title = article.get("title", "").lower()
            snippet = article.get("snippet", "").lower()
            
            # Si encontramos palabras de bajas, sumamos al contador
            for word in bad_news_keywords:
                if word in title or word in snippet:
                    news_count += 1
                    break 

    # Si hay más de 2 noticias sobre bajas, aplicamos el penalty
    if news_count >= 2:
        print(f"⚠️ Alerta: Detectadas posibles bajas en {team_name}. Aplicando penalty de xG.")
        impact_score = NEWS_PENALTY
        
    return impact_score

def get_stats_from_serp(team_name):
    """Obtiene xG base (Ataque y Defensa)"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"{team_name} xG stats 2024 per game"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url).json()
    text = str(response.get("organic_results", ""))
    numbers = [float(v) for v in re.findall(r'(\d\.\d+)', text) if 0.4 < float(v) < 3.0]
    
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    return 1.3, 1.3

def predict_real_time(h_team, a_team):
    # 1. Stats base
    h_att, h_def = get_stats_from_serp(h_team)
    a_att, a_def = get_stats_from_serp(a_team)
    
    # 2. Impacto de noticias en tiempo real
    h_news_mod = get_news_impact(h_team)
    a_news_mod = get_news_impact(a_team)
    
    # 3. Cálculo final de xG proyectado con todos los factores
    # (Ataque * Defensa_Rival) * Localía * Noticias
    final_h_xg = (h_att * a_def) * HOME_ADVANTAGE * h_news_mod
    final_a_xg = (a_att * h_def) * AWAY_PENALTY * a_news_mod
    
    # 4. Poisson
    h_probs = poisson.pmf(range(10), final_h_xg)
    a_probs = poisson.pmf(range(10), final_a_xg)
    matrix = np.outer(h_probs, a_probs)
    
    return np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1)), final_h_xg, final_a_xg

if __name__ == "__main__":
    # Ejemplo con un partido real actual
    local = "Manchester City"
    visitante = "Real Madrid"
    
    pl, pe, pv, xgh, xga = predict_real_time(local, visitante)
    
    print(f"\n--- PREDICCIÓN EN TIEMPO REAL ---")
    print(f"Partido: {local} vs {visitante}")
    print(f"xG Final (Ajustado): {xgh:.2f} - {xga:.2f}")
    print(f"Probabilidades: L:{pl:.1%} | E:{pe:.1%} | V:{pv:.1%}")
    
    # Recomendación basada en xG ajustado
    if xgh > xga + 0.5:
        print(f"Recomendación: Victoria clara de {local}")
    elif xga > xgh + 0.5:
        print(f"Recomendación: Victoria clara de {visitante}")
    else:
        print("Recomendación: Partido muy igualado o Empate")

import csv
from datetime import datetime

# ... (Mantén todas tus funciones anteriores: predict_real_time, get_stats, etc.) ...

def save_to_csv(home, away, prob_l, prob_e, prob_v, xgh, xga):
    file_name = "predictions_history.csv"
    file_exists = os.path.isfile(file_name)
    
    # Datos a guardar
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    nueva_fila = [fecha, home, away, f"{prob_l:.2%}", f"{prob_e:.2%}", f"{prob_v:.2%}", f"{xgh:.2f}", f"{xga:.2f}"]

    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Si el archivo es nuevo, escribimos la cabecera
        if not file_exists:
            writer.writerow(["Fecha", "Local", "Visitante", "Prob_L", "Prob_E", "Prob_V", "xG_H", "xG_A"])
        writer.writerow(nueva_fila)
    print(f"✅ Predicción guardada en {file_name}")

if __name__ == "__main__":
    # Ejemplo de ejecución
    local = "Manchester City"
    visitante = "Real Madrid"
    
    pl, pe, pv, xgh, xga = predict_real_time(local, visitante)
    
    # GUARDAR RESULTADOS
    save_to_csv(local, visitante, pl, pe, pv, xgh, xga)
    
    print(f"Análisis completado para {local} vs {visitante}")
