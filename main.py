import os
import requests
import re
import numpy as np
from scipy.stats import poisson
import csv
from datetime import datetime

# ==========================================
# CONFIGURACIÓN GLOBAL (AJUSTA A TU GUSTO)
# ==========================================
BANKROLL_MXN = 5000.00  # Tu presupuesto total en Pesos Mexicanos
KELLY_FRACTION = 0.25   # Riesgo conservador (25% de lo sugerido)
HOME_ADVANTAGE = 1.10   # +10% al local
AWAY_PENALTY = 0.90     # -10% al visitante
NEWS_PENALTY = 0.85     # -15% al ataque si hay bajas importantes
LEAGUE_TO_SCAN = "Liga MX" # Puedes cambiar a "Premier League", etc.

# ==========================================
# FUNCIONES DE OBTENCIÓN DE DATOS (SERPAPI)
# ==========================================

def get_upcoming_matches(league):
    """Busca los próximos partidos de la jornada"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"proximos partidos {league}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    try:
        response = requests.get(url).json()
        matches = []
        if "sports_results" in response and "games" in response["sports_results"]:
            for game in response["sports_results"]["games"]:
                if game.get("status") != "Final":
                    home = game["teams"][0]["name"]
                    away = game["teams"][1]["name"]
                    matches.append((home, away))
        
        # Backup si falla el widget de Google
        if not matches:
            matches = [("Club America", "Chivas"), ("Cruz Azul", "Pumas"), ("Tigres", "Monterrey")]
        return matches[:6] # Analizar 6 partidos para cuidar créditos de API
    except:
        return [("Club America", "Chivas")]

def get_stats_from_serp(team_name):
    """Obtiene xG de ataque y defensa"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"{team_name} xG stats 2024 per game"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    try:
        response = requests.get(url).json()
        text = str(response.get("organic_results", ""))
        numbers = [float(v) for v in re.findall(r'(\d\.\d+)', text) if 0.4 < float(v) < 3.5]
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        return 1.4, 1.4 # Valores neutros por defecto
    except:
        return 1.4, 1.4

def get_news_impact(team_name):
    """Analiza noticias de lesiones en tiempo real"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"{team_name} team news injuries out"
    url = f"https://serpapi.com/search.json?engine=google_news&q={query}&api_key={api_key}"
    
    try:
        response = requests.get(url).json()
        bad_news_keywords = ["injury", "out for", "suspended", "doubtful", "missing", "absent", "baja"]
        news_count = 0
        if "news_results" in response:
            for article in response["news_results"][:5]:
                text = (article.get("title", "") + article.get("snippet", "")).lower()
                if any(word in text for word in bad_news_keywords):
                    news_count += 1
        return NEWS_PENALTY if news_count >= 2 else 1.0
    except:
        return 1.0

def get_market_odds(home, away):
    """Busca cuotas reales en Google"""
    api_key = os.getenv("SERPAPI_KEY")
    query = f"odds {home} vs {away}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    
    try:
        response = requests.get(url).json()
        text = str(response.get("organic_results", ""))
        odds = re.findall(r'(\d\.\d{2})', text)
        if len(odds) >= 3:
            return [float(o) for o in odds[:3]]
        return [2.10, 3.20, 3.40] # Cuotas base si no encuentra
    except:
        return [2.00, 3.00, 3.50]

# ==========================================
# NÚCLEO MATEMÁTICO Y FINANCIERO
# ==========================================

def predict_match(home, away):
    """Pipeline completo de predicción para un partido"""
    # 1. Obtener Datos
    h_att, h_def = get_stats_from_serp(home)
    a_att, a_def = get_stats_from_serp(away)
    h_mod = get_news_impact(home)
    a_mod = get_news_impact(away)
    
    # 2. xG Proyectado Ajustado
    final_h_xg = (h_att * a_def) * HOME_ADVANTAGE * h_mod
    final_a_xg = (a_att * h_def) * AWAY_PENALTY * a_mod
    
    # 3. Poisson
    h_probs = poisson.pmf(range(10), final_h_xg)
    a_probs = poisson.pmf(range(10), final_a_xg)
    matrix = np.outer(h_probs, a_probs)
    
    prob_l = np.sum(np.tril(matrix, -1))
    prob_e = np.sum(np.diag(matrix))
    prob_v = np.sum(np.triu(matrix, 1))
    
    # 4. Cuotas y Valor (EV)
    odds = get_market_odds(home, away)
    cuota_l = odds[0]
    ev_l = (prob_l * cuota_l) - 1
    
    # 5. Kelly Criterion en MXN
    apuesta_mxn = 0
    if ev_l > 0:
        f_star = ((cuota_l - 1) * prob_l - (1 - prob_l)) / (cuota_l - 1)
        apuesta_mxn = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
        
    return {
        "local": home, "visitante": away,
        "prob_l": prob_l, "prob_e": prob_e, "prob_v": prob_v,
        "xgh": final_h_xg, "xga": final_a_xg,
        "cuota": cuota_l, "ev": ev_l, "apuesta": max(0, apuesta_mxn)
    }

def save_to_csv(res):
    """Guarda el resultado en el historial CSV"""
    file_name = "predictions_history.csv"
    file_exists = os.path.isfile(file_name)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha", "Partido", "Prob_L", "Prob_E", "Prob_V", "xG_H", "xG_A", "Cuota", "EV", "Apuesta_MXN"])
        
        writer.writerow([
            fecha, f"{res['local']} vs {res['visitante']}",
            f"{res['prob_l']:.1%}", f"{res['prob_e']:.1%}", f"{res['prob_v']:.1%}",
            f"{res['xgh']:.2f}", f"{res['xga']:.2f}",
            res['cuota'], f"{res['ev']:+.2f}", f"${res['apuesta']} MXN"
        ])

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    print(f"--- ESCÁNER PROFESIONAL: {LEAGUE_TO_SCAN} ---")
    jornada = get_upcoming_matches(LEAGUE_TO_SCAN)
    
    resultados_jornada = []
    for h, a in jornada:
        print(f"Analizando: {h} vs {a}...")
        res = predict_match(h, a)
        save_to_csv(res)
        resultados_jornada.append(res)
    
    # Mostrar Top 3 en consola
    top_picks = sorted(resultados_jornada, key=lambda x: x['ev'], reverse=True)
    
    print("\n" + "="*45)
    print("🏆 TOP APUESTAS SUGERIDAS (MXN) 🏆")
    print("="*45)
    for i, p in enumerate(top_picks[:3], 1):
        if p['ev'] > 0:
            print(f"{i}. {p['local']} vs {p['visitante']}")
            print(f"   Valor (EV): {p['ev']:+.2f} | Cuota: {p['cuota']}")
            print(f"   💰 APUESTA: ${p['apuesta']} MXN")
            print("-" * 35)
    
    print("\n✅ Proceso finalizado. Historial actualizado en GitHub.")
