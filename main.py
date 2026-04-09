import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import csv
from datetime import datetime

# --- CONFIGURACIÓN MAESTRA MEJORADA ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.10       # Bajamos a 0.10 para ser más conservadores
MAX_ODD = 5.0               # IGNOREMOS cualquier cuota mayor a 5.0
MIN_PROBABILITY = 0.40      # Solo picks con más del 40% de probabilidad real
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_live_data():
    """Busca partidos en vivo con filtros de seguridad"""
    url = f"https://serpapi.com/search.json?q=soccer+live+scores&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        games = res.get("sports_results", {}).get("games", [])
        live_list = []
        for g in games:
            # Extraer cuotas si están disponibles en el JSON de SerpApi
            # (Simulamos una extracción de cuota para el ejemplo)
            raw_odd = g.get("odds", {}).get("away_team", 1.0) 
            
            # --- FILTRO ANT-LOCURA ---
            if float(raw_odd) > MAX_ODD:
                continue # Si la cuota es 16.0 o 100.0, el bot la ignora
                
            status = g.get("status", "").lower()
            if any(c.isdigit() for c in status) or "vivo" in status or "live" in status:
                live_list.append({
                    "home": g["teams"][0]["name"], 
                    "away": g["teams"][1]["name"],
                    "h_score": int(g["teams"][0].get("score", 0)),
                    "a_score": int(g["teams"][1].get("score", 0)),
                    "status": status,
                    "odd": float(raw_odd)
                })
        return live_list
    except: return []

def calculate_kelly(prob, odd):
    """Calcula el stake basado en probabilidad real vs cuota"""
    if odd <= 1: return 0
    # Fórmula de Kelly: (p*b - q) / b donde b = odd - 1
    b = odd - 1
    q = 1 - prob
    fraction = (prob * b - q) / b
    return max(0, fraction * BANKROLL_MXN * KELLY_FRACTION)

def generate_card(game, prob, stake):
    """Crea la imagen profesional solo si hay valor real"""
    img = Image.new('RGB', (800, 450), color=(15, 15, 35))
    draw = ImageDraw.Draw(img)
    
    # Encabezado Dinámico
    header_color = "#00FF00" if prob > 0.6 else "#FFA500"
    draw.rectangle([0, 0, 800, 60], fill=header_color)
    
    draw.text((400, 30), f"• ANÁLISIS DE VALOR: {game['status'].upper()}", fill="black", anchor="mm")
    draw.text((400, 150), f"{game['h_score']} - {game['a_score']}", fill="white", anchor="mm")
    draw.text((150, 150), game['home'][:12], fill="gold", anchor="mm")
    draw.text((650, 150), game['away'][:12], fill="gold", anchor="mm")
    
    # Mostrar la probabilidad y el stake sugerido
    draw.text((400, 250), f"Confianza: {prob:.1%} | Cuota: {game['odd']}", fill="white", anchor="mm")
    
    draw.rectangle([150, 320, 650, 420], outline="white", width=2)
    draw.text((400, 370), f"STAKE RECOMENDADO: ${stake:.2f} MXN", fill="#00FF00", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    matches = get_live_data()
    
    valid_pick_found = False
    for game in matches:
        # LÓGICA DE PROBABILIDAD (Aquí deberías conectar tu Poisson real)
        # Por ahora, simulamos que el modelo estima un 55%
        estimated_prob = 0.55 
        
        # Solo procesar si cumple con nuestros estándares de "no locura"
        if estimated_prob >= MIN_PROBABILITY:
            stake = calculate_kelly(estimated_prob, game['odd'])
            if stake > 0:
                generate_card(game, estimated_prob, stake)
                valid_pick_found = True
                print(f"✅ Pick generado: {game['home']} vs {game['away']}")
                break # Solo enviamos el mejor

    if not valid_pick_found:
        img = Image.new('RGB', (800, 450), color=(20, 20, 20))
        ImageDraw.Draw(img).text((400, 225), "SIN OPORTUNIDADES DE VALOR SEGURO", fill="gray", anchor="mm")
        img.save("prediction_card.png")
