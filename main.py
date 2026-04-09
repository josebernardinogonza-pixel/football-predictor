import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
API_KEY = os.getenv("ALL_SPORTS_API_KEY")
BASE_URL = "https://apiv2.allsportsapi.com/football/"
BANKROLL_MXN = 4875.00
KELLY_FRACTION = 0.20

def initialize_files():
    """Crea los archivos necesarios para que el Workflow no de error"""
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write("<b>📊 ESCÁNER ALL-SPORTS-API</b>\nBuscando partidos en vivo...")
    
    # Imagen de respaldo
    img = Image.new('RGB', (800, 450), color=(15, 15, 15))
    img.save("prediction_card.png")

def get_live_data():
    """Obtiene datos reales de AllSportsApi.com"""
    params = {
        'met': 'Livescore',
        'APIkey': API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=15).json()
        if "result" in response:
            return response["result"]
        return []
    except Exception as e:
        print(f"Error de conexión con AllSportsApi: {e}")
        return []

def generate_card(game):
    """Genera la imagen usando los datos de la API"""
    img = Image.new('RGB', (800, 450), color=(10, 10, 30))
    draw = ImageDraw.Draw(img)
    
    # Descargar logos reales de la API
    def load_logo(url):
        try:
            res = requests.get(url, timeout=5)
            return Image.open(BytesIO(res.content)).convert("RGBA").resize((130, 130))
        except:
            return Image.new('RGBA', (130, 130), color=(40, 40, 40))

    logo_h = load_logo(game.get('home_team_logo'))
    logo_a = load_logo(game.get('away_team_logo'))
    img.paste(logo_h, (80, 100), logo_h)
    img.paste(logo_a, (590, 100), logo_a)

    # Marcador y Nombres
    draw.text((400, 40), f"EN VIVO: {game.get('event_status')}'", fill="red", anchor="mm")
    draw.text((400, 150), game.get('event_final_result'), fill="white", anchor="mm")
    draw.text((145, 250), game.get('event_home_team')[:12], fill="white", anchor="mm")
    draw.text((655, 250), game.get('event_away_team')[:12], fill="white", anchor="mm")
    
    # Predicción (Poisson simplificado)
    draw.text((400, 350), "ANÁLISIS DE VALOR EN CURSO", fill="gold", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Asegurar archivos para el Workflow
    initialize_files()
    
    # 2. Consultar AllSportsApi
    live_matches = get_live_data()
    
    if live_matches:
        # Tomamos el primer partido disponible (puedes filtrar por liga)
        match = live_matches[0]
        
        # Actualizar reporte de texto
        with open("audit_report.txt", "w", encoding="utf-8") as f:
            f.write(f"<b>✅ PARTIDO DETECTADO</b>\n{match['event_home_team']} vs {match['event_away_team']}\nMarcador: {match['event_final_result']}")
        
        # Generar imagen con logos de la API
        generate_card(match)
        
        # Guardar en historial CSV
        with open("predictions_history.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), f"{match['event_home_team']} vs {match['event_away_team']}", "LIVE", "API_DATA", "0"])
            
        print(f"Procesado: {match['event_home_team']}")
    else:
        print("No hay partidos en vivo en AllSportsApi ahora mismo.")
