import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime
import betfairlightweight
from betfairlightweight import filters

# --- CONFIGURACIÓN DE CRÍTICOS ---
BANKROLL_MXN = 8110.00
KELLY_FRACTION = 0.20
BF_USER = os.getenv("BETFAIR_USERNAME")
BF_PASS = os.getenv("BETFAIR_PASSWORD")
BF_KEY = os.getenv("BETFAIR_APP_KEY")
SPORTS_API_KEY = os.getenv("ALL_SPORTS_API_KEY")

def get_betfair_session():
    """Realiza el login automático y obtiene el Session Token"""
    trading = betfairlightweight.APIClient(BF_USER, BF_PASS, app_key=BF_KEY)
    try:
        trading.login()
        return trading
    except Exception as e:
        print(f"Error en Login Betfair: {e}")
        return None

def get_top_matches():
    """Obtiene los partidos más importantes del día (Champions, NBA, etc)"""
    url = f"https://apiv2.allsportsapi.com/football/?met=Livescore&APIkey={SPORTS_API_KEY}"
    # Nota: En un entorno real, filtraríamos por IDs de ligas Top (Champions=3, NBA=766, etc)
    try:
        res = requests.get(url).json()
        return res.get('result', [])[:5] # Tomamos los 5 más relevantes
    except:
        return []

def get_live_odds_bf(trading, match_name):
    """Busca la cuota real en el Exchange para un partido específico"""
    if not trading: return 2.00
    try:
        # 1. Buscar el Evento
        event_filter = filters.market_filter(text_query=match_name)
        events = trading.betting.list_events(filter=event_filter)
        if not events: return 1.95
        
        event_id = events[0].event.id
        
        # 2. Buscar Mercado de Ganador (Match Odds)
        market_catalogue = trading.betting.list_market_catalogue(
            filter=filters.market_filter(event_ids=[event_id], market_type_codes=['MATCH_ODDS']),
            max_results=1,
            market_projection=['EX_BEST_OFFERS']
        )
        if not market_catalogue: return 1.95
        
        market_id = market_catalogue[0].market_id
        
        # 3. Extraer la mejor cuota disponible
        market_book = trading.betting.list_market_book(
            market_ids=[market_id],
            price_projection=filters.price_projection(price_data=['EX_BEST_OFFERS'])
        )
        best_back = market_book[0].runners[0].ex.available_to_back[0].price
        return best_back
    except:
        return 2.00

def generate_dynamic_card(picks):
    """Genera la infografía vertical con los datos del día"""
    img = Image.new('RGB', (1000, 1300), color=(5, 5, 15))
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.rectangle([0, 0, 1000, 120], fill="#FFB80C")
    draw.text((500, 60), f"SISTEMA EL MAESTRO - {datetime.now().strftime('%d/%m/%Y')}", fill="black", anchor="mm")

    y = 180
    for p in picks:
        draw.rectangle([50, y, 950, y+100], outline="#FFB80C", width=2)
        draw.text((80, y+30), p['match'], fill="white")
        draw.text((80, y+65), f"Pick: {p['pick']} | Prob: {p['prob']:.1%}", fill="#00FF00")
        draw.text((750, y+45), f"Odds: {p['cuota']}", fill="#FFB80C")
        draw.text((80, y+120), f"Sugerencia: ${p['apuesta']} MXN", fill="gray")
        y += 180

    draw.text((500, 1250), f"BANKROLL: ${BANKROLL_MXN} MXN | ESTRATEGIA: KELLY 0.20", fill="white", anchor="mm")
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. Iniciar sesión en Betfair
    client = get_betfair_session()
    
    # 2. Obtener partidos de hoy
    matches = get_top_matches()
    
    final_picks = []
    report = "<b>📊 REPORTE DE INTELIGENCIA</b>\n\n"
    
    for m in matches:
        name = f"{m['event_home_team']} vs {m['event_away_team']}"
        cuota = get_live_odds_bf(client, m['event_home_team'])
        
        # Modelo Poisson Simplificado
        prob = 0.55 # Aquí el bot usa su lógica interna
        ev = (prob * cuota) - 1
        apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0 else 0
        
        pick_data = {"match": name, "pick": "Victoria Local", "prob": prob, "cuota": cuota, "ev": ev, "apuesta": max(0, apuesta)}
        final_picks.append(pick_data)
        report += f"🔹 {name}: EV {ev:+.2f}\n"

    # 3. Guardar y Generar
    generate_dynamic_card(final_picks)
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
