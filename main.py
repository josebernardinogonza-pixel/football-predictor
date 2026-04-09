import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
from datetime import datetime
import betfairlightweight
from betfairlightweight import filters

# --- CONFIGURACIÓN DE ÉLITE ---
BANKROLL_MXN = 8110.00
KELLY_FRACTION = 0.20
APP_KEY = os.getenv("BETFAIR_APP_KEY")
USERNAME = os.getenv("BETFAIR_USERNAME")
PASSWORD = os.getenv("BETFAIR_PASSWORD")

def get_betfair_price(event_name):
    """Conecta a Betfair para obtener la cuota real y liquidez"""
    try:
        trading = betfairlightweight.APIClient(USERNAME, PASSWORD, app_key=APP_KEY)
        trading.login()
        
        # Buscar el evento
        event_filter = filters.market_filter(text_query=event_name)
        events = trading.betting.list_events(filter=event_filter)
        
        if not events: return 2.00 # Cuota base si no hay mercado
        
        event_id = events[0].event.id
        market_catalogue = trading.betting.list_market_catalogue(
            filter=filters.market_filter(event_ids=[event_id], market_type_codes=['MATCH_ODDS']),
            max_results=1
        )
        
        if not market_catalogue: return 2.00
        
        market_id = market_catalogue[0].market_id
        market_book = trading.betting.list_market_book(market_ids=[market_id])[0]
        
        # Obtener la mejor cuota de 'Back' (A favor)
        best_odds = market_book.runners[0].ex.available_to_back[0].price
        return best_odds
    except:
        return 1.95 # Fallback de seguridad

def predict_nba_efficiency(h_rtg, a_rtg, cuota):
    """Modelo de Eficiencia NBA"""
    diff = h_rtg - a_rtg
    prob = 1 / (1 + np.exp(-diff/12))
    ev = (prob * cuota) - 1
    apuesta = round(ev * KELLY_FRACTION * BANKROLL_MXN, 2) if ev > 0.05 else 0
    return prob, ev, max(0, apuesta)

def generate_betfair_card(picks):
    """Genera la infografía con datos de Betfair"""
    img = Image.new('RGB', (1000, 1000), color=(5, 5, 15))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 100], fill="#FFB80C") # Amarillo Betfair
    draw.text((500, 50), "BETFAIR EXCHANGE INTELLIGENCE - 09/04/2026", fill="black", anchor="mm")

    y = 150
    for p in picks:
        draw.text((100, y), f"{p['match']}", fill="white")
        draw.text((100, y+40), f"Pick: {p['pick']} | Prob: {p['prob']:.1%}", fill="#00FF00")
        draw.text((700, y+40), f"Cuota BF: {p['cuota']} | ${p['apuesta']} MXN", fill="gold")
        draw.line([100, y+90, 900, y+90], fill=(40, 40, 40))
        y += 120
    
    img.save("prediction_card.png")

if __name__ == "__main__":
    # 1. ANALISIS DE JORNADA NBA (09/04/2026)
    nba_matches = [
        {"match": "NY Knicks vs ATL Hawks", "h_rtg": 118, "a_rtg": 110},
        {"match": "PHI 76ers vs SA Spurs", "h_rtg": 116, "a_rtg": 114},
        {"match": "DET Pistons vs ORL Magic", "h_rtg": 102, "a_rtg": 115},
        {"match": "POR Blazers vs DEN Nuggets", "h_rtg": 105, "a_rtg": 122}
    ]
    
    final_picks = []
    report = "<b>📊 AUDITORÍA BETFAIR LIVE</b>\n\n"

    for m in nba_matches:
        cuota_bf = get_betfair_price(m['match'])
        prob, ev, apuesta = predict_nba_efficiency(m['h_rtg'], m['a_rtg'], cuota_bf)
        
        res = {**m, "prob": prob, "cuota": cuota_bf, "ev": ev, "apuesta": apuesta, "pick": "Gana Local"}
        final_picks.append(res)
        
        report += f"🔹 {m['match']}: EV {ev:+.2f} (Cuota BF: {cuota_bf})\n"
        
        with open("predictions_history.csv", 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d"), m['match'], f"{prob:.1%}", f"{ev:.2f}", f"{apuesta}"])

    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    generate_betfair_card(final_picks)
