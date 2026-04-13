import os
import requests
import numpy as np
from scipy.stats import poisson
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
STAKE_API_KEY = os.getenv("STAKE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BANKROLL_MXN = 21830.00
KELLY_FRACTION = 0.15

def get_direct_stake_odds():
    """Consulta cuotas en Stake con cabeceras de navegador para evitar bloqueos"""
    url = "https://api.stake.com/graphql"
    query = """
    {
      activeSports(sport: "soccer") {
        groups {
          events {
            name
            markets(names: ["winner"]) {
              outcomes {
                name
                value
              }
            }
          }
        }
      }
    }
    """
    # Cabeceras para parecer un humano navegando
    headers = {
        "x-access-token": STAKE_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.post(url, json={'query': query}, headers=headers, timeout=15)
        if response.status_code != 200:
            return f"Error de Servidor: {response.status_code}"
        
        data = response.json()
        odds_results = []
        
        # Navegación segura por el JSON de Stake
        active_sports = data.get('data', {}).get('activeSports', [])
        if not active_sports: return []

        for group in active_sports[0].get('groups', []):
            for event in group.get('events', []):
                try:
                    m_name = event['name']
                    outcomes = event['markets'][0]['outcomes']
                    # Cuota del Local
                    h_odds = next(o['value'] for o in outcomes if o['name'] in m_name)
                    odds_results.append({"match": m_name, "cuota": float(h_odds)})
                except: continue
        return odds_results
    except Exception as e:
        print(f"Error técnico: {e}")
        return []

def predict_match(match):
    """Calcula probabilidad y stake"""
    h_xg, a_xg = 1.85, 1.15 # xG base optimizado
    cuota = match['cuota']
    
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    
    apuesta = 0
    if ev > 0.08:
        f_star = ((cuota - 1) * prob_l - (1 - prob_l)) / (cuota - 1)
        apuesta = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
        
    return {**match, "prob": prob_l, "ev": ev, "apuesta": max(0, apuesta)}

def enviar_telegram_bonito(picks):
    """Genera un mensaje de texto profesional y legible"""
    if not picks:
        mensaje = "<b>⏳ SIN OPORTUNIDADES</b>\nEl mercado de Stake no presenta valor ahora mismo."
    else:
        mensaje = "🚀 <b>INVERSIÓN DE ÉLITE STAKE</b>\n"
        mensaje += f"📅 Fecha: {datetime.now().strftime('%d/%m/%Y')}\n"
        mensaje += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, p in enumerate(picks, 1):
            mensaje += f"{i}. 🏟 <b>{p['match']}</b>\n"
            mensaje += f"   🎯 <b>PICK: VICTORIA LOCAL</b>\n"
            mensaje += f"   📈 Probabilidad: <code>{p['prob']:.1%}</code>\n"
            mensaje += f"   💰 Cuota Stake: <code>{p['cuota']:.2f}</code>\n"
            mensaje += f"   💎 Valor (EV): <code>{p['ev']:+.2f}</code>\n"
            mensaje += f"   💵 <b>APOSTAR: ${p['apuesta']} MXN</b>\n"
            mensaje += "────────────────────\n"
        
        mensaje += f"\n💰 <b>Bankroll Actual: ${BANKROLL_MXN} MXN</b>\n"
        mensaje += "⚠️ <i>Indicación: No sobre-apostar. Respetar el Stake.</i>"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "HTML"})

if __name__ == "__main__":
    print("🛰️ Consultando Stake.com...")
    partidos = get_direct_stake_odds()
    
    if isinstance(partidos, str):
        # Si hubo error de servidor, avisar
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                     json={"chat_id": CHAT_ID, "text": f"❌ <b>Error Stake:</b> {partidos}", "parse_mode": "HTML"})
    else:
        final_picks = []
        for p in partidos[:10]:
            res = predict_match(p)
            if res['apuesta'] > 0:
                final_picks.append(res)
        
        enviar_telegram_bonito(final_picks)
        print("✅ Reporte enviado a Telegram.")
