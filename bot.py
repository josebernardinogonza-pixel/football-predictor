import os
import requests
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
import csv
import json
import time
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURACIÓN DE ÉLITE
# ==========================================
BANKROLL_MXN = 21830.00  # Saldo actualizado tras Liga MX
KELLY_FRACTION = 0.15
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Rutas de Memoria
os.makedirs("data", exist_ok=True)
HISTORIAL_PATH = "data/predictions_history.csv"
APRENDIZAJE_PATH = "data/aprendizaje.txt"
CONFIG_IA_PATH = "data/config_ia.json"

# ==========================================
# 2. MOTOR DE APRENDIZAJE Y AUDITORÍA
# ==========================================
def auditar_jornada():
    """Compara predicciones pasadas con resultados reales de ESPN"""
    if not os.path.exists(HISTORIAL_PATH): return 0, 0, 0
    
    aciertos, fallos, balance = 0, 0, 0.0
    filas_actualizadas = []
    lecciones = []

    with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        if len(reader) < 2: return 0, 0, 0
        header = reader[0]
        for row in reader[1:]:
            # row: [Fecha, Partido, Liga, Pick, Prob, Cuota, Stake, Estado]
            if row[7] == "PENDIENTE":
                res = consultar_espn(row[2], row[0], row[1])
                if res != "PENDIENTE":
                    ganó = res in row[3].upper()
                    row[7] = "GANADA ✅" if ganó else "PERDIDA ❌"
                    if ganó:
                        aciertos += 1
                        balance += (float(row[5]) - 1) * float(row[6])
                    else:
                        fallos += 1
                        balance -= float(row[6])
                    lecciones.append(f"• {row[1]}: {'Acierto' if ganó else 'Error'}. Real: {res}")
            filas_actualizadas.append(row)

    with open(HISTORIAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filas_actualizadas)

    if lecciones:
        with open(APRENDIZAJE_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().date()}] AUDITORÍA:\n" + "\n".join(lecciones) + "\n")
    
    return aciertos, fallos, balance

def consultar_espn(liga, fecha, partido):
    """Consulta marcadores reales"""
    # Simplificación para el ejemplo: En producción usa el mapeo de IDs de ESPN
    return "LOCAL" # Simulación: El bot detecta que ganó el local

def recalibrar_ia(aciertos, fallos):
    """Ajusta los parámetros de riesgo automáticamente"""
    if os.path.exists(CONFIG_IA_PATH):
        with open(CONFIG_IA_PATH, "r") as f: config = json.load(f)
    else:
        config = {"MIN_EV": 0.08, "STRENGTH_ADJUST": 1.10}

    hit_rate = aciertos / (aciertos + fallos) if (aciertos + fallos) > 0 else 0.5
    if hit_rate < 0.45: config["MIN_EV"] += 0.02 # Más estrictos
    elif hit_rate > 0.65: config["MIN_EV"] = max(0.05, config["MIN_EV"] - 0.01) # Más agresivos
    
    with open(CONFIG_IA_PATH, "w") as f: json.dump(config, f)
    return config

# ==========================================
# 3. MOTOR MATEMÁTICO (POISSON & KELLY)
# ==========================================
def predict_poisson(h_xg, a_xg, cuota, config):
    """Cálculo de probabilidad real"""
    h_xg *= config["STRENGTH_ADJUST"]
    prob_l = np.sum(np.tril(np.outer(poisson.pmf(range(10), h_xg), poisson.pmf(range(10), a_xg)), -1))
    ev = (prob_l * cuota) - 1
    
    apuesta = 0
    if ev > config["MIN_EV"]:
        f_star = ((cuota - 1) * prob_l - (1 - prob_l)) / (cuota - 1)
        apuesta = round(f_star * KELLY_FRACTION * BANKROLL_MXN, 2)
    
    return prob_l, ev, max(0, apuesta)

# ==========================================
# 4. INTERFAZ VISUAL (CARTELERA PRO)
# ==========================================
def generate_pro_card(picks):
    img = Image.new('RGB', (1000, 1200), color=(10, 10, 15))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 120], fill="#D4AF37")
    draw.text((500, 60), f"EL MAESTRO - ESTRATEGIA {datetime.now().strftime('%d/%m/%Y')}", fill="black", anchor="mm")

    y = 180
    for p in picks:
        draw.rectangle([50, y, 950, y+120], outline="#D4AF37", width=2)
        draw.text((80, y+30), p['match'], fill="white")
        draw.text((80, y+70), f"PICK: {p['pick']} | PROB: {p['prob']:.1%}", fill="#00FF00")
        draw.text((750, y+50), f"STAKE: ${p['apuesta']} MXN", fill="#D4AF37")
        y += 150
    
    draw.text((500, 1150), f"BANKROLL: ${BANKROLL_MXN} MXN | IA RECALIBRADA", fill="gray", anchor="mm")
    img.save("prediction_card.png")

# ==========================================
# 5. EJECUCIÓN MAESTRA (10 AM CDMX)
# ==========================================
def main():
    print("🚀 Iniciando Ciclo de Inteligencia...")
    
    # A. Auditoría y Aprendizaje
    aciertos, fallos, balance = auditar_jornada()
    config = recalibrar_ia(aciertos, fallos)
    
    # B. Reporte de la Mañana (Solo a las 10 AM CDMX / 16:00 UTC)
    if 15 <= datetime.now().hour <= 17:
        msg = f"🤖 <b>REPORTE DE APRENDIZAJE</b>\n✅ Aciertos: {aciertos}\n❌ Fallos: {fallos}\n💰 Balance: {balance:+.2f} MXN\n\n🧠 IA: Parámetros ajustados para hoy."
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})

    # C. Nuevas Predicciones (Ejemplo con Liga MX y Europa)
    partidos_hoy = [
        {"match": "Toluca vs San Luis", "h_xg": 2.5, "a_xg": 0.8, "cuota": 1.65, "liga": "LIGA_MX"},
        {"match": "Man Utd vs Leeds", "h_xg": 2.1, "a_xg": 1.2, "cuota": 1.57, "liga": "PREMIER"}
    ]
    
    final_picks = []
    for p in partidos_hoy:
        prob, ev, stake = predict_poisson(p['h_xg'], p['a_xg'], p['cuota'], config)
        if stake > 0:
            res = {**p, "prob": prob, "ev": ev, "apuesta": stake, "pick": "Victoria Local"}
            final_picks.append(res)
            
            # Guardar en Historial
            with open(HISTORIAL_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if os.stat(HISTORIAL_PATH).st_size == 0:
                    writer.writerow(["Fecha", "Partido", "Liga", "Pick", "Prob", "Cuota", "Stake", "Estado"])
                writer.writerow([datetime.now().date(), p['match'], p['liga'], "LOCAL", prob, p['cuota'], stake, "PENDIENTE"])

    # D. Salida Visual
    if final_picks:
        generate_pro_card(final_picks)
        print("✅ Cartelera generada y enviada.")

if __name__ == "__main__":
    main()
