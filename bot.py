import os
import requests
import csv
import time
from datetime import datetime, timedelta

# ==========================================
# CONFIGURACIÓN DE ACCESO
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Ligas soportadas por ESPN
LEAGUES = {
    "NBA": "basketball/nba",
    "MLB": "baseball/mlb",
    "LIGA_MX": "soccer/mex.1",
    "PREMIER_LEAGUE": "soccer/eng.1",
    "LALIGA": "soccer/esp.1"
}

os.makedirs("data", exist_ok=True)
HISTORIAL_PATH = "data/predictions_history.csv"
APRENDIZAJE_PATH = "data/aprendizaje.txt"

# ==========================================
# 1. MÓDULO DE AUDITORÍA (VER ERRORES Y ACIERTOS)
# ==========================================
def auditar_y_aprender():
    if not os.path.exists(HISTORIAL_PATH): return
    
    print("🔍 Analizando aciertos y errores de ayer...")
    filas_actualizadas = []
    lecciones = []
    
    with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        if len(reader) < 2: return
        header = reader[0]
        
        for row in reader[1:]:
            # row: [ID, Fecha, Partido, Liga, Pick, Prob, Estado]
            if row[6] == "PENDIENTE":
                resultado = consultar_resultado_real(row[3], row[1], row[2])
                if resultado != "PENDIENTE":
                    es_acierto = resultado in row[4].upper()
                    row[6] = "GANADA ✅" if es_acierto else "PERDIDA ❌"
                    lecciones.append(f"Partido: {row[2]} | Predicción: {row[4]} | Real: {resultado} | {'LOGRADO' if es_acierto else 'ERROR'}")
            filas_actualizadas.append(row)

    # Guardar resultados
    with open(HISTORIAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filas_actualizadas)

    # Escribir lecciones para que la IA las lea
    if lecciones:
        with open(APRENDIZAJE_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- SESIÓN DE APRENDIZAJE {datetime.now()} ---\n")
            f.write("\n".join(lecciones) + "\n")

def consultar_resultado_real(liga_key, fecha, partido_nombre):
    path = LEAGUES.get(liga_key)
    date_str = fecha.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={date_str}"
    
    try:
        data = requests.get(url).json()
        for event in data.get("events", []):
            if event["status"]["type"]["state"] == "post":
                teams = event["competitions[0]"]["competitors"]
                h = next(t for t in teams if t["homeAway"] == "home")
                a = next(t for t in teams if t["homeAway"] == "away")
                
                res = "EMPATE"
                if int(h["score"]) > int(a["score"]): res = "LOCAL"
                elif int(a["score"]) > int(h["score"]): res = "VISITANTE"
                
                if h["team"]["name"] in partido_nombre: return res
    except: pass
    return "PENDIENTE"

# ==========================================
# 2. MÓDULO DE PREDICCIÓN (CONEXIÓN IA)
# ==========================================
def obtener_datos_ia(partido_str):
    # Leer lo que aprendimos de los errores antes de predecir
    aprendizaje = ""
    if os.path.exists(APRENDIZAJE_PATH):
        with open(APRENDIZAJE_PATH, "r") as f:
            aprendizaje = f.read()[-3000:]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""[SISTEMA DE APRENDIZAJE ACTIVO]
    Errores y aciertos previos: {aprendizaje}
    
    [PARTIDO A ANALIZAR]
    {partido_str}
    
    [INSTRUCCIÓN]
    Analiza el partido. Si el valor es alto, responde APROBADO. Si no, DESCARTADO.
    Usa 'LOCAL' o 'VISITANTE' para definir al ganador.
    
    Formato:
    🏟️ Partido: [Nombres]
    🎯 Pick: [LOCAL/VISITANTE + Mercado]
    📊 Probabilidad: [X%]
    ⚖️ Veredicto: [APROBADO/DESCARTADO]"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload).json()
        return res["choices"][0]["message"]["content"]
    except: return "DESCARTADO"

# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    # Primero: Aprender de los errores de ayer
    auditar_y_aprender()
    
    # Segundo: Buscar partidos de hoy
    print("🚀 Escaneando cartelera de hoy...")
    for liga_key, path in LEAGUES.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
        try:
            data = requests.get(url).json()
            for event in data.get("events", []):
                if event["status"]["type"]["state"] == "pre":
                    p_str = event["name"]
                    analisis = obtener_datos_ia(f"Liga: {liga_key} | Partido: {p_str}")
                    
                    if "APROBADO" in analisis.upper():
                        # Enviar a Telegram
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                                     json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🤖 <b>EDGE BOT PRO</b>\n{analisis}", "parse_mode": "HTML"})
                        
                        # Guardar en historial para auditar mañana
                        with open(HISTORIAL_PATH, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([event["id"], datetime.now().date(), p_str, liga_key, analisis, "IA", "PENDIENTE"])
                    
                    time.sleep(2) # Evitar baneo de API
        except: continue

if __name__ == "__main__":
    main()
