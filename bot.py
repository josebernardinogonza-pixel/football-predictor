import os
import requests
import time
import csv
import json
from datetime import datetime, timedelta

# ==========================================
# CONFIGURACIÓN
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

os.makedirs("data", exist_ok=True)

# Mapeo de ligas para la API de ESPN
LEAGUE_MAP = {
    "NBA": "basketball/nba",
    "MLB": "baseball/mlb",
    "Liga MX": "soccer/mex.1",
    "English Premier League": "soccer/eng.1",
    "Spanish LaLiga": "soccer/esp.1",
    "Italian Serie A": "soccer/ita.1",
    "German Bundesliga": "soccer/ger.1",
    "French Ligue 1": "soccer/fra.1"
}

def leer_aprendizaje():
    ruta = "data/aprendizaje.txt"
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()[-2500:]
    return "No hay datos previos. Sé conservador."

def auditar_resultados():
    """Busca resultados de partidos pendientes y actualiza el aprendizaje"""
    ruta_csv = "data/predictions_history.csv"
    if not os.path.exists(ruta_csv): return
    
    print("🔍 Iniciando Auditoría Automática...")
    filas_actualizadas = []
    aciertos = 0
    fallos = 0
    lecciones = []

    with open(ruta_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        if len(reader) < 2: return
        header = reader[0]
        for row in reader[1:]:
            # row format: [ID, Fecha, Partido, Liga, Pick, Prob, Estado]
            if row[6] == "PENDIENTE":
                p_id, fecha, partido, liga, pick, prob, estado = row
                resultado = buscar_marcador_espn(liga, fecha, partido)
                
                if resultado != "PENDIENTE":
                    # Lógica simple de validación
                    ganador_real = resultado # "LOCAL", "VISITANTE" o "EMPATE"
                    es_acierto = ganador_real in pick.upper()
                    
                    row[6] = "GANADA" if es_acierto else "PERDIDA"
                    if es_acierto: aciertos += 1
                    else: fallos += 1
                    lecciones.append(f"Partido {partido}: {'Acierto' if es_acierto else 'Fallo'}. Predicción fue {pick}, resultado real {ganador_real}.")
            
            filas_actualizadas.append(row)

    # Guardar CSV actualizado
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filas_actualizadas)

    # Escribir en el diario de aprendizaje
    if lecciones:
        with open("data/aprendizaje.txt", "a", encoding="utf-8") as f:
            resumen = f"\n--- AUDITORÍA {datetime.now().date()} ---\n"
            resumen += f"Resultado: {aciertos} aciertos, {fallos} fallos.\n"
            resumen += "\n".join(lecciones) + "\n"
            f.write(resumen)
        print(f"✅ Auditoría finalizada: {aciertos}W - {fallos}L. Memoria actualizada.")

def buscar_marcador_espn(liga_nombre, fecha_str, partido_nombre):
    """Consulta la API de ESPN para obtener el ganador real"""
    path = LEAGUE_MAP.get(liga_nombre)
    if not path: return "PENDIENTE"
    
    # Formato fecha para ESPN: YYYYMMDD
    date_api = fecha_str.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={date_api}"
    
    try:
        data = requests.get(url, timeout=10).json()
        for event in data.get("events", []):
            name = event.get("name", "")
            if event["status"]["type"]["state"] == "post": # Solo partidos terminados
                # Determinar ganador
                teams = event["competitions"][0]["competitors"]
                home = next(t for t in teams if t["homeAway"] == "home")
                away = next(t for t in teams if t["homeAway"] == "away")
                
                h_score = int(home["score"])
                a_score = int(away["score"])
                
                ganador = "EMPATE"
                if h_score > a_score: ganador = "LOCAL"
                elif a_score > h_score: ganador = "VISITANTE"
                
                # Verificar si es el partido que buscamos (comparación simple de nombres)
                if home["team"]["name"] in partido_nombre:
                    return ganador
    except: pass
    return "PENDIENTE"

def analizar_con_ia(historial, partido_str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""Eres EDGE BOT PRO. Analiza basándote en este aprendizaje previo:
    {historial}
    
    Partido: {partido_str}
    
    [REGLA] Si el pick es para el Local, incluye la palabra 'LOCAL'. Si es para el Visitante, 'VISITANTE'.
    [FORMATO]
    🏟️ Partido: [Nombres]
    🏆 Competición: [Liga]
    🧠 Análisis: [Técnico corto]
    🎯 Pick Principal: [LOCAL/VISITANTE/EMPATE + Mercado]
    📊 Probabilidad: [X%]
    ⚖️ Veredicto: [APROBADO/DESCARTADO]"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        res = requests.post(url, headers=headers, json=payload).json()
        return res["choices"][0]["message"]["content"]
    except: return "DESCARTADO"

def main():
    print("🚀 EDGE BOT PRO: SISTEMA DE APRENDIZAJE ACTIVO")
    
    # 1. Aprender de lo que pasó ayer
    auditar_resultados()
    
    # 2. Obtener nuevos partidos
    historial = leer_aprendizaje()
    # (Aquí llamarías a obtener_partidos_hoy() del código anterior)
    # ... resto de la lógica de envío a Telegram ...

if __name__ == "__main__":
    main()
