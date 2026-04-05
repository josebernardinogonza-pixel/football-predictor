import os
import requests
import re
import numpy as np
from scipy.stats import poisson
from PIL import Image, ImageDraw
from io import BytesIO
import csv
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
BANKROLL_MXN = 5000.00
KELLY_FRACTION = 0.20
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def get_real_result(home, away):
    """Busca el resultado real de un partido pasado en Google"""
    query = f"resultado final {home} vs {away}"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        res = requests.get(url).json()
        # Buscamos el marcador en el widget de deportes o snippets
        snippet = str(res.get("sports_results", "")) + str(res.get("organic_results", ""))
        scores = re.findall(r'(\d+)\s*-\s*(\d+)', snippet)
        if scores:
            h_score, a_score = int(scores[0][0]), int(scores[0][1])
            if h_score > a_score: return "L"
            if h_score == a_score: return "E"
            return "V"
    except: return None
    return None

def audit_past_predictions():
    """Revisa las últimas predicciones y calcula la precisión"""
    file = "predictions_history.csv"
    if not os.path.isfile(file): return "Sin historial aún."
    
    hits = 0
    total_audited = 0
    report = "📊 *AUDITORÍA DE RESULTADOS*\n"

    with open(file, 'r') as f:
        rows = list(csv.reader(f))
        # Analizamos los últimos 5 partidos del historial
        for row in rows[-5:]:
            fecha, partido, prob, ev, apuesta = row
            home, away = partido.split(" vs ")
            
            resultado_real = get_real_result(home, away)
            if resultado_real:
                total_audited += 1
                # Si predijimos Victoria Local (L) y así fue:
                if resultado_real == "L":
                    hits += 1
                    report += f"✅ {home} vs {away}: ACERTADO\n"
                else:
                    report += f"❌ {home} vs {away}: FALLADO\n"
    
    if total_audited > 0:
        precision = (hits / total_audited) * 100
        report += f"\n🎯 *Precisión Reciente: {precision:.1f}%*"
    else:
        report = "⏳ Esperando que terminen los partidos para auditar..."
    
    return report

# --- (Mantén tus funciones get_stats, get_news, generate_card igual que antes) ---

if __name__ == "__main__":
    # 1. AUDITORÍA (¿Qué pasó con lo que predije ayer?)
    resumen_auditoria = audit_past_predictions()
    
    # Guardamos el resumen en un txt para que el Workflow lo envíe a Telegram
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write(resumen_auditoria)

    # 2. NUEVAS PREDICCIONES (Igual que antes)
    print("Iniciando nuevas predicciones...")
    # ... (Aquí va el resto de tu código de búsqueda de partidos y generación de imagen)
    # [Usa el código del paso anterior para generar la imagen del mejor pick]
