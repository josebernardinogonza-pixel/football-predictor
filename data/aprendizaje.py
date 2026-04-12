import os

def actualizar_historial_aprendizaje():
    """
    Genera y actualiza el archivo aprendizaje.txt con las últimas 
    auditorías y reglas matemáticas (Regla 4) del modelo.
    """
    directorio = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(directorio, exist_ok=True)
    ruta_txt = os.path.join(directorio, "aprendizaje.txt")
    
    contenido_aprendizaje = """[HISTORIAL RECIENTE PARA APRENDIZAJE Y CALIBRACIÓN]

### REGLAS BASE (CHAMPIONS Y CONCACAF)
- [Away Elite Defense]: Reducir penalización a visitantes si son Top 5 europeo en PPDA.
- [Home Advantage Decay]: En eliminatorias con alto xG (>3.0), factor cancha pierde 15% de peso.
- [Extreme Altitude]: Multiplicar xGA del visitante por 1.35 en la 2da mitad en altitud (Ej: Pachuca, Toluca).

### REGLAS DE LIGAS DOMÉSTICAS Y DEPORTES AMERICANOS
- [Anti-Underdog Trap]: PROHIBIDO recomendar visitantes underdogs en ligas con alta disparidad económica.
- [Elite Roster Home Protection]: Penalizar picks en contra de equipos de Súper Élite en casa.
-[Playoff Seeding Motivation]: En NBA/MLB al final de temporada, potenciar a equipos peleando por playoffs vs eliminados.

### NUEVAS REGLAS TRAS AUDITORÍA DEL 11 DE ABRIL (CRÍTICAS)

### Evento A: Éxito Absoluto en Liga MX (Tigres 4-1, Pachuca 4-2, América 1-1)
- Lección Extraída: El modelo está perfectamente calibrado para México. La regla de altitud (Pachuca) y la detección de EV negativo (América -0.01) funcionaron a la perfección para evitar trampas.
- Regla a Aplicar[Liga MX Confidence]: Mantener máxima confianza algorítmica en locales de Liga MX con EV positivo. Buscar siempre el Over de goles como Pick Secundario en partidos de altitud.

### Evento B: Arsenal 1 - 2 Bournemouth (Premier League)
- Lección Extraída: Un equipo de Súper Élite perdió en casa ante un rival inferior en el mes de abril. Esto ocurre por rotaciones masivas previas/posteriores a los Cuartos de Final de la Champions League.
- Regla a Aplicar [European Rotation Trap]: En abril y mayo, REDUCIR la probabilidad de victoria de equipos Top de Europa (Arsenal, City, Madrid, Bayern, PSG) en sus ligas domésticas. Exigir un mercado de Goles (Over/Under) o Hándicap a favor del visitante en lugar de victoria directa del favorito.

### Evento C: Ligas Menores Asiáticas y de Europa del Este
- Lección Extraída: El sistema asignó un EV genérico (+0.10) a ligas con poca profundidad de datos, exponiéndonos a alta varianza.
- Regla a Aplicar [Data Depth Filter]: Para ligas fuera de las 8 principales (Top 5 Europa, Liga MX, MLS, NBA), el Edge/Valor mínimo exigido debe ser superior al +15%. Si no hay datos claros para justificar un Edge alto, el veredicto DEBE ser DESCARTADO.
"""

    with open(ruta_txt, "w", encoding="utf-8") as file:
        file.write(contenido_aprendizaje)
        
    print(f"✅ Archivo de aprendizaje actualizado exitosamente en: {ruta_txt}")

if __name__ == "__main__":
    actualizar_historial_aprendizaje()
