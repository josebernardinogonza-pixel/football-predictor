# 🤖 EDGE BOT PRO - Sistema de Análisis Predictivo Evolutivo

**Edge Bot Pro** es un bot automatizado y ultraligero diseñado para el análisis cuantitativo de apuestas deportivas (Value Betting / +EV). Utiliza la potencia de **Groq Cloud (LLaMA 3.3 70B)** para encontrar ineficiencias en el mercado basándose en un sistema de aprendizaje evolutivo.

El bot se ejecuta de forma 100% gratuita y automática cada hora mediante **GitHub Actions**, obteniendo los partidos de la jornada (NBA, MLB, Liga MX, Premier League, LaLiga, Serie A, Bundesliga, Ligue 1) y enviando los pronósticos directamente a **Telegram**.

## 🚀 Características Principales

- **Aprendizaje Evolutivo (Regla 4):** El bot no predice a ciegas. Antes de cada análisis, lee un historial de fallos y aciertos (`aprendizaje.txt`) para calibrar sus sesgos, penalizar falsos positivos y adaptar su modelo matemático a la realidad empírica.
- **Memoria a Corto Plazo:** El bot guarda un registro (`procesados.txt`) de los partidos que ya analizó en el día para no enviar alertas duplicadas.
- **Inferencia Ultrarrápida:** Utiliza la API de Groq Cloud para procesar modelos de lenguaje masivos (LLMs) en milisegundos.
- **Automatización Total:** Configurado con un *CRON job* en GitHub Actions para ejecutarse cada hora sin intervención humana.
- **Filtro de Calidad:** Solo envía notificaciones a Telegram si el análisis matemático dictamina que el partido está **APROBADO** por tener un Edge claro.

## ⚠️ Disclaimer Legal y Financiero

Este proyecto tiene fines estrictamente educativos, estadísticos y de investigación cuantitativa. **No constituye asesoramiento financiero ni garantiza ganancias.** Las apuestas deportivas conllevan un alto riesgo de pérdida de capital. Utiliza este software bajo tu propia responsabilidad.
