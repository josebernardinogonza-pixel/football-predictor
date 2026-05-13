import pandas as pd
import os

def load_real_data():
    filepath = 'data/match_history.csv'  # Ajusta esta ruta si tu archivo está en otro lugar

    if not os.path.exists(filepath):
        raise RuntimeError(f"Archivo no encontrado: {filepath}")

    df_raw = pd.read_csv(filepath)

    expected_raw_cols = {'Date', 'Home_Team', 'Away_Team', 'Home_Score', 'Away_Score', 'League'}
    missing = expected_raw_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en datos originales: {missing}")

    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    # Crear filas para equipos locales
    home_df = pd.DataFrame({
        'date': df_raw['Date'],
        'team': df_raw['Home_Team'],
        'opponent': df_raw['Away_Team'],
        'is_home': True,
        'scored': df_raw['Home_Score'],
        'conceded': df_raw['Away_Score'],
        'league': df_raw['League'],
    })

    # Crear filas para equipos visitantes
    away_df = pd.DataFrame({
        'date': df_raw['Date'],
        'team': df_raw['Away_Team'],
        'opponent': df_raw['Home_Team'],
        'is_home': False,
        'scored': df_raw['Away_Score'],
        'conceded': df_raw['Home_Score'],
        'league': df_raw['League'],
    })

    df = pd.concat([home_df, away_df], ignore_index=True)

    # Crear columna target: 1 para victoria, 0 empate, -1 derrota
    def get_target(row):
        if row['scored'] > row['conceded']:
            return 1
        elif row['scored'] == row['conceded']:
            return 0
        else:
            return -1

    df['target'] = df.apply(get_target, axis=1)

    expected_final_cols = {'is_home', 'team', 'date', 'conceded', 'opponent', 'target', 'scored', 'league'}
    missing_final = expected_final_cols - set(df.columns)
    if missing_final:
        raise ValueError(f"Columnas faltantes tras transformación: {missing_final}")

    return df

def train_all():
    print("🚀 Iniciando entrenamiento de modelos...")
    try:
        df = load_real_data()
    except Exception as e:
        print(f"❌ ERROR INESPERADO: Error procesando datos: {e}")
        return None

    print(f"Datos cargados correctamente, {len(df)} filas disponibles.")
    
    # Aquí puedes agregar la lógica de entrenamiento con tu dataframe df.
    # Por ejemplo, preparar datos, entrenar modelos, evaluar, etc.

    # Resultado simulado para ejemplo
    results = {"status": "ok", "rows": len(df)}
    return results


if __name__ == '__main__':
    results = train_all()
    if results:
        print("✅ Entrenamiento finalizado con éxito.")
    else:
        print("❌ El entrenamiento terminó con errores.")
