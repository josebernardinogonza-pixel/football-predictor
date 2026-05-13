"""
EDGE BOT PRO - Training Script v2.0 (Machine Learning Real)
Genera modelos calibrados y optimizados para soccer, NBA y MLB
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from datetime import datetime

os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------------
# 1. Generación de datos sintéticos avanzados (15 features + target)
# -------------------------------------------------------------------
def generate_soccer_data(n=10000):
    np.random.seed(42)
    data = {
        'goals_scored_avg_5': np.random.uniform(0.8, 2.2, n),
        'goals_conceded_avg_5': np.random.uniform(0.5, 1.8, n),
        'goals_scored_avg_10': np.random.uniform(0.9, 2.0, n),
        'goals_conceded_avg_10': np.random.uniform(0.6, 1.7, n),
        'opp_goals_scored_avg_5': np.random.uniform(0.7, 2.1, n),
        'opp_goals_conceded_avg_5': np.random.uniform(0.5, 1.9, n),
        'opp_goals_scored_avg_10': np.random.uniform(0.8, 2.0, n),
        'opp_goals_conceded_avg_10': np.random.uniform(0.6, 1.8, n),
        'home_advantage': np.random.uniform(0.05, 0.25, n),
        'rest_days': np.random.uniform(1, 7, n),
        'rest_diff': np.random.uniform(-2, 2, n),
        'rest_advantage': np.random.uniform(-0.3, 0.3, n),
        'strength_diff_5': np.random.uniform(-1.5, 1.5, n),
        'strength_diff_10': np.random.uniform(-1.2, 1.2, n),
        'form_rating': np.random.uniform(0.2, 0.8, n)
    }
    df = pd.DataFrame(data)
    # Target realista: prob = sigmoide(combinación lineal)
    logit = (0.8*df['home_advantage'] + 0.5*df['goals_scored_avg_5'] - 0.4*df['goals_conceded_avg_5']
             + 0.3*df['strength_diff_5'] + 0.1*df['form_rating'])
    prob = 1 / (1 + np.exp(-logit))  # sigmoide
    df['target'] = (np.random.uniform(0,1,n) < prob).astype(int)
    return df

def generate_nba_data(n=10000):
    np.random.seed(43)
    data = {
        'points_scored_avg_5': np.random.uniform(100, 120, n),
        'points_conceded_avg_5': np.random.uniform(100, 120, n),
        'points_scored_avg_10': np.random.uniform(101, 119, n),
        'points_conceded_avg_10': np.random.uniform(101, 119, n),
        'opp_points_scored_avg_5': np.random.uniform(100, 120, n),
        'opp_points_conceded_avg_5': np.random.uniform(100, 120, n),
        'opp_points_scored_avg_10': np.random.uniform(101, 119, n),
        'opp_points_conceded_avg_10': np.random.uniform(101, 119, n),
        'home_advantage': np.random.uniform(2, 8, n),
        'rest_days': np.random.uniform(0, 4, n),
        'rest_diff': np.random.uniform(-2, 2, n),
        'rest_advantage': np.random.uniform(-3, 3, n),
        'strength_diff_5': np.random.uniform(-12, 12, n),
        'strength_diff_10': np.random.uniform(-10, 10, n),
        'form_rating': np.random.uniform(0.2, 0.8, n)
    }
    df = pd.DataFrame(data)
    logit = (0.02*df['home_advantage'] + 0.01*(df['points_scored_avg_5']-110) - 0.01*(df['points_conceded_avg_5']-110)
             + 0.005*df['strength_diff_5'] + 0.1*df['form_rating'])
    prob = 1 / (1 + np.exp(-logit))
    df['target'] = (np.random.uniform(0,1,n) < prob).astype(int)
    return df

def generate_mlb_data(n=10000):
    np.random.seed(44)
    data = {
        'runs_scored_avg_5': np.random.uniform(3.0, 6.0, n),
        'runs_conceded_avg_5': np.random.uniform(3.0, 6.0, n),
        'runs_scored_avg_10': np.random.uniform(3.1, 5.9, n),
        'runs_conceded_avg_10': np.random.uniform(3.1, 5.9, n),
        'opp_runs_scored_avg_5': np.random.uniform(3.0, 6.0, n),
        'opp_runs_conceded_avg_5': np.random.uniform(3.0, 6.0, n),
        'opp_runs_scored_avg_10': np.random.uniform(3.1, 5.9, n),
        'opp_runs_conceded_avg_10': np.random.uniform(3.1, 5.9, n),
        'home_advantage': np.random.uniform(0.1, 0.6, n),
        'rest_days': np.random.uniform(0, 3, n),
        'rest_diff': np.random.uniform(-1, 1, n),
        'rest_advantage': np.random.uniform(-0.3, 0.3, n),
        'strength_diff_5': np.random.uniform(-2.5, 2.5, n),
        'strength_diff_10': np.random.uniform(-2.0, 2.0, n),
        'form_rating': np.random.uniform(0.2, 0.8, n)
    }
    df = pd.DataFrame(data)
    logit = (0.2*df['home_advantage'] + 0.3*(df['runs_scored_avg_5']-4.5) - 0.3*(df['runs_conceded_avg_5']-4.5)
             + 0.1*df['strength_diff_5'] + 0.15*df['form_rating'])
    prob = 1 / (1 + np.exp(-logit))
    df['target'] = (np.random.uniform(0,1,n) < prob).astype(int)
    return df

# -------------------------------------------------------------------
# 2. Feature engineering común
# -------------------------------------------------------------------
FEATURE_COLS = [
    'scored_rolling_5', 'conceded_rolling_5', 'scored_rolling_10', 'conceded_rolling_10',
    'opponent_scored_rolling_5', 'opponent_conceded_rolling_5', 'opponent_scored_rolling_10',
    'opponent_conceded_rolling_10', 'home_advantage', 'rest_days', 'rest_diff',
    'rest_advantage', 'strength_diff_5', 'strength_diff_10', 'form_rating'
]

def rename_features(df, sport):
    """Renombra columnas genéricas a las que espera el modelo."""
    mapping = {
        'soccer': {'goals_scored_avg_5': 'scored_rolling_5', 'goals_conceded_avg_5': 'conceded_rolling_5',
                   'goals_scored_avg_10': 'scored_rolling_10', 'goals_conceded_avg_10': 'conceded_rolling_10',
                   'opp_goals_scored_avg_5': 'opponent_scored_rolling_5', 'opp_goals_conceded_avg_5': 'opponent_conceded_rolling_5',
                   'opp_goals_scored_avg_10': 'opponent_scored_rolling_10', 'opp_goals_conceded_avg_10': 'opponent_conceded_rolling_10'},
        'nba': {'points_scored_avg_5': 'scored_rolling_5', 'points_conceded_avg_5': 'conceded_rolling_5',
                'points_scored_avg_10': 'scored_rolling_10', 'points_conceded_avg_10': 'conceded_rolling_10',
                'opp_points_scored_avg_5': 'opponent_scored_rolling_5', 'opp_points_conceded_avg_5': 'opponent_conceded_rolling_5',
                'opp_points_scored_avg_10': 'opponent_scored_rolling_10', 'opp_points_conceded_avg_10': 'opponent_conceded_rolling_10'},
        'mlb': {'runs_scored_avg_5': 'scored_rolling_5', 'runs_conceded_avg_5': 'conceded_rolling_5',
                'runs_scored_avg_10': 'scored_rolling_10', 'runs_conceded_avg_10': 'conceded_rolling_10',
                'opp_runs_scored_avg_5': 'opponent_scored_rolling_5', 'opp_runs_conceded_avg_5': 'opponent_conceded_rolling_5',
                'opp_runs_scored_avg_10': 'opponent_scored_rolling_10', 'opp_runs_conceded_avg_10': 'opponent_conceded_rolling_10'}
    }
    df.rename(columns=mapping.get(sport, {}), inplace=True)
    # Las columnas restantes ya tienen el nombre correcto (home_advantage, rest_*, strength_*, form_rating)
    return df[FEATURE_COLS + ['target']]

# -------------------------------------------------------------------
# 3. Entrenamiento con optimización de hiperparámetros
# -------------------------------------------------------------------
def train_and_save(df, sport_name):
    df = rename_features(df, sport_name)
    X = df[FEATURE_COLS]
    y = df['target']

    # División train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo base: Gradient Boosting con búsqueda ligera de hiperparámetros
    base_model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    # Búsqueda rápida con 2 folds para ahorrar tiempo
    grid = GridSearchCV(base_model, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    # Calibración de probabilidades (Platt scaling) para mejorar estimaciones
    calibrated = CalibratedClassifierCV(grid.best_estimator_, method='sigmoid', cv=3)
    calibrated.fit(X_train, y_train)

    # Evaluación
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = calibrated.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    print(f"✅ {sport_name} modelo entrenado - AUC: {auc:.3f} | Brier: {brier:.3f}")
    print(f"   Mejores hiperparámetros: {grid.best_params_}")

    # Guardar modelo
    path = f"models/{sport_name}_model.pkl"
    with open(path, 'wb') as f:
        pickle.dump(calibrated, f)
    print(f"   Guardado en {path}")

# -------------------------------------------------------------------
# 4. Ejecución principal
# -------------------------------------------------------------------
print("🔄 Generando datos sintéticos avanzados...")
soccer_df = generate_soccer_data()
nba_df = generate_nba_data()
mlb_df = generate_mlb_data()

train_and_save(soccer_df, "soccer")
train_and_save(nba_df, "nba")
train_and_save(mlb_df, "mlb")

print("🎉 Todos los modelos listos. Ejecuta: python bot.py")
