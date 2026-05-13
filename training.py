"""
EDGE BOT PRO - Training Script v1.0
Genera modelos .pkl para soccer, NBA y MLB usando datos sintéticos
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

def generate_soccer_data(n=5000):
    np.random.seed(42)
    data = {
        'scored_rolling_5': np.random.uniform(0.5, 2.5, n),
        'conceded_rolling_5': np.random.uniform(0.3, 2.2, n),
        'scored_rolling_10': np.random.uniform(0.6, 2.4, n),
        'conceded_rolling_10': np.random.uniform(0.4, 2.1, n),
        'opponent_scored_rolling_5': np.random.uniform(0.5, 2.3, n),
        'opponent_conceded_rolling_5': np.random.uniform(0.4, 2.2, n),
        'opponent_scored_rolling_10': np.random.uniform(0.5, 2.2, n),
        'opponent_conceded_rolling_10': np.random.uniform(0.4, 2.0, n),
        'home_advantage': np.random.uniform(0.05, 0.25, n),
        'rest_days': np.random.uniform(1, 7, n),
        'rest_diff': np.random.uniform(-2, 2, n),
        'rest_advantage': np.random.uniform(-0.3, 0.3, n),
        'strength_diff_5': np.random.uniform(-1, 1, n),
        'strength_diff_10': np.random.uniform(-1, 1, n)
    }
    df = pd.DataFrame(data)
    # Target sintético: local gana si cumple ciertas condiciones
    prob = 0.3 + 0.2*df['scored_rolling_5'] - 0.15*df['conceded_rolling_5'] + 0.1*df['home_advantage']
    prob = np.clip(prob, 0, 1)
    df['target'] = (np.random.uniform(0,1,n) < prob).astype(int)
    return df

def generate_nba_data(n=5000):
    np.random.seed(43)
    data = {
        'scored_rolling_5': np.random.uniform(95, 125, n),
        'conceded_rolling_5': np.random.uniform(95, 125, n),
        'scored_rolling_10': np.random.uniform(96, 124, n),
        'conceded_rolling_10': np.random.uniform(96, 124, n),
        'opponent_scored_rolling_5': np.random.uniform(95, 125, n),
        'opponent_conceded_rolling_5': np.random.uniform(95, 125, n),
        'opponent_scored_rolling_10': np.random.uniform(95, 124, n),
        'opponent_conceded_rolling_10': np.random.uniform(95, 124, n),
        'home_advantage': np.random.uniform(2, 8, n),
        'rest_days': np.random.uniform(0, 4, n),
        'rest_diff': np.random.uniform(-2, 2, n),
        'rest_advantage': np.random.uniform(-3, 3, n),
        'strength_diff_5': np.random.uniform(-10, 10, n),
        'strength_diff_10': np.random.uniform(-10, 10, n)
    }
    df = pd.DataFrame(data)
    prob = 0.35 + 0.005*(df['scored_rolling_5'] - 110) - 0.005*(df['conceded_rolling_5'] - 110) + 0.02*df['home_advantage']
    prob = np.clip(prob, 0, 1)
    df['target'] = (np.random.uniform(0,1,n) < prob).astype(int)
    return df

def generate_mlb_data(n=5000):
    np.random.seed(44)
    data = {
        'scored_rolling_5': np.random.uniform(2.5, 6.5, n),
        'conceded_rolling_5': np.random.uniform(2.5, 6.5, n),
        'scored_rolling_10': np.random.uniform(2.6, 6.4, n),
        'conceded_rolling_10': np.random.uniform(2.6, 6.4, n),
        'opponent_scored_rolling_5': np.random.uniform(2.5, 6.5, n),
        'opponent_conceded_rolling_5': np.random.uniform(2.5, 6.5, n),
        'opponent_scored_rolling_10': np.random.uniform(2.6, 6.4, n),
        'opponent_conceded_rolling_10': np.random.uniform(2.6, 6.4, n),
        'home_advantage': np.random.uniform(0.1, 0.6, n),
        'rest_days': np.random.uniform(0, 3, n),
        'rest_diff': np.random.uniform(-1, 1, n),
        'rest_advantage': np.random.uniform(-0.3, 0.3, n),
        'strength_diff_5': np.random.uniform(-2, 2, n),
        'strength_diff_10': np.random.uniform(-2, 2, n)
    }
    df = pd.DataFrame(data)
    prob = 0.4 + 0.1*(df['scored_rolling_5'] - 4.5) - 0.1*(df['conceded_rolling_5'] - 4.5) + 0.15*df['home_advantage']
    prob = np.clip(prob, 0, 1)
    df['target'] = (np.random.uniform(0,1,n) < prob).astype(int)
    return df

print("🔄 Generando datos sintéticos...")
soccer_df = generate_soccer_data()
nba_df = generate_nba_data()
mlb_df = generate_mlb_data()

feature_cols = ['scored_rolling_5', 'conceded_rolling_5', 'scored_rolling_10', 'conceded_rolling_10',
                'opponent_scored_rolling_5', 'opponent_conceded_rolling_5', 'opponent_scored_rolling_10',
                'opponent_conceded_rolling_10', 'home_advantage', 'rest_days', 'rest_diff',
                'rest_advantage', 'strength_diff_5', 'strength_diff_10']

def train_and_save(df, sport_name):
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"✅ {sport_name} modelo entrenado - Accuracy: {acc:.3f}")
    path = f"models/{sport_name}_model.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Guardado en {path}")

train_and_save(soccer_df, "soccer")
train_and_save(nba_df, "nba")
train_and_save(mlb_df, "mlb")

print("🎉 Todos los modelos listos. Ahora puedes ejecutar: python bot.py")
