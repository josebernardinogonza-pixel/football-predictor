"""
EDGE BOT PRO v3.2 - Entrenamiento Multi-Sport | Auto-Rolling Features (CSV/JSON)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from scipy.stats import poisson
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Config
ROLLING_WINDOWS = [5, 10]
HOME_ADV_MX = 0.15
MIN_REST_DAYS = 3

def load_data(filepath='data/match_history.csv'):
    """Carga CSV o JSON, soporta ambos."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Formato no soportado: usa CSV o JSON")
        print(f"✅ Datos cargados: {len(df)} filas desde {filepath}")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("⚠️ Archivo no encontrado → Gen sample data")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading {filepath}: {e} → Gen sample")
        return generate_sample_data()

def generate_sample_data(sport='soccer', n_samples=2000):
    """Sample data como fallback."""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='2D')  # Más realista
    leagues = ['Premier League', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Liga MX'] if sport=='soccer' else [sport]
    teams = [f'Team{i}' for i in range(20)] * (n_samples // 20)
    
    data = {
        'date': dates,
        'league': np.random.choice(leagues, n_samples),
        'team': teams,
        'opponent': np.random.choice(teams, n_samples),
        'is_home': np.random.choice([0,1], n_samples, p=[0.5, 0.5]),
    }
    if sport == 'soccer':
        data['scored'] = np.random.poisson(1.4, n_samples)
        data['conceded'] = np.random.poisson(1.2, n_samples)
        data['target'] = np.where(data['scored'] > data['conceded'], 0,  # 0: home win
                                  np.where(data['scored'] == data['conceded'], 1, 2))
        data['target'] = data['target'] if data['is_home'].all() else data['target']  # Adjust for is_home
    elif sport in ['nba', 'mlb']:
        lmb_scored, lmb_conc = (110,108) if sport=='nba' else (4.6,4.4)
        data['scored'] = np.random.poisson(lmb_scored, n_samples)
        data['conceded'] = np.random.poisson(lmb_conc, n_samples)
        data['target'] = np.where(data['scored'] > data['conceded'], 0, 2)  # No draw
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    return df

def compute_rolling_features(df, windows=ROLLING_WINDOWS):
    """Calcula team & opponent rolling si faltan."""
    print("🔄 Computando rolling averages...")
    df = df.sort_values(['team', 'date'])
    
    for w in windows:
        # Team perspective
        df[f'scored_rolling_{w}'] = df.groupby('team')['scored'].rolling(w, min_periods=1).mean().reset_index(0, drop=True).shift(1)
        df[f'conceded_rolling_{w}'] = df.groupby('team')['conceded'].rolling(w, min_periods=1).mean().reset_index(0, drop=True).shift(1)
        # Opponent perspective: scored_by_opp = conceded_by_team
        df[f'opponent_scored_rolling_{w}'] = df.groupby('opponent')['conceded'].rolling(w, min_periods=1).mean().reset_index(0, drop=True).shift(1)
        df[f'opponent_conceded_rolling_{w}'] = df.groupby('opponent')['scored'].rolling(w, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    
    df.fillna(0, inplace=True)  # Manejo NaN iniciales
    print("✅ Rolling computados + NaN → 0")
    return df

def ensure_rolling_features(df):
    """Auto-verifica y crea si no existen."""
    required = [f'{base}_rolling_{w}' for w in ROLLING_WINDOWS for base in ['scored', 'conceded', 'opponent_scored', 'opponent_conceded']]
    missing = set(required) - set(df.columns)
    if missing:
        print(f"🔧 Creando {len(missing)} columnas faltantes: {missing}")
        df = compute_rolling_features(df)
    return df

def prepare_features(df, sport):
    """Prepara features: auto-rolling + más."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = ensure_rolling_features(df)  # ← CAMBIO CLAVE
    
    # Rest days
    df_sorted = df.sort_values(['team', 'date'])
    df_sorted['last_match'] = df_sorted.groupby('team')['date'].shift(1)
    df_sorted['rest_days'] = (df_sorted['date'] - df_sorted['last_match']).dt.days.fillna(MIN_REST_DAYS)
    df['rest_days'] = df_sorted['rest_days']
    df['rest_diff'] = df['rest_days'] - df_sorted.groupby('opponent')['rest_days'].transform('mean')  # Approx
    df['rest_advantage'] = (df['rest_days'] >= MIN_REST_DAYS).astype(float) * 0.1
    
    # Liga MX & home adv
    df['is_liga_mx'] = df['league'].str.contains('MX|mx', na=False).astype(int)
    df['home_advantage'] = df['is_home'] * (HOME_ADV_MX * df['is_liga_mx'] + 0.10 * (1 - df['is_liga_mx']))
    
    # Strength diffs
    for w in ROLLING_WINDOWS:
        df[f'strength_diff_{w}'] = (
            (df[f'scored_rolling_{w}'] - df[f'conceded_rolling_{w}']) -
            (df[f'opponent_scored_rolling_{w}'] - df[f'opponent_conceded_rolling_{w}'])
        )
    
    df.fillna(0, inplace=True)
    # Opcional: df = df.dropna(subset=['target'])  # Elimina iniciales si prefieres
    return df

class PoissonGoalModel:
    def fit(self, df):
        self.avg_scored = df['scored'].mean()
        return self
    
    def calculate_1x2_prob(self, h_lambda, a_lambda, max_g=10):
        home_p = [poisson.pmf(i, h_lambda) for i in range(max_g)]
        away_p = [poisson.pmf(i, a_lambda) for i in range(max_g)]
        p_home = sum(home_p[i] * sum(away_p[:i]) for i in range(1, max_g))
        p_draw = sum(home_p[i] * away_p[i] for i in range(max_g))
        p_away = 1 - p_home - p_draw
        return np.array([p_home, p_draw, p_away])

class MultiSportEnsemble:
    def __init__(self, sport):
        self.sport = sport
        self.n_classes = 3 if sport == 'soccer' else 2
        self.poisson = PoissonGoalModel()
        self.xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_classes=self.n_classes)
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        self.feature_cols = None
    
    def fit(self, df):
        df_feat = prepare_features(df, self.sport)
        self.feature_cols = [col for col in df_feat.columns if any(kw in col.lower() for kw in ['rolling', 'advantage', 'rest', 'strength'])]
        self.feature_cols = [col for col in self.feature_cols if col not in ['target', 'date', 'team', 'opponent', 'league']]  # Clean
        X = df_feat[self.feature_cols].fillna(0)
        y = df_feat['target'].values
        self.poisson.fit(df_feat)
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        print(f"✅ {self.sport} fitted | Features: {len(self.feature_cols)}")
        return self
    
    def predict_proba(self, df_input):
        if self.feature_cols is None:
            raise ValueError("Fit first!")
        X_ml = df_input[self.feature_cols].fillna(0)
        probs_ml = (self.xgb.predict_proba(X_ml) + self.rf.predict_proba(X_ml)) / 2
        
        # Poisson approx
        h_lambda = X_ml['scored_rolling_5'].mean()
        a_lambda = X_ml['opponent_scored_rolling_5'].mean()
        if self.sport == 'soccer':
            poisson_p = self.poisson.calculate_1x2_prob(h_lambda, a_lambda)
            ensemble = 0.4 * poisson_p + 0.6 * probs_ml.mean(axis=0)
        else:
            p_home = 1 - poisson.cdf(0, h_lambda)  # Approx no draw
            poisson_p = np.array([p_home, 1-p_home])
            ensemble = 0.4 * poisson_p[:self.n_classes] + 0.6 * probs_ml.mean(axis=0)
        return ensemble / ensemble.sum(axis=-1, keepdims=True)  # Normalize

def time_series_validation(df, sport, n_splits=5):
    """TS validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    df_feat = prepare_features(df, sport)
    feature_cols = [col for col in df_feat if any(kw in col for kw in ['rolling','advantage','rest','strength'])]
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['target']
    model = MultiSportEnsemble(sport)
    accs = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(df.iloc[train_idx])
        pred = model.predict_proba(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], np.argmax(pred, axis=1)))
    mean_acc = np.mean(accs)
    print(f"📊 {sport.upper()} TS Acc: {mean_acc:.3f} (+/- {np.std(accs):.3f})")
    return mean_acc

def train_model(sport, data_path='data/match_history.csv', save_path='models'):
    df = load_data(data_path) if sport=='soccer' else generate_sample_data(sport)  # Custom path for soccer CSV
    print(f"🧹 Datos raw: {len(df)}")
    time_series_validation(df.copy(), sport)
    model = MultiSportEnsemble(sport).fit(df)
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(model, open(f'{save_path}/{sport}_model.pkl', 'wb'))
    print(f"✅ {sport.upper()} auto-saved")

def train_all():
    for sport in ['soccer', 'nba', 'mlb']:
        train_model(sport)

if __name__ == '__main__':
    train_all()
    print("🎯 Listo! Rolling auto-computados.")
