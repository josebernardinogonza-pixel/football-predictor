"""
EDGE BOT PRO v3.2 - Entrenamiento Multi-Sport Optimizado
Auto-Rolling Features (CSV/JSON) | TimeSeriesSplit | Ensemble Poisson + XGBoost + RF
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

# ==========================================
# CONFIGURACIÓN
# ==========================================
ROLLING_WINDOWS = [5, 10]
HOME_ADV_MX = 0.15
MIN_REST_DAYS = 3

# ==========================================
# CARGA DE DATOS (CSV/JSON)
# ==========================================

def load_data(filepath='data/match_history.csv'):
    """Carga CSV o JSON automáticamente."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Formato no soportado: usa CSV o JSON")
        print(f"✅ Datos cargados: {len(df)} filas desde {filepath}")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        print("⚠️ Archivo no encontrado → Generando datos de muestra")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading {filepath}: {e} → Generando datos de muestra")
        return generate_sample_data()

def generate_sample_data(sport='soccer', n_samples=2000):
    """Genera datos históricos realistas por deporte."""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='2D')
    
    if sport == 'soccer':
        leagues = ['Premier League', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Liga MX']
        teams = [f'Team{i}' for i in range(20)]
        scored_lambda, conceded_lambda = 1.4, 1.2
        data = {
            'date': dates,
            'league': np.random.choice(leagues, n_samples),
            'team': [teams[i % 20] for i in range(n_samples)],
            'opponent': [teams[(i + 1) % 20] for i in range(n_samples)],
            'is_home': np.random.choice([0, 1], n_samples),
            'scored': np.random.poisson(scored_lambda, n_samples),
            'conceded': np.random.poisson(conceded_lambda, n_samples),
        }
        data['target'] = np.where(data['scored'] > data['conceded'], 0,
                                  np.where(data['scored'] == data['conceded'], 1, 2))
    elif sport == 'nba':
        teams = [f'NBA_Team{i}' for i in range(30)]
        data = {
            'date': dates, 'league': ['NBA'] * n_samples,
            'team': [teams[i % 30] for i in range(n_samples)],
            'opponent': [teams[(i + 1) % 30] for i in range(n_samples)],
            'is_home': np.random.choice([0, 1], n_samples),
            'scored': np.random.poisson(110, n_samples),
            'conceded': np.random.poisson(108, n_samples),
        }
        data['target'] = np.where(data['scored'] > data['conceded'], 0, 2)
    elif sport == 'mlb':
        teams = [f'MLB_Team{i}' for i in range(30)]
        data = {
            'date': dates, 'league': ['MLB'] * n_samples,
            'team': [teams[i % 30] for i in range(n_samples)],
            'opponent': [teams[(i + 1) % 30] for i in range(n_samples)],
            'is_home': np.random.choice([0, 1], n_samples),
            'scored': np.random.poisson(4.6, n_samples),
            'conceded': np.random.poisson(4.4, n_samples),
        }
        data['target'] = np.where(data['scored'] > data['conceded'], 0, 2)
    else:
        raise ValueError(f"Deporte no soportado: {sport}")
    
    df = pd.DataFrame(data).sort_values('date')
    return df

# ==========================================
# INGENIERÍA DE FEATURES (AUTO-ROLLING)
# ==========================================

def compute_rolling_features(df, windows=ROLLING_WINDOWS):
    """Calcula promedios móviles para team y opponent."""
    print("🔄 Computando rolling averages...")
    df = df.sort_values(['team', 'date']).copy()
    
    for w in windows:
        # Team perspective: goles anotados/concedidos por el equipo
        df[f'scored_rolling_{w}'] = (
            df.groupby('team')['scored']
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
            .shift(1)  # Evitar data leakage
        )
        df[f'conceded_rolling_{w}'] = (
            df.groupby('team')['conceded']
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
            .shift(1)
        )
        # Opponent perspective: lo que el oponente anotó/recibió
        df[f'opponent_scored_rolling_{w}'] = (
            df.groupby('opponent')['conceded']
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
            .shift(1)
        )
        df[f'opponent_conceded_rolling_{w}'] = (
            df.groupby('opponent')['scored']
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
            .shift(1)
        )
    
    df.fillna(0, inplace=True)  # Manejo de NaN iniciales
    print("✅ Rolling computados + NaN → 0")
    return df

def ensure_rolling_features(df):
    """Verifica y crea columnas rolling si no existen."""
    required = [f'{base}_rolling_{w}' 
                for w in ROLLING_WINDOWS 
                for base in ['scored', 'conceded', 'opponent_scored', 'opponent_conceded']]
    missing = set(required) - set(df.columns)
    
    if missing:
        print(f"🔧 Creando {len(missing)} columnas faltantes: {missing}")
        df = compute_rolling_features(df)
    else:
        print("✅ Columnas rolling ya existen")
    return df

def prepare_features(df, sport):
    """Prepara todas las features para el modelo."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Auto-computar rolling si faltan
    df = ensure_rolling_features(df)
    
    # Días de descanso
    df = df.sort_values(['team', 'date'])
    df['last_match'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_match']).dt.days.fillna(MIN_REST_DAYS)
    df['rest_diff'] = df['rest_days'].fillna(MIN_REST_DAYS) - MIN_REST_DAYS
    
    # Ventaja por descanso
    df['rest_advantage'] = (df['rest_days'] >= MIN_REST_DAYS).astype(float) * 0.1
    
    # Liga MX detection + home advantage
    df['is_liga_mx'] = df['league'].astype(str).str.contains('MX|mx', na=False).astype(int)
    df['home_advantage'] = df['is_home'] * (
        HOME_ADV_MX * df['is_liga_mx'] + 0.10 * (1 - df['is_liga_mx'])
    )
    
    # Diferencia de fortaleza entre equipos
    for w in ROLLING_WINDOWS:
        df[f'strength_diff_{w}'] = (
            (df[f'scored_rolling_{w}'] - df[f'conceded_rolling_{w}']) -
            (df[f'opponent_scored_rolling_{w}'] - df[f'opponent_conceded_rolling_{w}'])
        )
    
    df.fillna(0, inplace=True)
    return df

# ==========================================
# MODELOS
# ==========================================

class PoissonGoalModel:
    """Modelo de Poisson para distribución de goles/puntos."""
    
    def fit(self, df):
        self.avg_scored = df['scored'].mean()
        self.avg_conceded = df['conceded'].mean()
        return self
    
    def calculate_1x2_prob(self, h_lambda, a_lambda, max_g=10):
        """Calcula probabilidades 1X2 usando Poisson."""
        home_p = np.array([poisson.pmf(i, h_lambda) for i in range(max_g)])
        away_p = np.array([poisson.pmf(i, a_lambda) for i in range(max_g)])
        
        p_home = sum(home_p[i] * sum(away_p[:i]) for i in range(1, max_g))
        p_draw = sum(home_p[i] * away_p[i] for i in range(max_g))
        p_away = 1 - p_home - p_draw
        
        return np.array([p_home, p_draw, p_away])
    
    def calculate_ml_prob(self, h_lambda, a_lambda):
        """Calcula probabilidad simple (NBA/MLB - sin empate)."""
        p_home = 1 - poisson.cdf(0, h_lambda)
        return np.array([p_home, 1 - p_home])

class MultiSportEnsemble:
    """Ensamble Poisson + XGBoost + RandomForest."""
    
    def __init__(self, sport):
        self.sport = sport
        self.n_classes = 3 if sport == 'soccer' else 2
        self.poisson = PoissonGoalModel()
        self.xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        self.feature_cols = None
    
    def fit(self, df):
        df_feat = prepare_features(df, self.sport)
        
        # Definir columnas de features
        self.feature_cols = [
            col for col in df_feat.columns 
            if any(kw in col.lower() for kw in ['rolling', 'advantage', 'rest', 'strength'])
        ]
        self.feature_cols = [
            col for col in self.feature_cols 
            if col not in ['target', 'date', 'team', 'opponent', 'league', 'is_home']
        ]
        
        X = df_feat[self.feature_cols].fillna(0)
        y = df_feat['target'].values
        
        self.poisson.fit(df_feat)
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        
        print(f"✅ {self.sport} fitted | Features: {len(self.feature_cols)}")
        return self
    
    def predict_proba(self, df_input):
        if self.feature_cols is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")
        
        # Asegurar que todas las columnas existan
        for col in self.feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0
        
        X_ml = df_input[self.feature_cols].fillna(0)
        probs_ml = (self.xgb.predict_proba(X_ml) + self.rf.predict_proba(X_ml)) / 2
        
        # Poisson approximation
        h_lambda = df_input.get('scored_rolling_5', pd.Series([1.4])).mean()
        a_lambda = df_input.get('opponent_scored_rolling_5', pd.Series([1.2])).mean()
        
        if self.sport == 'soccer':
            poisson_p = self.poisson.calculate_1x2_prob(h_lambda, a_lambda)
            # Asegurar misma dimensión
            poisson_p = poisson_p[:self.n_classes]
        else:
            poisson_p = self.poisson.calculate_ml_prob(h_lambda, a_lambda)
        
        # Combinar ensamble (40% Poisson, 60% ML)
        ensemble = 0.4 * poisson_p + 0.6 * probs_ml.mean(axis=0)
        
        # Normalizar
        ensemble = ensemble / ensemble.sum()
        return ensemble.reshape(1, -1)

# ==========================================
# VALIDACIÓN (TIMESERIES SPLIT)
# ==========================================

def time_series_validation(df, sport, n_splits=5):
    """Validación cronológica sin data leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    df_feat = prepare_features(df, sport)
    
    feature_cols = [
        col for col in df_feat.columns 
        if any(kw in col for kw in ['rolling', 'advantage', 'rest', 'strength'])
    ]
    feature_cols = [col for col in feature_cols if col not in ['target', 'date', 'team', 'opponent', 'league']]
    
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['target']
    
    model = MultiSportEnsemble(sport)
    accs = []
    
    for train_idx, test_idx in tscv.split(X):
        model.fit(df.iloc[train_idx])
        pred = model.predict_proba(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], np.argmax(pred, axis=1)))
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"📊 {sport.upper()} TS Acc: {mean_acc:.3f} (+/- {std_acc:.3f})")
    return mean_acc

# ==========================================
# ENTRENAMIENTO PRINCIPAL
# ==========================================

def train_model(sport, data_path='data/match_history.csv', save_path='models'):
    """Entrena y guarda modelo por deporte."""
    print(f"\n🏁 Entrenando {sport.upper()}...")
    
    # Cargar datos (soccer usa CSV, otros genera sample)
    if sport == 'soccer':
        df = load_data(data_path)
    else:
        df = generate_sample_data(sport)
    
    print(f"🧹 Datos raw: {len(df)} filas")
    
    # Validación TimeSeries
    time_series_validation(df.copy(), sport)
    
    # Entrenamiento final
    model = MultiSportEnsemble(sport).fit(df)
    
    # Auto-guardado
    os.makedirs(save_path, exist_ok=True)
    model_path = f'{save_path}/{sport}_model.pkl'
    pickle.dump(model, open(model_path, 'wb'))
    print(f"✅ {sport.upper()} modelo guardado: {model_path}")
    
    return model

def train_all():
    """Entrena todos los modelos (soccer, nba, mlb)."""
    for sport in ['soccer', 'nba', 'mlb']:
        train_model(sport)

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 EDGE BOT PRO v3.2 - ENTRENAMIENTO DE MODELOS")
    print("=" * 60)
    train_all()
    print("\n🎯 ¡Todos los modelos entrenados y guardados!")
    print("   Ejecuta: python bot.py")
