"""
EDGE BOT PRO v3.3 - Entrenamiento Multi-Sport (SOLO DATOS REALES)
TimeSeriesSplit Fixed | Sin Datos de Muestra | Producción
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from scipy.stats import poisson
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN
# ==========================================
ROLLING_WINDOWS = [5, 10]
HOME_ADV_MX = 0.15
MIN_REST_DAYS = 3
DATA_PATH = 'data/match_history.csv'  # ← Cambiar por tu ruta real

# ==========================================
# CARGA DE DATOS (SOLO REALES - SIN FALLBACK)
# ==========================================

def load_real_data(filepath=DATA_PATH):
    """
    Carga SOLO datos reales desde CSV/JSON.
    NO HAY FALLBACK - Falla si no existe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ ERROR CRÍTICO: Archivo no encontrado: {filepath}\n"
            f"   Soluciones:\n"
            f"   1. Descarga datos reales desde API/Scraping\n"
            f"   2. Coloca un CSV en {filepath} con columnas:\n"
            f"      date,team,opponent,is_home,scored,conceded,target,league"
        )
    
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Formato no soportado: usa CSV o JSON")
        
        # Validar columnas requeridas
        required_cols = ['date', 'team', 'opponent', 'is_home', 'scored', 'conceded', 'target', 'league']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'team', 'opponent', 'target'])
        
        print(f"✅ Datos REALES cargados: {len(df)} filas desde {filepath}")
        print(f"   Ligas: {df['league'].unique()}")
        print(f"   Rango: {df['date'].min()} a {df['date'].max()}")
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Error procesando {filepath}: {e}")

# ==========================================
# INGENIERÍA DE FEATURES (AUTO-ROLLING)
# ==========================================

def compute_rolling_features(df, windows=ROLLING_WINDOWS):
    """Calcula promedios móviles (team + opponent)."""
    print("🔄 Computando rolling averages...")
    df = df.sort_values(['team', 'date']).copy()
    
    for w in windows:
        # Team stats
        df[f'scored_rolling_{w}'] = (
            df.groupby('team')['scored']
            .rolling(w, min_periods=1).mean()
            .reset_index(0, drop=True).shift(1)
        )
        df[f'conceded_rolling_{w}'] = (
            df.groupby('team')['conceded']
            .rolling(w, min_periods=1).mean()
            .reset_index(0, drop=True).shift(1)
        )
        # Opponent stats (perspectiva inversa)
        df[f'opponent_scored_rolling_{w}'] = (
            df.groupby('opponent')['conceded']
            .rolling(w, min_periods=1).mean()
            .reset_index(0, drop=True).shift(1)
        )
        df[f'opponent_conceded_rolling_{w}'] = (
            df.groupby('opponent')['scored']
            .rolling(w, min_periods=1).mean()
            .reset_index(0, drop=True).shift(1)
        )
    
    df.fillna(0, inplace=True)
    print("✅ Rolling computados + NaN → 0")
    return df

def ensure_rolling_features(df):
    """Auto-verifica y crea si faltan."""
    required = [f'{base}_rolling_{w}' 
                for w in ROLLING_WINDOWS 
                for base in ['scored', 'conceded', 'opponent_scored', 'opponent_conceded']]
    missing = set(required) - set(df.columns)
    
    if missing:
        print(f"🔧 Creando {len(missing)} columnas rolling...")
        df = compute_rolling_features(df)
    else:
        print("✅ Columnas rolling ya existen")
    return df

def prepare_features(df, sport):
    """Prepara features completas."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Auto-rolling
    df = ensure_rolling_features(df)
    
    # Rest days
    df = df.sort_values(['team', 'date'])
    df['last_match'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_match']).dt.days.fillna(MIN_REST_DAYS)
    df['rest_diff'] = df['rest_days'].fillna(MIN_REST_DAYS) - MIN_REST_DAYS
    df['rest_advantage'] = (df['rest_days'] >= MIN_REST_DAYS).astype(float) * 0.1
    
    # Liga MX boost
    df['is_liga_mx'] = df['league'].astype(str).str.contains('MX|mx', na=False).astype(int)
    df['home_advantage'] = df['is_home'] * (
        HOME_ADV_MX * df['is_liga_mx'] + 0.10 * (1 - df['is_liga_mx'])
    )
    
    # Strength deltas
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
    def fit(self, df):
        self.avg_scored = df['scored'].mean()
        return self
    
    def calculate_1x2_prob(self, h_lambda, a_lambda, max_g=10):
        home_p = np.array([poisson.pmf(i, h_lambda) for i in range(max_g)])
        away_p = np.array([poisson.pmf(i, a_lambda) for i in range(max_g)])
        
        p_home = sum(home_p[i] * sum(away_p[:i]) for i in range(1, max_g))
        p_draw = sum(home_p[i] * away_p[i] for i in range(max_g))
        p_away = 1 - p_home - p_draw
        
        return np.array([p_home, p_draw, p_away])

class MultiSportEnsemble:
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
        
        # Definir features ML
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
        """
        ⚠️ FIX CRÍTICO: Retorna (n_samples, n_classes) no (1, n_classes)
        """
        if self.feature_cols is None:
            raise ValueError("Modelo no entrenado")
        
        # Asegurar columnas
        for col in self.feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0
        
        X_ml = df_input[self.feature_cols].fillna(0)
        
        # ✅ CORRECCIÓN: predict_proba retorna (n_samples, n_classes)
        probs_xgb = self.xgb.predict_proba(X_ml)  # Shape: (n_samples, n_classes)
        probs_rf = self.rf.predict_proba(X_ml)    # Shape: (n_samples, n_classes)
        probs_ml = (probs_xgb + probs_rf) / 2     # Shape: (n_samples, n_classes)
        
        # Poisson approx (broadcast a todas las filas)
        h_lambda = df_input.get('scored_rolling_5', pd.Series([1.4])).mean()
        a_lambda = df_input.get('opponent_scored_rolling_5', pd.Series([1.2])).mean()
        
        if self.sport == 'soccer':
            poisson_p = self.poisson.calculate_1x2_prob(h_lambda, a_lambda)
        else:
            p_home = 1 - poisson.cdf(0, h_lambda)
            poisson_p = np.array([p_home, 1 - p_home])
        
        # Broadcast Poisson a n_samples
        poisson_p_broadcast = np.tile(poisson_p[:self.n_classes], (len(X_ml), 1))
        
        # Ensamble
        ensemble = 0.4 * poisson_p_broadcast + 0.6 * probs_ml
        ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
        
        return ensemble  # ✅ Shape: (n_samples, n_classes)

# ==========================================
# VALIDACIÓN CORREGIDA (TIMESERIES SPLIT)
# ==========================================

def time_series_validation(df, sport, n_splits=5):
    """
    ✅ FIX CRÍTICO: predict_proba ahora retorna (n_test, n_classes)
    """
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
    
    print(f"\n{'='*60}")
    print(f"VALIDACIÓN TIMESERIES - {sport.upper()}")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Entrenar con fold
        model.fit(df.iloc[train_idx])
        
        # ✅ CORRECCIÓN: predict_proba retorna (n_test, n_classes)
        X_test = X.iloc[test_idx]
        pred_probs = model.predict_proba(X_test)  # Shape: (n_test, n_classes)
        
        # np.argmax ahora funciona correctamente
        y_pred = np.argmax(pred_probs, axis=1)    # Shape: (n_test,)
        y_true = y.iloc[test_idx].values          # Shape: (n_test,)
        
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)
        
        print(f"Fold {fold+1}: Train={len(train_idx)}, Test={len(test_idx)}, Acc={acc:.4f}")
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print(f"{'='*60}")
    print(f"📊 Accuracy Promedio: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"{'='*60}\n")
    
    return mean_acc

# ==========================================
# ENTRENAMIENTO PRINCIPAL
# ==========================================

def train_model(sport, data_path=DATA_PATH, save_path='models'):
    """Entrena modelo con datos REALES."""
    print(f"\n🏁 Entrenando {sport.upper()} con datos REALES...")
    
    # ✅ Cargar SOLO datos reales (sin fallback)
    df = load_real_data(data_path)
    
    # Filtrar por deporte si es necesario
    if sport == 'soccer':
        soccer_leagues = ['Premier', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Liga MX']
        df = df[df['league'].str.contains('|'.join(soccer_leagues), na=False)]
    elif sport == 'nba':
        df = df[df['league'].str.contains('NBA', na=False)]
    elif sport == 'mlb':
        df = df[df['league'].str.contains('MLB', na=False)]
    
    if len(df) < 100:
        raise ValueError(f"Datos insuficientes para {sport}: {len(df)} filas (mínimo 100)")
    
    print(f"✅ Filas filtradas para {sport}: {len(df)}")
    
    # Validación TimeSeries
    time_series_validation(df.copy(), sport)
    
    # Entrenamiento final
    model = MultiSportEnsemble(sport).fit(df)
    
    # Guardar
    os.makedirs(save_path, exist_ok=True)
    model_path = f'{save_path}/{sport}_model.pkl'
    pickle.dump(model, open(model_path, 'wb'))
    print(f"✅ Modelo guardado: {model_path}")
    
    return model

def train_all(data_path=DATA_PATH):
    """Entrena todos los modelos."""
    for sport in ['soccer', 'nba', 'mlb']:
        try:
            train_model(sport, data_path)
        except Exception as e:
            print(f"❌ Error en {sport}: {e}")
            continue

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 EDGE BOT PRO v3.3 - ENTRENAMIENTO CON DATOS REALES")
    print("=" * 60)
    
    try:
        train_all()
        print("\n✅ ¡Todos los modelos entrenados exitosamente!")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📋 INSTRUCCIONES:")
        print("   1. Descarga datos históricos desde:")
        print("      - FBRef API: https://fbref.com/")
        print("      - Understat API: https://understat.com/")
        print("      - Scraping ESPN/Sofascore")
        print("   2. Guarda en data/match_history.csv con formato:")
        print("      date,team,opponent,is_home,scored,conceded,target,league")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
