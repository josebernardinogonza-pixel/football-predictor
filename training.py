"""
EDGE BOT PRO v3.0 - Entrenamiento Multi-Sport Optimizado
Soporte: Soccer (6 ligas), NBA, MLB | Auto-Save Modelos por Deporte
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
import warnings
warnings.filterwarnings('ignore')

# Config
ROLLING_WINDOWS = [5, 10]
HOME_ADV_MX = 0.15
MIN_REST_DAYS = 3

def generate_sample_data(sport, n_samples=2000):
    """Datos realistas por deporte"""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    leagues_soccer = ['Premier', 'LaLiga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Liga MX']
    
    if sport == 'soccer':
        scored_lambda, conceded_lambda = 1.4, 1.2
        data = {
            'date': dates, 'league': np.random.choice(leagues_soccer, n_samples),
            'team': [f'Team{i}' for i in range(20)] * (n_samples//20),
            'opponent': [f'Team{j}' for j in range(20)] * (n_samples//20),
            'is_home': np.random.choice([0,1], n_samples),
            'scored': np.random.poisson(scored_lambda, n_samples),
            'conceded': np.random.poisson(conceded_lambda, n_samples),
        }
        data['target'] = np.where(data['scored'] > data['conceded'], 0, np.where(data['scored'] == data['conceded'], 1, 2))
    elif sport == 'nba':
        scored_lambda, conceded_lambda = 110, 108
        data = {'date': dates, 'league': ['NBA']*n_samples, 'team': [f'Team{i}' for i in range(30)]*(n_samples//30),
                'opponent': [f'Team{j}' for j in range(30)]*(n_samples//30), 'is_home': np.random.choice([0,1], n_samples),
                'scored': np.random.poisson(scored_lambda, n_samples), 'conceded': np.random.poisson(conceded_lambda, n_samples)}
        data['target'] = np.where(data['scored'] > data['conceded'], 0, 2)  # No draw
    elif sport == 'mlb':
        scored_lambda, conceded_lambda = 4.6, 4.4
        data = {'date': dates, 'league': ['MLB']*n_samples, 'team': [f'Team{i}' for i in range(30)]*(n_samples//30),
                'opponent': [f'Team{j}' for j in range(30)]*(n_samples//30), 'is_home': np.random.choice([0,1], n_samples),
                'scored': np.random.poisson(scored_lambda, n_samples), 'conceded': np.random.poisson(conceded_lambda, n_samples)}
        data['target'] = np.where(data['scored'] > data['conceded'], 0, 2)  # No draw
    else:
        raise ValueError("Sport no soportado")
    return pd.DataFrame(data).sort_values('date')

def calculate_rolling_averages(df, team_col, metric_col, windows=[5,10]):
    """Rolling avg múltiples ventanas"""
    for w in windows:
        df[f'{metric_col}_rolling_{w}'] = df.groupby(team_col)[metric_col].rolling(w, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    return df

def add_rest_days_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team', 'date'])
    df['last_match'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_match']).dt.days.fillna(MIN_REST_DAYS)
    return df

def prepare_features(df, sport):
    df = calculate_rolling_averages(df, 'team', 'scored', ROLLING_WINDOWS)
    df = calculate_rolling_averages(df, 'team', 'conceded', ROLLING_WINDOWS)
    df = add_rest_days_feature(df)
    
    # Home/away rolling
    for w in ROLLING_WINDOWS:
        df[f'home_scored_rolling_{w}'] = df[f'scored_rolling_{w}']
        df[f'away_scored_rolling_{w}'] = df[f'opponent_scored_rolling_{w}']  # Approx
        df[f'home_conceded_rolling_{w}'] = df[f'conceded_rolling_{w}']
        df[f'away_conceded_rolling_{w}'] = df[f'opponent_conceded_rolling_{w}']
    
    # Liga MX
    df['is_liga_mx'] = df['league'].str.contains('MX').astype(int)
    df['home_advantage'] = df['is_home'] * (HOME_ADV_MX if sport=='soccer' else 0.12) * df['is_liga_mx']
    
    # Deltas
    for w in ROLLING_WINDOWS:
        df[f'strength_diff_{w}'] = (df[f'scored_rolling_{w}'] - df[f'conceded_rolling_{w}']) - (df.get(f'opponent_scored_rolling_{w}',0) - df.get(f'opponent_conceded_rolling_{w}',0))
    
    df['rest_diff'] = df['rest_days'].diff()  # Approx opp
    df['rest_advantage'] = (df['rest_days'] >= MIN_REST_DAYS).astype(int) * 0.1
    df.fillna(0, inplace=True)
    return df

class MultiSportEnsemble:
    def __init__(self, sport):
        self.sport = sport
        self.n_classes = 3 if sport == 'soccer' else 2
        self.poisson_model = PoissonGoalModel()  # Generic
        self.xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_classes=self.n_classes)
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    def fit(self, df):
        df_feat = prepare_features(df, self.sport)
        feature_cols = [col for col in df_feat.columns if any(kw in col for kw in ['rolling', 'advantage', 'rest', 'strength'])]
        X = df_feat[feature_cols]
        y = df_feat['target']
        self.poisson_model.fit(df_feat)
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        return self
    
    def predict_proba(self, df_input):
        # Poisson probs
        xg_home = df_input.get('xg_home', 1.4).iloc[0]
        xg_away = df_input.get('xg_away', 1.1).iloc[0]
        poisson_p = self.poisson_model.calculate_1x2_prob(xg_home, xg_away) if self.sport=='soccer' else [1-sum(poisson.pmf(range(11), xg_away)/sum(poisson.pmf(range(11), xg_away)) > poisson.pmf(range(11), xg_home)/sum(poisson.pmf(range(11), xg_home))),  ...]  # Simplified home prob approx no draw
        # ML probs avg
        probs_ml = (self.xgb.predict_proba(df_input[feature_cols]) + self.rf.predict_proba(df_input[feature_cols])) / 2
        return 0.4 * np.array(poisson_p[:self.n_classes]) + 0.6 * probs_ml.mean(axis=0)  # Normalized

# Poisson class (adapted)
class PoissonGoalModel:
    def fit(self, df): self.avg_xg_home = df['scored'].mean() * 0.6  # Approx
    def calculate_1x2_prob(self, h_xg, a_xg):
        home_p = np.array([poisson.pmf(i, h_xg) for i in range(10)])
        away_p = np.array([poisson.pmf(i, a_xg) for i in range(10)])
        p_home = sum(home_p[i] * away_p[j] for i in range(10) for j in range(i))
        p_draw = sum(home_p[i] * away_p[i] for i in range(10))
        p_away = 1 - p_home - p_draw
        return [p_home, p_draw, p_away]

def time_series_validation(df, sport, n_splits=5):
    model = MultiSportEnsemble(sport)
    tscv = TimeSeriesSplit(n_splits)
    X = prepare_features(df, sport).drop(['target'], axis=1)
    y = df['target']
    accs = [accuracy_score(y.iloc[test], model.fit(X.iloc[train]).predict(X.iloc[test])) for train, test in tscv.split(X)]
    print(f"{sport.upper()} Acc: {np.mean(accs):.3f}")
    return np.mean(accs)

def train_model(sport, save_path='models'):
    df = generate_sample_data(sport)
    time_series_validation(df, sport)
    model = MultiSportEnsemble(sport).fit(df)
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(model, open(f'{save_path}/{sport}_model.pkl', 'wb'))
    print(f"✅ {sport.upper()} model auto-saved: {save_path}/{sport}_model.pkl")

def train_all():
    for sport in ['soccer', 'nba', 'mlb']:
        train_model(sport)

if __name__ == '__main__':
    train_all()
    print("🎯 Todos los modelos entrenados y guardados. Listo para bot.py!")
