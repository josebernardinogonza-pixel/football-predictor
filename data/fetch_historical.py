"""
Fetcher de Datos Históricos desde ESPN API
Genera data/match_history.csv para training
"""
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
LEAGUES = {
    'soccer': {'eng.1': 'Premier League', 'esp.1': 'LaLiga', 'ita.1': 'Serie A', 
                'ger.1': 'Bundesliga', 'fra.1': 'Ligue 1', 'ligamx': 'Liga MX'},
    'basketball': {'nba': 'NBA'},
    'baseball': {'mlb': 'MLB'}
}

def fetch_league_data(sport, league_key, league_name, days=90):
    """Fetches completed matches from ESPN."""
    url = f"{ESPN_BASE}/{sport}/{league_key}/scoreboard"
    matches = []
    
    try:
        resp = requests.get(url, timeout=15).json()
        events = resp.get('events', [])
        
        for event in events:
            if event['status']['type']['state'] == 'post':
                comp = event['competitions'][0]
                home = comp['competitors'][0]
                away = comp['competitors'][1]
                
                scored_h = int(home.get('score', '0') or 0)
                scored_a = int(away.get('score', '0') or 0)
                
                # Home perspective
                matches.append({
                    'date': event.get('date', ''),
                    'league': league_name,
                    'team': home['team']['name'],
                    'opponent': away['team']['name'],
                    'is_home': 1,
                    'scored': scored_h,
                    'conceded': scored_a,
                    'target': 0 if scored_h > scored_a else (1 if scored_h == scored_a else 2)
                })
                # Away perspective
                matches.append({
                    'date': event.get('date', ''),
                    'league': league_name,
                    'team': away['team']['name'],
                    'opponent': home['team']['name'],
                    'is_home': 0,
                    'scored': scored_a,
                    'conceded': scored_h,
                    'target': 2 if scored_h > scored_a else (1 if scored_h == scored_a else 0)
                })
    except Exception as e:
        print(f"⚠️ Error {league_name}: {e}")
    
    return matches

def fetch_all():
    """Fetches all leagues and saves CSV."""
    print("🔄 Fetching historical data from ESPN...")
    all_matches = []
    
    for sport, leagues in LEAGUES.items():
        for key, name in leagues.items():
            matches = fetch_league_data(sport, key, name)
            all_matches.extend(matches)
            print(f"✅ {name}: {len(matches)} matches")
    
    if not all_matches:
        print("❌ No data fetched")
        return
    
    df = pd.DataFrame(all_matches)
    df = df.sort_values('date')
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/match_history.csv', index=False)
    print(f"\n✅ Saved: data/match_history.csv ({len(df)} rows)")

if __name__ == '__main__':
    fetch_all()
