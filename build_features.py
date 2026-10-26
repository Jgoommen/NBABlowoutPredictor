import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
import time

df = pd.read_csv("nbaGamesDataRaw.csv")
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
# sorts data by team and date
df = df.sort_values(['TEAM_NAME', 'GAME_DATE']).reset_index(drop=True)

df['IS_HOME'] = (~df['MATCHUP'].str.contains('@')).astype(int)
#statistical features
stat_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS']
# rolling last 10 game averages
for col in stat_cols:
    df[f'{col}_avg_l10'] = (
        df.groupby('TEAM_NAME')[col]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
# opponent points allowed feature
opp_pts = df[['GAME_ID', 'TEAM_NAME', 'PTS']].rename(columns={'TEAM_NAME': 'OPPONENT', 'PTS': 'PTS_ALLOWED'})

def extract_opponent(row):
    parts = row['MATCHUP'].split(' ')
    return parts[-1]

df['OPP_ABBR'] = df.apply(extract_opponent, axis=1)

abbr_to_name = df[['TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates().set_index('TEAM_ABBREVIATION')['TEAM_NAME'].to_dict()
df['OPPONENT'] = df['OPP_ABBR'].map(abbr_to_name)

df = df.merge(opp_pts, left_on=['GAME_ID', 'OPPONENT'], right_on=['GAME_ID', 'OPPONENT'], how='left')

df['PTS_ALLOWED_avg_l10'] = (
    df.groupby('TEAM_NAME')['PTS_ALLOWED']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)
## days of rest a team as
df['DAYS_REST'] = df.groupby(['TEAM_NAME','SEASON'])['GAME_DATE'].diff().dt.days
df['DAYS_REST'] = df['DAYS_REST'].fillna(7)
df['BACK_TO_BACK'] = (df['DAYS_REST'] <= 1).astype(int)

print(f"Rows after team-level feature engineering: {len(df)}")



seasons = df['SEASON'].unique().tolist()
roster_frames = []
# pulling player stats for specified seasons
for season in seasons:
    print(f"Pulling player season stats for {season}...")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame',
    ).get_data_frames()[0]
    stats['SEASON'] = season
    roster_frames.append(stats)
    time.sleep(1)

player_stats = pd.concat(roster_frames, ignore_index=True)
# measureing team roster stength using aggregated team player stats
team_roster_strength = player_stats.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg(
    roster_avg_pts=('PTS', 'mean'),
    roster_top_scorer_pts=('PTS', 'max'),
    roster_total_pts=('PTS', 'sum'),
    roster_avg_fg_pct=('FG_PCT', 'mean'),
    roster_avg_min=('MIN', 'mean'),
).reset_index()

print(f"Roster strength computed for {len(team_roster_strength)} teams")
print(team_roster_strength.head())



df = df.merge(team_roster_strength, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')

print(f"Final rows: {len(df)}")
print(df[['TEAM_NAME','GAME_DATE','roster_avg_pts','roster_top_scorer_pts']].head())

df.to_csv('nbaTeamFeatures.csv', index=False)
print("Saved nbaTeamFeatures.csv")