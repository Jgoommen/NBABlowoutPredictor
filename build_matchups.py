import pandas as pd

df = pd.read_csv('nbaTeamFeatures.csv')

df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# columns we want on both the home and away side of each matchup row
# data for home and away team
feature_cols = [
    'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'PTS', 'IS_HOME',
    'PTS_avg_l10', 'FG_PCT_avg_l10', 'FG3_PCT_avg_l10', 'FT_PCT_avg_l10',
    'REB_avg_l10', 'AST_avg_l10', 'TOV_avg_l10', 'PLUS_MINUS_avg_l10',
    'PTS_ALLOWED_avg_l10', 'DAYS_REST', 'BACK_TO_BACK',
    'roster_avg_pts', 'roster_top_scorer_pts', 'roster_total_pts',
    'roster_avg_fg_pct', 'roster_avg_min',
]
# separate home and away games
home_rows = df[df['IS_HOME'] == 1][feature_cols]
away_rows = df[df['IS_HOME'] == 0][feature_cols]
# merging into matchups using game_id so that each game because a single row
matchups = home_rows.merge(
    away_rows,
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)

# target variables
matchups['POINT_DIFF'] = matchups['PTS_HOME'] - matchups['PTS_AWAY']
matchups['BLOWOUT'] = (matchups['POINT_DIFF'].abs() >= 15).astype(int)
matchups['HOME_WIN'] = (matchups['POINT_DIFF'] > 0).astype(int)

# drop rows without enough rolling history yet (early-season games)
model_feature_cols = [c for c in matchups.columns if c.endswith('_HOME') or c.endswith('_AWAY')]
model_feature_cols = [c for c in model_feature_cols if c not in ('TEAM_NAME_HOME','TEAM_NAME_AWAY','GAME_DATE_HOME','GAME_DATE_AWAY','IS_HOME_HOME','IS_HOME_AWAY','PTS_HOME','PTS_AWAY')]

before = len(matchups)
matchups = matchups.dropna(subset=model_feature_cols)
after = len(matchups)

print(f"Matchup rows before dropping incomplete history: {before}")
print(f"Matchup rows after: {after}")
print(f"Model feature columns ({len(model_feature_cols)}): {model_feature_cols}")

print(matchups[['GAME_DATE_HOME', 'TEAM_NAME_HOME', 'TEAM_NAME_AWAY', 'POINT_DIFF', 'BLOWOUT', 'HOME_WIN']].head(10))

matchups.to_csv('nbaMatchups.csv', index=False)
print("Saved nbaMatchups.csv")