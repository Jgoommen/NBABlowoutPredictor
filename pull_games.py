from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

##pulls games from a specified season
def pull_season_games(season):
    print(f"Pulling games for {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    df = gamefinder.get_data_frames()[0]
    df['SEASON'] = season
    time.sleep(1)
    return df
seasons = ['2023-24', '2024-25', '2025-26']
all_dfs = [pull_season_games(s) for s in seasons]
games_raw = pd.concat(all_dfs, ignore_index = True)
print(f"Total team-game rows pulled: {len(games_raw)}")
print(games_raw.columns.tolist())
print(games_raw.head())
##saving the pulled games as a csv
games_raw.to_csv('nbaGamesDataRaw.csv', index=False)
print("Saved nbaGamesDataRaw.csv")