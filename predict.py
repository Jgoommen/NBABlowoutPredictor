import pandas as pd
import numpy as np
import joblib
import boto3
from scipy.stats import norm

BUCKET_NAME = "student-s3-josh-2026-161012475511-us-east-2-an"
REGION = "us-east-2"

TEAM_FEATURES_S3_KEY = "raw/nbaTeamFeatures.csv"
LOCAL_TEAM_FEATURES_FILE = "nbaTeamFeatures.csv"

MODEL_FILES = {
    "model_point_diff.pkl": "models/model_point_diff.pkl",
    "model_feature_cols.pkl": "models/model_feature_cols.pkl",
    "model_residual_std.pkl": "models/model_residual_std.pkl",
}

BLOWOUT_THRESHOLD = 15


BASE_FEATURE_COLS = [
    'PTS_avg_l10', 'FG_PCT_avg_l10', 'FG3_PCT_avg_l10', 'FT_PCT_avg_l10',
    'REB_avg_l10', 'AST_avg_l10', 'TOV_avg_l10', 'PLUS_MINUS_avg_l10',
    'PTS_ALLOWED_avg_l10', 'DAYS_REST', 'BACK_TO_BACK',
    'roster_avg_pts', 'roster_top_scorer_pts', 'roster_total_pts',
    'roster_avg_fg_pct', 'roster_avg_min',
]

s3 = boto3.client("s3", region_name=REGION)


def download_resources():
    """
    Pulls the trained model, its feature column order, its residual std,
    and each team's latest stats -- all from S3. This is the part that
    satisfies the rubric's "model reads data from S3" requirement at
    prediction/demo time.
    """
    print("Downloading model files from S3...")
    for local_path, s3_key in MODEL_FILES.items():
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"  Downloaded s3://{BUCKET_NAME}/{s3_key} -> {local_path}")

    print(f"Downloading s3://{BUCKET_NAME}/{TEAM_FEATURES_S3_KEY} ...")
    s3.download_file(BUCKET_NAME, TEAM_FEATURES_S3_KEY, LOCAL_TEAM_FEATURES_FILE)

    diff_model = joblib.load("model_point_diff.pkl")
    feature_cols = joblib.load("model_feature_cols.pkl")
    residual_std = joblib.load("model_residual_std.pkl")

    team_features = pd.read_csv(LOCAL_TEAM_FEATURES_FILE)
    team_features['GAME_DATE'] = pd.to_datetime(team_features['GAME_DATE'])

    
    latest = (
        team_features.sort_values('GAME_DATE')
        .groupby('TEAM_NAME')
        .tail(1)
        .set_index('TEAM_NAME')
    )

    return diff_model, feature_cols, residual_std, latest


def get_team_list(team_latest):
    return sorted(team_latest.index.tolist())


def build_matchup_row(home_team, away_team, team_latest, feature_cols):
    """
    Builds a single-row DataFrame shaped exactly like the training data:
    each base stat column duplicated as _HOME and _AWAY, in the same
    column order the model was trained on.
    """
    if home_team not in team_latest.index:
        raise ValueError(f"No stats found for '{home_team}'")
    if away_team not in team_latest.index:
        raise ValueError(f"No stats found for '{away_team}'")

    home_stats = team_latest.loc[home_team]
    away_stats = team_latest.loc[away_team]

    row = {}
    for col in BASE_FEATURE_COLS:
        row[f"{col}_HOME"] = home_stats[col]
        row[f"{col}_AWAY"] = away_stats[col]

    # reindex to guarantee the exact column order the model was trained on
    return pd.DataFrame([row])[feature_cols]


def blowout_probability(predicted_diff, residual_std, threshold=BLOWOUT_THRESHOLD):
    p_home_blowout = 1 - norm.cdf(threshold, loc=predicted_diff, scale=residual_std)
    p_away_blowout = norm.cdf(-threshold, loc=predicted_diff, scale=residual_std)
    return p_home_blowout + p_away_blowout


def recommendation(blowout_prob):
    if blowout_prob < 0.30:
        return "Worth attending -- this projects as a competitive game."
    elif blowout_prob < 0.50:
        return "Toss-up -- could go either way, reasonable watch."
    else:
        return "Likely a blowout -- may not be worth attending live."


def predict_matchup(home_team, away_team, diff_model, feature_cols, residual_std, team_latest):
    X = build_matchup_row(home_team, away_team, team_latest, feature_cols)
    predicted_diff = diff_model.predict(X)[0]
    blowout_prob = blowout_probability(predicted_diff, residual_std)

    predicted_winner = home_team if predicted_diff > 0 else away_team

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_point_diff": round(float(predicted_diff), 1),
        "predicted_winner": predicted_winner,
        "blowout_probability": round(float(blowout_prob), 3),
        "recommendation": recommendation(blowout_prob),
    }


if __name__ == "__main__":
    diff_model, feature_cols, residual_std, team_latest = download_resources()
    teams = get_team_list(team_latest)

    print(f"\n{len(teams)} teams available: {teams}")

    # quick CLI test -- swap these two team names to try others
    home_team = teams[0]
    away_team = teams[1]

    result = predict_matchup(home_team, away_team, diff_model, feature_cols, residual_std, team_latest)

    print(f"\n--- {result['home_team']} (home) vs {result['away_team']} (away) ---")
    print(f"Predicted point differential (home - away): {result['predicted_point_diff']}")
    print(f"Predicted winner: {result['predicted_winner']}")
    print(f"Blowout probability: {result['blowout_probability']:.1%}")
    print(f"Recommendation: {result['recommendation']}")