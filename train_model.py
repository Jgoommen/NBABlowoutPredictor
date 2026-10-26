import pandas as pd
import numpy as np
import joblib
import boto3
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

BUCKET_NAME = "student-s3-josh-2026-161012475511-us-east-2-an"
REGION = "us-east-2"
MATCHUPS_S3_KEY = "raw/nbaMatchups.csv"
LOCAL_MATCHUPS_FILE = "nbaMatchups.csv"  # temp local copy pulled down from S3

BLOWOUT_THRESHOLD = 15  # point margin that counts as a blowout, matches build_matchups.py

s3 = boto3.client("s3", region_name=REGION)

#pull training data FROM S3, not local disk
print(f"Downloading s3://{BUCKET_NAME}/{MATCHUPS_S3_KEY} ...")
s3.download_file(BUCKET_NAME, MATCHUPS_S3_KEY, LOCAL_MATCHUPS_FILE)
print("Download complete.")

df = pd.read_csv(LOCAL_MATCHUPS_FILE)
df['GAME_DATE_HOME'] = pd.to_datetime(df['GAME_DATE_HOME'])
df = df.sort_values('GAME_DATE_HOME').reset_index(drop=True)

feature_cols = [c for c in df.columns if c.endswith('_HOME') or c.endswith('_AWAY')]
feature_cols = [c for c in feature_cols if c not in (
    'TEAM_NAME_HOME', 'TEAM_NAME_AWAY', 'GAME_DATE_HOME', 'GAME_DATE_AWAY',
    'IS_HOME_HOME', 'IS_HOME_AWAY', 'PTS_HOME', 'PTS_AWAY'
)]

print(f"Using {len(feature_cols)} features")


split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Train: {len(train_df)} games ({train_df['GAME_DATE_HOME'].min().date()} to {train_df['GAME_DATE_HOME'].max().date()})")
print(f"Test:  {len(test_df)} games ({test_df['GAME_DATE_HOME'].min().date()} to {test_df['GAME_DATE_HOME'].max().date()})")
#training and test data
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]


y_train_diff = train_df['POINT_DIFF']
y_test_diff = test_df['POINT_DIFF']

diff_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
diff_model.fit(X_train, y_train_diff)

diff_preds = diff_model.predict(X_test)
mae = mean_absolute_error(y_test_diff, diff_preds)
rmse = np.sqrt(mean_squared_error(y_test_diff, diff_preds))

print(f"\n--- Point Differential Model ---")
print(f"MAE:  {mae:.2f} points")
print(f"RMSE: {rmse:.2f} points")


residual_std = rmse  

def blowout_probability(predicted_diff, residual_std, threshold=BLOWOUT_THRESHOLD):
    """
    Probability that the true point differential falls outside
    [-threshold, threshold], assuming errors are roughly normally
    distributed around the model's prediction.
    """
    p_home_blowout = 1 - norm.cdf(threshold, loc=predicted_diff, scale=residual_std)
    p_away_blowout = norm.cdf(-threshold, loc=predicted_diff, scale=residual_std)
    return p_home_blowout + p_away_blowout

test_blowout_probs = blowout_probability(diff_preds, residual_std)
test_blowout_actual = test_df['BLOWOUT'].values

derived_auc = roc_auc_score(test_blowout_actual, test_blowout_probs)

print(f"\n--- Derived Blowout Probability ---")
print(f"Test set blowout rate: {test_blowout_actual.mean():.2%}")
print(f"AUC (derived from regression residuals): {derived_auc:.3f}")


importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': diff_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 features for point differential:")
print(importance.head(10).to_string(index=False))


joblib.dump(diff_model, 'model_point_diff.pkl')
joblib.dump(feature_cols, 'model_feature_cols.pkl')
joblib.dump(residual_std, 'model_residual_std.pkl')

print("\nSaved local copies: model_point_diff.pkl, model_feature_cols.pkl, model_residual_std.pkl")

model_files = {
    'model_point_diff.pkl': 'models/model_point_diff.pkl',
    'model_feature_cols.pkl': 'models/model_feature_cols.pkl',
    'model_residual_std.pkl': 'models/model_residual_std.pkl',
}

print("\nUploading trained models to S3...")
for local_path, s3_key in model_files.items():
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    print(f"  Uploaded {local_path} -> s3://{BUCKET_NAME}/{s3_key}")

print("\nDone. predict.py can now pull these straight from S3.")