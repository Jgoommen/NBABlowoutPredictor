# NBABlowoutPredictor
## Overview

The goal of the project is to create an app that consumers can use to gauge whether they want to attend a given NBA game. This comes from the idea that blowouts are generally not fun to attend, and, if given the choice, people would much rather attend a close game than a very lopsided one. The project has a secondary use, which is the ability to be used by NBA front offices to assess the difficulty of future games. They could use informtion from this project to better allocate film and prepare better.

We used data from NBA_stats API stored in an S3 bucket to train a model, which is then stored in the same S3 bucket. The predict.py file calls this model to pull data and make predictions.

You can run the uploaded streamlit app here https://nbablowoutpredictor-htnarsuqsjczenfumtx6uh.streamlit.app/ 
For any questions regarding this project, contact us at joommen@charlotte.edu

## How It Works

The project is a pipeline that turns raw NBA game logs into a trained model, plus a Streamlit app that uses the model to predict how lopsided a matchup will be.

### 1. Pull raw game data (`pull_games.py`)
Uses the `nba_api` package to pull every regular-season team game for the 2023-24, 2024-25 and 2025-26 seasons. Each row is one team's performance in one game. The result is saved as `nbaGamesDataRaw.csv`.

### 2. Build team-level features (`build_features.py`)
Turns the raw logs into features a model can learn from:
- **Home/away flag** from the matchup text (an "@" means away).
- **Rolling last-10-game averages** for points, FG%, 3P%, FT%, rebounds, assists, turnovers and plus/minus. The current game is excluded (`shift(1)`), so a game is never described using its own result.
- **Points allowed** (last-10 average), found by matching each game to the opponent's score.
- **Rest features:** days of rest and a back-to-back flag.
- **Roster strength:** per-game player stats for each season, aggregated by team (average points, top scorer, total points, average FG%, average minutes).

The result is saved as `nbaTeamFeatures.csv`.

### 3. Build matchup rows (`build_matchups.py`)
Merges the home team's and away team's rows for each game into a single row, with columns suffixed `_HOME` and `_AWAY`. It also creates the targets:
- `POINT_DIFF`: home score minus away score
- `BLOWOUT`: 1 if the margin is 15 points or more in either direction
- `HOME_WIN`: 1 if the home team won

Early-season games without enough rolling history are dropped. The result is saved as `nbaMatchups.csv`.

### 4. Upload data to S3 (`upload_to_s3.py`)
Uploads `nbaMatchups.csv` and `nbaTeamFeatures.csv` to the `raw/` folder of the S3 bucket.

### 5. Train the model (`train_model.py`)
1. Downloads the matchup data from S3.
2. Splits the games **by date**, using the first 80% for training and the last 20% for testing, so the model is tested on games it hasn't seen from the future.
3. Trains a Random Forest regressor (300 trees) to predict `POINT_DIFF`.
4. Reports MAE and RMSE on the test set.
5. Saves the RMSE as the model's typical prediction error (`residual_std`).
6. Derives a blowout probability and checks it with AUC (see below).
7. Saves the model, its feature column order and the residual std as `.pkl` files, and uploads them to the `models/` folder in S3.

### 6. Predict a matchup (`predict.py`)
1. Downloads the model files and `nbaTeamFeatures.csv` from S3.
2. Takes each team's most recent row as its current form.
3. Builds one matchup row (home stats and away stats, in the exact column order used in training).
4. Predicts the point differential with the model.
5. Converts that into a **blowout probability**. It assumes the prediction error follows a normal distribution centered on the predicted differential and calculates the chance the final margin lands beyond +15 or −15:

   `P(blowout) = P(diff > 15) + P(diff < -15)`
6. Returns the predicted winner, differential, blowout probability and a recommendation (competitive, toss-up, or likely blowout).

### 7. The web app (`app.py`)
A Streamlit interface where the user picks two teams and chooses which one is at home. When they click **Calculate Matchup**, the app calls `predict.py` and shows the results. Model and data loading is cached so S3 is only called once per session.

### Data flow

```
nba_api → pull_games.py → build_features.py → build_matchups.py
        → upload_to_s3.py → S3 → train_model.py → S3 (models)
        → predict.py → app.py (Streamlit)
```

### Limitations
- Roster strength uses full-season player averages, so it includes games that happen after the one being predicted.
- A team's "current form" comes from its most recent game in the data, so the app needs refreshed data to stay current.
- The blowout probability comes from a normal-error assumption, not a separately trained classifier.
  
## 📊 Data Source 
The data comes from NBA.com stats via the NBA_api stats wrapper. The API is free and simple to use. The data includes games pulled as far back as specified, and player data from specific seasons as specified. 

## 🛠️ Tools and Technologies Used
- **Languages:** *Python*
- **Libraries:** *streamlit, pandas, , SciPy, joblib, boto3, numpy, sklearn*
- **IDE:** *VS Code*
- **Simple Storage Service**

## 🚀 How to Run the Project
1. Clone the repository or copy the code into a Jupyter Notebook or Python file (Only predict.py and app.py are necessary to run the project)
2. Install the required libraries;
   ```bash
   pip install streamlit, pandas, boto3, joblib, numpym etc.
3. Run the streamlit app
   ```bash
   streamlit run app.py
You can also run the uploaded streamlit app from this link https://nbablowoutpredictor-htnarsuqsjczenfumtx6uh.streamlit.app/ 

If you are wanting to train your own data, make sure to change the seasons in pullgames.py to reflect the years you want the data to come from.
   
