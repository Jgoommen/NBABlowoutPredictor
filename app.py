import streamlit as st
from predict import download_resources, get_team_list, predict_matchup

st.set_page_config(page_title="NBA Blowout Predictor", page_icon="🏀")

st.title("🏀 NBA Blowout Predictor")
st.write(
    "Pick two teams to see a predicted score differential, blowout probability, "
    "and whether the game looks worth attending in person."
)


# Cached so the model + data only get downloaded from S3 once per app session,
# not on every button click / rerun.
@st.cache_resource
def load_resources():
    return download_resources()


with st.spinner("Loading model and team data from S3..."):
    diff_model, feature_cols, residual_std, team_latest = load_resources()

teams = get_team_list(team_latest)

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", teams, index=teams.index("Los Angeles Lakers") if "Los Angeles Lakers" in teams else 0)

with col2:
    remaining_teams = [t for t in teams if t != team_a]
    team_b = st.selectbox("Team B", remaining_teams, index=0)

home_choice = st.radio(
    "Who's the home team?",
    options=[team_a, team_b],
    horizontal=True,
)

if st.button("Calculate Matchup", type="primary"):
    home_team = home_choice
    away_team = team_b if home_choice == team_a else team_a

    with st.spinner("Running prediction..."):
        result = predict_matchup(
            home_team, away_team, diff_model, feature_cols, residual_std, team_latest
        )

    st.subheader(f"{result['home_team']} (Home) vs {result['away_team']} (Away)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Winner", result["predicted_winner"])
    m2.metric("Predicted Point Diff", f"{result['predicted_point_diff']:+.1f}")
    m3.metric("Blowout Probability", f"{result['blowout_probability']:.1%}")

    if result["blowout_probability"] < 0.30:
        st.success(result["recommendation"])
    elif result["blowout_probability"] < 0.50:
        st.warning(result["recommendation"])
    else:
        st.error(result["recommendation"])

    with st.expander("How to read this"):
        st.write(
            "**Predicted Point Diff** is home score minus away score -- positive means "
            "the home team is favored, negative means the away team is favored.\n\n"
            "**Blowout Probability** is the model's estimated chance the final margin "
            "is 15+ points in either direction, based on the model's predicted "
            "differential and its typical prediction error."
        )