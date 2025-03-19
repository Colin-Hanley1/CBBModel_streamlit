import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
import json

# Define the Feedforward Neural Network model
class CBBNet(nn.Module):
    def __init__(self, input_dim):
        super(CBBNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model and data
@st.cache_resource
def load_model_and_data():
    input_dim = 18
    model = CBBNet(input_dim)
    try:
        model.load_state_dict(torch.load('cbb_fnn_model.pth', map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, None, None
    
    try:
        df = pd.read_csv('data2.csv')
        teamsheet = pd.read_csv('2024ts.csv')
        features = pd.read_csv('fixedtrain2.csv')
        features = features.drop(['t1', 't2', 'year', 'month', 'day', 'result', 'pointdiff'], axis=1)
        scaler = StandardScaler()
        scaler.fit(features)
        background_data = torch.tensor(scaler.transform(features.iloc[:100]), dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background_data)
        feature_names = features.columns.tolist()
        return model, scaler, df, teamsheet, explainer, feature_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_resource
def train_probability_model():
    try:
        df = pd.read_csv('ModelOutput.csv')
        X = df[['Points']].values
        y = df['Win?'].values
        model = LogisticRegression()
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error training probability model: {e}")
        return None

# Matchup function with stat differences
def matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names):
    try:
        t1_stats = df.loc[df['Team'] == t1].squeeze()
        t2_stats = df.loc[df['Team'] == t2].squeeze()
        class1 = float(teamsheet.loc[teamsheet['Team'] == t1, 'Recruiting 3-Year Average'])
        class2 = float(teamsheet.loc[teamsheet['Team'] == t2, 'Recruiting 3-Year Average'])

        stat_differences = [
            (float(t1_stats['PPG']) + float(t2_stats['PPGA'])) / 2 - (float(t2_stats['PPG']) + float(t1_stats['PPGA'])) / 2,
            (float(t1_stats['ScoringMargin'])) - (float(t2_stats['ScoringMargin'])),
            (float(t1_stats['OffensiveEff']) + float(t2_stats['DefensiveEff'])) / 2 - (float(t2_stats['OffensiveEff']) + float(t1_stats['DefensiveEff'])) / 2,
            (float(t1_stats['FloorP']) + float(t2_stats['FloorPA'])) / 2 - (float(t2_stats['FloorP']) + float(t1_stats['FloorPA'])) / 2,
            (float(t1_stats['TWOPP']) + float(t2_stats['TWOPPA'])) / 2 - (float(t2_stats['TWOPP']) + float(t1_stats['TWOPPA'])) / 2,
            (float(t1_stats['THREEPP']) + float(t2_stats['THREEPPA'])) / 2 - (float(t2_stats['THREEPP']) + float(t1_stats['THREEPPA'])) / 2,
            (float(t1_stats['FTR']) + float(t2_stats['FTRA'])) / 2 - (float(t2_stats['FTR']) + float(t1_stats['FTRA'])) / 2,
            (float(t1_stats['ORB']) + float(t2_stats['ORBA'])) / 2 - (float(t2_stats['ORB']) + float(t1_stats['ORBA'])) / 2,
            (float(t1_stats['DRB']) + float(t2_stats['DRBA'])) / 2 - (float(t2_stats['DRB']) + float(t1_stats['DRBA'])) / 2,
            (float(t1_stats['BlockP']) + float(t2_stats['BlockPA'])) / 2 - (float(t2_stats['BlockP']) + float(t1_stats['BlockPA'])) / 2,
            (float(t1_stats['StealP']) + float(t2_stats['StealPA'])) / 2 - (float(t2_stats['StealP']) + float(t1_stats['StealPA'])) / 2,
            (float(t1_stats['AssistR']) + float(t2_stats['AssistRA'])) / 2 - (float(t2_stats['AssistR']) + float(t1_stats['AssistRA'])) / 2,
            (float(t1_stats['FoulR']) + float(t2_stats['FoulRA'])) / 2 - (float(t2_stats['FoulR']) + float(t1_stats['FoulRA'])) / 2,
            (float(t1_stats['ESCPG'])) - (float(t2_stats['ESCPG'])),
            (float(t1_stats['EPR']) + float(t2_stats['EPRA'])) / 2 - (float(t2_stats['EPR']) + float(t1_stats['EPRA'])) / 2,
            (float(t1_stats['TOVR']) + float(t2_stats['TOVRA'])) / 2 - (float(t2_stats['TOVR']) + float(t1_stats['TOVRA'])) / 2,
            (float(t1_stats['WinP'])) - (float(t2_stats['WinP'])),
            class1 - class2
        ]
        
        stat_diffs_scaled = scaler.transform([stat_differences])
        tensor = torch.tensor(stat_diffs_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(tensor)
        
        stat_labels = [
            "PPG vs PPGA", "Scoring Margin", "Offensive vs Defensive Eff", "Floor %", 
            "2-Point %", "3-Point %", "Free Throw Rate", "Offensive Rebound %", 
            "Defensive Rebound %", "Block %", "Steal %", "Assist Ratio", 
            "Foul Rate", "ESCPG", "EPR", "Turnover Rate", "Win %", "Recruiting Avg"
        ]
        return prediction.item(), list(zip(stat_labels, stat_differences))

    except KeyError as e:
        st.error(f"Error: Team not found - {e}")
        return None, None

def neutral_court_matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names):
    home_pred, home_stats = matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names)
    away_pred, away_stats = matchup(t2, t1, model, scaler, df, teamsheet, explainer, feature_names)
    return ((home_pred - away_pred) / 2), home_stats

def predict_probability(points, prob_model):
    points = np.array([[points]])
    prob = prob_model.predict_proba(points)[0]
    return float(prob[1])

def load_bracket(csv_file):
    df = pd.read_csv(csv_file)
    bracket = {}
    for region in df["Region"].unique():
        region_teams = df[df["Region"] == region]["Team"].tolist()
        bracket[region] = region_teams
    return bracket

def simulate_bracket(bracket, model, prob_model, scaler, df, teamsheet, explainer, feature_names, use_random=True):
    regions = {
        'South': bracket.get('South', []),
        'East': bracket.get('East', []),
        'Midwest': bracket.get('Midwest', []),
        'West': bracket.get('West', [])
    }
    
    R32, S16, E8, F4, NC = [], [], [], [], []
    all_games = []
    
    for region_name, teams in regions.items():
        region_winners = []
        temp = teams.copy()
        round_number = 1
        
        while len(temp) > 1:
            next_round = []
            for i in range(0, len(temp), 2):
                if i + 1 < len(temp):
                    team1 = temp[i]
                    team2 = temp[i+1]
                    
                    score, _ = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
                    probability = predict_probability(score, prob_model)
                    
                    if use_random:
                        game = random.uniform(0, 1)
                        winner = team1 if game <= probability else team2
                    else:
                        winner = team1 if probability >= 0.5 else team2
                    
                    next_round.append(winner)
                    all_games.append({
                        "region": region_name, "round": round_number, "team1": team1, "team2": team2,
                        "predicted_point_diff": score, "win_probability": probability, "winner": winner
                    })
                    
                    if round_number == 1: R32.append(winner)
                    elif round_number == 2: S16.append(winner)
                    elif round_number == 3: E8.append(winner)
                    
            temp = next_round
            round_number += 1
            
        if temp: F4.append(temp[0])
    
    if len(F4) >= 4:
        for teams, region in [((F4[0], F4[1]), "Final Four"), ((F4[2], F4[3]), "Final Four")]:
            score, _ = neutral_court_matchup(teams[0], teams[1], model, scaler, df, teamsheet, explainer, feature_names)
            probability = predict_probability(score, prob_model)
            winner = teams[0] if (use_random and random.uniform(0, 1) <= probability) or (not use_random and probability >= 0.5) else teams[1]
            all_games.append({"region": region, "round": 4, "team1": teams[0], "team2": teams[1], "predicted_point_diff": score, "win_probability": probability, "winner": winner})
            if region == "Final Four": F4 = [winner] + F4[2:] if teams[0] == F4[0] else F4[:2] + [winner]
        
        score, _ = neutral_court_matchup(F4[0], F4[2], model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, prob_model)
        champion = F4[0] if (use_random and random.uniform(0, 1) <= probability) or (not use_random and probability >= 0.5) else F4[2]
        all_games.append({"region": "Championship", "round": 5, "team1": F4[0], "team2": F4[2], "predicted_point_diff": score, "win_probability": probability, "winner": champion})
        NC.append(champion)

    return {'R32': R32, 'S16': S16, 'E8': E8, 'F4': F4, 'NC': NC, 'all_games': all_games}

# Custom styling for stat differences (no percentages)
# Custom styling for stat differences with turnover rate color inverted
def style_stat_differences(df):
    def color_val(row):
        val = row['Difference']
        stat = row['Statistic']
        
        # Invert color logic for Turnover Rate
        if stat == "Turnover Rate":
            color = 'green' if val < 0 else 'red' if val > 0 else 'black'
        else:
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        
        return f'color: {color}'
    
    return df.style.format({"Difference": "{:.2f}"}).apply(
        lambda row: [color_val(row) if col == 'Difference' else '' for col in df.columns], 
        axis=1
    )
def main():
    st.set_page_config(page_title="College Basketball Prediction Model", layout="wide")
    
    st.title("College Basketball Prediction Model")
    st.markdown("Model last updated on 3/19/25 at 10:17 CDT")
    
    model, scaler, df, teamsheet, explainer, feature_names = load_model_and_data()
    prob_model = train_probability_model()
    
    if model is None or prob_model is None:
        st.error("Failed to load required models or data. Please ensure all files are present.")
        return
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Team vs Team Prediction", "Rankings", "Tournament Simulator"])
    
    if page == "Team vs Team Prediction":
        st.header("Team vs Team Prediction")
        st.markdown("Compare two teams and see the predicted outcome based on their stats.")
        
        team_list = sorted(df['Team'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", team_list, help="Select the first team")
        with col2:
            team2 = st.selectbox("Team 2", team_list, index=1, help="Select the opposing team")
        
        location = st.radio("Game Location", ["Home/Away", "Neutral Court"], help="Choose where the game is played")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            predict_button = st.button("Predict Outcome")
        
        
        if predict_button:
            with st.spinner("Calculating prediction..."):
                if location == "Home/Away":
                    home_score, home_stats = matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
                    away_score, away_stats = matchup(team2, team1, model, scaler, df, teamsheet, explainer, feature_names)
                    
                    if home_score is not None and away_score is not None:
                        net_score = (home_score + -away_score) / 2
                        win_probability = predict_probability(net_score, prob_model)
                        
                        st.subheader("Prediction Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Home Advantage", f"{home_score:.2f}", delta=team1 if home_score > 0 else team2)
                        with col2:
                            st.metric("Away Advantage", f"{-away_score:.2f}", delta=team1 if -away_score > 0 else team2)

                        with st.expander("🔍 Stat Differences", expanded=False):
                            st.markdown(f"**{team1} (Home) vs {team2} (Away)**")
                            home_df = pd.DataFrame(home_stats, columns=["Statistic", "Difference"])
                            st.dataframe(style_stat_differences(home_df))

                            st.markdown(f"**{team2} (Home) vs {team1} (Away)**")
                            away_df = pd.DataFrame(away_stats, columns=["Statistic", "Difference"])
                            st.dataframe(style_stat_differences(away_df))

                else:
                    net_score, stats_diff = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
                    if net_score is not None:
                        win_probability = predict_probability(net_score, prob_model)
                        
                        st.subheader("Prediction Results (Neutral Court)")
                        st.metric("Predicted Point Difference", f"{net_score:.2f}", delta=team1 if net_score > 0 else team2)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{team1} Win Probability", f"{win_probability:.1%}")
                        with col2:
                            st.metric(f"{team2} Win Probability", f"{1-win_probability:.1%}")
                        
                        st.success(f"Predicted Winner: {team1 if win_probability > 0.5 else team2} with {(win_probability if win_probability > 0.5 else 1-win_probability):.1%} probability")
                        
                        with st.expander("🔍 Stat Differences", expanded=False):
                            st.markdown(f"**{team1} vs {team2} (Neutral Court)**")
                            stats_df = pd.DataFrame(stats_diff, columns=["Statistic", "Difference"])
                            st.dataframe(style_stat_differences(stats_df))

    

    elif page == "Tournament Simulator":
        st.header("Tournament Simulator")
        st.markdown("Simulate an NCAA tournament bracket with your loaded teams.")
        
        bracket_df = pd.read_csv('bracket.csv')
        bracket = load_bracket('bracket.csv')
        st.success("Bracket loaded successfully!")
        st.write("Teams by region:", bracket)
        
        simulation_type = st.radio("Simulation Type", ["Random (Monte Carlo)", "Deterministic"], help="Random uses Monte Carlo simulation; Deterministic uses highest probability.")
        
        if st.button("Run Tournament Simulation"):
            use_random = (simulation_type == "Random (Monte Carlo)")
            with st.spinner("Simulating tournament..."):
                results = simulate_bracket(bracket, model, prob_model, scaler, df, teamsheet, explainer, feature_names, use_random)
            
            st.subheader("Tournament Results")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Final Results", "All Games", "Elite Eight", "Sweet 16", "Round of 32"])
            
            with tab1:
                st.header("Champion")
                if results['NC']: st.subheader(f"🏆 {results['NC'][0]} 🏆")
                st.header("Final Four")
                col1, col2, col3, col4 = st.columns(4)
                for i, team in enumerate(results['F4']):
                    with [col1, col2, col3, col4][i]: st.write(team)
            
            with tab2:
                games_df = pd.DataFrame(results['all_games'])
                st.dataframe(games_df.sort_values(by=['round', 'region']).assign(
                    predicted_point_diff=lambda x: x['predicted_point_diff'].round(2),
                    win_probability=lambda x: (x['win_probability'] * 100).round(1).astype(str) + '%'
                ))
            
            with tab3: st.header("Elite Eight"); st.write("\n".join(results['E8']))
            with tab4: st.header("Sweet 16"); st.write("\n".join(results['S16']))
            with tab5: st.header("Round of 32"); st.write("\n".join(results['R32']))

    elif page == "Rankings":
        st.header("Rankings")
        ranking_show = st.radio("Show", ["Momentum", "Performance Rankings"], help="Choose which rankings to display")
        if ranking_show == 'Performance Rankings':
            ranking_df = pd.read_csv('ranking2.csv')
            st.dataframe(ranking_df)
        elif ranking_show == 'Momentum':
            ranking_df = pd.read_csv('Momentum.csv')
            st.dataframe(ranking_df)

if __name__ == "__main__":
    main()
