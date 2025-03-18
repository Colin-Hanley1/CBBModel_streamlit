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

# Load the trained model, scaler, and datasets
@st.cache_resource
def load_model_and_data():
    # Load model
    input_dim = 18  # Set to match your feature count
    model = CBBNet(input_dim)
    try:
        model.load_state_dict(torch.load('cbb_fnn_model.pth', map_location=torch.device('cpu')))
        model.eval()
    except:
        st.error("Failed to load model. Please make sure 'cbb_fnn_model.pth' is in the app directory.")
        return None, None, None, None, None
    
    # Load data
    try:
        df = pd.read_csv('data2.csv')
        teamsheet = pd.read_csv('2024ts.csv')
        
        # Load and fit scaler
        features = pd.read_csv('fixedtrain2.csv')
        features = features.drop(['t1', 't2', 'year', 'month', 'day', 'result', 'pointdiff'], axis=1)
        scaler = StandardScaler()
        scaler.fit(features)
        
        # Create background data for SHAP
        background_data = torch.tensor(scaler.transform(features.iloc[:100]), dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background_data)
        
        # Feature names
        feature_names = features.columns.tolist()
        
        return model, scaler, df, teamsheet, explainer, feature_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Function to train probability model
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

# Matchup prediction function
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
        
        return prediction.item()

    except KeyError as e:
        st.error(f"Error: Team not found - {e}")
        return None

def neutral_court_matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names):
    home = matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names)
    away = matchup(t2, t1, model, scaler, df, teamsheet, explainer, feature_names)
    return ((home - away) / 2)

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
    # Get all regions
    regions = {
        'South': bracket.get('South', []),
        'East': bracket.get('East', []),
        'Midwest': bracket.get('Midwest', []),
        'West': bracket.get('West', [])
    }
    
    # Initialize rounds
    R32 = []  # Round 32
    S16 = []  # Sweet 16
    E8 = []   # Elite 8
    F4 = []   # Final 4
    NC = []   # National Championship
    
    # Track all games and results
    all_games = []
    
    # Process each region separately for the first rounds
    for region_name, teams in regions.items():
        region_winners = []
        temp = teams.copy()
        round_number = 1  # Track the round within each region
        
        # Process until we get a single winner from the region
        while len(temp) > 1:
            next_round = []
            for i in range(0, len(temp), 2):
                if i + 1 < len(temp):  # Make sure we have a pair
                    team1 = temp[i]
                    team2 = temp[i+1]
                    
                    score = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
                    probability = predict_probability(score, prob_model)
                    
                    if use_random:
                        game = random.uniform(0, 1)
                        winner = team1 if game <= probability else team2
                    else:
                        winner = team1 if probability >= 0.5 else team2
                    
                    next_round.append(winner)
                    
                    # Record game details
                    game_info = {
                        "region": region_name,
                        "round": round_number,
                        "team1": team1,
                        "team2": team2,
                        "predicted_point_diff": score,
                        "win_probability": probability,
                        "winner": winner
                    }
                    all_games.append(game_info)
                    
                    # Track teams at appropriate rounds based on round number
                    if round_number == 1:
                        R32.append(winner)
                    elif round_number == 2:
                        S16.append(winner)
                    elif round_number == 3:
                        E8.append(winner)
                    
            temp = next_round
            round_number += 1
            
        # Add the region winner to Final Four
        if temp:
            F4.append(temp[0])
    
    # Process Final Four
    if len(F4) >= 4:
        # Semifinal 1
        team1, team2 = F4[0], F4[1]
        score = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, prob_model)
        
        if use_random:
            game = random.uniform(0, 1)
            finalist1 = team1 if game <= probability else team2
        else:
            finalist1 = team1 if probability >= 0.5 else team2
        
        game_info = {
            "region": "Final Four",
            "round": 4,
            "team1": team1,
            "team2": team2,
            "predicted_point_diff": score,
            "win_probability": probability,
            "winner": finalist1
        }
        all_games.append(game_info)
        
        # Semifinal 2
        team1, team2 = F4[2], F4[3]
        score = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, prob_model)
        
        if use_random:
            game = random.uniform(0, 1)
            finalist2 = team1 if game <= probability else team2
        else:
            finalist2 = team1 if probability >= 0.5 else team2
            
        game_info = {
            "region": "Final Four",
            "round": 4,
            "team1": team1,
            "team2": team2,
            "predicted_point_diff": score,
            "win_probability": probability,
            "winner": finalist2
        }
        all_games.append(game_info)
        
        # Championship game
        team1, team2 = finalist1, finalist2
        score = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, prob_model)
        
        if use_random:
            game = random.uniform(0, 1)
            champion = team1 if game <= probability else team2
        else:
            champion = team1 if probability >= 0.5 else team2
            
        game_info = {
            "region": "Championship",
            "round": 5,
            "team1": team1,
            "team2": team2,
            "predicted_point_diff": score,
            "win_probability": probability,
            "winner": champion
        }
        all_games.append(game_info)
        
        NC.append(champion)

    round_data = {
        'R32': R32,
        'S16': S16,
        'E8': E8,
        'F4': F4,
        'NC': NC,
        'all_games': all_games
    }
    
    return round_data

# Streamlit app
def main():
    st.set_page_config(page_title="College Basketball Prediction Model", layout="wide")
    
    st.title("College Basketball Prediction Model")
    st.write("This app uses a neural network model to predict college basketball games and tournament outcomes.")
    
    # Load model and data
    model, scaler, df, teamsheet, explainer, feature_names = load_model_and_data()
    prob_model = train_probability_model()
    
    if model is None or prob_model is None:
        st.error("Failed to load required models or data. Please check that all files are available.")
        return
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Team vs Team Prediction", "Rankings", "Tournament Simulator"])
    
    if page == "Team vs Team Prediction":
        st.header("Team vs Team Prediction")
        
        # Get list of teams
        team_list = sorted(df['Team'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", team_list)
        with col2:
            team2 = st.selectbox("Team 2", team_list, index=1)
        
        location = st.radio("Game Location", ["Home/Away", "Neutral Court"])
        
        if st.button("Predict Outcome"):
            if location == "Home/Away":
                home_score = matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
                away_score = matchup(team2, team1, model, scaler, df, teamsheet, explainer, feature_names)
                
                if home_score is not None and away_score is not None:
                    net_score = (home_score + -away_score)/2
                    win_probability = predict_probability(net_score, prob_model)
                    
                    st.subheader("Prediction Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Home Advantage", f"{home_score:.2f}",
                                  delta=team1 if home_score > 0 else team2)
                    with col2:
                        st.metric("Away Advantage", f"{-away_score:.2f}",
                                  delta=team1 if -away_score > 0 else team2)

                    
            else:
                # Neutral court prediction
                net_score = neutral_court_matchup(team1, team2, model, scaler, df, teamsheet, explainer, feature_names)
                if net_score is not None:
                    win_probability = predict_probability(net_score, prob_model)
                    
                    st.subheader("Prediction Results (Neutral Court)")
                    st.metric("Predicted Point Difference", f"{net_score:.2f}", 
                              delta=team1 if net_score > 0 else team2)
                    
                    # Display win probability
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{team1} Win Probability", f"{win_probability:.1%}")
                    with col2:
                        st.metric(f"{team2} Win Probability", f"{1-win_probability:.1%}")
                    
                    # Show the likely winner
                    if win_probability > 0.5:
                        st.success(f"Predicted Winner: {team1} with {win_probability:.1%} probability")
                    else:
                        st.success(f"Predicted Winner: {team2} with {(1-win_probability):.1%} probability")
    
    elif page == "Tournament Simulator":
        st.header("Tournament Simulator")
        
        # Option to upload bracket or use a demo
        
        
        bracket_df = pd.read_csv('bracket.csv')
        bracket = {}
        for region in bracket_df["Region"].unique():
            region_teams = bracket_df[bracket_df["Region"] == region]["Team"].tolist()
            bracket[region] = region_teams
                    
        st.success("Bracket loaded successfully!")
        st.write("Teams by region:")
        st.write(bracket)

        if bracket is not None:
            simulation_type = st.radio("Simulation Type", ["Random (Monte Carlo)", "Deterministic"])
            
            if st.button("Run Tournament Simulation"):
                use_random = (simulation_type == "Random (Monte Carlo)")
                
                with st.spinner("Simulating tournament..."):
                    results = simulate_bracket(bracket, model, prob_model, scaler, df, teamsheet, explainer, feature_names, use_random)
                
                # Display results
                st.subheader("Tournament Results")
                
                # Create tabs for different rounds
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Final Results", "All Games", "Elite Eight", "Sweet 16", "Round of 32", "Raw Data"])
                
                with tab1:
                    st.header("Champion")
                    if results['NC']:
                        st.subheader(f"🏆 {results['NC'][0]} 🏆")
                    
                    st.header("Final Four")
                    col1, col2, col3, col4 = st.columns(4)
                    for i, team in enumerate(results['F4']):
                        with [col1, col2, col3, col4][i]:
                            st.write(team)
                
                with tab2:
                    # Filter and display all games
                    games_df = pd.DataFrame(results['all_games'])
                    st.dataframe(
                        games_df.sort_values(by=['round', 'region'])
                        .assign(predicted_point_diff=lambda x: x['predicted_point_diff'].round(2))
                        .assign(win_probability=lambda x: (x['win_probability'] * 100).round(1).astype(str) + '%')
                    )
                
                with tab3:
                    st.header("Elite Eight")
                    for team in results['E8']:
                        st.write(team)
                
                with tab4:
                    st.header("Sweet 16")
                    for team in results['S16']:
                        st.write(team)
                
                with tab5:
                    st.header("Round of 32")
                    for team in results['R32']:
                        st.write(team)
                
                with tab6:
                    st.json(results)
                
                # Option to save results
                
    elif page == "Rankings":
        st.header("Rankings")
        rankingShow = st.radio("Show", ["Momentum", "Performance Rankings"])
        if rankingShow == 'Performance Rankings':
            ranking_df = pd.read_csv('ranking2.csv')
            st.write(ranking_df)
        if rankingShow == 'Momentum':
            ranking_df = pd.read_csv('Momentum.csv')
            st.write(ranking_df)
        
        
                
                # Save rankings


if __name__ == "__main__":
    main()
