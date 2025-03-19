import pandas as pd
import warnings
import time
import numpy as np
import csv
import random
import requests
from torchviz import make_dot
import shap
import json
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from csv import writer
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# Load and preprocess the data
def load_and_preprocess_data():
    features = pd.read_csv('fixedtrain2.csv')
    dataset = features.drop(['t1', 't2', 'year', 'month', 'day'], axis=1)
    y = np.array(features['pointdiff'])
    features = features.drop(['t1', 't2', 'year', 'month', 'day', 'result', 'pointdiff'], axis=1)
    X = features
    return X, y

# Split and standardize data
def split_and_standardize_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# Convert data to PyTorch tensors
def convert_to_tensors(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# Create PyTorch Dataset
class CBBDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define Feedforward Neural Network (FNN)
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

# Evaluate model on test data
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in test_loader:
            predictions = model(features)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Training loop with early stopping
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        test_loss = evaluate_model(model, test_loader, criterion)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, test_losses

# Evaluate model performance
def evaluate_performance(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            predictions = model(features)
            y_pred.extend(predictions.numpy())
            y_true.extend(labels.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    baseline_pred = np.full_like(y_true, np.mean(y_true))
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Baseline MAE: {mean_absolute_error(y_true, baseline_pred)}")

def get_html_document(url):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/115.0.0.0"
    ]

    session = requests.Session()
    headers = {
        "User-Agent": random.choice(user_agents)
    }
    session.headers.update(headers)
    
    retries = 3
    delay = 2

    for attempt in range(retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                print(f"403 Error encountered. Attempt {attempt + 1} of {retries}. Retrying...")
                headers["User-Agent"] = random.choice(user_agents)
                session.headers.update(headers)
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Attempt {attempt + 1} of {retries}. Retrying...")
            time.sleep(delay)

    raise Exception("Failed to fetch the HTML document after multiple attempts.")

def score_scraper(url, model, scaler, df, teamsheet, explainer, feature_names):
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    matchups = soup.find_all('td', class_='text-left nowrap')
    results = []
    
    for game in matchups:  # Renamed 'matchup' to 'game' to avoid confusion with the function name
        a_tag = game.find('a')
        if a_tag:
            matchup_text = a_tag.text.strip()
            
            if " at " in matchup_text:
                team1, rest = matchup_text.split(" at ")
                connector = "at"
                team2 = rest.strip()
            elif " vs. " in matchup_text:
                team1, rest = matchup_text.split(" vs. ")
                connector = "vs"
                team2 = rest.strip()
            else:
                continue
            
            ateam = " ".join(team1.split()[1:]).strip()
            hteam = " ".join(team2.split()[1:]).strip()
            
            if connector == "at":
                points = matchup(hteam, ateam, model, scaler, df, teamsheet, explainer, feature_names)
            elif connector == "vs":
                points = ncourtmatchup(hteam, ateam, model, scaler, df, teamsheet, explainer, feature_names)
            else:
                continue
            
            result = [hteam, ateam, points]
            with open('winprob.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(result)
            results.append(result)
    
    return results

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

        model.eval()
        with torch.no_grad():
            prediction = model(tensor)
        predicted_value = prediction.item()

    # SHAP visualization
        
        shap_values = explainer.shap_values(tensor)
        sum = 0
        
        # Proper handling for single-output regression model
        for i in range(len(feature_names)):
            shap_values_array = np.array(shap_values[0])  # Shape: [1, num_features]
            shap_values_instance = shap_values_array[i]    # Shape: [num_features]
            #print(str(feature_names[i])+str(shap_values_instance))
            sum = sum + shap_values_instance
        
        #print(sum)
        
        return predicted_value


    except KeyError as e:
        print(f"Error: Team not found - {e}")
        return None

def ranking(df, model, scaler, teamsheet, explainer, feature_names):
    mod = pd.read_csv("Overperformance.csv")
    matchup_cache = {}
    rankings = []

    for index1, row1 in df.iterrows():
        t1 = row1['Team']
        sums = 0

        for index2, row2 in df.iterrows():
            t2 = row2['Team']
            if t1 != t2:
                key_home = (t1, t2)
                key_away = (t2, t1)

                t1_over = mod.loc[mod['Team Name'] == t1, 'Performance'].values
                t2_over = mod.loc[mod['Team Name'] == t2, 'Performance'].values
                t1_games = mod.loc[mod['Team Name'] == t1, 'Total Games'].values
                t2_games = mod.loc[mod['Team Name'] == t2, 'Total Games'].values
                
                t1_mod = t1_over/t1_games if len(t1_over) > 0 and len(t1_games) > 0 else np.array([1])
                t2_mod = t2_over/t2_games if len(t2_over) > 0 and len(t2_games) > 0 else np.array([1])
                
                t1_mod = t1_mod[0]
                t2_mod = t2_mod[0]
                
                avg_modifier = (t1_mod + t2_mod) / 2
                avg_modifier = avg_modifier*1.5
                if key_home not in matchup_cache:
                    points_home = float(matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names)) + avg_modifier
                    matchup_cache[key_home] = points_home
                else:
                    points_home = matchup_cache[key_home]

                if key_away not in matchup_cache:
                    points_away = float(matchup(t2, t1, model, scaler, df, teamsheet, explainer, feature_names)) - avg_modifier
                    matchup_cache[key_away] = points_away
                else:
                    points_away = matchup_cache[key_away]

                sums += points_home
                sums -= points_away

        rankings.append([t1, sums])

    rankings.sort(key=lambda x: x[1], reverse=True)

    with open('ranking2.csv', 'w', newline='') as f_object:
        writer = csv.writer(f_object)
        writer.writerow(["Team", "Ranking Score"])
        writer.writerows(rankings)

    print("Ranking has been written to 'ranking2.csv' successfully.")

def ncourtmatchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names):
    home = matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names)
    away = matchup(t2, t1, model, scaler, df, teamsheet, explainer, feature_names)
    return ((home - away) / 2)

def probabilitytrain():
    df = pd.read_csv('ModelOutput.csv')
    X = df[['Points']].values
    y = df['Win?'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_probability(points, model):
    points = np.array([[points]])
    prob = model.predict_proba(points)[0]
    return float(prob[1])

def load_bracket(csv_file):
    df = pd.read_csv(csv_file)
    bracket = {}
    for region in df["Region"].unique():
        region_teams = df[df["Region"] == region]["Team"].tolist()
        bracket[region] = region_teams
    return bracket

def parsebracket(bracket, model, probmodel, scaler, df, teamsheet, explainer, feature_names):
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
    round_data = {
        'R32': R32,
        'S16': S16,
        'E8': E8,
        'F4': F4,
        'NC': NC
    }
    
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
                    print(f"{temp[i]} v {temp[i+1]}")
                    score = ncourtmatchup(temp[i], temp[i+1], model, scaler, df, teamsheet, explainer, feature_names)
                    probability = predict_probability(score, probmodel)
                    game = random.uniform(0, 1)
                    
                    winner = temp[i] if game <= probability else temp[i+1]
                    next_round.append(winner)
                    print(f"Winner: {winner}")
                    
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
        print(f"{F4[0]} v {F4[1]}")
        score = ncourtmatchup(F4[0], F4[1], model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, probmodel)
        game = random.uniform(0, 1)
        finalist1 = F4[0] if game <= probability else F4[1]
        
        # Semifinal 2
        print(f"{F4[2]} v {F4[3]}")
        score = ncourtmatchup(F4[2], F4[3], model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, probmodel)
        game = random.uniform(0, 1)
        finalist2 = F4[2] if game <= probability else F4[3]
        
        # Championship game
        print(f"Championship: {finalist1} v {finalist2}")
        score = ncourtmatchup(finalist1, finalist2, model, scaler, df, teamsheet, explainer, feature_names)
        probability = predict_probability(score, probmodel)
        game = random.uniform(0, 1)
        champion = finalist1 if game <= probability else finalist2
        print(champion)
        NC.append(champion)

    append_to_json_file('brackets.json', round_data)
    return R32, S16, E8, F4, NC
def append_to_json_file(file_name, new_data):
    try:
        try:
            with open(file_name, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except FileNotFoundError:
            existing_data = []

        existing_data.append(new_data)

        with open(file_name, 'w') as f:
            json.dump(existing_data, f, indent=4)

    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}. Starting fresh.")
        with open(file_name, 'w') as f:
            json.dump([new_data], f, indent=4)

def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, scaler = split_and_standardize_data(X, y)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_to_tensors(X_train, X_test, y_train, y_test)

    train_dataset = CBBDataset(X_train_tensor, y_train_tensor)
    test_dataset = CBBDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    model = CBBNet(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    #train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
    #torch.save(model.state_dict(), 'cbb_fnn_model.pth')

    model.load_state_dict(torch.load('cbb_fnn_model.pth'))
    feature_names = X.columns.tolist()
    print(feature_names)
    background_data = X_train_tensor[:100]
    explainer = shap.DeepExplainer(model, background_data)
    probmodel = probabilitytrain()
    df = pd.read_csv('data2.csv')
    teamsheet = pd.read_csv('2024ts.csv')
    #score_scraper('https://www.teamrankings.com/ncb/schedules/?date=2025-03-18', model, scaler, df, teamsheet, explainer, feature_names)
    ranking(df, model, scaler, teamsheet, explainer, feature_names)
    csv_file = "bracket.csv"
    bracket = load_bracket(csv_file)
    #parsebracket(bracket, model, probmodel, scaler, df, teamsheet, explainer, feature_names)
    while True:
        t1 = input('Home Team: ')
        t2 = input('Away Team: ')
        home = matchup(t1, t2, model, scaler, df, teamsheet, explainer, feature_names)
        away = matchup(t2, t1, model, scaler, df, teamsheet, explainer, feature_names)
        print(f"\nFinal Prediction: {t1} vs {t2}")
        print(f"Home Advantage: {home:.2f}")
        print(f"Away Disadvantage: {away:.2f}")
        print(f"Predicted Point Difference: {(home + -away)/2:.2f}\n")
        print(predict_probability((home + -away)/2, probmodel))
if __name__ == "__main__":
    main()
