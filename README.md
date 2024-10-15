# IPL Win Probability Predictor

![Screenshot 2024-10-15 231254](https://github.com/user-attachments/assets/7196a303-52f1-49e9-877e-98c10d8afeef)

This project predicts the probability of an IPL team's victory during a match based on the current state of the game. The model uses historical IPL match data to generate real-time win probabilities for a given scenario.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Streamlit Web App](#streamlit-web-app)
7. [Project Structure](#project-structure)
8. [Future Improvements](#future-improvements)

## Project Overview

The IPL Win Probability Predictor is built using **Logistic Regression** to classify the likelihood of a team winning at any point during the match. The model takes the following input:
- Batting team
- Bowling team
- Host city
- Current score
- Overs completed
- Wickets lost
- Target runs

It outputs the win probability of the batting team and the probability of the bowling team preventing a win.

## Features
- **Real-time match prediction**: Enter current match data and receive probabilities for both teams.
- **Interactive web app**: A user-friendly interface built with **Streamlit** to enter inputs and display predictions.
- **Model pipeline**: The model is built using **Scikit-learn** with preprocessing steps for categorical and numerical features.

## Installation

To run this project, you'll need to install the required dependencies. Follow the steps below:

### 1. Clone the repository

```bash
git clone https://github.com/Aakash109-hub/IPL-Match-Win-Predictor.git
cd IPL-Match-Win-Predictor
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

Ensure that the trained model file `pipe.pkl` is in the root directory of the project. If the model file is not available, follow the [Model Training](#model-training) steps below.

## Usage

### Running the Streamlit Web App

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command to start the Streamlit web app:

```bash
streamlit run app.py
```

4. The web app will open in your default browser. Enter the required match data to get win probabilities for both teams.

## Model Training

If you need to train the model, follow these steps:

1. **Preprocess the data**: The training data should include categorical features (`batting_team`, `bowling_team`, `city`) and numerical features (e.g., `runs_left`, `balls_left`, `wickets`, `total_runs_x`, `crr`, `rrr`).

2. **Model Pipeline**: The model is built using a Scikit-learn pipeline. The categorical data is encoded using `OneHotEncoder`, and `LogisticRegression` is used for training. 

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Column Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['batting_team', 'bowling_team', 'city']),
        ('num', StandardScaler(), ['runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'])
    ], remainder='passthrough'
)

# Model pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# Save the model
import pickle
pickle.dump(pipe, open('pipe.pkl', 'wb'))
```

3. **Save the model**: The trained model is saved as `pipe.pkl` and used in the Streamlit web app.

## Streamlit Web App

The web app is built using **Streamlit**. The user can input current match details (teams, score, overs, etc.), and the app will return the predicted probabilities for both teams.

### Key Components of the App:
- `selectbox`: For team and city selection.
- `number_input`: For entering numerical values like target, score, overs, and wickets.
- `button`: A button to trigger the prediction function.

### Example Code Snippet from `app.py`

```python
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
pipe = pickle.load(open('pipe.pkl','rb'))

# User input fields
teams = [...]
cities = [...]

batting_team = st.selectbox('Select the batting team', sorted(teams))
bowling_team = st.selectbox('Select the bowling team', sorted(teams))
city = st.selectbox('Select the host city', sorted(cities))
target = st.number_input('Target')
score = st.number_input('Current Score')
overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets fallen')

# Calculate derived features
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_left = 10 - wickets
crr = score / overs
rrr = (runs_left * 6) / balls_left

# Create input DataFrame
input_df = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets': [wickets_left],
    'total_runs_x': [target],
    'crr': [crr],
    'rrr': [rrr]
})

# Predict the probability
if st.button('Predict'):
    result = pipe.predict_proba(input_df)
    win_prob = result[0][1]
    st.header(f"{batting_team} win probability: {win_prob*100:.2f}%")
```

## Project Structure

```bash
ipl-win-predictor/
│
├── app.py                  # Streamlit web app code
├── model_training.ipynb     # Jupyter notebook for model training
├── pipe.pkl                 # Trained model (pickle file)
├── requirements.txt         # Dependencies
└── README.md                # Project documentation (this file)
```

## Future Improvements

- **Model Performance**: Experiment with more advanced models (e.g., XGBoost, Random Forest) for potentially better accuracy.
- **Additional Features**: Add more features like weather conditions or team strength based on player stats.
- **Dynamic Updates**: Incorporate real-time data updates using APIs for live match prediction.
