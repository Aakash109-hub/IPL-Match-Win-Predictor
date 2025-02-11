# IPL Match Win Probability Predictor

This web application predicts win probabilities for IPL (Indian Premier League) teams in real-time using a machine learning model. The model leverages historical match data to estimate the likelihood of winning for both the batting and bowling teams based on current match dynamics

**Live App**: [IPL Win Probability Predictor on Render](https://ipl-matchwinpedicter-ak109.onrender.com)

![Screenshot 2024-11-08 124216](https://github.com/user-attachments/assets/fbcd9acf-7038-4adb-8fd4-d2998100200c)


---

## Project Overview

The IPL Win Probability Predictor is useful for cricket fans and analysts who want to track a match's likely outcome as it progresses. The model is trained using logistic regression, a popular classification technique suited for probability prediction in sports analytics.

### Features
- **Team Selection**: Choose the batting and bowling teams.
- **Match Location**: Select the city where the match is being held.
- **Match Stats Input**: Enter key details like target score, current score, overs completed, and wickets lost.
- **Prediction**: Calculates win probabilities for both teams.

---

## How to Use Locally

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the App**:
    ```bash
    streamlit run app.py
    ```

---

## Machine Learning Model

The model was developed using **logistic regression**, which is effective for binary classification tasks. Logistic regression predicts probabilities, which fits well for this application where the output is the likelihood of winning for each team.

### Model Details
- **Algorithm**: Logistic Regression
- **Training Data**: Historical IPL match data
- **Features**: Includes current score, wickets fallen, target score, current run rate (CRR), and required run rate (RRR)

---

## Deployment

This project is deployed on Render. You can access the live application here:  
https://ipl-matchwinpedicter-ak109.onrender.com

---

## Files

- **app.py**: Streamlit app code.
- **pipe1.pkl**: Pre-trained logistic regression model file.
- **requirements.txt**: Lists required Python packages.
