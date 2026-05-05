# 💳 Credit Card Fraud Detection System

A machine learning-based web application that predicts the probability of fraudulent transactions using a trained classification model and an interactive Streamlit UI.

---

## 🚀 Project Overview

This project detects fraudulent credit card transactions by analyzing transaction details such as amount, time, merchant category, and user behavior.

It combines:

* 📊 Data preprocessing & feature engineering
* 🤖 Machine learning model training
* 🌐 Interactive UI using Streamlit

---

## 🧠 Features

* Real-time fraud prediction
* Probability-based risk scoring
* Custom threshold tuning
* Clean and modern UI
* Actionable recommendations for detected fraud

---

## 🖥️ Streamlit UI

The app allows users to input transaction details and get instant predictions.

Example:

* Transaction amount
* Time & date
* Merchant category
* Entry mode (chip/swipe/online)
* Account details

👉 The UI is implemented in:
`app.py` 

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Streamlit
* Joblib (model persistence)

---

## 📁 Project Structure

```
creditcard_project/
│
├── app.py                  # Streamlit UI
├── training.ipynb          # Model training notebook
├── model/
│   ├── fraud_model.joblib
│   ├── scaler.joblib
│   ├── encoders.joblib
│   ├── feature_means.joblib
│   ├── feature_columns.joblib
│   └── model_meta.joblib
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/creditcardfrauddetection.git
cd creditcardfrauddetection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Details

* Model: Random Forest Classifier
* Handles imbalanced data using:

  * Feature engineering
  * Threshold tuning
* Outputs:

  * Fraud probability
  * Binary classification (Fraud / Legitimate)

---

## ⚠️ Important Notes

* Ensure model files are present in the `model/` directory before running the app
* If missing, train the model using `training.ipynb`

---

## 📈 Future Improvements

* Add real-time API integration
* Deploy on cloud (Streamlit Cloud / AWS)
* Improve model performance on imbalanced data
* Add user authentication & logging

---

## 👨‍💻 Author

Tirth Patel
B.Tech CSE (AI-ML)

---

## ⭐ If you like this project

Give it a star on GitHub!
