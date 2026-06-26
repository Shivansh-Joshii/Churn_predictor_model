#Customer Churn Prediction System

A Machine Learning project that predicts whether a telecom customer is likely to churn using the Telco Customer Churn dataset. The project includes data preprocessing, exploratory data analysis (EDA), handling class imbalance with SMOTE, model comparison, and a Flask web application for real-time predictions.

---

## Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. Identifying customers who are likely to leave allows companies to take proactive measures and improve customer retention.

This project develops a machine learning model that predicts customer churn based on customer demographics, account information, and service usage.

---

## Objectives

- Analyze customer behavior using EDA.
- Clean and preprocess the dataset.
- Handle class imbalance using SMOTE.
- Train and compare multiple machine learning models.
- Select the best-performing model.
- Deploy the model using Flask.

---

## Dataset

**Dataset:** Telco Customer Churn

Source:
https://www.kaggle.com/blastchar/telco-customer-churn

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn Status

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SMOTE (imbalanced-learn)
- Flask

---

## Exploratory Data Analysis

The project includes:

- Missing value analysis
- Churn distribution
- Correlation heatmap
- Customer demographics analysis
- Monthly charges analysis
- Tenure analysis
- Service usage analysis
- Feature relationships

---

## Data Preprocessing

- Removed missing values
- Label Encoding
- Feature Scaling
- Train-Test Split
- SMOTE for class balancing

---

## Machine Learning Models

The following models were trained and compared:

- Decision Tree Classifier
- Random Forest Classifier

The best-performing model was Random Forest Clssifier.

---

## Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

---

## Flask Web Application

The trained model is deployed using Flask.

Users can:

- Enter customer information
- Predict churn probability
- View prediction instantly through a web interface

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Move into the project folder

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python app.py
```

---

## Screenshots

### Home Page

<img width="317" height="397" alt="image" src="https://github.com/user-attachments/assets/092094e7-48e6-499f-87c9-038ed1749293" />


### Prediction Result

(Add screenshot here)

### EDA Visualizations

(Add screenshots here)

---

## Future Improvements

- Hyperparameter tuning
- XGBoost implementation
- Explainable AI using SHAP
- Streamlit dashboard
- Cloud deployment
- Docker containerization

---

## Skills Demonstrated

- Data Cleaning
- Data Visualization
- Exploratory Data Analysis
- Feature Engineering
- Handling Imbalanced Data
- Machine Learning
- Model Evaluation
- Flask Deployment
- Git & GitHub

---

## Author

**Shivansh Joshi**

AI & Data Science Student

Feel free to connect or contribute to this project.
