# Loan Approval Prediction System

## 📌 Project Overview

The Loan Approval Prediction System is a Machine Learning classification project that analyzes applicant information and predicts whether a loan application is likely to be Approved or Rejected.

The project demonstrates the complete Data Science workflow, including data cleaning, exploratory data analysis, feature engineering, preprocessing, classification model training, evaluation, and prediction.

## 🎯 Objectives

- Analyze loan application data
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis
- Engineer useful financial features
- Build and compare classification models
- Evaluate model performance
- Create an interactive loan prediction application
- Present insights through a dashboard

## 📊 Dataset

The dataset contains 3,192 loan applications and includes information such as:

- Gender
- Marital Status
- Dependents
- Education
- Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Age
- Loan Status

The target variable is `Loan_Status`.

The dataset contains:

- Approved: 1,596
- Rejected: 1,596

## 🔍 Exploratory Data Analysis

The project includes analysis of:

- Loan approval distribution
- Applicant income distribution
- Loan amount distribution
- Credit history and approval
- Education and approval
- Property area and approval
- Employment status and approval
- Age distribution
- Correlation between numerical features

## 🛠️ Feature Engineering

Additional features were created:

- `Total_Income`
- `Loan_to_Income_Ratio`

## 🤖 Machine Learning Models

The following classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Best Performing Model

Logistic Regression achieved the best overall performance with approximately:

- Accuracy: 98.28%
- Precision: 98.34%
- Recall: 98.28%
- F1 Score: 98.28%

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

## 💻 Application

A Streamlit web application was developed to provide:

- Loan application overview
- Loan analytics
- Applicant input form
- Loan approval prediction
- Prediction probability
- Model performance comparison

## 📁 Project Structure

```text
Phase 2 - Task 4 - Loan Approval Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── train.csv
│   └── cleaned_loan_approval_dataset.csv
│
├── models/
│   ├── loan_approval_model.pkl
│   ├── feature_columns.pkl
│   └── model_performance.csv
│
├── notebooks/
│   └── Loan_Approval_Prediction.ipynb
│
└── assets/
