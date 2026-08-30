# Customer Churn Prediction

## Project Overview

This project develops a machine learning system to predict whether a telecom customer is likely to churn.

The project includes data preprocessing, exploratory data analysis, feature engineering, classification model training, model evaluation, customer churn prediction, and model saving.

## Dataset

The project uses the Telco Customer Churn dataset.

Dataset file:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information related to demographics, services, account details, billing, and churn status.

## Objectives

- Analyze customer churn patterns
- Perform data preprocessing
- Explore customer characteristics
- Perform feature engineering
- Train classification models
- Compare model performance
- Select the best-performing model
- Predict customer churn probability
- Save the trained model and preprocessing pipeline

## Models Tested

1. Logistic Regression
2. Random Forest
3. Decision Tree

## Best Model

The best-performing model selected for this project is:

**Logistic Regression**

Performance:

- Accuracy: 0.8028
- Precision: 0.6610
- Recall: 0.5242
- F1 Score: 0.5847

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Classification Model Training
6. Model Evaluation
7. Customer Churn Prediction
8. Model Saving and Verification

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Files

- `Customer_Churn_Prediction.ipynb` – Main project notebook
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` – Dataset
- `customer_churn_model.pkl` – Trained Logistic Regression model
- `customer_churn_preprocessor.pkl` – Saved preprocessing pipeline
- `README.md` – Project documentation

## How to Run

1. Open `Customer_Churn_Prediction.ipynb` in Jupyter Notebook or JupyterLab.
2. Make sure the required Python libraries are installed.
3. Run the notebook cells from beginning to end.
4. The trained model and preprocessing pipeline can be used for customer churn prediction.

## Prediction Example

The system predicts:

- Churn status
- Churn probability
- Risk level

## Model Evaluation

The classification models are evaluated using accuracy, precision, recall, F1 score, classification reports, and confusion matrices.

## Author

Neeharika Pediredla
