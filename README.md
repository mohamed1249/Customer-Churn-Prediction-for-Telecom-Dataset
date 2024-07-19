# Telecom Customer Churn Prediction

## Overview

This repository contains code and documentation for predicting customer churn in a telecom dataset. The project involves the application of machine learning models to identify customers who are likely to churn. Four different models are used: Logistic Regression, Support Vector Classifier (SVC), Random Forest, and XGBoost.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Files](#files)
- [Usage](#usage)
- [License](#license)

## Requirements

To run the code, you'll need the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `xgboost`
- `M-ana-package`
- `eli5`

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost M-ana-package eli5
```

## Data

The dataset used for this project is in the following files:

- `Dataset_Sep2023/Data file.xlsx`: The main dataset containing customer information.
- `Dataset_Sep2023/Discunnect Number.xlsx`: Contains information about disconnected numbers.

## Preprocessing

1. **Data Loading**: Load the data from the provided Excel files.
2. **Data Cleaning**:
   - Convert date columns to datetime format.
   - Identify and handle disconnected numbers.
   - Drop duplicate entries based on mobile number and last recharge date.
3. **Feature Engineering**:
   - Create features related to days since last recharge.
   - Label customers as churned based on the gap between recharges.
   - Convert date features to numerical values for modeling.

## Modeling

The following models were applied to predict customer churn:

1. **Logistic Regression**:
   - Trained and evaluated using basic settings.
   - Hyperparameter tuning performed with GridSearchCV.
   
2. **Support Vector Classifier (SVC)**:
   - Trained and evaluated using basic settings.
   - Hyperparameter tuning performed with GridSearchCV.
   
3. **Random Forest Classifier**:
   - Trained and evaluated using basic settings.
   - Hyperparameter tuning performed with GridSearchCV.
   
4. **XGBoost**:
   - Trained and evaluated using basic settings.
   - Feature importance analysis conducted.

Each model was evaluated based on accuracy, precision, recall, F1 score, and ROC AUC score.

## Results

The results of each model are summarized in the `scores` DataFrame and include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

## Files

- `Data file.xlsx`: Contains customer data.
- `Discunnect Number.xlsx`: Contains disconnected numbers.
- `full_training_data_db.csv`: The processed dataset with churn labels.
- `modeling_data_db.csv`: The final dataset used for modeling.
- `LR_trained_model.pkl`: Serialized Logistic Regression model.
- `SVC_trained_model.pkl`: Serialized SVC model.
- `RF_trained_model.pkl`: Serialized Random Forest model.
- `XGB_trained_model.pkl`: Serialized XGBoost model.
- `correlations.csv`: Feature correlation matrix.
- `scores.csv`: Model performance scores.
