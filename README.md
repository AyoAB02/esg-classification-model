# ESG Classification Model for Institutional Wealth Management

Supervised machine learning pipeline classifying companies into four ESG tiers for institutional portfolio construction, comparing Logistic Regression and Random Forest with class imbalance correction.

## Overview

This project builds a classification pipeline to categorise companies into ESG tiers (e.g., Leader, Average, Laggard, Poor) for use in institutional wealth management. The pipeline addresses real-world challenges including class imbalance across tiers and feature selection from ESG sub-scores and financial metrics.

## Methodology

- **Data Preprocessing** — feature engineering from ESG sub-scores and financial indicators
- **Class Imbalance Correction** — SMOTE / class weighting to handle uneven tier distribution
- **Logistic Regression** — interpretable baseline classifier with multinomial extension
- **Random Forest** — ensemble classifier for comparison
- **Model Evaluation** — ROC/AUC curves, confusion matrices, precision-recall analysis
- **Feature Importance** — identifying the most predictive ESG and financial factors

## Tools & Libraries

- **Python** (scikit-learn, pandas, NumPy, matplotlib, seaborn)
- Classification modelling and hyperparameter tuning
- ROC/AUC and multi-class evaluation metrics
- SMOTE for class imbalance

## Key Outputs

- ROC/AUC curves for both classifiers across all ESG tiers
- Confusion matrices and classification reports
- Feature importance rankings
- Model comparison and recommendation for deployment

## Module

Machine Learning, Artificial Intelligence, and Big Data in Finance — MSc Finance & Financial Technology, Henley Business School

## Usage

*Jupyter notebook with full pipeline and visualisations coming soon.*
