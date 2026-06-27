# ESG Classification Model for Institutional Wealth Management

Supervised machine learning pipeline classifying publicly listed corporations into four ESG tiers (Loser, Normal, Winner, Champion) for institutional portfolio construction.

## Overview

This project builds a classification pipeline to categorise companies into ESG tiers for use in institutional wealth management. The pipeline addresses real-world challenges including class imbalance (ESG Champions represent only 2.4% of the dataset), feature redundancy detection, outlier handling, and model governance considerations for regulated deployment.

## Methodology

### Section 1 — Exploratory Data Analysis
- Dataset overview and data type assessment (2,000 training / 1,000 test)
- Class distribution analysis revealing severe imbalance (Cat 3: 2.4%)
- Missing value audit (FEAT_GOV3: 9.2% missing)
- IQR and Z-score outlier detection (FEAT_GOV2 data entry error at 1000)
- Feature redundancy: FEAT_SOC4 = -FEAT_SOC5 (perfect negative mirror, r = -1.000)
- Correlation heatmap and feature-target relationship analysis

### Section 2 — Model Development
- **Preprocessing** — redundant feature removal, outlier correction, median imputation, label encoding, StandardScaler
- **Class Imbalance Correction** — Category 3 oversampled x7 (48 to 384 rows)
- **Logistic Regression** — multinomial with L-BFGS solver
- **Random Forest** — 100-tree ensemble
- **Hyperparameter Sensitivity** — C values and n_estimators tested
- **Evaluation** — confusion matrices, ROC/AUC curves, per-class accuracy, feature importances

## Tools & Libraries

- **Python** (pandas, NumPy, scikit-learn, matplotlib, seaborn)
- Classification modelling and hyperparameter tuning
- ROC/AUC and multi-class evaluation metrics

## Key Findings

- Random Forest achieves higher overall accuracy (80.4%) but misclassifies 79% of ESG Champions
- Logistic Regression achieves 78.6% accuracy with 79.2% Champion detection — recommended for deployment on minority-class reliability and auditability
- Social features (SOC1, SOC5, SOC3) are the strongest predictors across both models
- Random Forest shows severe overfitting (100% train vs 80.4% test)

## Module

Machine Learning, Artificial Intelligence, and Big Data in Finance — MSc Finance & Financial Technology, Henley Business School

## Usage

Open [`esg_classification_model.ipynb`](esg_classification_model.ipynb) for the full pipeline. Training and test datasets are included.
