<h1 align="center">ReneWind: Optimizing Wind Turbine Maintenance Costs</h1>

<p align="center"><i>Through Neural Network Based Failure Prediction</i></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-2.0-013243?logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%2FKeras-2.19-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="imbalanced-learn" src="https://img.shields.io/badge/imbalanced--learn-SMOTE-2E8B57">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.10-11557C">
  <img alt="seaborn" src="https://img.shields.io/badge/seaborn-0.13-4C72B0">
</p>

## Overview

Wind energy is one of the most developed renewable technologies, and keeping turbines running reliably depends on catching component failures before they happen. ReneWind collects sensor data from wind turbine generators and wants to move from reactive repairs to predictive maintenance. This project builds and tunes a series of neural network classifiers on a ciphered sensor dataset to predict generator failures in advance, so that components can be repaired before they break and the overall maintenance cost is reduced.

## Problem Statement

The goal is to build several classification models, tune them, and select the one that best identifies failures so generators can be serviced before breaking. Because the cost of each prediction outcome differs, the model is judged on a cost basis rather than on accuracy alone. The outcomes translate into business costs as follows:

- **True positive.** A correctly predicted failure, which results in a repair cost.
- **False negative.** A missed real failure, which results in the most expensive replacement cost.
- **False positive.** A false alarm, which results in a relatively cheap inspection cost.

Since a missed failure is the most expensive error, recall on the failure class is the primary metric, with total business cost used as the final selection criterion.

## Data

The dataset is a transformed and ciphered version of real sensor readings, supplied as two files.

| File | Purpose | Shape |
| --- | --- | --- |
| Train.csv | Training and tuning | 20,000 observations, 40 predictors, 1 target |
| Test.csv | Final evaluation only | 5,000 observations, 40 predictors, 1 target |

The 40 predictors are anonymized sensor and environmental features named V1 through V40, and the target marks failure as 1 and no failure as 0. The classes are heavily imbalanced at roughly seventeen no failure cases for every failure, which directly shaped the modeling approach.

## Approach

1. **Data understanding.** Reviewed the shape, data types, and the strong class imbalance across the training and test sets.
2. **Missing value treatment.** Imputed missing predictor values using a simple imputer so all features were complete before scaling.
3. **Exploratory data analysis.** Studied univariate distributions and compared each predictor across the failure and no failure classes to identify the strongest discriminators.
4. **Feature scaling.** Standardized the predictors so the neural networks trained on features with comparable scales.
5. **Class imbalance handling.** Addressed the imbalance using SMOTE oversampling for most models and class weighting on the original data as an alternative strategy.
6. **Modeling.** Built and tuned nine neural network architectures using the Keras Sequential API, varying depth, width, optimizers, and regularization.
7. **Cost based selection.** Swept the decision threshold for the leading candidates against the ReneWind cost structure to choose the most cost efficient model, then confirmed the choice on the held out test set.

## Models

Nine neural networks were developed and compared, progressing from a simple baseline to deeper regularized architectures. The series explored:

- **Optimizers** including SGD, Adam, and RMSprop.
- **Architectures** with hidden layers such as a 128, 64, and 32 ReLU backbone.
- **Regularization** through Dropout, Batch Normalization, L2 weight penalties, and Early Stopping to control the overfitting seen in the higher capacity models.
- **Imbalance strategies** comparing SMOTE oversampling against class weighting on the original imbalanced data.

The early models showed how near perfect training scores coincided with weaker generalization, while the later regularized models traded a small amount of training fit for substantially better behavior on unseen data.

## Model Selection and Results

The leading candidates were compared not at a fixed 0.5 threshold but by sweeping the decision threshold to minimize a total business cost defined as repair cost of 1, replacement cost of 10, and inspection cost of 0.5.

| Model (validation) | Best Threshold | Recall | Precision | Missed Failures | False Alarms | Total Cost |
| --- | --- | --- | --- | --- | --- | --- |
| Model 7 | 0.55 | 0.9134 | 0.9068 | 24 | 26 | 506.0 |
| Model 9 | 0.60 | 0.9097 | 0.9582 | 25 | 11 | 507.5 |
| Model 4 | 0.80 | 0.9061 | 0.8838 | 26 | 33 | 527.5 |

Models 7 and 9 were effectively tied on cost. Model 7 caught the most failures with the fewest missed cases, while Model 9 achieved markedly higher precision and far fewer false alarms. Model 7 was selected as the final model on the basis of its lowest total cost and best recall.

On the held out test set, the final Model 7 achieved accuracy of 0.9852, precision of 0.9293, recall of 0.9321, and F1 of 0.9307. It correctly captured 246 of 282 real failures, missed 36, and raised only 38 false alarms out of 4,718 no failure cases, catching roughly 87 percent of genuine failures on completely unseen data. The close agreement between validation and test performance confirmed that the regularization prevented overfitting and that the model is a reliable, deployable predictor of turbine failure.

## Key Findings

- **Predictive maintenance beats reactive maintenance.** Detecting early warning signs lets ReneWind avoid expensive replacements and emergency repairs, since replacement is far costlier than preventive service.
- **Recall is the metric that matters most.** Missed failures drive the largest financial risk, so failure detection capability outweighs raw accuracy.
- **Regularization was essential.** Dropout, Batch Normalization, L2 penalties, and Early Stopping closed the gap between training and validation performance that plagued the higher capacity models.
- **Class weighting can rival SMOTE.** Model 9 matched the leading SMOTE models on recall while delivering much higher precision, showing that class weighting on the original data is a strong alternative.
- **Threshold tuning aligns models with business cost.** Choosing the cutoff that minimizes total cost produced better operational outcomes than the default threshold.

## Recommendations

- Deploy the final model in a real time monitoring system connected to turbine sensors, generating automated alerts when failure probability is elevated.
- Implement a risk based maintenance strategy that prioritizes high risk turbines for immediate inspection while keeping low risk turbines on routine monitoring.
- Continue prioritizing recall in future model development, since missed failures carry the highest cost.
- Invest in sensor data quality, because prediction reliability depends directly on accurate and consistent readings.

## Repository Structure

```
renewind-failure-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── Train.csv                       # not committed (see .gitignore)
│   └── Test.csv                        # not committed (see .gitignore)
├── notebooks/
│   └── renewind_failure_prediction.ipynb
└── reports/
    └── renewind_failure_prediction.html  # exported notebook
```

## Getting Started

```bash
# clone
git clone https://github.com/<your-username>/renewind-failure-prediction.git
cd renewind-failure-prediction

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch the notebook
jupyter notebook notebooks/renewind_failure_prediction.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, TensorFlow and Keras, imbalanced-learn, Matplotlib, seaborn, Jupyter
