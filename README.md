# Metaheuristic-Optimized-Diabetes-Prediction-using-Machine-Learning

This project implements a robust system for **predicting diabetes** using machine learning classifiers—Random Forest, Logistic Regression, and XGBoost—augmented with **metaheuristic feature selection techniques** like **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**.

By integrating traditional models with bio-inspired optimization strategies, the system demonstrates significant improvements in predictive accuracy and generalization for healthcare applications.


## 📊 Overview

- **Problem**: Accurate early detection of diabetes from clinical and demographic data.
- **Solution**: Apply feature selection using Genetic Algorithm and PSO to enhance classification performance.
- **Dataset**: Pima Indians Diabetes Dataset (oversampled to 1,500 entries).
- **Models Used**:
  - Random Forest (RF)
  - Logistic Regression (LR)
  - XGBoost
  - Stacked Ensemble (RF + XGBoost with LR meta-model)


##  Key Features

-  **Feature Selection**:
  - Traditional: Chi-Squared Test
  - Advanced: Genetic Algorithm (GA), Particle Swarm Optimization (PSO)

-  **Machine Learning**:
  - Base classifiers: RF, LR
  - Advanced: XGBoost, Stacked Ensemble

-  **Performance Metrics**:
  - Accuracy: up to **93.4%** with GA-optimized Random Forest
  - F1-Score: up to **0.952**
  - Log Loss & Optimality Gap analysis
  - Statistical Tests: Mann–Whitney U, t-test, Cohen’s d


##  Experimental Highlights

| Method        | Selected Features                                 | Accuracy (%) |
|---------------|---------------------------------------------------|--------------|
| Chi-Squared   | Glucose, Age, Pregnancies, BMI, Insulin           | 91.3         |
| GA            | Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age | 93.4         |
| PSO           | Pregnancies, Glucose, BMI                         | 91.9         |

- GA achieved the highest predictive accuracy and diversity in feature subset selection.
- PSO converged faster but had a slightly lower final accuracy.
- Ensemble model using XGBoost + RF yielded best F1-score (0.952).


##  Requirements

- Python 3.8+
- scikit-learn
- xgboost
- numpy
- pandas
- matplotlib
- scipy

📊 Dataset Source
This project uses https://www.kaggle.com/uciml/pima-indians-diabetes-database 
