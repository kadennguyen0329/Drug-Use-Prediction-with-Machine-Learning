# Drug Use Prediction with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![LIME](https://img.shields.io/badge/LIME-Model_Interpretability-brightgreen.svg)]()

Welcome to the **Drug Use Prediction** project! This repository contains an end-to-end machine learning pipeline designed to predict **Cannabis Usage Levels** based on demographic data and psychological traits. By leveraging the UCI Machine Learning Repository's Drug Consumption Dataset, this project explores the intricate relationship between personality and drug consumption.

---

## Project Overview

Predicting drug consumption patterns is a complex challenge that involves understanding human psychology, socioeconomic backgrounds, and behavioral tendencies. This project applies advanced data science techniques to classify and predict usage levels, focusing specifically on cannabis. 

### Key Objectives:
- Uncover underlying patterns between personality scores (e.g., Neuroticism, Extraversion, Impulsiveness) and drug usage.
- Build and evaluate multiple classification models to accurately predict consumption levels.
- Demystify machine learning predictions using cutting-edge interpretability tools.

---

## Key Features & Methodology

- **Comprehensive Data Preprocessing**: Filtering anomalous data (e.g., Semeron users), label encoding, and feature scaling using `StandardScaler` and `MinMaxScaler`.
- **Dimensionality Reduction & Visualization**: Utilizing **Principal Component Analysis (PCA)** to reduce feature space and visualize complex distributions of usage levels.
- **Unsupervised Learning**: Applying **K-Means Clustering** accompanied by Silhouette Score evaluation to identify natural groupings within the demographic and psychological data.
- **Robust Classification Models**:
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
- **Hyperparameter Optimization**: Extensive tuning using `GridSearchCV` to find the optimal model parameters.
- **Advanced Model Interpretability**:
  - **LIME (Local Interpretable Model-agnostic Explanations)** to explain individual predictions and build trust in the model.
  - **Permutation Importance** to determine the global impact of each psychological trait on model performance.

---

## Technology Stack

- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`sklearn`)
- **Interpretability**: LIME, Permutation Importance
- **Visualization**: Matplotlib
- **Data Source**: `ucimlrepo` (UCI Machine Learning Repository)

---

## Dataset Insights

The project utilizes dataset ID `373` from the UCI ML Repository, containing features such as:
- **Demographics**: Age, Gender, Education Level, Country, Ethnicity.
- **Psychological Metrics**: Neuroticism (N-Score), Extraversion (E-Score), Openness (O-Score), Agreeableness (A-Score), Conscientiousness (C-Score), Impulsiveness, and Sensation Seeking.

---

## Getting Started

To explore the notebook and run the models locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/kadennguyen0329/Drug-Use-Prediction-with-Machine-Learning.git
cd Drug-Use-Prediction-with-Machine-Learning
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries by running:
```bash
pip install pandas numpy matplotlib scikit-learn lime ucimlrepo
```

### 3. Run the Jupyter Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```
Open `DS3001 Final Project Code.ipynb` and run the cells sequentially to see the data processing, model training, and LIME interpretations in action!

---

## Future Scope
- Expand predictions to other substances in the dataset (e.g., Amphetamines, Cocaine, Mushrooms).
- Deploy the optimal model as a web application using Streamlit or Flask for real-time predictions.
- Integrate deep learning approaches for comparative performance analysis.

---
*Created by Kaden Nguyen. Feel free to connect or reach out with any questions regarding the methodology or findings!*
