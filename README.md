<div align="center">

# ❤️ Heart Disease Prediction using Machine Learning

### End-to-End Machine Learning Pipeline for Heart Disease Prediction with an Interactive Streamlit Web Application

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

### 📌 Predict the likelihood of Heart Disease using Machine Learning

This project implements a complete Machine Learning pipeline—from **data preprocessing and exploratory data analysis (EDA)** to **model training, evaluation, feature importance analysis, and deployment using Streamlit**.

The application allows users to enter patient information through an intuitive web interface and instantly receive a heart disease prediction together with the model's confidence score.

<br>

## 🚀 Live Demo

**Coming Soon**

</div>

---

# 📖 Table of Contents

- Project Overview
- Why This Project?
- Key Features
- Machine Learning Workflow
- Project Architecture
- Dataset
- Exploratory Data Analysis
- Data Preprocessing
- Machine Learning Models
- Model Evaluation
- Cross Validation
- Feature Importance
- Streamlit Web Application
- Installation
- Usage
- Project Structure
- Technologies Used
- Future Improvements
- License
- Author

---

# 📖 Project Overview

Heart disease remains one of the leading causes of mortality worldwide. Early prediction can significantly improve treatment outcomes and assist healthcare professionals in making informed clinical decisions.

This project demonstrates how Machine Learning techniques can be applied to structured clinical data to predict whether a patient is likely to have heart disease.

The repository follows a modular software engineering approach where every stage of the Machine Learning pipeline is implemented as an independent module, making the codebase clean, reusable, and easy to maintain.

---

# ❓ Why This Project?

Instead of training a single Machine Learning model, this project performs a comprehensive comparison of multiple classification algorithms.

The workflow includes:

- Data Cleaning
- Duplicate Removal
- Exploratory Data Analysis
- Data Preprocessing
- Feature Scaling
- Training Multiple ML Models
- Model Comparison
- Cross Validation
- Feature Importance Analysis
- ROC Curve Analysis
- Interactive Deployment using Streamlit

This makes the project significantly more comprehensive than a basic classification notebook.

---

# ✨ Key Features

✅ End-to-End Machine Learning Pipeline

✅ Duplicate Detection and Removal

✅ Modular Python Project Structure

✅ Exploratory Data Analysis (EDA)

✅ Data Visualization

✅ Data Preprocessing

✅ Feature Scaling

✅ Five Machine Learning Algorithms

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

✅ Automatic Model Evaluation

✅ Confusion Matrix Generation

✅ ROC Curve Analysis

✅ Cross Validation

✅ Feature Importance Visualization

✅ Model Persistence using Joblib

✅ Interactive Command Line Prediction System

✅ Streamlit Web Application

---

# 📸 Application Preview

> Replace these images with screenshots from your application.

## 🏠 Home Page

```
<img width="1920" height="940" alt="image" src="https://github.com/user-attachments/assets/3287dc66-4c62-4361-9f08-e4a2941b84ef" />

```

---

## 📋 Patient Information Form

```
assets/form.png
```

---

## ❤️ Prediction Result

```
assets/result.png
```

---

# 🔄 Complete Machine Learning Workflow

```text
                 Heart Disease Dataset
                          │
                          ▼
                  Data Cleaning
                          │
                          ▼
               Duplicate Removal
                          │
                          ▼
          Exploratory Data Analysis
                          │
                          ▼
              Data Preprocessing
                          │
                          ▼
          Train-Test Split (80-20)
                          │
                          ▼
              Feature Scaling
                          │
                          ▼
          Train Multiple ML Models
                          │
                          ▼
           Performance Evaluation
                          │
                          ▼
              Model Comparison
                          │
                          ▼
             Cross Validation
                          │
                          ▼
          Feature Importance Analysis
                          │
                          ▼
           Select Best Performing Model
                          │
                          ▼
               Save Model (Joblib)
                          │
                          ▼
         Streamlit Web Application
```

---

# 🏗️ Project Architecture

```text
Disease_Prediction
│
├── dataset/
│     └── heart.csv
│
├── models/
│     ├── best_model.pkl
│     └── scaler.pkl
│
├── reports/
│     ├── metrics/
│     └── plots/
│
├── src/
│     ├── preprocessing.py
│     ├── models.py
│     ├── train.py
│     ├── evaluate.py
│     ├── feature_importance.py
│     ├── cross_validation.py
│     ├── roc_curve.py
│     ├── compare_models.py
│     ├── predict.py
│     └── utils.py
│
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 💡 Highlights

- Modular and maintainable project structure
- Comparison of multiple ML algorithms
- Professional evaluation pipeline
- Interactive Streamlit deployment
- Clean CLI prediction system
- Automatic report generation
- Publication-quality plots
- Reusable utility functions
- Resume-ready project

---

# 📊 Dataset

This project uses the **Heart Disease Dataset**, which contains clinical and diagnostic information collected from patients for heart disease prediction.

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| Samples | 302 (after removing duplicates) |
| Original Samples | 1025 |
| Features | 13 |
| Target Classes | 2 (Heart Disease / No Heart Disease) |
| Problem Type | Binary Classification |

### Features Used

| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest Pain Type |
| trestbps | Resting Blood Pressure |
| chol | Serum Cholesterol |
| fbs | Fasting Blood Sugar |
| restecg | Resting ECG Results |
| thalach | Maximum Heart Rate Achieved |
| exang | Exercise Induced Angina |
| oldpeak | ST Depression |
| slope | Slope of Peak Exercise ST Segment |
| ca | Number of Major Vessels |
| thal | Thalassemia |
| target | Heart Disease (1) / No Heart Disease (0) |

---

# 🧹 Data Cleaning

Before training the Machine Learning models, several preprocessing steps were performed.

✔ Removed duplicate records

✔ Separated features and target variable

✔ Stratified train-test split

✔ Standardized numerical features (where applicable)

✔ Saved fitted scaler for deployment

This ensures that the deployed model receives data in the same format as the training pipeline.

---

# 📈 Exploratory Data Analysis (EDA)

Several exploratory analyses were performed to better understand the dataset.

These include:

- Class Distribution
- Age Distribution
- Correlation Analysis
- Feature Statistics

Generated plots are automatically saved under

```text
reports/plots/
```

### Example Visualizations

Replace these placeholders with your generated figures.

| Plot | Image |
|------|-------|
| Class Distribution | `reports/plots/class_distribution.png` |
| Age Distribution | `reports/plots/age_distribution.png` |

---

# 🤖 Machine Learning Models

Five different Machine Learning algorithms were trained and evaluated.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Linear baseline classifier |
| Support Vector Machine (SVM) | Maximum-margin classifier |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble learning model |
| K-Nearest Neighbors | Distance-based classifier |

Each model was trained independently and evaluated using identical train-test splits to ensure a fair comparison.

---

# 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|--------:|---------:|
| Logistic Regression | **83.12%** | 83.80% | 86.55% | 84.54% |
| SVM | 82.79% | 82.23% | **87.80%** | **84.55%** |
| Decision Tree | 77.48% | 78.03% | 81.70% | 79.60% |
| **Random Forest ⭐** | **83.61%** | 82.68% | 84.15% | 83.02% |
| KNN | 80.81% | 80.31% | 85.98% | 82.95% |

> **Final Model Selected:** **Random Forest Classifier**

Random Forest was chosen as the final deployment model due to its strong predictive performance, robustness, and excellent generalization capability.

---

# 📊 Performance Comparison

The project automatically generates comparison plots for all evaluated models.

Generated figures include:

- Accuracy Comparison
- Precision Comparison
- Recall Comparison
- F1 Score Comparison

Example:

```text
reports/plots/accuracy_comparison.png
```

---

# 🔁 Cross Validation

To evaluate model robustness, **5-Fold Cross Validation** was performed.

This helps estimate how well the model generalizes to unseen data rather than relying on a single train-test split.

### Cross Validation Summary

| Model | Mean Accuracy | Standard Deviation |
|--------|--------------:|------------------:|
| Logistic Regression | 83.12% | ±2.81% |
| SVM | 82.79% | ±2.61% |
| Random Forest | 81.48% | ±4.00% |
| KNN | 80.81% | ±2.74% |
| Decision Tree | 77.48% | ±4.76% |

The relatively low standard deviations indicate consistent model performance across different folds.

---

# 🌲 Feature Importance

Understanding which clinical features contribute most to predictions is essential in medical Machine Learning.

This project includes feature importance analysis using:

- Random Forest Feature Importance
- Logistic Regression Coefficients

Automatically generated plots include:

- Overall Feature Importance
- Top 10 Important Features
- Logistic Regression Coefficient Analysis

Saved under

```text
reports/plots/
```

---

# 📈 ROC Curve Analysis

Receiver Operating Characteristic (ROC) analysis was performed to evaluate classification performance across different thresholds.

Metrics include:

- ROC Curve
- Area Under Curve (AUC)

These plots provide additional insight into model discrimination capability beyond accuracy alone.

---

# 📂 Generated Reports

During training, the project automatically generates reports.

```text
reports/
│
├── metrics/
│   ├── Logistic Regression_metrics.txt
│   ├── SVM_metrics.txt
│   ├── Decision Tree_metrics.txt
│   ├── Random Forest_metrics.txt
│   ├── KNN_metrics.txt
│   └── model_results.csv
│
└── plots/
    ├── confusion matrices
    ├── ROC curves
    ├── feature importance
    ├── comparison graphs
    └── EDA plots
```

This makes experiment tracking and model comparison significantly easier.

---

# 🌐 Streamlit Web Application

An interactive web application has been developed using **Streamlit** to make predictions in real time.

The application provides an intuitive interface where users can enter patient information and instantly receive:

- ❤️ Heart Disease Prediction
- 📊 Prediction Confidence
- ⚠ Risk Level
- 💡 Health Recommendations

### Application Workflow

```text
User Input
      │
      ▼
Input Validation
      │
      ▼
Load Trained Random Forest Model
      │
      ▼
Prediction
      │
      ▼
Probability Estimation
      │
      ▼
Display Results
```

---


# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/mayanksinharay/Disease_Prediction.git
```

Move into the project directory

```bash
cd Disease_Prediction
```

Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

## Train Models

```bash
python src/train.py
```

---

## Compare Models

```bash
python src/compare_models.py
```

---

## Cross Validation

```bash
python src/cross_validation.py
```

---

## Feature Importance

```bash
python src/feature_importance.py
```

---

## ROC Curve

```bash
python src/roc_curve.py
```

---

## CLI Prediction

```bash
python src/predict.py
```

---

## Launch Streamlit App

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```text
Disease_Prediction/
│
├── dataset/
│   └── heart.csv
│
├── models/
│   ├── Random Forest.pkl
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── metrics/
│   └── plots/
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── cross_validation.py
│   ├── feature_importance.py
│   ├── roc_curve.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🛠 Technology Stack

### Programming Language

- Python 3

### Machine Learning

- Scikit-Learn

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Web Application

- Streamlit

### Model Persistence

- Joblib

---

# 📌 Future Improvements

This project can be extended further by incorporating:

- 🔹 Hyperparameter Optimization using GridSearchCV
- 🔹 Explainable AI using SHAP
- 🔹 REST API using FastAPI
- 🔹 Docker Containerization
- 🔹 CI/CD Pipeline with GitHub Actions
- 🔹 Cloud Deployment on AWS/Azure/GCP
- 🔹 Deep Learning Models
- 🔹 Integration with Electronic Health Records (EHR)

---

# 📚 Learning Outcomes

This project demonstrates practical experience in:

- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Machine Learning
- Model Evaluation
- Cross Validation
- Feature Importance
- ROC Analysis
- Model Persistence
- Streamlit Deployment
- Software Project Organization
- Git & GitHub

---

# 🤝 Contributing

Contributions are always welcome.

If you would like to improve this project:

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added a new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

See the **LICENSE** file for more details.

---

# 👨‍💻 Author

**Mayank Sinharay**

B.Tech in Computer Science & Engineering

Aspiring Machine Learning & Data Science Engineer

GitHub: https://github.com/mayanksinharay

LinkedIn: www.linkedin.com/in/mayank-sinharay-2013a3346

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a Star!

It motivates me to build and share more Machine Learning projects.

---

Made with ❤️ using Python, Scikit-Learn & Streamlit

</div>
