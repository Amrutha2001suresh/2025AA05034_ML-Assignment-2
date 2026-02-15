## 👤 Author
* **Student Name:** Amrutha Sureshkumar Sunitha
* **Student ID:** 2025AA05034
# Breast Cancer Classification App (ML Assignment 2)

## 📌 Overview
This repository contains the code and resources for Machine Learning Assignment 2. The project involves building a classification model to predict Breast Cancer (Malignant vs. Benign) using the Breast Cancer Wisconsin (Diagnostic) Dataset and deploying it as an interactive Streamlit web application.

## 📊 Problem Statement
The goal is to develop a machine learning pipeline that:
1.  Preprocesses the Breast Cancer dataset (handling missing values, encoding targets).
2.  Trains **six different classification models** (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost).
3.  Evaluates models based on Accuracy, Precision, Recall, F1 Score, and MCC.
4.  Deploys the best solution via a user-friendly web interface.

## 📂 Dataset Description
* **Source:** [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Features:** 30 numerical features computed from digitized images of fine needle aspirate (FNA) of a breast mass.
* **Target:** Diagnosis (M = Malignant, B = Benign).
* **Size:** 569 instances, 32 columns.


## 📈 Model Performance Comparison
The following metrics were obtained during the training phase on the BITS Virtual Lab:

| ML Model Name | Accuracy | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9649 | 0.9583 | 0.9857 | 0.9718 | 0.9254 |
| **Decision Tree** | 0.9298 | 0.9315 | 0.9571 | 0.9441 | 0.8504 |
| **kNN** | 0.9474 | 0.9444 | 0.9714 | 0.9577 | 0.8872 |
| **Naive Bayes** | 0.9649 | 0.9583 | 0.9857 | 0.9718 | 0.9254 |
| **Random Forest** | 0.9649 | 0.9583 | 0.9857 | 0.9718 | 0.9254 |
| **XGBoost** | 0.9649 | 0.9583 | 0.9857 | 0.9718 | 0.9254 |

*Note: Random Forest and XGBoost provided the most robust performance with high stability.*
## 📝 Model Performance Observations
*(Based on the evaluation metrics obtained from BITS Virtual Lab)*

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well (~96.5% accuracy), indicating that the dataset has linearly separable features which this model exploits effectively. |
| **Decision Tree** | Showed the lowest comparative accuracy (~93%). This suggests it may have slightly overfitted the training data or struggled to capture the smooth decision boundaries compared to ensemble methods. |
| **kNN** | Achieved strong performance (~94.7%), proving that tumors with similar feature magnitudes tend to belong to the same class. |
| **Naive Bayes** | Surprisingly high accuracy (~96.5%) and recall, suggesting that the assumption of feature independence holds reasonably well for these medical metrics. |
| **Random Forest** | (Ensemble) Tied for top performance (~96.5%). By averaging multiple trees, it reduced the variance seen in the single Decision Tree model, resulting in a more robust prediction. |
| **XGBoost** | (Ensemble) delivered top-tier performance (~96.5%) with high AUC. Its gradient boosting approach effectively corrected errors from previous iterations, making it highly reliable for this classification task. |

### 🏆 Final Conclusion
**Logistic Regression** and **Random Forest** are the recommended models for this application due to their high accuracy and balance between Precision and Recall.
## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud
