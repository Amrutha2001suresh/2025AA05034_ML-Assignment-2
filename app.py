import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# ==========================================
# 1. App Title & Description
# ==========================================
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")
st.title("🔬 Breast Cancer Prediction App")
st.write("This app allows you to upload data and predict using 6 different Machine Learning models.")

# ==========================================
# 2. Sidebar: Model Selection
# ==========================================
st.sidebar.header("User Input Features")
model_options = [
    "Logistic Regression", 
    "Decision Tree", 
    "kNN", 
    "Naive Bayes", 
    "Random Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select a Model", model_options)

# ==========================================
# 3. File Upload
# ==========================================
st.write("### Upload Data")

# --- NEW CODE: Download Button for Sample Data ---
# This checks if the file exists (it should be in your repo) and creates a button
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.download_button(
            label="⬇️ Download Sample Test Data (CSV)",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
            help="Download this file to test the app if you don't have one."
        )
else:
    st.warning("⚠️ 'test_data.csv' not found in repository.")
# -------------------------------------------------

uploaded_file = st.file_uploader("Upload your input CSV file (Test Data)", type=["csv"])

    # ==========================================
    # 4. Load Model & Predict
    # ==========================================
    # Map friendly name to filename
    model_filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    model_path = os.path.join("model", model_filename_map[selected_model_name])
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}. Please check your 'model' folder on GitHub.")
        st.stop()

    # Make Predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Error during prediction. Ensure your input columns match the model features.\nDetails: {e}")
        st.stop()

    # ==========================================
    # 5. Display Evaluation Metrics (If target exists)
    # ==========================================
    if y_test is not None:
        st.write(f"### 📊 Performance Metrics: {selected_model_name}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        col5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # ==========================================
        # 6. Confusion Matrix
        # ==========================================
        st.write("### 📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
    else:
        st.success("Predictions generated successfully!")
        st.write(y_pred)

else:
    st.info("Please upload a CSV file to proceed.")
