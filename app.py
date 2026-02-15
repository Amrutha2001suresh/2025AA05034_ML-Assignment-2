import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import xgboost # Required for XGBoost models

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
# 3. File Upload & Sample Data
# ==========================================
st.write("### Upload Data")

# --- Download Button for Sample Data ---
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

uploaded_file = st.file_uploader("Upload your input CSV file (Test Data)", type=["csv"])

# ==========================================
# 4. Main Execution Block
# ==========================================
if uploaded_file is not None:
    # --------------------------------------
    # A. Read and Preprocess Data
    # --------------------------------------
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Identify Target Column
    target_col = None
    if 'diagnosis' in df.columns:
        target_col = 'diagnosis'
        # Convert M/B to 0/1 if needed
        if df[target_col].dtype == 'object':
             df[target_col] = df[target_col].map({'M': 0, 'B': 1})
    elif 'target' in df.columns:
        target_col = 'target'
    
    # Define X_test and y_test
    if target_col:
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
    else:
        st.warning("⚠️ No 'diagnosis' or 'target' column found. Metrics cannot be calculated.")
        X_test = df
        y_test = None

    # Clean extraneous columns (Common in Kaggle data)
    cols_to_drop = [col for col in ['id', 'Unnamed: 32'] if col in X_test.columns]
    if cols_to_drop:
        X_test = X_test.drop(columns=cols_to_drop)

    # --------------------------------------
    # B. Load Model & Predict
    # --------------------------------------
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
        st.error(f"❌ Model file not found: {model_path}. Check your 'model' folder.")
        st.stop()

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # --------------------------------------
    # C. Display Metrics
    # --------------------------------------
    if y_test is not None:
        st.write(f"### 📊 Performance Metrics: {selected_model_name}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        col5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # Confusion Matrix
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
