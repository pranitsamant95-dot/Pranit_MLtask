import streamlit as st
import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
 
st.title("ML Task Selector: Classification vs Regression")
 
# -----------------------------
# 1. Dataset Input
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
 
# -----------------------------
# 2. Task Selection
# -----------------------------
task_type = st.radio(
    "Select Task Type",
    ["Classification", "Regression"]
)
 
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
 
    # -----------------------------
    # 3. Target Variable
    # -----------------------------
    target = st.selectbox("Select Target Variable", df.columns)
 
    # -----------------------------
    # 4. Feature Variables
    # -----------------------------
    features = st.multiselect(
        "Select Feature Variables",
        [col for col in df.columns if col != target]
    )
 
    # -----------------------------
    # 5. Test Size
    # -----------------------------
    test_size = st.slider("Test Size", 0.1, 0.5, 0.3)
 
    if st.button("Run Model"):
 
        if not features:
            st.error("Please select at least one feature variable.")
            st.stop()
 
        X = df[features]
        y = df[target]
 
        # -----------------------------
        # 6. Target Validation
        # -----------------------------
        if task_type == "Classification":
            if y.nunique() > 10 or y.dtype in ["float64", "float32"]:
                st.error(
                    "❌ Invalid target for Classification.\n"
                    "Target must be a DISCRETE variable."
                )
                st.stop()
 
        if task_type == "Regression":
            if y.dtype not in ["float64", "int64"]:
                st.error(
                    "❌ Invalid target for Regression.\n"
                    "Target must be a CONTINUOUS variable."
                )
                st.stop()
 
        # -----------------------------
        # 7. Train-Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )
 
        # -----------------------------
        # 8. Classification
        # -----------------------------
        if task_type == "Classification":
            model = GaussianNB()
            model.fit(X_train, y_train)
 
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
 
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            cm = confusion_matrix(y_test, y_test_pred)
 
            st.subheader("Results (Classification)")
            st.write("Training Accuracy:", round(train_acc, 4))
            st.write("Testing Accuracy:", round(test_acc, 4))
            st.write("Confusion Matrix:")
            st.dataframe(cm)
 
        # -----------------------------
        # 9. Regression
        # -----------------------------
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
 
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
 
            st.subheader("Results (Regression)")
            st.write("Training R² Score:", round(train_r2, 4))
            st.write("Testing R² Score:", round(test_r2, 4))
 