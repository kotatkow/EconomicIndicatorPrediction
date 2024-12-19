import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title and Description
st.title("Economic Indicator Prediction: Linear Regression")
st.write("This app demonstrates a linear regression model to predict unemployment rates based on economic indicators.")

# Sidebar: File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Feature Selection
    features = ['Interest_Rate_Lagged', 'Money_Supply_Lagged', 'PPI_Lagged', 'CPI_Lagged', 'GDP_Growth', 'Consumer_Confidence']
    target = 'Unemployment_Rate'

    X = data[features]
    y = data[target]

    # Split Data
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))

    st.write("### Model Performance")
    st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"- R² Score: {r2:.2f}")

    # Feature Importance
    st.write("### Feature Importance")
    coefficients = model.coef_
    plt.figure(figsize=(10, 6))
    plt.barh(features, coefficients, color='skyblue')
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.xlabel("Coefficient Value")
    st.pyplot(plt)

    # Residuals
    st.write("### Residual Analysis")
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, color='orange', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Residuals of Linear Regression Model")
    plt.xlabel("Actual Unemployment Rate")
    plt.ylabel("Residuals")
    st.pyplot(plt)

    # Predicted vs Actual
    st.write("### Predicted vs Actual")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title("Predicted vs Actual Unemployment Rates")
    plt.xlabel("Actual Unemployment Rate")
    plt.ylabel("Predicted Unemployment Rate")
    plt.legend()
    st.pyplot(plt)
else:
    st.write("Upload a dataset to get started!")

