#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set page title
st.title("Polynomial Regression for Unemployment Rate Prediction")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Extract lagged predictors and additional features as specified
    X_lagged = data[['Interest_Rate_Lagged', 'Money_Supply_Lagged', 'PPI_Lagged', 
                     'CPI_Lagged', 'Consumer_Confidence', 'GDP_Growth']]
    y = data['Unemployment_Rate']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_lagged, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply Polynomial Features
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Fit a Linear Regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_test_pred = model.predict(X_test_poly)

    # Evaluate the model
    mse_lagged = mean_squared_error(y_test, y_test_pred)
    rmse_lagged = np.sqrt(mse_lagged)
    mae_lagged = mean_absolute_error(y_test, y_test_pred)
    r2_lagged = r2_score(y_test, y_test_pred)

    st.write("### Model Evaluation Metrics")
    st.write({
        "Mean Squared Error (MSE)": mse_lagged,
        "Root Mean Squared Error (RMSE)": rmse_lagged,
        "Mean Absolute Error (MAE)": mae_lagged,
        "R-squared": r2_lagged
    })

    # Plot: Predicted vs Actual Unemployment Rates
    st.write("### Predicted vs Actual Unemployment Rates")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
    ax.set_title('Predicted vs Actual Unemployment Rates')
    ax.set_xlabel('Actual Unemployment Rate')
    ax.set_ylabel('Predicted Unemployment Rate')
    ax.legend()
    st.pyplot(fig)

    # Plot: Residuals
    residuals = y_test - y_test_pred
    st.write("### Residuals of Polynomial Regression Model")
    fig, ax = plt.subplots()
    ax.scatter(y_test, residuals, color='orange', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Residuals of Polynomial Regression Model')
    ax.set_xlabel('Actual Unemployment Rate')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)

    # Retrieve feature names after polynomial transformation
    feature_names = poly.get_feature_names_out(['Interest_Rate_Lagged', 'Money_Supply_Lagged',
                                                'PPI_Lagged', 'CPI_Lagged', 
                                                'Consumer_Confidence', 'GDP_Growth'])

    # Retrieve coefficients from the trained model
    coefficients = model.coef_

    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    st.write("### Top 10 Features")
    st.dataframe(feature_importance.head(10))

