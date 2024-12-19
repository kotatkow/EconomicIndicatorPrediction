#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the data series
consumer_confidence = pd.read_csv("../data/processed/preprocessed/Consumer_Confidence_data_preprocessed.csv")
cpi = pd.read_csv("../data/processed/preprocessed/CPI_data_preprocessed.csv")
gdp_growth = pd.read_csv("../data/processed/preprocessed/GDP_Growth_data_preprocessed.csv")
interest_rate = pd.read_csv("../data/processed/preprocessed/Interest_Rate_data_preprocessed.csv")
money_supply = pd.read_csv("../data/processed/preprocessed//Money_Supply_data_preprocessed.csv")
ppi = pd.read_csv("../data/processed/preprocessed/PPI_data_preprocessed.csv")
unemployment_rate = pd.read_csv("../data/processed/preprocessed/Unemployment_Rate_data_preprocessed.csv")

# Preview the first few rows to check the structure and columns
print(consumer_confidence.head())
print(cpi.head())
print(gdp_growth.head())
print(interest_rate.head())
print(money_supply.head())
print(ppi.head())
print(unemployment_rate.head())


# Merging on 'Date'
data = consumer_confidence.merge(cpi, on="Date") \
                          .merge(gdp_growth, on="Date") \
                          .merge(interest_rate, on="Date") \
                          .merge(money_supply, on="Date") \
                          .merge(ppi, on="Date") \
                          .merge(unemployment_rate, on="Date")

print("\n")
print("data merged:")
print(data.head())
output_dir = 'C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/EDA/correlation_matrix'
csv_file_path = os.path.join(output_dir, 'merged_datasets.csv')
data.to_csv(csv_file_path,index=False)
print(f"data saved to {csv_file_path}")

correlation_matrix = data.select_dtypes(include=['number']).corr()
print('\n')
print("correlation_matrix")
print(correlation_matrix)
correlation_csv_file_path = os.path.join(output_dir, 'correlation_matrix.csv')
correlation_matrix.to_csv(correlation_csv_file_path,index=True)
print(f"data saved to {csv_file_path}")

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix for Unemployment Prediction Features")

png_save_path = os.path.join(output_dir,'heatmap.png')
# Save the plot as an image file
plt.savefig(png_save_path, format="png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:




