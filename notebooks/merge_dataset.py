#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import os

# Reload datasets (ensure alignment for merging with lagged predictors)
# Load the provided preprocessed files
unemployment_rate = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/Unemployment_Rate_data_preprocessed.csv')
consumer_confidence = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/Consumer_Confidence_data_preprocessed.csv')
cpi = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/CPI_data_preprocessed.csv')
gdp_growth = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/GDP_Growth_data_preprocessed.csv')
interest_rate = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/Interest_Rate_data_preprocessed.csv')
money_supply = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/Money_Supply_data_preprocessed.csv')
ppi = pd.read_csv('C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed/PPI_data_preprocessed.csv')

# Convert 'Date' to datetime for all datasets
datasets = [interest_rate, money_supply, ppi, cpi, unemployment_rate, gdp_growth, consumer_confidence]
for df in datasets:
    df['Date'] = pd.to_datetime(df['Date'])

# Merge datasets on 'Date'
merged = unemployment_rate.copy()
merged = merged.merge(interest_rate, on='Date', how='inner', suffixes=('', '_int'))
merged = merged.merge(money_supply, on='Date', how='inner', suffixes=('', '_money'))
merged = merged.merge(ppi, on='Date', how='inner', suffixes=('', '_ppi'))
merged = merged.merge(cpi, on='Date', how='inner', suffixes=('', '_cpi'))
merged = merged.merge(gdp_growth, on='Date', how='inner', suffixes=('', '_gdp'))
merged = merged.merge(consumer_confidence, on='Date', how='inner', suffixes=('', '_conf'))

# Rename columns for clarity
merged.columns = [
    'Date', 'Unemployment_Rate', 'Interest_Rate', 'Money_Supply', 'PPI',
    'CPI', 'GDP_Growth', 'Consumer_Confidence'
]

# Apply 12-month lag for Interest Rate, Money Supply, PPI, and CPI
lag = 12
merged['Interest_Rate_Lagged'] = merged['Interest_Rate'].shift(lag)
merged['Money_Supply_Lagged'] = merged['Money_Supply'].shift(lag)
merged['PPI_Lagged'] = merged['PPI'].shift(lag)
merged['CPI_Lagged'] = merged['CPI'].shift(lag)

# Drop rows with NaN values due to lagging
merged_lagged = merged.dropna()

# Save the resulting dataset for review
output_file = '../data/lagged_dataset'
save_path = os.path.join(output_file, 'lagged_dataset_merged.csv')
merged_lagged.to_csv(save_path, index=False)
output_file


# In[ ]:




