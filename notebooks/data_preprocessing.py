#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Add the 'src' directory to the Python path
src_path = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_path)

# Now you can import your function
from data_collection import fetch_data_from_fred

# Define your FRED API key
api_key = 'e916710d165717e6348556cdce8111f3'

# Define the series IDs for the indicators you want to collect
series_ids = {
    'Unemployment_Rate': 'UNRATE',
    'GDP_Growth': 'A191RL1Q225SBEA',
    'CPI': 'CPIAUCSL',
    'Interest_Rate': 'FEDFUNDS',
    'Money_Supply': 'M2SL',
    'PPI': 'PPIACO',
    'Consumer_Confidence': 'UMCSENT'
}

scaler = MinMaxScaler()

# Define output directory for files
output_dir = 'C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed'
png_out_dir = 'C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/processed/preprocessed'

starting_dates = []
ending_dates = []

for name, series_id in series_ids.items():
    # Fetch the data
    data = fetch_data_from_fred(series_id, api_key)
    
    # Ensure the index is datetime
    data.index = pd.to_datetime(data.index)
    
    # Record the first and last date
    starting_dates.append(data.index.min())
    ending_dates.append(data.index.max())

# Find the latest starting date and earliest ending date
latest_start_date = max(starting_dates)
earliest_end_date = min(ending_dates)
print(f"Latest starting date among all series: {latest_start_date.date()}")
print(f"Earliest ending date among all series: {earliest_end_date.date()}")

# Show plot for each data series and save as CSV and PNG files
for name, series_id in series_ids.items():
    # Fetch the data
    data = fetch_data_from_fred(series_id, api_key)

    data.index = pd.to_datetime(data.index)

    if name == 'CPI':
        data['CPIAUCSL'] = data['CPIAUCSL'].pct_change(periods=12)*100
    if name == 'Money_Supply':
        data['M2SL'] = data['M2SL'].pct_change(periods=12)*100
    if name == 'PPI':
        data['PPIACO'] = data['PPIACO'].pct_change(periods=12)*100

    # Trim data to the date range
    data = data[(data.index >= latest_start_date) & (data.index <= earliest_end_date)]
    
    # Resample to monthly frequency
    data_monthly = data.resample('ME').mean()
    
    # Apply linear interpolation for missing values
    data_interpolated = data_monthly.interpolate(method='linear')
    data_interpolated = data_interpolated.bfill()

    # Normalize the data
    data_normalized = data_interpolated
    data_normalized[[series_id]] = scaler.fit_transform(data_normalized[[series_id]])
    
    # Save the file as csv
    csv_file_path = os.path.join(output_dir, f'{name}_data_preprocessed.csv')
    data_normalized.index.name='Date'
    data_normalized.to_csv(csv_file_path, index=True)
    print(f'first and last several elements of {name}')
    print(data_normalized.head())
    print(data_normalized.tail())
    print(f'Data saved to {csv_file_path}')
    print('\n')

    # Plot and save each data series as a PNG
    plt.figure(figsize=(10, 5))
    plt.plot(data_normalized.index, data_normalized[series_id], label=name)
    plt.xlabel('Date')
    plt.ylabel(name)
    plt.title(f'Normalized {name} Over Time')
    plt.legend()
    plt.grid(True)
    # Save the plot as a PNG file
    png_file_path = os.path.join(png_out_dir, f'{name} normalized.png')
    plt.savefig(png_file_path)
    print(f"Plot saved to {png_file_path}")
    plt.show()
    plt.close()






# In[ ]:




