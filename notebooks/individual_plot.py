#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import os
import pandas as pd
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

# Define output directory for files
output_dir = 'C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/raw'
png_output_path = 'C:/Users/ghkjs/OneDrive/바탕 화면/EconomicIndicatorPrediction/data/raw/png'
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Show plot for each data series and save as CSV and PNG files
for name, series_id in series_ids.items():
    # Fetch the data
    data = fetch_data_from_fred(series_id, api_key)
    
    # Save data as CSV file
    csv_file_path = os.path.join(output_dir, f'{name}.csv')
    data.index = pd.to_datetime(data.index)
    data.index.name = 'Date'
    data.to_csv(csv_file_path, index=True)
    print(f"Data saved to {csv_file_path}")

 # Plot and save each data series as a PNG
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data[series_id], label=name)
    plt.xlabel('Date')
    plt.ylabel(name)
    plt.title(f'{name} Over Time')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    png_file_path = os.path.join(png_output_path, f'{name}.png')
    plt.savefig(png_file_path)
    print(f"Plot saved to {png_file_path}")
    plt.show()
    plt.close()    
    print(data.head())


# In[ ]:




