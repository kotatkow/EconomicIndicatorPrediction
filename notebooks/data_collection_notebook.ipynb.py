#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

# Add the 'src' directory to the Python path
src_path = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_path)

# Now you can import your function
from data_collection import fetch_multiple_series

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

# Fetch the data using the imported function
data = fetch_multiple_series(series_ids, api_key)

# Display the first few rows of the data
print(data.head())


# In[ ]:




