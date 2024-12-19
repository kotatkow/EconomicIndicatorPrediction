#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import os


# Load the user's data to check the structure and proceed with the analysis
file_path = '../data/raw/Unemployment_Rate.csv'
unemployment_data = pd.read_csv(file_path)

# Display the first few rows and columns info to understand the data
unemployment_data.head(), unemployment_data.info()

# Plotting the historical trend of the unemployment rate
plt.figure(figsize=(12, 6))
plt.plot(unemployment_data['UNRATE'], label='Unemployment Rate')
plt.title('Historical Unemployment Rate')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.show()


# Decomposing the time series to observe trend and seasonality
decomposition = sm.tsa.seasonal_decompose(unemployment_data['UNRATE'], model='additive', period=12)

# Plotting the decomposed components: Observed, Trend, Seasonal, Residual
decomposition.plot()

#save the file and show the plot
png_file_path = os.path.join('../data/EDA/trend_analysis', 'Unemployment Rate Trend Analysis.png')
plt.savefig(png_file_path)
plt.show()
print('file saved')


# In[ ]:




