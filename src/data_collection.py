from fredapi import Fred
import pandas as pd

def fetch_data_from_fred(series_id, api_key):
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id)
    return pd.DataFrame(data, columns=[series_id])
