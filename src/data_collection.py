from fredapi import Fred
import pandas as pd

def fetch_data_from_fred(series_id, api_key):
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id)
    return pd.DataFrame(data, columns=[series_id])

def fetch_multiple_series(series_ids, api_key):
    fred = Fred(api_key=api_key)
    data_frames = []
    
    for name, series_id in series_ids.items():
        series_data = fred.get_series(series_id)
        df = pd.DataFrame(series_data, columns=[name])
        data_frames.append(df)
    
    # Combine all data frames on the date index
    combined_data = pd.concat(data_frames, axis=1)
    return combined_data
