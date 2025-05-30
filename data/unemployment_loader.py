import pandas_datareader.data as web
import pandas as pd
import os

def load_unemployment():
    """
    Loads US unemployment rate data from FRED, using a local CSV cache if available and only updating with new data.
    Returns a DataFrame with monthly unemployment rates.
    """
    start_date = "1900-01-01"
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    cache_path = os.path.join(os.path.dirname(__file__), 'unemployment_data.csv')

    # Try to load local cache
    if os.path.exists(cache_path):
        df_local = pd.read_csv(cache_path, parse_dates=['Date'])
        last_local = df_local['Date'].max()
        # Only fetch new data if needed
        if pd.to_datetime(end_date) > last_local:
            df_new = web.DataReader("UNRATE", "fred", start=last_local + pd.Timedelta(days=1), end=end_date)
            df_new = df_new.reset_index()
            df_new.columns = ['Date', 'Unemployment']
            df_new.dropna(inplace=True)
            df_local = pd.concat([df_local, df_new], ignore_index=True)
            df_local.drop_duplicates(subset=['Date'], inplace=True)
            df_local.sort_values('Date', inplace=True)
            df_local.to_csv(cache_path, index=False)
        df_unemp = df_local
    else:
        # No local cache, fetch all
        df_unemp = web.DataReader("UNRATE", "fred", start=start_date, end=end_date)
        df_unemp = df_unemp.reset_index()
        df_unemp.columns = ['Date', 'Unemployment']
        df_unemp.dropna(inplace=True)
        df_unemp.to_csv(cache_path, index=False)

    print(df_unemp.head())
    return df_unemp

if __name__ == "__main__":
    load_unemployment()