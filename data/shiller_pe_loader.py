import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import io
import os


def load_shiller_pe():
    url_pe = "https://www.multpl.com/shiller-pe/table/by-month"
    url_sp500 = "https://www.multpl.com/s-p-500-historical-prices/table/by-month"  # Using nominal S&P 500 prices instead of inflation-adjusted
    cache_pe = os.path.join(os.path.dirname(__file__), 'shiller_pe_data.csv')
    cache_sp500 = os.path.join(os.path.dirname(__file__), 'sp500_data.csv')

    # Helper to load and update a dataset
    def load_and_update(url, cache_path, col_names, value_col):
        # Try to load local cache
        if os.path.exists(cache_path):
            df_local = pd.read_csv(cache_path, parse_dates=['Date'])
        else:
            df_local = pd.DataFrame(columns=col_names)
        # Always pull the latest data from the web
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        html = str(soup)
        df_web = pd.read_html(io.StringIO(html))[0]
        df_web.columns = col_names
        df_web['Date'] = pd.to_datetime(df_web['Date'], format="mixed")
        df_web[value_col] = pd.to_numeric(df_web[value_col], errors='coerce')
        df_web.sort_values('Date', inplace=True)
        # If local cache exists, check if update is needed
        if not df_local.empty:
            last_local = df_local['Date'].max()
            last_web = df_web['Date'].max()
            if last_web > last_local:
                # Append only new rows
                df_new = df_web[df_web['Date'] > last_local]
                df_local = pd.concat([df_local, df_new], ignore_index=True)
                df_local.drop_duplicates(subset=['Date'], inplace=True)
                df_local.sort_values('Date', inplace=True)
                df_local.to_csv(cache_path, index=False)
            # else: no update needed
        else:
            # No local cache, save all
            df_local = df_web.copy()
            df_local.to_csv(cache_path, index=False)
        return df_local

    data_pe = load_and_update(url_pe, cache_pe, ['Date', 'PE_Ratio'], 'PE_Ratio')
    data_sp500 = load_and_update(url_sp500, cache_sp500, ['Date', 'S&P_500'], 'S&P_500')

    print(data_pe.head())
    print(data_sp500.head())

    return data_pe, data_sp500

if __name__ == "__main__":
    load_shiller_pe()