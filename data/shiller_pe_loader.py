import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import io
import os
import yfinance as yf
from datetime import datetime, timedelta

def load_shiller_pe():
    url_pe = "https://www.multpl.com/shiller-pe/table/by-month"
    url_sp500 = "https://www.multpl.com/s-p-500-historical-prices/table/by-month"
    cache_pe = os.path.join(os.path.dirname(__file__), 'shiller_pe_data.csv')
    cache_sp500 = os.path.join(os.path.dirname(__file__), 'sp500_data.csv')

    def load_and_update(url, cache_path, col_names, value_col):
        # Check if cache exists and is recent (< 24 hours)
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (datetime.now().timestamp() - mtime) < 86400: # 24 hours
                return pd.read_csv(cache_path, parse_dates=['Date'])
            
            df_local = pd.read_csv(cache_path, parse_dates=['Date'])
        else:
            df_local = pd.DataFrame(columns=col_names)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            html = str(soup)
            df_web = pd.read_html(io.StringIO(html))[0]
            df_web.columns = col_names
            df_web['Date'] = pd.to_datetime(df_web['Date'], format="mixed")
            df_web[value_col] = pd.to_numeric(df_web[value_col], errors='coerce')
            df_web.sort_values('Date', inplace=True)
            
            if not df_local.empty:
                last_local = df_local['Date'].max()
                last_web = df_web['Date'].max()
                if last_web > last_local:
                    df_new = df_web[df_web['Date'] > last_local]
                    df_local = pd.concat([df_local, df_new], ignore_index=True)
                    df_local.drop_duplicates(subset=['Date'], inplace=True)
                    df_local.sort_values('Date', inplace=True)
            else:
                df_local = df_web.copy()

            df_local['Month'] = df_local['Date'].dt.to_period('M')
            current_period = pd.Timestamp.now().to_period('M')
            
            df_historical = df_local[df_local['Month'] < current_period].copy()
            df_current = df_local[df_local['Month'] == current_period].copy()
            
            if not df_historical.empty:
                df_historical = df_historical.sort_values('Date').groupby('Month').last().reset_index()
                df_historical['Date'] = df_historical['Month'].dt.to_timestamp()
                df_historical = df_historical.drop(columns=['Month'])
                
            if not df_current.empty:
                df_current = df_current.drop(columns=['Month'])
                
            df_to_save = pd.concat([df_historical, df_current], ignore_index=True)
            df_to_save.sort_values('Date', inplace=True)
            df_to_save.to_csv(cache_path, index=False)
            
            return df_to_save
        except Exception as e:
            print(f"Error updating from web: {e}")
            return df_local

    data_pe = load_and_update(url_pe, cache_pe, ['Date', 'PE_Ratio'], 'PE_Ratio')
    data_sp500 = load_and_update(url_sp500, cache_sp500, ['Date', 'S&P_500'], 'S&P_500')

    # --- Live Data Logic ---
    try:
        # Get latest monthly data point (Anchor)
        last_date = data_pe['Date'].iloc[-1]
        last_pe = data_pe['PE_Ratio'].iloc[-1]
        last_price = data_sp500[data_sp500['Date'] == last_date]['S&P_500'].iloc[0]
        
        # Calculate implied Earnings (E10)
        implied_earnings = last_price / last_pe

        # Fetch Live Price
        ticker = yf.Ticker("^GSPC")
        # Get today's data (or last trading day)
        live_df = ticker.history(period="1d")
        
        if not live_df.empty:
            live_price = live_df['Close'].iloc[-1]
            live_date = live_df.index[-1].replace(tzinfo=None) # Remove timezone for compatibility
            
            # Check if we need to append
            # We append if the live date is strictly greater than the last monthly date
            if live_date > last_date:
                # Calculate Live PE
                live_pe = live_price / implied_earnings
                
                # Create new rows
                new_pe_row = pd.DataFrame({'Date': [live_date], 'PE_Ratio': [live_pe]})
                new_sp500_row = pd.DataFrame({'Date': [live_date], 'S&P_500': [live_price]})
                
                # Append to dataframes (in memory only)
                data_pe = pd.concat([data_pe, new_pe_row], ignore_index=True)
                data_sp500 = pd.concat([data_sp500, new_sp500_row], ignore_index=True)
                
    except Exception as e:
        print(f"Error fetching live data: {e}")

    return data_pe, data_sp500