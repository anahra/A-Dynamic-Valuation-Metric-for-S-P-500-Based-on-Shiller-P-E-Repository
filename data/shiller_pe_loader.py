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
        df_local = pd.DataFrame(columns=col_names + ['Date'])
        needs_web_update = True
        
        # 1. Load local cache if it exists
        if os.path.exists(cache_path):
            df_local = pd.read_csv(cache_path, parse_dates=['Date'])
            mtime = os.path.getmtime(cache_path)
            
            # Check if cache has data for the current month
            current_month_start = pd.Timestamp.now().normalize().replace(day=1)
            has_current_month = False
            if not df_local.empty:
                has_current_month = (df_local['Date'].dt.to_period('M') == current_month_start.to_period('M')).any()
            
            # Only skip web update if cache is recent AND has current month data
            if (datetime.now().timestamp() - mtime) < 86400 and has_current_month:
                needs_web_update = False
        
        # 2. Fetch from web if needed
        if needs_web_update:
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                html = str(soup)
                df_web = pd.read_html(io.StringIO(html))[0]
                df_web.columns = col_names
                df_web['Date'] = pd.to_datetime(df_web['Date'], format="mixed")
                df_web[value_col] = pd.to_numeric(df_web[value_col], errors='coerce')
                
                # Merge web data with local data, preferring local cached values
                # (cache has accumulated actual end-of-month prices from yfinance)
                if not df_local.empty:
                    df_local = pd.concat([df_local, df_web], ignore_index=True)
                    df_local.drop_duplicates(subset=['Date'], keep='first', inplace=True)
                else:
                    df_local = df_web.copy()
            except Exception as e:
                print(f"Error updating {url} from web: {e}")
        
        # 3. Process and group the data to ensure correct display
        if not df_local.empty:
            df_local = df_local.sort_values('Date').reset_index(drop=True)
            df_local['Month'] = df_local['Date'].dt.to_period('M')
            current_period = pd.Timestamp.now().to_period('M')
            
            # Historical months: Keep only the LAST available point of each previous month,
            # and label it as the 1st of that month on the chart.
            df_historical = df_local[df_local['Month'] < current_period].copy()
            if not df_historical.empty:
                df_historical = df_historical.sort_values('Date').groupby('Month').last().reset_index()
                df_historical['Date'] = df_historical['Month'].dt.to_timestamp()
                df_historical = df_historical.drop(columns=['Month'])
            
            # Current month: Keep ONLY the first available point (anchor)
            # and the absolute latest available point.
            df_current = df_local[df_local['Month'] == current_period].copy()
            if not df_current.empty:
                df_current = df_current.sort_values('Date')
                if len(df_current) > 2:
                    df_current = pd.concat([df_current.iloc[[0]], df_current.iloc[[-1]]])
                df_current = df_current.drop_duplicates(subset=['Date'])
                df_current = df_current.drop(columns=['Month'])
            
            df_to_save = pd.concat([df_historical, df_current], ignore_index=True)
            df_to_save.sort_values('Date', inplace=True)
            df_to_save.drop_duplicates(subset=['Date'], inplace=True)
            df_to_save = df_to_save.reset_index(drop=True)
            
            # Save the processed data back to cache
            df_to_save.to_csv(cache_path, index=False)
            return df_to_save
        
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
        
        # --- Ensure we have a 1st-of-month anchor for the current month ---
        current_month_start = pd.Timestamp.now().normalize().replace(day=1)
        has_month_start = (data_pe['Date'] == current_month_start).any()
        
        if not has_month_start:
            # The web source doesn't have the 1st of this month yet.
            # Fetch the first trading day's close from yfinance as the anchor.
            try:
                month_start_data = ticker.history(
                    start=current_month_start, 
                    end=current_month_start + timedelta(days=7)
                )
                if not month_start_data.empty:
                    first_price = month_start_data['Close'].iloc[0]
                    first_pe = first_price / implied_earnings
                    
                    # Use the 1st of the month as the display date (consistent with historical)
                    new_pe_row = pd.DataFrame({'Date': [current_month_start], 'PE_Ratio': [first_pe]})
                    new_sp500_row = pd.DataFrame({'Date': [current_month_start], 'S&P_500': [first_price]})
                    
                    data_pe = pd.concat([data_pe, new_pe_row], ignore_index=True)
                    data_pe.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                    data_sp500 = pd.concat([data_sp500, new_sp500_row], ignore_index=True)
                    data_sp500.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            except Exception as e:
                print(f"Error fetching 1st-of-month anchor: {e}")
        
        # --- Fetch today's live price ---
        live_df = ticker.history(period="1d")
        
        if not live_df.empty:
            live_price = live_df['Close'].iloc[-1]
            live_date = live_df.index[-1].replace(tzinfo=None)
            
            # Avoid duplicating the 1st-of-month point
            if live_date.day == 1 and live_date.month == current_month_start.month:
                # Today IS the 1st — update the anchor instead of adding a new point
                data_pe.loc[data_pe['Date'] == current_month_start, 'PE_Ratio'] = live_price / implied_earnings
                data_sp500.loc[data_sp500['Date'] == current_month_start, 'S&P_500'] = live_price
            elif live_date >= last_date:
                # Normal case: add today's live data as a separate point
                live_pe = live_price / implied_earnings
                
                new_pe_row = pd.DataFrame({'Date': [live_date], 'PE_Ratio': [live_pe]})
                new_sp500_row = pd.DataFrame({'Date': [live_date], 'S&P_500': [live_price]})
                
                data_pe = pd.concat([data_pe, new_pe_row], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')
                data_sp500 = pd.concat([data_sp500, new_sp500_row], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')
                
    except Exception as e:
        print(f"Error fetching live data: {e}")

    # --- Final Cleanup ---
    # Ensure current month only shows TWO points: 1st of month (anchor) and absolute latest.
    def finalize_month_display(df):
        if df.empty: return df
        current_period = pd.Timestamp.now().to_period('M')
        mask_current = df['Date'].dt.to_period('M') == current_period
        if not mask_current.any():
            return df
        df_hist = df[~mask_current].copy()
        df_curr = df[mask_current].sort_values('Date')
        if len(df_curr) > 2:
            # Keep only the first (1st of month anchor) and the absolute last point
            df_curr = pd.concat([df_curr.iloc[[0]], df_curr.iloc[[-1]]])
        # If both points have the same date, keep just one
        df_curr = df_curr.drop_duplicates(subset=['Date'])
        return pd.concat([df_hist, df_curr], ignore_index=True).sort_values('Date').reset_index(drop=True)

    data_pe = finalize_month_display(data_pe)
    data_sp500 = finalize_month_display(data_sp500)

    return data_pe, data_sp500