import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import io
import plotly.graph_objects as go

# Load unemployment rate data
file_path = r"C:\Users\agusn\OneDrive - HEC Paris\Escritorio\General\4_Investing\2_Stocks\Risk\UNRATE.xlsx"
unemployment_data = pd.read_excel(file_path, sheet_name='Monthly')

# Ensure proper column names
unemployment_data.columns = ["Date", "Unemployment_Rate"]
unemployment_data['Date'] = pd.to_datetime(unemployment_data['Date'])
unemployment_data.sort_values('Date', inplace=True)

# Cap extreme values on both high and low ends
lower_cap = unemployment_data['Unemployment_Rate'].quantile(0.01)  # 1st percentile
upper_cap = unemployment_data['Unemployment_Rate'].quantile(0.99)  # 99th percentile

unemployment_data['Unemployment_Rate'] = np.clip(unemployment_data['Unemployment_Rate'], lower_cap, upper_cap)

# Apply a 6-month moving average
unemployment_data['Unemployment_Rate_Smoothed'] = unemployment_data['Unemployment_Rate'].rolling(window=6).mean()

# Apply natural log transformation
unemployment_data['Log_Unemployment_Rate'] = np.log(unemployment_data['Unemployment_Rate_Smoothed'])

# Compute historical mean and standard deviation in log space
log_mean = unemployment_data['Log_Unemployment_Rate'].mean()
log_std = unemployment_data['Log_Unemployment_Rate'].std()

# Compute upper and lower bounds in log scale (Mean ± 2 Std Dev)
log_upper_bound = log_mean + 2 * log_std
log_lower_bound = log_mean - 2 * log_std

# Convert bounds back to original scale
unemployment_data['Hist_Upper_Bound'] = np.exp(log_upper_bound)
unemployment_data['Hist_Lower_Bound'] = np.exp(log_lower_bound)

# Fetch Inflation-Adjusted S&P 500 data
url_sp500 = "https://www.multpl.com/inflation-adjusted-s-p-500/table/by-month"
response_sp500 = requests.get(url_sp500)
response_sp500.raise_for_status()
soup_sp500 = BeautifulSoup(response_sp500.text, 'html.parser')
html_sp500 = str(soup_sp500)
data_sp500 = pd.read_html(io.StringIO(html_sp500))[0]

data_sp500.columns = ['Date', 'S&P_500']
data_sp500['Date'] = pd.to_datetime(data_sp500['Date'], format="mixed")
data_sp500['S&P_500'] = pd.to_numeric(data_sp500['S&P_500'], errors='coerce')
data_sp500.sort_values('Date', inplace=True)

# Merge both datasets
data = pd.merge(unemployment_data, data_sp500, on='Date', how='inner')
data.dropna(inplace=True)

# --- Calculate "Risk" Using Historical Bounds ---
data['Risk'] = 1 - (data['Unemployment_Rate_Smoothed'] - data['Hist_Lower_Bound']) / (data['Hist_Upper_Bound'] - data['Hist_Lower_Bound'])
data['Risk'] = (data['Risk'] - data['Risk'].min()) / (data['Risk'].max() - data['Risk'].min())  # Normalize

# --- Figure 3: Unemployment Rate with Historical Bounds Only ---
fig3 = go.Figure()

# Unemployment Rate Smoothed
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Unemployment_Rate_Smoothed'], 
                          mode='lines', name='Unemployment Rate (Smoothed)', 
                          line=dict(color='blue')))

# Historical Upper and Lower Bounds (Mean ± 2 Std Dev in Log Space)
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Hist_Upper_Bound'], 
                          mode='lines', name='Upper Bound (Historical)', 
                          line=dict(color='green', dash='dash')))

fig3.add_trace(go.Scatter(x=data['Date'], y=data['Hist_Lower_Bound'], 
                          mode='lines', name='Lower Bound (Historical)', 
                          line=dict(color='green', dash='dash')))

fig3.update_layout(
    title='Unemployment Rate with Historical Bounds (Log-Normal)',
    xaxis_title='Date',
    yaxis_title='Unemployment Rate',
    template='plotly_dark'
)

fig3.show()

# --- Figure 4: S&P 500 Scatter Plot Color-Coded by Risk ---
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=data['Date'],
    y=data['S&P_500'],
    mode='markers',
    marker=dict(
        size=8,
        color=data['Risk'],
        colorscale='jet',
        showscale=True,
        colorbar=dict(title="Risk")
    ),
    text=[f"Risk: {r:.2f}" for r in data['Risk']],
    hoverinfo="text+x+y",
    name="S&P 500 (Color-Coded by Risk)"
))

# Update layout for the plot
fig4.update_layout(
    title="S&P 500 with Unemployment Risk-Based Color Coding ",
    xaxis_title="Date",
    yaxis_title="S&P 500 (Inflation-Adjusted)",
    template="plotly_dark",
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=700,
    xaxis=dict(
        tickangle=45,
        tickformat='%Y-%m'
    ),
    yaxis=dict(type='log')
)

fig4.show()
