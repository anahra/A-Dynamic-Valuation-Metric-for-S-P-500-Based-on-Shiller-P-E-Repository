# compute_risk.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.shiller_pe_loader import load_shiller_pe  # Importing the function from the loader

# Risk computation logic
def compute_risk(data_pe, data_sp500, rolling_window_upper=300, rolling_window_lower=300, number_standard_deviations=3):
    # Merge both datasets on the Date column
    data = pd.merge(data_pe, data_sp500, on='Date', how='inner')

    # Apply log transformation to the P/E ratio
    data['Log_PE'] = np.log(data['PE_Ratio'])

    # Compute rolling statistics in log space
    data['Rolling_Mean_upper'] = data['Log_PE'].rolling(window=rolling_window_upper).mean()
    data['Rolling_Std_upper'] = data['Log_PE'].rolling(window=rolling_window_upper).std()
    data['Rolling_Mean_lower'] = data['Log_PE'].rolling(window=rolling_window_lower).mean()
    data['Rolling_Std_lower'] = data['Log_PE'].rolling(window=rolling_window_lower).std()

    # Upper and Lower Bound calculations
    data['Upper_Bound'] = np.exp(data['Rolling_Mean_upper'] + number_standard_deviations * data['Rolling_Std_upper'])
    data['Lower_Bound'] = np.exp(data['Rolling_Mean_lower'] - number_standard_deviations * data['Rolling_Std_lower'])
    data['Rolling_Mean_upper'] = np.exp(data['Rolling_Mean_upper'])  # Re-map mean back from log scale

    # Drop rows with missing values in Rolling Mean
    data.dropna(subset=['Rolling_Mean_upper'], inplace=True)

    # Historical average and standard deviation in log space
    historical_avg_log = data['Log_PE'].mean()
    historical_std_log = data['Log_PE'].std()

    # Historical stats
    data['Historical_Avg'] = np.exp(historical_avg_log)
    data['Historical_Upper'] = np.exp(historical_avg_log + 2 * historical_std_log)
    data['Historical_Lower'] = np.exp(historical_avg_log - 2 * historical_std_log)

    # Compute Risk metric
    data['Risk'] = (data['PE_Ratio'] - data['Lower_Bound']) / (data['Upper_Bound'] - data['Lower_Bound'])
    data['Risk'] = (data['Risk'] - data['Risk'].min()) / (data['Risk'].max() - data['Risk'].min())  # Normalize between 0 and 1

    # show the risk data header
    print(data[['Date', 'PE_Ratio', 'Risk']].head())

    return data

# Chart plotting functions
def plot_charts(data):
    import plotly.io as pio
    pio.renderers.default = "browser"  # Ensure charts open in the default web browser    # Common chart settings
    chart_settings = dict(
        template="plotly_white",
        height=800,
        font=dict(size=24, color='black'),  # Increased base font size
        xaxis=dict(
            tickformat='%Y',  # Show only years
            tickangle=0,      # Horizontal labels
            title_font=dict(size=32, color='black'),  # Increased axis title
            tickfont=dict(size=24),  # Increased tick labels
        ),
        yaxis=dict(
            title_font=dict(size=32, color='black'),  # Increased axis title
            tickfont=dict(size=24),  # Increased tick labels
        ),
        title_font=dict(size=36, color='black'),  # Increased chart title
        legend=dict(
            font=dict(size=24),  # Increased legend text
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # Create first chart: Rolling statistics
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['PE_Ratio'], mode='lines', name='Shiller P/E Ratio', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Rolling_Mean_upper'], mode='lines', name='Rolling Mean', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Bound'], mode='lines', name='Upper Bound (±3 Std Dev)', line=dict(color='gray', dash='dot')))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Bound'], mode='lines', name='Lower Bound (±3 Std Dev)', line=dict(color='gray', dash='dot'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['S&P_500'], mode='lines', name='S&P 500 (Nominal)', line=dict(color='red'), yaxis='y2'))
    fig1.update_layout(
        title='Shiller P/E Ratio and S&P 500 (Nominal Values)',
        xaxis_title='Date',
        yaxis_title=dict(text='P/E Ratio', font=dict(size=16)),
        yaxis2=dict(
            title=dict(text='S&P 500 (Nominal, Log Scale)', font=dict(size=16)),
            overlaying='y',
            side='right',
            type='log'
        ),
        **chart_settings
    )    # Create second chart: Historical stats
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['PE_Ratio'], mode='lines', name='Shiller P/E Ratio', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Historical_Avg'], mode='lines', name='Historical Average', line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Historical_Upper'], mode='lines', name='Historical Avg + 2 Std Dev', line=dict(color='red', dash='dash')))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Historical_Lower'], mode='lines', name='Historical Avg - 2 Std Dev', line=dict(color='green', dash='dash')))
    fig2.update_layout(
        title='Historical Shiller P/E Ratio with Mean and Std Dev',
        xaxis_title='Date',
        yaxis_title=dict(text='P/E Ratio', font=dict(size=16)),
        **chart_settings
    )

    # Create third chart: S&P 500 risk-based color coding
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data['Date'], y=data['S&P_500'], mode='markers', 
        marker=dict(
            size=8, 
            color=data['Risk'], 
            colorscale='Jet',  # Jet, reversed
            showscale=True, 
            colorbar=dict(
                title=dict(text='Risk', font=dict(size=32)),
                tickfont=dict(size=24)
            )
        ), 
        text=[f"Risk: {r:.2f}" for r in data['Risk']], 
        hoverinfo='text+x+y', 
        name='S&P 500 (Color-Coded by Risk)')
    )    # Update the yaxis settings in chart_settings for this specific chart
    specific_settings = chart_settings.copy()
    specific_settings.update({
        'yaxis': dict(
            type='log',
            title=dict(text='S&P 500 (Nominal, Log Scale)', font=dict(size=32, color='black')),
            tickfont=dict(size=24)
        )
    })
    
    fig3.update_layout(
        title='S&P 500 (Nominal) with P/E Risk-Based Color Coding',
        **specific_settings
    )

    # Create fourth chart: Logarithmic scale corridor
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['PE_Ratio'], mode='lines', name='Shiller P/E Ratio', line=dict(color='blue')))
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['Rolling_Mean_upper'], mode='lines', name='Rolling Mean (Log Scale)', line=dict(color='orange')))
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Bound'], mode='lines', name='Upper Bound (+3 Std Dev, Log Scale)', line=dict(color='red', dash='dot')))
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Bound'], mode='lines', name='Lower Bound (-3 Std Dev, Log Scale)', line=dict(color='green', dash='dot'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)'))# Update the yaxis settings in chart_settings for this specific chart
    specific_settings = chart_settings.copy()
    specific_settings.update({
        'yaxis': dict(
            type='log',
            title=dict(text='P/E Ratio (Log Scale)', font=dict(size=20, color='black')),
            tickfont=dict(size=16)
        )
    })
    
    fig4.update_layout(
        title='Shiller P/E Ratio with Logarithmic Corridor',
        **specific_settings
    )

    # Show plots
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()

# Now, you can fetch the data and calculate the risk
if __name__ == "__main__":
    # Fetch the data by calling the function from data_loader.py
    data_pe, data_sp500 = load_shiller_pe()

    # Compute the risk
    risk_data = compute_risk(data_pe, data_sp500)

    # Plot the charts
    plot_charts(risk_data)
