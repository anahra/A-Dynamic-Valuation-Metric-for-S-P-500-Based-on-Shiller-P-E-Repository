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
    
    # Ensure no duplicates after merge
    data.drop_duplicates(subset=['Date'], inplace=True)

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
    # Common chart settings
    chart_settings = dict(
        template="plotly_white",
        height=800,
        font=dict(size=24, color='black'),  # Increased base font size
        xaxis=dict(
            tickformat='%Y',  # Show only years
            tickangle=0,      # Horizontal labels
            title_font=dict(size=32, color='#CCCCCC'),  # Increased axis title
            tickfont=dict(size=24, color='#CCCCCC'),  # Increased tick labels
        ),
        yaxis=dict(
            title_font=dict(size=20, color='#CCCCCC'),  # Axis title
            tickfont=dict(size=16, color='#CCCCCC'),  # Tick labels
        ),
        title_font=dict(size=36, color='#555555'),  # Lighter title color
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
            bgcolor='rgba(0,0,0,0)' # Transparent background
        )
    )

    # Create first chart: Rolling statistics
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['PE_Ratio'], mode='lines', name='Shiller P/E Ratio', line=dict(color='blue'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Rolling_Mean_upper'], mode='lines', name='Rolling Mean', line=dict(color='orange'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Bound'], mode='lines', name='Upper Bound (±3 Std Dev)', line=dict(color='gray', dash='dot'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Bound'], mode='lines', name='Lower Bound (±3 Std Dev)', line=dict(color='gray', dash='dot'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['S&P_500'], mode='lines', name='S&P 500 (Nominal)', line=dict(color='red'), yaxis='y2', hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig1.update_layout(
        title='Shiller P/E Ratio and S&P 500 (Nominal Values)',
        xaxis_title='Date',
        yaxis_title=dict(text='P/E Ratio', font=dict(size=16)),
        yaxis2=dict(
            title=dict(text='S&P 500 (Nominal, Log Scale)', font=dict(size=16, color='#CCCCCC')),
            overlaying='y',
            side='right',
            type='log',
            tickfont=dict(color='#CCCCCC')
        ),
        **chart_settings
    )    # Create second chart: Historical stats
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['PE_Ratio'], mode='lines', name='Shiller P/E Ratio', line=dict(color='blue'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Historical_Avg'], mode='lines', name='Historical Average', line=dict(color='black', dash='dash'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Historical_Upper'], mode='lines', name='Historical Avg + 2 Std Dev', line=dict(color='red', dash='dash'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Historical_Lower'], mode='lines', name='Historical Avg - 2 Std Dev', line=dict(color='green', dash='dash'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
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
        text=[f"Date: {d.strftime('%d %b %Y')}<br>Risk: {r:.2f}" for d, r in zip(data['Date'], data['Risk'])], 
        hovertemplate='%{text}<br>S&P 500: %{y:.2f}<extra></extra>', 
        name='S&P 500 (Color-Coded by Risk)')
    )    # Update the yaxis settings in chart_settings for this specific chart
    specific_settings = chart_settings.copy()
    specific_settings.update({
        'yaxis': dict(
            type='log',
            title=dict(text='S&P 500 (Nominal, Log Scale)', font=dict(size=20, color='#CCCCCC')),
            tickfont=dict(size=16, color='#CCCCCC')
        )
    })
    
    fig3.update_layout(
        title='S&P 500 (Nominal) with P/E Risk-Based Color Coding',
        **specific_settings
    )

    # Create fourth chart: Logarithmic scale corridor
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['PE_Ratio'], mode='lines', name='Shiller P/E Ratio', line=dict(color='blue'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['Rolling_Mean_upper'], mode='lines', name='Rolling Mean (Log Scale)', line=dict(color='orange'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Bound'], mode='lines', name='Upper Bound (+3 Std Dev, Log Scale)', line=dict(color='red', dash='dot'), hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))
    fig4.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Bound'], mode='lines', name='Lower Bound (-3 Std Dev, Log Scale)', line=dict(color='green', dash='dot'), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', hovertemplate='%{x|%d %b %Y}: %{y:.2f}<extra></extra>'))# Update the yaxis settings in chart_settings for this specific chart
    specific_settings = chart_settings.copy()
    specific_settings.update({
        'yaxis': dict(
            type='log',
            title=dict(text='P/E Ratio (Log Scale)', font=dict(size=20, color='#CCCCCC')),
            tickfont=dict(size=16, color='#CCCCCC')
        )
    })
    
    fig4.update_layout(
        title='Shiller P/E Ratio with Logarithmic Corridor',
        **specific_settings
    )

    return [fig1, fig2, fig3, fig4]

# Now, you can fetch the data and calculate the risk
if __name__ == "__main__":
    # Fetch the data by calling the function from data_loader.py
    data_pe, data_sp500 = load_shiller_pe()

    # Compute the risk
    risk_data = compute_risk(data_pe, data_sp500)


    # Plot the charts
    figs = plot_charts(risk_data)
    for fig in figs:
        fig.show()

def compute_forward_returns(data, years=10):
    """
    Computes the N-year forward annualized price return.
    """
    data = data.copy()
    months = years * 12
    # Shift future price back to current row
    # S&P_500 is nominal price
    data['S&P_500_Future'] = data['S&P_500'].shift(-months)
    
    # Calculate annualized return: (P_future / P_current)^(1/years) - 1
    # We use (1/years) because it's annualized
    data['10Y_Return'] = (data['S&P_500_Future'] / data['S&P_500']) ** (1/years) - 1
    
    return data

def plot_correlation_charts(data):
    """
    Plots scatter plots for 10Y Forward Return vs Risk and vs P/E Ratio.
    """
    # Compute returns if not present
    if '10Y_Return' not in data.columns:
        data = compute_forward_returns(data)
    
    # Filter out NaN values (the most recent 10 years won't have future returns)
    corr_data = data.dropna(subset=['10Y_Return']).copy()
    
    # Common settings from plot_charts but customized
    chart_settings = dict(
        template="plotly_white",
        height=600,
        font=dict(size=14, color='black'),
        title_font=dict(size=24, color='#333333'),
    )
    
    # 1. 10Y Return vs Risk
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=corr_data['Risk'],
        y=corr_data['10Y_Return'],
        mode='markers',
        marker=dict(
            size=6,
            color=corr_data['Date'].dt.year, # Color by year
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        text=corr_data['Date'].dt.strftime('%b %Y'),
        hovertemplate='Date: %{text}<br>Risk: %{x:.2f}<br>10Y Ann. Return: %{y:.1%}<extra></extra>'
    ))
    
    # Add trendline (Linear Regression)
    # Using numpy for simple linear regression
    idx = np.isfinite(corr_data['Risk']) & np.isfinite(corr_data['10Y_Return'])
    if idx.sum() > 1:
        m, b = np.polyfit(corr_data.loc[idx, 'Risk'], corr_data.loc[idx, '10Y_Return'], 1)
        # Create line points
        x_range = np.linspace(corr_data['Risk'].min(), corr_data['Risk'].max(), 100)
        y_range = m * x_range + b
        
        fig1.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Trendline',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Calculate Correlation Coefficient
        correlation = corr_data.loc[idx, 'Risk'].corr(corr_data.loc[idx, '10Y_Return'])
        title_suffix = f" (Correlation: {correlation:.2f})"
    else:
        title_suffix = ""

    fig1.update_layout(
        title=f'10-Year Annualized Return vs. Risk Metric{title_suffix}',
        xaxis_title='Risk Metric (0=Undervalued, 1=Overvalued)',
        yaxis_title='10-Year Annualized Return',
        yaxis=dict(tickformat='.1%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **chart_settings
    )

    # 2. 10Y Return vs P/E Ratio
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=corr_data['PE_Ratio'],
        y=corr_data['10Y_Return'],
        mode='markers',
        marker=dict(
            size=6,
            color=corr_data['Date'].dt.year,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        text=corr_data['Date'].dt.strftime('%b %Y'),
        hovertemplate='Date: %{text}<br>Shiller P/E: %{x:.2f}<br>10Y Ann. Return: %{y:.1%}<extra></extra>'
    ))

    # Add trendline
    idx_pe = np.isfinite(corr_data['PE_Ratio']) & np.isfinite(corr_data['10Y_Return'])
    if idx_pe.sum() > 1:
        m_pe, b_pe = np.polyfit(corr_data.loc[idx_pe, 'PE_Ratio'], corr_data.loc[idx_pe, '10Y_Return'], 1)
        x_range_pe = np.linspace(corr_data['PE_Ratio'].min(), corr_data['PE_Ratio'].max(), 100)
        y_range_pe = m_pe * x_range_pe + b_pe

        fig2.add_trace(go.Scatter(
            x=x_range_pe,
            y=y_range_pe,
            mode='lines',
            name='Trendline',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        correlation_pe = corr_data.loc[idx_pe, 'PE_Ratio'].corr(corr_data.loc[idx_pe, '10Y_Return'])
        title_suffix_pe = f" (Correlation: {correlation_pe:.2f})"
    else:
        title_suffix_pe = ""


    fig2.update_layout(
        title=f'10-Year Annualized Return vs. Shiller P/E Ratio (Monthly Data){title_suffix_pe}',
        xaxis_title='Shiller P/E Ratio',
        yaxis_title='10-Year Annualized Return',
        yaxis=dict(tickformat='.1%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **chart_settings
    )

    # --- Yearly Analysis ---
    # Resample to Yearly frequency, taking the mean of values
    yearly_data = data.copy()
    yearly_data.set_index('Date', inplace=True)
    yearly_data = yearly_data.resample('YE').mean() # Average of the year
    yearly_data.reset_index(inplace=True)
    
    # Compute 10Y Returns for Yearly Data (shift by 10 rows/years)
    yearly_data['S&P_500_Future'] = yearly_data['S&P_500'].shift(-10)
    yearly_data['10Y_Return'] = (yearly_data['S&P_500_Future'] / yearly_data['S&P_500']) ** (1/10) - 1
    
    corr_yearly = yearly_data.dropna(subset=['10Y_Return']).copy()
    
    # 3. Yearly 10Y Return vs Risk
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=corr_yearly['Risk'],
        y=corr_yearly['10Y_Return'],
        mode='markers',
        marker=dict(
            size=10, # Larger markers for yearly
            color=corr_yearly['Date'].dt.year,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        text=corr_yearly['Date'].dt.year,
        hovertemplate='Year: %{text}<br>Avg Risk: %{x:.2f}<br>10Y Ann. Return: %{y:.1%}<extra></extra>'
    ))
    
    # Yearly Trendline
    idx_y = np.isfinite(corr_yearly['Risk']) & np.isfinite(corr_yearly['10Y_Return'])
    if idx_y.sum() > 1:
        m_y, b_y = np.polyfit(corr_yearly.loc[idx_y, 'Risk'], corr_yearly.loc[idx_y, '10Y_Return'], 1)
        x_range_y = np.linspace(corr_yearly['Risk'].min(), corr_yearly['Risk'].max(), 100)
        y_range_y = m_y * x_range_y + b_y
        
        fig3.add_trace(go.Scatter(
            x=x_range_y, y=y_range_y, mode='lines', name='Trendline',
            line=dict(color='red', width=2, dash='dash')
        ))
        corr_coef_y = corr_yearly.loc[idx_y, 'Risk'].corr(corr_yearly.loc[idx_y, '10Y_Return'])
        title_suffix_y = f" (Correlation: {corr_coef_y:.2f})"
    else:
        title_suffix_y = ""
        
    fig3.update_layout(
        title=f'10-Year Annualized Return vs. Yearly Avg Risk{title_suffix_y}',
        xaxis_title='Average Risk Metric (Yearly)',
        yaxis_title='10-Year Annualized Return',
        yaxis=dict(tickformat='.1%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **chart_settings
    )

    # 4. Yearly 10Y Return vs P/E
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=corr_yearly['PE_Ratio'],
        y=corr_yearly['10Y_Return'],
        mode='markers',
        marker=dict(
            size=10, 
            color=corr_yearly['Date'].dt.year,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        text=corr_yearly['Date'].dt.year,
        hovertemplate='Year: %{text}<br>Avg Shiller P/E: %{x:.2f}<br>10Y Ann. Return: %{y:.1%}<extra></extra>'
    ))
    
    # Yearly Trendline (PE)
    idx_pe_y = np.isfinite(corr_yearly['PE_Ratio']) & np.isfinite(corr_yearly['10Y_Return'])
    if idx_pe_y.sum() > 1:
        m_pe_y, b_pe_y = np.polyfit(corr_yearly.loc[idx_pe_y, 'PE_Ratio'], corr_yearly.loc[idx_pe_y, '10Y_Return'], 1)
        x_range_pe_y = np.linspace(corr_yearly['PE_Ratio'].min(), corr_yearly['PE_Ratio'].max(), 100)
        y_range_pe_y = m_pe_y * x_range_pe_y + b_pe_y
        
        fig4.add_trace(go.Scatter(
            x=x_range_pe_y, y=y_range_pe_y, mode='lines', name='Trendline',
            line=dict(color='red', width=2, dash='dash')
        ))
        corr_coef_pe_y = corr_yearly.loc[idx_pe_y, 'PE_Ratio'].corr(corr_yearly.loc[idx_pe_y, '10Y_Return'])
        title_suffix_pe_y = f" (Correlation: {corr_coef_pe_y:.2f})"
    else:
        title_suffix_pe_y = ""

    fig4.update_layout(
        title=f'10-Year Annualized Return vs. Yearly Avg Shiller P/E{title_suffix_pe_y}',
        xaxis_title='Average Shiller P/E Ratio (Yearly)',
        yaxis_title='10-Year Annualized Return',
        yaxis=dict(tickformat='.1%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **chart_settings
    )

    return [fig1, fig2, fig3, fig4]
