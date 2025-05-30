import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas_datareader import data as pdr
from datetime import datetime

# Add parent directory to Python path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Download VIX data from FRED
def get_vix_monthly():
    try:
        # Get VIX data from FRED (Federal Reserve Economic Data)
        start = datetime(1990, 1, 1)
        end = datetime.now()
        vix = pdr.DataReader('VIXCLS', 'fred', start, end)
        vix = vix.rename(columns={'VIXCLS': 'VIX'})
        
        # Resample to month start and forward fill missing values
        vix = vix.resample('MS').first().ffill()  # Using ffill() instead of fillna(method='ffill')
        return vix
    except Exception as e:
        print(f"Error downloading VIX data: {e}")
        return pd.DataFrame()

# Load Shiller PE risk metric
def get_shiller_pe_risk():
    from risk.shiller_pe_risk import load_shiller_pe, compute_risk
    data_pe, data_sp500 = load_shiller_pe()
    risk_data = compute_risk(data_pe, data_sp500)
    # Resample to month start for alignment
    risk_monthly = risk_data.set_index('Date').resample('MS').first()
    return risk_monthly[['Risk']]

if __name__ == "__main__":
    # Get data
    vix = get_vix_monthly()
    risk = get_shiller_pe_risk()

    # Align on dates and calculate correlation
    combined = pd.merge(vix, risk, left_index=True, right_index=True, how='inner')
    corr = combined['VIX'].corr(combined['Risk'])
    print(f"\nCorrelation between VIX and Shiller PE Risk metric: {corr:.4f}")

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot VIX
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX', color='tab:blue')
    line1 = ax1.plot(combined.index, combined['VIX'], 
                     color='tab:blue', label='VIX', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot Risk on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Shiller PE Risk', color='tab:orange')
    line2 = ax2.plot(combined.index, combined['Risk'], 
                     color='tab:orange', label='Risk', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('VIX vs Shiller PE Risk (Monthly)')
    plt.grid(True, alpha=0.3)
    plt.show()
