"""
Module for analyzing risk-adjusted returns (Sharpe ratio) and Internal Rate of Return (IRR)
for investment strategies.

The Sharpe ratio calculation properly accounts for monthly contributions by:
1. Calculating "pure" returns excluding the effect of new investments
2. Using formula: (end_value - start_value - new_contributions) / start_value
3. Using 3-month Treasury Bill rates as risk-free rate from FRED data
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import os

# Control whether to use historical rates or a flat 2% rate
USE_HISTORICAL_RATES = False  # Set to False to use flat 2% rate instead of TB3MS historical data
FLAT_RATE_VALUE = 0.02  # 2% annual rate

def get_risk_free_rate():
    """
    Load 3-month Treasury Bill rate data from TB3MS.csv file or use flat rate.
    Returns monthly rates as a Series with Date index or a constant rate.
    
    If USE_HISTORICAL_RATES is True:
      - Returns a Series of historical T-bill rates
    If USE_HISTORICAL_RATES is False:
      - Returns a constant rate (FLAT_RATE_VALUE)
    """
    if not USE_HISTORICAL_RATES:
        print(f"Using flat {FLAT_RATE_VALUE:.1%} annual risk-free rate as configured")
        return FLAT_RATE_VALUE  # Return flat annual rate
    
    # If using historical rates, try to load TB3MS data
    try:
        # Read the TB3MS.csv file
        file_path = os.path.join(os.path.dirname(__file__), 'TB3MS.csv')
        tbill_data = pd.read_csv(file_path)
        
        # Convert observation_date to datetime
        tbill_data['observation_date'] = pd.to_datetime(tbill_data['observation_date'])
        
        # Set date as index
        tbill_data.set_index('observation_date', inplace=True)
        
        # Convert annual percentage to decimal
        # TB3MS data is in percentage, so divide by 100 to get decimal
        tbill_monthly = tbill_data['TB3MS'] / 100
        
        # Set the frequency to monthly start
        tbill_monthly = tbill_monthly.asfreq('MS')
        
        print(f"Loaded {len(tbill_monthly)} months of risk-free rate data")
        print(f"Date range: {tbill_monthly.index.min()} to {tbill_monthly.index.max()}")
        print(f"Average rate: {tbill_monthly.mean():.4f} ({tbill_monthly.mean()*100:.2f}%)")
        
        return tbill_monthly
        
    except Exception as e:
        print(f"Warning: Could not load TB3MS data: {e}")
        print(f"Using default {FLAT_RATE_VALUE:.1%} annual risk-free rate")
        return FLAT_RATE_VALUE

def calculate_cashflows(results):
    """
    Calculate actual cashflows for both strategies.
    
    For benchmark: Fixed -200 investment at start of each month
    For strategy: Difference between consecutive accumulated cashflow values
    Last cashflow for both: Final portfolio value
    
    Args:
        results (pd.DataFrame): DataFrame with Date, Strategy_Cashflow, and portfolio values
    """
    # Group by month start to get monthly data points
    monthly_data = results.groupby(pd.Grouper(key='Date', freq='MS')).agg({
        'Strategy_Cashflow': 'first',
        'Benchmark_Portfolio': 'first',
        'Strategy_Portfolio': 'first',
        'Risk': 'first'  # Include Risk for consistency
    }).reset_index()
    
    # Initialize lists for cashflows
    benchmark_cashflows = []
    strategy_cashflows = []
      # Process all months except the last
    for i in range(len(monthly_data) - 1):
        # Benchmark: Always -200 monthly investment
        benchmark_cashflows.append(-200)
        
        # Strategy: Difference in accumulated cashflows from previous month
        if i == 0:
            # First month's cashflow is the initial cashflow value
            strategy_cf = monthly_data['Strategy_Cashflow'].iloc[0]
        else:
            # Other months: take difference from previous month
            current_acc = monthly_data['Strategy_Cashflow'].iloc[i]
            prev_acc = monthly_data['Strategy_Cashflow'].iloc[i-1]
            strategy_cf = current_acc - prev_acc
        strategy_cashflows.append(strategy_cf)
    
    # For the last period, add the final portfolio values (positive cashflow)
    benchmark_cashflows.append(monthly_data['Benchmark_Portfolio'].iloc[-1])
    strategy_cashflows.append(monthly_data['Strategy_Portfolio'].iloc[-1])
    
    return pd.Series(benchmark_cashflows), pd.Series(strategy_cashflows)

def calculate_monthly_returns(portfolio_values, cashflows):
    """
    Calculate monthly returns properly accounting for contributions/withdrawals.
    
    Formula: (end_value - start_value - new_contributions) / start_value
    
    Args:
        portfolio_values (pd.Series): Monthly portfolio values
        cashflows (pd.Series): Monthly cashflows (negative for investments)
    
    Returns:
        pd.Series: Monthly returns excluding effect of contributions
    """
    returns = []
    
    for i in range(1, len(portfolio_values)):
        start_value = portfolio_values.iloc[i-1]
        end_value = portfolio_values.iloc[i]
        # Contributions are negative in our cashflows, so we negate them
        new_contribution = -cashflows.iloc[i] if i < len(cashflows) else 0
        
        if start_value > 0:  # Avoid division by zero
            monthly_return = (end_value - start_value - new_contribution) / start_value
            returns.append(monthly_return)
        else:
            returns.append(0.0)
    
    # Add 0 for the first month since we can't calculate return
    returns.insert(0, 0.0)
    return pd.Series(returns, index=portfolio_values.index)

def analyze_risk_adjusted_returns(results):
    """
    Calculate Sharpe ratio and IRR for both strategies using actual T-bill rates
    and proper return calculation accounting for monthly contributions.
    
    Args:
        results (pd.DataFrame): Results dataframe with daily portfolio values
    """
    # Calculate cashflows
    benchmark_cashflows, strategy_cashflows = calculate_cashflows(results)
      # Calculate monthly data
    monthly_data = results.resample('MS', on='Date').first()
    
    # Calculate total values (portfolio + positive cash)
    monthly_data['Benchmark_Total_Value'] = monthly_data['Benchmark_Portfolio'] + monthly_data['Benchmark_Cashflow'].clip(lower=0)
    monthly_data['Strategy_Total_Value'] = monthly_data['Strategy_Portfolio'] + monthly_data['Strategy_Cashflow'].clip(lower=0)
    
    # Calculate returns properly accounting for contributions
    benchmark_monthly_returns = calculate_monthly_returns(
        monthly_data['Benchmark_Total_Value'], 
        pd.Series([-200] * len(monthly_data))  # Fixed monthly contribution for benchmark
    )
    
    strategy_monthly_returns = calculate_monthly_returns(
        monthly_data['Strategy_Total_Value'],
        strategy_cashflows
    )
    
    # Get risk-free rate data
    rf_rates = get_risk_free_rate()
      # Calculate excess returns over risk-free rate
    if isinstance(rf_rates, pd.Series):
        # If we have historical rates, align them with our data
        aligned_rf = rf_rates.reindex(monthly_data.index, method='ffill')
        # TB3MS data is already in percentage terms (e.g., 4.27 means 4.27%)
        # The rates are annual rates, need to convert to monthly equivalent
        aligned_rf_monthly = (1 + aligned_rf) ** (1/12) - 1
        benchmark_excess_returns = benchmark_monthly_returns - aligned_rf_monthly
        strategy_excess_returns = strategy_monthly_returns - aligned_rf_monthly
        # Use average T-bill rate for overall metrics
        rf_rate = aligned_rf.mean()
    else:
        # If we're using constant rate
        rf_rate_monthly = (1 + rf_rates) ** (1/12) - 1
        benchmark_excess_returns = benchmark_monthly_returns - rf_rate_monthly
        strategy_excess_returns = strategy_monthly_returns - rf_rate_monthly
        rf_rate = rf_rates
    # Calculate monthly statistics first
    # Standard deviation of raw monthly returns (for reporting annual std dev)
    benchmark_std_raw_monthly = np.nanstd(benchmark_monthly_returns)
    strategy_std_raw_monthly = np.nanstd(strategy_monthly_returns)
    
    # Mean and Standard deviation of monthly excess returns (for Sharpe Ratio)
    benchmark_mean_excess = np.nanmean(benchmark_excess_returns)
    strategy_mean_excess = np.nanmean(strategy_excess_returns)
    benchmark_std_excess_monthly = np.nanstd(benchmark_excess_returns)
    strategy_std_excess_monthly = np.nanstd(strategy_excess_returns)
    
    # Calculate monthly Sharpe ratios and then annualize
    # Sharpe = Mean(Excess Return) / Std(Excess Return)
    benchmark_sharpe = (benchmark_mean_excess / benchmark_std_excess_monthly) * np.sqrt(12) if benchmark_std_excess_monthly > 0 else 0
    strategy_sharpe = (strategy_mean_excess / strategy_std_excess_monthly) * np.sqrt(12) if strategy_std_excess_monthly > 0 else 0
    
    # Calculate annualized returns for reporting
    benchmark_mean_annual = np.nanmean(benchmark_monthly_returns) * 12
    strategy_mean_annual = np.nanmean(strategy_monthly_returns) * 12
    
    # Calculate IRRs
    try:
        benchmark_monthly_irr = npf.irr(benchmark_cashflows)
        benchmark_annual_irr = (1 + benchmark_monthly_irr) ** 12 - 1
    except:
        benchmark_annual_irr = 0
        
    try:
        strategy_monthly_irr = npf.irr(strategy_cashflows)
        strategy_annual_irr = (1 + strategy_monthly_irr) ** 12 - 1
    except:
        strategy_annual_irr = 0
    
    metrics = {
        'Benchmark Mean Return (Annual)': benchmark_mean_annual,
        'Strategy Mean Return (Annual)': strategy_mean_annual,
        'Benchmark Std Dev (Annual)': benchmark_std_raw_monthly * np.sqrt(12), # Annualized StdDev of raw returns
        'Strategy Std Dev (Annual)': strategy_std_raw_monthly * np.sqrt(12),   # Annualized StdDev of raw returns
        'Risk-Free Rate (Annual)': rf_rate,
        'Benchmark Sharpe Ratio': benchmark_sharpe,
        'Strategy Sharpe Ratio': strategy_sharpe,
        'Benchmark IRR (Annual)': benchmark_annual_irr,
        'Strategy IRR (Annual)': strategy_annual_irr
    }
    
    # Add the calculated data to monthly_data for reference
    monthly_data['Benchmark_Return'] = benchmark_monthly_returns
    monthly_data['Strategy_Return'] = strategy_monthly_returns
    monthly_data['Risk_Free_Rate'] = aligned_rf if isinstance(rf_rates, pd.Series) else rf_rate
    monthly_data['Risk_Free_Rate_Monthly'] = aligned_rf_monthly if isinstance(rf_rates, pd.Series) else rf_rate_monthly
    monthly_data['Benchmark_Excess_Return'] = benchmark_excess_returns
    monthly_data['Strategy_Excess_Return'] = strategy_excess_returns
    
    return monthly_data, metrics