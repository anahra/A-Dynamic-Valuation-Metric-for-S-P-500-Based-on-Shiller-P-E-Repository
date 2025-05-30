# run.py
"""
This script runs the Shiller P/E risk analysis by calling the main functions from shiller_pe_risk.py.
It pulls the data, computes the risk metrics, and displays the charts.
"""

from risk.shiller_pe_risk import load_shiller_pe, compute_risk
from strategies.strat_test import run_strategy, plot_strategy_results
from strategies.analyze_performance import analyze_by_decades
from risk.shiller_pe_risk import plot_charts
import time
import pandas as pd
import os

if __name__ == "__main__":    # User parameters
    start_year = 1950
    initial_investment = 0
    monthly_investment = 200

    # Load and process data
    start = time.time()
    data_pe, data_sp500 = load_shiller_pe()
    print("Loaded data in", time.time() - start, "seconds")

    start = time.time()
    risk_data = compute_risk(data_pe, data_sp500)
    print("Computed risk in", time.time() - start, "seconds")

     # Analyze performance by decades
    print("\nAnalyzing performance across decades...")
    returns_df, risk_metrics_df = analyze_by_decades(risk_data, start_year=start_year, 
                                                    initial_investment=initial_investment, 
                                                    monthly_investment=monthly_investment)
    
    # Format the display of the performance tables
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Display returns table
    print("\nTotal Returns Comparison Across Time Periods:")
    print("=========================================")
    returns_cols = ['Period', 'Years', 'Benchmark Return %', 'Strategy Return %', 'Outperformance %']
    print(returns_df[returns_cols].to_string(index=False))

      # Display risk metrics table
    print("\nRisk-Adjusted Metrics Across Time Periods:")
    print("=========================================")
    risk_cols = ['Period', 'Years', 'Benchmark Sharpe', 'Strategy Sharpe', 'Benchmark IRR %', 'Strategy IRR %']
    print(risk_metrics_df[risk_cols].to_string(index=False))

    # Display maximum drawdown table
    drawdown_cols = ['Period', 'Years', 'Benchmark Max Drawdown %', 'Strategy Max Drawdown %']
    print("\nMaximum Drawdown Across Time Periods:")
    print("=========================================")
    print(returns_df[drawdown_cols].to_string(index=False))
    
    # Run final strategy from user-defined start_year to present for plotting
    start = time.time()
    results = run_strategy(risk_data, start_year=start_year, initial_investment=initial_investment, monthly_investment=monthly_investment)
    print("\nRan complete strategy in", time.time() - start, "seconds")
      # Calculate monthly data and metrics
    from strategies.analyze_sharpe_ratio import analyze_risk_adjusted_returns
    monthly_data, final_metrics = analyze_risk_adjusted_returns(results)
    
        
    # Create monthly cashflow data
    strategy_monthly_cashflow = monthly_data['Strategy_Cashflow'].diff().fillna(monthly_data['Strategy_Cashflow'])
    # Replace last month's cashflow with final portfolio value
    strategy_monthly_cashflow.iloc[-1] = monthly_data['Strategy_Portfolio'].iloc[-1]
    
    monthly_cashflow_data = pd.DataFrame({
        'Date': monthly_data.index,
        'Benchmark_Monthly_Investment': [-200] * (len(monthly_data) - 1) + [monthly_data['Benchmark_Portfolio'].iloc[-1]],
        'Benchmark_Portfolio_Value': monthly_data['Benchmark_Portfolio'],
        'Strategy_Monthly_Cashflow': strategy_monthly_cashflow,
        'Strategy_Portfolio_Value': monthly_data['Strategy_Portfolio'],
        'Risk_Level': monthly_data['Risk']
    })    # Save monthly data to CSV in strategies folder
    csv_path = os.path.join('strategies', 'cashflows.csv')
    try:
        monthly_cashflow_data.to_csv(csv_path, index=False)
        print(f"\nMonthly cashflow data saved to: {csv_path}")
    except PermissionError:
        print(f"\n[ERROR] Could not save {csv_path}. Please close the file if it is open in another program.")

    # Plot final results
    start = time.time()
    plot_strategy_results(results)
    print("Plotted strategy results in", time.time() - start, "seconds")
    
    # Also add Shiller PE charts if you want to see them
    plot_charts(risk_data)
